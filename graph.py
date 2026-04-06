# ═══════════════════════════════════════════════════════════════════════════════
# graph.py — LangGraph ReAct Agent
# Converted from the LangChain ReAct agent to a LangGraph workflow
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import math as mathlib
import operator
import requests
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# ─── Configuration ────────────────────────────────────────────────────────────
load_dotenv()  # loads GROQ_API_KEY and TAVILY_API_KEY from .env

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

REACT_SYSTEM = """You are a ReAct agent. Strictly follow this loop:
Thought → Action (tool call) → Observation → Thought → ...

RULES:
1. ALWAYS use a tool for factual information — never answer from memory.
2. For multi-part questions, make one tool call per fact.
3. ALWAYS use calculator for any arithmetic — never compute in your head.
4. Only give Final Answer AFTER all required tool calls are complete.
5. When you have gathered all needed information, provide a comprehensive Final Answer."""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TOOL DEFINITIONS (ported from the MCP tool servers)
# ═══════════════════════════════════════════════════════════════════════════════

CITY_COORDS = {
    "london":     (51.5074, -0.1278),
    "paris":      (48.8566,  2.3522),
    "new york":   (40.7128, -74.0060),
    "tokyo":      (35.6762, 139.6503),
    "karachi":    (24.8607,  67.0011),
    "lahore":     (31.5204,  74.3587),
    "islamabad":  (33.6844,  73.0479),
    "rawalpindi": (33.5651,  73.0169),
    "dubai":      (25.2048,  55.2708),
    "berlin":     (52.5200,  13.4050),
    "sydney":    (-33.8688, 151.2093),
    "chicago":    (41.8781, -87.6298),
}


@tool
def get_current_weather(city: str) -> str:
    """Get real-time current weather for a city using Open-Meteo API.
    Returns temperature, wind speed, humidity, and sky condition.
    Available cities: London, Paris, New York, Tokyo, Karachi, Lahore,
    Islamabad, Rawalpindi, Dubai, Berlin, Sydney, Chicago."""
    coords = CITY_COORDS.get(city.lower().strip())
    if not coords:
        available = ", ".join(c.title() for c in CITY_COORDS)
        return f"City '{city}' not found. Available cities: {available}"

    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&hourly=relativehumidity_2m,apparent_temperature"
    )
    try:
        data = requests.get(url, timeout=10).json()
        cw       = data.get("current_weather", {})
        temp     = cw.get("temperature", "N/A")
        wind     = cw.get("windspeed",   "N/A")
        wcode    = cw.get("weathercode",  0)
        cond     = "Sunny" if wcode < 3 else "Cloudy" if wcode < 50 else "Rainy"
        humidity = data.get("hourly", {}).get("relativehumidity_2m",  ["N/A"])[0]
        feels    = data.get("hourly", {}).get("apparent_temperature", ["N/A"])[0]
        return (
            f"Current weather in {city.title()}:\n"
            f"  Condition : {cond}\n"
            f"  Temp      : {temp} °C\n"
            f"  Feels like: {feels} °C\n"
            f"  Wind      : {wind} km/h\n"
            f"  Humidity  : {humidity}%"
        )
    except Exception as e:
        return f"Weather API error: {e}"


@tool
def search_web(query: str) -> str:
    """Search the web for real-time information.
    Use this for factual questions, current events, people, history,
    or any general knowledge lookups."""
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        results = response.get("results", [])
        if not results:
            return f"No results found for: '{query}'"
        return "\n\n".join(
            f"[{i+1}] {r['title']}\n    {r['content']}"
            for i, r in enumerate(results)
        )
    except Exception as e:
        return f"Search error: {e}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a full mathematical expression safely.
    Supports: +, -, *, /, **, sqrt, log, sin, cos, pi, e, abs, round.
    Examples: '2026 - 1991', 'sqrt(144)', '15 * 8 + 20'"""
    try:
        safe_globals = {
            "__builtins__": {},
            "sqrt":  mathlib.sqrt,
            "log":   mathlib.log,
            "log2":  mathlib.log2,
            "log10": mathlib.log10,
            "sin":   mathlib.sin,
            "cos":   mathlib.cos,
            "tan":   mathlib.tan,
            "ceil":  mathlib.ceil,
            "floor": mathlib.floor,
            "pi":    mathlib.pi,
            "e":     mathlib.e,
            "abs":   abs,
            "round": round,
            "pow":   pow,
        }
        result = eval(expression, safe_globals)
        return f"{expression} = {round(float(result), 6)}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except NameError as err:
        return f"Error: Unknown function — {err}"
    except SyntaxError:
        return f"Error: Invalid syntax in '{expression}'"
    except Exception as err:
        return f"Error evaluating '{expression}': {err}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    input: str                                          # User query
    messages: Annotated[list, operator.add]              # LLM conversation history
    agent_scratchpad: Annotated[str, operator.add]       # Accumulated reasoning log
    final_answer: str                                    # Final response to user
    steps: Annotated[list, operator.add]                 # Action/observation tracking


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TOOLS SETUP — bind tools to LLM
# ═══════════════════════════════════════════════════════════════════════════════

tools_list = [get_current_weather, search_web, calculator]
tools_map  = {t.name: t for t in tools_list}
llm_with_tools = llm.bind_tools(tools_list)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NODE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def react_node(state: AgentState) -> dict:
    """
    ReAct reasoning node.
    Calls the LLM with the full message history and tool definitions.
    Produces EITHER an action (tool call) OR a final answer.
    """
    response = llm_with_tools.invoke(state["messages"])
    scratchpad_delta = ""

    if response.tool_calls:
        # ── LLM decided to call one or more tools ──
        thought = response.content or ""
        if thought:
            scratchpad_delta += f"\nThought: {thought}"
        for tc in response.tool_calls:
            scratchpad_delta += f"\nAction: {tc['name']}({tc['args']})"
            print(f"  [Action] {tc['name']} | Args: {tc['args']}")

        return {
            "messages": [response],
            "agent_scratchpad": scratchpad_delta,
        }
    else:
        # ── LLM produced a final answer ──
        scratchpad_delta += f"\nFinal Answer: {response.content}"
        print(f"\n[Final Answer]\n{response.content}")

        return {
            "messages": [response],
            "agent_scratchpad": scratchpad_delta,
            "final_answer": response.content,
        }


def tool_node(state: AgentState) -> dict:
    """
    Tool execution node.
    Reads the last AI message's tool_calls, executes each tool,
    and stores the observations back into the state.
    """
    last_message = state["messages"][-1]
    new_tool_messages = []
    scratchpad_delta  = ""
    new_steps         = []

    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        print(f"  [Executing] {tool_name}({tool_args})")

        # Execute the tool
        result      = tools_map[tool_name].invoke(tool_args)
        observation = str(result)

        print(f"  [Observation] {observation[:200]}")

        # Add ToolMessage so the LLM sees the result
        new_tool_messages.append(
            ToolMessage(content=observation, tool_call_id=tc["id"])
        )

        # Update scratchpad
        scratchpad_delta += f"\nObservation: {observation}"

        # Track this step
        new_steps.append({
            "action":      tool_name,
            "args":        tool_args,
            "observation": observation,
        })

    return {
        "messages":          new_tool_messages,
        "agent_scratchpad":  scratchpad_delta,
        "steps":             new_steps,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONDITIONAL ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> str:
    """
    Routing logic after react_node:
      - If the last message has tool_calls → route to 'tool_node'
      - Otherwise (final answer)           → route to 'end'
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return "end"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Construct the LangGraph workflow:
        START → react_node → (if action → tool_node → react_node)
                           → (if final  → END)
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("react_node", react_node)
    workflow.add_node("tool_node",  tool_node)

    # Entry point
    workflow.set_entry_point("react_node")

    # Conditional edge: react_node decides next step
    workflow.add_conditional_edges(
        "react_node",
        should_continue,
        {
            "tool_node": "tool_node",   # action  → execute tool
            "end":       END,           # final   → stop
        },
    )

    # After tool execution, always loop back to react_node
    workflow.add_edge("tool_node", "react_node")

    return workflow.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RUN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent(query: str, max_iterations: int = 25) -> dict:
    """
    Run the ReAct LangGraph agent on a user query.
    Returns the full final state including final_answer, steps, and scratchpad.
    """
    app = build_graph()

    # Initial state
    initial_state = {
        "input": query,
        "messages": [
            SystemMessage(content=REACT_SYSTEM),
            HumanMessage(content=query),
        ],
        "agent_scratchpad": "",
        "final_answer": "",
        "steps": [],
    }

    print(f"{'='*60}")
    print(f"  ReAct LangGraph Agent")
    print(f"  Query: {query}")
    print(f"{'='*60}\n")

    # Invoke the graph with a recursion limit
    result = app.invoke(initial_state, config={"recursion_limit": max_iterations})

    # Print summary
    print(f"\n{'='*60}")
    print("  AGENT SCRATCHPAD (full reasoning trace):")
    print(f"{'='*60}")
    print(result["agent_scratchpad"])

    print(f"\n{'='*60}")
    print(f"  STEPS TAKEN: {len(result['steps'])}")
    print(f"{'='*60}")
    for i, step in enumerate(result["steps"], 1):
        obs_preview = step["observation"][:100]
        print(f"  Step {i}: {step['action']}({step['args']}) -> {obs_preview}...")

    return result
