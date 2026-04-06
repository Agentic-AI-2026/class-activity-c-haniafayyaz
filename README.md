# Quiz: Convert ReAct Agent to LangGraph 🦜🕸️

## Objective
Convert a standard working **ReAct agent** (implemented in LangChain) into a **LangGraph workflow**. Your implementation must preserve the iterative reasoning and tool-usage behavior inherent to the ReAct framework.

---

## 🛠 Provided Resources
* **Existing ReAct agent code** (LangChain-based).
* **Tool implementations** (functional and ready for use).

---

## 📋 Requirements

### 1. Define State
Create a state structure (TypedDict or Pydantic) to represent the workflow. Your state must include:
* `input`: The original user query.
* `agent_scratchpad`: Stores intermediate reasoning (Thoughts, Actions, Observations).
* `final_answer`: The final response delivered to the user.
* `steps`: (Optional) A list to track history of actions and observations.

### 2. ReAct Node (Reasoning + Action)
Implement a node that:
1.  Takes the current state.
2.  Calls the LLM using **ReAct-style prompting**.
3.  Produces either an **Action** (tool name + arguments) or a **Final Answer**.
4.  Updates the state accordingly.

### 3. Tool Execution Node
Implement a node that:
1.  Executes the tool selected by the ReAct node.
2.  Passes the correct arguments to the tool.
3.  Stores the **Observation** (result) back in the state.
4.  Updates the scratchpad to prepare for the next reasoning step.

### 4. Graph Flow
Construct a LangGraph workflow that follows this logic:

> **START** $\rightarrow$ `react_node` $\rightarrow$ **Conditional Edge**
> * If **Action** $\rightarrow$ `tool_node` $\rightarrow$ `react_node`
> * If **Final Answer** $\rightarrow$ **END**

**The graph must:**
* Support iterative reasoning loops.
* Continue execution until a terminal state (Final Answer) is reached.

### 5. Conditional Routing
Implement the router logic to determine the next step based on the model's output:
- `is_action` $\rightarrow$ Route to `tool_node`.
- `is_final` $\rightarrow$ Route to `END`.

---

## 🧪 Test Case
Your implementation should successfully process complex, multi-step queries such as:

> *"What is the weather in Lahore and who is the current Prime Minister of Pakistan? Now get the age of PM and tell us will this weather suits PM health."*

---

## ⚠️ Constraints
* **No Hardcoding:** Do not hardcode outputs; the logic must be dynamic.
* **Reasoning Integrity:** Maintain the "Thought $\rightarrow$ Action $\rightarrow$ Observation" flow.
* **Scalability:** The agent must be capable of calling tools multiple times in a single run.
* **State Management:** Ensure proper state updates to prevent infinite loops or data loss between iterations.

---

## 🚀 Submission
Push your solution to this repository using the following structure:
```text
.
├── main.py          # Entry point for execution
├── graph.py         # LangGraph definition (optional)
└── README.md        # Project documentation
```

---

## ✅ Solution

### How It Works

The agent follows the **ReAct loop**:
```
Thought → Action (tool call) → Observation → Thought → ... → Final Answer
```

### Graph Flow
```
START → react_node → [tool_node → react_node]* → END
```

- **`react_node`** — Calls the LLM (`llama-3.1-8b-instant` via Groq) with bound tools. Produces either tool calls or a final answer, and updates `agent_scratchpad`.
- **`tool_node`** — Executes the chosen tool(s), stores observations back into state, and updates the scratchpad.
- **Conditional routing (`should_continue`)** — If the last message contains tool calls → `tool_node`; otherwise → `END`.

### State (`AgentState`)

| Field              | Type   | Description                                  |
|--------------------|--------|----------------------------------------------|
| `input`            | `str`  | Original user query                          |
| `messages`         | `list` | Full LLM conversation history (accumulated) |
| `agent_scratchpad` | `str`  | Accumulated Thought / Action / Observation   |
| `final_answer`     | `str`  | Final response from the agent                |
| `steps`            | `list` | Per-step action + observation records        |

### Tools

| Tool                  | Description                              |
|-----------------------|------------------------------------------|
| `get_current_weather` | Real-time weather via Open-Meteo API     |
| `search_web`          | Web search via Tavily API                |
| `calculator`          | Safe arithmetic expression evaluator     |

### Setup & Run

```bash
pip install -r requirements.txt
python main.py
```

### LLM
- **Model:** `llama-3.1-8b-instant` via [Groq](https://groq.com)

---

## 🖥️ Sample Output

```
============================================================
  ReAct LangGraph Agent
  Query: What is the weather in Lahore and who is the current Prime
         Minister of Pakistan? Now get the age of PM and tell us will
         this weather suits PM health.
============================================================

  [Action] get_current_weather | Args: {'city': 'Lahore'}
  [Action] search_web | Args: {'query': 'current Prime Minister of Pakistan'}
  [Action] calculator | Args: {'expression': '2024 - 1962'}
  [Action] search_web | Args: {'query': 'health effects of weather in Lahore'}

  [Executing] get_current_weather({'city': 'Lahore'})
  [Observation] Current weather in Lahore:
    Condition : Sunny
    Temp      : 30.1 °C
    Feels like: 18.7 °C
    Wind      : 4.8 km/h
    Humidity  : 76%

  [Executing] search_web({'query': 'current Prime Minister of Pakistan'})
  [Observation] Mian Muhammad Shehbaz Sharif is currently serving as
                the 24th Prime Minister of Pakistan since March 2024.

  [Executing] calculator({'expression': '2024 - 1962'})
  [Observation] 2024 - 1962 = 62.0

  [Executing] search_web({'query': 'health effects of weather in Lahore'})
  [Observation] Smog in Lahore caused serious health problems —
                breathing difficulties, coughing (AQI 180-340).

[Final Answer]
The current weather in Lahore is partly cloudy with a temperature of
30.0°C and a humidity of 40%. The wind speed is 7.6 km/h.

The current Prime Minister of Pakistan is Shehbaz Sharif, who is
approximately 62 years old.

Considering the high temperature, humidity, and Lahore's historically
poor air quality, this weather is NOT suitable for the Prime Minister's
health — it can cause breathing difficulties and other heat-related
health problems.

============================================================
  STEPS TAKEN: 4
============================================================
  Step 1: get_current_weather({'city': 'Lahore'})
  Step 2: search_web({'query': 'current Prime Minister of Pakistan'})
  Step 3: calculator({'expression': '2024 - 1962'})
  Step 4: search_web({'query': 'health effects of weather in Lahore'})
```
