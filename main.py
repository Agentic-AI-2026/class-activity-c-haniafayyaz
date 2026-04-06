# ═══════════════════════════════════════════════════════════════════════════════
# main.py — Entry point for the LangGraph ReAct Agent
# ═══════════════════════════════════════════════════════════════════════════════

from graph import run_agent

# ─── Test Query ───────────────────────────────────────────────────────────────
query = (
    "What is the weather in Lahore and who is the current Prime Minister of Pakistan? "
    "Now get the age of PM and tell us will this weather suits PM health."
)

# ─── Run the agent ────────────────────────────────────────────────────────────
result = run_agent(query)

# ─── Display Final Answer ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL ANSWER")
print("=" * 60)
print(result["final_answer"])
