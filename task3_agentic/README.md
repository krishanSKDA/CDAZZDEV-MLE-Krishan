# Task 3 - Agentic Workflows: Multi-Agent Financial Research System

## Structure
- `src/tools.py` - the five required tools, each `@observed_tool`-wrapped for logging, fails soft on error
- `src/memory.py` - observability decorator, short-term (in-session) memory, persistent (on-disk) cache
- `src/schemas.py` - Pydantic schemas for the Agent A -> Agent B structured handoff
- `src/single_agent.py` - Task 3A: manual ReAct loop over Groq tool-calling; autonomous tool selection, visible observe/replan trace
- `src/multi_agent.py` - Task 3B: Data Analyst + Research Writer agents with restricted tool access and a one-shot critique loop
- `notebook_task3_agentic.ipynb` - Colab notebook running all of 3A/3B/3C end to end
- `logs/agent_trace.jsonl` - generated at runtime; committed after a real run

## Run in Colab
1. Open `notebook_task3_agentic.ipynb`.
2. Add your free Groq API key as a Colab secret named `GROQ_API_KEY`.
3. Run all cells. `logs/agent_trace.jsonl` will be populated as tools are called.

## Design notes
- **Autonomy, not a fixed pipeline**: `single_agent.py` sends the full
  conversation (including every prior tool result) back to the LLM at each
  step and lets it choose the next tool via native function-calling --
  there is no hard-coded `if/else` sequence of tool calls.
- **Tool restriction is enforced by construction**: `DataAnalystAgent` and
  `ResearchWriterAgent` each only import/call their allowed tool functions;
  Agent A never imports `web_search` or `get_news`, Agent B never imports
  `get_price_data` or `calculate_volatility`.
- **Structured handoff**: `DataBrief` (Pydantic) is the only thing passed
  from Agent A to Agent B -- never a raw string.
- **Critique loop**: `ResearchWriterAgent.maybe_request_clarification`
  inspects the brief and, if `news_sentiment_score` is missing, emits one
  `ClarificationRequest`; Agent A answers it via `respond_to_clarification`,
  and the updated brief is used for the final report.
- **Observability**: the `@observed_tool` decorator in `memory.py` is the
  single place that writes to `agent_trace.jsonl`, so every tool call from
  both the single-agent and multi-agent runs is captured uniformly.
