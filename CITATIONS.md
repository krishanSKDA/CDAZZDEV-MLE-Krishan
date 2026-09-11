# CITATIONS.md

Per Section 2.2 of the assessment, this file documents every instance of AI
assistance used in this submission.

## AI-assisted code (all tasks)

The initial code scaffolding for all three tasks -- the Task 1 data
pipeline and indicator math, the Task 1B Pydantic schemas and LLM
prompt/reasoning layer, the Task 2 dataset-generation/QLoRA/evaluation
modules, and the Task 3 tool implementations, single-agent ReAct loop, and
multi-agent critique-loop pipeline -- was drafted with the assistance of
Claude (Anthropic), via the claude.ai chat interface, on 2026-09-11.

The implementation was then iteratively adjusted from earlier Mistral/Groq
provider attempts to the final Gemini-based configuration because several
free-tier or restricted API paths were blocked or inconsistent in practice,
so Gemini was selected as the reliable provider for the final submission.

Representative prompts used:
- "Build a Task 1A data pipeline: yfinance OHLCV fetch (>=2yr, no hardcoded
  dates), first-principles SMA/RSI/MACD/Bollinger Bands, news retrieval,
  summary dict, graceful handling of missing data."
- "Add Pydantic-validated LLM sentiment classification per headline and a
  Buy/Hold/Sell signal that reasons over the combination of indicators, with
  prompts as separate template constants."
- "Design a fine-tuning use case with a fixed, checkable output taxonomy
  (not a generic chatbot), a teacher-model dataset generator with diversity
  reporting, a QLoRA config with every hyperparameter justified in
  comments, and a ROUGE-L / BERTScore / LLM-judge / hallucination-rate
  evaluation harness."
- "Build a single tool-using research agent with a manual ReAct loop over
  Mistral AI function-calling so the observe/replan cycle is visible, then a
  two-agent pipeline with restricted tool access, a Pydantic handoff schema,
  and a one-shot critique loop, plus an @observed_tool decorator that logs to
  agent_trace.jsonl and short-term/persistent memory helpers."

**All AI-drafted code was reviewed, adjusted, and partially validated by
running indicator math, the observability decorator, dataset diversity
reporting, JSONL formatting, and ROUGE-L scoring against synthetic inputs
before inclusion.** I take responsibility for every design decision in this
submission and can defend each one in the follow-up interview.

## What still needs to be done by the candidate before submission
- Running each notebook against a live Gemini API key and a real T4 GPU
  session in Colab (this could not be executed in the AI assistant's
  sandbox, which has no GPU and no direct access to the required external
  endpoints; some earlier free-tier/restricted provider attempts were blocked,
  which is why the final project is configured for Gemini).
- Filling in the Task 2C qualitative-analysis paragraphs with real
  observations from the actual fine-tuning run.
- Manually labelling the >= 10 fine-tuned responses in
  `manual_review_and_hallucination_rate` (currently a placeholder loop).
- Replacing `[YourName]` in the repo name and adding real screenshots/links
  (W&B run, HF model, agent_trace.jsonl from an actual run).

## Open-source code adapted
No third-party repository code was copied or adapted beyond standard,
publicly-documented library usage (yfinance, transformers, peft, trl,
bitsandbytes, google-generativeai, duckduckgo-search, rouge_score,
bert_score) per their respective official APIs/docs.

## Teacher model system prompt (Task 2A)
The full teacher system prompt used for synthetic training-data generation
is included verbatim in `task2_genai/src/dataset_generation.py` as the
`TEACHER_SYSTEM_PROMPT` constant, per the Section 2.2 requirement to include
it in an appendix/README.
