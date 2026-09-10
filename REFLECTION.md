# REFLECTION.md

*(Template -- personalise after running the notebooks; keep under 600 words.)*

## Architectural decisions

**Task 1.** I separated indicator math (`indicators.py`) from data
orchestration (`data_pipeline.py`) from LLM reasoning (`llm_reasoning.py`)
so each piece is independently testable. RSI uses genuine Wilder smoothing
(EMA with alpha=1/period) rather than the common but incorrect simple
rolling-mean shortcut. The momentum signal computed in the pipeline is
deliberately a *feature*, not the final call -- the LLM prompt requires it
to reason over the combination of SMA crossover, RSI, and MACD together,
which is what the rubric is actually testing.

**Task 2.** I chose a financial compliance clause classifier over a
generic chatbot use case specifically because it has a checkable, fixed
output taxonomy -- without that, ROUGE-L and hallucination-rate metrics
are close to meaningless. The teacher (Groq-hosted Llama-3-70B) and student
(locally fine-tuned Phi-3-mini) are deliberately different models. Every
QLoRA hyperparameter is justified inline rather than left at a library
default, per the rubric's explicit requirement.

**Task 3.** Rather than using a framework's black-box AgentExecutor, I
implemented the single agent as an explicit ReAct loop over Groq's native
tool-calling, so the "observe result, decide next action" cycle is visible
in notebook output rather than hidden inside library internals. For the
multi-agent system, tool-access restriction is enforced by construction --
each agent class only imports the tools it's allowed to call -- and the
Agent A -> Agent B handoff uses a Pydantic `DataBrief`, never a raw string.
A single `@observed_tool` decorator centralises all `agent_trace.jsonl`
logging so both the single- and multi-agent paths are captured uniformly.

## What I would improve with more time

- **Task 1**: add backtesting of the LLM's Buy/Hold/Sell signal against
  actual forward returns to give the "reasoning quality" criterion an
  empirical anchor, not just a qualitative read.
- **Task 2**: the current dataset generation is capped at 150 examples for
  Colab-session practicality; a larger, harder-negative-mined set (e.g.
  deliberately near-adjacent clause types like `conflict_of_interest` vs
  `anti_bribery_corruption`) would stress-test the classifier's actual
  discriminative power rather than easy cases. I'd also add the RAG
  fallback bonus layer, retrieving from a small ChromaDB store of real
  (public, non-confidential) regulatory guidance when model confidence is low.
- **Task 3**: extend the critique loop to more than one round, and add a
  third agent role (e.g. a risk-scoring agent) to test whether the
  handoff schema and restriction pattern scale past two agents.

## Limitations encountered

- This was built and reviewed in an AI-assistant sandbox with no GPU and no
  access to the Groq/yfinance/Hugging Face network endpoints, so the pure
  logic (indicator math, Pydantic validation, dataset diversity reporting,
  JSONL formatting, the observability decorator, ROUGE-L scoring) was
  unit-tested against synthetic inputs, but the live API/GPU paths (Groq
  calls, yfinance fetches, the actual QLoRA training run, BERTScore, and
  the LLM-judge) have not yet been executed end-to-end -- that must happen
  in Colab before submission, with real outputs left visible in the
  notebooks per the assessment's requirements.
- The manual hallucination-rate review in Task 2C is currently a
  placeholder loop; it needs real human judgment against actual model
  outputs from a completed training run, not synthetic stand-ins.
- yfinance's news schema has shifted across versions, so `get_news`
  defensively checks multiple possible response shapes; if it still
  returns fewer than 10 headlines for a given ticker in practice, a
  secondary free source (e.g. an RSS feed) should be added as a fallback.
