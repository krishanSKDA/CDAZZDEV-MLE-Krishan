# Running Guide - Task 3 (Multi-Agent Financial Research System)

No GPU needed. Should take ~15-20 minutes. Do this after Task 1 so your repo
is already pushed.

## 1. Prerequisites
- Same Gemini API key as Task 1.
- Repo pushed to GitHub with `task1_financial/src/` present too -- Task 3
  reuses `indicators.py` from Task 1 for `get_price_data`.

## 2. Open the notebook in Colab
File > Open notebook > GitHub tab > your repo >
`task3_agentic/notebook_task3_agentic.ipynb`.

If you're uploading manually instead of opening from GitHub, make sure the
Colab file browser has this exact relative layout so the `sys.path.append`
calls in the notebook work:
```
/content/task3_agentic/src/...
/content/task1_financial/src/indicators.py
```

## 3. Add the Colab Secret
Same as Task 1: key icon > add `GEMINI_API_KEY` > toggle notebook access on.

## 4. Run all cells top to bottom
Watch for these sections in order:
1. **Task 3A** - the single agent's tool-call trace prints live: you'll see
   lines like `--- Step 0: agent decided to call get_news({...}) ---`
   followed by an observation, then the next decision. This is the
   "observe and replan" cycle the rubric wants -- it should visibly happen
   more than once before the final answer.
2. **Short-term memory demo** - confirms the second call to the same tool
   is a cache hit (`True`) with no re-fetch.
3. **Task 3B** - multi-agent run: watch for `[Agent A]` and `[Agent B]`
   prefixed lines, and specifically the `===== Critique loop =====` section
   where Agent B asks Agent A for the sentiment score it's missing.
4. **Persistent cache demo** - saves the report to `logs/cache/`, then
   reloads it to prove the cache hit.
5. **Observability** - prints the last 5 entries from `agent_trace.jsonl`.

## 5. Verify `agent_trace.jsonl` was generated
```python
!cat task3_agentic/logs/agent_trace.jsonl | wc -l
```
Should be nonzero. This file is a **mandatory** deliverable for Task 3 --
commit it to your repo after running.

## 6. Commit the trace log and executed notebook
After running, download or sync `task3_agentic/logs/agent_trace.jsonl` and
the executed notebook back into your local repo, then:
```
git add task3_agentic/logs/agent_trace.jsonl task3_agentic/notebook_task3_agentic.ipynb
git commit -m "Task 3: executed run with agent trace log"
git push
```

## 7. Change the ticker if you like
Edit `TICKER = 'AAPL'` in the third code cell.

## Common issues
| Symptom | Likely cause | Fix |
|---|---|---|
| Agent calls `get_news` before it has a reason to | Model's own reasoning, not a bug | This is fine -- the rubric wants autonomous ordering, not a specific order. Just make sure it isn't the exact same fixed order every single run (try 2 different tickers to demonstrate variation) |
| `duckduckgo_search` returns empty results | Rate limiting on the free library | Re-run the cell after a short pause, or reduce `max_results` |
| Critique loop doesn't trigger | Agent A already populated `news_sentiment_score` some other way | Check `multi_agent.py`'s `maybe_request_clarification` -- by design it only fires when that field is `None`, which is the expected first-run behaviour |
| `KeyError` on `GEMINI_API_KEY` inside `tools.py` | Secret not exported to `os.environ` yet | Make sure you ran the secrets cell (step 3) *before* any tool-using cell |
