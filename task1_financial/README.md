# Task 1 - Financial AI: LLM-Powered Equity Research Assistant

## Structure
- `src/indicators.py` - SMA/RSI/MACD/Bollinger Bands from first principles (no TA-Lib)
- `src/data_pipeline.py` - Task 1A: OHLCV fetch, news retrieval, summary dict, robustness
- `src/schemas.py` - Pydantic models enforcing structured LLM output
- `src/llm_reasoning.py` - Task 1B: headline sentiment + Buy/Hold/Sell signal reasoning via Gemini
- `src/report_renderer.py` - Bonus: styled HTML brief with embedded matplotlib chart
- `notebook_task1_equity_research.ipynb` - Colab notebook wiring it all together

## Run in Colab
1. Open `notebook_task1_equity_research.ipynb` in Google Colab.
2. Add your Gemini API key as a Colab secret named `GEMINI_API_KEY`.
3. Run all cells top to bottom. Change `TICKER` to analyse a different stock.

## Colab badge
Replace `YOUR_GITHUB_USERNAME` after pushing to GitHub:
`[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/CDAZZDEV-MLE-YourName/blob/main/task1_financial/notebook_task1_equity_research.ipynb)`

## Design notes
- Momentum signal in `data_pipeline.py` is a simple rule-based feature fed
  *into* the LLM prompt -- the LLM is required to reason over the combination
  of indicators for the final Buy/Hold/Sell call, not just echo this feature.
- All external calls (price, news, P/E) fail soft: partial data produces
  `data_warnings` rather than raising, so a single missing field never crashes
  the whole pipeline.
- Prompts live as module-level constants in `llm_reasoning.py`, decoupled from
  the calling code, per the prompt-engineering criterion.
