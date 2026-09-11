# Running Guide - Task 1 (Financial AI Equity Research Assistant)

No GPU needed. Should take ~10-15 minutes including API key setup.

## 1. Get a Gemini API key
1. Go to https://aistudio.google.com/app/apikey and create an API key.
2. Copy the key and keep it secure.
3. Set it as the environment variable `GEMINI_API_KEY` before running the notebook or scripts.

## 2. Push the repo to GitHub (if you haven't yet)
1. Create a new **public** repo named `CDAZZDEV-MLE-[YourName]` on github.com.
2. From the unzipped folder locally:
   ```
   cd CDAZZDEV-MLE-Candidate
   git init
   git add .
   git commit -m "Initial submission scaffold"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/CDAZZDEV-MLE-YourName.git
   git push -u origin main
   ```

## 3. Open the notebook in Colab
- Easiest: go to https://colab.research.google.com > File > Open notebook >
  GitHub tab > paste your repo URL > select `task1_financial/notebook_task1_equity_research.ipynb`.
- Alternative: upload the `.ipynb` file directly via File > Upload notebook,
  then also upload the `task1_financial/src/` folder into the Colab file
  browser (left sidebar, folder icon) so the imports resolve.

## 4. Add your Gemini key as a Colab Secret (do NOT paste it into a cell)
1. Click the key icon in Colab's left sidebar ("Secrets").
2. Add new secret: name = `GEMINI_API_KEY`, value = your key.
3. Toggle "Notebook access" on for this notebook.

## 5. Run all cells
- Runtime > Run all (or run cells one by one top to bottom).
- The pip install cell takes ~30-60 seconds.
- You'll see: the summary dict + OHLCV tail printed, per-headline sentiment
  classifications, the trade signal with justification, and a rendered HTML
  report inline at the end.

## 6. Change the ticker if you like
Edit the `TICKER = 'AAPL'` line in the third code cell to any valid ticker
(e.g. `'MSFT'`, `'TSLA'`) before running.

## 7. Before submitting
- Do **not** clear outputs (Edit > Clear all outputs is the thing to avoid).
- Save the notebook (File > Save a copy in GitHub, or download and commit
  the executed `.ipynb` back to your repo).
- Double-check no API key text appears anywhere in the saved notebook.

## Common issues
| Symptom | Likely cause | Fix |
|---|---|---|
| `EnvironmentError: GEMINI_API_KEY not set` | Secret not added or toggle off | Re-check step 4, make sure "Notebook access" is on |
| `ModuleNotFoundError: No module named 'indicators'` | `src/` not visible to the notebook | Make sure you opened the notebook *from within* the cloned repo structure, or manually upload `src/` next to the notebook in Colab's file browser |
| News list has fewer than 10 headlines | yfinance's news endpoint varies by ticker/time | Try a large-cap ticker (AAPL, MSFT); if still short, note it as a limitation in REFLECTION.md, or add an RSS fallback |
| `Signal generation failed validation` in the HTML report | LLM produced a justification under 3 sentences | Re-run that cell; the retry logic already handles one retry, a second manual re-run usually succeeds |
