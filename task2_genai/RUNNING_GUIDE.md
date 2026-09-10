# Running Guide - Task 2 (Fine-Tuning Pipeline)

**Needs a GPU. Budget 1-2 hours** including dataset generation (~10-15 min of
API calls), training (~20-40 min on a T4 for 3 epochs on ~120 examples), and
evaluation (~15-20 min, mostly model loading + generation). Do this last.

## 1. Prerequisites
- Groq API key (same as Tasks 1 & 3).
- (Optional but recommended) Hugging Face account + access token, so you can
  push the merged model: huggingface.co/settings/tokens > New token > role
  "write".
- (Optional) Weights & Biases free account for loss-curve logging:
  wandb.ai > sign up > copy API key from wandb.ai/authorize. If you skip
  this, the notebook still prints train/eval loss to console -- that alone
  satisfies the "loss monitoring" requirement, W&B is just nicer.

## 2. Open the notebook and switch to a GPU runtime
1. Open `task2_genai/notebook_task2_finetune.ipynb` in Colab (same GitHub-open
   method as Tasks 1/3).
2. **Runtime > Change runtime type > T4 GPU > Save.** This is easy to forget
   and training will fail or be unusably slow without it.

## 3. Add Colab Secrets
Key icon > left sidebar:
- `GROQ_API_KEY` (required)
- `HF_TOKEN` (optional -- only needed to push the merged model to the Hub;
  if you skip it, the model still saves locally and you can use the Google
  Drive alternative deliverable instead)

## 4. Run Task 2A cells (dataset generation)
- This calls Groq ~150 times (one per training example) -- expect a few
  minutes, and Groq's free tier has rate limits, so if you hit a 429 error,
  just re-run the cell; it will pick up where reasonable (or reduce `n=150`
  to `n=100` for the minimum requirement if you're rate-limited).
- **Check the diversity report output carefully.** If `max_single_topic_share`
  is printed with a warning, don't proceed to fine-tuning yet -- the rubric
  gives zero marks for a homogeneous dataset. Re-run generation if needed.
- Confirms `train.jsonl` / `val.jsonl` / `test.jsonl` are written with an
  80/10/10 split -- the printed sizes should roughly match.

## 5. Run Task 2B cells (QLoRA fine-tuning)
1. `wandb.login()` will prompt for your API key inline (paste it, or run
   `wandb.init(mode='disabled')` instead if you're skipping W&B).
2. Model download (~7-8GB for Phi-3-mini) takes a few minutes on first run.
3. Training prints loss per logging step; **watch that `eval_loss` decreases
   across the 3 epochs** (printed automatically after training in the next
   cell) -- this is a scored requirement, not optional.
4. If you hit a CUDA out-of-memory error: the exact fallback sequence
   (reduce batch size -> reduce max_seq_length -> enable gradient
   checkpointing) is documented as a comment directly above `trainer.train()`
   in `finetune_qlora.py`. Apply it, **and paste the actual error + fix into
   REFLECTION.md** -- the rubric explicitly rewards documenting this.
5. At the end, the merged model saves to
   `compliance-clause-classifier-lora-merged/` and pushes to HF Hub if
   `HF_TOKEN` was set.

## 6. Run Task 2C cells (evaluation)
1. Loads the *base* (non-fine-tuned) Phi-3-mini fresh and generates on the
   test set -- this is your baseline comparison.
2. Loads your *merged fine-tuned* model from step 5 and generates on the
   same test set.
3. `run_comparison(...)` prints the ROUGE-L and BERTScore-F1 table -- the
   fine-tuned row should score higher than the base row on both. If it
   doesn't, that's a real result to discuss honestly in REFLECTION.md, not
   something to hide.
4. LLM-as-judge cell scores each fine-tuned response 1-5 on three
   dimensions via Groq.
5. `manual_review_and_hallucination_rate(...)` currently has a **placeholder**
   (`label = "correct"` for every response) -- you must edit this to
   actually read each of the 10 printed responses and assign a real label
   (`correct` / `partially_correct` / `hallucinated`) before this counts for
   marks. Easiest: replace the placeholder line with manual entries after
   reading the printed output, e.g.:
   ```python
   manual_labels = ["correct", "partially_correct", "correct", "hallucinated", ...]  # your real judgments
   ```
   and adapt the function accordingly, or just hand-edit the loop.
6. Fill in the two qualitative-analysis paragraphs at the bottom of the
   notebook (markdown cell) with real, specific examples from what you
   just saw -- this is separately scored and generic text will lose marks.

## 7. Save everything back to the repo
- Download the executed notebook (or sync via git in Colab) with all
  outputs visible.
- Commit `train.jsonl`, `val.jsonl`, `test.jsonl` if you want the exact
  dataset reproducible (optional but good practice).
- Add your HF model link (or Google Drive link with public view access) to
  `task2_genai/README.md`.

## Common issues
| Symptom | Likely cause | Fix |
|---|---|---|
| CUDA OOM during training | T4's 16GB exceeded | See step 5.4 and the comment in `finetune_qlora.py` |
| `bert_score` import/download hangs | First-time download of the BERT scoring model (~400MB) | Just wait; it only downloads once per Colab session |
| Groq 429 rate-limit errors during dataset generation | Free tier request-per-minute limit | Add a short `time.sleep(1)` between calls in `generate_dataset`, or reduce `n` |
| `push_to_hub` fails with 401 | `HF_TOKEN` missing or lacks "write" role | Regenerate the token with write access, re-add as a Colab secret |
| Fine-tuned ROUGE-L is *lower* than base | Possible: too few epochs, dataset too homogeneous, or LR too high/low | Don't fudge results -- document it honestly in REFLECTION.md and note what you'd try (more epochs, different rank, cleaner data) |
