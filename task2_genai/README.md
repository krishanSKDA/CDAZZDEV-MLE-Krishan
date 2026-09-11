# Task 2 - Generative AI: Domain-Specific Fine-Tuning Pipeline

## Use case
**Financial Compliance Clause Classifier** -- given a synthetic policy/contract
clause, classify it into a fixed taxonomy of compliance clause types with a
risk level, flag reason, and recommended action. Full problem statement is in
the `dataset_generation.py` module docstring. Chosen deliberately for its
checkable, fixed-taxonomy output (needed for meaningful ROUGE-L/hallucination
evaluation), not a generic chatbot task.

## Structure
- `src/schemas.py` - Pydantic models for training examples and eval outputs
- `src/dataset_generation.py` - Task 2A: teacher-model (Mistral AI) data generation, diversity report, JSONL chat-format split
- `src/finetune_qlora.py` - Task 2B: QLoRA fine-tuning of Phi-3-mini-4k-instruct with every hyperparameter justified in comments
- `src/evaluate.py` - Task 2C: ROUGE-L, BERTScore F1, LLM-as-judge, manual hallucination review
- `notebook_task2_finetune.ipynb` - Colab notebook running 2A -> 2B -> 2C end to end

## Run in Colab
1. Runtime > Change runtime type > T4 GPU.
2. Add Colab secrets: `MISTRAL_API_KEY` (required), `HF_TOKEN` (optional, to push
   the merged model to Hugging Face Hub), W&B API key (optional, for loss
   logging -- otherwise skip `wandb.login()` and console logs are still printed).
3. Run all cells top to bottom.

## Key design decisions
- **Teacher != student**: teacher is Mistral AI; student is a
  locally fine-tuned Phi-3-mini-4k-instruct -- different models, as required.
- **Diversity enforcement**: `generate_dataset` cycles evenly through every
  (clause_type x risk_level) combination rather than sampling randomly, so no
  single topic can dominate the set; `report_diversity` flags it if one ever does.
- **All hyperparameters justified**: see inline comments in `finetune_qlora.py`
  for the reasoning behind r=16, alpha=32, target_modules, lr=2e-4, cosine
  schedule, 3 epochs, batch size 2 x grad-accum 8, max_seq_length=512.
- **OOM handling documented**: the fallback sequence (reduce batch size ->
  reduce max_seq_length -> enable gradient checkpointing) is written directly
  into `finetune_qlora.py` as the debugging playbook, to be filled in with the
  actual error message/traceback if OOM occurs during the real Colab run.
- **Two evaluation angles**: ROUGE-L catches exact-structure mismatches;
  BERTScore/LLM-judge catches semantically-correct-but-differently-worded
  flag_reason text that ROUGE-L would unfairly penalise.
