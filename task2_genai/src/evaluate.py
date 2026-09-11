"""
Task 2C - Evaluation and Baseline Comparison

Compares the base model (system prompt only, no fine-tuning) against the
fine-tuned model on the held-out test set, using:
  1. ROUGE-L (structural/lexical overlap with gold JSON)
  2. BERTScore F1 (semantic similarity, catches paraphrased-but-correct
     answers that ROUGE-L would under-score) -- an LLM-as-judge is
     provided as an alternative/additional metric.
  3. Manual hallucination review on >= 10 fine-tuned responses.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from rouge_score import rouge_scorer
from bert_score import score as bertscore
from mistralai import Mistral

from schemas import JudgeScore, ManualReviewLabel

JUDGE_MODEL = "mistral-large-latest"

JUDGE_SYSTEM_PROMPT = """You are grading a financial compliance clause
classifier's output against a gold-standard answer. Score on a 1-5 scale
for each dimension. Respond with ONLY a JSON object:
{
  "correctness": <1-5, does clause_type and risk_level match the gold answer>,
  "completeness": <1-5, are all four required fields present and populated>,
  "domain_accuracy": <1-5, is the flag_reason grounded in the actual clause text>,
  "overall": <float, average of the three above>,
  "rationale": "<one sentence>"
}"""


@dataclass
class ComparisonResult:
    rouge_l_base: float
    rouge_l_finetuned: float
    bertscore_f1_base: float
    bertscore_f1_finetuned: float
    hallucination_rate_pct: float


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores) if scores else 0.0


def compute_bertscore_f1(predictions: list[str], references: list[str]) -> float:
    _, _, f1 = bertscore(predictions, references, lang="en", verbose=False)
    return float(f1.mean())


def llm_judge(client: Mistral, prediction: str, reference: str) -> JudgeScore | None:
    try:
        resp = client.chat.complete(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Gold answer: {reference}\n\nModel output: {prediction}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = json.loads(resp.choices[0].message.content)
        return JudgeScore(**raw)
    except Exception as exc:
        print(f"Judge call failed: {exc}")
        return None


def run_comparison(
    base_predictions: list[str], finetuned_predictions: list[str], references: list[str],
) -> ComparisonResult:
    assert len(base_predictions) == len(finetuned_predictions) == len(references), \
        "predictions and references must be the same length (identical test set)"

    rouge_base = compute_rouge_l(base_predictions, references)
    rouge_ft = compute_rouge_l(finetuned_predictions, references)

    bert_base = compute_bertscore_f1(base_predictions, references)
    bert_ft = compute_bertscore_f1(finetuned_predictions, references)

    print("=== ROUGE-L / BERTScore F1 comparison ===")
    print(f"{'Model':<15}{'ROUGE-L':>12}{'BERTScore-F1':>15}")
    print(f"{'Base':<15}{rouge_base:>12.4f}{bert_base:>15.4f}")
    print(f"{'Fine-tuned':<15}{rouge_ft:>12.4f}{bert_ft:>15.4f}")

    return ComparisonResult(
        rouge_l_base=rouge_base, rouge_l_finetuned=rouge_ft,
        bertscore_f1_base=bert_base, bertscore_f1_finetuned=bert_ft,
        hallucination_rate_pct=None,  # filled in by manual_review_and_hallucination_rate
    )


def manual_review_and_hallucination_rate(
    predictions: list[str], min_reviewed: int = 10,
) -> tuple[list[ManualReviewLabel], float]:
    """
    Interactive-style manual review. In the notebook, replace the
    placeholder label assignment below with your own read of each
    fine-tuned response against its source clause.

    A response is:
      correct           - clause_type & risk_level match, flag_reason is grounded
      partially_correct - one field wrong OR flag_reason vague but not fabricated
      hallucinated       - invents a clause_type outside the taxonomy, or cites
                            a risk driver / fact not present in the input clause
    """
    if len(predictions) < min_reviewed:
        raise ValueError(f"Need at least {min_reviewed} responses to review, got {len(predictions)}")

    labels: list[ManualReviewLabel] = []
    for i, pred in enumerate(predictions[:min_reviewed]):
        print(f"\n--- Response {i} ---\n{pred}")
        # NOTE: in the actual Colab run, replace this with a real judgment
        # per response (e.g. `label = input("correct/partially_correct/hallucinated: ")`)
        # and record it below. Placeholder shown for pipeline demonstration:
        label = "correct"
        labels.append(ManualReviewLabel(example_id=i, label=label))

    hallucinated = sum(1 for l in labels if l.label == "hallucinated")
    rate = 100 * hallucinated / len(labels)
    print(f"\nHallucination rate: {rate:.1f}% ({hallucinated}/{len(labels)})")
    return labels, rate
