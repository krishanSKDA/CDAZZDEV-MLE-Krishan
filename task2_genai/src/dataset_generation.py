"""
Task 2A - Use Case Definition and Dataset Engineering

Use case: Financial Compliance Clause Classifier.

  INPUT:  a single clause/paragraph excerpted from a financial-services
          policy document, client agreement, or internal procedure
          (synthetic, teacher-generated -- not real client documents).
  OUTPUT: a structured JSON classification:
          {
            "clause_type": one of a fixed taxonomy (see CLAUSE_TYPES),
            "risk_level": "low" | "medium" | "high",
            "flag_reason": one sentence explaining the risk_level,
            "recommended_action": one sentence, e.g. "escalate to compliance
                officer", "no action needed", "require legal review"
          }
  CORRECT vs INCORRECT: a response is correct if clause_type and
  risk_level exactly match the gold label and flag_reason references the
  actual risk driver in the clause (not a generic restatement); it is a
  hallucination if it invents a clause_type outside the taxonomy or cites
  a risk factor not present in the input text.

This is chosen deliberately over a generic chatbot/creative task: outputs
are checkable against a fixed taxonomy, which is what makes rigorous
ROUGE-L / hallucination-rate evaluation in Task 2C meaningful.

Dataset generation uses Gemini as the teacher model. The full system
prompt used is TEACHER_SYSTEM_PROMPT below (also required in the submission
per Section 2.2).
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter

import google.generativeai as genai
from pydantic import ValidationError

from schemas import TrainingExample

TEACHER_MODEL = "gemini-2.0-flash"

CLAUSE_TYPES = [
    "data_privacy", "conflict_of_interest", "anti_money_laundering",
    "suitability_obligation", "recordkeeping", "whistleblower_protection",
    "insider_trading", "fee_disclosure", "client_communication",
    "outsourcing_third_party", "cybersecurity_incident_response",
    "anti_bribery_corruption",
]

# The full teacher system prompt, included verbatim as required by the
# citation policy for teacher-model data generation (Section 2.2).
TEACHER_SYSTEM_PROMPT = f"""You are generating synthetic training data for a
financial compliance clause classifier. Given a clause_type from this fixed
taxonomy: {CLAUSE_TYPES}

Generate ONE realistic (but fictional -- do not reference real companies or
real people) policy/contract clause of 2-4 sentences that clearly belongs to
the given clause_type, at a randomly assigned risk level (low, medium, or
high severity of the underlying compliance issue).

Then produce the correct structured classification for that clause.

Respond with ONLY a JSON object, no markdown fences, in this exact schema:
{{
  "clause_text": "<the synthetic clause, 2-4 sentences>",
  "clause_type": "<one of the taxonomy values, must match what you were asked for>",
  "risk_level": "low" | "medium" | "high",
  "flag_reason": "<one sentence, must reference specific wording/content of the clause>",
  "recommended_action": "<one concrete sentence>"
}}"""

CLASSIFIER_SYSTEM_PROMPT = """You are a financial compliance clause
classifier. Given a clause excerpted from a policy or client agreement,
classify it. Respond with ONLY a JSON object, no markdown fences:
{
  "clause_type": "<taxonomy value>",
  "risk_level": "low" | "medium" | "high",
  "flag_reason": "<one sentence referencing the specific clause content>",
  "recommended_action": "<one concrete sentence>"
}"""


def _client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai


def generate_one_example(client, clause_type: str, risk_level: str) -> TrainingExample | None:
    user_prompt = f"clause_type: {clause_type}\nrisk_level: {risk_level}"
    try:
        model = client.GenerativeModel(TEACHER_MODEL)
        resp = model.generate_content(
            [
                {"text": TEACHER_SYSTEM_PROMPT},
                {"text": user_prompt},
            ],
            generation_config={
                "temperature": 0.9,
                "response_mime_type": "application/json",
            },
        )
        raw = json.loads(resp.text)
        assistant_json = json.dumps({
            "clause_type": raw["clause_type"],
            "risk_level": raw["risk_level"],
            "flag_reason": raw["flag_reason"],
            "recommended_action": raw["recommended_action"],
        })
        return TrainingExample(
            system=CLASSIFIER_SYSTEM_PROMPT,
            user=raw["clause_text"],
            assistant=assistant_json,
            topic_tag=raw["clause_type"],
        )
    except (KeyError, ValidationError, json.JSONDecodeError) as exc:
        print(f"Skipping malformed example ({clause_type}/{risk_level}): {exc}")
        return None


def generate_dataset(n: int = 150, seed: int = 42) -> list[TrainingExample]:
    """
    Generates n examples spread across the full clause_type taxonomy and
    all three risk levels, to avoid the "near-duplicate scenario" failure
    mode the rubric explicitly penalises.
    """
    random.seed(seed)
    client = _client()
    combos = [(ct, rl) for ct in CLAUSE_TYPES for rl in ["low", "medium", "high"]]
    random.shuffle(combos)
    # cycle through combos until we hit n, so coverage is even
    plan = (combos * (n // len(combos) + 1))[:n]

    examples = []
    for clause_type, risk_level in plan:
        ex = generate_one_example(client, clause_type, risk_level)
        if ex is not None:
            examples.append(ex)
    return examples


def report_diversity(examples: list[TrainingExample]) -> dict:
    """Prompt-length distribution + topic frequency, per the rubric requirement."""
    lengths = [len(ex.user.split()) for ex in examples]
    topic_counts = Counter(ex.topic_tag for ex in examples)

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    report = {
        "n_examples": n,
        "prompt_length_words": {
            "min": lengths_sorted[0] if n else None,
            "p25": lengths_sorted[n // 4] if n else None,
            "median": lengths_sorted[n // 2] if n else None,
            "p75": lengths_sorted[(3 * n) // 4] if n else None,
            "max": lengths_sorted[-1] if n else None,
        },
        "topic_frequency": dict(topic_counts),
        "unique_topics": len(topic_counts),
        "max_single_topic_share": (max(topic_counts.values()) / n) if n else 0,
    }
    if report["max_single_topic_share"] > 0.5:
        print(
            "WARNING: over half the dataset is a single topic -- "
            "diversity criterion will likely score zero. Regenerate with "
            "wider clause_type coverage."
        )
    return report


def to_chat_jsonl(examples: list[TrainingExample], path: str) -> None:
    """Writes examples as JSONL with system/user/assistant chat turns."""
    with open(path, "w") as f:
        for ex in examples:
            record = {
                "messages": [
                    {"role": "system", "content": ex.system},
                    {"role": "user", "content": ex.user},
                    {"role": "assistant", "content": ex.assistant},
                ]
            }
            f.write(json.dumps(record) + "\n")


def split_dataset(examples: list[TrainingExample], seed: int = 42):
    random.seed(seed)
    shuffled = examples[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


if __name__ == "__main__":
    data = generate_dataset(n=150)
    print(f"Generated {len(data)} examples")
    print(json.dumps(report_diversity(data), indent=2))

    train, val, test = split_dataset(data)
    print(f"Split sizes -- train: {len(train)}, val: {len(val)}, test: {len(test)}")

    to_chat_jsonl(train, "train.jsonl")
    to_chat_jsonl(val, "val.jsonl")
    to_chat_jsonl(test, "test.jsonl")
