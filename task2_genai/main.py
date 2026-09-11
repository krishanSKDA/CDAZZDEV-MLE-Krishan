from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            os.environ.setdefault(key, value)


_load_env()

from dataset_generation import generate_dataset, report_diversity, split_dataset, to_chat_jsonl


def main() -> None:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MISTRAL_API_KEY is not set. Export it before running this script. "
            "Example: $env:MISTRAL_API_KEY='your_key'"
        )

    dataset = generate_dataset(n=150)
    print(f"Generated {len(dataset)} examples")

    diversity = report_diversity(dataset)
    print("Diversity report:")
    print(diversity)

    train, val, test = split_dataset(dataset)
    to_chat_jsonl(train, str(ROOT / "train.jsonl"))
    to_chat_jsonl(val, str(ROOT / "val.jsonl"))
    to_chat_jsonl(test, str(ROOT / "test.jsonl"))

    print(f"Saved train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Files written to {ROOT}")


if __name__ == "__main__":
    main()
