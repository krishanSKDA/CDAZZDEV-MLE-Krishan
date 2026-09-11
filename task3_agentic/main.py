from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TASK1_SRC = (ROOT.parent / "task1_financial" / "src").resolve()
if str(TASK1_SRC) not in sys.path:
    sys.path.insert(0, str(TASK1_SRC))


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

from multi_agent import run_multi_agent_pipeline
from single_agent import run_agent


def main() -> None:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MISTRAL_API_KEY is not set. Export it before running this script. "
            "Example: $env:MISTRAL_API_KEY='your_key'"
        )

    ticker = os.environ.get("TICKER", "AAPL")
    query = (
        f"Analyse the current financial health and market sentiment of {ticker}. "
        "Identify the top three risks to its share price over the next 90 days "
        "and suggest one data-driven hedge strategy."
    )

    single = run_agent(query, verbose=False)
    print("=== Single-agent result ===")
    print(single["final_report"])
    print(f"Trace length: {len(single['trace'])}")

    multi = run_multi_agent_pipeline(ticker)
    print("=== Multi-agent result ===")
    print(multi.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
