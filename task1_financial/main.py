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

from data_pipeline import run_pipeline
from llm_reasoning import _get_client, classify_all_headlines, generate_trade_signal
from report_renderer import render_report


def main() -> None:
    ticker = os.environ.get("TICKER", "AAPL")
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MISTRAL_API_KEY is not set. Export it before running this script. "
            "Example: $env:MISTRAL_API_KEY='your_key'"
        )

    result = run_pipeline(ticker)
    client = _get_client()
    sentiment_results, aggregate = classify_all_headlines(client, result["news"])
    signal = generate_trade_signal(
        client,
        ticker,
        {
            "Close": result["summary"].current_price,
            "SMA_50": result["ohlcv"].tail(1)["SMA_50"].iloc[0] if "SMA_50" in result["ohlcv"].columns else None,
            "SMA_200": result["ohlcv"].tail(1)["SMA_200"].iloc[0] if "SMA_200" in result["ohlcv"].columns else None,
            "RSI_14": result["ohlcv"].tail(1)["RSI_14"].iloc[0] if "RSI_14" in result["ohlcv"].columns else None,
            "MACD": result["ohlcv"].tail(1)["MACD"].iloc[0] if "MACD" in result["ohlcv"].columns else None,
            "MACD_signal": result["ohlcv"].tail(1)["MACD_signal"].iloc[0] if "MACD_signal" in result["ohlcv"].columns else None,
            "MACD_hist": result["ohlcv"].tail(1)["MACD_hist"].iloc[0] if "MACD_hist" in result["ohlcv"].columns else None,
            "BB_percent_b": result["ohlcv"].tail(1)["BB_percent_b"].iloc[0] if "BB_percent_b" in result["ohlcv"].columns else None,
            "fifty_two_week_low": result["summary"].fifty_two_week_low,
            "fifty_two_week_high": result["summary"].fifty_two_week_high,
            "ytd_return_pct": result["summary"].ytd_return_pct,
        },
        aggregate,
    )

    html = render_report(
        ticker,
        result["summary"],
        sentiment_results,
        aggregate,
        signal,
        result["ohlcv"],
    )

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / f"{ticker}_brief.html"
    report_path.write_text(html, encoding="utf-8")

    print(f"Ticker: {ticker}")
    print(f"Prices and indicators fetched for {len(result['ohlcv'])} rows")
    print(f"News headlines processed: {len(result['news'])}")
    print(f"Aggregate sentiment: {aggregate.overall_score}")
    print(f"Trade signal: {signal.signal if signal else 'N/A'}")
    print(f"HTML report saved to: {report_path}")


if __name__ == "__main__":
    main()
