"""
Task 1A - Financial Data Pipeline

Fetches >= 2 years of OHLCV data via yfinance, computes indicators,
retrieves recent news, and produces a clean summary dictionary.

Design notes:
- Dates are computed relative to `datetime.now()`, never hardcoded, so
  the "minimum two years" window is always satisfied regardless of
  when this is run.
- Every external call (price fetch, news fetch, fundamentals lookup)
  is wrapped so a failure degrades gracefully instead of raising.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import compute_all_indicators, derive_momentum_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_pipeline")

MIN_HISTORY_YEARS = 2
MIN_NEWS_HEADLINES = 10


@dataclass
class TickerSnapshot:
    ticker: str
    current_price: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    pe_ratio: Optional[float] = None
    ytd_return_pct: Optional[float] = None
    momentum_signal: Optional[str] = None
    data_warnings: list = field(default_factory=list)


def fetch_ohlcv(ticker: str, years: int = MIN_HISTORY_YEARS) -> pd.DataFrame:
    """Fetch >= `years` of daily OHLCV data. Dates computed relative to today."""
    end = datetime.now()
    start = end - timedelta(days=365 * years + 30)  # small buffer for weekends/holidays
    try:
        df = yf.download(
            ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
            progress=False, auto_adjust=False,
        )
        if df.empty:
            raise ValueError(f"No OHLCV data returned for ticker '{ticker}'")
        # yfinance sometimes returns MultiIndex columns for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as exc:
        logger.error("OHLCV fetch failed for %s: %s", ticker, exc)
        raise


def fetch_news(ticker: str, n: int = MIN_NEWS_HEADLINES) -> list[dict]:
    """
    Retrieve recent headlines via the yfinance news endpoint.
    Returns a list of dicts: {title, publisher, link, published}.
    Never raises -- returns an empty list on failure so downstream
    code (Task 1B) can handle "no news" without crashing.
    """
    try:
        t = yf.Ticker(ticker)
        raw = t.news or []
        headlines = []
        for item in raw[:n]:
            content = item.get("content", item)  # yfinance schema has shifted versions
            title = content.get("title") or item.get("title")
            if not title:
                continue
            headlines.append(
                {
                    "title": title,
                    "publisher": (content.get("provider") or {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else item.get("publisher", "unknown"),
                    "link": (content.get("canonicalUrl") or {}).get("url")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else item.get("link", ""),
                    "published": content.get("pubDate", item.get("providerPublishTime", "")),
                }
            )
        if len(headlines) < n:
            logger.warning(
                "Only %d headlines retrieved for %s (requested %d)",
                len(headlines), ticker, n,
            )
        return headlines
    except Exception as exc:
        logger.error("News fetch failed for %s: %s", ticker, exc)
        return []


def safe_pe_ratio(ticker: str) -> Optional[float]:
    try:
        info = yf.Ticker(ticker).info
        pe = info.get("trailingPE")
        return float(pe) if pe is not None else None
    except Exception as exc:
        logger.warning("P/E lookup failed for %s: %s", ticker, exc)
        return None


def build_summary(ticker: str, df_with_indicators: pd.DataFrame) -> TickerSnapshot:
    """Assemble the required summary dictionary fields, tolerating missing data."""
    snap = TickerSnapshot(ticker=ticker)
    warnings = []

    if df_with_indicators.empty:
        warnings.append("price_history_empty")
        snap.data_warnings = warnings
        return snap

    close = df_with_indicators["Close"].dropna()
    if close.empty:
        warnings.append("close_series_empty")
        snap.data_warnings = warnings
        return snap

    snap.current_price = float(close.iloc[-1])

    lookback = close.tail(252)  # ~52 trading weeks
    snap.fifty_two_week_high = float(lookback.max())
    snap.fifty_two_week_low = float(lookback.min())

    # YTD return
    this_year = close[close.index.year == datetime.now().year]
    if len(this_year) >= 2:
        snap.ytd_return_pct = float((this_year.iloc[-1] / this_year.iloc[0] - 1) * 100)
    else:
        warnings.append("insufficient_data_for_ytd")

    snap.pe_ratio = safe_pe_ratio(ticker)
    if snap.pe_ratio is None:
        warnings.append("pe_ratio_unavailable")

    latest_row = df_with_indicators.iloc[-1]
    try:
        snap.momentum_signal = derive_momentum_signal(latest_row)
    except Exception as exc:
        logger.warning("Momentum signal derivation failed: %s", exc)
        snap.momentum_signal = "unknown"
        warnings.append("momentum_signal_failed")

    snap.data_warnings = warnings
    return snap


def run_pipeline(ticker: str) -> dict:
    """End-to-end Task 1A entry point. Returns a plain dict, ready for Task 1B."""
    ohlcv = fetch_ohlcv(ticker)
    enriched = compute_all_indicators(ohlcv)
    news = fetch_news(ticker)
    summary = build_summary(ticker, enriched)

    return {
        "ticker": ticker,
        "ohlcv": enriched,
        "news": news,
        "summary": summary,
    }


if __name__ == "__main__":
    result = run_pipeline("AAPL")
    print(result["summary"])
    print(f"Retrieved {len(result['news'])} headlines")
    print(result["ohlcv"].tail())
