"""
Task 3A - Five tools the research agent(s) can call.

Each tool is wrapped with @observed_tool (see memory.py) so every call
is logged to agent_trace.jsonl automatically -- this satisfies the
Task 3C observability requirement without scattering logging code
through the agent logic itself.

Every tool fails soft: on error it returns a dict with an "error" key
rather than raising, so the calling agent can see the failure and
decide on an alternative approach (Task 3A error-handling criterion).
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from duckduckgo_search import DDGS
import google.generativeai as genai

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "task1_financial", "src"))
from indicators import compute_all_indicators  # reuse Task 1's indicator math

from memory import observed_tool


@observed_tool
def get_price_data(ticker: str, period: str = "1y") -> dict:
    """Wraps yfinance; returns OHLCV data with computed indicators as JSON-safe dict."""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if df.empty:
            return {"error": f"no price data for {ticker}"}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        enriched = compute_all_indicators(df)
        latest = enriched.iloc[-1].replace({np.nan: None}).to_dict()
        return {
            "ticker": ticker,
            "latest_close": latest.get("Close"),
            "indicators": {
                "SMA_50": latest.get("SMA_50"),
                "SMA_200": latest.get("SMA_200"),
                "RSI_14": latest.get("RSI_14"),
                "MACD": latest.get("MACD"),
                "MACD_signal": latest.get("MACD_signal"),
                "BB_percent_b": latest.get("BB_percent_b"),
            },
            "rows_returned": len(enriched),
        }
    except Exception as exc:
        return {"error": str(exc)}


@observed_tool
def get_news(ticker: str, n: int = 10) -> dict:
    """Retrieves recent headlines for a ticker via yfinance's news endpoint."""
    try:
        raw = yf.Ticker(ticker).news or []
        headlines = []
        for item in raw[:n]:
            content = item.get("content", item)
            title = content.get("title") or item.get("title")
            if title:
                headlines.append(title)
        if not headlines:
            return {"error": f"no news found for {ticker}", "headlines": []}
        return {"ticker": ticker, "headlines": headlines}
    except Exception as exc:
        return {"error": str(exc), "headlines": []}


@observed_tool
def calculate_volatility(ticker: str, window: int = 30) -> dict:
    """Computes annualised historical volatility over a rolling window (trading days)."""
    try:
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if df.empty:
            return {"error": f"no price data for {ticker}"}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        recent = log_returns.tail(window)
        if len(recent) < 5:
            return {"error": "insufficient data for volatility window"}
        daily_vol = recent.std()
        annualised_vol = float(daily_vol * np.sqrt(252))
        return {"ticker": ticker, "window_days": window, "annualised_volatility": round(annualised_vol, 4)}
    except Exception as exc:
        return {"error": str(exc)}


@observed_tool
def llm_sentiment(headlines: list[str]) -> dict:
    """Calls Gemini to produce a structured aggregate sentiment score for a headline list."""
    try:
        if not headlines:
            return {"error": "no headlines provided", "overall_score": 0.0}
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"error": "GEMINI_API_KEY not set"}
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.6-flash")
        prompt = (
            "Classify the overall sentiment of these financial news headlines as a "
            "JSON object with fields overall_score (-1 to 1), positive_count, "
            "negative_count, neutral_count. Headlines:\n"
            + "\n".join(f"- {h}" for h in headlines)
        )
        response = model.generate_content(
            [
                {"text": "You are a financial sentiment classifier. Respond with only JSON."},
                {"text": prompt},
            ],
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        )
        return json.loads(response.text)
    except Exception as exc:
        return {"error": str(exc)}


@observed_tool
def web_search(query: str, max_results: int = 5) -> dict:
    """Uses duckduckgo-search to retrieve analyst commentary / general web results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return {"error": "no results found", "results": []}
        return {
            "query": query,
            "results": [{"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")} for r in results],
        }
    except Exception as exc:
        return {"error": str(exc), "results": []}
