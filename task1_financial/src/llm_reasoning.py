"""
Task 1B - LLM Sentiment and Signal Reasoning

Uses the Gemini API for inference. Prompts are kept as module-level
template constants, separate from business logic. All responses are parsed
as JSON and validated against the Pydantic schemas in schemas.py;
validation failures are logged and handled (retried once, then skipped)
rather than crashing the pipeline.

Set GEMINI_API_KEY as an environment variable before running -- never
hardcode it in this file.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import google.generativeai as genai
from pydantic import ValidationError

from schemas import HeadlineSentiment, AggregateSentiment, TradeSignal

logger = logging.getLogger("llm_reasoning")

GEMINI_MODEL = "gemini-2.0-flash"

# ---------------------------------------------------------------------------
# Prompt templates (kept separate from business logic)
# ---------------------------------------------------------------------------

SENTIMENT_SYSTEM_PROMPT = """You are a financial news sentiment analyst.
Given a single news headline about a publicly traded company, classify its
sentiment and explain your reasoning briefly. Respond with ONLY a JSON
object matching this exact schema, no markdown fences, no extra text:
{
  "headline": "<the original headline verbatim>",
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": <float between 0 and 1>,
  "brief_reason": "<one short sentence>"
}"""

SENTIMENT_USER_TEMPLATE = "Headline: {headline}"

SIGNAL_SYSTEM_PROMPT = """You are a senior equity research analyst producing a
first-pass trading signal. You will be given a set of technical indicators
for a stock. Reason over the COMBINATION of indicators together (e.g. does
a rising RSI near overbought territory conflict with a bullish MACD
crossover? does price sit near the upper Bollinger Band while momentum is
fading?) -- do not simply restate each indicator's value in isolation.

Respond with ONLY a JSON object matching this exact schema, no markdown
fences, no extra text:
{
  "signal": "Buy" | "Hold" | "Sell",
  "justification": "<3 to 5 full sentences reasoning over the indicator combination>"
}"""

SIGNAL_USER_TEMPLATE = """Ticker: {ticker}
Current price: {current_price}
50-day SMA: {sma_50}
200-day SMA: {sma_200}
RSI(14): {rsi_14}
MACD: {macd} | Signal line: {macd_signal} | Histogram: {macd_hist}
Bollinger %B: {bb_percent_b} (0 = at lower band, 1 = at upper band)
52-week range: {low_52w} - {high_52w}
YTD return: {ytd_return}%
Aggregate news sentiment score: {sentiment_score} (-1 very negative to +1 very positive)"""


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Export it as an environment variable "
            "(never hardcode it in source)."
        )
    genai.configure(api_key=api_key)
    return genai


def _call_llm_json(client, system_prompt: str, user_prompt: str) -> Optional[dict]:
    """Call the LLM and parse the response as JSON. Returns None on failure."""
    try:
        model = client.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            [
                {"text": system_prompt},
                {"text": user_prompt},
            ],
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )
        text = getattr(response, "text", "")
        if not text:
            return None
        return json.loads(text)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("LLM call/parse failed: %s", exc)
        return None


def classify_headline(client, headline: str, retries: int = 1) -> Optional[HeadlineSentiment]:
    """Classify one headline, validating against HeadlineSentiment. Retries once on failure."""
    user_prompt = SENTIMENT_USER_TEMPLATE.format(headline=headline)
    for attempt in range(retries + 1):
        raw = _call_llm_json(client, SENTIMENT_SYSTEM_PROMPT, user_prompt)
        if raw is None:
            continue
        try:
            return HeadlineSentiment(**raw)
        except ValidationError as exc:
            logger.warning(
                "Validation failed for headline classification (attempt %d): %s",
                attempt + 1, exc,
            )
    logger.error("Giving up on headline after %d attempts: %s", retries + 1, headline)
    return None


def aggregate_sentiment(results: list[HeadlineSentiment]) -> AggregateSentiment:
    pos = sum(1 for r in results if r.sentiment == "positive")
    neg = sum(1 for r in results if r.sentiment == "negative")
    neu = sum(1 for r in results if r.sentiment == "neutral")
    total = len(results)

    if total == 0:
        return AggregateSentiment(
            overall_score=0.0, positive_count=0, negative_count=0,
            neutral_count=0, total_headlines=0,
        )

    # confidence-weighted score in [-1, 1]
    weighted_sum = sum(
        (1 if r.sentiment == "positive" else -1 if r.sentiment == "negative" else 0) * r.confidence
        for r in results
    )
    score = weighted_sum / total

    return AggregateSentiment(
        overall_score=round(score, 4),
        positive_count=pos, negative_count=neg, neutral_count=neu,
        total_headlines=total,
    )


def classify_all_headlines(client, headlines: list[dict]) -> tuple[list[HeadlineSentiment], AggregateSentiment]:
    results = []
    for item in headlines:
        title = item.get("title", "")
        if not title:
            continue
        parsed = classify_headline(client, title)
        if parsed is not None:
            results.append(parsed)
    return results, aggregate_sentiment(results)


def generate_trade_signal(
    client, ticker: str, latest_indicators: dict, sentiment: AggregateSentiment,
    retries: int = 1,
) -> Optional[TradeSignal]:
    user_prompt = SIGNAL_USER_TEMPLATE.format(
        ticker=ticker,
        current_price=latest_indicators.get("Close"),
        sma_50=latest_indicators.get("SMA_50"),
        sma_200=latest_indicators.get("SMA_200"),
        rsi_14=latest_indicators.get("RSI_14"),
        macd=latest_indicators.get("MACD"),
        macd_signal=latest_indicators.get("MACD_signal"),
        macd_hist=latest_indicators.get("MACD_hist"),
        bb_percent_b=latest_indicators.get("BB_percent_b"),
        low_52w=latest_indicators.get("fifty_two_week_low"),
        high_52w=latest_indicators.get("fifty_two_week_high"),
        ytd_return=latest_indicators.get("ytd_return_pct"),
        sentiment_score=sentiment.overall_score,
    )
    for attempt in range(retries + 1):
        raw = _call_llm_json(client, SIGNAL_SYSTEM_PROMPT, user_prompt)
        if raw is None:
            continue
        try:
            return TradeSignal(**raw)
        except ValidationError as exc:
            logger.warning("Signal validation failed (attempt %d): %s", attempt + 1, exc)
    logger.error("Giving up on trade signal generation after %d attempts", retries + 1)
    return None
