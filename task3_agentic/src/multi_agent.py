"""
Task 3B - Multi-Agent Coordination

Agent A (Data Analyst): quantitative only. Tools: get_price_data,
    calculate_volatility, llm_sentiment. No web_search access.
Agent B (Research Writer): qualitative only. Tools: web_search, get_news.
    No direct price-data access.

Agent A hands off a structured DataBrief (Pydantic) to Agent B -- never a
raw string. Agent B may issue exactly one ClarificationRequest back to
Agent A; Agent A responds with the requested field, Agent B incorporates
it, then produces the final ResearchReport. The full message trace is
printed so it's visible in notebook output.
"""
from __future__ import annotations

import json
import os

try:
    from mistralai import Mistral
except ImportError:  # newer SDK layout
    from mistralai.client import Mistral

from tools import get_price_data, calculate_volatility, llm_sentiment, get_news, web_search
from schemas import DataBrief, ClarificationRequest, ResearchReport, RiskItem

MODEL = "mistral-large-latest"


def _client() -> Mistral:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY not set")
    return Mistral(api_key=api_key)


def _llm_json(client: Mistral, system: str, user: str) -> dict:
    resp = client.chat.complete(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------------------
# Agent A - Data Analyst
# ---------------------------------------------------------------------------

class DataAnalystAgent:
    """Restricted to quantitative tools only."""

    ALLOWED_TOOLS = {"get_price_data", "calculate_volatility", "llm_sentiment"}

    def __init__(self, client: Mistral):
        self.client = client

    def build_brief(self, ticker: str) -> DataBrief:
        print(f"[Agent A] calling get_price_data({ticker})")
        price = get_price_data(ticker)

        print(f"[Agent A] calling calculate_volatility({ticker})")
        vol = calculate_volatility(ticker)

        # Agent A is not allowed to call get_news directly (Agent B owns news),
        # so it derives sentiment context only if headlines are supplied to it.
        # For the initial brief, sentiment starts unset and is filled if Agent B
        # requests it via the critique loop, or left None.
        notes = []
        if "error" in price:
            notes.append(f"price data issue: {price['error']}")
        if "error" in vol:
            notes.append(f"volatility issue: {vol['error']}")

        indicators = price.get("indicators", {})
        brief = DataBrief(
            ticker=ticker,
            latest_close=price.get("latest_close"),
            sma_50=indicators.get("SMA_50"),
            sma_200=indicators.get("SMA_200"),
            rsi_14=indicators.get("RSI_14"),
            macd=indicators.get("MACD"),
            macd_signal=indicators.get("MACD_signal"),
            annualised_volatility=vol.get("annualised_volatility"),
            notes="; ".join(notes),
        )
        print(f"[Agent A] handoff -> {brief.model_dump()}")
        return brief

    def respond_to_clarification(self, req: ClarificationRequest, headlines: list[str]) -> dict:
        """Agent A doesn't own news/sentiment tools directly in this split, but it
        does own llm_sentiment -- so a clarification about sentiment can be
        answered using headlines Agent B supplies (Agent B owns get_news)."""
        print(f"[Agent A] received clarification request: {req.model_dump()}")
        if "sentiment" in req.field_requested.lower():
            result = llm_sentiment(headlines)
            print(f"[Agent A] clarification response -> {result}")
            return {"news_sentiment_score": result.get("overall_score")}
        return {"error": f"Agent A cannot fulfil clarification for '{req.field_requested}'"}


# ---------------------------------------------------------------------------
# Agent B - Research Writer
# ---------------------------------------------------------------------------

class ResearchWriterAgent:
    """Restricted to qualitative tools only: web_search, get_news."""

    ALLOWED_TOOLS = {"web_search", "get_news"}

    def __init__(self, client: Mistral):
        self.client = client

    def maybe_request_clarification(self, brief: DataBrief) -> ClarificationRequest | None:
        """One-shot critique: if the brief lacks a sentiment score, ask Agent A for it."""
        if brief.news_sentiment_score is None:
            return ClarificationRequest(
                field_requested="news_sentiment_score",
                reason="Cannot assess market mood for the report without a sentiment score.",
            )
        return None

    def write_report(self, ticker: str, brief: DataBrief) -> ResearchReport:
        print(f"[Agent B] calling get_news({ticker})")
        news = get_news(ticker)
        headlines = news.get("headlines", [])

        print(f"[Agent B] calling web_search for analyst commentary on {ticker}")
        commentary = web_search(f"{ticker} stock analyst commentary risks 90 days")

        system = (
            "You are a senior equity research writer. Using the structured data "
            "brief and qualitative context provided, produce a JSON object with "
            "keys: financial_health_summary (2-3 sentences), "
            "top_three_risks (list of {risk, evidence}), hedge_strategy (1-2 sentences)."
        )
        user = json.dumps({
            "ticker": ticker,
            "data_brief": brief.model_dump(),
            "headlines": headlines[:10],
            "web_commentary": [r.get("snippet") for r in commentary.get("results", [])][:5],
        })
        raw = _llm_json(self.client, system, user)
        report = ResearchReport(
            ticker=ticker,
            financial_health_summary=raw.get("financial_health_summary", ""),
            top_three_risks=[RiskItem(**r) for r in raw.get("top_three_risks", [])],
            hedge_strategy=raw.get("hedge_strategy", ""),
        )
        print(f"[Agent B] final report drafted -> {report.model_dump()}")
        return report, headlines


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_multi_agent_pipeline(ticker: str) -> ResearchReport:
    client = _client()
    agent_a = DataAnalystAgent(client)
    agent_b = ResearchWriterAgent(client)

    print(f"\n===== Agent A: building data brief for {ticker} =====")
    brief = agent_a.build_brief(ticker)

    print(f"\n===== Agent B: reviewing brief, deciding whether to critique =====")
    clarification = agent_b.maybe_request_clarification(brief)

    headlines_for_sentiment: list[str] = []
    if clarification is not None:
        print(f"\n===== Critique loop: Agent B -> Agent A =====")
        # Agent B fetches headlines itself (it owns get_news) so it can hand
        # them to Agent A for the sentiment clarification.
        news = get_news(ticker)
        headlines_for_sentiment = news.get("headlines", [])
        clarification_response = agent_a.respond_to_clarification(clarification, headlines_for_sentiment)
        if "news_sentiment_score" in clarification_response:
            brief.news_sentiment_score = clarification_response["news_sentiment_score"]
            print(f"[Agent B] incorporated clarification -> updated brief: {brief.model_dump()}")

    print(f"\n===== Agent B: writing final report =====")
    report, _ = agent_b.write_report(ticker, brief)
    return report


if __name__ == "__main__":
    final_report = run_multi_agent_pipeline("AAPL")
    print("\n\n=== FINAL REPORT ===")
    print(final_report.model_dump_json(indent=2))
