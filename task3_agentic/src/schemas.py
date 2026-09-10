"""Structured schemas for Task 3 -- the multi-agent handoff and final report."""
from typing import Optional
from pydantic import BaseModel, Field


class DataBrief(BaseModel):
    """Agent A (Data Analyst) -> Agent B (Research Writer) structured handoff."""
    ticker: str
    latest_close: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    annualised_volatility: Optional[float] = None
    news_sentiment_score: Optional[float] = None
    notes: str = Field(default="", description="Any data-quality caveats from Agent A")


class ClarificationRequest(BaseModel):
    """Agent B -> Agent A critique-loop request for additional/clarified data."""
    field_requested: str
    reason: str


class RiskItem(BaseModel):
    risk: str
    evidence: str


class ResearchReport(BaseModel):
    ticker: str
    financial_health_summary: str
    top_three_risks: list[RiskItem]
    hedge_strategy: str
