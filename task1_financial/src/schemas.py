"""Pydantic schemas enforcing structured output from the LLM (Task 1B)."""
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class HeadlineSentiment(BaseModel):
    headline: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str

    @field_validator("brief_reason")
    @classmethod
    def reason_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("brief_reason must not be empty")
        return v


class AggregateSentiment(BaseModel):
    overall_score: float = Field(
        ge=-1.0, le=1.0,
        description="positive - negative fraction of headlines, weighted by confidence",
    )
    positive_count: int
    negative_count: int
    neutral_count: int
    total_headlines: int


class TradeSignal(BaseModel):
    signal: Literal["Buy", "Hold", "Sell"]
    justification: str = Field(min_length=1)

    @field_validator("justification")
    @classmethod
    def justification_length(cls, v: str) -> str:
        sentence_count = v.count(".") + v.count("!") + v.count("?")
        if sentence_count < 3:
            raise ValueError(
                f"justification must be 3-5 sentences, found ~{sentence_count}"
            )
        return v
