"""Schemas for Task 2 dataset generation and evaluation."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    """One chat-format training example: system/user/assistant turns."""
    system: str
    user: str
    assistant: str
    topic_tag: str = Field(description="short keyword/topic label used for diversity analysis")


class JudgeScore(BaseModel):
    """LLM-as-judge structured output for Task 2C evaluation."""
    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    domain_accuracy: int = Field(ge=1, le=5)
    overall: float
    rationale: str


class ManualReviewLabel(BaseModel):
    example_id: int
    label: Literal["correct", "partially_correct", "hallucinated"]
    note: Optional[str] = None
