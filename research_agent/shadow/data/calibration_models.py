"""Strict structured contracts for Calibration Agent inputs and outputs."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CATEGORIES = ("volatility", "event_risk", "valuation", "liquidity", "concentration")
Category = Literal["volatility", "event_risk", "valuation", "liquidity", "concentration"]
AdviceValue = Literal["tighten", "loosen", "keep"]
Confidence = Literal["low", "medium", "high"]
HypothesisStatus = Literal["pending", "held", "failed"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CategoryAdvice(StrictModel):
    advice: AdviceValue
    confidence: Confidence
    reasoning: str = Field(min_length=1, max_length=2400)
    conditions: list[str] = Field(default_factory=list, max_length=12)
    supporting_ids: list[str] = Field(default_factory=list, max_length=50)
    contradicting_ids: list[str] = Field(default_factory=list, max_length=50)
    watch_next: list[str] = Field(default_factory=list, max_length=12)
    conflict_rationale: str | None = Field(default=None, max_length=1600)


class NumericClaim(StrictModel):
    """A numeric fact tied to an exact value returned by one tool call."""

    statement: str = Field(min_length=1, max_length=500)
    value: float
    unit: str = Field(default="", max_length=40)
    tool_call_seq: int = Field(ge=1)
    json_pointer: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def finite_value(self):
        if not math.isfinite(self.value):
            raise ValueError("numeric claim value must be finite")
        return self


class Hypothesis(StrictModel):
    statement: str = Field(min_length=1, max_length=1200)
    category: Category
    subgroup: dict = Field(default_factory=dict)


class HypothesisCheck(StrictModel):
    hypothesis_id: str = Field(min_length=1, max_length=160)
    status: HypothesisStatus
    evidence_tool_call_seq: int = Field(ge=1)
    evidence_json_pointer: str = Field(min_length=1, max_length=500)
    evidence_record_ids: list[str] = Field(default_factory=list, max_length=50)


class CalibrationOutput(StrictModel):
    summary: str = Field(min_length=1, max_length=2400)
    risk_level: int = Field(ge=1, le=5)
    risk_level_rationale: str = Field(min_length=1, max_length=2400)
    advice: dict[Category, CategoryAdvice]
    numeric_claims: list[NumericClaim] = Field(default_factory=list, max_length=80)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=30)
    hypothesis_checks: list[HypothesisCheck] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def require_all_categories(self):
        if set(self.advice) != set(CATEGORIES):
            raise ValueError(f"advice must contain exactly {list(CATEGORIES)}")
        return self

    def as_json_dict(self) -> dict:
        return self.model_dump(mode="json")
