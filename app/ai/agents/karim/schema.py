"""Evidence-backed actual campaign-performance report for Karim."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PerformanceEvidence(BaseModel):
    """One measured value tied to the system that returned it and its period."""

    metric: str = ""
    value: str = ""
    source: str = ""
    date_range: str = ""


class BrandContextFinding(BaseModel):
    """Relevant saved profile/research context, kept separate from live metrics."""

    finding: str = ""
    source: str = ""
    source_url: str | None = None
    recorded_at: str | None = None


class PerformanceRecommendation(BaseModel):
    """A proposed experiment, not an executed campaign change."""

    action: str = ""
    rationale: str = ""
    priority: Literal["high", "medium", "low"] | None = None
    evidence: list[PerformanceEvidence] = Field(default_factory=list)
    supporting_context: list[str] = Field(default_factory=list)
    next_test: str | None = None


class PerformanceReport(BaseModel):
    """Report only metrics actually returned by Meta/GA4 or supplied by the user."""

    summary: str = ""
    details: list[str] = Field(default_factory=list)
    campaign_name: str | None = None
    campaign_id: str | None = None
    date_range: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    breakdowns: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[PerformanceEvidence] = Field(default_factory=list)
    brand_context: list[BrandContextFinding] = Field(default_factory=list)
    recommendations: list[PerformanceRecommendation] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
