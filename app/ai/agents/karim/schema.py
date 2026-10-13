"""Evidence-backed actual campaign-performance report for Karim."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PerformanceEvidence(BaseModel):
    metric: str = ""
    value: str = ""
    source: str = ""
    date_range: str = ""


class PerformanceReport(BaseModel):
    """Report only metrics actually returned by Meta or supplied by the user."""

    summary: str = ""
    details: list[str] = Field(default_factory=list)
    campaign_id: str | None = None
    date_range: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    breakdowns: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[PerformanceEvidence] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
