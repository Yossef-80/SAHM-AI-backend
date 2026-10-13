"""Human-approval-ready optimization proposal schema for Salma."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OptimizationPlan(BaseModel):
    """A proposed change, never an execution instruction."""

    summary: str = ""
    details: list[str] = Field(default_factory=list)
    recommendation_id: str | None = None
    action_type: str | None = None
    title: str | None = None
    proposed_change: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    estimated_impact: str | None = None
    requires_approval: bool = True
    missing_inputs: list[str] = Field(default_factory=list)
