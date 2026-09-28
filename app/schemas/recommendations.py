"""Schemas for recommendations and the human-in-the-loop approval flow.

The approve endpoint is two-phase: it first records a diff, then executes only
when an explicit ``confirm`` flag is sent.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import ApiModel, Evidence, Option, RecommendationStatus


class ChangeDiff(ApiModel):
    """before -> after for a single field the change touches."""

    field: str
    before: object = None
    after: object = None


class Recommendation(ApiModel):
    id: str
    campaign_id: str | None = None
    insight_id: str | None = None
    action_type: str
    title: str
    rationale: str = ""
    payload: dict[str, object] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    status: RecommendationStatus = RecommendationStatus.PENDING
    persona_id: str | None = None
    created_at: datetime
    updated_at: datetime
    executed_at: datetime | None = None
    requires_approval: bool = False
    estimated_impact: str | None = None


class ApproveRecommendationRequest(ApiModel):
    """Phase 1 (no confirm) returns the diff. Phase 2 (confirm=True) executes."""

    confirm: bool = False
    note: str | None = None


class ApproveRecommendationResponse(ApiModel):
    recommendation: Recommendation
    diff: list[ChangeDiff] = Field(default_factory=list)
    executed: bool = False
    # Present when execution needs the second, explicit confirm call.
    requires_confirmation: bool = False
    message: str = ""
    audit_log_id: str | None = None


class RejectRecommendationRequest(ApiModel):
    reason: str | None = None


class RecommendationCreatedByAgent(ApiModel):
    """What create_recommendation returns to the agent."""

    recommendation_id: str
    status: RecommendationStatus = RecommendationStatus.PENDING
    requires_approval: bool = True
    title: str
    action_type: str


__all__ = [
    "ApproveRecommendationRequest",
    "ApproveRecommendationResponse",
    "ChangeDiff",
    "Recommendation",
    "RecommendationCreatedByAgent",
    "RejectRecommendationRequest",
    "Option",
    "Evidence",
]
