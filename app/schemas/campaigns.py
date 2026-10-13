"""Schemas for campaign endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.common import ApiModel


class CampaignCreateRequest(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    objective: str | None = None
    channel: str | None = None
    budget: float | None = None
    currency: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    target_audience: str | None = None
    meta_campaign_id: str | None = None
    notes: str | None = None


class CampaignUpdateRequest(ApiModel):
    name: str | None = None
    objective: str | None = None
    channel: str | None = None
    budget: float | None = None
    status: str | None = None
    notes: str | None = None
    end_date: datetime | None = None


class Campaign(ApiModel):
    id: str
    business_id: str
    name: str
    status: str = "draft"
    objective: str | None = None
    channel: str | None = None
    budget: float | None = None
    currency: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    target_audience: str | None = None
    meta_campaign_id: str | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime
    creative_count: int = 0
    recommendation_count: int = 0


class CampaignListResponse(ApiModel):
    campaigns: list[Campaign]
    total: int = 0


class CampaignStrategyGenerationRequest(ApiModel):
    """Inputs for Omar's non-publishing campaign strategy draft."""

    campaign_name: str | None = None
    objective: Literal["awareness", "traffic", "engagement", "leads", "sales", "app_installs"]
    audience: dict[str, Any] = Field(default_factory=dict)
    daily_budget: float = Field(gt=0)
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    brief: str | None = None
    language: Literal["ar", "en"] = "en"


class CampaignStrategyChannel(ApiModel):
    channel: str
    role: str = ""
    budget_share_pct: float | None = None
    placements: list[str] = Field(default_factory=list)
    rationale: str = ""


class GeneratedCampaignStrategy(ApiModel):
    """Structured draft shown in the campaign wizard; not an execution request."""

    summary: str = ""
    objective: str | None = None
    audience: str | None = None
    positioning: str | None = None
    key_messages: list[str] = Field(default_factory=list)
    channels: list[CampaignStrategyChannel] = Field(default_factory=list)
    placements: list[str] = Field(default_factory=list)
    creative_direction: str | None = None
    cta_direction: str | None = None
    creatives_needed: list[str] = Field(default_factory=list)
    success_metrics: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)


class MetaPlanAssistAudience(ApiModel):
    """Current, user-chosen audience values sent to the read-only assistant."""

    name: str = Field(default="", max_length=200)
    age_min: int = Field(ge=13, le=65)
    age_max: int = Field(ge=13, le=65)
    gender: Literal["all", "male", "female"] = "all"
    countries: list[str] = Field(default_factory=list, max_length=12)
    interests: list[str] = Field(default_factory=list, max_length=20)


class MetaAdPlanAssistRequest(ApiModel):
    """Wizard state for generating an editable, non-publishing Meta plan draft."""

    campaign_name: str = Field(min_length=2, max_length=200)
    objective: Literal["awareness", "traffic", "engagement", "leads", "sales", "app_installs"]
    daily_budget: float = Field(gt=0)
    currency: str = Field(min_length=3, max_length=3)
    brief: str | None = Field(default=None, max_length=4000)
    language: Literal["ar", "en"] = "en"
    strategy: GeneratedCampaignStrategy
    audience: MetaPlanAssistAudience
    destination_url: str | None = Field(default=None, max_length=2000)
    primary_text: str | None = Field(default=None, max_length=3000)
    headline: str | None = Field(default=None, max_length=255)
    call_to_action_type: str | None = Field(default=None, max_length=80)
    billing_event: str | None = Field(default=None, max_length=80)
    optimization_goal: str | None = Field(default=None, max_length=80)
    destination_type: str | None = Field(default=None, max_length=80)
    special_ad_categories: list[str] = Field(default_factory=list, max_length=5)


class MetaAdPlanAssistResponse(ApiModel):
    """MCP-checked suggestions; the UI applies them only to its editable draft."""

    primary_text: str | None = None
    headline: str | None = None
    call_to_action_type: str | None = None
    billing_event: str | None = None
    optimization_goal: str | None = None
    destination_type: str | None = None
    special_ad_categories: list[str] | None = None
    bid_strategy: Literal["LOWEST_COST_WITHOUT_CAP"] = "LOWEST_COST_WITHOUT_CAP"
    allowed_values: dict[str, list[str]] = Field(default_factory=dict)
    field_validation: dict[str, Literal["valid", "invalid", "unchecked"]] = Field(
        default_factory=dict
    )
    assumptions: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    mcp_context_available: bool = True
    review_required: bool = True
    writes_to_meta: bool = False


class CampaignDetailResponse(ApiModel):
    campaign: Campaign
    creatives: list[dict[str, object]] = Field(default_factory=list)
    recommendations: list[dict[str, object]] = Field(default_factory=list)
    recent_insights: dict[str, object] | None = None
