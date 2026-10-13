"""Schemas for campaign endpoints."""

from __future__ import annotations

from datetime import datetime

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


class CampaignDetailResponse(ApiModel):
    campaign: Campaign
    creatives: list[dict[str, object]] = Field(default_factory=list)
    recommendations: list[dict[str, object]] = Field(default_factory=list)
    recent_insights: dict[str, object] | None = None


class StrategyPhase(ApiModel):
    name: str
    durationDays: int
    focus: str


class StrategyKPI(ApiModel):
    metric: str
    target: float


class CampaignStrategy(ApiModel):
    summary: str
    keyMessages: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=lambda: ["instagram", "facebook"])
    placements: list[str] = Field(default_factory=lambda: ["feed", "stories", "reels"])
    biddingStrategy: str = "lowest_cost"
    phases: list[StrategyPhase] = Field(default_factory=list)
    kpis: list[StrategyKPI] = Field(default_factory=list)


class GenerateStrategyRequest(ApiModel):
    objective: str
    dailyBudget: float
    audience: dict[str, object] = Field(default_factory=dict)
    brief: str | None = None

