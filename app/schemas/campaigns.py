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
