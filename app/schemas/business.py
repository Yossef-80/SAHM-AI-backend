"""Schemas for POST /business and GET /business/profile."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import ApiModel


class BusinessCreateRequest(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    industry: str | None = None
    website_url: str | None = None
    country: str | None = Field(default=None, max_length=2)
    language: str | None = Field(default=None, description="en | ar")
    currency: str | None = Field(default=None, max_length=8)
    monthly_budget: float | None = None
    goals: list[str] = Field(default_factory=list)


class BrandProfile(ApiModel):
    """Stable, always-in-context brand fields (the compact part of memory)."""

    business_id: str
    name: str
    industry: str | None = None
    website_url: str | None = None
    country: str | None = None
    language: str = "en"
    currency: str | None = None
    monthly_budget: float | None = None
    goals: list[str] = Field(default_factory=list)
    brand_voice: str | None = None
    visual_style: str | None = None
    target_audience: str | None = None
    key_offers: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None


class BusinessProfileResponse(ApiModel):
    business: BrandProfile
    memory_item_count: int = 0
    learning_count: int = 0
