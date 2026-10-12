"""Schemas for content calendar endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import ApiModel

ContentChannel = Literal["instagram", "facebook", "tiktok", "email", "blog"]
ContentFormat = Literal["post", "reel", "story", "ad", "email", "article"]
ContentStatus = Literal["idea", "draft", "scheduled", "published"]


class ContentItemOut(ApiModel):
    id: str
    date: str
    time: str | None = None
    channel: str
    format: str
    title: str
    hook: str | None = None
    caption: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    status: str = "scheduled"
    campaign_id: str | None = None
    creative_id: str | None = None
    pillar: str | None = None
    created_at: datetime
    updated_at: datetime


class GenerateContentCalendarRequest(ApiModel):
    month: str = Field(description="YYYY-MM format, e.g. 2026-10")
    campaign_id: str | None = None
    posts_per_week: int = Field(default=4, ge=1, le=14)
    channels: list[str] = Field(default_factory=lambda: ["instagram", "facebook"])
    brief: str | None = None


class UpdateContentItemRequest(ApiModel):
    date: str | None = None
    time: str | None = None
    channel: str | None = None
    format: str | None = None
    title: str | None = None
    hook: str | None = None
    caption: str | None = None
    hashtags: list[str] | None = None
    status: str | None = None
    pillar: str | None = None
