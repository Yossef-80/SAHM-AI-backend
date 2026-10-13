"""Schemas for the workspace-scoped content calendar API."""

from __future__ import annotations

import re
from datetime import date as CalendarDate
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.common import ApiModel

ContentChannel = Literal["instagram", "facebook", "tiktok", "email", "blog"]
ContentFormat = Literal["post", "reel", "story", "ad", "email", "article"]
ContentStatus = Literal["idea", "draft", "scheduled", "published"]


class ContentCalendarGenerateRequest(ApiModel):
    """Generation controls already collected by the calendar dialog."""

    month: str = Field(pattern=r"^\d{4}-\d{2}$")
    campaign_id: str | None = None
    posts_per_week: int = Field(default=4, ge=1, le=14)
    channels: list[ContentChannel] = Field(
        default_factory=lambda: ["instagram", "facebook"], min_length=1
    )
    brief: str | None = Field(default=None, max_length=12_000)

    @field_validator("month")
    @classmethod
    def validate_month(cls, value: str) -> str:
        year, month = (int(part) for part in value.split("-"))
        if not 1 <= month <= 12 or year < 1:
            raise ValueError("month must be a real calendar month in YYYY-MM format")
        return value

    @field_validator("channels")
    @classmethod
    def validate_unique_channels(cls, value: list[ContentChannel]) -> list[ContentChannel]:
        if len(set(value)) != len(value):
            raise ValueError("channels must not contain duplicates")
        return value


class ContentItemPatchRequest(ApiModel):
    """Editable fields from the content-item side sheet; null clears time/caption."""

    date: CalendarDate | None = None
    time: str | None = None
    title: str | None = Field(default=None, min_length=1, max_length=500)
    caption: str | None = None
    status: ContentStatus | None = None
    channel: ContentChannel | None = None
    format: ContentFormat | None = None

    @field_validator("time")
    @classmethod
    def validate_time(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
            raise ValueError("time must use HH:MM format")
        return value

    @model_validator(mode="after")
    def validate_patch(self) -> ContentItemPatchRequest:
        if not self.model_fields_set:
            raise ValueError("send at least one editable content-item field")
        for field_name in ("date", "title", "status", "channel", "format"):
            if field_name in self.model_fields_set and getattr(self, field_name) is None:
                raise ValueError(f"{field_name} cannot be cleared")
        return self


class ContentItemResponse(ApiModel):
    id: str
    date: CalendarDate
    time: str | None = None
    channel: ContentChannel
    format: ContentFormat
    title: str
    hook: str | None = None
    caption: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    status: ContentStatus
    campaign_id: str | None
    creative_id: str | None = None
    pillar: str | None = None
    created_at: datetime
    updated_at: datetime


class ContentItemDeleteResponse(ApiModel):
    id: str
