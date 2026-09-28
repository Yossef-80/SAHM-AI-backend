"""Schemas for insights endpoints and the meta_read tools."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field

from app.core.tools.base import ToolOutput
from app.schemas.common import ApiModel, Evidence


class DateRange(ApiModel):
    start: date
    end: date

    def model_post_init(self, __context: object) -> None:
        if self.end < self.start:
            raise ValueError("end must be >= start")


class InsightsQuery(ApiModel):
    campaign_id: str
    date_range: DateRange | None = None
    # When the campaign is not linked to a Meta campaign, this is used instead.
    meta_campaign_id: str | None = None


class MetricSeries(ApiModel):
    label: str
    value: float
    unit: str | None = None


class InsightsPayload(ToolOutput):
    """Campaign-level metrics. ``status``/``source`` come from ToolOutput.

    Fields default rather than being required so ``ToolOutput.unavailable()`` can
    build a valid instance when Meta is not connected.
    """

    campaign_id: str = ""
    meta_campaign_id: str | None = None
    date_start: date | None = None
    date_end: date | None = None
    currency: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    series: list[MetricSeries] = Field(default_factory=list)
    top_ads: list[dict[str, object]] = Field(default_factory=list)
    retrieved_at: datetime | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class BreakdownPayload(ToolOutput):
    """A single breakdown dimension. ``status``/``source`` from ToolOutput."""

    campaign_id: str = ""
    by: str = ""
    date_start: date | None = None
    date_end: date | None = None
    rows: list[dict[str, object]] = Field(default_factory=list)
    retrieved_at: datetime | None = None


class InsightRecord(ApiModel):
    id: str
    campaign_id: str
    kind: str
    title: str
    detail: str = ""
    severity: str = "info"
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime


class CampaignInsightsResponse(ApiModel):
    campaign_id: str
    insights: list[InsightRecord] = Field(default_factory=list)
    latest: InsightsPayload | None = None
    breakdowns: list[BreakdownPayload] = Field(default_factory=list)
