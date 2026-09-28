"""Insights endpoints: GET /campaigns/{id}/insights."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import SessionDep
from app.core.tools.meta_read import _handle_get_campaign_insights  # noqa: F401  (see note)
from app.db.models import Campaign, Insight
from app.schemas.insights import CampaignInsightsResponse, InsightRecord, InsightsPayload

router = APIRouter(tags=["insights"])


@router.get("/campaigns/{campaign_id}/insights", response_model=CampaignInsightsResponse)
async def get_campaign_insights(
    campaign_id: str,
    db: SessionDep,
    date_start: date | None = Query(default=None),
    date_end: date | None = Query(default=None),
) -> CampaignInsightsResponse:
    """Stored insights for a campaign, plus a fresh Meta pull when connected."""
    campaign = await db.get(Campaign, campaign_id)
    if campaign is None:
        from app.core.errors import NotFoundError

        raise NotFoundError(f"campaign '{campaign_id}' not found")

    rows = (
        await db.execute(
            select(Insight)
            .where(Insight.campaign_id == campaign_id)
            .order_by(Insight.created_at.desc())
            .limit(50)
        )
    ).scalars().all()

    insights = [
        InsightRecord(
            id=r.id,
            campaign_id=r.campaign_id,
            kind=r.kind,
            title=r.title,
            detail=r.detail,
            severity=r.severity,
            metrics=r.metrics or {},
            created_at=r.created_at,
        )
        for r in rows
    ]

    # A fresh pull is best-effort; it must never fail the endpoint.
    latest: InsightsPayload | None = None
    try:
        from app.core.llm.mock_client import MockClient
        from app.core.tools.base import ToolContext
        from app.schemas.tools import GetCampaignInsightsInput

        ctx = ToolContext(
            workspace_id="default",
            business_id=campaign.business_id,
            db=db,
            llm=MockClient(),
            request_id="",
            campaign_id=campaign_id,
        )
        latest = await _handle_get_campaign_insights(
            GetCampaignInsightsInput(
                campaign_id=campaign_id, date_start=date_start, date_end=date_end
            ),
            ctx,
        )
    except Exception:  # pragma: no cover - insights must never 500
        latest = None

    return CampaignInsightsResponse(
        campaign_id=campaign_id, insights=insights, latest=latest, breakdowns=[]
    )


__all__ = ["get_campaign_insights"]
