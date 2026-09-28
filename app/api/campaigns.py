"""Campaign endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import SessionDep
from app.db.models import Business, Campaign as CampaignRow, Creative, Recommendation
from app.schemas.campaigns import (
    Campaign,
    CampaignCreateRequest,
    CampaignDetailResponse,
    CampaignListResponse,
)

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


async def _business(db) -> Business:
    stmt = select(Business).order_by(Business.created_at).limit(1)
    business = (await db.execute(stmt)).scalars().first()
    if business is None:
        business = Business(id=f"biz_{uuid.uuid4().hex[:20]}", name="My Business")
        db.add(business)
        await db.flush()
    return business


def _to_schema(row: CampaignRow, *, creatives: int = 0, recommendations: int = 0) -> Campaign:
    return Campaign(
        id=row.id,
        business_id=row.business_id,
        name=row.name,
        status=row.status,
        objective=row.objective,
        channel=row.channel,
        budget=row.budget,
        currency=row.currency,
        start_date=row.start_date,
        end_date=row.end_date,
        target_audience=row.target_audience,
        meta_campaign_id=row.meta_campaign_id,
        notes=row.notes,
        created_at=row.created_at,
        updated_at=row.updated_at,
        creative_count=creatives,
        recommendation_count=recommendations,
    )


@router.post("", response_model=Campaign)
async def create_campaign(payload: CampaignCreateRequest, db: SessionDep) -> Campaign:
    business = await _business(db)
    row = CampaignRow(
        id=f"cmp_{uuid.uuid4().hex[:20]}",
        business_id=business.id,
        name=payload.name,
        objective=payload.objective,
        channel=payload.channel,
        budget=payload.budget,
        currency=payload.currency,
        start_date=payload.start_date,
        end_date=payload.end_date,
        target_audience=payload.target_audience,
        meta_campaign_id=payload.meta_campaign_id,
        notes=payload.notes,
        status="draft",
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _to_schema(row)


@router.get("", response_model=CampaignListResponse)
async def list_campaigns(db: SessionDep) -> CampaignListResponse:
    business = await _business(db)
    stmt = (
        select(CampaignRow)
        .where(CampaignRow.business_id == business.id)
        .order_by(CampaignRow.created_at.desc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    return CampaignListResponse(
        campaigns=[_to_schema(r) for r in rows], total=len(rows)
    )


@router.get("/{campaign_id}", response_model=CampaignDetailResponse)
async def get_campaign(campaign_id: str, db: SessionDep) -> CampaignDetailResponse:
    row = await db.get(CampaignRow, campaign_id)
    if row is None:
        from app.core.errors import NotFoundError

        raise NotFoundError(f"campaign '{campaign_id}' not found")

    creatives = (
        await db.execute(select(Creative).where(Creative.campaign_id == campaign_id))
    ).scalars().all()
    recommendations = (
        await db.execute(
            select(Recommendation).where(Recommendation.campaign_id == campaign_id)
        )
    ).scalars().all()

    return CampaignDetailResponse(
        campaign=_to_schema(
            row, creatives=len(creatives), recommendations=len(recommendations)
        ),
        creatives=[
            {
                "id": c.id,
                "url": c.url,
                "prompt": c.prompt,
                "aspect_ratio": c.aspect_ratio,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in creatives
        ],
        recommendations=[
            {
                "id": r.id,
                "action_type": r.action_type,
                "title": r.title,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in recommendations
        ],
    )


__all__ = ["create_campaign", "get_campaign", "list_campaigns"]
