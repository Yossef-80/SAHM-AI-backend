"""Business endpoints: POST /business, GET /business/profile."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import SessionDep
from app.core.memory.store import BrandMemoryStore
from app.db.models import BrandProfile as BrandProfileRow
from app.db.models import Business
from app.schemas.business import (
    BrandProfile,
    BusinessCreateRequest,
    BusinessProfileResponse,
)

router = APIRouter(prefix="/business", tags=["business"])


async def _get_or_create_business(db) -> Business:
    stmt = select(Business).order_by(Business.created_at).limit(1)
    business = (await db.execute(stmt)).scalars().first()
    if business is None:
        business = Business(id=f"biz_{__import__('uuid').uuid4().hex[:20]}", name="My Business")
        db.add(business)
        await db.flush()
    return business


@router.post("", response_model=BusinessProfileResponse)
async def create_business(payload: BusinessCreateRequest, db: SessionDep) -> BusinessProfileResponse:
    """Create (or replace) the workspace's business and brand profile."""
    business = await _get_or_create_business(db)
    business.name = payload.name
    business.industry = payload.industry
    business.website_url = payload.website_url
    business.country = (payload.country or "").upper() or None
    business.language = payload.language or business.language
    business.currency = payload.currency
    business.monthly_budget = payload.monthly_budget

    stmt = select(BrandProfileRow).where(BrandProfileRow.business_id == business.id)
    profile = (await db.execute(stmt)).scalars().first()
    if profile is None:
        profile = BrandProfileRow(id=f"bp_{business.id}", business_id=business.id)
        db.add(profile)
    profile.goals = payload.goals

    await db.commit()
    return await _profile_response(db, business.id)


@router.get("/profile", response_model=BusinessProfileResponse)
async def get_business_profile(db: SessionDep) -> BusinessProfileResponse:
    business = await _get_or_create_business(db)
    return await _profile_response(db, business.id)


async def _profile_response(db, business_id: str) -> BusinessProfileResponse:
    store = BrandMemoryStore(db)
    business = await store.get_business(business_id)
    profile = await store.get_profile(business_id)
    count = await store.count_items(business_id)
    learning_count = sum(
        1 for _ in await store.recent_learnings(business_id=business_id, limit=1000)
    )
    schema = BrandProfile(
        business_id=business.id,
        name=business.name,
        industry=business.industry,
        website_url=business.website_url,
        country=business.country,
        language=business.language,
        currency=business.currency,
        monthly_budget=business.monthly_budget,
        brand_voice=profile.brand_voice if profile else None,
        visual_style=profile.visual_style if profile else None,
        target_audience=profile.target_audience if profile else None,
        key_offers=list(profile.key_offers) if profile else [],
        goals=list(profile.goals) if profile else [],
        competitors=list(profile.competitors) if profile else [],
        updated_at=business.updated_at,
    )
    return BusinessProfileResponse(
        business=schema, memory_item_count=count, learning_count=learning_count
    )
