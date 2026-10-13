"""Campaign endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import LLMDep, SessionDep, WorkspaceDep, assert_owned
from app.db.models import BrandProfile, Campaign as CampaignRow, Creative, Recommendation
from app.schemas.campaigns import (
    Campaign,
    CampaignCreateRequest,
    CampaignDetailResponse,
    CampaignListResponse,
    CampaignStrategy,
    GenerateStrategyRequest,
    StrategyKPI,
    StrategyPhase,
)

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


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
async def create_campaign(
    payload: CampaignCreateRequest, ws: WorkspaceDep, db: SessionDep
) -> Campaign:
    row = CampaignRow(
        id=f"cmp_{uuid.uuid4().hex[:20]}",
        business_id=ws.business_id,
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
async def list_campaigns(
    ws: WorkspaceDep, db: SessionDep
) -> CampaignListResponse:
    stmt = (
        select(CampaignRow)
        .where(CampaignRow.business_id == ws.business_id)
        .order_by(CampaignRow.created_at.desc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    return CampaignListResponse(
        campaigns=[_to_schema(r) for r in rows], total=len(rows)
    )


@router.get("/{campaign_id}", response_model=CampaignDetailResponse)
async def get_campaign(
    campaign_id: str, ws: WorkspaceDep, db: SessionDep
) -> CampaignDetailResponse:
    """One campaign. A campaign owned by another workspace is a 404, not a 403."""
    stmt = select(CampaignRow).where(
        CampaignRow.id == campaign_id, CampaignRow.business_id == ws.business_id
    )
    row = assert_owned(
        await db.scalar(stmt), ws, entity=f"campaign '{campaign_id}'"
    )

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


@router.post("/generate-strategy", response_model=CampaignStrategy)
async def generate_campaign_strategy_endpoint(
    payload: GenerateStrategyRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    llm: LLMDep,
) -> CampaignStrategy:
    """Generate an actionable AI marketing campaign strategy for the wizard."""
    bp_stmt = select(BrandProfile).where(BrandProfile.business_id == ws.business_id)
    bp = (await db.execute(bp_stmt)).scalars().first()
    
    brand_name = ws.business.name or "Brand"
    industry = ws.business.industry or "Marketing"
    voice = bp.brand_voice if bp and bp.brand_voice else "Engaging, authentic and direct"
    offers = bp.key_offers if bp and bp.key_offers else ["Signature Product Offering"]

    obj = payload.objective.lower()
    summary = (
        f"A multi-stage {payload.objective.capitalize()} strategy for {brand_name} ({industry}) "
        f"focused on driving high-intent conversions with a daily budget of {payload.dailyBudget} EGP. "
        f"Combines high-impact creative storytelling with aggressive retargeting."
    )
    key_messages = [
        f"Discover the distinctive quality of {brand_name}.",
        f"Crafted for customers who value excellence and authenticity in {industry}.",
        f"Exclusive offer: {offers[0] if offers else 'Order today with fast delivery'}.",
    ]
    if payload.brief:
        summary += f" Strategic focus: {payload.brief}"

    phases = [
        StrategyPhase(name="Phase 1: Awareness & Hook Testing", durationDays=7, focus="Test 3 video creative hooks and gather initial pixel data."),
        StrategyPhase(name="Phase 2: Core Conversion Scaling", durationDays=14, focus="Scale winning ad creatives to high-intent lookalike audiences."),
        StrategyPhase(name="Phase 3: Retargeting & LTV Lift", durationDays=9, focus="Retarget engaged profile visitors and abandon-cart users with urgency incentives."),
    ]

    kpis = [
        StrategyKPI(metric="Target ROAS", target=3.2),
        StrategyKPI(metric="Cost Per Acquisition (CPA)", target=round(payload.dailyBudget * 0.25, 2)),
        StrategyKPI(metric="Click-Through Rate (CTR)", target=2.4),
    ]

    if llm:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from pydantic import BaseModel, Field

            class ExtractedStrategy(BaseModel):
                summary: str = Field(description="Strategic executive summary of the campaign")
                key_messages: list[str] = Field(description="3-4 compelling ad messages / value propositions")
                bidding_strategy: str = Field(default="lowest_cost", description="lowest_cost | cost_cap | bid_cap | roas_goal")

            extractor = llm.chat.with_structured_output(ExtractedStrategy)
            res = await extractor.ainvoke([
                SystemMessage(content="You are Omar, Chief Marketing Strategist. Formulate a sharp campaign strategy."),
                HumanMessage(
                    content=(
                        f"Brand: {brand_name} ({industry})\n"
                        f"Brand Voice: {voice}\n"
                        f"Objective: {payload.objective}\n"
                        f"Daily Budget: {payload.dailyBudget}\n"
                        f"Audience: {payload.audience}\n"
                        f"Brief: {payload.brief or 'None'}\n"
                    )
                ),
            ])
            if res.summary:
                summary = res.summary
            if res.key_messages:
                key_messages = res.key_messages
        except Exception:
            pass

    return CampaignStrategy(
        summary=summary,
        keyMessages=key_messages,
        channels=["instagram", "facebook"],
        placements=["feed", "stories", "reels"],
        biddingStrategy="lowest_cost",
        phases=phases,
        kpis=kpis,
    )


__all__ = ["create_campaign", "get_campaign", "list_campaigns", "generate_campaign_strategy_endpoint"]

