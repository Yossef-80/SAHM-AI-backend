"""Campaign endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.marketing.meta_ads import (
    HttpMetaGraphClient,
    MetaAdPlanSettings,
    MetaAdPlanValidationError,
    MetaGraphError,
    create_meta_ad_plan_draft,
    resolve_meta_connection,
)
from app.ai.marketing.meta_ads_mcp import MetaAdsMcpClient, MetaAdsMcpError
from app.ai.marketing.meta_plan_assist import MetaPlanAssistError, assist_meta_ad_plan
from app.api.deps import (
    AuthDep,
    LLMDep,
    SessionDep,
    SettingsDep,
    WorkspaceDep,
    assert_owned,
    require_role,
)
from app.config import Settings
from app.core.campaigns.service import generate_campaign_strategy_draft
from app.core.errors import ValidationFailed
from app.db.models import BrandProfile, Campaign as CampaignRow, Creative, Insight, Recommendation
from app.schemas.campaigns import (
    Campaign,
    CampaignCreateRequest,
    CampaignDetailResponse,
    CampaignListResponse,
    CampaignStrategy,
    CampaignStrategyGenerationRequest,
    CampaignUpdateRequest,
    GenerateStrategyRequest,
    GeneratedCampaignStrategy,
    MetaAdPlanAssistRequest,
    MetaAdPlanAssistResponse,
    StrategyKPI,
)
from app.schemas.common import ApiModel

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


class MetaAdPlanCreateRequest(ApiModel):
    campaign: CampaignCreateRequest
    plan: MetaAdPlanSettings


class MetaAdPlanCreatedResponse(ApiModel):
    campaign_id: str
    recommendation_id: str
    title: str
    status: str = "pending"
    message: str = "Plan saved locally. No Meta request has been made."


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


#: Campaign statuses that mean something to Meta. Local-only statuses (draft,
#: scheduled, completed, failed) are never mirrored.
_META_STATUS = {"active": "ACTIVE", "paused": "PAUSED"}


async def _mirror_status_to_meta(
    db: AsyncSession, row: CampaignRow, meta_status: str, settings: Settings
) -> None:
    """Push a status change to Meta, or raise so nothing local is written.

    Failing loudly is the point: a locally-paused campaign that keeps spending
    on Meta would silently burn budget while the UI said otherwise.
    """
    connection = await resolve_meta_connection(db, row.business_id, settings)
    if not connection.access_token:
        raise HTTPException(
            status_code=409,
            detail=(
                "This campaign exists on Meta but Meta is not connected for this "
                "workspace, so its delivery state cannot be changed safely. "
                "Connect Meta first."
            ),
        )
    graph = HttpMetaGraphClient(
        settings=settings,
        access_token=connection.access_token,
        base_url=connection.graph_base_url,
    )
    try:
        await graph.post(str(row.meta_campaign_id), {"status": meta_status})
    except MetaGraphError as exc:
        raise HTTPException(
            status_code=502, detail=f"Meta rejected the status change: {exc}"
        ) from exc


@router.patch("/{campaign_id}", response_model=Campaign)
async def update_campaign(
    campaign_id: str,
    payload: CampaignUpdateRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> Campaign:
    """Update name, status, budget, notes or end date.

    ``status`` is mirrored to Meta when the campaign is linked to a Meta campaign,
    before any local write, so the two can never disagree about delivery.
    """
    await require_role(principal, settings, ws, minimum="member")
    stmt = select(CampaignRow).where(
        CampaignRow.id == campaign_id, CampaignRow.business_id == ws.business_id
    )
    row = assert_owned(
        await db.scalar(stmt), ws, entity=f"campaign '{campaign_id}'"
    )

    meta_status = _META_STATUS.get((payload.status or "").lower())
    if payload.status is not None and payload.status != row.status:
        if meta_status is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{payload.status}' is not a status Meta understands. "
                    "Use 'active' or 'paused' to change delivery."
                ),
            )
        if not row.meta_campaign_id:
            # Nothing on Meta to mirror to. Writing `active` locally would claim
            # delivery for a campaign that does not exist, so refuse instead.
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Campaign '{row.name}' has not been published to Meta yet, so "
                    "its delivery state cannot be changed. Publish it from the "
                    "campaign wizard first."
                ),
            )
        await _mirror_status_to_meta(db, row, meta_status, settings)

    if payload.name is not None:
        row.name = payload.name
    if payload.objective is not None:
        row.objective = payload.objective
    if payload.channel is not None:
        row.channel = payload.channel
    if payload.budget is not None:
        if payload.budget < 0:
            raise HTTPException(status_code=422, detail="budget must not be negative")
        row.budget = payload.budget
    if payload.status is not None:
        row.status = payload.status
    if payload.notes is not None:
        row.notes = payload.notes
    if payload.end_date is not None:
        row.end_date = payload.end_date

    await db.commit()
    # The commit expires the row; converting without this refresh would trigger
    # lazy IO outside the greenlet.
    await db.refresh(row)
    return _to_schema(row)


@router.post("/meta-ad-plan", response_model=MetaAdPlanCreatedResponse)
async def create_meta_ad_plan(
    payload: MetaAdPlanCreateRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> MetaAdPlanCreatedResponse:
    """Save a local campaign draft and a pending, human-reviewable Meta plan.

    This route performs no Meta or AI request. The actual writes are reachable
    only through the existing two-phase recommendation approval endpoint.
    """
    await require_role(principal, settings, ws, minimum="member")
    try:
        # Page identity is owner-managed connector configuration, not a field a
        # campaign user should have to copy into the wizard.
        meta_connection = await resolve_meta_connection(db, ws.business_id, settings)
        plan = payload.plan
        if not plan.page_id and meta_connection.page_id:
            plan = plan.model_copy(update={"page_id": meta_connection.page_id})
        campaign, recommendation = await create_meta_ad_plan_draft(
            db,
            business_id=ws.business_id,
            campaign_input=payload.campaign,
            plan=plan,
        )
    except MetaAdPlanValidationError as exc:
        raise ValidationFailed(
            str(exc), detail={"missingFields": exc.missing_fields}
        ) from exc
    await db.commit()
    await db.refresh(campaign)
    await db.refresh(recommendation)
    return MetaAdPlanCreatedResponse(
        campaign_id=campaign.id,
        recommendation_id=recommendation.id,
        title=recommendation.title,
        status=recommendation.status,
        message="Plan saved locally. No Meta request has been made.",
    )


@router.post("/meta-ad-plan/assist", response_model=MetaAdPlanAssistResponse)
async def assist_meta_ad_plan_draft(
    payload: MetaAdPlanAssistRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
    llm: LLMDep,
) -> MetaAdPlanAssistResponse:
    """Use LLM + read-only Meta MCP to prepare editable wizard suggestions.

    The MCP client is allowlisted to ``ads_get_field_context`` only. This route
    does not write to the database or Meta; the existing Graph API approval
    flow remains the only path that creates paused Meta entities.
    """
    await require_role(principal, settings, ws, minimum="member")
    connection = await resolve_meta_connection(db, ws.business_id, settings)
    if not connection.access_token:
        raise HTTPException(
            status_code=409,
            detail=(
                "Connect Meta before using the campaign assistant. Meta Ads MCP also "
                "requires ads_mcp_management plus ads_read or ads_management. "
                "Do not paste an access token into chat."
            ),
        )
    try:
        async with MetaAdsMcpClient(connection.access_token) as mcp:
            return await assist_meta_ad_plan(llm=llm, mcp=mcp, request=payload)
    except MetaAdsMcpError as exc:
        status_code = 409 if exc.status_code in {401, 403} else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except MetaPlanAssistError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/strategy", response_model=GeneratedCampaignStrategy)
async def generate_campaign_strategy(
    payload: CampaignStrategyGenerationRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> GeneratedCampaignStrategy:
    """Generate an Omar strategy draft; this endpoint never writes to Meta."""
    await require_role(principal, settings, ws, minimum="member")
    return await generate_campaign_strategy_draft(
        db,
        business_id=ws.business_id,
        request=payload,
    )


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

    # The campaign's Overview/Performance tabs read `recentInsights.metrics`. This
    # was never populated, so those tabs rendered their empty state even for a
    # campaign with real Meta performance rows.
    latest_insight = (
        await db.execute(
            select(Insight)
            .where(Insight.campaign_id == campaign_id)
            .order_by(Insight.created_at.desc())
            .limit(1)
        )
    ).scalars().first()

    return CampaignDetailResponse(
        campaign=_to_schema(
            row, creatives=len(creatives), recommendations=len(recommendations)
        ),
        recent_insights=(
            {"metrics": latest_insight.metrics or {}}
            if latest_insight is not None
            else None
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


__all__ = [
    "create_campaign",
    "generate_campaign_strategy",
    "generate_campaign_strategy_endpoint",
    "get_campaign",
    "list_campaigns",
]
