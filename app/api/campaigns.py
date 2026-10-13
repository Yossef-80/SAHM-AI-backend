"""Campaign endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings

from app.ai.marketing.meta_ads import (
    HttpMetaGraphClient,
    MetaAdPlanSettings,
    MetaAdPlanValidationError,
    MetaGraphError,
    create_meta_ad_plan_draft,
    resolve_meta_connection,
)
from app.core.errors import ValidationFailed
from app.core.campaigns.service import generate_campaign_strategy_draft
from app.schemas.common import ApiModel
from app.api.deps import (
    AuthDep,
    SessionDep,
    SettingsDep,
    WorkspaceDep,
    assert_owned,
    require_role,
)
from app.db.models import Campaign as CampaignRow, Creative, Insight, Recommendation
from app.schemas.campaigns import (
    Campaign,
    CampaignCreateRequest,
    CampaignDetailResponse,
    CampaignListResponse,
    CampaignStrategyGenerationRequest,
    CampaignUpdateRequest,
    GeneratedCampaignStrategy,
)

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


__all__ = ["create_campaign", "generate_campaign_strategy", "get_campaign", "list_campaigns"]
