"""Recommendation endpoints, including the two-phase approval flow."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    AuthDep,
    RequestIdDep,
    SessionDep,
    SettingsDep,
    WorkspaceContext,
    WorkspaceDep,
    assert_owned,
    require_role,
)
from app.ai.marketing import meta_ads
from app.core.approvals import service as approvals
from app.db.models import Recommendation as RecommendationRow
from app.schemas.recommendations import (
    ActivateRecommendationRequest,
    ActivateRecommendationResponse,
    ApproveRecommendationRequest,
    ApproveRecommendationResponse,
    Recommendation,
    RejectRecommendationRequest,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def _owned_recommendations(ws: WorkspaceContext):
    """Recommendations belonging to this workspace, newest first.

    A single indexed equality on the denormalised ``workspace_id`` column --
    the join through ``campaign_id`` that P0 needed is gone. ``NULL`` still
    matches nothing, so a recommendation raised outside a campaign, or one whose
    business has been deleted (``ondelete="SET NULL"``), stays invisible to
    everyone: the column fails closed by construction.
    """
    return (
        select(RecommendationRow)
        .where(RecommendationRow.workspace_id == ws.business_id)
        .order_by(RecommendationRow.created_at.desc())
        .limit(100)
    )


async def _load_owned(
    db: AsyncSession, recommendation_id: str, ws: WorkspaceContext
) -> RecommendationRow:
    """Load one recommendation, 404 across a workspace boundary.

    ``assert_owned`` reads the row's own ``workspace_id`` here, so no join and
    no lazy relationship access is needed -- a row whose tenant cannot be
    resolved is unowned, and unowned is a 404 for everyone.
    """
    row = await db.scalar(
        select(RecommendationRow).where(RecommendationRow.id == recommendation_id)
    )
    return assert_owned(
        row, ws, entity=f"recommendation '{recommendation_id}'", owner=row.workspace_id if row else None
    )


@router.get("", response_model=list[Recommendation])
async def list_recommendations(
    ws: WorkspaceDep, db: SessionDep, status: str | None = None
) -> list[Recommendation]:
    """This workspace's recommendations only (D3: there was no filter at all)."""
    stmt = _owned_recommendations(ws)
    if status:
        stmt = stmt.where(RecommendationRow.status == status)
    rows = (await db.execute(stmt)).scalars().all()
    return [approvals._to_schema(r) for r in rows]


@router.get("/{recommendation_id}", response_model=Recommendation)
async def get_recommendation(
    recommendation_id: str, ws: WorkspaceDep, db: SessionDep
) -> Recommendation:
    return approvals._to_schema(await _load_owned(db, recommendation_id, ws))


@router.post("/{recommendation_id}/approve", response_model=ApproveRecommendationResponse)
async def approve_recommendation(
    recommendation_id: str,
    payload: ApproveRecommendationRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    request_id: RequestIdDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> ApproveRecommendationResponse:
    """Phase 1 returns the diff; phase 2 (confirm=true) executes on Meta.

    This is the ONLY route that reaches the protected tools, via
    approvals/service.py. It is therefore the single most consequential write in
    the product -- which is why it requires ``member`` or better and attributes
    the approval to the caller's real user id rather than the literal "user".
    """
    actor = await require_role(principal, settings, ws, minimum="member")
    await _load_owned(db, recommendation_id, ws)
    return await approvals.approve_recommendation(
        db,
        recommendation_id,
        payload,
        actor=actor.user_id if actor else "anonymous",
        request_id=request_id,
        settings=settings,
    )


def _activation_targets(row: RecommendationRow) -> list[str]:
    """Meta object labels recorded by plan execution, in parent-first order."""
    execution = row.execution_result if isinstance(row.execution_result, dict) else {}
    created_raw = execution.get("created_ids")
    created = created_raw if isinstance(created_raw, dict) else {}
    targets: list[str] = []
    for step in ("campaign", "ad_set"):
        value = created.get(step)
        if isinstance(value, str) and value:
            targets.append(step)
    ads_raw = created.get("ads")
    if isinstance(ads_raw, list):
        targets.extend(f"ad:{ad}" for ad in ads_raw if isinstance(ad, str) and ad)
    return targets


@router.post("/{recommendation_id}/activate", response_model=ActivateRecommendationResponse)
async def activate_recommendation(
    recommendation_id: str,
    payload: ActivateRecommendationRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> ActivateRecommendationResponse:
    """Start delivery on an already-created PAUSED Meta plan.

    Everything the wizard creates is PAUSED, so without this a campaign can
    never actually run. It is two-phase like approval: creating Meta objects is
    reversible, switching them to ACTIVE spends real budget and is not.
    """
    actor = await require_role(principal, settings, ws, minimum="member")
    row = await _load_owned(db, recommendation_id, ws)
    targets = _activation_targets(row)

    if not payload.confirm:
        return ActivateRecommendationResponse(
            recommendation=approvals._to_schema(row),
            executed=False,
            requires_confirmation=True,
            targets=targets,
            message=(
                "Confirm to switch these Meta objects to ACTIVE. The campaign "
                "will start delivering and spending its budget."
            ),
        )

    result = await meta_ads.activate_meta_ad_plan(
        db,
        row,
        actor=actor.user_id if actor else "anonymous",
        settings=settings,
    )
    await db.commit()
    # The commit expires every attribute on the row; without this refresh the
    # schema conversion below would trigger lazy IO outside the greenlet and 500.
    await db.refresh(row)
    return ActivateRecommendationResponse(
        recommendation=approvals._to_schema(row),
        executed=result.status == "ok",
        requires_confirmation=False,
        targets=targets,
        activated=result.activated,
        message=result.detail,
        unavailable_reason=result.unavailable_reason,
    )


@router.post("/{recommendation_id}/reject", response_model=Recommendation)
async def reject_recommendation(
    recommendation_id: str,
    payload: RejectRecommendationRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    request_id: RequestIdDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> Recommendation:
    actor = await require_role(principal, settings, ws, minimum="member")
    await _load_owned(db, recommendation_id, ws)
    return await approvals.reject_recommendation(
        db,
        recommendation_id,
        reason=payload.reason,
        actor=actor.user_id if actor else "anonymous",
        request_id=request_id,
    )


__all__ = [
    "approve_recommendation",
    "get_recommendation",
    "list_recommendations",
    "reject_recommendation",
]
