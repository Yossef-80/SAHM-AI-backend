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
from app.core.approvals import service as approvals
from app.db.models import Recommendation as RecommendationRow
from app.schemas.recommendations import (
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
