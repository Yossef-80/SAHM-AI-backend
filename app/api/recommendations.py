"""Recommendation endpoints, including the two-phase approval flow."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import RequestIdDep, SessionDep
from app.core.approvals import service as approvals
from app.db.models import Recommendation as RecommendationRow
from app.schemas.recommendations import (
    ApproveRecommendationRequest,
    ApproveRecommendationResponse,
    Recommendation,
    RejectRecommendationRequest,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("", response_model=list[Recommendation])
async def list_recommendations(
    db: SessionDep, status: str | None = None
) -> list[Recommendation]:
    stmt = select(RecommendationRow).order_by(RecommendationRow.created_at.desc()).limit(100)
    if status:
        stmt = stmt.where(RecommendationRow.status == status)
    rows = (await db.execute(stmt)).scalars().all()
    return [approvals._to_schema(r) for r in rows]


@router.get("/{recommendation_id}", response_model=Recommendation)
async def get_recommendation(recommendation_id: str, db: SessionDep) -> Recommendation:
    return await approvals.get_recommendation(db, recommendation_id)


@router.post("/{recommendation_id}/approve", response_model=ApproveRecommendationResponse)
async def approve_recommendation(
    recommendation_id: str,
    payload: ApproveRecommendationRequest,
    db: SessionDep,
    request_id: RequestIdDep,
) -> ApproveRecommendationResponse:
    """Phase 1 returns the diff; phase 2 (confirm=true) executes on Meta.

    This is the ONLY route that reaches the protected tools, via
    approvals/service.py.
    """
    return await approvals.approve_recommendation(
        db, recommendation_id, payload, actor="user", request_id=request_id
    )


@router.post("/{recommendation_id}/reject", response_model=Recommendation)
async def reject_recommendation(
    recommendation_id: str,
    payload: RejectRecommendationRequest,
    db: SessionDep,
    request_id: RequestIdDep,
) -> Recommendation:
    return await approvals.reject_recommendation(
        db, recommendation_id, reason=payload.reason, actor="user", request_id=request_id
    )


__all__ = [
    "approve_recommendation",
    "get_recommendation",
    "list_recommendations",
    "reject_recommendation",
]
