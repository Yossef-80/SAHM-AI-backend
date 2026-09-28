"""Layer 7: human-in-the-loop approvals.

This module is the ONLY importer of ``app.core.tools.protected``. That is the
structural guarantee that no agent can reach an external write: the registry
refuses to hold external_write tools, and the only two callers of those handlers
live behind this file.

Two-phase approval:
    POST /recommendations/{id}/approve            -> phase 1: returns the diff
    POST /recommendations/{id}/approve {confirm}  -> phase 2: executes

Every approval and every execution writes an ``audit_log`` row.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import (
    ApprovalNotFoundError,
    ApprovalRequired,
    ApprovalStateError,
)
from app.core.logging import get_logger
from app.core.tools import protected
from app.core.tools.protected import (
    ApplyChangeInput,
    ApplyChangeOutput,
    PublishCampaignInput,
    PublishCampaignOutput,
)
from app.db.models import AuditLog, Campaign
from app.db.models import Recommendation as RecommendationRow
from app.schemas.recommendations import (
    ApproveRecommendationRequest,
    ApproveRecommendationResponse,
    ChangeDiff,
    Recommendation,
    RecommendationStatus,
)

_logger = get_logger(__name__)


# ------------------------------------------------------------------ helpers --


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


async def _write_audit(
    db: AsyncSession,
    *,
    actor: str,
    action: str,
    entity_id: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    result: dict[str, Any] | None,
    request_id: str | None = None,
    business_id: str | None = None,
) -> AuditLog:
    row = AuditLog(
        id=_new_id("aud"),
        actor=actor,
        action=action,
        entity_type="recommendation",
        entity_id=entity_id,
        before=before,
        after=after,
        result=result,
        request_id=request_id,
        business_id=business_id,
    )
    db.add(row)
    await db.flush()
    _logger.info(
        "audit_written",
        extra={"action": action, "entity_id": entity_id, "actor": actor},
    )
    return row


def _current_value_for(db: AsyncSession, recommendation: RecommendationRow, key: str) -> Any:
    """Best-known current value of the field being changed.

    Read from our own records. When we do not hold it (e.g. the value lives only
    in Meta), return None so the UI shows "unknown" rather than a guess.
    """
    if key in ("budget", "daily_budget", "lifetime_budget"):
        if recommendation.campaign_id:
            campaign = db.get(Campaign, recommendation.campaign_id)
            if campaign is not None and key == "budget":
                return campaign.budget
    return None


async def build_diff(db: AsyncSession, recommendation: RecommendationRow) -> list[ChangeDiff]:
    """The before -> after diff the review dialog shows.

    Only fields present in the recommendation's payload are included, so the
    dialog never implies a change that is not being made.
    """
    diffs: list[ChangeDiff] = []
    payload = recommendation.payload or {}
    for key in sorted(payload):
        proposed = payload[key]
        current = _current_value_for(db, recommendation, key)
        diffs.append(ChangeDiff(field=key, before=current, after=proposed))
    if not diffs:
        diffs.append(
            ChangeDiff(
                field="action",
                before=None,
                after={
                    "action_type": recommendation.action_type,
                    "title": recommendation.title,
                },
            )
        )
    return diffs


def _to_schema(row: RecommendationRow) -> Recommendation:
    return Recommendation(
        id=row.id,
        campaign_id=row.campaign_id,
        insight_id=row.insight_id,
        action_type=row.action_type,
        title=row.title,
        rationale=row.rationale,
        payload=row.payload or {},
        evidence=row.evidence or [],
        status=RecommendationStatus(row.status),
        persona_id=row.persona_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        executed_at=row.executed_at,
        requires_approval=bool(row.requires_approval),
        estimated_impact=row.estimated_impact,
    )


async def get_recommendation(db: AsyncSession, recommendation_id: str) -> Recommendation:
    row = await db.get(RecommendationRow, recommendation_id)
    if row is None:
        raise ApprovalNotFoundError(f"recommendation '{recommendation_id}' not found")
    return _to_schema(row)


# -------------------------------------------------------------- the flow ----


async def approve_recommendation(
    db: AsyncSession,
    recommendation_id: str,
    request: ApproveRecommendationRequest,
    *,
    actor: str = "user",
    request_id: str | None = None,
) -> ApproveRecommendationResponse:
    """Phase 1 records the diff; phase 2 executes.

    Phase 1 (``confirm=False``): marks the recommendation approved and returns the
    diff. Nothing is written to Meta.
    Phase 2 (``confirm=True``): calls ``protected.apply_change`` and writes an
    execution audit row.
    """
    row = await db.get(RecommendationRow, recommendation_id)
    if row is None:
        raise ApprovalNotFoundError(f"recommendation '{recommendation_id}' not found")

    if row.status == RecommendationStatus.EXECUTED.value:
        raise ApprovalStateError(
            f"recommendation '{recommendation_id}' has already been executed"
        )
    if row.status == RecommendationStatus.REJECTED.value:
        raise ApprovalStateError(
            f"recommendation '{recommendation_id}' was rejected and cannot be approved"
        )

    before = {
        "status": row.status,
        "payload": row.payload or {},
        "approved_by": row.approved_by,
    }
    diffs = await build_diff(db, row)

    # ---------------------------------------------------------- phase 1 ----
    if not request.confirm:
        if row.status != RecommendationStatus.APPROVED.value:
            row.status = RecommendationStatus.APPROVED.value
            row.approved_by = actor
            row.approved_at = _now()
            row.diff = [d.model_dump(mode="json") for d in diffs]
            await db.flush()
        await _write_audit(
            db,
            actor=actor,
            action="recommendation.approved",
            entity_id=recommendation_id,
            before=before,
            after={"status": row.status, "diff": row.diff},
            result={"phase": 1, "executed": False},
            request_id=request_id,
        )
        await db.commit()
        await db.refresh(row)
        return ApproveRecommendationResponse(
            recommendation=_to_schema(row),
            diff=diffs,
            executed=False,
            requires_confirmation=True,
            message=(
                "Approval recorded. Review the diff above and confirm to execute "
                "the change on Meta."
            ),
            audit_log_id=None,
        )

    # ---------------------------------------------------------- phase 2 ----
    if row.status != RecommendationStatus.APPROVED.value:
        # Must go through phase 1 first so the diff is always reviewed.
        row.status = RecommendationStatus.APPROVED.value
        row.approved_by = actor
        row.approved_at = _now()
        row.diff = [d.model_dump(mode="json") for d in diffs]
        await db.flush()

    outcome = await _execute(db, row, actor=actor, request_id=request_id)

    after = {
        "status": row.status,
        "executed_at": row.executed_at.isoformat() if row.executed_at else None,
        "execution_result": row.execution_result,
    }
    audit = await _write_audit(
        db,
        actor=actor,
        action="recommendation.executed",
        entity_id=recommendation_id,
        before=before,
        after=after,
        result=outcome.model_dump(mode="json"),
        request_id=request_id,
    )
    await db.commit()
    await db.refresh(row)

    executed = row.status == RecommendationStatus.EXECUTED.value
    return ApproveRecommendationResponse(
        recommendation=_to_schema(row),
        diff=diffs,
        executed=executed,
        requires_confirmation=False,
        message=(
            "Change applied to Meta."
            if executed
            else f"Execution did not complete: {outcome.unavailable_reason or outcome.detail}"
        ),
        audit_log_id=audit.id,
    )


async def _execute(
    db: AsyncSession, row: RecommendationRow, *, actor: str, request_id: str | None
) -> ApplyChangeOutput:
    """The single call site of the protected ``apply_change`` tool."""
    from app.core.tools.base import ToolContext

    # A ToolContext is required by the handler. It is built here, inside the
    # approvals service, which is the only place allowed to do this.
    ctx = ToolContext(
        workspace_id="default",
        business_id=row.campaign_id or "",
        db=db,
        llm=None,  # type: ignore[arg-type]
        request_id=request_id or "",
        campaign_id=row.campaign_id,
        persona_id=row.persona_id,
    )
    # The protected handler does not use the LLM; pass a stub to keep the type.
    from app.core.llm.mock_client import MockClient

    ctx.llm = MockClient()

    outcome = await protected.apply_change.handler(
        ApplyChangeInput(
            recommendation_id=row.id,
            confirmed_by=actor,
            note="approved via API",
        ),
        ctx,
    )
    if outcome.status == "ok":
        row.status = RecommendationStatus.EXECUTED.value
        row.executed_at = _now()
        await db.flush()
    else:
        row.status = RecommendationStatus.FAILED.value
        await db.flush()
    return outcome


async def publish_campaign(
    db: AsyncSession,
    campaign_id: str,
    *,
    actor: str = "user",
    request_id: str | None = None,
) -> PublishCampaignOutput:
    """Publish a campaign. The second (and only other) protected call site."""
    from app.core.llm.mock_client import MockClient
    from app.core.tools.base import ToolContext

    campaign = await db.get(Campaign, campaign_id)
    if campaign is None:
        raise ApprovalNotFoundError(f"campaign '{campaign_id}' not found")

    before = {"status": campaign.status, "meta_campaign_id": campaign.meta_campaign_id}

    ctx = ToolContext(
        workspace_id="default",
        business_id=campaign.business_id,
        db=db,
        llm=MockClient(),
        request_id=request_id or "",
        campaign_id=campaign_id,
    )
    outcome = await protected.publish_campaign.handler(
        PublishCampaignInput(campaign_id=campaign_id, confirmed_by=actor),
        ctx,
    )
    await _write_audit(
        db,
        actor=actor,
        action="campaign.published",
        entity_id=campaign_id,
        before=before,
        after={
            "status": campaign.status,
            "meta_campaign_id": campaign.meta_campaign_id,
        },
        result=outcome.model_dump(mode="json"),
        request_id=request_id,
    )
    await db.commit()
    return outcome


async def reject_recommendation(
    db: AsyncSession,
    recommendation_id: str,
    *,
    reason: str | None = None,
    actor: str = "user",
    request_id: str | None = None,
) -> Recommendation:
    row = await db.get(RecommendationRow, recommendation_id)
    if row is None:
        raise ApprovalNotFoundError(f"recommendation '{recommendation_id}' not found")
    if row.status == RecommendationStatus.EXECUTED.value:
        raise ApprovalStateError(
            f"recommendation '{recommendation_id}' has already been executed"
        )
    before = {"status": row.status}
    row.status = RecommendationStatus.REJECTED.value
    await db.flush()
    await _write_audit(
        db,
        actor=actor,
        action="recommendation.rejected",
        entity_id=recommendation_id,
        before=before,
        after={"status": row.status},
        result={"reason": reason},
        request_id=request_id,
    )
    await db.commit()
    await db.refresh(row)
    return _to_schema(row)


async def list_audit(
    db: AsyncSession, *, entity_id: str | None = None, limit: int = 50
) -> list[AuditLog]:
    stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    if entity_id:
        stmt = stmt.where(AuditLog.entity_id == entity_id)
    return list((await db.execute(stmt)).scalars().all())


__all__ = [
    "approve_recommendation",
    "build_diff",
    "get_recommendation",
    "list_audit",
    "publish_campaign",
    "reject_recommendation",
]
