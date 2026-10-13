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
from app.db.models import AuditLog, Business, Campaign
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


async def _current_value_for(
    db: AsyncSession, recommendation: RecommendationRow, key: str
) -> Any:
    """Best-known current value of the field being changed.

    Read from our own records. When we do not hold it (e.g. the value lives only
    in Meta), return None so the UI shows "unknown" rather than a guess.

    This is async because it queries through an ``AsyncSession``: the previous
    synchronous ``db.get(...)`` returned an un-awaited coroutine, so any diff on
    a budget field would have raised instead of showing a "before" value.
    """
    if key in ("budget", "daily_budget", "lifetime_budget"):
        if recommendation.campaign_id:
            campaign = await db.get(Campaign, recommendation.campaign_id)
            if campaign is not None and key == "budget":
                return campaign.budget
    return None


async def build_diff(db: AsyncSession, recommendation: RecommendationRow) -> list[ChangeDiff]:
    """The before -> after diff the review dialog shows.

    Only fields present in the recommendation's payload are included, so the
    dialog never implies a change that is not being made. A Meta ad plan gets a
    compact, explicit hierarchy summary; its complete non-secret settings remain
    on ``recommendation.payload.plan`` for the review UI.
    """
    if recommendation.action_type == "create_meta_ad_plan":
        payload = recommendation.payload or {}
        summary = payload.get("campaign_summary") if isinstance(payload, dict) else {}
        plan = payload.get("plan") if isinstance(payload, dict) else {}
        summary = summary if isinstance(summary, dict) else {}
        plan = plan if isinstance(plan, dict) else {}
        return [
            ChangeDiff(
                field="meta_creation_plan",
                before=None,
                after={
                    "campaign": summary.get("name"),
                    "objective": summary.get("objective"),
                    "daily_budget": summary.get("daily_budget"),
                    "currency": summary.get("currency"),
                    "special_ad_categories": plan.get("specialAdCategories"),
                    "page_id": plan.get("pageId"),
                    "billing_event": plan.get("billingEvent"),
                    "optimization_goal": plan.get("optimizationGoal"),
                    "destination_type": plan.get("destinationType"),
                    "destination_url": plan.get("destinationUrl"),
                    "creative_ids": plan.get("creativeIds", []),
                    "meta_status": "PAUSED",
                    "hierarchy": "Campaign → Ad Set → Creative(s) → Ad(s)",
                },
            )
        ]

    diffs: list[ChangeDiff] = []
    payload = recommendation.payload or {}
    for key in sorted(payload):
        proposed = payload[key]
        current = await _current_value_for(db, recommendation, key)
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
        execution_result=row.execution_result,
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
    settings: Any | None = None,
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
    if (
        row.action_type == "create_meta_ad_plan"
        and request.confirm
        and row.status != RecommendationStatus.APPROVED.value
    ):
        raise ApprovalStateError(
            "a Meta ad plan must complete phase 1 and have its diff reviewed before confirmation"
        )
    if row.action_type == "create_meta_ad_plan" and not request.confirm:
        from app.ai.marketing.meta_ads import preflight_meta_ad_plan
        from app.config import get_settings

        issues = await preflight_meta_ad_plan(
            db, row, settings=settings or get_settings()
        )
        if issues:
            raise ApprovalRequired(
                "The Meta ad plan cannot be approved until its required settings are complete.",
                detail={"issues": issues},
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

    outcome = await _execute(
        db, row, actor=actor, request_id=request_id, settings=settings
    )

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


async def _tenant_of_campaign(
    db: AsyncSession, campaign_id: str | None
) -> tuple[str, str]:
    """(workspace_id, business_id) owning ``campaign_id``.

    D2: the execute path used to build ``ToolContext(business_id=row.campaign_id)``,
    so every tool that resolves an integration or brand memory by
    ``ctx.business_id`` during execution looked up a business id that does not
    exist. The owning business is read back off the campaign instead.
    """
    if not campaign_id:
        return "", ""
    campaign = await db.get(Campaign, campaign_id)
    if campaign is None:
        return "", ""
    business = await db.get(Business, campaign.business_id)
    workspace_id = business.workspace_id or "" if business is not None else ""
    return workspace_id, campaign.business_id


async def _execute(
    db: AsyncSession,
    row: RecommendationRow,
    *,
    actor: str,
    request_id: str | None,
    settings: Any | None = None,
) -> Any:
    """The only call site of protected Meta write bridges.

    The legacy generic action path retains its existing ToolContext-based
    handler. The new multi-object campaign path deliberately does not create a
    ToolContext; it delegates through the narrow protected bridge.
    """
    if row.action_type == "create_meta_ad_plan":
        from app.config import get_settings

        outcome = await protected.execute_meta_ad_plan_protected(
            db,
            row,
            confirmed_by=actor,
            settings=settings or get_settings(),
        )
        if outcome.status == "ok":
            row.status = RecommendationStatus.EXECUTED.value
            row.executed_at = _now()
        else:
            # Failed/partial external writes stay visibly failed. The executor
            # records any created IDs and independently blocks an automatic
            # replay on the next confirmation attempt.
            row.status = RecommendationStatus.FAILED.value
        await db.flush()
        return outcome

    from app.core.tools.base import ToolContext

    workspace_id, business_id = await _tenant_of_campaign(db, row.campaign_id)

    # A ToolContext is required by the legacy handler. It is built here, inside
    # the approvals service, which is the only place allowed to do this.
    ctx = ToolContext(
        workspace_id=workspace_id,
        business_id=business_id,
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
    workspace_id, _ = await _tenant_of_campaign(db, campaign_id)

    ctx = ToolContext(
        workspace_id=workspace_id,
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
    db: AsyncSession,
    *,
    entity_id: str | None = None,
    business_id: str | None = None,
    entity_type: str | None = None,
    actor: str | None = None,
    action: str | None = None,
    limit: int = 50,
) -> list[AuditLog]:
    """Audit rows, newest first, with the filters ``GET /audit`` exposes.

    ``business_id`` was added in P4.4. Every filter is optional and the result
    is always a tenant-scoped list once ``business_id`` is passed -- the
    unscoped call is what the pre-P4.4 internal callers used, and it now has
    exactly one HTTP caller (``app/api/audit.py``) which always scopes it.
    """
    stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    if entity_id:
        stmt = stmt.where(AuditLog.entity_id == entity_id)
    if business_id:
        stmt = stmt.where(AuditLog.business_id == business_id)
    if entity_type:
        stmt = stmt.where(AuditLog.entity_type == entity_type)
    if actor:
        stmt = stmt.where(AuditLog.actor == actor)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    return list((await db.execute(stmt)).scalars().all())


__all__ = [
    "approve_recommendation",
    "build_diff",
    "get_recommendation",
    "list_audit",
    "publish_campaign",
    "reject_recommendation",
]
