"""``GET /api/v1/audit`` -- the audit log, finally readable (P4.4).

``audit_log`` has been written by every approval and execution since the
two-phase flow shipped. Before P4.4 nothing ever read it: the table existed, the
rows were correct, and there was no endpoint. This router is the read side.

Two decisions worth stating
---------------------------
**Owner only.** The task brief and the roadmap disagree here: §6 P4.4 says
"owner/member/viewer may read" while the task specifies owner-only. Owner-only is
implemented, because an audit log answers "what did a human change and whose
account did it" and the actor ids are customer-identifying. A narrower read is
the reversible mistake; a wider one is not.

**Redaction before serialisation.** ``before`` / ``after`` / ``result`` are
arbitrary JSON a tool wrote. They go through ``app.core.logging.redact`` -- the
function that scrubs log lines -- before being put in a response body, so the
audit screen can never become the place a token is rediscovered.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.api.deps import AuthDep, SessionDep, SettingsDep, WorkspaceDep, require_role
from app.core.approvals.service import list_audit
from app.core.logging import redact
from app.db.models import AuditLog
from app.schemas.audit import AuditEntry, AuditList

router = APIRouter(prefix="/audit", tags=["audit"])

#: Owner only -- see the module docstring.
READ_AUDIT = "owner"

DEFAULT_LIMIT = 50
MAX_LIMIT = 500


def _entry(row: AuditLog) -> AuditEntry:
    """Serialise one row with every payload redacted."""
    return AuditEntry(
        id=row.id,
        actor=row.actor,
        action=row.action,
        entity_type=row.entity_type,
        entity_id=row.entity_id,
        business_id=row.business_id,
        request_id=row.request_id,
        before=redact(row.before) if row.before else None,
        after=redact(row.after) if row.after else None,
        result=redact(row.result) if row.result else None,
        created_at=row.created_at,
    )


@router.get("", response_model=AuditList)
async def get_audit(
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    db: SessionDep,
    entity_type: str | None = Query(default=None, max_length=60),
    entity_id: str | None = Query(default=None, max_length=64),
    actor: str | None = Query(default=None, max_length=120),
    action: str | None = Query(default=None, max_length=80),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
) -> AuditList:
    """This workspace's audit entries, newest first.

    Scoped to the caller's business on every read: the tenant predicate is passed
    into ``list_audit`` (P4.4 added it there for exactly this endpoint), so the
    query lives in one place rather than being duplicated in the router.
    """
    await require_role(principal, settings, ws, minimum=READ_AUDIT)

    rows = await list_audit(
        db,
        entity_id=entity_id,
        business_id=ws.business_id,
        entity_type=entity_type,
        actor=actor,
        action=action,
        limit=limit,
    )
    total = int(
        await db.scalar(
            select(func.count()).select_from(AuditLog).where(
                AuditLog.business_id == ws.business_id
            )
        )
        or 0
    )

    return AuditList(
        entries=[_entry(row) for row in rows],
        limit=limit,
        total=total,
        entity_type=entity_type,
        entity_id=entity_id,
        actor=actor,
    )


__all__ = ["READ_AUDIT", "router"]