"""Schemas for ``GET /api/v1/audit`` (P4.4).

``audit_log`` has been written by every approval and execution since the
approval flow shipped; before P4.4 no endpoint ever read it, so the table was
invisible. These schemas are what makes it a screen.

The redaction rule
------------------
``before`` / ``after`` / ``result`` are arbitrary JSON blobs that a tool chose to
put there. They are passed through ``app.core.logging.redact`` before
serialisation -- the same function that scrubs log lines -- because an audit
table is exactly the place a token would be stored "temporarily" and then
rediscovered eighteen months later.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import ApiModel


class AuditEntry(ApiModel):
    """One audit row."""

    id: str
    #: A real ``users.id`` since P1 fixed the hardcoded ``"user"`` literal.
    actor: str = "system"
    action: str
    entity_type: str = "recommendation"
    entity_id: str | None = None
    business_id: str | None = None
    request_id: str | None = None
    #: Redacted before serialisation; never a raw secret.
    before: dict[str, object] | None = None
    after: dict[str, object] | None = None
    result: dict[str, object] | None = None
    created_at: datetime | None = None


class AuditList(ApiModel):
    """``GET /audit`` response."""

    entries: list[AuditEntry] = Field(default_factory=list)
    limit: int = 50
    total: int = 0
    entity_type: str | None = None
    entity_id: str | None = None
    actor: str | None = None


__all__ = ["AuditEntry", "AuditList"]