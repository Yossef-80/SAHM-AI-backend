"""Cost guard: per-workspace daily token budget and per-run caps.

Every LLM call's usage is recorded in ``usage_records``. Before a run starts and
before each subsequent LLM call, the guard checks the workspace's spend for the
current UTC day against the configured budget and raises ``TokenBudgetExceeded``
with a clear message rather than silently running up a bill.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select

from app.core.errors import TokenBudgetExceeded
from app.core.logging import get_logger
from app.db.models import UsageRecord
from app.schemas.common import Usage

_logger = get_logger(__name__)


@dataclass
class BudgetStatus:
    workspace_id: str
    used_today: int
    budget: int
    remaining: int

    @property
    def exceeded(self) -> bool:
        return self.used_today >= self.budget


async def workspace_usage_today(db: Any, workspace_id: str) -> int:
    """Tokens consumed by this workspace since UTC midnight."""
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    stmt = (
        select(func.coalesce(func.sum(UsageRecord.tokens_in), 0))
        .where(
            UsageRecord.workspace_id == workspace_id,
            UsageRecord.created_at >= start_of_day,
        )
    )
    stmt_out = (
        select(func.coalesce(func.sum(UsageRecord.tokens_out), 0))
        .where(
            UsageRecord.workspace_id == workspace_id,
            UsageRecord.created_at >= start_of_day,
        )
    )
    total_in = int((await db.execute(stmt)).scalar() or 0)
    total_out = int((await db.execute(stmt_out)).scalar() or 0)
    return total_in + total_out


async def check_budget(db: Any, workspace_id: str, budget: int) -> BudgetStatus:
    used = await workspace_usage_today(db, workspace_id)
    status = BudgetStatus(
        workspace_id=workspace_id,
        used_today=used,
        budget=budget,
        remaining=max(0, budget - used),
    )
    if status.exceeded:
        _logger.warning(
            "token_budget_exceeded",
            extra={"workspace_id": workspace_id, "used": used, "budget": budget},
        )
        raise TokenBudgetExceeded(
            f"This workspace has used {used:,} of its {budget:,} daily token budget. "
            "Raise DAILY_TOKEN_BUDGET or wait until tomorrow (UTC).",
            detail={"used_today": used, "budget": budget},
        )
    return status


async def record_usage(
    db: Any,
    *,
    business_id: str,
    workspace_id: str,
    usage: Usage,
    task_id: str | None = None,
    persona_id: str | None = None,
    tier: str | None = None,
) -> None:
    """Append one usage row. Never raises: accounting must not break a run."""
    try:
        db.add(
            UsageRecord(
                id=f"ur_{uuid.uuid4().hex[:24]}",
                business_id=business_id,
                workspace_id=workspace_id,
                task_id=task_id,
                persona_id=persona_id,
                tier=tier,
                model=usage.model,
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
            )
        )
        await db.flush()
    except Exception as exc:  # pragma: no cover
        _logger.warning("usage_record_failed", extra={"error": str(exc)})


class RunUsageTracker:
    """Accumulates usage for one run and enforces the per-run token cap."""

    def __init__(self, *, max_tokens_per_run: int) -> None:
        self.max_tokens_per_run = max_tokens_per_run
        self.total = Usage()

    def add(self, usage: Usage) -> None:
        self.total = self.total + usage

    @property
    def tokens(self) -> int:
        return self.total.tokens_in + self.total.tokens_out

    @property
    def exceeded(self) -> bool:
        return self.tokens > self.max_tokens_per_run


__all__ = [
    "BudgetStatus",
    "RunUsageTracker",
    "check_budget",
    "record_usage",
    "workspace_usage_today",
]
