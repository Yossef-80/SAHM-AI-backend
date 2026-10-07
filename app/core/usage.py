"""Cost guard: per-workspace daily token budget and per-run caps.

Every LLM call's usage is recorded in ``usage_records``. Before a run starts and
before each subsequent LLM call, the guard checks the workspace's spend for the
current UTC day against the configured budget and raises ``TokenBudgetExceeded``
with a clear message rather than silently running up a bill.

P4.1 added what the meter needs to be honest:

* a *period* window (``today`` / ``month``), not just the UTC day;
* per-workspace budget resolution (``businesses.daily_token_budget`` with the
  process env as the fallback) so two tenants can hold different limits;
* burn rate and projection, so ``GET /usage`` can say "at this rate you will hit
  the ceiling at 18:40" instead of only "you have used 40%".

Periods are UTC-bounded. That is a deliberate, documented limitation rather than
an oversight: the roadmap's Q5 flags the UTC-midnight reset as wrong for an
Egyptian or Gulf customer, and fixing it properly needs a per-workspace timezone
column. Until then one honest UTC boundary beats a per-customer boundary that is
only approximately right.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select

from app.config import get_settings
from app.core.errors import TokenBudgetExceeded
from app.core.logging import get_logger
from app.db.models import Business, UsageRecord
from app.schemas.common import Usage

_logger = get_logger(__name__)

#: The periods ``GET /usage`` accepts.
USAGE_PERIODS: tuple[str, ...] = ("today", "month")


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


# ------------------------------------------------------------------ windows ---


@dataclass(frozen=True)
class UsageWindow:
    """Tokens and call counts over one time window."""

    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


async def workspace_usage_since(db: Any, workspace_id: str, since: datetime) -> UsageWindow:
    """Everything this workspace has spent since ``since`` (inclusive).

    One query, three aggregates, rather than the two queries
    :func:`workspace_usage_today` used to run: the meter is called on every poll
    of the usage screen and the difference is visible at any real row count.
    """
    row = (
        await db.execute(
            select(
                func.coalesce(func.sum(UsageRecord.tokens_in), 0),
                func.coalesce(func.sum(UsageRecord.tokens_out), 0),
                func.count(UsageRecord.id),
            ).where(
                UsageRecord.workspace_id == workspace_id,
                UsageRecord.created_at >= since,
            )
        )
    ).one()
    return UsageWindow(
        tokens_in=int(row[0] or 0), tokens_out=int(row[1] or 0), calls=int(row[2] or 0)
    )


def period_start(period: str, now: datetime | None = None) -> datetime:
    """Start of ``today`` (UTC midnight) or of ``month`` (1st, UTC)."""
    moment = now or datetime.now(timezone.utc)
    if period == "today":
        return moment.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "month":
        return moment.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
    raise ValueError(f"unknown usage period '{period}'; expected {USAGE_PERIODS}")


def period_end(period: str, now: datetime | None = None) -> datetime:
    """When the period resets. This is what ``GET /usage`` returns as ``resetAt``."""
    start = period_start(period, now)
    if period == "today":
        return start + timedelta(days=1)
    if period == "month":
        return (start + timedelta(days=32)).replace(day=1)
    raise ValueError(f"unknown usage period '{period}'; expected {USAGE_PERIODS}")


async def workspace_usage_period(
    db: Any, workspace_id: str, period: str = "today", now: datetime | None = None
) -> UsageWindow:
    """Usage for one period."""
    return await workspace_usage_since(db, workspace_id, period_start(period, now))


async def usage_breakdown(
    db: Any,
    workspace_id: str,
    *,
    since: datetime,
    column: Any,
    limit: int = 20,
) -> list[tuple[str | None, int]]:
    """``[(group key, tokens)]`` for ``byModel`` / ``byPersona``.

    A NULL key (usage recorded outside any persona/model) is folded into
    ``"unknown"`` rather than dropped: a total that does not add up to the
    headline number is worse than an ugly label.
    """
    key = func.coalesce(column, "unknown")
    tokens = func.coalesce(
        func.sum(UsageRecord.tokens_in + UsageRecord.tokens_out), 0
    )
    rows = (
        await db.execute(
            select(key, tokens)
            .where(UsageRecord.workspace_id == workspace_id, UsageRecord.created_at >= since)
            .group_by(key)
            .order_by(tokens.desc())
            .limit(limit)
        )
    ).all()
    return [(str(r[0]), int(r[1] or 0)) for r in rows]


# ------------------------------------------------------------------- budget ---


async def effective_daily_budget(db: Any, workspace_id: str, default: int | None = None) -> int:
    """This workspace's daily budget.

    Resolution order: ``businesses.daily_token_budget`` (the per-workspace
    override P2.7 added) -> the caller's fallback -> ``Settings.daily_token_budget``.
    A NULL column therefore means "inherit", not "zero", so a workspace created
    before the override existed keeps working.
    """
    override = await db.scalar(
        select(Business.daily_token_budget).where(
            Business.workspace_id == workspace_id
        )
    )
    if override is not None:
        return int(override)
    if default is not None:
        return int(default)
    return int(get_settings().daily_token_budget)


async def check_budget(db: Any, workspace_id: str, budget: int | None = None) -> BudgetStatus:
    """Raise ``TokenBudgetExceeded`` when the workspace is over its daily budget.

    ``budget`` is the *fallback* used when the workspace has no override of its
    own, so the caller (the agent runtime) keeps passing its configured value and
    two workspaces with different limits still stop at different thresholds.
    """
    effective = await effective_daily_budget(db, workspace_id, budget)
    used = await workspace_usage_today(db, workspace_id)
    status = BudgetStatus(
        workspace_id=workspace_id,
        used_today=used,
        budget=effective,
        remaining=max(0, effective - used),
    )
    if status.exceeded:
        _logger.warning(
            "token_budget_exceeded",
            extra={"workspace_id": workspace_id, "used": used, "budget": effective},
        )
        raise TokenBudgetExceeded(
            f"This workspace has used {used:,} of its {effective:,} daily token "
            "budget. Raise the workspace budget in settings, raise "
            "DAILY_TOKEN_BUDGET, or wait until tomorrow (UTC).",
            detail={"used_today": used, "budget": effective},
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
    "USAGE_PERIODS",
    "BudgetStatus",
    "RunUsageTracker",
    "UsageWindow",
    "check_budget",
    "effective_daily_budget",
    "period_end",
    "period_start",
    "record_usage",
    "usage_breakdown",
    "workspace_usage_period",
    "workspace_usage_since",
    "workspace_usage_today",
]
