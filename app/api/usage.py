"""``GET /api/v1/usage`` -- the meter the UI shows (P4.1).

Why the maths is in this file and not only in ``app/core/usage.py``
------------------------------------------------------------------
``app/core/usage.py`` owns the *query* and the enforcement rule. This file owns
the *arithmetic a customer is shown*, and it is deliberately explicit about the
three places a meter usually lies:

1. **Budget.** A monthly figure is 30 daily budgets, not the daily number with a
   bigger label. Using the daily ceiling for a month view would tell a customer
   they have used 400% of their plan.
2. **Percent.** Unclamped. 140% is 140%.
3. **Projection.** Linear, and derived from the *elapsed fraction* of the
   period, so it is 0 before any spend and equal to the spend at the very end
   of the period rather than dividing by zero.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.api.deps import SessionDep, WorkspaceDep
from app.config import get_settings
from app.core.errors import ValidationFailed
from app.core.usage import (
    USAGE_PERIODS,
    effective_daily_budget,
    period_end,
    period_start,
    usage_breakdown,
    workspace_usage_since,
)
from app.db.models import Business, UsageRecord
from app.schemas.usage import UsageBreakdown, UsageResponse

router = APIRouter(prefix="/usage", tags=["usage"])

#: The daily budget is enforced per UTC day, so a month's ceiling is the daily
#: one multiplied by this. 30 rather than 31/28: a month view is a budget
#: *signal*, and using the real calendar length would make the same spend report
#: a different percentage on the 28th than on the 30th, which is worse than a
#: stated approximation.
DAYS_PER_MONTH_BUDGET = 30


def _percent(used: int, budget: int) -> float:
    """``used / budget`` as a percentage. Unclamped; 0.0 when the budget is 0.

    A zero budget is a "no limit configured" state, not a division by zero: it
    reports 0% rather than an error, because returning an error for "you have no
    limit" is unhelpful.
    """
    if budget <= 0:
        return 0.0
    return round(used * 100.0 / budget, 2)


def _breakdown(
    rows: list[tuple[str | None, int]], total: int
) -> list[UsageBreakdown]:
    """Attach each group's share of the period total."""
    return [
        UsageBreakdown(
            key=key or "unknown",
            tokens=tokens,
            share=round(tokens / total, 4) if total > 0 else 0.0,
        )
        for key, tokens in rows
    ]


@router.get("", response_model=UsageResponse)
async def get_usage(
    ws: WorkspaceDep,
    db: SessionDep,
    period: str = Query(
        default="today", description="today | month", enum=list(USAGE_PERIODS)
    ),
) -> UsageResponse:
    """Token usage and budget for this workspace.

    Readable by every role -- a usage screen that only owners can see is a
    support ticket waiting to happen -- and scoped to the caller's workspace by
    the ``WorkspaceDep`` dependency, not by a filter added here.
    """
    if period not in USAGE_PERIODS:
        raise ValidationFailed(
            f"unknown period '{period}'",
            detail={"allowed": list(USAGE_PERIODS)},
        )

    now = datetime.now(timezone.utc)
    start = period_start(period, now)
    end = period_end(period, now)

    window = await workspace_usage_since(db, ws.workspace_id, start)
    total = window.total_tokens

    daily = await effective_daily_budget(db, ws.workspace_id)
    budget = daily * DAYS_PER_MONTH_BUDGET if period == "month" else daily
    override = await db.scalar(
        select(Business.daily_token_budget).where(
            Business.workspace_id == ws.workspace_id
        )
    )

    # Elapsed fraction, floored at one minute so a request at 00:00:30 does not
    # project a whole day from thirty seconds of spend.
    elapsed = max((now - start).total_seconds(), 60.0)
    total_seconds = max((end - start).total_seconds(), 1.0)
    hours_elapsed = elapsed / 3600.0
    burn_rate = round(total / hours_elapsed, 1) if hours_elapsed > 0 else 0.0
    projected = int(round(total * total_seconds / elapsed))

    models = await usage_breakdown(
        db, ws.workspace_id, since=start, column=UsageRecord.model
    )
    personas = await usage_breakdown(
        db, ws.workspace_id, since=start, column=UsageRecord.persona_id
    )
    actions = await usage_breakdown(
        db, ws.workspace_id, since=start, column=UsageRecord.action
    )
    estimated_calls = int(
        (
            await db.execute(
                select(func.count(UsageRecord.id)).where(
                    UsageRecord.workspace_id == ws.workspace_id,
                    UsageRecord.created_at >= start,
                    UsageRecord.estimated.is_(True),
                )
            )
        ).scalar()
        or 0
    )

    settings = get_settings()
    price_in = float(settings.token_price_in_usd_per_million or 0.0)
    price_out = float(settings.token_price_out_usd_per_million or 0.0)
    cost = (
        round(
            (window.tokens_in / 1_000_000) * price_in
            + (window.tokens_out / 1_000_000) * price_out,
            6,
        )
        if (price_in > 0 or price_out > 0)
        else None
    )

    return UsageResponse(
        workspace_id=ws.workspace_id,
        business_id=ws.business_id,
        period=period,
        tokens_in=window.tokens_in,
        tokens_out=window.tokens_out,
        total_tokens=total,
        budget_tokens=budget,
        remaining_tokens=max(0, budget - total),
        percent_used=_percent(total, budget),
        exceeded=total >= budget and budget > 0,
        reset_at=end,
        period_start=start,
        calls=window.calls,
        burn_rate_per_hour=burn_rate,
        projected_total_tokens=projected,
        budget_source="workspace" if override is not None else "default",
        estimated_calls=estimated_calls,
        estimated_cost_usd=cost,
        cost_per_million_in_usd=price_in,
        cost_per_million_out_usd=price_out,
        by_model=_breakdown(models, total),
        by_persona=_breakdown(personas, total),
        by_action=_breakdown(actions, total),
    )


__all__ = ["DAYS_PER_MONTH_BUDGET", "router"]