"""Schemas for ``GET /api/v1/usage`` (P4.1).

The numbers have to be *honest*, so the shape carries three things a bare
``used/budget`` pair cannot:

``percentUsed``
    May exceed 100. It is not clamped: a workspace that is 140% of its budget
    needs to be able to say so, and clamping to 100 makes "at the limit" and
    "double the limit" indistinguishable. The UI decides how to render it.

``burnRate``
    Tokens per hour over the period so far. Useful on its own: 2M/day looks
    alarming until you learn the workspace spends 3k/h.

``projectedTotal``
    The same rate extrapolated to the end of the period. A linear extrapolation,
    stated as such -- real usage is spiky, and a number presented without that
    caveat would be a lie dressed as a forecast.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import ApiModel


class UsageBreakdown(ApiModel):
    """One row of ``byModel`` / ``byPersona``."""

    key: str
    tokens: int = 0
    #: Share of the period total, 0..1 (or >1 in a degenerate single-group case).
    share: float = 0.0


class UsageResponse(ApiModel):
    """``GET /usage?period=today|month``."""

    workspace_id: str
    business_id: str
    period: str
    tokens_in: int = 0
    tokens_out: int = 0
    total_tokens: int = 0
    #: The period total (30x daily for ``month``), resolved per workspace.
    budget_tokens: int = 0
    remaining_tokens: int = 0
    #: Unclamped. Can exceed 100 when the workspace is over budget.
    percent_used: float = 0.0
    #: ``true`` when ``percent_used`` >= 100.
    exceeded: bool = False
    #: When the period resets (UTC).
    reset_at: datetime
    period_start: datetime
    calls: int = 0
    #: Tokens per hour so far, for the "am I burning money?" question.
    burn_rate_per_hour: float = 0.0
    #: Straight-line projection to the end of the period.
    projected_total_tokens: int = 0
    #: Where ``budget_tokens`` came from: ``workspace`` or ``default``.
    budget_source: str = "default"
    #: Rows whose token counts were estimated (provider reported no usage).
    estimated_calls: int = 0
    #: ``tokens_in``/``tokens_out`` priced at the configured per-million rates.
    #: ``null`` when no rate is configured -- never a fabricated number.
    estimated_cost_usd: float | None = None
    cost_per_million_in_usd: float = 0.0
    cost_per_million_out_usd: float = 0.0
    by_model: list[UsageBreakdown] = Field(default_factory=list)
    by_persona: list[UsageBreakdown] = Field(default_factory=list)
    #: Per-action spend: which part of the product is consuming the budget.
    by_action: list[UsageBreakdown] = Field(default_factory=list)


__all__ = ["UsageBreakdown", "UsageResponse"]