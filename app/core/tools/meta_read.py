"""meta_read tools (9-10). READ-ONLY against the Meta Marketing API.

These use the stored Meta token and never write. When the token is missing, the
campaign is not linked to a Meta campaign, or the API errors, they return
``status="unavailable"`` with a reason. They never return fabricated numbers.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import httpx

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.core.security import TokenCipher
from app.core.tools.base import Tool, ToolContext
from app.db.models import Campaign, Integration
from app.schemas.common import Evidence, ToolSideEffect
from app.schemas.insights import BreakdownPayload, InsightsPayload, MetricSeries
from app.schemas.tools import GetBreakdownsInput, GetCampaignInsightsInput

_logger = get_logger(__name__)

#: Default metric set. Kept small and explicit.
DEFAULT_METRICS = [
    "spend",
    "impressions",
    "clicks",
    "ctr",
    "cpc",
    "cpm",
    "reach",
    "frequency",
    "conversions",
    "cost_per_conversion",
]

#: Breakdowns the API supports, mapped to the API's own parameter values.
_BREAKDOWNS = {
    "ad": "ad_id",
    "placement": "publisher_platform,platform_position",
    "audience": "age,gender",
    "age": "age",
    "gender": "gender",
}


async def _resolve_meta_campaign_id(
    ctx: ToolContext, campaign_id: str, override: str | None
) -> tuple[str | None, str | None]:
    """Return (meta_campaign_id, error_reason)."""
    if override:
        return override, None
    campaign = await ctx.db.get(Campaign, campaign_id)
    if campaign is None:
        return None, f"Campaign '{campaign_id}' was not found."
    if not campaign.meta_campaign_id:
        return None, (
            f"Campaign '{campaign.name}' is not linked to a Meta campaign. Link it in "
            "the campaign settings to read real performance data."
        )
    return campaign.meta_campaign_id, None


async def _get_meta_token(ctx: ToolContext) -> str | None:
    """Read and decrypt the stored Meta token for this business."""
    from sqlalchemy import select

    stmt = select(Integration).where(
        Integration.business_id == ctx.business_id,
        Integration.provider == "meta",
        Integration.status == "connected",
    )
    integration = (await ctx.db.execute(stmt)).scalars().first()
    if integration is None or not integration.access_token_enc:
        return None
    cfg = ctx.require_settings()
    cipher = TokenCipher(cfg.token_encryption_key)
    try:
        return cipher.decrypt(integration.access_token_enc)
    except Exception as exc:  # pragma: no cover - corrupt/rotated key
        _logger.warning("meta_token_decrypt_failed", extra={"error": str(exc)})
        return None


def _default_range(payload: Any) -> tuple[date, date]:
    end = payload.date_end or date.today()
    start = payload.date_start or (end - timedelta(days=29))
    return start, end


async def _handle_get_campaign_insights(
    payload: GetCampaignInsightsInput, ctx: ToolContext
) -> InsightsPayload:
    cfg = ctx.require_settings()
    source = "meta_marketing_api"
    start, end = _default_range(payload)

    meta_campaign_id, error = await _resolve_meta_campaign_id(
        ctx, payload.campaign_id, payload.meta_campaign_id
    )
    if error:
        return InsightsPayload.unavailable(source=source, reason=error).model_copy(
            update={
                "campaign_id": payload.campaign_id,
                "meta_campaign_id": meta_campaign_id,
            }
        )

    token = await _get_meta_token(ctx)
    if not token:
        return InsightsPayload.unavailable(
            source=source,
            reason=(
                "Meta is not connected for this workspace, so campaign performance "
                "cannot be read. Connect Meta in Integrations, or export the numbers "
                "and paste them in."
            ),
        ).model_copy(
            update={
                "campaign_id": payload.campaign_id,
                "meta_campaign_id": meta_campaign_id,
            }
        )

    metrics = payload.metrics or DEFAULT_METRICS
    try:
        async with httpx.AsyncClient(timeout=cfg.meta_api_timeout_seconds) as client:
            resp = await client.get(
                f"{cfg.meta_graph_base_url}/{cfg.meta_graph_version}/{meta_campaign_id}/insights",
                params={
                    "access_token": token,
                    "time_range": json_dumps({"since": start.isoformat(), "until": end.isoformat()}),
                    "fields": ",".join(metrics),
                    "level": "campaign",
                },
            )
            resp.raise_for_status()
            rows = resp.json().get("data", [])
    except httpx.HTTPError as exc:
        return InsightsPayload.unavailable(
            source=source, reason=f"Meta insights request failed: {exc}"
        )

    if not rows:
        return InsightsPayload.partial(
            source=source,
            reason="Meta returned no insight rows for the requested date range.",
        ).model_copy(
            update={
                "campaign_id": payload.campaign_id,
                "meta_campaign_id": meta_campaign_id,
                "date_start": start,
                "date_end": end,
            }
        )

    raw = rows[0]
    metrics_out = {
        k: float(v) for k, v in raw.items() if k not in ("date_start", "date_stop") and _is_num(v)
    }
    return InsightsPayload(
        campaign_id=payload.campaign_id,
        meta_campaign_id=meta_campaign_id,
        date_start=start,
        date_end=end,
        status="ok",
        source=source,
        metrics=metrics_out,
        series=[
            MetricSeries(
                label=f"{k} ({start.isoformat()} to {end.isoformat()})", value=v
            )
            for k, v in metrics_out.items()
        ],
        retrieved_at=_now(),
        evidence=[
            Evidence(
                source=source,
                detail=f"Meta campaign {meta_campaign_id}, {start} to {end}",
                retrieved_at=_now(),
            )
        ],
    )


async def _handle_get_breakdowns(
    payload: GetBreakdownsInput, ctx: ToolContext
) -> BreakdownPayload:
    cfg = ctx.require_settings()
    source = "meta_marketing_api"
    start, end = _default_range(payload)

    if payload.by not in _BREAKDOWNS:
        return BreakdownPayload.unavailable(
            source=source,
            reason=(
                f"Breakdown '{payload.by}' is not supported. Use one of: "
                f"{', '.join(sorted(_BREAKDOWNS))}."
            ),
        )

    meta_campaign_id, error = await _resolve_meta_campaign_id(
        ctx, payload.campaign_id, payload.meta_campaign_id
    )
    if error:
        return BreakdownPayload.unavailable(source=source, reason=error)

    token = await _get_meta_token(ctx)
    if not token:
        return BreakdownPayload.unavailable(
            source=source,
            reason=(
                "Meta is not connected for this workspace, so breakdowns cannot be "
                "read. Connect Meta in Integrations, or paste the numbers you want "
                "analysed."
            ),
        )

    try:
        async with httpx.AsyncClient(timeout=cfg.meta_api_timeout_seconds) as client:
            resp = await client.get(
                f"{cfg.meta_graph_base_url}/{cfg.meta_graph_version}/{meta_campaign_id}/insights",
                params={
                    "access_token": token,
                    "time_range": json_dumps({"since": start.isoformat(), "until": end.isoformat()}),
                    "fields": join_metrics(DEFAULT_METRICS),
                    "breakdowns": _BREAKDOWNS[payload.by],
                    "level": "campaign",
                },
            )
            resp.raise_for_status()
            rows = resp.json().get("data", [])
    except httpx.HTTPError as exc:
        return BreakdownPayload.unavailable(
            source=source, reason=f"Meta breakdown request failed: {exc}"
        )

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        entry = {k: v for k, v in row.items() if k not in ("date_start", "date_stop")}
        out_rows.append(entry)

    return BreakdownPayload(
        campaign_id=payload.campaign_id,
        by=payload.by,
        date_start=start,
        date_end=end,
        status="ok",
        source=source,
        rows=out_rows,
        retrieved_at=_now(),
    )


def _is_num(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value)


def join_metrics(metrics: list[str]) -> str:
    return ",".join(metrics)


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


get_campaign_insights_tool = Tool(
    name="get_campaign_insights",
    description=(
        "Read performance metrics (spend, impressions, clicks, CTR, CPC, conversions) "
        "for a campaign from Meta over a date range. Use it whenever the user asks how "
        "a campaign is performing. It never invents numbers: if Meta is not connected "
        "or the campaign is not linked it returns status='unavailable' and you must "
        "tell the user and ask them to paste the numbers."
    ),
    input_model=GetCampaignInsightsInput,
    output_model=InsightsPayload,
    side_effect=ToolSideEffect.NONE,
    handler=_handle_get_campaign_insights,
)

get_breakdowns_tool = Tool(
    name="get_breakdowns",
    description=(
        "Break a campaign's performance down by ad, placement, audience, age or "
        "gender. Use it to find which segment is underperforming or fatiguing. Same "
        "unavailability rules as get_campaign_insights."
    ),
    input_model=GetBreakdownsInput,
    output_model=BreakdownPayload,
    side_effect=ToolSideEffect.NONE,
    handler=_handle_get_breakdowns,
)

META_READ_TOOLS: list[Tool] = [get_campaign_insights_tool, get_breakdowns_tool]

__all__ = [
    "DEFAULT_METRICS",
    "META_READ_TOOLS",
    "get_breakdowns_tool",
    "get_campaign_insights_tool",
]
