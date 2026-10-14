"""meta_read tools (9-10). READ-ONLY against the Meta Marketing API.

These use the stored Meta token and never write. When the token is missing, the
campaign is not linked to a Meta campaign, or the API errors, they return
``status="unavailable"`` with a reason. They never return fabricated numbers.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.logging import get_logger
from app.core.security import TokenCipher
from app.core.tools.base import Tool, ToolContext
from app.db.models import Campaign, Integration
from app.schemas.common import Evidence, ToolSideEffect, ToolStatus
from app.schemas.insights import (
    BreakdownPayload,
    DailyMetrics,
    InsightsPayload,
    MetricSeries,
)
from app.schemas.tools import (
    GetBreakdownsInput,
    GetCampaignInsightsInput,
    ListMetaCampaignsInput,
    ListMetaCampaignsOutput,
    MetaCampaignSummary,
)

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
_META_CAMPAIGN_ID_RE = re.compile(r"^[0-9]{1,64}$")


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
    # Remember the name on the context: `get_campaign_insights` reports back what
    # it read, and "which campaign?" is the first thing a user asks about numbers.
    ctx.user_supplied.setdefault("campaign_name", str(campaign.name or ""))
    return campaign.meta_campaign_id, None


def _campaign_label(ctx: ToolContext) -> str | None:
    """The Sahm campaign name resolved for this call, when there is one.

    ``_resolve_meta_campaign_id`` stores it on the context so every return path --
    ok, partial, unavailable -- can say which campaign it is talking about.
    """
    supplied = getattr(ctx, "user_supplied", None) or {}
    if not isinstance(supplied, dict):
        return None
    name = str(supplied.get("campaign_name") or "").strip()
    return name or None


async def _get_meta_token(ctx: ToolContext) -> str | None:
    """The Meta token for this business, or None.

    Two sources, in this order:

    1. A connected ``Integration`` row -- an explicit OAuth grant the user made in
       Integrations, which is always the most deliberate signal.
    2. The ``tool_settings`` Meta row (P2.4) -- configuration typed into the
       admin panel, decrypted through the single resolver in
       ``app.core.tools.settings``.

    Falling through to (2) rather than to ``Settings.meta_access_token`` is the
    point of P2.5: with no env var set at all, a workspace that saved a token in
    the UI gets real insights.
    """
    from sqlalchemy import select

    from app.core.tools.settings import resolve_tool_settings

    stmt = select(Integration).where(
        Integration.business_id == ctx.business_id,
        Integration.provider == "meta",
        Integration.status == "connected",
    )
    integration = (await ctx.db.execute(stmt)).scalars().first()
    cfg = ctx.require_settings()
    if integration is not None and integration.access_token_enc:
        cipher = TokenCipher(cfg.token_encryption_key)
        try:
            return cipher.decrypt(integration.access_token_enc)
        except Exception as exc:  # pragma: no cover - corrupt/rotated key
            _logger.warning("meta_token_decrypt_failed", extra={"error": str(exc)})
            return None
    tools_cfg = await resolve_tool_settings(ctx)
    # The resolver already merges env; the explicit fallback is for the case
    # where a caller passed no workspace at all (scripts, exercise_tools.py).
    return tools_cfg.meta_access_token or cfg.meta_access_token


async def _meta_config(ctx: ToolContext, cfg: Any) -> tuple[str, str]:
    """(graph base url, access token) with the workspace row preferred.

    Resolved once per call and cached on the context, so the insights handler and
    the token lookup cannot disagree about which configuration is in force.
    """
    from app.core.tools.settings import resolve_tool_settings

    tools_cfg = await resolve_tool_settings(ctx)
    return (
        tools_cfg.meta_graph_base_url or cfg.meta_graph_base_url,
        tools_cfg.meta_access_token or cfg.meta_access_token or "",
    )


#: Metrics that are safe to add across days.
_ADDITIVE_METRICS = frozenset({"spend", "impressions", "clicks", "conversions", "reach"})


def _parse_day(value: object) -> date | None:
    """Parse Meta's ``date_start`` (``YYYY-MM-DD``) without raising."""
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _derive_ratio_metrics(totals: dict[str, float]) -> dict[str, float]:
    """Recompute ratio metrics from the summed additive base.

    Summing ctr/cpc/cpm/cost_per_conversion across days would be wrong, so they
    are derived from the totals instead. ``frequency`` is an approximation: it
    uses the sum of daily reach, which double-counts users seen on several days.
    """
    spend = totals.get("spend", 0.0)
    impressions = totals.get("impressions", 0.0)
    clicks = totals.get("clicks", 0.0)
    conversions = totals.get("conversions", 0.0)
    reach = totals.get("reach", 0.0)
    derived: dict[str, float] = {}
    if impressions:
        derived["ctr"] = round(clicks / impressions * 100, 4)
        derived["cpm"] = round(spend / impressions * 1000, 4)
    if clicks:
        derived["cpc"] = round(spend / clicks, 4)
    if conversions:
        derived["cost_per_conversion"] = round(spend / conversions, 4)
    if reach:
        derived["frequency"] = round(impressions / reach, 4)
    return derived


def _default_range(payload: Any) -> tuple[date, date]:
    end = payload.date_end or date.today()
    start = payload.date_start or (end - timedelta(days=29))
    return start, end


async def list_meta_campaigns_for_workspace(
    db: AsyncSession,
    business_id: str,
    settings: Settings,
    payload: ListMetaCampaignsInput,
) -> ListMetaCampaignsOutput:
    """Discover campaigns from the connected Meta account without importing or writing them."""
    from app.ai.marketing.meta_ads import (
        HttpMetaGraphClient,
        MetaGraphError,
        resolve_meta_connection,
    )

    cfg = settings
    source = "meta_marketing_api"
    try:
        connection = await resolve_meta_connection(db, business_id, cfg)
    except Exception as exc:
        _logger.warning(
            "meta_campaign_connection_resolution_failed",
            extra={"error_type": type(exc).__name__},
        )
        return ListMetaCampaignsOutput.unavailable(
            source=source,
            reason="Could not resolve this workspace's Meta connection. Check the connection settings.",
        )

    if not connection.access_token:
        return ListMetaCampaignsOutput.unavailable(
            source=source,
            reason="Meta is not connected for this workspace, so existing ad-account campaigns cannot be listed.",
        )
    if not connection.ad_account_id:
        return ListMetaCampaignsOutput.unavailable(
            source=source,
            reason="No Meta ad account is configured for this workspace.",
        )
    if not _META_CAMPAIGN_ID_RE.fullmatch(connection.ad_account_id):
        return ListMetaCampaignsOutput.unavailable(
            source=source,
            reason="The configured Meta ad account ID is invalid; reconnect or update the account settings.",
        )

    params: dict[str, Any] = {
        "fields": "id,name,objective,status,effective_status,created_time,start_time,stop_time",
        "limit": payload.limit,
    }
    if payload.after:
        params["after"] = payload.after

    graph = HttpMetaGraphClient(
        settings=cfg,
        access_token=connection.access_token,
        base_url=connection.graph_base_url,
    )
    try:
        response = await graph.get(f"act_{connection.ad_account_id}/campaigns", params)
    except MetaGraphError as exc:
        # Meta errors are sanitized at the HTTP boundary; redact the exact grant
        # once more here as defense in depth (including short test/development tokens).
        detail = str(exc)
        if connection.access_token:
            detail = detail.replace(connection.access_token, "[redacted]")
        return ListMetaCampaignsOutput(
            status=ToolStatus.UNAVAILABLE,
            source=source,
            unavailable_reason=f"Meta campaigns request failed: {detail}",
            account_id=connection.ad_account_id,
            retrieved_at=_now(),
        )
    except Exception as exc:
        _logger.warning(
            "meta_campaign_list_failed",
            extra={"error_type": type(exc).__name__},
        )
        return ListMetaCampaignsOutput.unavailable(
            source=source,
            reason="Meta campaigns could not be read. Check the connection and ads_read permission.",
        ).model_copy(
            update={"account_id": connection.ad_account_id, "retrieved_at": _now()}
        )

    raw_rows = response.get("data", [])
    rows = raw_rows if isinstance(raw_rows, list) else []
    query = (payload.search or "").strip().casefold()
    campaigns: list[MetaCampaignSummary] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        meta_id = row.get("id")
        if not isinstance(meta_id, str) or not _META_CAMPAIGN_ID_RE.fullmatch(meta_id.strip()):
            continue
        name = row.get("name")
        safe_name = name.strip() if isinstance(name, str) else ""
        if query and query not in f"{meta_id} {safe_name}".casefold():
            continue

        def text_or_none(key: str) -> str | None:
            value = row.get(key)
            return value.strip() if isinstance(value, str) and value.strip() else None

        campaigns.append(
            MetaCampaignSummary(
                meta_campaign_id=meta_id.strip(),
                name=safe_name,
                objective=text_or_none("objective"),
                status=text_or_none("status"),
                effective_status=text_or_none("effective_status"),
                created_time=text_or_none("created_time"),
                start_time=text_or_none("start_time"),
                stop_time=text_or_none("stop_time"),
            )
        )

    paging = response.get("paging")
    paging = paging if isinstance(paging, dict) else {}
    cursors = paging.get("cursors")
    cursors = cursors if isinstance(cursors, dict) else {}
    after = cursors.get("after")
    next_cursor = after if isinstance(after, str) and after else None
    # Never return Meta's `paging.next` URL: it may contain the access token.
    has_more = bool(next_cursor and paging.get("next"))
    return ListMetaCampaignsOutput(
        status=ToolStatus.PARTIAL if has_more else ToolStatus.OK,
        source=source,
        unavailable_reason=(
            "More campaigns are available; continue with next_cursor to search older campaigns."
            if has_more
            else None
        ),
        account_id=connection.ad_account_id,
        campaigns=campaigns,
        has_more=has_more,
        next_cursor=next_cursor,
        retrieved_at=_now(),
    )


async def _handle_list_meta_campaigns(
    payload: ListMetaCampaignsInput, ctx: ToolContext
) -> ListMetaCampaignsOutput:
    """Agent-tool adapter for shared, read-only Meta campaign discovery."""
    return await list_meta_campaigns_for_workspace(
        ctx.db, ctx.business_id, ctx.require_settings(), payload
    )


async def _handle_get_campaign_insights(
    payload: GetCampaignInsightsInput, ctx: ToolContext
) -> InsightsPayload:
    cfg = ctx.require_settings()
    meta_base, _fallback_token = await _meta_config(ctx, cfg)
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

    if not meta_campaign_id or not _META_CAMPAIGN_ID_RE.fullmatch(meta_campaign_id):
        return InsightsPayload.unavailable(
            source=source,
            reason="The linked Meta campaign ID is invalid; reconnect or relink the campaign before reading insights.",
        ).model_copy(
            update={
                "campaign_id": payload.campaign_id,
                "meta_campaign_id": meta_campaign_id,
                "date_start": start,
                "date_end": end,
            }
        )

    from app.ai.marketing.meta_ads import HttpMetaGraphClient, MetaGraphError

    metrics = payload.metrics or DEFAULT_METRICS
    graph = HttpMetaGraphClient(settings=cfg, access_token=token, base_url=meta_base)
    try:
        response = await graph.get(
            f"{meta_campaign_id}/insights",
            {
                "time_range": json_dumps({"since": start.isoformat(), "until": end.isoformat()}),
                "fields": ",".join(metrics),
                "level": "campaign",
                # One row per day so the UI can draw a trend, not one total.
                "time_increment": "1",
            },
        )
        raw_rows = response.get("data", [])
        rows = raw_rows if isinstance(raw_rows, list) else []
    except MetaGraphError as exc:
        detail = str(exc).replace(token, "[redacted]")
        return InsightsPayload.unavailable(
            source=source, reason=f"Meta insights request failed: {detail}"
        ).model_copy(
            update={
                "campaign_id": payload.campaign_id,
                "meta_campaign_id": meta_campaign_id,
                "date_start": start,
                "date_end": end,
            }
        )
    except Exception as exc:
        _logger.warning(
            "meta_campaign_insights_failed",
            extra={"error_type": type(exc).__name__},
        )
        return InsightsPayload.unavailable(
            source=source,
            reason="Meta insights could not be read. Check the connection and ads_read permission.",
        ).model_copy(
            update={
                "campaign_id": payload.campaign_id,
                "campaign_name": _campaign_label(ctx),
                "meta_campaign_id": meta_campaign_id,
                "date_start": start,
                "date_end": end,
            }
        )

    if not rows:
        return InsightsPayload.partial(
            source=source,
            reason="Meta returned no insight rows for the requested date range.",
        ).model_copy(
            update={
                "campaign_id": payload.campaign_id,
                "campaign_name": _campaign_label(ctx),
                "meta_campaign_id": meta_campaign_id,
                "date_start": start,
                "date_end": end,
            }
        )

    daily: list[DailyMetrics] = []
    totals: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        values = {
            k: float(v)
            for k, v in row.items()
            if k not in ("date_start", "date_stop") and _is_num(v)
        }
        for key, value in values.items():
            if key in _ADDITIVE_METRICS:
                totals[key] = totals.get(key, 0.0) + value
        daily.append(
            DailyMetrics(
                date=_parse_day(row.get("date_start")) or start,
                impressions=values.get("impressions", 0.0),
                clicks=values.get("clicks", 0.0),
                conversions=values.get("conversions", 0.0),
                spend=values.get("spend", 0.0),
            )
        )
    daily.sort(key=lambda item: item.date)

    # Ratios are derived from the totals, never summed across days.
    metrics_out = {**totals, **_derive_ratio_metrics(totals)}
    campaign_name = _campaign_label(ctx)
    return InsightsPayload(
        campaign_id=payload.campaign_id,
        campaign_name=campaign_name,
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
        daily=daily,
        retrieved_at=_now(),
        evidence=[
            Evidence(
                source=source,
                detail=(
                    f"Meta campaign {meta_campaign_id}"
                    + (f" (Sahm campaign '{campaign_name}')" if campaign_name else "")
                    + f", {start} to {end}: "
                    f"{len(daily)} daily row(s). Ratio metrics are recomputed "
                    "from summed spend/impressions/clicks/conversions; "
                    "frequency uses the sum of daily reach and is approximate."
                ),
                retrieved_at=_now(),
            )
        ],
    )


async def _handle_get_breakdowns(
    payload: GetBreakdownsInput, ctx: ToolContext
) -> BreakdownPayload:
    cfg = ctx.require_settings()
    meta_base, _fallback_token = await _meta_config(ctx, cfg)
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

    if not meta_campaign_id or not _META_CAMPAIGN_ID_RE.fullmatch(meta_campaign_id):
        return BreakdownPayload.unavailable(
            source=source,
            reason="The linked Meta campaign ID is invalid; reconnect or relink the campaign before reading breakdowns.",
        )

    from app.ai.marketing.meta_ads import HttpMetaGraphClient, MetaGraphError

    graph = HttpMetaGraphClient(settings=cfg, access_token=token, base_url=meta_base)
    try:
        response = await graph.get(
            f"{meta_campaign_id}/insights",
            {
                "time_range": json_dumps({"since": start.isoformat(), "until": end.isoformat()}),
                "fields": join_metrics(DEFAULT_METRICS),
                "breakdowns": _BREAKDOWNS[payload.by],
                "level": "campaign",
            },
        )
        raw_rows = response.get("data", [])
        rows = raw_rows if isinstance(raw_rows, list) else []
    except MetaGraphError as exc:
        detail = str(exc).replace(token, "[redacted]")
        return BreakdownPayload.unavailable(
            source=source, reason=f"Meta breakdown request failed: {detail}"
        )
    except Exception as exc:
        _logger.warning(
            "meta_campaign_breakdown_failed",
            extra={"error_type": type(exc).__name__},
        )
        return BreakdownPayload.unavailable(
            source=source,
            reason="Meta breakdowns could not be read. Check the connection and ads_read permission.",
        )

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
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
        "a campaign is performing. For campaigns outside Sahm, first call list_meta_campaigns "
        "and pass the returned Meta ID in both campaign_id and meta_campaign_id; no local "
        "campaign row is required. It never invents numbers: if Meta is not connected or "
        "the campaign cannot be resolved, it returns status='unavailable'."
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
        "unavailability rules as get_campaign_insights. For an external campaign "
        "returned by list_meta_campaigns, pass its exact Meta ID in meta_campaign_id."
    ),
    input_model=GetBreakdownsInput,
    output_model=BreakdownPayload,
    side_effect=ToolSideEffect.NONE,
    handler=_handle_get_breakdowns,
)

list_meta_campaigns_tool = Tool(
    name="list_meta_campaigns",
    description=(
        "Read-only discovery of campaigns already present in this workspace's connected Meta ad account, "
        "including campaigns not created in Sahm and paused/archived campaigns returned by Meta. "
        "Use search to match a campaign name or ID; results expose the exact identifier as meta_campaign_id. "
        "If multiple names match, ask the user to choose. Continue with next_cursor when has_more is true. "
        "This does not import, create, edit, or publish anything."
    ),
    input_model=ListMetaCampaignsInput,
    output_model=ListMetaCampaignsOutput,
    side_effect=ToolSideEffect.NONE,
    handler=_handle_list_meta_campaigns,
)

META_READ_TOOLS: list[Tool] = [
    list_meta_campaigns_tool,
    get_campaign_insights_tool,
    get_breakdowns_tool,
]

__all__ = [
    "DEFAULT_METRICS",
    "META_READ_TOOLS",
    "get_breakdowns_tool",
    "get_campaign_insights_tool",
    "list_meta_campaigns_tool",
]
