"""Read-only, workspace-scoped Meta campaign performance service."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.security import TokenCipher
from app.core.tools.settings import load_connector_settings
from app.db.models import Campaign, Integration
from app.schemas.common import Evidence
from app.schemas.insights import DailyMetric, InsightsPayload, MetricSeries

DEFAULT_METRICS = (
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
)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        # List-shaped action metrics require an explicit action-type choice;
        # do not sum unlike conversion types into a misleading total.
        return None


async def _meta_credentials(
    db: AsyncSession, *, business_id: str, cfg: Settings
) -> tuple[str | None, str]:
    integration = await db.scalar(
        select(Integration).where(
            Integration.business_id == business_id,
            Integration.provider == "meta",
            Integration.status == "connected",
        )
    )
    connectors = await load_connector_settings(db, business_id, cfg)
    if integration and integration.access_token_enc:
        try:
            token = TokenCipher(cfg.token_encryption_key).decrypt(integration.access_token_enc)
            if token:
                return token, connectors.meta_graph_base_url or cfg.meta_graph_base_url
        except Exception:
            # Fail closed: never surface ciphertext or key details.
            pass
    token = connectors.meta_access_token or cfg.meta_access_token
    base_url = connectors.meta_graph_base_url or cfg.meta_graph_base_url
    return token, base_url


async def read_campaign_performance(
    db: AsyncSession,
    *,
    business_id: str,
    campaign_id: str,
    date_start: date | None = None,
    date_end: date | None = None,
    settings: Settings | None = None,
) -> InsightsPayload:
    """Return real daily Meta rows or an explicit unavailable/partial result."""
    cfg = settings or get_settings()
    campaign = await db.scalar(
        select(Campaign).where(Campaign.id == campaign_id, Campaign.business_id == business_id)
    )
    if campaign is None:
        from app.core.errors import NotFoundError

        raise NotFoundError(f"campaign '{campaign_id}' not found")
    if not campaign.meta_campaign_id:
        return InsightsPayload.unavailable(
            source="meta_marketing_api",
            reason="This local draft is not linked to a Meta campaign, so real performance is unavailable.",
        ).model_copy(update={"campaign_id": campaign_id})

    end = date_end or date.today()
    start = date_start or (end - timedelta(days=29))
    if end < start:
        return InsightsPayload.unavailable(
            source="meta_marketing_api", reason="The requested date range is invalid."
        ).model_copy(update={"campaign_id": campaign_id, "meta_campaign_id": campaign.meta_campaign_id})

    token, base_url = await _meta_credentials(db, business_id=business_id, cfg=cfg)
    if not token:
        return InsightsPayload.unavailable(
            source="meta_marketing_api",
            reason="Meta is not connected for this workspace; no campaign performance numbers are available.",
        ).model_copy(update={"campaign_id": campaign_id, "meta_campaign_id": campaign.meta_campaign_id})

    params = {
        "access_token": token,
        "time_range": json.dumps({"since": start.isoformat(), "until": end.isoformat()}),
        "fields": ",".join(DEFAULT_METRICS),
        "level": "campaign",
        "time_increment": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=cfg.meta_api_timeout_seconds) as client:
            response = await client.get(
                f"{base_url.rstrip('/')}/{cfg.meta_graph_version}/{campaign.meta_campaign_id}/insights",
                params=params,
            )
        if response.is_error:
            return InsightsPayload.unavailable(
                source="meta_marketing_api",
                reason=f"Meta insights could not be read (HTTP {response.status_code}); no numbers were substituted.",
            ).model_copy(update={"campaign_id": campaign_id, "meta_campaign_id": campaign.meta_campaign_id})
        payload = response.json()
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            rows = []
    except (httpx.HTTPError, ValueError, TypeError):
        return InsightsPayload.unavailable(
            source="meta_marketing_api",
            reason="Meta insights request failed; no numbers were substituted.",
        ).model_copy(update={"campaign_id": campaign_id, "meta_campaign_id": campaign.meta_campaign_id})

    if not rows:
        return InsightsPayload.partial(
            source="meta_marketing_api",
            reason="Meta returned no insight rows for the requested date range.",
        ).model_copy(
            update={
                "campaign_id": campaign_id,
                "meta_campaign_id": campaign.meta_campaign_id,
                "date_start": start,
                "date_end": end,
            }
        )

    daily: list[DailyMetric] = []
    numeric_totals: dict[str, float] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        row_date = item.get("date_start") or item.get("date_stop")
        if not isinstance(row_date, str):
            continue
        try:
            parsed_date = date.fromisoformat(row_date[:10])
        except ValueError:
            continue
        metrics = {
            key: _numeric(item.get(key))
            for key in ("impressions", "clicks", "conversions", "spend")
        }
        daily.append(DailyMetric(date=parsed_date, **metrics))
        for key in ("spend", "impressions", "clicks", "reach", "conversions"):
            number = _numeric(item.get(key))
            if number is not None:
                numeric_totals[key] = numeric_totals.get(key, 0.0) + number

    if not daily:
        return InsightsPayload.partial(
            source="meta_marketing_api",
            reason="Meta returned rows without usable daily dates; totals are not displayed.",
        ).model_copy(
            update={"campaign_id": campaign_id, "meta_campaign_id": campaign.meta_campaign_id, "date_start": start, "date_end": end}
        )

    now = datetime.now(timezone.utc)
    return InsightsPayload(
        campaign_id=campaign_id,
        meta_campaign_id=campaign.meta_campaign_id,
        date_start=start,
        date_end=end,
        status="ok",
        source="meta_marketing_api",
        metrics=numeric_totals,
        daily=daily,
        series=[MetricSeries(label=key, value=value) for key, value in numeric_totals.items()],
        retrieved_at=now,
        evidence=[Evidence(source="meta_marketing_api", detail=f"Meta campaign {campaign.meta_campaign_id}; {start} to {end}", retrieved_at=now)],
    )


__all__ = ["DEFAULT_METRICS", "read_campaign_performance"]
