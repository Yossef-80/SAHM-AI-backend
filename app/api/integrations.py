"""Integration endpoints: Meta status and connect."""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    AuthDep,
    RequestIdDep,
    SessionDep,
    SettingsDep,
    WorkspaceDep,
    require_role,
)
from app.ai.marketing.meta_ads import (
    HttpMetaGraphClient,
    MetaGraphError,
    resolve_meta_connection,
)
from app.core.logging import get_logger
from app.core.tools.meta_read import list_meta_campaigns_for_workspace
from app.core.security import TokenCipher
from app.core.tools.settings import load_connector_settings
from app.db.models import Integration
from app.schemas.targeting import TargetingCategory
from app.schemas.tools import ListMetaCampaignsInput
from app.schemas.integrations import (
    MetaConnectRequest,
    MetaConnectResponse,
    MetaInterestSuggestion,
    MetaIntegrationStatus,
    MetaTargetingSearchResponse,
    MetaAccountCampaign,
    MetaAccountCampaignsResponse,
)

router = APIRouter(prefix="/integrations", tags=["integrations"])

_logger = get_logger(__name__)

# Categories that Meta serves as a browse list rather than a text autocomplete.
# We cache these short-lived account-level catalogues, then filter in Python by
# the user's phrase; all values remain selected from Meta's own response.
_META_TARGETING_CLASS: dict[str, str] = {
    "behaviors": "behaviors",
    "life_events": "life_events",
    "industries": "industries",
    "income": "income",
    "family_statuses": "family_statuses",
    "devices": "user_device",
    "operating_systems": "user_os",
}
_META_TARGETING_AUTOCOMPLETE: dict[str, str] = {
    "interests": "adinterest",
    "job_titles": "adworkposition",
    "employers": "adworkemployer",
    "education_majors": "adeducationmajor",
    "education_schools": "adeducationschool",
}
_TARGETING_CATALOG_CACHE: dict[
    tuple[str, str, str], tuple[float, list[dict[str, Any]], bool]
] = {}
_TARGETING_CATALOG_TTL_SECONDS = 600
_TARGETING_CATALOG_PAGE_LIMIT = 500
_TARGETING_CATALOG_MAX_PAGES = 8


async def _load_targeting_catalog(
    graph: HttpMetaGraphClient,
    *,
    business_id: str,
    account_id: str,
    category: str,
) -> tuple[list[dict[str, Any]], bool]:
    cache_key = (business_id, account_id, category)
    now = time.monotonic()
    cached = _TARGETING_CATALOG_CACHE.get(cache_key)
    if cached and cached[0] > now:
        return cached[1], cached[2]

    meta_class = _META_TARGETING_CLASS[category]
    rows: list[dict[str, Any]] = []
    after: str | None = None
    has_more = False
    for _ in range(_TARGETING_CATALOG_MAX_PAGES):
        params: dict[str, Any] = {
            "type": "adTargetingCategory",
            "class": meta_class,
            "limit": _TARGETING_CATALOG_PAGE_LIMIT,
        }
        if after:
            params["after"] = after
        response = await graph.get("search", params)
        page_rows = response.get("data", [])
        if isinstance(page_rows, list):
            rows.extend(row for row in page_rows if isinstance(row, dict))

        paging = response.get("paging")
        paging = paging if isinstance(paging, dict) else {}
        cursors = paging.get("cursors")
        cursors = cursors if isinstance(cursors, dict) else {}
        cursor = cursors.get("after")
        has_more = bool(cursor and paging.get("next"))
        if not has_more or not isinstance(cursor, str):
            has_more = False
            break
        after = cursor

    _TARGETING_CATALOG_CACHE[cache_key] = (
        time.monotonic() + _TARGETING_CATALOG_TTL_SECONDS,
        rows,
        has_more,
    )
    return rows, has_more


async def _meta_integration(
    db: AsyncSession, business_id: str
) -> Integration | None:
    """The Meta grant for one business. Always scoped to a resolved workspace."""
    stmt = select(Integration).where(
        Integration.business_id == business_id, Integration.provider == "meta"
    )
    return (await db.execute(stmt)).scalars().first()


@router.get("/meta/targeting-search", response_model=MetaTargetingSearchResponse)
async def meta_targeting_search(
    ws: WorkspaceDep,
    db: SessionDep,
    settings: SettingsDep,
    q: str = Query(min_length=1, max_length=120),
    category: TargetingCategory = Query(default="interests"),
) -> MetaTargetingSearchResponse:
    """Resolve a phrase in the selected Meta detailed-targeting category.

    Searchable categories use Meta's type-ahead endpoints. Behavior and other
    catalog categories are returned by Meta as browse lists, so those are cached
    briefly and filtered locally. No free-text phrase is ever applied directly
    to campaign targeting.
    """
    parts = [part.strip() for part in re.split(r"[,،;؛\r\n]+", q) if part.strip()]
    query = parts[-1] if parts else ""
    if not query:
        return MetaTargetingSearchResponse(available=False, unavailable_reason="empty_query")

    conn = await resolve_meta_connection(db, ws.business_id, settings)
    if not conn.access_token:
        return MetaTargetingSearchResponse(
            available=False, unavailable_reason="meta_not_connected"
        )

    graph = HttpMetaGraphClient(
        settings=settings,
        access_token=conn.access_token,
        base_url=conn.graph_base_url,
    )
    has_more = False
    try:
        if category in _META_TARGETING_CLASS:
            rows, has_more = await _load_targeting_catalog(
                graph,
                business_id=ws.business_id,
                account_id=conn.ad_account_id or "",
                category=category,
            )
            folded_query = query.casefold()
            matching_rows = []
            for row in rows:
                path_value = row.get("path")
                path_text = " ".join(
                    item for item in path_value if isinstance(item, str)
                ) if isinstance(path_value, list) else ""
                searchable = " ".join(
                    str(row.get(key) or "") for key in ("name", "description", "subtext")
                ) + " " + path_text
                if folded_query in searchable.casefold():
                    matching_rows.append(row)
            rows = matching_rows
            has_more = has_more or len(rows) > 12
        else:
            meta_type = _META_TARGETING_AUTOCOMPLETE[category]
            response = await graph.get(
                "search",
                {"type": meta_type, "q": query, "limit": 12},
            )
            raw_rows = response.get("data", [])
            rows = raw_rows if isinstance(raw_rows, list) else []
            paging = response.get("paging")
            has_more = isinstance(paging, dict) and bool(paging.get("next"))
    except MetaGraphError as exc:
        _logger.warning(
            "meta_targeting_search_failed",
            extra={
                "status_code": exc.status_code,
                "reason": str(exc),
                "category": category,
                "query_length": len(query),
            },
        )
        return MetaTargetingSearchResponse(
            available=False, unavailable_reason="meta_lookup_failed"
        )
    except Exception as exc:
        _logger.warning(
            "meta_targeting_search_failed",
            extra={"error_type": type(exc).__name__, "category": category},
        )
        return MetaTargetingSearchResponse(
            available=False, unavailable_reason="meta_lookup_failed"
        )

    results: list[MetaInterestSuggestion] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        target_id = str(row.get("id") or "").strip()
        name = str(row.get("name") or "").strip()
        if not target_id or not name:
            continue
        size = row.get("audience_size")
        if size is None:
            size = row.get("coverage")
        if size is None:
            size = row.get("audience_size_upper_bound")
        if size is None:
            size = row.get("audience_size_lower_bound")
        try:
            audience_size = int(size) if isinstance(size, (int, float, str)) else None
        except (TypeError, ValueError):
            audience_size = None
        raw_path = row.get("path")
        description = row.get("description") or row.get("subtext")
        results.append(
            MetaInterestSuggestion(
                id=target_id,
                name=name,
                category=category,
                audience_size=audience_size,
                description=description[:600] if isinstance(description, str) else None,
                path=[item for item in raw_path if isinstance(item, str)]
                if isinstance(raw_path, list)
                else [],
            )
        )
        if len(results) >= 12:
            break
    return MetaTargetingSearchResponse(results=results, has_more=has_more)


@router.get("/meta/campaigns", response_model=MetaAccountCampaignsResponse)
async def meta_account_campaigns(
    ws: WorkspaceDep,
    db: SessionDep,
    settings: SettingsDep,
    q: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=50, ge=1, le=100),
    after: str | None = Query(default=None, max_length=500),
) -> MetaAccountCampaignsResponse:
    """List connected-account campaigns for the UI, without importing or writing."""
    payload = ListMetaCampaignsInput(search=q, limit=limit, after=after)
    result = await list_meta_campaigns_for_workspace(
        db, ws.business_id, settings, payload
    )
    return MetaAccountCampaignsResponse(
        status=result.status.value if hasattr(result.status, "value") else str(result.status),
        source=result.source,
        unavailable_reason=result.unavailable_reason,
        account_id=result.account_id or None,
        campaigns=[MetaAccountCampaign.model_validate(row.model_dump()) for row in result.campaigns],
        has_more=result.has_more,
        next_cursor=result.next_cursor,
        retrieved_at=result.retrieved_at,
    )


@router.get("/meta/status", response_model=MetaIntegrationStatus)
async def meta_status(
    ws: WorkspaceDep, db: SessionDep, settings: SettingsDep
) -> MetaIntegrationStatus:
    """Report local Meta configuration without calling Meta or exposing secrets."""
    integration = await _meta_integration(db, ws.business_id)
    connector = await load_connector_settings(db, ws.business_id, settings)

    integration_token = None
    if integration is not None and integration.status == "connected" and integration.access_token_enc:
        try:
            integration_token = TokenCipher(settings.token_encryption_key).decrypt(
                integration.access_token_enc
            )
        except Exception:
            integration_token = None
    token_available = bool(integration_token or connector.meta_access_token or settings.meta_access_token)
    ad_account_id = (
        integration.account_id
        if integration is not None and integration.status == "connected" and integration.account_id
        else connector.meta_ad_account_id
    )
    configured = token_available and bool(ad_account_id)
    page_configured = bool(connector.meta_page_id or settings.meta_page_id)
    campaign_ready = configured and page_configured
    return MetaIntegrationStatus(
        connected=configured,
        campaign_ready=campaign_ready,
        page_configured=page_configured,
        ad_account_id=ad_account_id,
        ad_account_name=integration.account_name if integration is not None else None,
        token_expires_at=integration.token_expires_at if integration is not None else None,
        scopes=list(integration.scopes or []) if integration is not None else [],
        last_synced_at=integration.last_synced_at if integration is not None else None,
        message=(
            "Meta ad account and publishing page are configured."
            if campaign_ready
            else "Meta ad account is configured; set the Facebook Page ID before creating ads."
            if configured
            else "Meta configuration is incomplete; a token and Ad Account ID are required."
        ),
    )


@router.post("/meta/connect", response_model=MetaConnectResponse)
async def meta_connect(
    payload: MetaConnectRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
    request_id: RequestIdDep = "",
) -> MetaConnectResponse:
    """Store a Meta access token, encrypted at rest.

    The token is never logged and never returned by any endpoint.

    ``owner`` only. Connecting an ad account is a workspace-level credential
    change: it decides which Meta ad account every agent in this workspace
    writes to, so it is the one integration action a ``member`` or ``viewer``
    must not reach. ``GET /meta/status`` stays open to every role.
    """
    await require_role(principal, settings, ws, minimum="owner")
    cfg = settings
    cipher = TokenCipher(cfg.token_encryption_key)

    integration = await _meta_integration(db, ws.business_id)
    if integration is None:
        integration = Integration(
            id=f"int_{uuid.uuid4().hex[:20]}",
            business_id=ws.business_id,
            provider="meta",
        )
        db.add(integration)

    integration.access_token_enc = cipher.encrypt(payload.access_token)
    integration.account_id = payload.ad_account_id
    integration.status = "connected"
    integration.scopes = ["ads_read", "ads_management"]
    integration.last_synced_at = datetime.now(timezone.utc)
    await db.commit()

    # Best-effort verification of the token; never blocks the connect.
    verified = False
    detail = ""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=cfg.meta_api_timeout_seconds) as client:
            resp = await client.get(
                f"{cfg.meta_graph_base_url}/{cfg.meta_graph_version}/me",
                params={"access_token": payload.access_token, "fields": "id,name"},
            )
            if resp.status_code == 200:
                data = resp.json()
                verified = True
                integration.account_name = data.get("name")
                await db.commit()
            else:
                detail = f"Meta returned {resp.status_code} when verifying the token."
    except Exception as exc:  # pragma: no cover - network optional
        detail = f"Could not verify the token: {exc}"

    status = MetaIntegrationStatus(
        connected=True,
        ad_account_id=integration.account_id,
        ad_account_name=integration.account_name,
        scopes=list(integration.scopes or []),
        last_synced_at=integration.last_synced_at,
        message="Token stored." if verified else "Token stored, but not verified.",
        unavailable_reason=detail or None,
    )
    return MetaConnectResponse(status=status, message=detail or "Meta connected.")


@router.post("/meta/disconnect")
async def meta_disconnect(
    ws: WorkspaceDep, db: SessionDep, principal: AuthDep, settings: SettingsDep
) -> dict[str, str]:
    """Drop the stored Meta grant. ``owner`` only, same as connect."""
    await require_role(principal, settings, ws, minimum="owner")
    integration = await _meta_integration(db, ws.business_id)
    if integration is not None:
        integration.status = "disconnected"
        integration.access_token_enc = None
        await db.commit()
    return {"status": "disconnected"}


__all__ = ["meta_connect", "meta_disconnect", "meta_status"]
