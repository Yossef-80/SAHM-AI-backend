"""Integration endpoints: Meta status and connect."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

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
from app.core.security import TokenCipher
from app.core.tools.settings import load_connector_settings
from app.db.models import Integration
from app.schemas.integrations import (
    MetaConnectRequest,
    MetaConnectResponse,
    MetaInterestSuggestion,
    MetaIntegrationStatus,
    MetaTargetingSearchResponse,
)

router = APIRouter(prefix="/integrations", tags=["integrations"])

_logger = get_logger(__name__)


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
) -> MetaTargetingSearchResponse:
    """Resolve an interest name to the Meta IDs that targeting actually accepts.

    Returns `available: false` (not an error) when Meta is not configured or
    rejects the lookup, so the wizard can fall back to no interest targeting.
    """
    query = q.strip()
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
    try:
        payload = await graph.get(
            "search", {"type": "adinterest", "q": query, "limit": 12}
        )
    except MetaGraphError:
        # Message is deliberately generic: Meta errors can echo user input, and
        # the caller only needs to know the lookup did not succeed.
        return MetaTargetingSearchResponse(
            available=False, unavailable_reason="meta_lookup_failed"
        )

    rows = payload.get("data") if isinstance(payload, dict) else None
    results: list[MetaInterestSuggestion] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        interest_id = str(row.get("id") or "").strip()
        name = str(row.get("name") or "").strip()
        if not interest_id or not name:
            continue
        size = row.get("audience_size")
        path = row.get("path")
        results.append(
            MetaInterestSuggestion(
                id=interest_id,
                name=name,
                audience_size=int(size) if isinstance(size, (int, float)) else None,
                path=[str(p) for p in path if isinstance(p, str)] if isinstance(path, list) else [],
            )
        )
    return MetaTargetingSearchResponse(results=results)


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
