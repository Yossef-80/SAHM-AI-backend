"""Integration endpoints: Meta status and connect."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import RequestIdDep, SessionDep
from app.core.logging import get_logger
from app.core.security import TokenCipher
from app.db.models import Business, Integration
from app.schemas.integrations import (
    MetaConnectRequest,
    MetaConnectResponse,
    MetaIntegrationStatus,
)

router = APIRouter(prefix="/integrations", tags=["integrations"])

_logger = get_logger(__name__)


async def _business(db) -> Business:
    stmt = select(Business).order_by(Business.created_at).limit(1)
    business = (await db.execute(stmt)).scalars().first()
    if business is None:
        business = Business(id=f"biz_{uuid.uuid4().hex[:20]}", name="My Business")
        db.add(business)
        await db.flush()
    return business


async def _meta_integration(db, business_id: str) -> Integration | None:
    stmt = select(Integration).where(
        Integration.business_id == business_id, Integration.provider == "meta"
    )
    return (await db.execute(stmt)).scalars().first()


@router.get("/meta/status", response_model=MetaIntegrationStatus)
async def meta_status(db: SessionDep) -> MetaIntegrationStatus:
    business = await _business(db)
    integration = await _meta_integration(db, business.id)
    if integration is None or integration.status != "connected":
        return MetaIntegrationStatus(
            connected=False,
            message=(
                "Meta is not connected. Campaign performance and ad-library tools "
                "will return 'unavailable' until you connect."
            ),
        )
    return MetaIntegrationStatus(
        connected=True,
        ad_account_id=integration.account_id,
        ad_account_name=integration.account_name,
        token_expires_at=integration.token_expires_at,
        scopes=list(integration.scopes or []),
        last_synced_at=integration.last_synced_at,
        message="Meta is connected.",
    )


@router.post("/meta/connect", response_model=MetaConnectResponse)
async def meta_connect(
    payload: MetaConnectRequest, db: SessionDep, request_id: RequestIdDep = ""
) -> MetaConnectResponse:
    """Store a Meta access token, encrypted at rest.

    The token is never logged and never returned by any endpoint.
    """
    from app.config import get_settings

    business = await _business(db)
    cfg = get_settings()
    cipher = TokenCipher(cfg.token_encryption_key)

    integration = await _meta_integration(db, business.id)
    if integration is None:
        integration = Integration(
            id=f"int_{uuid.uuid4().hex[:20]}",
            business_id=business.id,
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
async def meta_disconnect(db: SessionDep) -> dict[str, str]:
    business = await _business(db)
    integration = await _meta_integration(db, business.id)
    if integration is not None:
        integration.status = "disconnected"
        integration.access_token_enc = None
        await db.commit()
    return {"status": "disconnected"}


__all__ = ["meta_connect", "meta_disconnect", "meta_status"]
