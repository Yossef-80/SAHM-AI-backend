"""Supabase token exchange endpoint.

The frontend sends a Supabase access token (JWT) and we:
1. Verify it via JWKS (already done in get_principal)
2. Load/create the local profile
3. Return workspace memberships in the format the frontend expects
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import SessionDep, SettingsDep, get_principal
from app.core.supabase_auth import Principal, SupabaseAuthError, verify_supabase_token
from app.config import get_settings
from app.core.logging import get_logger

_logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/supabase-exchange")
async def supabase_exchange(
    authorization: str | None = Header(None),
    db: SessionDep = None,
    settings: SettingsDep = None,
) -> dict:
    """Exchange a Supabase access token for workspace membership info.

    The frontend calls this after Supabase auth state changes.
    Returns the workspaces and active workspace for the authenticated user.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )

    token = authorization[7:].strip()  # Remove "Bearer "

    try:
        claims = await verify_supabase_token(token)
    except SupabaseAuthError as exc:
        _logger.warning("supabase_exchange_failed", extra={"error": exc.message})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.message,
        ) from exc

    supabase_user_id = claims.get("sub")
    email = claims.get("email")
    full_name = claims.get("user_metadata", {}).get("full_name") or claims.get("user_metadata", {}).get("name")

    if not supabase_user_id or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token claims",
        )

    # Get or create profile
    from app.core.supabase_auth import get_or_create_profile, load_principal_from_profile
    profile = await get_or_create_profile(db, supabase_user_id, email, full_name)
    # Persist the profile/workspace before returning its ID to the frontend.
    # Otherwise the request-scoped session rolls it back and later calls 404.
    await db.commit()
    principal = await load_principal_from_profile(db, profile)

    # Override with JWT claims
    principal = Principal(
        user_id=principal.user_id,
        email=email,
        full_name=full_name,
        memberships=principal.memberships,
    )

    # Build workspace response
    from app.api.auth import _workspaces_for, _active_workspace
    workspaces = await _workspaces_for(db, principal)
    active_workspace_id = _active_workspace(workspaces)

    return {
        "workspaces": [
            {
                "workspaceId": w.workspace_id,
                "businessId": w.business_id,
                "businessName": w.business_name,
                "role": w.role,
                "isPrimary": w.is_primary,
            }
            for w in workspaces
        ],
        "activeWorkspaceId": active_workspace_id,
    }
