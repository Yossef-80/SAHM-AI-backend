"""Supabase Auth integration: JWT verification and profile loading.

Replaces the custom auth module (password hashing, opaque sessions, Principal).
Supabase issues JWTs; we verify them via JWKS and map the `sub` claim to our
local `profiles` table to get business_id + role.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import httpx
from jose import jwt
from jose.exceptions import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.logging import get_logger
from app.db.models import Business, Profile

_logger = get_logger(__name__)

# Role hierarchy (same as before)
ROLE_RANK: dict[str, int] = {"viewer": 1, "member": 2, "owner": 3}


@dataclass(frozen=True, slots=True)
class MembershipGrant:
    """One workspace grant for a principal."""
    workspace_id: str
    business_id: str
    role: str


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated principal from a verified Supabase JWT."""
    user_id: str          # Supabase auth.users.id (UUID string)
    email: str
    full_name: str | None = None
    memberships: tuple[MembershipGrant, ...] = ()

    def membership_for_business(self, business_id: str) -> MembershipGrant | None:
        for grant in self.memberships:
            if grant.business_id == business_id:
                return grant
        return None

    def primary_workspace(self) -> MembershipGrant | None:
        if not self.memberships:
            return None
        return sorted(self.memberships, key=lambda g: (g.workspace_id, g.business_id))[0]

    def role_for(self, business_id: str) -> str | None:
        grant = self.membership_for_business(business_id)
        return grant.role if grant else None

    def has_role(self, business_id: str, role: str) -> bool:
        actual = self.role_for(business_id)
        if actual is None:
            return False
        return ROLE_RANK.get(actual, 0) >= ROLE_RANK.get(role, 99)


# ---------------------------------------------------------------------------
# JWKS caching
# ---------------------------------------------------------------------------

_JWKS_CACHE: dict[str, tuple[dict[str, Any], float]] = {}  # kid -> (key, expires_at)
_JWKS_FETCHED_AT: float = 0
_JWKS_TTL = 3600  # 1 hour


async def _fetch_jwks() -> dict[str, Any]:
    """Fetch and cache Supabase JWKS."""
    global _JWKS_FETCHED_AT
    settings = get_settings()
    jwks_url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"

    now = time.time()
    if now - _JWKS_FETCHED_AT < _JWKS_TTL and _JWKS_CACHE:
        return {kid: key for kid, (key, exp) in _JWKS_CACHE.items() if exp > now}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(jwks_url)
        resp.raise_for_status()
        jwks = resp.json()

    _JWKS_CACHE.clear()
    for key_data in jwks.get("keys", []):
        kid = key_data.get("kid")
        if kid:
            _JWKS_CACHE[kid] = (key_data, now + _JWKS_TTL)
    _JWKS_FETCHED_AT = now
    return {kid: key for kid, (key, _) in _JWKS_CACHE.items()}


def _get_signing_key(kid: str) -> dict[str, Any] | None:
    """Get a signing key from cache, fetching JWKS if needed."""
    # This is called synchronously from jwt.decode, so we can't await here.
    # The cache is populated by _fetch_jwks which runs at startup / on first request.
    return _JWKS_CACHE.get(kid, (None, 0))[0]


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------


class SupabaseAuthError(Exception):
    """Raised when Supabase JWT verification fails."""
    def __init__(self, message: str, code: str = "unauthorized"):
        self.message = message
        self.code = code
        super().__init__(message)


async def verify_supabase_token(token: str) -> dict[str, Any]:
    """Verify a Supabase access token (JWT) and return its claims.

    Raises SupabaseAuthError on any verification failure.
    """
    settings = get_settings()

    # Ensure JWKS is fresh
    await _fetch_jwks()

    try:
        # Get the key ID from the header
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        algorithm = header.get("alg")
        if not kid:
            raise SupabaseAuthError("Token missing key ID", "invalid_token")
        if algorithm not in {"RS256", "ES256"}:
            raise SupabaseAuthError("Token uses an unsupported signing algorithm", "invalid_token")

        # Find the signing key
        key_data = _get_signing_key(kid)
        if not key_data:
            # Force a JWKS refresh and try once more
            await _fetch_jwks()
            key_data = _get_signing_key(kid)
            if not key_data:
                raise SupabaseAuthError("Signing key not found", "invalid_token")

        # Supabase projects can sign with either RSA or P-256 EC keys. Never
        # trust the token's alg alone: it must match the public key metadata.
        key_algorithm = key_data.get("alg")
        if not key_algorithm:
            key_type = key_data.get("kty")
            curve = key_data.get("crv")
            key_algorithm = "RS256" if key_type == "RSA" else (
                "ES256" if key_type == "EC" and curve == "P-256" else None
            )
        if key_data.get("use", "sig") != "sig" or key_algorithm != algorithm:
            raise SupabaseAuthError("Token signing key does not match its algorithm", "invalid_token")

        # Verify and decode
        claims = jwt.decode(
            token,
            key_data,
            algorithms=[algorithm],
            audience="authenticated",
            issuer=f"{settings.supabase_url}/auth/v1",
            options={"verify_exp": True, "verify_aud": True, "verify_iss": True},
        )
        return claims

    except JWTError as exc:
        _logger.warning("supabase_token_invalid", extra={"error": str(exc)})
        raise SupabaseAuthError("Invalid or expired token", "invalid_token") from exc
    except Exception as exc:
        _logger.error("supabase_token_verification_failed", extra={"error": str(exc)})
        raise SupabaseAuthError("Token verification failed", "verification_failed") from exc


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------


async def load_profile(db: AsyncSession, supabase_user_id: str) -> Profile | None:
    """Load a profile by Supabase user ID."""
    stmt = select(Profile).where(Profile.supabase_user_id == supabase_user_id)
    return await db.scalar(stmt)


async def load_principal_from_profile(db: AsyncSession, profile: Profile) -> Principal:
    """Build a Principal from a Profile row."""
    # Load the business to get workspace_id
    business = await db.scalar(select(Business).where(Business.id == profile.business_id))
    if not business:
        raise SupabaseAuthError("Profile references missing business", "profile_invalid")

    grant = MembershipGrant(
        workspace_id=business.workspace_id,
        business_id=business.id,
        role=profile.role,
    )
    return Principal(
        user_id=str(profile.supabase_user_id),
        email="",  # Will be filled from JWT claims
        full_name=None,
        memberships=(grant,),
    )


async def get_or_create_profile(
    db: AsyncSession,
    supabase_user_id: str,
    email: str,
    full_name: str | None,
) -> Profile:
    """Get existing profile or create one for a new Supabase user.

    For brand-new users, we create a business + profile with owner role.
    This mirrors the old registration flow but triggered by first login.
    """
    profile = await load_profile(db, supabase_user_id)
    if profile:
        return profile

    # In local development, attach a first-time Supabase account to the seeded
    # demo tenant when it exists. Production users always receive a private one.
    settings = get_settings()
    demo_business = await db.get(Business, "biz_demo") if settings.environment == "dev" else None
    if demo_business is not None:
        business_id = demo_business.id
    else:
        import uuid
        business_id = f"biz_{uuid.uuid4().hex[:20]}"
        workspace_id = f"ws_{uuid.uuid4().hex[:20]}"
        business = Business(
            id=business_id,
            workspace_id=workspace_id,
            name="My Business",
        )
        db.add(business)

    profile = Profile(
        supabase_user_id=supabase_user_id,
        business_id=business_id,
        role="owner",
    )
    db.add(profile)
    await db.flush()

    _logger.info("profile_created", extra={"supabase_user_id": supabase_user_id, "business_id": business_id})
    return profile


# ---------------------------------------------------------------------------
# Backwards-compatible helpers (for gradual migration)
# ---------------------------------------------------------------------------


def create_legacy_token() -> str:
    """Not used with Supabase Auth. Kept for any legacy code paths."""
    import secrets
    return secrets.token_urlsafe(32)


__all__ = [
    "MembershipGrant",
    "Principal",
    "ROLE_RANK",
    "SupabaseAuthError",
    "verify_supabase_token",
    "load_profile",
    "load_principal_from_profile",
    "get_or_create_profile",
    "create_legacy_token",
]
