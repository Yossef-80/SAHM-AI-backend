"""Shared API dependencies.

One resolver, one place
-----------------------
``resolve_workspace`` is the only code in the repository that decides which
business a request acts on. It lives here, next to ``SessionDep`` and
``LLMDep``, rather than in ``app/core/`` because it is an HTTP-layer concern
(it reads a header and is declared with FastAPI ``Depends``) and because every
router already imports from this module -- putting it anywhere else would add a
second import edge and a second place for a future author to forget.

Resolution order (P0 wrote steps 2 and 3; P1 put step 1.5 in front of them):

0. **A principal is present** -> the tenant comes from their memberships. The
   ``X-Workspace-Id`` header may still *choose* among workspaces they already
   belong to, but it can no longer name someone else's. This is what replaced
   the P0 note below: the seam was not "read the header" but "decide the
   tenant", and the decision now consults the database of memberships rather
   than trusting a caller-supplied string.
1. **No principal, ``X-Workspace-Id`` present** -> that ``businesses.id``. A
   header that names a row which does not exist is a ``404``. It never falls
   through to the single-tenant fallback: silently downgrading a misconfigured
   client to "whatever business is oldest" is exactly the cross-tenant leak
   this dependency exists to remove.
2. **No principal, no header, ``Settings.require_workspace_header``** -> explicit
   ``WorkspaceRequiredError``.
3. **No principal, no header, not strict** -> the historical behaviour: the
   oldest business, created on first use. Dev keeps working untouched.

Step 0 is additive by design. Every public read endpoint keeps answering for an
anonymous caller, so the existing frontend keeps working before its login page
exists; what authentication adds is that an *authenticated* caller is confined
to their own tenants, and ``Settings.require_auth`` can later flip the
sensitive surface to mandatory credentials with one env var.

Binding happens in the dependency rather than in the request middleware because
the workspace is only known once the dependency has run, and dependencies run
before the handler -- which is what lets ``bind_workspace`` put a tenant id on
every log line (D7).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, TypeVar

from fastapi import Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.auth import load_principal, resolve_session
from app.core.errors import (
    ForbiddenError,
    NotFoundError,
    UnauthorizedError,
    WorkspaceRequiredError,
)
from app.core.llm.base import LLMClient
from app.core.llm.router import get_workspace_llm
from app.core.logging import bind_workspace, get_logger
from app.core.supabase_auth import MembershipGrant, Principal, SupabaseAuthError, get_or_create_profile, load_principal_from_profile, verify_supabase_token
from app.db.models import Business, Campaign, Profile, User
from app.db.session import get_session

#: Header a client uses to name its tenant. Carries a ``businesses.id``.
WORKSPACE_HEADER = "X-Workspace-Id"

_logger = get_logger(__name__)

SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]


def request_id_of(request: Request) -> str:
    """The request id set by middleware, or a fresh one."""
    return getattr(request.state, "request_id", None) or f"req_{id(request):x}"


RequestIdDep = Annotated[str, Depends(request_id_of)]


# ------------------------------------------------------------------ the user --

#: Prefix every bearer token is sent with.
BEARER_PREFIX = "bearer "


def bearer_token(request: Request) -> str | None:
    """The raw token from ``Authorization: Bearer <token>``, or None.

    Deliberately header-only. Cookies would work too, and are the more
    conventional choice -- but the browser calls this API cross-origin (there is
    no Next.js proxy), so cookies would require
    ``Access-Control-Allow-Credentials``, a SameSite decision and CSRF tokens on
    every mutating route. A bearer header has none of that surface. See the
    module docstring in ``app/core/auth.py``.
    """
    header = request.headers.get("authorization")
    if not header:
        return None
    if not header.lower().startswith(BEARER_PREFIX):
        return None
    token = header[len(BEARER_PREFIX) :].strip()
    return token or None


async def get_principal(request: Request, db: SessionDep, settings: SettingsDep) -> Principal | None:
    """Resolve the bearer token to a :class:`Principal`, or None.

    With Supabase Auth enabled, the token is a Supabase-issued JWT.
    We verify it via JWKS, then load/create the local profile.
    """
    token = bearer_token(request)
    if token is None:
        return None

    # If Supabase Auth is not enabled, fall back to legacy session auth
    if not settings.use_supabase_auth:
        # Legacy path - kept for gradual migration
        cached_user_id = getattr(request.state, "auth_user_id", None)
        if cached_user_id:
            user = await db.get(User, cached_user_id)
            if user is None:
                return None
            return await load_principal(db, user)

        user = await resolve_session(db, token)
        if user is None:
            return None
        await db.commit()
        return await load_principal(db, user)

    # Supabase Auth path
    try:
        claims = await verify_supabase_token(token)
    except SupabaseAuthError:
        return None

    supabase_user_id = claims.get("sub")
    email = claims.get("email")
    full_name = claims.get("user_metadata", {}).get("full_name") or claims.get("user_metadata", {}).get("name")

    if not supabase_user_id or not email:
        return None

    # Get or create local profile
    profile = await get_or_create_profile(db, supabase_user_id, email, full_name)
    principal = await load_principal_from_profile(db, profile)

    # Override email/full_name from JWT claims (source of truth)
    principal = Principal(
        user_id=principal.user_id,
        email=email,
        full_name=full_name,
        memberships=principal.memberships,
    )

    return principal


#: Optional authentication. ``None`` when the request carried no valid token.
AuthDep = Annotated[Principal | None, Depends(get_principal)]


async def get_current_principal(
    principal: AuthDep, settings: SettingsDep
) -> Principal:
    """Require authentication, in the standard error envelope.

    Raises ``UnauthorizedError`` -> ``401 unauthorized`` with
    ``{code, message, detail, requestId}``, the same shape every other failure
    uses, so a client needs one error parser.
    """
    if principal is None:
        if settings.require_auth:
            raise UnauthorizedError(
                "authentication is required for this endpoint",
                detail={
                    "reason": "missing_or_invalid_token",
                    "hint": "send 'Authorization: Bearer <token>' from POST /auth/login",
                },
            )
        raise UnauthorizedError(
            "authentication is required for this endpoint",
            detail={"reason": "missing_or_invalid_token"},
        )
    return principal


#: Required authentication. 401s when no valid token is present.
CurrentUserDep = Annotated[Principal, Depends(get_current_principal)]


# ------------------------------------------------------------------- roles ----

#: What each role may do, enforced by :func:`require_role`.
#:
#: =============  ==================================================================
#: ``owner``      everything a member can do, plus disconnect/rotate an integration
#:                and change workspace-level connector settings.
#: ``member``     run tools and agent tasks, create/edit campaigns, write brand
#:                memory, edit personas, approve/reject recommendations.
#: ``viewer``     read-only. No writes of any kind.
#: =============  ==================================================================
ROLE_RANKS: dict[str, int] = {"viewer": 1, "member": 2, "owner": 3}


async def require_role(
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceContext,
    *,
    minimum: str,
) -> Principal | None:
    """Authorise a write against the caller's role on the resolved workspace.

    Returns the principal so a handler can use it as its single auth dependency.

    Three cases, and the middle one is the point:

    * No principal and ``require_auth`` off -> ``None``, i.e. the route behaves
      exactly as it did before authentication existed. The frontend keeps
      working until it has a login page.
    * No principal and ``require_auth`` on -> ``401``.
    * A principal whose role is below ``minimum`` -> ``403 forbidden`` with
      ``required_role`` in ``detail``. This is a 403 rather than a 404 on
      purpose: the caller is a known member of this tenant who lacks permission,
      so there is nothing to conceal from them.
    """
    if principal is None:
        if settings.require_auth:
            raise UnauthorizedError(
                "authentication is required for this endpoint",
                detail={
                    "reason": "missing_or_invalid_token",
                    "required_role": minimum,
                    "hint": "send 'Authorization: Bearer <token>' from POST /auth/login",
                },
            )
        return None
    if not principal.has_role(ws.business_id, minimum):
        raise ForbiddenError(
            f"role '{principal.role_for(ws.business_id) or 'none'}' may not perform "
            f"this action; '{minimum}' or higher is required",
            detail={
                "required_role": minimum,
                "actual_role": principal.role_for(ws.business_id),
                "workspace_id": ws.workspace_id,
            },
        )
    return principal


# ------------------------------------------------------------ the workspace --


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    """The tenant a request acts on.

    Both ids are carried explicitly rather than re-derived by each caller:
    ``business_id`` is what every tenant-scoped query filters on, and
    ``workspace_id`` is the axis the token budget, the persona overrides and
    P1's membership table hang off.
    """

    business_id: str
    workspace_id: str
    business: Business


async def _single_tenant_business(db: AsyncSession) -> Business:
    """The single-tenant fallback: the oldest business, created on first use.

    Only reached when the request named no workspace and strict tenancy is off.
    """
    business = await db.scalar(select(Business).order_by(Business.created_at).limit(1))
    if business is None:
        business = Business(id=f"biz_{uuid.uuid4().hex[:20]}", name="My Business")
        db.add(business)
        await db.flush()
    return business


def _bind(business: Business) -> WorkspaceContext:
    """Wrap a business for routers and publish it to the log context."""
    workspace_id = business.workspace_id or ""
    bind_workspace(workspace_id, business.id)
    return WorkspaceContext(
        business_id=business.id, workspace_id=workspace_id, business=business
    )


async def _resolve_for_principal(
    db: AsyncSession, principal: Principal, requested: str | None
) -> WorkspaceContext:
    """Pick one of ``principal``'s own workspaces. Never anyone else's.

    Two inputs, and the header can only ever *narrow*:

    * No header -> ``primary_workspace()``, i.e. their only membership (the
      normal case) or the oldest of several.
    * Header -> that business, provided one of their grants covers it.

    Every rejection is a ``404`` and never a ``403``. A 403 would tell a stranger
    "that workspace exists, you just may not have it", turning a guessed id into
    an enumeration oracle over the customer list; a 404 is indistinguishable from
    a workspace that was never created.
    """
    if requested:
        business = await db.scalar(select(Business).where(Business.id == requested))
        if business is None:
            raise NotFoundError(
                f"workspace '{requested}' not found",
                detail={"header": WORKSPACE_HEADER, "value": requested},
            )
        grant = principal.membership_for_business(business.id)
        if grant is None:
            # A workspace-level grant also covers the businesses inside it: they
            # may act on this workspace, so the business resolves.
            workspace_grant = principal.membership_for_workspace(
                business.workspace_id or ""
            )
            if workspace_grant is not None:
                grant = MembershipGrant(
                    workspace_id=workspace_grant.workspace_id,
                    business_id=business.id,
                    role=workspace_grant.role,
                )
        if grant is None:
            # Fail closed, and identically to a row that does not exist at all.
            raise NotFoundError(
                f"workspace '{requested}' not found",
                detail={"header": WORKSPACE_HEADER, "value": requested},
            )
        _logger.info(
            "workspace_resolved_from_membership",
            extra={
                "user_id": principal.user_id,
                "requested_workspace": requested,
                "granted_role": grant.role,
            },
        )
        return _bind(business)

    grant = principal.primary_workspace()
    if grant is None:
        # A user row with no membership is a broken account, not a tenant with
        # no rows. Refusing is the point: falling back here would hand an
        # authenticated user the *oldest* business in the table.
        raise NotFoundError(
            "this account is not a member of any workspace",
            detail={"user_id": principal.user_id},
        )
    business = await db.scalar(select(Business).where(Business.id == grant.business_id))
    if business is None:
        # The membership points at a business that no longer exists.
        raise NotFoundError(
            "workspace not found",
            detail={"workspace_id": grant.workspace_id, "value": grant.business_id},
        )
    return _bind(business)


async def resolve_workspace(
    request: Request,
    db: SessionDep,
    settings: SettingsDep,
    principal: AuthDep = None,
) -> WorkspaceContext:
    """Bind this request to a workspace. The single tenant resolver.

    Every router takes the result as ``ws: WorkspaceDep``; no router resolves a
    business itself. See the module docstring for the resolution order.

    ``principal`` carries a default only so this stays callable directly with
    the three P0 positional arguments (the isolation tests do exactly that).
    FastAPI always injects it through ``AuthDep``.
    """
    requested = request.headers.get(WORKSPACE_HEADER)

    if principal is not None:
        # Authenticated: the membership table decides, not the header alone.
        return await _resolve_for_principal(db, principal, requested)

    if requested:
        business = await db.scalar(select(Business).where(Business.id == requested))
        if business is None:
            raise NotFoundError(
                f"workspace '{requested}' not found",
                detail={"header": WORKSPACE_HEADER, "value": requested},
            )
        return _bind(business)

    if settings.require_workspace_header:
        raise WorkspaceRequiredError(
            "no workspace was identified for this request",
            detail={
                "header": WORKSPACE_HEADER,
                "hint": "send the X-Workspace-Id header naming the business to act on",
            },
        )
    if settings.environment == "prod":
        _logger.warning(
            "workspace_not_resolved_falling_back_to_single_tenant",
            extra={"environment": settings.environment},
        )
    return _bind(await _single_tenant_business(db))


WorkspaceDep = Annotated[WorkspaceContext, Depends(resolve_workspace)]


async def get_llm_client(
    db: SessionDep,
    ws: WorkspaceDep,
    settings: SettingsDep,
) -> AsyncIterator[LLMClient]:
    """Resolve the workspace's configured strong-tier LLM client."""
    yield await get_workspace_llm(
        "strong", db=db, business_id=ws.business_id, settings=settings
    )


LLMDep = Annotated[LLMClient, Depends(get_llm_client)]


async def get_fast_llm_client(
    db: SessionDep,
    ws: WorkspaceDep,
    settings: SettingsDep,
) -> AsyncIterator[LLMClient]:
    """Resolve the workspace's configured fast-tier LLM client."""
    yield await get_workspace_llm(
        "fast", db=db, business_id=ws.business_id, settings=settings
    )


FastLLMDep = Annotated[LLMClient, Depends(get_fast_llm_client)]


# ------------------------------------------------------------- ownership -----


RowT = TypeVar("RowT")


def assert_owned(
    row: RowT | None,
    workspace: WorkspaceContext,
    *,
    entity: str,
    owner: str | None = None,
) -> RowT:
    """Return ``row`` when ``workspace`` owns it, otherwise raise ``NotFoundError``.

    404, never 403. A 403 confirms to a stranger that the row exists, which is
    the information leak that turns a guessed id into an enumeration oracle; a
    404 is indistinguishable from "no such id".

    ``owner`` is the owning ``businesses.id``. It defaults to the row's own
    ``business_id``, which is what campaigns, conversations, tasks, creatives,
    insights and (since P1) recommendations carry. Pass it explicitly for rows
    whose tenant is reached some other way; doing that in SQL rather than by
    walking a relationship matters, because a lazy access outside the request's
    async context raises ``MissingGreenlet``.

    A row whose owner cannot be resolved is treated as unowned, so the check
    fails closed on orphans and NULL foreign keys.
    """
    if row is None:
        raise NotFoundError(f"{entity} not found")
    owner_id = owner if owner is not None else getattr(row, "business_id", None)
    if owner_id != workspace.business_id:
        raise NotFoundError(f"{entity} not found")
    return row


async def assert_campaign_owned(
    db: AsyncSession, campaign_id: str | None, workspace: WorkspaceContext
) -> Campaign | None:
    """Guard a campaign id that arrived in a request body; return it, or None.

    Body-supplied ids are the one place a caller can point us at another
    tenant's row without naming it in the path, so they are checked before
    they are used. A missing id is not an error: the endpoint simply has no
    campaign context.
    """
    if not campaign_id:
        return None
    stmt = select(Campaign).where(
        Campaign.id == campaign_id, Campaign.business_id == workspace.business_id
    )
    return assert_owned(
        await db.scalar(stmt), workspace, entity=f"campaign '{campaign_id}'"
    )


__all__ = [
    "BEARER_PREFIX",
    "AuthDep",
    "CurrentUserDep",
    "LLMDep",
    "FastLLMDep",
    "ROLE_RANKS",
    "RequestIdDep",
    "SessionDep",
    "SettingsDep",
    "WORKSPACE_HEADER",
    "WorkspaceContext",
    "WorkspaceDep",
    "assert_campaign_owned",
    "assert_owned",
    "bearer_token",
    "get_current_principal",
    "get_llm_client",
    "get_principal",
    "request_id_of",
    "require_role",
    "resolve_workspace",
]
