"""Authentication endpoints: register, login, refresh, logout, me.

Five routes, all under ``/api/v1/auth``. They are the only place a password or
a raw session token is ever accepted or produced, and none of them logs one.

The session token is returned in the JSON body for the client to send back as
``Authorization: Bearer <token>``. It is deliberately *not* a cookie -- see
``app/core/auth.py`` for why (cross-origin browser -> FastAPI with no proxy).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUserDep, SessionDep, SettingsDep, bearer_token
from app.core.auth import (
    MembershipGrant,
    Principal,
    create_session,
    hash_password,
    hash_token,
    load_principal,
    needs_rehash,
    new_membership_id,
    new_user_id,
    normalise_email,
    revoke_session,
    verify_password,
)
from app.core.errors import (
    EmailTakenError,
    ForbiddenError,
    NotFoundError,
    SahmError,
    UnauthorizedError,
)
from app.core.logging import get_logger
from app.db.models import Business, Membership, Session, User
from app.schemas.auth import (
    AuthTokenResponse,
    LoginRequest,
    LogoutResponse,
    MeResponse,
    RefreshRequest,
    RegisterRequest,
    UserOut,
    WorkspaceMembership,
)

router = APIRouter(prefix="/auth", tags=["auth"])

_logger = get_logger(__name__)

#: Hash of a password nobody can supply, computed once at import. Verifying a
#: dummy hash when the email is unknown keeps the timing of "no such user" and
#: "wrong password" the same, so login cannot be used to enumerate accounts.
_DUMMY_HASH = hash_password("sahm-timing-equaliser-not-a-real-password")


# ------------------------------------------------------- failure throttling ----


class _FailureCounter:
    """In-process failed-login counter. P4.2 replaces this with the real limiter.

    Honest about what it is: a ``dict`` behind a lock, which bounds attempts per
    process and resets on restart. It stops a script hammering one account in a
    single box; it does not survive a multi-worker deployment, and it is not the
    rate limiter P4.2 specifies. Deliberately not persisted -- writing failed
    attempts to the DB would turn a throttling aid into a write-amplification
    target for an unauthenticated endpoint.
    """

    def __init__(self) -> None:
        self._failures: dict[str, list[datetime]] = {}
        self._locked_until: dict[str, datetime] = {}

    def is_locked(self, key: str, *, max_failures: int, window: timedelta) -> bool:
        stamp = self._locked_until.get(key)
        if stamp is None:
            return False
        if stamp <= datetime.now(UTC):
            del self._locked_until[key]
            self._failures.pop(key, None)
            return False
        return True

    def record_failure(self, key: str, *, max_failures: int, window: timedelta) -> None:
        now = datetime.now(UTC)
        recent = [t for t in self._failures.get(key, ()) if now - t < window]
        recent.append(now)
        self._failures[key] = recent
        if len(recent) >= max_failures:
            self._locked_until[key] = now + window

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)
        self._locked_until.pop(key, None)


_failures = _FailureCounter()


class TooManyAttempts(SahmError):
    """Too many failed logins for this address. 429 with a Retry-After hint."""

    code = "too_many_attempts"
    http_status = 429


# ------------------------------------------------------------------ helpers ----


def _client_ip(request: Request) -> str | None:
    """The caller's IP, or None. Recorded on the session row, never on a token.

    ``request.client`` is the proxy's address behind a load balancer until P5.4
    turns on ``--proxy-headers``; that is a known, documented limitation rather
    than a reason to parse ``X-Forwarded-For`` ourselves and get it wrong.
    """
    client = request.client
    return client.host if client else None


def _user_out(user: User) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=bool(user.is_active),
        last_login_at=user.last_login_at,
        created_at=user.created_at,
    )


async def _workspaces_for(
    db: AsyncSession, principal: Principal
) -> list[WorkspaceMembership]:
    """The caller's workspaces with their business names, oldest first.

    One query for the grants, one for the business rows they name. A membership
    whose business has been deleted is skipped rather than rendered as a broken
    entry, which matches the 404 ``resolve_workspace`` would give it.
    """
    if not principal.memberships:
        return []
    business_ids = [grant.business_id for grant in principal.memberships]
    rows = (
        (await db.execute(select(Business).where(Business.id.in_(business_ids))))
        .scalars()
        .all()
    )
    names = {row.id: row.name for row in rows}
    primary = principal.primary_workspace()
    primary_id = primary.business_id if primary else None
    out: list[WorkspaceMembership] = []
    for grant in principal.memberships:
        if grant.business_id not in names:
            continue
        out.append(
            WorkspaceMembership(
                workspace_id=grant.workspace_id,
                business_id=grant.business_id,
                business_name=names[grant.business_id],
                role=grant.role,
                is_primary=grant.business_id == primary_id,
            )
        )
    return out


def _active_workspace(workspaces: list[WorkspaceMembership]) -> str | None:
    """The workspace a request resolves to with no header: the primary one."""
    for item in workspaces:
        if item.is_primary:
            return item.business_id
    return workspaces[0].business_id if workspaces else None


def _token_response(
    principal: Principal, workspaces: list[WorkspaceMembership], token: str
) -> AuthTokenResponse:
    from app.config import get_settings

    settings = get_settings()
    return AuthTokenResponse(
        access_token=token,
        token_type="Bearer",
        expires_in=settings.session_ttl_days * 24 * 3600,
        user=_user_out_from_principal(principal),
        workspaces=workspaces,
        active_workspace_id=_active_workspace(workspaces),
    )


def _user_out_from_principal(principal: Principal) -> UserOut:
    """A :class:`UserOut` for a principal. Last-login time is not carried here."""
    return UserOut(
        id=principal.user_id,
        email=principal.email,
        full_name=principal.full_name,
        is_active=True,
    )


# ---------------------------------------------------------------- register ----


@router.post(
    "/register", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED
)
async def register(
    payload: RegisterRequest,
    db: SessionDep,
    request: Request,
) -> AuthTokenResponse:
    """Create an account, a business and an ``owner`` membership, atomically.

    All four writes -- user, business, membership, session -- commit together.
    That is the whole point: an account with no membership could not resolve a
    workspace, so a partial registration would produce a user who registers
    successfully and is then locked out of their own data.

    Registering an address that already exists is a ``409``. That necessarily
    confirms the address is taken, which is why the *login* messages below are
    generic -- registration is where a user is told the truth about a conflict.
    """
    email = normalise_email(payload.email)
    existing = await db.scalar(select(User).where(User.email == email))
    if existing is not None:
        raise _email_taken()

    user = User(
        id=new_user_id(),
        email=email,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        is_active=True,
    )
    db.add(user)

    business = Business(
        id=f"biz_{uuid.uuid4().hex[:20]}",
        workspace_id=f"ws_{uuid.uuid4().hex[:20]}",
        name=payload.workspace_name,
    )
    db.add(business)

    # These models intentionally have no ORM relationships, so SQLAlchemy does
    # not know that the user/business rows must precede their membership row.
    # Flush the FK targets explicitly before adding the membership.
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        existing = await db.scalar(select(User).where(User.email == email))
        if existing is not None:
            raise _email_taken() from exc
        _logger.exception(
            "registration_integrity_error",
            extra={"email_domain": email.rsplit("@", 1)[-1]},
        )
        raise

    membership = Membership(
        id=new_membership_id(),
        user_id=user.id,
        workspace_id=business.workspace_id,
        business_id=business.id,
        role="owner",
    )
    db.add(membership)

    try:
        # Flush so the inserts hit the unique index inside our own transaction;
        # a concurrent registration of the same address loses here rather than
        # surfacing an IntegrityError at COMMIT, after the token was minted.
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        # Only translate an actual email collision to 409. Registration also
        # creates a business, membership and session; a constraint failure in
        # any of those writes must not be misreported as "email already exists".
        existing = await db.scalar(select(User).where(User.email == email))
        if existing is not None:
            raise _email_taken() from exc
        _logger.exception(
            "registration_integrity_error",
            extra={"email_domain": email.rsplit("@", 1)[-1]},
        )
        raise

    token = await create_session(
        db,
        user,
        user_agent=request.headers.get("user-agent"),
        ip=_client_ip(request),
        ttl=timedelta(days=7),
    )
    user.last_login_at = datetime.now(UTC)
    await db.commit()

    principal = Principal(
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
        memberships=(
            MembershipGrant(
                workspace_id=membership.workspace_id,
                business_id=membership.business_id,
                role=membership.role,
            ),
        ),
    )
    _logger.info("user_registered", extra={"user_id": user.id})
    return _token_response(principal, await _workspaces_for(db, principal), token)


def _email_taken() -> EmailTakenError:
    """The 409 for a duplicate address, in the standard envelope."""
    return EmailTakenError(
        "an account already exists for this email address",
        detail={"field": "email", "hint": "sign in instead, or use another address"},
    )


# ------------------------------------------------------------------- login ----


@router.post("/login", response_model=AuthTokenResponse)
async def login(
    payload: LoginRequest, db: SessionDep, request: Request, settings: SettingsDep
) -> AuthTokenResponse:
    """Verify a password and issue a session.

    Two rules that matter more than the mechanics:

    * **One message for both failures.** "no such email" and "wrong password"
      return the identical ``401 unauthorized`` with identical text. The
      response is also the same *shape*, so a timing or size difference is the
      only channel left -- hence the dummy-hash verify below.
    * **Failures are counted.** After ``login_max_failures`` in the window the
      address is locked for the window. In-process only; P4.2 replaces this.
    """
    email = normalise_email(payload.email)
    window = timedelta(seconds=settings.login_lockout_seconds)
    if _failures.is_locked(
        email, max_failures=settings.login_max_failures, window=window
    ):
        raise TooManyAttempts(
            "too many failed login attempts; try again later",
            detail={"retry_after_seconds": settings.login_lockout_seconds},
        )

    user = await db.scalar(select(User).where(User.email == email))
    # Always verify a hash, even with no user row, so the two branches below
    # cost the same.
    ok = verify_password(payload.password, user.password_hash if user else _DUMMY_HASH)

    if user is None or not ok:
        _failures.record_failure(
            email, max_failures=settings.login_max_failures, window=window
        )
        _logger.info(
            "login_failed",
            # The email is logged; the password never is. It is PII, but it is
            # the field an operator needs to see a lockout report.
            extra={"email": email, "user_exists": user is not None},
        )
        raise _invalid_credentials()

    if not user.is_active:
        # Distinct from a bad password: the account is real and the credentials
        # were right, so a generic message would be actively confusing. It still
        # discloses that the address exists -- only to someone who already has
        # the password, which is the only audience for whom that is new.
        raise ForbiddenError(
            "this account is deactivated",
            detail={"reason": "account_disabled"},
        )

    _failures.reset(email)

    # Opportunistic upgrade: a hash written with weaker parameters is redone now
    # that we hold the plaintext. This is how a raised iteration count actually
    # reaches the existing user base.
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)

    token = await create_session(
        db,
        user,
        user_agent=request.headers.get("user-agent"),
        ip=_client_ip(request),
        ttl=timedelta(days=settings.session_ttl_days),
    )
    user.last_login_at = datetime.now(UTC)
    await db.commit()

    principal = await load_principal(db, user)
    _logger.info("login_succeeded", extra={"user_id": user.id})
    return _token_response(principal, await _workspaces_for(db, principal), token)


def _invalid_credentials() -> UnauthorizedError:
    """The single message both login failures share."""
    return UnauthorizedError(
        "email or password is incorrect",
        detail={"reason": "invalid_credentials"},
    )


# ----------------------------------------------------------------- refresh ----


@router.post("/refresh", response_model=AuthTokenResponse)
async def refresh(
    payload: RefreshRequest, db: SessionDep, request: Request, settings: SettingsDep
) -> AuthTokenResponse:
    """Rotate a session: issue a new token, revoke the old one.

    Rotation means a stolen token is useful at most until the real client next
    refreshes, and the old row is stamped ``revoked_at`` in the same
    transaction as the new one is created -- so there is no window in which both
    are live.
    """
    row = await db.scalar(
        select(Session).where(Session.token_hash == hash_token(payload.token))
    )
    now = datetime.now(UTC)
    if row is None or row.revoked_at is not None:
        raise _invalid_credentials()
    expires_at = row.expires_at
    if expires_at is not None:
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at <= now:
            raise _invalid_credentials()

    user = await db.scalar(select(User).where(User.id == row.user_id))
    if user is None or not user.is_active:
        raise _invalid_credentials()

    token = await create_session(
        db,
        user,
        user_agent=request.headers.get("user-agent"),
        ip=_client_ip(request),
        ttl=timedelta(days=settings.session_ttl_days),
    )
    row.revoked_at = now
    await db.commit()

    principal = await load_principal(db, user)
    _logger.info("session_refreshed", extra={"user_id": user.id})
    return _token_response(principal, await _workspaces_for(db, principal), token)


# ------------------------------------------------------------------ logout ----


@router.post("/logout", response_model=LogoutResponse)
async def logout(db: SessionDep, request: Request) -> LogoutResponse:
    """Revoke the caller's current session.

    Idempotent: logging out twice, or with a token that was never valid, both
    return ``{"revoked": false}`` with a ``200``. A logout endpoint that tells
    you whether your token was real is an oracle for the same reason login is
    generic. The body token is honoured too, so a client that lost the header
    can still log out.
    """
    token = bearer_token(request)
    revoked = await revoke_session(db, token)
    await db.commit()
    return LogoutResponse(revoked=revoked)


# --------------------------------------------------------------------- me ----


@router.get("/me", response_model=MeResponse)
async def me(
    principal: CurrentUserDep, db: SessionDep, request: Request
) -> MeResponse:
    """The caller's identity, workspaces and role -- the frontend's boot check.

    Requires a token unconditionally, unlike the rest of the API: ``/auth/me`` is
    *about* authentication, so there is no anonymous answer to give.

    ``401`` uses the standard envelope, so the frontend needs exactly one error
    parser to detect an expired session.
    """
    del request  # the principal already carries everything this route returns
    workspaces = await _workspaces_for(db, principal)
    active = _active_workspace(workspaces)
    if active is None:
        # A user with no usable membership. Not a 404 on a row they named --
        # they named nothing -- so this is the honest status for the state.
        raise NotFoundError(
            "this account is not a member of any workspace",
            detail={"user_id": principal.user_id},
        )
    return MeResponse(
        user=_user_out_from_principal(principal),
        workspaces=workspaces,
        active_workspace_id=active,
        active_role=principal.role_for(active),
    )


__all__ = [
    "TooManyAttempts",
    "login",
    "logout",
    "me",
    "refresh",
    "register",
    "router",
]
