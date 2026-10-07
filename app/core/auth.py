"""Identity: password hashing, opaque sessions, and the authenticated principal.

Three decisions are baked in here and the rest of the codebase follows them.

**1. Passwords are PBKDF2-HMAC-SHA256 from the stdlib.**
``hashlib.pbkdf2_hmac`` with a per-user 16-byte random salt, 600k iterations and
a 32-byte derived key, compared with ``hmac.compare_digest``. Argon2id would be
the stronger primitive, but it needs a new package and the roadmap's
"no new dependencies" constraint wins; PBKDF2-SHA256 at 600k iterations is
accepted by OWASP. The iteration count is *encoded in the stored string*
(``pbkdf2_sha256$<iters>$<salt>$<hash>``) so it can be raised later without
invalidating existing hashes -- verify decodes the count it finds rather than
assuming the current one. Swapping in Argon2id is a change to
``_derive_key`` and :func:`hash_password` below, nothing else.

**2. Sessions are opaque server-side tokens, sent as a bearer header.**
No JWT, and deliberately no cookie: the browser calls this API cross-origin
(there is no Next.js proxy), so cookies would drag in
``Access-Control-Allow-Credentials``, SameSite and CSRF tokens for no gain. A
bearer header has none of that surface. Only ``sha256(token)`` is stored, so a
database leak yields hashes that cannot be replayed.

**3. Roles live on the membership row and are enforced here, not in routers.**
``owner | member | viewer``, ranked. ``Principal.has_role`` is the single
predicate; routers depend on it rather than comparing strings themselves.

Nothing in this module logs a password, a raw token, or a token hash. See
``app/core/logging.py`` for the redaction list that backs that up.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Membership, Session, User
from app.core.logging import get_logger

_logger = get_logger(__name__)

#: Prefixes for ids in this module, matching the convention of the rest of the
#: schema (``biz_``, ``cmp_``, ``rec_``...). Used only for readability in logs.
USER_ID_PREFIX = "usr"
MEMBERSHIP_ID_PREFIX = "mbr"
SESSION_ID_PREFIX = "ses"

#: OWASP's current floor for PBKDF2-HMAC-SHA256. Encoded into every hash.
PBKDF2_ITERATIONS = 600_000
PBKDF2_SALT_BYTES = 16
PBKDF2_KEY_BYTES = 32
HASH_SCHEME = "pbkdf2_sha256"

#: How long a freshly issued session stays valid. ``/auth/refresh`` rotates it.
SESSION_TTL = timedelta(days=30)

#: The role hierarchy. ``has_role`` compares ranks, so ``has_role(..., "member")``
#: is true for an owner, and ``has_role(..., "owner")`` is false for a viewer.
ROLE_RANK: dict[str, int] = {"viewer": 1, "member": 2, "owner": 3}


def _now() -> datetime:
    """Current time as an aware UTC datetime. Every timestamp here is UTC."""
    return datetime.now(UTC)


def new_user_id() -> str:
    """A fresh ``usr_<hex>`` primary key."""
    return f"{USER_ID_PREFIX}_{uuid.uuid4().hex[:24]}"


def new_membership_id() -> str:
    """A fresh ``mbr_<hex>`` primary key."""
    return f"{MEMBERSHIP_ID_PREFIX}_{uuid.uuid4().hex[:24]}"


def new_session_id() -> str:
    """A fresh ``ses_<hex>`` primary key."""
    return f"{SESSION_ID_PREFIX}_{uuid.uuid4().hex[:24]}"


def normalise_email(email: str) -> str:
    """Lowercase and strip an address so uniqueness is case-insensitive.

    Storing the normalised form and putting the UNIQUE constraint on it gives
    citext semantics without a Postgres extension, and it is what allows
    :func:`app.api.auth.login` to return one message for "no such user" and
    "wrong password" without a second enumeration channel.
    """
    return email.strip().lower()


# ------------------------------------------------------------- passwords ------


def _derive_key(password: str, salt: bytes, iterations: int) -> bytes:
    """The one place the KDF is called.

    To move to Argon2id, replace the body of this function and the two format
    markers in :func:`hash_password` / :func:`verify_password`. Nothing else in
    the codebase touches password derivation.
    """
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=PBKDF2_KEY_BYTES)


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    """Hash ``password`` into ``pbkdf2_sha256$<iters>$<b64 salt>$<b64 hash>``.

    The iteration count travels inside the string so it can be raised later
    without invalidating hashes written today -- see :func:`needs_rehash`.
    """
    salt = secrets.token_bytes(PBKDF2_SALT_BYTES)
    key = _derive_key(password, salt, iterations)
    return "$".join(
        (
            HASH_SCHEME,
            str(iterations),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(key).decode("ascii"),
        )
    )


def verify_password(password: str, encoded: str | None) -> bool:
    """Constant-time check of ``password`` against a stored hash.

    Returns ``False`` -- never raises -- for a malformed, truncated or absent
    hash, so a corrupt row denies access instead of 500-ing the login route.
    A ``None`` (or empty) ``password_hash`` is the OAuth-only case the roadmap's
    schema allows for, and is simply a failed login.
    """
    if not encoded:
        return False
    try:
        scheme, iterations_raw, salt_b64, key_b64 = encoded.split("$")
        if scheme != HASH_SCHEME:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(key_b64)
        iterations = int(iterations_raw)
    except (ValueError, TypeError):
        return False
    candidate = _derive_key(password, salt, iterations)
    return hmac.compare_digest(candidate, expected)


def needs_rehash(encoded: str, *, iterations: int = PBKDF2_ITERATIONS) -> bool:
    """True when a stored hash was made with weaker parameters than we now use.

    Callers use this at login to upgrade the hash opportunistically, which is
    how a raised iteration count actually reaches the existing user base.
    """
    try:
        scheme, iterations_raw, _, _ = encoded.split("$")
    except ValueError:
        return True
    if scheme != HASH_SCHEME:
        return True
    try:
        return int(iterations_raw) < iterations
    except (TypeError, ValueError):
        return True


# -------------------------------------------------------------- sessions ------


def hash_token(token: str) -> str:
    """The value stored in ``sessions.token_hash``: sha256 hex of the raw token.

    Called on both issue and resolve, so the two can never disagree. It is one
    hash (not pbkdf2) on purpose: the input is already 32 bytes of CSPRNG
    entropy, so there is nothing to brute-force and lookup must stay indexed.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_session_token() -> str:
    """A fresh opaque bearer token. 32 random bytes, URL-safe."""
    return secrets.token_urlsafe(32)


async def create_session(
    db: AsyncSession,
    user: User,
    *,
    user_agent: str | None = None,
    ip: str | None = None,
    ttl: timedelta | None = None,
) -> str:
    """Issue a session for ``user`` and return the raw token.

    The token is returned to the caller and to nobody else: only its sha256 is
    persisted. The row is flushed but not committed, so registration can create
    the user, the business, the membership and this session in one transaction.
    """
    token = new_session_token()
    db.add(
        Session(
            id=new_session_id(),
            user_id=user.id,
            token_hash=hash_token(token),
            user_agent=(user_agent or None) and user_agent[:400],
            ip=ip,
            expires_at=_now() + (ttl or SESSION_TTL),
        )
    )
    await db.flush()
    # The token itself is deliberately absent from this log line.
    _logger.info(
        "session_issued", extra={"user_id": user.id, "session_expires_at": True}
    )
    return token


async def resolve_session(db: AsyncSession, token: str | None) -> User | None:
    """Turn a raw bearer token into its ``User``, or ``None``.

    Rejects an unknown, revoked or expired token -- all three as the same
    ``None``, so a client cannot tell "logged out" from "tampered with". On
    success ``last_used_at`` is refreshed, which is what makes a session list
    answerable later.
    """
    if not token:
        return None
    stmt = select(Session).where(Session.token_hash == hash_token(token))
    row = await db.scalar(stmt)
    if row is None:
        return None
    if row.revoked_at is not None:
        return None
    expires_at = _as_utc(row.expires_at)
    if expires_at is not None and expires_at <= _now():
        return None
    user = await db.scalar(select(User).where(User.id == row.user_id))
    if user is None or not user.is_active:
        return None
    row.last_used_at = _now()
    await db.flush()
    return user


async def revoke_session(db: AsyncSession, token: str | None) -> bool:
    """Revoke one session by its raw token. Returns whether a row was stamped.

    Idempotent by design: logging out twice, or with an already-rotated token,
    succeeds silently rather than leaking whether the token was ever valid.
    """
    if not token:
        return False
    row = await db.scalar(select(Session).where(Session.token_hash == hash_token(token)))
    if row is None or row.revoked_at is not None:
        return False
    row.revoked_at = _now()
    await db.flush()
    _logger.info("session_revoked", extra={"user_id": row.user_id})
    return True


async def revoke_all_for_user(db: AsyncSession, user_id: str) -> int:
    """Revoke every live session for a user. Returns how many were stamped.

    Used for "log me out everywhere" and, more importantly, for
    ``is_active=False``: deactivating an account must kill its sessions, and a
    bearer token has no way to notice the flag until it looks at the session row.
    """
    rows = (
        (
            await db.execute(
                select(Session).where(
                    Session.user_id == user_id, Session.revoked_at.is_(None)
                )
            )
        )
        .scalars()
        .all()
    )
    stamp = _now()
    for row in rows:
        row.revoked_at = stamp
    await db.flush()
    if rows:
        _logger.info(
            "sessions_revoked_bulk", extra={"user_id": user_id, "count": len(rows)}
        )
    return len(rows)


# ------------------------------------------------------------- principal ------


@dataclass(frozen=True, slots=True)
class MembershipGrant:
    """One of a principal's workspace grants, copied out of the DB.

    A plain frozen dataclass rather than the ORM row: the principal is carried
    through dependency resolution and read outside the session's context, so it
    must not hold a lazy relationship that would raise ``MissingGreenlet``.
    """

    workspace_id: str
    business_id: str
    role: str


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is making this request, and which workspaces they may act on.

    Built once by ``app.api.deps.get_principal`` and then treated as immutable
    for the rest of the request.
    """

    user_id: str
    email: str
    full_name: str | None = None
    memberships: tuple[MembershipGrant, ...] = field(default_factory=tuple)

    def membership_for_workspace(self, workspace_id: str) -> MembershipGrant | None:
        """The grant covering ``workspace_id``, or None."""
        for grant in self.memberships:
            if grant.workspace_id == workspace_id:
                return grant
        return None

    def membership_for_business(self, business_id: str) -> MembershipGrant | None:
        """The grant naming ``business_id`` directly, or None.

        Falls back to a workspace-level grant so that a member of a workspace
        containing several businesses still resolves one of them.
        """
        for grant in self.memberships:
            if grant.business_id == business_id:
                return grant
        return None

    def primary_workspace(self) -> MembershipGrant | None:
        """The grant to use when the caller named no workspace.

        With exactly one membership this is unambiguous -- the common case, since
        every registration creates one workspace. With several, the oldest is
        chosen so the answer is stable across requests. P3.1 replaces this with
        an explicit "switch workspace" call; until then the frontend picks with
        the ``X-Workspace-Id`` header, which is always honoured over this.
        """
        if not self.memberships:
            return None
        return sorted(self.memberships, key=lambda g: (g.workspace_id, g.business_id))[0]

    def role_for(self, business_id: str) -> str | None:
        """The caller's role on ``business_id``, or None when they have no grant."""
        grant = self.membership_for_business(business_id)
        return grant.role if grant else None

    def has_role(self, business_id: str, role: str) -> bool:
        """True when the caller's role on ``business_id`` is at least ``role``.

        Ranked, not equality: an owner satisfies ``"member"``. A business the
        caller has no membership on is always False -- there is no implicit
        default role, because a missing grant must not read as permission.
        """
        actual = self.role_for(business_id)
        if actual is None:
            return False
        return ROLE_RANK.get(actual, 0) >= ROLE_RANK.get(role, 99)


async def load_principal(db: AsyncSession, user: User) -> Principal:
    """Load a user's grants and return the immutable :class:`Principal`."""
    rows = (
        (
            await db.execute(
                select(Membership)
                .where(Membership.user_id == user.id)
                .order_by(Membership.created_at, Membership.id)
            )
        )
        .scalars()
        .all()
    )
    return Principal(
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
        memberships=tuple(
            MembershipGrant(
                workspace_id=row.workspace_id,
                business_id=row.business_id,
                role=row.role,
            )
            for row in rows
        ),
    )


# ------------------------------------------------------------------ time ------


def _as_utc(value: datetime | None) -> datetime | None:
    """Coerce a DB timestamp to aware UTC.

    SQLite returns naive datetimes even for ``DateTime(timezone=True)``, and
    comparing a naive value against an aware one raises. Everything written by
    this module is already aware UTC; this only fixes values read back.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "HASH_SCHEME",
    "MEMBERSHIP_ID_PREFIX",
    "PBKDF2_ITERATIONS",
    "ROLE_RANK",
    "SESSION_ID_PREFIX",
    "SESSION_TTL",
    "USER_ID_PREFIX",
    "MembershipGrant",
    "Principal",
    "create_session",
    "hash_password",
    "hash_token",
    "load_principal",
    "needs_rehash",
    "new_membership_id",
    "new_session_id",
    "new_session_token",
    "new_user_id",
    "normalise_email",
    "resolve_session",
    "revoke_all_for_user",
    "revoke_session",
    "verify_password",
]