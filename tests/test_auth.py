"""P1 auth: registration, login, sessions, workspace binding and roles.

Before this phase every endpoint was anonymous and every request resolved "the
first business in the table", so two clients could not see different data
because there was no way to be a second client. These tests create two real
accounts and assert the properties that now hold:

* A password is hashed, and only its salt/derived key are stored.
* Login cannot be used to enumerate accounts -- one message, one shape, for
  "no such email" and "wrong password".
* A session is an opaque token; only ``sha256(token)`` reaches the database, so
  a DB leak is not a session leak.
* Rotation invalidates the old token immediately.
* An expired or revoked session is rejected.
* A user's workspace comes from their membership. Naming someone else's
  workspace in ``X-Workspace-Id`` is a **404**, never a 403, because a 403
  confirms the id is real.
* Roles are enforced server-side: a viewer cannot approve a recommendation.
* ``REQUIRE_AUTH`` flips the sensitive surface to mandatory credentials.

Every failure is asserted to use the one envelope
``{error: {code, message, detail, requestId}}``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import (
    MembershipGrant,
    Principal,
    hash_password,
    hash_token,
    needs_rehash,
    normalise_email,
    resolve_session,
    revoke_session,
    verify_password,
)
from app.db.models import (
    AuditLog,
    Business,
    Campaign,
    Membership,
    Recommendation,
    Session,
    User,
)

PASSWORD = "correct horse battery staple"
WEAK = "short"


# ------------------------------------------------------------------ helpers ----


def _assert_envelope(response: Any, status: int, code: str) -> dict[str, Any]:
    """Assert the standard error envelope and return its ``error`` object."""
    assert response.status_code == status, response.text
    body = response.json()
    assert "detail" not in body, "FastAPI's default shape leaked through"
    error = body["error"]
    assert error["code"] == code, error
    assert error["message"]
    assert error["requestId"]
    assert isinstance(error["detail"], dict)
    return error


async def _register(client: Any, email: str, name: str = "Acme") -> dict[str, Any]:
    """POST /auth/register and return the decoded body (201 asserted)."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": PASSWORD,
            "full_name": "Owner Person",
            "workspace_name": name,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------- fixtures ----


@pytest.fixture()
def cleared_failures() -> Any:
    """Reset the in-process login throttle between tests."""
    from app.api.auth import _failures

    _failures._failures.clear()
    _failures._locked_until.clear()
    yield
    _failures._failures.clear()
    _failures._locked_until.clear()


# ------------------------------------------------------------- password core ---


def test_password_roundtrip_and_wrong_password() -> None:
    encoded = hash_password(PASSWORD)
    assert verify_password(PASSWORD, encoded) is True
    assert verify_password("Correct horse battery staple", encoded) is False
    assert verify_password("", encoded) is False


def test_password_hash_is_not_the_password_and_is_salted() -> None:
    """The stored string must be unusable as a credential, and unique per user."""
    a = hash_password(PASSWORD)
    b = hash_password(PASSWORD)
    assert PASSWORD not in a
    assert a != b, "two hashes of one password must differ -- the salt is per-user"
    assert verify_password(PASSWORD, a) and verify_password(PASSWORD, b)


def test_encoded_hash_carries_its_iteration_count() -> None:
    """So the cost can be raised later without invalidating existing hashes."""
    encoded = hash_password(PASSWORD, iterations=1000)
    scheme, iterations, salt, key = encoded.split("$")
    assert scheme == "pbkdf2_sha256"
    assert int(iterations) == 1000
    assert salt and key
    assert verify_password(PASSWORD, encoded), "must honour the stored count"
    assert needs_rehash(encoded, iterations=600_000), "1000 < 600k, so upgrade"
    assert not needs_rehash(hash_password(PASSWORD))


def test_malformed_hashes_deny_rather_than_raise() -> None:
    """A corrupt row must 401 the login route, not 500 it."""
    for bad in (None, "", "not-a-hash", "pbkdf2_sha256$abc$x$y", "bcrypt$1$a$b"):
        assert verify_password(PASSWORD, bad) is False


def test_normalise_email_is_case_insensitive() -> None:
    assert normalise_email("  Foo@Example.COM ") == "foo@example.com"


# ----------------------------------------------------------------- register ----


async def test_register_creates_user_business_and_owner_membership(
    api_client, seeded_db: AsyncSession
) -> None:
    """A new account lands on a usable workspace, not an empty shell."""
    body = await _register(api_client, "owner@example.com")

    assert body["tokenType"] == "Bearer"
    assert body["accessToken"]
    assert body["user"]["email"] == "owner@example.com"
    assert body["user"]["id"].startswith("usr_")

    user = await seeded_db.scalar(
        select(User).where(User.email == "owner@example.com")
    )
    assert user is not None
    assert user.password_hash != PASSWORD
    assert user.is_active is True

    business = await seeded_db.get(Business, body["workspaces"][0]["businessId"])
    assert business is not None, "register must create the business too"
    assert business.name == "Acme"

    membership = await seeded_db.scalar(
        select(Membership).where(Membership.user_id == user.id)
    )
    assert membership is not None, "no membership means no resolvable workspace"
    assert membership.role == "owner"
    assert membership.business_id == business.id
    assert membership.workspace_id == business.workspace_id


async def test_register_normalises_the_email(
    api_client, seeded_db: AsyncSession
) -> None:
    """Mixed case must be one account, not two -- that is the citext behaviour."""
    await _register(api_client, "Founder@Example.COM")
    user = await seeded_db.scalar(select(User).where(User.email == "founder@example.com"))
    assert user is not None
    assert user.email == user.email.lower()


async def test_duplicate_registration_is_409(api_client) -> None:
    await _register(api_client, "dup@example.com")
    resp = await api_client.post(
        "/api/v1/auth/register",
        json={"email": "dup@example.com", "password": PASSWORD, "workspaceName": "X"},
    )
    _assert_envelope(resp, 409, "email_taken")


async def test_weak_passwords_are_rejected_by_validation(api_client) -> None:
    resp = await api_client.post(
        "/api/v1/auth/register",
        json={"email": "weak@example.com", "password": WEAK, "workspaceName": "X"},
    )
    assert resp.status_code == 422
    _assert_envelope(resp, 422, "validation_failed")


async def test_breached_password_is_rejected(api_client) -> None:
    resp = await api_client.post(
        "/api/v1/auth/register",
        json={"email": "pwn@example.com", "password": "password123", "workspaceName": "X"},
    )
    _assert_envelope(resp, 422, "validation_failed")


# -------------------------------------------------------------------- login ----


async def test_login_with_the_right_password_issues_a_session(api_client) -> None:
    await _register(api_client, "login@example.com")
    resp = await api_client.post(
        "/api/v1/auth/login",
        json={"email": "login@example.com", "password": PASSWORD},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["accessToken"]
    assert body["user"]["email"] == "login@example.com"
    assert body["expiresIn"] > 0
    assert body["workspaces"][0]["role"] == "owner"


async def test_login_with_a_wrong_password_is_401(api_client, cleared_failures) -> None:
    await _register(api_client, "wrong@example.com")
    resp = await api_client.post(
        "/api/v1/auth/login",
        json={"email": "wrong@example.com", "password": "not the password"},
    )
    _assert_envelope(resp, 401, "unauthorized")


async def test_unknown_email_and_wrong_password_are_indistinguishable(
    api_client, cleared_failures
) -> None:
    """The whole point: login must not confirm which addresses are registered.

    Same status, same code, same message, same detail -- a client cannot tell the
    two failures apart, and a non-existent address is still given a real hash
    verify so the timing does not give it away either.
    """
    await _register(api_client, "real@example.com")

    wrong_password = await api_client.post(
        "/api/v1/auth/login",
        json={"email": "real@example.com", "password": "not the password"},
    )
    unknown_email = await api_client.post(
        "/api/v1/auth/login",
        json={"email": "ghost@example.com", "password": "not the password"},
    )

    assert wrong_password.status_code == unknown_email.status_code == 401
    a = wrong_password.json()["error"]
    b = unknown_email.json()["error"]
    assert a["code"] == b["code"] == "unauthorized"
    assert a["message"] == b["message"], "the message must not distinguish the cases"
    assert a["detail"] == b["detail"]
    assert "exist" not in a["message"].lower()


async def test_repeated_failures_are_throttled(
    api_client, cleared_failures, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The counter bounds guessing; P4.2 replaces it with a real limiter."""
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "login_max_failures", 3)
    await _register(api_client, "brute@example.com")

    for _ in range(3):
        resp = await api_client.post(
            "/api/v1/auth/login",
            json={"email": "brute@example.com", "password": "wrong wrong wrong"},
        )
        assert resp.status_code == 401

    locked = await api_client.post(
        "/api/v1/auth/login",
        json={"email": "brute@example.com", "password": PASSWORD},
    )
    _assert_envelope(locked, 429, "too_many_attempts")


# --------------------------------------------------------------- session row ----


async def test_session_token_is_never_stored_in_plaintext(
    api_client, seeded_db: AsyncSession
) -> None:
    """Only sha256(token) is persisted, so a DB leak is not a session leak."""
    body = await _register(api_client, "hash@example.com")
    token = body["accessToken"]

    rows = (await seeded_db.execute(select(Session))).scalars().all()
    assert rows, "register must have created a session row"
    for row in rows:
        assert row.token_hash != token
        assert token not in row.token_hash
        assert row.token_hash == hash_token(token)
        assert len(row.token_hash) == 64, "sha256 hex"
        assert row.user_id == body["user"]["id"]
        assert row.expires_at is not None


async def test_a_bearer_token_authenticates(api_client) -> None:
    body = await _register(api_client, "bearer@example.com")
    resp = await api_client.get("/api/v1/auth/me", headers=_auth(body["accessToken"]))
    assert resp.status_code == 200, resp.text
    assert resp.json()["user"]["email"] == "bearer@example.com"


@pytest.mark.parametrize(
    "header",
    [
        {"Authorization": "Bearer not-a-real-token"},
        {"Authorization": "Bearer "},
        {"Authorization": "Basic YWRtaW46YWRtaW4="},
        {"Authorization": "bearer"},
        {},
    ],
)
async def test_bad_authorization_headers_are_simply_anonymous(
    api_client, header: dict[str, str]
) -> None:
    """A malformed token is "anonymous", not an error -- except on /auth/me.

    Every other endpoint must keep answering for an anonymous caller while the
    frontend has no login page.
    """
    resp = await api_client.get("/api/v1/campaigns", headers=header)
    assert resp.status_code == 200, resp.text


async def test_logout_revokes_the_session(api_client, seeded_db: AsyncSession) -> None:
    body = await _register(api_client, "logout@example.com")
    headers = _auth(body["accessToken"])

    assert (await api_client.get("/api/v1/auth/me", headers=headers)).status_code == 200

    resp = await api_client.post("/api/v1/auth/logout", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["revoked"] is True

    _assert_envelope(
        await api_client.get("/api/v1/auth/me", headers=headers), 401, "unauthorized"
    )


async def test_logout_is_idempotent_and_says_nothing(api_client) -> None:
    """Logging out twice must not reveal whether the token was ever valid."""
    body = await _register(api_client, "twice@example.com")
    headers = _auth(body["accessToken"])
    await api_client.post("/api/v1/auth/logout", headers=headers)
    second = await api_client.post("/api/v1/auth/logout", headers=headers)
    assert second.status_code == 200
    assert second.json()["revoked"] is False


async def test_refresh_rotates_and_invalidates_the_old_token(api_client) -> None:
    """Rotation: the old token dies in the same transaction as the new one."""
    body = await _register(api_client, "rotate@example.com")
    old = body["accessToken"]

    resp = await api_client.post("/api/v1/auth/refresh", json={"token": old})
    assert resp.status_code == 200, resp.text
    new = resp.json()["accessToken"]
    assert new != old, "a refresh that returns the same token is not a rotation"

    assert (await api_client.get("/api/v1/auth/me", headers=_auth(new))).status_code == 200
    _assert_envelope(
        await api_client.get("/api/v1/auth/me", headers=_auth(old)), 401, "unauthorized"
    )

    again = await api_client.post("/api/v1/auth/refresh", json={"token": old})
    _assert_envelope(again, 401, "unauthorized")


async def test_expired_session_is_rejected(api_client, seeded_db: AsyncSession) -> None:
    body = await _register(api_client, "expired@example.com")
    token = body["accessToken"]

    row = await seeded_db.scalar(select(Session))
    assert row is not None
    row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await seeded_db.commit()

    _assert_envelope(
        await api_client.get("/api/v1/auth/me", headers=_auth(token)), 401, "unauthorized"
    )


async def test_revoked_session_row_is_rejected_directly(
    seeded_db: AsyncSession
) -> None:
    """``resolve_session`` is the single gate; revoked must fail at it."""
    user = User(
        id="usr_revoked",
        email="revoked@example.com",
        password_hash=hash_password(PASSWORD),
    )
    business = Business(id="biz_revoked", workspace_id="ws_revoked", name="R")
    db = seeded_db
    db.add(user)
    db.add(business)
    db.add(
        Membership(
            id="mbr_revoked",
            user_id=user.id,
            workspace_id=business.workspace_id,
            business_id=business.id,
            role="owner",
        )
    )
    from app.core.auth import create_session

    token = await create_session(db, user, user_agent="pytest", ip="127.0.0.1")
    await db.commit()
    assert (await resolve_session(db, token)) is not None

    assert await revoke_session(db, token) is True
    await db.commit()
    assert await resolve_session(db, token) is None
    assert await resolve_session(db, "never-issued") is None
    assert await resolve_session(db, None) is None


async def test_last_used_at_is_refreshed_on_use(
    api_client, seeded_db: AsyncSession
) -> None:
    body = await _register(api_client, "used@example.com")
    await api_client.get("/api/v1/auth/me", headers=_auth(body["accessToken"]))
    row = await seeded_db.scalar(select(Session))
    assert row is not None
    assert row.last_used_at is not None


# ----------------------------------------------------------------- auth/me ----


async def test_me_returns_the_user_and_workspaces(api_client) -> None:
    body = await _register(api_client, "me@example.com", name="Me Inc")
    resp = await api_client.get("/api/v1/auth/me", headers=_auth(body["accessToken"]))
    assert resp.status_code == 200, resp.text
    me = resp.json()

    assert me["user"]["email"] == "me@example.com"
    assert me["activeWorkspaceId"] == body["workspaces"][0]["businessId"]
    assert me["activeRole"] == "owner"

    assert len(me["workspaces"]) == 1
    workspace = me["workspaces"][0]
    assert workspace["businessName"] == "Me Inc"
    assert workspace["role"] == "owner"
    assert workspace["isPrimary"] is True
    assert workspace["businessId"] and workspace["workspaceId"]
    # Never the hash, never the password.
    assert "passwordHash" not in resp.text
    assert PASSWORD not in resp.text


async def test_me_requires_a_token_with_the_standard_envelope(api_client) -> None:
    _assert_envelope(await api_client.get("/api/v1/auth/me"), 401, "unauthorized")


# ------------------------------------------- the workspace/principal contract ----


async def test_a_user_cannot_resolve_another_users_workspace_via_the_header(
    api_client,
) -> None:
    """404, never 403: a 403 would confirm the id is real.

    User A holds one business. User B holds another. B's token plus A's
    ``X-Workspace-Id`` must find nothing -- and must not confirm that the id
    exists.
    """
    a = await _register(api_client, "iso-a@example.com", name="Alpha")
    b = await _register(api_client, "iso-b@example.com", name="Bravo")

    resp = await api_client.get(
        "/api/v1/campaigns",
        headers={**_auth(b["accessToken"]), "X-Workspace-Id": a["workspaces"][0]["businessId"]},
    )
    _assert_envelope(resp, 404, "not_found")

    # And the same id is visible to its own owner, so the 404 above is the
    # membership check and not a missing row.
    own = await api_client.get(
        "/api/v1/campaigns",
        headers={**_auth(a["accessToken"]), "X-Workspace-Id": a["workspaces"][0]["businessId"]},
    )
    assert own.status_code == 200


async def test_with_a_token_the_header_is_optional(api_client) -> None:
    """No header still resolves -- to the caller's own workspace.

    The business name is the probe: two accounts register different names and
    each must read back its own with no ``X-Workspace-Id`` at all. That is the
    whole point of resolving from the membership instead of falling back to the
    oldest business in the table.
    """
    a = await _register(api_client, "nohdr-a@example.com", name="Solo Alpha")
    b = await _register(api_client, "nohdr-b@example.com", name="Solo Bravo")

    for token, expected in (
        (a["accessToken"], "Solo Alpha"),
        (b["accessToken"], "Solo Bravo"),
    ):
        resp = await api_client.get("/api/v1/business/profile", headers=_auth(token))
        assert resp.status_code == 200, resp.text
        assert resp.json()["business"]["name"] == expected


async def test_a_bearer_token_cannot_see_another_users_data(api_client) -> None:
    """The end-to-end promise: two accounts, two disjoint sets of campaigns."""
    a = await _register(api_client, "data-a@example.com", name="Alpha")
    b = await _register(api_client, "data-b@example.com", name="Bravo")
    biz_a = a["workspaces"][0]["businessId"]
    biz_b = b["workspaces"][0]["businessId"]

    # Seed one campaign each, as the owner of that campaign's own workspace.
    for token, business_id, campaign_name in (
        (a["accessToken"], biz_a, "Alpha Campaign"),
        (b["accessToken"], biz_b, "Bravo Campaign"),
    ):
        resp = await api_client.post(
            "/api/v1/campaigns",
            headers={**_auth(token), "X-Workspace-Id": business_id},
            json={"name": campaign_name, "budget": 100.0},
        )
        assert resp.status_code in (200, 201), resp.text

    alice = await api_client.get("/api/v1/campaigns", headers=_auth(a["accessToken"]))
    bob = await api_client.get("/api/v1/campaigns", headers=_auth(b["accessToken"]))
    assert alice.status_code == 200 and bob.status_code == 200

    a_names = {c["name"] for c in alice.json()["campaigns"]}
    b_names = {c["name"] for c in bob.json()["campaigns"]}
    assert "Alpha Campaign" in a_names and "Bravo Campaign" not in a_names
    assert "Bravo Campaign" in b_names and "Alpha Campaign" not in b_names


# ------------------------------------------------------------------- roles ----


async def _make_member(
    db: AsyncSession, email: str, owner_business_id: str, role: str
) -> str:
    """Register a user and re-point their membership at an existing business."""
    owner_business = await db.get(Business, owner_business_id)
    assert owner_business is not None
    user = User(
        id=f"usr_{email.replace('@', '_').replace('.', '_')}",
        email=email,
        password_hash=hash_password(PASSWORD),
    )
    db.add(user)
    await db.flush()
    db.add(
        Membership(
            id=f"mbr_{email.replace('@', '_').replace('.', '_')}",
            user_id=user.id,
            workspace_id=owner_business.workspace_id,
            business_id=owner_business.id,
            role=role,
        )
    )
    from app.core.auth import create_session

    token = await create_session(db, user, user_agent="pytest", ip="127.0.0.1")
    # ``create_session`` only flushes -- its caller owns the transaction -- so the
    # session row is committed here or the token never resolves for anyone.
    await db.commit()
    return token


async def _seed_recommendation(db: AsyncSession, business_id: str) -> str:
    db.add(Campaign(id="cmp_role", business_id=business_id, name="Role Campaign"))
    db.add(
        Recommendation(
            id="rec_role",
            campaign_id="cmp_role",
            workspace_id=business_id,
            action_type="pause_ad",
            title="Pause the reel",
            rationale="CTR fell.",
            payload={"status": "PAUSED"},
            status="pending",
            requires_approval=True,
        )
    )
    await db.commit()
    return "rec_role"


async def test_a_viewer_cannot_approve_a_recommendation(
    api_client, seeded_db: AsyncSession
) -> None:
    """P1.7's headline case: read-only means read-only, server-side."""
    owner = await _register(api_client, "role-owner@example.com", name="Role Co")
    business_id = owner["workspaces"][0]["businessId"]
    rec_id = await _seed_recommendation(seeded_db, business_id)
    viewer_token = await _make_member(
        seeded_db, "viewer@example.com", business_id, "viewer"
    )

    resp = await api_client.post(
        f"/api/v1/recommendations/{rec_id}/approve",
        headers=_auth(viewer_token),
        json={"confirm": False},
    )
    error = _assert_envelope(resp, 403, "forbidden")
    assert error["detail"]["required_role"] == "member"
    assert error["detail"]["actual_role"] == "viewer"


async def test_a_member_can_approve_and_the_actor_is_the_real_user(
    api_client, seeded_db: AsyncSession
) -> None:
    """Approval is attributed to a user id, not the literal "user" (P4.5)."""
    owner = await _register(api_client, "act-owner@example.com", name="Actor Co")
    business_id = owner["workspaces"][0]["businessId"]
    rec_id = await _seed_recommendation(seeded_db, business_id)
    member_token = await _make_member(
        seeded_db, "member@example.com", business_id, "member"
    )
    member_id = f"usr_{'member@example.com'.replace('@', '_').replace('.', '_')}"

    resp = await api_client.post(
        f"/api/v1/recommendations/{rec_id}/approve",
        headers=_auth(member_token),
        json={"confirm": False},
    )
    assert resp.status_code == 200, resp.text

    # The audit row is where the actor lands -- the response schema does not
    # carry approved_by, and the audit log is what P4.4 will expose.
    audit = await seeded_db.scalar(
        select(AuditLog).where(AuditLog.entity_id == rec_id)
    )
    assert audit is not None
    assert audit.actor == member_id, "the actor must name a real user id"


async def test_only_an_owner_can_change_a_connector(
    api_client, seeded_db: AsyncSession
) -> None:
    """Disconnecting Meta is a workspace-level credential change."""
    owner = await _register(api_client, "conn-owner@example.com", name="Conn Co")
    business_id = owner["workspaces"][0]["businessId"]
    member_token = await _make_member(
        seeded_db, "conn-member@example.com", business_id, "member"
    )

    denied = await api_client.post(
        "/api/v1/integrations/meta/disconnect", headers=_auth(member_token)
    )
    error = _assert_envelope(denied, 403, "forbidden")
    assert error["detail"]["required_role"] == "owner"
    assert error["detail"]["actual_role"] == "member"

    allowed = await api_client.post(
        "/api/v1/integrations/meta/disconnect", headers=_auth(owner["accessToken"])
    )
    assert allowed.status_code == 200, allowed.text


async def test_a_viewer_may_still_read(api_client, seeded_db: AsyncSession) -> None:
    """Read-only must mean *can read*, not *is locked out of everything*."""
    owner = await _register(api_client, "read-owner@example.com", name="Read Co")
    business_id = owner["workspaces"][0]["businessId"]
    viewer_token = await _make_member(
        seeded_db, "read-viewer@example.com", business_id, "viewer"
    )

    for path in (
        "/api/v1/campaigns",
        "/api/v1/recommendations",
        "/api/v1/personas",
        "/api/v1/integrations/meta/status",
    ):
        resp = await api_client.get(path, headers=_auth(viewer_token))
        assert resp.status_code == 200, f"{path} -> {resp.text}"


async def test_a_viewer_cannot_edit_a_persona(api_client, seeded_db: AsyncSession) -> None:
    owner = await _register(api_client, "pers-owner@example.com", name="Pers Co")
    business_id = owner["workspaces"][0]["businessId"]
    viewer_token = await _make_member(
        seeded_db, "pers-viewer@example.com", business_id, "viewer"
    )

    denied = await api_client.patch(
        "/api/v1/personas/layla",
        headers=_auth(viewer_token),
        json={"tone": "bold"},
    )
    _assert_envelope(denied, 403, "forbidden")

    allowed = await api_client.patch(
        "/api/v1/personas/layla",
        headers=_auth(owner["accessToken"]),
        json={"tone": "bold"},
    )
    assert allowed.status_code == 200, allowed.text


# --------------------------------------------------- the require_auth switch ----


async def test_require_auth_off_is_todays_behaviour(
    api_client, seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default off: the frontend keeps working before it has a login page."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "require_auth", False)

    resp = await api_client.post(
        "/api/v1/integrations/meta/disconnect",
        headers={"X-Workspace-Id": "biz_test"},
    )
    assert resp.status_code == 200, resp.text


async def test_require_auth_on_is_401_in_the_standard_envelope(
    api_client, seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One env var locks the sensitive surface, with the one error shape."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "require_auth", True)

    resp = await api_client.post(
        "/api/v1/integrations/meta/disconnect",
        headers={"X-Workspace-Id": "biz_test"},
    )
    error = _assert_envelope(resp, 401, "unauthorized")
    assert error["detail"]["reason"] == "missing_or_invalid_token"


async def test_require_auth_on_locks_tool_invocation(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "require_auth", True)
    resp = await api_client.post(
        "/api/v1/tools/generate_ad_copy/invoke",
        headers={"X-Workspace-Id": "biz_test"},
        json={},
    )
    _assert_envelope(resp, 401, "unauthorized")


async def test_require_auth_on_still_leaves_reads_and_health_open(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The switch is scoped to writes; the catalogue must stay browsable."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "require_auth", True)

    assert (await api_client.get("/health")).status_code == 200
    for path in ("/api/v1/campaigns", "/api/v1/tools", "/api/v1/personas"):
        resp = await api_client.get(path, headers={"X-Workspace-Id": "biz_test"})
        assert resp.status_code == 200, f"{path} -> {resp.text}"


async def test_require_auth_on_still_admits_a_valid_token(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "require_auth", True)
    owner = await _register(api_client, "locked-owner@example.com", name="Locked Co")
    resp = await api_client.post(
        "/api/v1/integrations/meta/disconnect", headers=_auth(owner["accessToken"])
    )
    assert resp.status_code == 200, resp.text


# --------------------------------------------------------- principal helpers ----


def test_has_role_is_ranked_and_fails_closed_for_strangers() -> None:
    grants = (
        MembershipGrant(workspace_id="ws_1", business_id="biz_1", role="owner"),
        MembershipGrant(workspace_id="ws_2", business_id="biz_2", role="viewer"),
    )
    principal = Principal(user_id="usr_1", email="a@b.c", memberships=grants)

    assert principal.has_role("biz_1", "owner") is True
    assert principal.has_role("biz_1", "member") is True, "ranked, not equality"
    assert principal.has_role("biz_1", "viewer") is True

    assert principal.has_role("biz_2", "viewer") is True
    assert principal.has_role("biz_2", "member") is False
    assert principal.has_role("biz_2", "owner") is False

    # No grant at all must never read as permission.
    assert principal.has_role("biz_stranger", "viewer") is False
    assert principal.role_for("biz_stranger") is None

    assert principal.primary_workspace() is not None
    assert Principal(user_id="u", email="e").primary_workspace() is None