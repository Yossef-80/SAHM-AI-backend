"""P4: usage meter, rate limiting, audit log, security headers, readiness.

What is asserted here, and why it matters more than it looks
-----------------------------------------------------------
* The meter is **honest**: the total matches ``SELECT SUM(tokens_in +
  tokens_out)`` exactly, ``percentUsed`` is not clamped at 100, ``resetAt`` is the
  real UTC boundary, and the projection is derived from the elapsed fraction.
* A 429 carries the standard envelope, ``Retry-After`` and the ``X-RateLimit-*``
  family, and is keyed per caller -- one tenant cannot throttle another.
* The audit log is workspace-scoped and owner-only.
* Security headers are on every response, and a CORS wildcard that no browser
  would honour refuses to start rather than silently breaking the frontend.

The limiter is per app (one app per process in production). ``api_client`` resets
the buckets, so these tests drive the limiter directly with a known capacity
instead of firing hundreds of requests through the stack.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import create_session, hash_password
from app.db.models import (
    AuditLog,
    Business,
    Membership,
    UsageRecord,
    User,
)

PASSWORD = "correct horse battery staple"

#: Small enough that a test does not need to fire 200 requests to see a 429.
SMALL_CAPACITY = 3


# ------------------------------------------------------------------ helpers ----


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _register(client: Any, email: str, name: str = "Acme") -> dict[str, Any]:
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": PASSWORD,
            "fullName": "Owner Person",
            "workspaceName": name,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _make_member(
    db: AsyncSession, email: str, business_id: str, role: str
) -> str:
    business = await db.get(Business, business_id)
    assert business is not None
    slug = email.replace("@", "_").replace(".", "_")
    user = User(id=f"usr_{slug}", email=email, password_hash=hash_password(PASSWORD))
    db.add(user)
    await db.flush()
    db.add(
        Membership(
            id=f"mbr_{slug}",
            user_id=user.id,
            workspace_id=business.workspace_id,
            business_id=business.id,
            role=role,
        )
    )
    token = await create_session(db, user, user_agent="pytest", ip="127.0.0.1")
    await db.commit()
    return token


def _record(db: AsyncSession, workspace_id: str, *, tokens_in: int, tokens_out: int,
            model: str = "mock-strong-1", persona_id: str = "layla") -> UsageRecord:
    row = UsageRecord(
        id=f"ur_{workspace_id}_{tokens_in}_{tokens_out}_{model}_{persona_id}",
        business_id="biz_usage",
        workspace_id=workspace_id,
        model=model,
        persona_id=persona_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        created_at=datetime.now(UTC),
    )
    db.add(row)
    return row


@asynccontextmanager
async def _metered_workspace(
    api_client: Any, seeded_db: AsyncSession, email: str, *, budget: int = 1000
) -> Any:
    """Register an owner whose workspace has its own budget; yield its ids."""
    owner = await _register(api_client, email, "Metered Co")
    business_id = owner["workspaces"][0]["businessId"]
    business = await seeded_db.get(Business, business_id)
    assert business is not None
    workspace_id = business.workspace_id
    business.daily_token_budget = budget
    await seeded_db.commit()
    yield {
        "token": owner["accessToken"],
        "user_id": owner["user"]["id"],
        "business_id": business_id,
        "workspace_id": workspace_id,
        "headers": _auth(owner["accessToken"]),
    }


# ------------------------------------------------------------------- the meter --


async def test_usage_matches_the_raw_sum(
    api_client, seeded_db: AsyncSession
) -> None:
    """The headline: the meter is not a guess."""
    async with _metered_workspace(api_client, seeded_db, "usage@example.com") as ws:
        workspace_id = ws["workspace_id"]
        headers = ws["headers"]
        for index in range(4):
            _record(
                seeded_db,
                workspace_id,
                tokens_in=100 * (index + 1),
                tokens_out=50 * (index + 1),
                model="mock-strong-1" if index % 2 else "mock-fast-1",
                persona_id="layla" if index < 3 else "omar",
            )
        await seeded_db.commit()

        resp = await api_client.get("/api/v1/usage?period=today", headers=headers)
        assert resp.status_code == 200, resp.text
        body = resp.json()

        total_in = sum(100 * (i + 1) for i in range(4))  # 1000
        total_out = sum(50 * (i + 1) for i in range(4))  # 500
        assert body["tokensIn"] == total_in
        assert body["tokensOut"] == total_out
        assert body["totalTokens"] == total_in + total_out
        assert body["calls"] == 4
        assert body["workspaceId"] == workspace_id
        assert body["period"] == "today"

        # index 0 and 2 used the fast tier, 1 and 3 the strong one.
        by_model = {row["key"]: row["tokens"] for row in body["byModel"]}
        assert sum(by_model.values()) == total_in + total_out
        assert by_model["mock-fast-1"] == 150 + 450
        assert by_model["mock-strong-1"] == 300 + 600
        by_persona = {row["key"]: row["tokens"] for row in body["byPersona"]}
        assert set(by_persona) == {"layla", "omar"}
        assert round(sum(r["share"] for r in body["byPersona"]), 3) == 1.0


async def test_usage_percent_used_is_not_clamped_and_reset_is_utc(
    api_client, seeded_db: AsyncSession
) -> None:
    """140% must read as 140%: clamping hides a real overage."""
    async with _metered_workspace(api_client, seeded_db, "over@example.com") as ws:
        _record(seeded_db, ws["workspace_id"], tokens_in=1_000, tokens_out=400)
        await seeded_db.commit()

        body = (
            await api_client.get("/api/v1/usage", headers=ws["headers"])
        ).json()
        assert body["budgetTokens"] == 1000
        assert body["budgetSource"] == "workspace"
        assert body["percentUsed"] == 140.0
        assert body["exceeded"] is True
        assert body["remainingTokens"] == 0, "remaining is floored, never negative"

        reset = datetime.fromisoformat(body["resetAt"])
        assert reset.tzinfo is not None
        assert (reset.hour, reset.minute, reset.second) == (0, 0, 0), "UTC midnight"
        assert reset > datetime.now(UTC)
        assert reset - datetime.now(UTC) <= timedelta(days=1)
        assert datetime.fromisoformat(body["periodStart"]).hour == 0


async def test_usage_month_budget_and_window(
    api_client, seeded_db: AsyncSession
) -> None:
    """A month view uses a month-sized budget, not the daily one relabelled."""
    async with _metered_workspace(api_client, seeded_db, "month@example.com") as ws:
        _record(seeded_db, ws["workspace_id"], tokens_in=60_000, tokens_out=0)
        await seeded_db.commit()

        today = (
            await api_client.get("/api/v1/usage?period=today", headers=ws["headers"])
        ).json()
        month = (
            await api_client.get("/api/v1/usage?period=month", headers=ws["headers"])
        ).json()

        assert today["budgetTokens"] == 1000
        assert month["budgetTokens"] == 1000 * 30
        assert month["totalTokens"] == today["totalTokens"] == 60_000
        assert month["percentUsed"] == round(60_000 / 30_000 * 100, 2)
        start = datetime.fromisoformat(month["periodStart"])
        assert start.day == 1
        assert datetime.fromisoformat(month["resetAt"]).day == 1
        assert datetime.fromisoformat(month["resetAt"]) > start

        assert month["burnRatePerHour"] > 0
        assert month["projectedTotalTokens"] >= month["totalTokens"]


async def test_usage_falls_back_to_the_process_budget(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workspace with no override inherits ``DAILY_TOKEN_BUDGET``."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "daily_token_budget", 777)
    owner = await _register(api_client, "inherit@example.com", "Inherit Co")
    body = (
        await api_client.get("/api/v1/usage", headers=_auth(owner["accessToken"]))
    ).json()
    assert body["budgetSource"] == "default"
    assert body["budgetTokens"] == 777


async def test_usage_is_workspace_scoped(
    api_client, seeded_db: AsyncSession
) -> None:
    seeded_db.add(
        UsageRecord(
            id="ur_stranger",
            business_id="biz_stranger",
            workspace_id="ws_stranger",
            model="mock-strong-1",
            tokens_in=999,
            tokens_out=1,
        )
    )
    async with _metered_workspace(api_client, seeded_db, "scoped-a@example.com") as ws:
        await seeded_db.commit()
        body = (
            await api_client.get("/api/v1/usage", headers=ws["headers"])
        ).json()
        assert body["totalTokens"] == 0, "another workspace's spend is not yours"
        assert body["workspaceId"] == ws["workspace_id"]
        assert body["calls"] == 0


async def test_usage_rejects_an_unknown_period(api_client) -> None:
    owner = await _register(api_client, "period@example.com")
    resp = await api_client.get(
        "/api/v1/usage?period=year", headers=_auth(owner["accessToken"])
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_failed"


# ------------------------------------------------------------- rate limiting --


def test_token_bucket_allows_burst_then_refills() -> None:
    from app.core.ratelimit import (
        ROUTE_CLASS_READ,
        Limit,
        RateLimiter,
    )

    limiter = RateLimiter(
        limits={ROUTE_CLASS_READ: Limit(capacity=3, refill_per_second=10.0)},
        enabled=True,
    )
    verdicts = [asyncio_run(limiter.consume(ROUTE_CLASS_READ, "user:a")) for _ in range(4)]
    assert [v.allowed for v in verdicts] == [True, True, True, False]
    assert verdicts[0].remaining == 2
    assert verdicts[-1].retry_after >= 1
    assert verdicts[-1].limit == 3

    # A different key has its own bucket: one noisy tenant cannot throttle another.
    other = asyncio_run(limiter.consume(ROUTE_CLASS_READ, "user:b"))
    assert other.allowed is True


def test_route_classes_are_ordered_strictest_first() -> None:
    from app.core.ratelimit import (
        ROUTE_CLASS_AUTH,
        ROUTE_CLASS_READ,
        ROUTE_CLASS_TOOL,
        ROUTE_CLASS_WRITE,
        classify,
    )

    assert classify("POST", "/api/v1/auth/login") == ROUTE_CLASS_AUTH
    assert classify("GET", "/api/v1/auth/me") == ROUTE_CLASS_AUTH
    assert classify("POST", "/api/v1/tools/invoke") == ROUTE_CLASS_TOOL
    assert classify("POST", "/api/v1/recommendations/x/approve") == ROUTE_CLASS_WRITE
    assert classify("GET", "/api/v1/campaigns") == ROUTE_CLASS_READ


async def test_rate_limit_returns_429_with_retry_after_and_the_envelope(
    api_client, rate_limiter: Any
) -> None:
    """The exact response shape a client has to parse."""
    from app.core.ratelimit import ROUTE_CLASS_READ, Limit

    rate_limiter.limits[ROUTE_CLASS_READ] = Limit(
        capacity=SMALL_CAPACITY, refill_per_second=0.001
    )
    owner = await _register(api_client, "limited@example.com")
    headers = _auth(owner["accessToken"])

    statuses = [
        (await api_client.get("/api/v1/campaigns", headers=headers)).status_code
        for _ in range(SMALL_CAPACITY + 1)
    ]
    assert statuses[:SMALL_CAPACITY] == [200] * SMALL_CAPACITY

    resp = await api_client.get("/api/v1/campaigns", headers=headers)
    assert resp.status_code == 429, resp.text
    assert int(resp.headers["retry-after"]) >= 1
    assert resp.headers["x-ratelimit-limit"] == str(SMALL_CAPACITY)
    assert resp.headers["x-ratelimit-remaining"] == "0"

    body = resp.json()
    assert "detail" not in body, "FastAPI's default shape leaked through"
    error = body["error"]
    assert error["code"] == "rate_limited"
    assert error["message"]
    assert error["detail"]["retry_after"] == int(resp.headers["retry-after"])
    assert error["detail"]["route_class"] == "read"

    # Security headers ride on the 429 too -- it is still an HTTP response.
    assert resp.headers["x-content-type-options"] == "nosniff"


async def test_successful_responses_carry_the_rate_limit_headers(
    api_client,
) -> None:
    owner = await _register(api_client, "headers@example.com")
    resp = await api_client.get(
        "/api/v1/campaigns", headers=_auth(owner["accessToken"])
    )
    assert resp.status_code == 200, resp.text
    assert "x-ratelimit-limit" in resp.headers
    assert "x-ratelimit-remaining" in resp.headers


async def test_rate_limit_is_keyed_per_caller(
    api_client, rate_limiter: Any
) -> None:
    """Tenant A exhausting its bucket must not throttle tenant B."""
    from app.core.ratelimit import ROUTE_CLASS_AUTH, Limit

    # Register both users first, so the arithmetic below is about /auth/me alone
    # rather than about how many setup calls the test happened to make.
    a_token = (await _register(api_client, "a@example.com"))["accessToken"]
    b_token = (await _register(api_client, "b@example.com"))["accessToken"]
    # Capacity 1 and no refill: A's second call is denied, B's first is not.
    rate_limiter.limits[ROUTE_CLASS_AUTH] = Limit(capacity=1, refill_per_second=0.0)
    rate_limiter.reset()

    # Exhaust A's bucket with authenticated reads.
    await api_client.get("/api/v1/auth/me", headers=_auth(a_token))
    throttled = await api_client.get("/api/v1/auth/me", headers=_auth(a_token))
    assert throttled.status_code == 429, throttled.text

    # B is a different key and still answers.
    resp = await api_client.get("/api/v1/auth/me", headers=_auth(b_token))
    assert resp.status_code == 200, resp.text


async def test_health_probes_are_never_rate_limited(
    api_client, rate_limiter: Any
) -> None:
    """A load balancer hammering /health must never get a 429 from the app."""
    from app.core.ratelimit import ROUTE_CLASS_READ, Limit

    rate_limiter.limits[ROUTE_CLASS_READ] = Limit(capacity=1, refill_per_second=0.001)
    for _ in range(5):
        resp = await api_client.get("/health")
        assert resp.status_code == 200, resp.text
        resp = await api_client.get("/health/ready")
        assert resp.status_code == 200, resp.text


async def test_rate_limit_can_be_turned_off(api_client, rate_limiter: Any) -> None:
    """The documented escape hatch: ``RATE_LIMIT_ENABLED=false``."""
    from app.core.ratelimit import Limit, ROUTE_CLASS_READ

    rate_limiter.limits[ROUTE_CLASS_READ] = Limit(capacity=1, refill_per_second=0.0)
    assert (await api_client.get("/api/v1/campaigns")).status_code == 200
    assert (await api_client.get("/api/v1/campaigns")).status_code == 429

    rate_limiter.enabled = False
    for index in range(25):
        resp = await api_client.get("/api/v1/campaigns")
        assert resp.status_code == 200, f"{index} -> {resp.text}"


def test_the_in_process_limit_documents_its_worker_limitation() -> None:
    """A limit that silently multiplies with replicas is worse than none.

    The module docstring is the contract an operator reads; this asserts the two
    facts it promises are actually wired: the buckets are per instance and the
    switch that disables them is honoured by the middleware.
    """
    from app.config import Settings
    from app.core.ratelimit import RateLimitMiddleware, build_limiter

    limiter = build_limiter(Settings(rate_limit_enabled=True))
    assert limiter.enabled is True
    assert limiter.max_keys > 0, "the bucket dict must stay bounded"

    disabled = build_limiter(Settings(rate_limit_enabled=False))
    assert disabled.enabled is False
    assert RateLimitMiddleware(lambda *a, **k: None, disabled).limiter.enabled is False

    import inspect

    from app.core import ratelimit

    doc = inspect.getdoc(ratelimit) or ""
    assert "per worker" in doc or "per-process" in doc


# --------------------------------------------------------------- the audit log --


async def test_audit_is_owner_only_and_workspace_scoped(
    api_client, seeded_db: AsyncSession
) -> None:
    owner = await _register(api_client, "audit-owner@example.com", "Audit Co")
    business_id = owner["workspaces"][0]["businessId"]
    headers = _auth(owner["accessToken"])

    seeded_db.add(
        AuditLog(
            id="aud_owner",
            actor=owner["user"]["id"],
            action="recommendation.approved",
            entity_type="recommendation",
            entity_id="rec_1",
            business_id=business_id,
            before={"budget": 100},
            after={"budget": 250},
            result={"note": "ok"},
        )
    )
    seeded_db.add(
        AuditLog(
            id="aud_other",
            actor="usr_stranger",
            action="recommendation.executed",
            entity_type="recommendation",
            entity_id="rec_2",
            business_id="biz_someone_else",
            result={"api_key": "should-never-be-served"},
        )
    )
    await seeded_db.commit()

    resp = await api_client.get("/api/v1/audit", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    ids = {entry["id"] for entry in body["entries"]}
    assert ids == {"aud_owner"}, "another tenant's rows must not appear"
    assert body["total"] == 1

    entry = body["entries"][0]
    assert entry["actor"] == owner["user"]["id"], "a real user id, not 'user'"
    assert entry["action"] == "recommendation.approved"
    assert entry["before"] == {"budget": 100}
    assert entry["after"] == {"budget": 250}

    # Filters.
    resp = await api_client.get("/api/v1/audit?entity_id=rec_2", headers=headers)
    assert resp.json()["entries"] == []
    resp = await api_client.get(
        f"/api/v1/audit?actor={owner['user']['id']}", headers=headers
    )
    assert len(resp.json()["entries"]) == 1
    resp = await api_client.get(
        "/api/v1/audit?action=recommendation.executed", headers=headers
    )
    assert resp.json()["entries"] == []

    # A viewer may not read the audit log.
    viewer = await _make_member(
        seeded_db, "audit-viewer@example.com", business_id, "viewer"
    )
    resp = await api_client.get("/api/v1/audit", headers=_auth(viewer))
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["detail"]["required_role"] == "owner"


async def test_audit_payloads_are_redacted(
    api_client, seeded_db: AsyncSession
) -> None:
    owner = await _register(api_client, "audit-redact@example.com", "Redact Co")
    business_id = owner["workspaces"][0]["businessId"]
    seeded_db.add(
        AuditLog(
            id="aud_secret",
            actor="usr_1",
            action="connector.updated",
            entity_type="connector",
            business_id=business_id,
            after={"api_key": "tvly-realkey", "nested": [{"access_token": "abc123"}]},
        )
    )
    await seeded_db.commit()

    resp = await api_client.get("/api/v1/audit", headers=_auth(owner["accessToken"]))
    assert resp.status_code == 200
    assert "tvly-realkey" not in resp.text
    assert "abc123" not in resp.text
    entry = resp.json()["entries"][0]
    assert entry["after"]["api_key"] == "***REDACTED***"
    assert entry["after"]["nested"][0]["access_token"] == "***REDACTED***"


# --------------------------------------------------------- headers and CORS --


async def test_security_headers_are_present_on_every_response(
    api_client,
) -> None:
    resp = await api_client.get("/health")
    assert resp.status_code == 200
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert resp.headers["content-security-policy"] == (
        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    )
    hsts = resp.headers["strict-transport-security"]
    assert hsts.startswith("max-age=") and "includeSubDomains" in hsts

    # ...and on an API response, not only on the probe.
    resp = await api_client.get("/api/v1/campaigns")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"


async def test_readiness_reports_each_component(api_client) -> None:
    """/health stays liveness-only; /health/ready is the dependency check."""
    liveness = await api_client.get("/health")
    assert liveness.status_code == 200
    assert liveness.json()["status"] == "ok"

    ready = await api_client.get("/health/ready")
    assert ready.status_code == 200, ready.text
    checks = ready.json()["checks"]
    assert checks["database"] == "ok"
    assert checks["encryption_key"] == "ok"


def test_cors_configuration_is_explicit_and_the_wildcard_is_refused() -> None:
    from app.config import Settings
    from app.core.security_headers import validate_cors

    assert Settings().cors_allow_methods == (
        "GET",
        "POST",
        "PATCH",
        "PUT",
        "DELETE",
        "OPTIONS",
    )
    assert "*" not in Settings().cors_allow_headers
    assert validate_cors(Settings(cors_origins=("https://app.example",))) == []

    with pytest.raises(ValueError) as excinfo:
        validate_cors(Settings(cors_origins=("*", "http://localhost:3000")))
    assert "CORS_ORIGINS" in str(excinfo.value)


# ------------------------------------------------------------------ utilities ---


def asyncio_run(coro: Any) -> Any:
    """Run one coroutine to completion. A helper so a sync test can drive the bucket."""
    import asyncio

    return asyncio.run(coro)