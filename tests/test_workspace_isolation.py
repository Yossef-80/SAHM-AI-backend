"""P0 workspace isolation: two tenants, no bleed.

Before P0 every router resolved "the first business in the table", so two
clients could not see different data because there was no way to be a second
client. These tests seed two workspaces and assert that naming one in
``X-Workspace-Id`` is the only thing that decides what a request can read.

Two rules are asserted throughout, not just the status code:

* A row that exists but belongs to another workspace is **404**, never 403. A
  403 confirms to a stranger that the id is real.
* Every failure uses the one envelope: ``{error: {code, message, detail,
  requestId}}``.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import WORKSPACE_HEADER, resolve_workspace
from app.core.errors import WorkspaceRequiredError
from app.core.logging import (
    ContextFilter,
    JsonFormatter,
    current_business_id,
    current_workspace_id,
    get_logger,
)
from app.core.security import DEV_FERNET_KEY
from app.core.tools import protected
from app.core.tools.base import ToolContext
from app.db.models import (
    AgentTask,
    Business,
    Campaign,
    Conversation,
    Creative,
    Insight,
    Recommendation,
)

_logger = get_logger(__name__)

#: A throwaway Fernet key for the production-boot tests. P4 made it a hard boot
#: requirement in prod, so a "prod" test has to look like a real prod process.
#: Reused from the module so it is a *valid* Fernet key by construction.
_TEST_FERNET_KEY = DEV_FERNET_KEY.decode()

WS_A = "ws_test"  # what ``seeded_db`` creates
WS_B = "ws_bravo"
BIZ_A = "biz_test"
BIZ_B = "biz_bravo"


def _headers(business_id: str) -> dict[str, str]:
    return {WORKSPACE_HEADER: business_id}


class _FakeRequest:
    """Minimal stand-in for ``Request``: the resolver only reads headers."""

    def __init__(self, business_id: str | None) -> None:
        self.headers: dict[str, str] = (
            {} if business_id is None else {WORKSPACE_HEADER: business_id}
        )


def _settings() -> Any:
    from app.config import get_settings

    return get_settings()


def _assert_envelope_404(response: Any) -> None:
    """A cross-tenant read is 404 in the standard envelope, not a bare 404."""
    assert response.status_code == 404, response.text
    body = response.json()
    assert "detail" not in body, "FastAPI's default shape leaked through"
    error = body["error"]
    assert error["code"] == "not_found"
    assert error["message"]
    assert error["requestId"]
    assert isinstance(error["detail"], dict)


# ------------------------------------------------------------------ fixtures --


@pytest.fixture()
async def two_workspaces(seeded_db: AsyncSession) -> AsyncSession:
    """``seeded_db`` is workspace A; add a second, disjoint business.

    Each side gets one campaign, conversation, creative, insight,
    recommendation and task, so every "can A reach B's X?" test has a target.
    """
    db = seeded_db
    db.add(Business(id=BIZ_B, workspace_id=WS_B, name="Bravo Rentals", industry="SaaS"))

    for prefix, business_id, label in (
        ("a", BIZ_A, "Alpha"),
        ("b", BIZ_B, "Bravo"),
    ):
        db.add(
            Campaign(
                id=f"cmp_{prefix}",
                business_id=business_id,
                name=f"{label} Spring Campaign",
                status="active",
                objective="OUTCOME_SALES",
                budget=1000.0,
                currency="EGP",
            )
        )
        db.add(
            Conversation(
                id=f"conv_{prefix}", business_id=business_id, title=f"{label} chat"
            )
        )
        db.add(
            Creative(
                id=f"cr_{prefix}",
                business_id=business_id,
                campaign_id=f"cmp_{prefix}",
                url=f"https://cdn.example.test/{prefix}.png",
                prompt=f"{label} hero shot",
            )
        )
        db.add(
            Insight(
                id=f"ins_{prefix}",
                campaign_id=f"cmp_{prefix}",
                kind="ctr_drop",
                title=f"{label} CTR fell",
                detail="Frequency fatigue.",
                severity="warning",
            )
        )
        db.add(
            Recommendation(
                id=f"rec_{prefix}",
                campaign_id=f"cmp_{prefix}",
                # P1: the owning business, denormalised. Before this column the
                # tenant was reached through the campaign join.
                workspace_id=business_id,
                action_type="pause_ad",
                title=f"{label}: pause the reel",
                rationale="CTR fell to 0.4%.",
                payload={"status": "PAUSED"},
                status="pending",
                requires_approval=True,
            )
        )
        db.add(
            AgentTask(
                id=f"task_{prefix}",
                business_id=business_id,
                task_name="ad_copy",
                persona_id="layla",
                status="queued",
            )
        )
    await db.commit()
    return db


# ------------------------------------------------ the header picks the tenant --


async def test_header_selects_the_workspace(two_workspaces: AsyncSession) -> None:
    """No header keeps today's single-tenant fallback; the header overrides it."""
    default = await resolve_workspace(_FakeRequest(None), two_workspaces, _settings())
    assert default.business_id == BIZ_A, "the oldest business, unchanged behaviour"

    alpha = await resolve_workspace(_FakeRequest(BIZ_A), two_workspaces, _settings())
    assert (alpha.business_id, alpha.workspace_id) == (BIZ_A, WS_A)

    bravo = await resolve_workspace(_FakeRequest(BIZ_B), two_workspaces, _settings())
    assert (bravo.business_id, bravo.workspace_id) == (BIZ_B, WS_B)


async def test_unknown_workspace_header_is_404_and_never_falls_through(
    api_client, two_workspaces: AsyncSession
) -> None:
    """A header naming a row that does not exist must not silently downgrade.

    Falling back to "the oldest business" here would hand a misconfigured client
    another tenant's data, which is the leak this dependency exists to remove.
    """
    resp = await api_client.get("/api/v1/campaigns", headers=_headers("biz_nope"))
    _assert_envelope_404(resp)


async def test_prod_without_the_header_is_rejected_when_required(
    two_workspaces: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict tenancy: an unnamed request is refused, never defaulted."""
    settings = _settings()
    monkeypatch.setattr(settings, "environment", "prod")
    monkeypatch.setattr(settings, "require_workspace_header", True)

    with pytest.raises(WorkspaceRequiredError) as excinfo:
        await resolve_workspace(_FakeRequest(None), two_workspaces, settings)
    assert excinfo.value.code == "workspace_required"
    assert excinfo.value.http_status == 400

    # With the header the same request resolves normally, even in prod.
    resolved = await resolve_workspace(_FakeRequest(BIZ_A), two_workspaces, settings)
    assert resolved.business_id == BIZ_A


# ----------------------------------------------- cross-tenant reads are 404 --


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/campaigns/cmp_b",
        "/api/v1/campaigns/cmp_b/insights",
        "/api/v1/assistant/conversations/conv_b",
        "/api/v1/recommendations/rec_b",
        "/api/v1/tasks/task_b",
    ],
)
async def test_workspace_a_cannot_read_workspace_b(
    api_client, two_workspaces: AsyncSession, path: str
) -> None:
    resp = await api_client.get(path, headers=_headers(BIZ_A))
    _assert_envelope_404(resp)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/v1/tasks/task_b/cancel"),
        ("get", "/api/v1/tasks/task_b/stream"),
    ],
)
async def test_workspace_a_cannot_reach_workspace_b_task_actions(
    api_client, two_workspaces: AsyncSession, method: str, path: str
) -> None:
    """The cancel/stream siblings are scoped exactly like the read."""
    resp = await getattr(api_client, method)(path, headers=_headers(BIZ_A))
    _assert_envelope_404(resp)


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/campaigns/cmp_a",
        "/api/v1/assistant/conversations/conv_a",
        "/api/v1/recommendations/rec_a",
        "/api/v1/tasks/task_a",
    ],
)
async def test_workspace_a_can_still_read_its_own_rows(
    api_client, two_workspaces: AsyncSession, path: str
) -> None:
    """The 404s above must be isolation, not breakage."""
    resp = await api_client.get(path, headers=_headers(BIZ_A))
    assert resp.status_code == 200, resp.text


async def test_creatives_body_campaign_id_is_checked(
    api_client, two_workspaces: AsyncSession
) -> None:
    """A campaign id in the request body cannot point at another tenant."""
    resp = await api_client.post(
        "/api/v1/creatives/generate",
        headers=_headers(BIZ_A),
        json={"prompt": "a summer offer", "campaign_id": "cmp_b"},
    )
    _assert_envelope_404(resp)


async def test_recommendation_with_null_campaign_is_unattributable(
    api_client, two_workspaces: AsyncSession
) -> None:
    """A recommendation that names no owner belongs to no workspace.

    Since P1 ``recommendations.workspace_id`` carries the owning business
    directly. This row leaves it NULL -- as a recommendation raised outside any
    campaign does -- so it matches nobody's list and is a 404 by id for
    everyone. Failing closed is the property under test, and it now holds because
    of the column rather than because of join semantics: NULL never equals a
    workspace id.
    """
    two_workspaces.add(
        Recommendation(
            id="rec_orphan",
            campaign_id=None,
            action_type="refresh_creative",
            title="No campaign",
            rationale="Raised outside a campaign.",
            payload={},
            status="pending",
            requires_approval=True,
        )
    )
    await two_workspaces.commit()

    resp = await api_client.get("/api/v1/recommendations", headers=_headers(BIZ_A))
    ids = [r["id"] for r in resp.json()]
    assert "rec_orphan" not in ids
    assert "rec_a" in ids
    _assert_envelope_404(
        await api_client.get("/api/v1/recommendations/rec_orphan", headers=_headers(BIZ_A))
    )


# --------------------------------------------- collection endpoints are scoped --


async def test_list_campaigns_returns_only_the_workspaces(
    api_client, two_workspaces: AsyncSession
) -> None:
    for business_id, expected, forbidden in (
        (BIZ_A, "cmp_a", "cmp_b"),
        (BIZ_B, "cmp_b", "cmp_a"),
    ):
        resp = await api_client.get("/api/v1/campaigns", headers=_headers(business_id))
        assert resp.status_code == 200
        body = resp.json()
        ids = [c["id"] for c in body["campaigns"]]
        assert expected in ids
        assert forbidden not in ids
        assert body["total"] == 1


async def test_list_recommendations_respects_the_workspace(
    api_client, two_workspaces: AsyncSession
) -> None:
    """D3: this endpoint used to filter only by status, with no tenant filter."""
    resp = await api_client.get("/api/v1/recommendations", headers=_headers(BIZ_A))
    assert resp.status_code == 200
    assert [r["id"] for r in resp.json()] == ["rec_a"]


async def test_list_creatives_respects_the_workspace(
    api_client, two_workspaces: AsyncSession
) -> None:
    resp = await api_client.get("/api/v1/creatives", headers=_headers(BIZ_A))
    assert resp.status_code == 200
    ids = [c["id"] for c in resp.json()]
    assert "cr_a" in ids and "cr_b" not in ids


async def test_list_conversations_respects_the_workspace(
    api_client, two_workspaces: AsyncSession
) -> None:
    resp = await api_client.get("/api/v1/assistant/conversations", headers=_headers(BIZ_B))
    assert resp.status_code == 200
    ids = [c["id"] for c in resp.json()]
    assert "conv_b" in ids and "conv_a" not in ids


async def test_meta_status_is_scoped_to_the_workspace(
    api_client, two_workspaces: AsyncSession
) -> None:
    """``GET /integrations/meta/status`` must report the caller's connection.

    Bravo has a connected Meta grant and Alpha has none, so if this endpoint
    still resolved "the first business" Alpha would report Bravo's token state.
    """
    from datetime import datetime, timezone

    from app.db.models import Integration

    two_workspaces.add(
        Integration(
            id="int_bravo",
            business_id=BIZ_B,
            provider="meta",
            account_id="act_999",
            account_name="Bravo Ads",
            status="connected",
            access_token_enc="ciphertext",
            scopes=["ads_read"],
            last_synced_at=datetime.now(timezone.utc),
        )
    )
    await two_workspaces.commit()

    alpha = await api_client.get(
        "/api/v1/integrations/meta/status", headers=_headers(BIZ_A)
    )
    bravo = await api_client.get(
        "/api/v1/integrations/meta/status", headers=_headers(BIZ_B)
    )
    assert alpha.json()["connected"] is False
    assert bravo.json()["connected"] is True
    assert bravo.json()["adAccountId"] == "act_999"


# --------------------------------------------------------------------- D2 -----


async def test_execution_resolves_the_owning_business_not_the_campaign_id(
    two_workspaces: AsyncSession,
) -> None:
    """D2 regression: the campaign id used to land in ``business_id``.

    The old line was ``ToolContext(workspace_id="default",
    business_id=row.campaign_id)``, so every tool that resolves an integration
    or brand memory by ``ctx.business_id`` during execution looked up a business
    id that does not exist. This fails on the old code and passes on the new.
    """
    from app.core.approvals import service as approvals
    from app.schemas.recommendations import ApproveRecommendationRequest

    seen: dict[str, str] = {}
    original = protected.apply_change.handler

    async def _capture(payload: Any, ctx: ToolContext) -> Any:
        seen["workspace_id"] = ctx.workspace_id
        seen["business_id"] = ctx.business_id
        return await original(payload, ctx)

    protected.apply_change.handler = _capture  # type: ignore[method-assign]
    try:
        await approvals.approve_recommendation(
            two_workspaces,
            "rec_b",
            ApproveRecommendationRequest(confirm=True),
            actor="nour",
            request_id="req-d2",
        )
    finally:
        protected.apply_change.handler = original  # type: ignore[method-assign]

    assert seen["business_id"] == BIZ_B
    assert seen["business_id"] != "cmp_b", "the campaign id leaked into business_id"
    assert seen["workspace_id"] == WS_B
    assert seen["workspace_id"] != "default"


# --------------------------------------------------------- uniform envelopes --


async def test_validation_failure_uses_the_standard_envelope(api_client) -> None:
    """FastAPI's own 422 must be reshaped, not left in its default shape."""
    resp = await api_client.post("/api/v1/campaigns", json={"budget": "not-a-number"})
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert "detail" not in body, "FastAPI's default shape leaked through"
    error = body["error"]
    assert error["code"] == "validation_failed"
    assert error["message"]
    assert error["requestId"]
    assert error["detail"]["issues"], "the failing fields must be reported"


async def test_missing_route_uses_the_standard_envelope(api_client) -> None:
    """A bare framework 404 (no such route) is enveloped too."""
    resp = await api_client.get("/api/v1/no-such-route")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "not_found"
    assert error["requestId"]


async def test_typed_not_found_uses_the_standard_envelope(api_client) -> None:
    resp = await api_client.get("/api/v1/campaigns/nope")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


# ---------------------------------------------------- logging (D7) + config --


async def test_bind_workspace_puts_the_tenant_on_log_lines(
    two_workspaces: AsyncSession,
) -> None:
    """D7: ``bind_workspace`` had zero callers, so tenant id was always null.

    Asserted on the real formatted JSON line, not on a mock, so this covers the
    filter and the formatter as well as the binding.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(ContextFilter())
    handler.setFormatter(JsonFormatter())
    _logger.logger.addHandler(handler)
    try:
        await resolve_workspace(_FakeRequest(BIZ_B), two_workspaces, _settings())
        assert current_workspace_id() == WS_B
        assert current_business_id() == BIZ_B
        _logger.info("inside_request")
    finally:
        _logger.logger.removeHandler(handler)

    payload = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert payload["msg"] == "inside_request"
    assert payload["workspace_id"] == WS_B
    assert payload["business_id"] == BIZ_B


async def test_embedding_column_derives_its_dimension_from_settings() -> None:
    """D6: ``EMBEDDING_DIM`` must reach the column, not just the provider."""
    from app.db.models import EMBEDDING_DIM, BrandMemoryItem

    assert EMBEDDING_DIM == _settings().embedding_dim
    assert BrandMemoryItem.__table__.c.embedding.type.dim == EMBEDDING_DIM


# ------------------------------------------------------------- boot + probes --


async def test_boot_creates_the_schema_outside_production(api_client) -> None:
    """Dev/test still get ``create_all`` (the lifespan ran over the test DB)."""
    resp = await api_client.get("/api/v1/campaigns")
    assert resp.status_code == 200, resp.text


async def test_boot_skips_ddl_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """In prod the schema comes from Alembic; booting must not issue DDL.

    ``token_encryption_key`` is set because P4 added a second production boot
    requirement: a prod process without one refuses to start at all
    (``app/core/security.py::verify_runtime_security``). Without it this test
    would now be asserting "boot raises" rather than "boot issues no DDL".
    """
    import app.main as main

    issued: list[str] = []

    async def _spy() -> None:  # pragma: no cover - must not run in prod
        issued.append("create_all")

    monkeypatch.setattr(main, "create_all", _spy)
    monkeypatch.setattr(_settings(), "environment", "prod")
    monkeypatch.setattr(_settings(), "token_encryption_key", _TEST_FERNET_KEY)

    async with main.lifespan(main.app):
        pass

    assert issued == [], "production boot must not create tables"


async def test_production_boot_refuses_without_an_encryption_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P4/D5: prod must not boot with the dev Fernet key published in the repo."""
    import app.main as main
    from app.core.errors import EncryptionKeyMissingError

    monkeypatch.setattr(_settings(), "environment", "prod")
    monkeypatch.setattr(_settings(), "token_encryption_key", None)

    with pytest.raises(EncryptionKeyMissingError):
        async with main.lifespan(main.app):
            pass


async def test_boot_creates_the_schema_in_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dev path is unchanged: create_all still runs outside prod."""
    import app.main as main

    issued: list[str] = []

    async def _spy() -> None:
        issued.append("create_all")

    monkeypatch.setattr(main, "create_all", _spy)
    monkeypatch.setattr(_settings(), "environment", "dev")

    async with main.lifespan(main.app):
        pass

    assert issued == ["create_all"]


async def test_readiness_reports_ready_against_a_live_database(api_client) -> None:
    resp = await api_client.get("/health/ready")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "ready"


async def test_readiness_returns_503_in_the_standard_envelope(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the database unreachable, readiness must fail loudly (D4)."""
    import app.main as main

    async def _down() -> tuple[bool, str]:
        return False, "OperationalError: connection refused"

    monkeypatch.setattr(main, "_check_database", _down)
    resp = await api_client.get("/health/ready")
    assert resp.status_code == 503
    error = resp.json()["error"]
    assert error["code"] == "not_ready"
    assert error["requestId"]
    assert "connection refused" in error["detail"]["checks"]["database"]


async def test_liveness_stays_cheap_and_does_not_touch_the_database(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/health`` must stay dependency-free so a DB blip cannot restart a pod."""
    import app.main as main

    async def _boom() -> tuple[bool, str]:  # pragma: no cover - must not run
        raise AssertionError("/health must not check the database")

    monkeypatch.setattr(main, "_check_database", _boom)
    resp = await api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"