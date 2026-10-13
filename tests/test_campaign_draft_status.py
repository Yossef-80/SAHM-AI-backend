"""Delivery state is only changeable once the campaign exists on Meta.

The PATCH endpoint accepts exactly the two statuses Meta understands, and it
refuses a delivery-state change outright when there is no Meta campaign to
mirror it to. Before that guard, a draft could be flipped to ``active`` locally
and the UI would claim delivery while nothing existed on Meta at all.

The campaign-detail header is state-aware as a result: a draft offers "continue
in the wizard", and only a campaign that exists on Meta offers pause / resume.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import WORKSPACE_HEADER
from app.db.models import Campaign

pytestmark = pytest.mark.anyio

BIZ = "biz_test"
HEADERS = {WORKSPACE_HEADER: BIZ}


async def _campaign(
    db: AsyncSession,
    *,
    status: str = "draft",
    meta_campaign_id: str | None = None,
    campaign_id: str = "cmp_draft",
) -> Campaign:
    row = Campaign(
        id=campaign_id,
        business_id=BIZ,
        name="عمرة موتور",
        status=status,
        objective="awareness",
        channel="meta_ads",
        budget=150.0,
        currency="EGP",
        meta_campaign_id=meta_campaign_id,
    )
    db.add(row)
    await db.commit()
    return row


@pytest.mark.parametrize("wanted", ["published", "live", "running", "scheduled", "ready"])
async def test_only_meta_statuswords_are_accepted(
    api_client: Any, seeded_db: AsyncSession, wanted: str
) -> None:
    await _campaign(seeded_db)

    resp = await api_client.patch(
        "/api/v1/campaigns/cmp_draft", json={"status": wanted}, headers=HEADERS
    )

    assert resp.status_code == 422, resp.text
    body = resp.json()["error"]
    assert body["code"] == "validation_failed"
    assert "not a status Meta understands" in body["message"]
    # A rejected request leaves the row alone.
    row = await seeded_db.get(Campaign, "cmp_draft")
    await seeded_db.refresh(row)
    assert row.status == "draft"


async def test_a_draft_cannot_be_marked_active(
    api_client: Any, seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trap this guard exists for: `active` with no Meta campaign behind it."""
    await _campaign(seeded_db, status="draft", meta_campaign_id=None)

    class _NeverCalled:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("a draft must not reach Meta from here")

    monkeypatch.setattr("app.api.campaigns.HttpMetaGraphClient", _NeverCalled)

    resp = await api_client.patch(
        "/api/v1/campaigns/cmp_draft", json={"status": "active"}, headers=HEADERS
    )

    assert resp.status_code == 409, resp.text
    assert "has not been published to Meta yet" in resp.json()["error"]["message"]

    row = await seeded_db.get(Campaign, "cmp_draft")
    await seeded_db.refresh(row)
    assert row.status == "draft"
    assert row.meta_campaign_id is None


async def test_a_paused_campaign_can_go_back_to_active(
    api_client: Any, seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The only transition the header's pause/resume button can produce."""
    await _campaign(seeded_db, status="paused", meta_campaign_id="meta_123")

    calls: list[tuple[str, dict[str, Any]]] = []

    class _Recording:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            calls.append((path, payload))
            return {"success": True}

        async def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
            return {"data": []}

    async def _connection(db: Any, business_id: str, settings: Any) -> Any:
        class _Connection:
            access_token = "token"
            ad_account_id = "123"
            graph_base_url = "https://graph.facebook.com"

        return _Connection()

    monkeypatch.setattr("app.api.campaigns.HttpMetaGraphClient", _Recording)
    monkeypatch.setattr("app.api.campaigns.resolve_meta_connection", _connection)

    resp = await api_client.patch(
        "/api/v1/campaigns/cmp_draft", json={"status": "active"}, headers=HEADERS
    )

    assert resp.status_code == 200, resp.text
    assert calls == [("meta_123", {"status": "ACTIVE"})]
