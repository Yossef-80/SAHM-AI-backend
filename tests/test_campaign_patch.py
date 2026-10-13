"""PATCH /campaigns/{id} (Phase 2.3d).

The important cases are the negative ones: a status change must reach Meta
*before* anything local is written, so a Meta failure can never leave the UI
claiming "paused" while the campaign keeps spending.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.marketing import meta_ads
from app.api.deps import WORKSPACE_HEADER
from app.db.models import Campaign, Recommendation

pytestmark = pytest.mark.anyio

BIZ = "biz_test"
HEADERS = {WORKSPACE_HEADER: BIZ}


class _RecordingGraph:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail = fail

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((path, payload))
        if self.fail:
            raise meta_ads.MetaGraphError(
                "Meta rejected the request (Meta error 100).",
                retry_safe=True,
                status_code=400,
            )
        return {"success": True}

    async def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}


async def _campaign(
    db: AsyncSession, *, meta_campaign_id: str | None, campaign_id: str = "cmp_patch"
) -> Campaign:
    row = Campaign(
        id=campaign_id,
        business_id=BIZ,
        name="Patchable campaign",
        status="paused",
        objective="sales",
        channel="meta_ads",
        budget=100.0,
        currency="EGP",
        meta_campaign_id=meta_campaign_id,
    )
    db.add(row)
    await db.commit()
    return row


async def test_patch_updates_local_fields_for_a_campaign_not_on_meta(
    api_client: Any, seeded_db: AsyncSession
) -> None:
    await _campaign(seeded_db, meta_campaign_id=None)

    resp = await api_client.patch(
        "/api/v1/campaigns/cmp_patch",
        json={"name": "Renamed", "budget": 250.0, "notes": "hello"},
        headers=HEADERS,
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "Renamed"
    assert body["budget"] == 250.0

    row = await seeded_db.get(Campaign, "cmp_patch")
    await seeded_db.refresh(row)
    assert row.name == "Renamed"
    assert row.budget == 250.0
    assert row.notes == "hello"


async def test_patch_status_is_mirrored_to_meta_before_the_local_write(
    api_client: Any,
    seeded_db: AsyncSession,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _campaign(seeded_db, meta_campaign_id="meta_c_1")
    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "act_987"
    fake = _RecordingGraph()
    # The route imports the client into its own namespace.
    monkeypatch.setattr("app.api.campaigns.HttpMetaGraphClient", lambda **kw: fake)

    resp = await api_client.patch(
        "/api/v1/campaigns/cmp_patch", json={"status": "active"}, headers=HEADERS
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "active"
    assert fake.calls == [("meta_c_1", {"status": "ACTIVE"})]

    row = await seeded_db.get(Campaign, "cmp_patch")
    await seeded_db.refresh(row)
    assert row.status == "active"


async def test_patch_leaves_the_row_untouched_when_meta_rejects_the_change(
    api_client: Any,
    seeded_db: AsyncSession,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed mirror must not produce a locally-paused, remotely-spending campaign."""
    await _campaign(seeded_db, meta_campaign_id="meta_c_1")
    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "act_987"
    fake = _RecordingGraph(fail=True)
    monkeypatch.setattr("app.api.campaigns.HttpMetaGraphClient", lambda **kw: fake)

    resp = await api_client.patch(
        "/api/v1/campaigns/cmp_patch",
        json={"status": "active", "name": "Should not stick"},
        headers=HEADERS,
    )

    assert resp.status_code == 502, resp.text
    assert len(fake.calls) == 1

    row = await seeded_db.get(Campaign, "cmp_patch")
    await seeded_db.refresh(row)
    assert row.status == "paused"  # unchanged
    assert row.name == "Patchable campaign"  # unchanged too


async def test_patch_refuses_status_change_when_meta_is_linked_but_disconnected(
    api_client: Any,
    seeded_db: AsyncSession,
    test_settings: Any,
) -> None:
    await _campaign(seeded_db, meta_campaign_id="meta_c_1")
    test_settings.meta_access_token = None
    test_settings.meta_ad_account_id = None

    resp = await api_client.patch(
        "/api/v1/campaigns/cmp_patch", json={"status": "active"}, headers=HEADERS
    )

    assert resp.status_code == 409, resp.text
    row = await seeded_db.get(Campaign, "cmp_patch")
    await seeded_db.refresh(row)
    assert row.status == "paused"


async def test_patch_rejects_a_status_meta_does_not_understand(
    api_client: Any, seeded_db: AsyncSession
) -> None:
    await _campaign(seeded_db, meta_campaign_id=None)

    resp = await api_client.patch(
        "/api/v1/campaigns/cmp_patch", json={"status": "exploded"}, headers=HEADERS
    )

    assert resp.status_code == 422, resp.text


async def test_patch_rejects_a_negative_budget(
    api_client: Any, seeded_db: AsyncSession
) -> None:
    await _campaign(seeded_db, meta_campaign_id=None)

    resp = await api_client.patch(
        "/api/v1/campaigns/cmp_patch", json={"budget": -5}, headers=HEADERS
    )

    assert resp.status_code == 422, resp.text


async def test_patch_404s_for_another_tenants_campaign(
    api_client: Any, seeded_db: AsyncSession
) -> None:
    await _campaign(seeded_db, meta_campaign_id=None)

    resp = await api_client.patch(
        "/api/v1/campaigns/cmp_patch",
        json={"name": "stolen"},
        headers={WORKSPACE_HEADER: "biz_other"},
    )

    assert resp.status_code == 404, resp.text
    row = await seeded_db.get(Campaign, "cmp_patch")
    await seeded_db.refresh(row)
    assert row.name == "Patchable campaign"
