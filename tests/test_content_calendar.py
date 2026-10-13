"""Offline tests for Omar's standalone calendar flow and persisted API contract."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.agents.omar import service as omar_service
from app.ai.agents.omar.schema import ContentCalendarOutput
from app.ai.agents.omar.schema import ContentItem as OmarContentItem
from app.ai.errors import ProviderRequestError
from app.db.models import Business, Campaign
from app.db.models import ContentItem as ContentItemRow


def _calendar_draft(weeks: int, posts_per_week: int, channels: list[str]) -> ContentCalendarOutput:
    items = []
    for week in range(1, weeks + 1):
        for position in range(posts_per_week):
            items.append(
                OmarContentItem(
                    week=week,
                    channel=channels[position % len(channels)],
                    format="post",
                    topic=f"Coffee topic {week}-{position + 1}",
                    title=f"Coffee title {week}-{position + 1}",
                    hook=f"A grounded hook for week {week}.",
                    caption=f"A useful caption for week {week}.",
                    hashtags=["coffee", "cairo"],
                    pillar="Education",
                    goal="engagement",
                    cta="Explore the details in the brief.",
                )
            )
    return ContentCalendarOutput(items=items, summary="An editable monthly draft.")


class _FakeStructuredRunnable:
    def __init__(self, result: ContentCalendarOutput) -> None:
        self.result = result
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> ContentCalendarOutput:
        self.messages = messages
        return self.result


class _FakeLLMHandle:
    def __init__(self, result: ContentCalendarOutput) -> None:
        self.result = result
        self.schema: type[Any] | None = None
        self.runnable = _FakeStructuredRunnable(result)

    def structured(self, schema: type[Any]) -> _FakeStructuredRunnable:
        self.schema = schema
        return self.runnable


async def test_omar_calendar_service_is_standalone_and_validates_monthly_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _FakeLLMHandle(_calendar_draft(4, 2, ["instagram"]))
    requested: list[dict[str, Any]] = []
    monkeypatch.setattr(
        omar_service,
        "get_ai_settings",
        lambda: SimpleNamespace(llm_provider="mock"),
    )
    monkeypatch.setattr(
        omar_service,
        "get_llm",
        lambda **kwargs: requested.append(kwargs) or handle,
    )

    result = await omar_service.generate_content_calendar(
        calendar_input={
            "month": "2026-02",
            "posts_per_week": 2,
            "channels": ["instagram"],
            "brief": "Focus on practical coffee education.",
            "language": "ar",
        },
        business_profile={"name": "Saved Cairo Coffee", "industry": "F&B"},
    )

    assert requested == [{"provider": "mock"}]
    assert handle.schema is ContentCalendarOutput
    assert len(result.items) == 8
    assert "Arabic" in handle.runnable.messages[1].content
    assert '"name": "Saved Cairo Coffee"' in handle.runnable.messages[1].content
    assert "does not require an earlier" in handle.runnable.messages[0].content
    assert "publish" in handle.runnable.messages[0].content
    assert "Explore the details in the brief." in result.items[0].caption


async def test_omar_rejects_incomplete_calendar_without_saving_or_network_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _FakeLLMHandle(_calendar_draft(1, 1, ["instagram"]))
    monkeypatch.setattr(
        omar_service,
        "get_ai_settings",
        lambda: SimpleNamespace(llm_provider="mock"),
    )
    monkeypatch.setattr(omar_service, "get_llm", lambda **_kwargs: handle)

    with pytest.raises(ProviderRequestError, match="incomplete"):
        await omar_service.generate_content_calendar(
            calendar_input={
                "month": "2026-02",
                "posts_per_week": 2,
                "channels": ["instagram"],
            },
            business_profile={"name": "Saved business"},
        )


async def test_generate_calendar_persists_and_lists_the_requested_month(
    api_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    async def fake_generate(**kwargs: Any) -> ContentCalendarOutput:
        calls.update(kwargs)
        return _calendar_draft(5, 2, ["instagram", "facebook"])

    monkeypatch.setattr(omar_service, "generate_content_calendar", fake_generate)
    response = await api_client.post(
        "/api/v1/content/generate",
        json={
            "month": "2026-07",
            "postsPerWeek": 2,
            "channels": ["instagram", "facebook"],
            "campaignId": None,
            "brief": "Use only facts from the saved profile.",
        },
    )

    assert response.status_code == 200, response.text
    generated = response.json()
    assert len(generated) == 10
    assert calls["business_profile"]["name"] == "Cairo Coffee Co"
    assert calls["business_profile"]["industry"] == "F&B"
    assert calls["campaign_context"] is None
    assert calls["calendar_input"]["language"] == "en"
    assert all(item["status"] == "idea" for item in generated)
    assert all(item["campaignId"] is None and item["creativeId"] is None for item in generated)
    assert all(item["date"].startswith("2026-07-") for item in generated)
    assert all(item["createdAt"] and item["updatedAt"] for item in generated)

    listed = await api_client.get("/api/v1/content", params={"month": "2026-07"})
    assert listed.status_code == 200, listed.text
    assert {item["id"] for item in listed.json()} == {item["id"] for item in generated}

    edited_id = generated[0]["id"]
    updated = await api_client.patch(
        f"/api/v1/content/{edited_id}",
        json={"title": "Edited content", "date": "2026-08-11", "time": None, "caption": None},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["title"] == "Edited content"
    assert updated.json()["date"] == "2026-08-11"
    assert updated.json()["time"] is None
    assert updated.json()["caption"] is None

    july = await api_client.get("/api/v1/content", params={"month": "2026-07"})
    august = await api_client.get("/api/v1/content", params={"month": "2026-08"})
    assert len(july.json()) == 9
    assert [item["id"] for item in august.json()] == [edited_id]

    deleted = await api_client.delete(f"/api/v1/content/{edited_id}")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"id": edited_id}
    assert (await api_client.get("/api/v1/content", params={"month": "2026-08"})).json() == []


async def test_regeneration_replaces_ideas_but_preserves_scheduled_items(
    api_client: Any,
    seeded_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded_db.add(
        Campaign(
            id="cmp_calendar",
            business_id="biz_test",
            name="Summer coffee",
            objective="engagement",
            channel="instagram",
            target_audience="The saved campaign audience.",
            notes="Keep the campaign context factual.",
        )
    )
    await seeded_db.commit()

    calls: list[dict[str, Any]] = []

    async def fake_generate(**kwargs: Any) -> ContentCalendarOutput:
        calls.append(kwargs)
        return _calendar_draft(4, 1, ["instagram"])

    monkeypatch.setattr(omar_service, "generate_content_calendar", fake_generate)
    request = {
        "month": "2026-02",
        "postsPerWeek": 1,
        "channels": ["instagram"],
        "campaignId": "cmp_calendar",
    }
    first = await api_client.post("/api/v1/content/generate", json=request)
    assert first.status_code == 200, first.text
    assert calls[0]["campaign_context"]["name"] == "Summer coffee"
    assert calls[0]["campaign_context"]["objective"] == "engagement"
    old_items = first.json()
    scheduled_id = old_items[0]["id"]
    scheduled = await api_client.patch(
        f"/api/v1/content/{scheduled_id}", json={"status": "scheduled"}
    )
    assert scheduled.status_code == 200, scheduled.text

    second = await api_client.post("/api/v1/content/generate", json=request)
    assert second.status_code == 200, second.text
    current = second.json()
    assert len(current) == 4
    assert {item["id"] for item in current}.isdisjoint({item["id"] for item in old_items})

    filtered = await api_client.get(
        "/api/v1/content", params={"month": "2026-02", "campaignId": "cmp_calendar"}
    )
    persisted = filtered.json()
    assert len(persisted) == 5
    assert any(item["id"] == scheduled_id and item["status"] == "scheduled" for item in persisted)
    assert {item["id"] for item in current}.issubset({item["id"] for item in persisted})
    assert {item["id"] for item in current}.isdisjoint({item["id"] for item in old_items})


async def test_content_items_are_isolated_by_workspace(
    api_client: Any,
    seeded_db: AsyncSession,
) -> None:
    seeded_db.add(Business(id="biz_bravo", workspace_id="ws_bravo", name="Bravo Rentals"))
    seeded_db.add(
        ContentItemRow(
            id="content_private_bravo",
            business_id="biz_bravo",
            date=date(2026, 7, 12),
            channel="instagram",
            format="post",
            title="Bravo private idea",
            hashtags=[],
            status="idea",
        )
    )
    await seeded_db.commit()

    headers = {"X-Workspace-Id": "biz_test"}
    listing = await api_client.get(
        "/api/v1/content", params={"month": "2026-07"}, headers=headers
    )
    assert listing.status_code == 200, listing.text
    assert listing.json() == []

    update = await api_client.patch(
        "/api/v1/content/content_private_bravo",
        json={"title": "Attempted cross-workspace edit"},
        headers=headers,
    )
    delete = await api_client.delete(
        "/api/v1/content/content_private_bravo", headers=headers
    )
    assert update.status_code == 404
    assert delete.status_code == 404
    assert update.json()["error"]["code"] == "not_found"
