"""Tests for Content Calendar API and Orchestrator persistence."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ContentItem
from app.ai.agents.orchestrator.brain import _persist_content_calendar_posts


@pytest.mark.anyio
async def test_content_calendar_crud_flow(api_client: AsyncClient, seeded_db: AsyncSession):
    headers = {"X-Workspace-Id": "biz_test"}

    # 1. Initially empty
    res = await api_client.get("/api/v1/content", headers=headers)
    assert res.status_code == 200
    items = res.json()
    assert items == []

    # 2. Generate calendar for 2026-11
    gen_payload = {
        "month": "2026-11",
        "postsPerWeek": 3,
        "channels": ["instagram", "facebook"],
        "brief": "Special November Launch",
    }
    res = await api_client.post("/api/v1/content/generate", json=gen_payload, headers=headers)
    assert res.status_code == 200
    generated = res.json()
    assert len(generated) > 0
    first_item = generated[0]
    assert first_item["date"].startswith("2026-11")
    assert first_item["channel"] in ["instagram", "facebook"]
    assert first_item["status"] == "scheduled"
    assert "Special November Launch" in first_item["title"]

    item_id = first_item["id"]

    # 3. List with month filter
    res = await api_client.get("/api/v1/content?month=2026-11", headers=headers)
    assert res.status_code == 200
    listed = res.json()
    assert len(listed) == len(generated)

    # 4. Patch one item
    patch_payload = {
        "title": "Updated Brand Story Reel",
        "status": "published",
        "format": "reel",
    }
    res = await api_client.patch(f"/api/v1/content/{item_id}", json=patch_payload, headers=headers)
    assert res.status_code == 200
    updated = res.json()
    assert updated["id"] == item_id
    assert updated["title"] == "Updated Brand Story Reel"
    assert updated["status"] == "published"
    assert updated["format"] == "reel"

    # 5. Delete the item
    res = await api_client.delete(f"/api/v1/content/{item_id}", headers=headers)
    assert res.status_code == 200
    assert res.json() == {"id": item_id}

    # Verify item is gone
    res = await api_client.get("/api/v1/content?month=2026-11", headers=headers)
    assert res.status_code == 200
    after_del = res.json()
    assert len(after_del) == len(generated) - 1
    assert not any(it["id"] == item_id for it in after_del)


@pytest.mark.anyio
async def test_persist_content_calendar_posts_helper(api_client: AsyncClient, seeded_db: AsyncSession):
    posts = [
        {
            "day": 1,
            "channel": "Instagram",
            "content_pillar": "Behind the Scenes",
            "format": "Reel",
            "concept": "Our artisan coffee roasting process in action",
            "caption_hook": "How real Cairo coffee is crafted ☕",
            "call_to_action": "Order your bag today!",
            "best_time": "18:00",
        },
        {
            "day": 3,
            "channel": "Facebook",
            "content_pillar": "Educational",
            "format": "Post",
            "concept": "3 tips to brew the best cup at home",
            "caption_hook": "Are you making this common coffee mistake?",
            "call_to_action": "Save this guide for later",
            "best_time": "12:30",
        },
    ]

    await _persist_content_calendar_posts("biz_test", posts)

    # Check rows in DB
    rows = (
        await seeded_db.execute(
            select(ContentItem).where(ContentItem.business_id == "biz_test")
        )
    ).scalars().all()

    assert len(rows) == 2
    titles = [r.title for r in rows]
    assert "Our artisan coffee roasting process in action" in titles
    assert "3 tips to brew the best cup at home" in titles
    assert all(r.status == "scheduled" for r in rows)
