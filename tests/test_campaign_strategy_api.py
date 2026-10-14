"""App.ai provider wiring and draft-only contract for both strategy API routes."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.agents.omar import service as omar_service
from app.ai.agents.omar.schema import CampaignStrategyOutput, ChannelPlan
from app.db.models import Campaign, Creative, Recommendation


def _install_fake_app_ai_provider(
    monkeypatch: pytest.MonkeyPatch,
    output: CampaignStrategyOutput,
) -> dict[str, Any]:
    calls: dict[str, Any] = {"providers": [], "schemas": [], "messages": []}

    class FakeRunnable:
        async def ainvoke(self, messages: list[Any]) -> CampaignStrategyOutput:
            calls["messages"].extend(messages)
            return output

    class FakeHandle:
        def structured(self, schema: type[Any]) -> FakeRunnable:
            calls["schemas"].append(schema)
            return FakeRunnable()

    def fake_get_llm(*, provider: str) -> FakeHandle:
        calls["providers"].append(provider)
        return FakeHandle()

    monkeypatch.setattr(
        omar_service,
        "get_ai_settings",
        lambda: SimpleNamespace(llm_provider="groq"),
    )
    monkeypatch.setattr(omar_service, "get_llm", fake_get_llm)
    return calls


async def test_strategy_route_uses_app_ai_provider_and_does_not_create_campaign(
    api_client: Any,
    seeded_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_app_ai_provider(
        monkeypatch,
        CampaignStrategyOutput(
            summary="Draft a lead strategy from the supplied business profile.",
            objective="leads",
            audience="Office managers in Egypt, as specified in the brief.",
            key_messages=["Use the verified saved offer."],
            success_metrics=["Qualified leads", "Cost per lead"],
            missing_inputs=["Confirm the lead conversion event."],
        ),
    )

    response = await api_client.post(
        "/api/v1/campaigns/strategy",
        headers={"X-Workspace-Id": "biz_test"},
        json={
            "campaignName": "Draft only",
            "objective": "leads",
            "audience": {"name": "Office managers", "countries": ["EG"]},
            "dailyBudget": 850,
            "currency": "EGP",
            "brief": "Use only the saved subscription offer.",
            "language": "en",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["summary"].startswith("Draft a lead strategy")
    assert calls["providers"] == ["groq"]
    assert calls["schemas"] == [CampaignStrategyOutput]
    human_text = next(message.content for message in calls["messages"] if message.type == "human")
    assert '"daily_budget": 850' in human_text

    # This endpoint only returns an editable strategy draft; it makes no
    # campaign, creative, recommendation, or Meta write.
    assert (await seeded_db.execute(select(Campaign))).scalars().all() == []
    assert (await seeded_db.execute(select(Creative))).scalars().all() == []
    assert (await seeded_db.execute(select(Recommendation))).scalars().all() == []


async def test_legacy_strategy_route_uses_same_app_ai_draft_without_fake_kpis(
    api_client: Any,
    seeded_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_app_ai_provider(
        monkeypatch,
        CampaignStrategyOutput(
            summary="A measured draft with no unsupported target values.",
            objective="sales",
            channels=[
                ChannelPlan(
                    channel="instagram",
                    role="Test the verified offer.",
                    placements=["Instagram Reels"],
                )
            ],
            placements=["Instagram Reels"],
            success_metrics=["Purchases", "Cost per purchase"],
            missing_inputs=["Confirm the purchase event before setting a target."],
        ),
    )

    response = await api_client.post(
        "/api/v1/campaigns/generate-strategy",
        headers={"X-Workspace-Id": "biz_test"},
        json={
            "objective": "sales",
            "dailyBudget": 1200,
            "audience": {"name": "Cairo customers"},
            "brief": "Use only verified saved information.",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["summary"] == "A measured draft with no unsupported target values."
    assert body["channels"] == ["instagram"]
    assert body["placements"] == ["Instagram Reels"]
    assert body["phases"] == []
    assert body["kpis"] == []
    assert calls["providers"] == ["groq"]
    assert calls["schemas"] == [CampaignStrategyOutput]
    assert (await seeded_db.execute(select(Campaign))).scalars().all() == []
    assert (await seeded_db.execute(select(Creative))).scalars().all() == []
    assert (await seeded_db.execute(select(Recommendation))).scalars().all() == []


def test_strong_llm_dependency_stays_on_the_ads_mcp_assist_route() -> None:
    from app.api.campaigns import (
        assist_meta_ad_plan_draft,
        generate_campaign_strategy,
        generate_campaign_strategy_endpoint,
    )

    assert "llm" in inspect.signature(assist_meta_ad_plan_draft).parameters
    assert "llm" not in inspect.signature(generate_campaign_strategy).parameters
    assert "llm" not in inspect.signature(generate_campaign_strategy_endpoint).parameters
