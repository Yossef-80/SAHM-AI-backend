"""Sahm-only saved-audience library: tenant isolation, immutable snapshots, and draft suggestions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Campaign, SavedAudience
from app.schemas.audiences import AudienceIdea, AudienceSuggestionResponse
from app.schemas.targeting import TargetingSearchTerm

HEADERS = {"X-Workspace-Id": "biz_test"}
BASE = "/api/v1/audiences"


def _targeting(*, age_min: int = 24, interest_id: str = "6001") -> dict[str, Any]:
    return {
        "ageMin": age_min,
        "ageMax": 55,
        "genders": ["all"],
        "locations": ["Cairo, EG"],
        "countries": ["EG"],
        "interests": ["Specialty coffee"],
        "interestTargets": [{"id": interest_id, "name": "Coffee"}],
        "targetingTargets": [
            {"id": interest_id, "name": "Coffee", "category": "interests"},
            {"id": "behavior_test", "name": "Frequent travelers", "category": "behaviors"},
        ],
        "behaviors": ["Frequent travelers"],
    }


async def test_saved_audience_crud_archives_softly_and_is_business_scoped(
    api_client: Any,
    seeded_db: AsyncSession,
) -> None:
    from app.db.models import Business

    seeded_db.add(
        Business(
            id="biz_other",
            workspace_id="ws_other",
            name="Other business",
            industry="Retail",
            country="EG",
            language="en",
            currency="EGP",
            monthly_budget=1000,
        )
    )
    await seeded_db.commit()

    created = await api_client.post(
        BASE,
        headers=HEADERS,
        json={
            "name": "Cairo coffee buyers",
            "description": "A reusable test segment",
            "source": "manual",
            "targeting": _targeting(),
        },
    )
    assert created.status_code == 201, created.text
    audience = created.json()
    audience_id = audience["id"]
    assert audience["businessId"] == "biz_test"
    assert audience["targeting"]["interestTargets"] == [{"id": "6001", "name": "Coffee"}]
    assert audience["targeting"]["targetingTargets"] == [
        {"id": "6001", "name": "Coffee", "category": "interests"},
        {"id": "behavior_test", "name": "Frequent travelers", "category": "behaviors"},
    ]

    listed = await api_client.get(BASE, headers=HEADERS)
    assert listed.status_code == 200, listed.text
    assert listed.json()["total"] == 1
    assert listed.json()["audiences"][0]["id"] == audience_id

    updated = await api_client.patch(
        f"{BASE}/{audience_id}",
        headers=HEADERS,
        json={"name": "Updated coffee buyers", "targeting": _targeting(age_min=30)},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "Updated coffee buyers"
    assert updated.json()["targeting"]["ageMin"] == 30

    # A different business cannot enumerate or read the first business's preset.
    other_list = await api_client.get(BASE, headers={"X-Workspace-Id": "biz_other"})
    assert other_list.status_code == 200, other_list.text
    assert other_list.json()["total"] == 0
    cross_tenant_read = await api_client.get(
        f"{BASE}/{audience_id}", headers={"X-Workspace-Id": "biz_other"}
    )
    assert cross_tenant_read.status_code == 404
    cross_tenant_link = await api_client.post(
        "/api/v1/campaigns",
        headers={"X-Workspace-Id": "biz_other"},
        json={"name": "Invalid link", "audienceId": audience_id},
    )
    assert cross_tenant_link.status_code == 404

    archived = await api_client.delete(f"{BASE}/{audience_id}", headers=HEADERS)
    assert archived.status_code == 200, archived.text
    assert archived.json()["isArchived"] is True
    assert (await api_client.get(BASE, headers=HEADERS)).json()["total"] == 0
    # Archived records stay readable for linked campaign history.
    historical = await api_client.get(f"{BASE}/{audience_id}", headers=HEADERS)
    assert historical.status_code == 200
    assert historical.json()["isArchived"] is True


async def test_campaign_keeps_creation_time_audience_snapshot_after_library_update(
    api_client: Any,
    seeded_db: AsyncSession,
) -> None:
    created = await api_client.post(
        BASE,
        headers=HEADERS,
        json={"name": "Reusable audience", "targeting": _targeting(age_min=24)},
    )
    assert created.status_code == 201, created.text
    audience_id = created.json()["id"]

    campaign_response = await api_client.post(
        "/api/v1/campaigns",
        headers=HEADERS,
        json={
            "name": "Snapshot campaign",
            "objective": "sales",
            "channel": "meta_ads",
            "budget": 250,
            "currency": "EGP",
            "targetAudience": "Cairo, EG",
            "audienceId": audience_id,
        },
    )
    assert campaign_response.status_code == 200, campaign_response.text
    campaign_id = campaign_response.json()["id"]
    assert campaign_response.json()["audienceSnapshot"]["targeting"]["ageMin"] == 24
    assert campaign_response.json()["audienceSnapshot"]["targeting"]["targetingTargets"][1]["category"] == "behaviors"

    updated = await api_client.patch(
        f"{BASE}/{audience_id}",
        headers=HEADERS,
        json={"targeting": _targeting(age_min=35)},
    )
    assert updated.status_code == 200, updated.text
    archived = await api_client.delete(f"{BASE}/{audience_id}", headers=HEADERS)
    assert archived.status_code == 200, archived.text

    live_audience = await api_client.get(f"{BASE}/{audience_id}", headers=HEADERS)
    historical_campaign = await api_client.get(
        f"/api/v1/campaigns/{campaign_id}", headers=HEADERS
    )
    assert live_audience.json()["targeting"]["ageMin"] == 35
    snapshot = historical_campaign.json()["campaign"]["audienceSnapshot"]
    assert snapshot["targeting"]["ageMin"] == 24
    assert snapshot["name"] == "Reusable audience"

    # Exercise the ORM state too: the snapshot is an immutable value copy.
    row = await seeded_db.scalar(select(Campaign).where(Campaign.id == campaign_id))
    assert row is not None
    assert row.audience_id == audience_id
    assert row.audience_snapshot["targeting"]["ageMin"] == 24


async def test_suggestions_use_current_app_ai_provider_and_never_create_or_write(
    api_client: Any,
    seeded_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.audiences import suggestions as suggestion_service

    calls: dict[str, Any] = {"providers": [], "schemas": [], "messages": []}
    response_model = AudienceSuggestionResponse(
        suggestions=[
            AudienceIdea(
                name="Broad Cairo test",
                summary="Keep delivery broad inside the user-selected market.",
                approach="broad",
                interest_search_terms=[],
                rationale="The brief does not justify narrowing the audience yet.",
            ),
            AudienceIdea(
                name="Interest hypothesis",
                summary="Test an interest that closely matches the supplied offer.",
                approach="targeting_hypothesis",
                interest_search_terms=[],
                targeting_search_terms=[
                    TargetingSearchTerm(category="behaviors", term="Frequent travelers")
                ],
                rationale="Search Meta for a real match before applying this phrase.",
            ),
        ],
        missing_inputs=["product price"],
    )

    class FakeRunnable:
        async def ainvoke(self, messages: list[Any]) -> AudienceSuggestionResponse:
            calls["messages"].extend(messages)
            return response_model

    class FakeHandle:
        def structured(self, schema: type[Any]) -> FakeRunnable:
            calls["schemas"].append(schema)
            return FakeRunnable()

    def fake_get_llm(*, provider: str) -> FakeHandle:
        calls["providers"].append(provider)
        return FakeHandle()

    monkeypatch.setattr(
        suggestion_service,
        "get_ai_settings",
        lambda: SimpleNamespace(llm_provider="groq"),
    )
    monkeypatch.setattr(suggestion_service, "get_llm", fake_get_llm)

    result = await api_client.post(
        f"{BASE}/suggest",
        headers=HEADERS,
        json={
            "campaignName": "Coffee subscriptions",
            "objective": "sales",
            "brief": "Promote the verified monthly beans subscription.",
            "countries": ["EG"],
            "language": "en",
        },
    )
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["suggestions"][1]["interestSearchTerms"] == []
    assert body["suggestions"][1]["targetingSearchTerms"] == [
        {"category": "behaviors", "term": "Frequent travelers"}
    ]
    assert "id" not in body["suggestions"][1]
    assert calls["providers"] == ["groq"]
    assert calls["schemas"] == [AudienceSuggestionResponse]
    human_message = next(message.content for message in calls["messages"] if message.type == "human")
    system_message = next(message.content for message in calls["messages"] if message.type == "system")
    assert '"user_selected_countries": ["EG"]' in human_message
    assert "targeting_search_terms" in system_message

    # Suggestions are ideas only: they do not create a saved audience, campaign,
    # or any Meta object.
    assert (await seeded_db.execute(select(SavedAudience))).scalars().all() == []
    assert (await seeded_db.execute(select(Campaign))).scalars().all() == []
