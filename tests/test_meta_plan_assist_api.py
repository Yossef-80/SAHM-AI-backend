"""API contract tests for the read-only campaign-assistance route."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Self

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Campaign, Recommendation
from app.schemas.campaigns import (
    GeneratedCampaignStrategy,
    MetaAdPlanAssistRequest,
    MetaAdPlanAssistResponse,
)


async def test_assist_route_is_read_only_and_returns_review_required_draft(
    api_client: Any,
    seeded_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMcp:
        def __init__(self, token: str) -> None:
            assert token == "fake-meta-token"

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    async def fake_assist(**kwargs: Any) -> MetaAdPlanAssistResponse:
        assert kwargs["request"].objective == "traffic"
        return MetaAdPlanAssistResponse(
            primary_text="Try the summer blend.",
            headline="Shop now",
            call_to_action_type="SHOP_NOW",
            billing_event="LINK_CLICKS",
            optimization_goal="LINK_CLICKS",
            destination_type="WEBSITE",
            bid_strategy="LOWEST_COST_WITHOUT_CAP",
            allowed_values={},
            field_validation={"callToActionType": "valid"},
            mcp_context_available=True,
            review_required=True,
            writes_to_meta=False,
        )

    from app.api import campaigns as campaigns_api

    monkeypatch.setattr(
        campaigns_api,
        "resolve_meta_connection",
        lambda *args, **kwargs: _async_value(SimpleNamespace(access_token="fake-meta-token")),
    )
    monkeypatch.setattr(campaigns_api, "MetaAdsMcpClient", FakeMcp)
    monkeypatch.setattr(campaigns_api, "assist_meta_ad_plan", fake_assist)

    request = MetaAdPlanAssistRequest(
        campaign_name="Summer offer",
        objective="traffic",
        daily_budget=500,
        currency="EGP",
        language="en",
        strategy=GeneratedCampaignStrategy(
            summary="Drive visits to the summer offer.",
            objective="traffic",
            key_messages=["Try the new blend"],
        ),
        audience={
            "name": "Coffee fans",
            "ageMin": 18,
            "ageMax": 45,
            "gender": "all",
            "countries": ["EG"],
            "interests": [],
        },
        billing_event="LINK_CLICKS",
        optimization_goal="LINK_CLICKS",
        destination_type="WEBSITE",
    )
    response = await api_client.post(
        "/api/v1/campaigns/meta-ad-plan/assist",
        json=request.model_dump(mode="json", by_alias=True),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["primaryText"] == "Try the summer blend."
    assert body["reviewRequired"] is True
    assert body["writesToMeta"] is False
    assert (await seeded_db.execute(select(Campaign))).scalars().all() == []
    assert (await seeded_db.execute(select(Recommendation))).scalars().all() == []


async def _async_value(value: Any) -> Any:
    return value
