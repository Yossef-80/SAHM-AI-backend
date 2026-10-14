from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from app.ai.agents.omar import service as omar_service
from app.ai.agents.omar.schema import CampaignStrategyOutput
from app.core.errors import LLMProviderError


def test_campaign_strategy_uses_current_app_ai_provider_and_returns_typed_output(
    monkeypatch: Any,
) -> None:
    captured: list[Any] = []
    providers: list[str] = []
    expected = CampaignStrategyOutput(
        summary="Draft a lead-generation campaign for the supplied audience.",
        objective="leads",
        audience="Adults in the locations specified by the user.",
        key_messages=["Use only offers confirmed in the saved profile."],
        missing_inputs=["Confirm the cost-per-lead target."],
    )

    class FakeRunnable:
        async def ainvoke(self, messages: list[Any]) -> CampaignStrategyOutput:
            captured.extend(messages)
            return expected

    class FakeHandle:
        def structured(self, schema: type[Any]) -> FakeRunnable:
            assert schema is CampaignStrategyOutput
            return FakeRunnable()

    def fake_get_llm(*, provider: str) -> FakeHandle:
        providers.append(provider)
        return FakeHandle()

    monkeypatch.setattr(
        omar_service,
        "get_ai_settings",
        lambda: SimpleNamespace(llm_provider="groq"),
    )
    monkeypatch.setattr(omar_service, "get_llm", fake_get_llm)

    result = asyncio.run(
        omar_service.generate_campaign_strategy(
            campaign_input={
                "objective": "leads",
                "daily_budget": 1000,
                "currency": "EGP",
                "language": "ar",
            },
            business_profile={"name": "Saved business", "country": "EG"},
        )
    )

    assert result is expected
    assert providers == ["groq"]
    human_text = next(message.content for message in captured if isinstance(message, HumanMessage))
    assert '"daily_budget": 1000' in human_text
    assert '"name": "Saved business"' in human_text
    assert "Arabic" in human_text
    assert "do not create a Meta campaign" in captured[0].content


def test_campaign_strategy_surfaces_a_retryable_google_503(
    monkeypatch: Any,
) -> None:
    class ProviderUnavailable(Exception):
        code = 503

    class FakeRunnable:
        async def ainvoke(self, messages: list[Any]) -> Any:
            raise ProviderUnavailable("503 UNAVAILABLE: temporary provider load")

    class FakeHandle:
        def structured(self, schema: type[Any]) -> FakeRunnable:
            assert schema is CampaignStrategyOutput
            return FakeRunnable()

    monkeypatch.setattr(
        omar_service,
        "get_ai_settings",
        lambda: SimpleNamespace(llm_provider="gemini"),
    )
    monkeypatch.setattr(omar_service, "get_llm", lambda **kwargs: FakeHandle())

    with pytest.raises(LLMProviderError) as caught:
        asyncio.run(
            omar_service.generate_campaign_strategy(
                campaign_input={"objective": "awareness", "language": "ar"},
                business_profile={"name": "Saved business"},
            )
        )

    assert caught.value.status_code == 503
    assert caught.value.http_status == 503
    assert caught.value.retryable is True
    assert caught.value.detail["provider_status_code"] == 503
