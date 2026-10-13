from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage

from app.ai.agents.omar import service as omar_service
from app.ai.agents.omar.schema import CampaignStrategyOutput


def test_campaign_strategy_service_uses_supplied_context_and_returns_typed_output(monkeypatch: Any) -> None:
    captured: list[Any] = []
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

    monkeypatch.setattr(
        omar_service,
        "get_ai_settings",
        lambda: SimpleNamespace(llm_provider="mock"),
    )
    monkeypatch.setattr(omar_service, "get_llm", lambda **_kwargs: FakeHandle())

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
    human_text = next(message.content for message in captured if isinstance(message, HumanMessage))
    assert '"daily_budget": 1000' in human_text
    assert '"name": "Saved business"' in human_text
    assert "Arabic" in human_text
    assert "do not create a Meta campaign" in captured[0].content
