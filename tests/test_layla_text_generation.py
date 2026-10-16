"""Mock-only tests for Layla's direct creative-workspace text endpoint."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.ai.agents.layla import service
from app.ai.agents.layla.schema import CreativeBrief, CreativeVariant
from app.ai.enums import LLMProvider
from app.ai.errors import ProviderNotConfiguredError


class FakeStructuredRunnable:
    def __init__(self, result: CreativeBrief) -> None:
        self.result = result
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> CreativeBrief:
        self.messages = messages
        return self.result


class FakeLLMHandle:
    def __init__(self, result: CreativeBrief) -> None:
        self.runnable = FakeStructuredRunnable(result)
        self.schema: type[Any] | None = None
        #: Every schema asked for, in order: the writing call first, then the
        #: review pass's verdict schema.
        self.schemas: list[type[Any]] = []

    def structured(self, schema: type[Any]) -> FakeStructuredRunnable:
        self.schema = schema
        self.schemas.append(schema)
        return self.runnable


async def test_layla_generation_uses_app_ai_and_shapes_ad_variations(monkeypatch: Any) -> None:
    model_result = CreativeBrief(
        deliverable_type="creative_brief",
        summary="A warmer morning routine",
        variants=[
            CreativeVariant(
                angle="Convenience",
                headline="Make mornings easier",
                primary_text="Start your day with less prep.",
                description="Ready when you are.",
                call_to_action="Learn more",
            )
        ],
    )
    handle = FakeLLMHandle(model_result)
    requested_provider: list[Any] = []
    monkeypatch.setattr(service, "get_ai_settings", lambda: SimpleNamespace(llm_provider=LLMProvider.MOCK))
    monkeypatch.setattr(service, "get_llm", lambda **kwargs: requested_provider.append(kwargs) or handle)

    output = await service.generate_layla_text(
        tool="generate_ad_variations",
        brief="Create several ads for a coffee subscription.",
        language="ar",
        brand_context={"brand_voice": "warm", "target_audience": "busy professionals"},
    )

    # The provider still comes from settings, but the *role* now decides the
    # sampling: ad variations are creative work, so the handle is built with the
    # creative temperature instead of the process-wide AI_LLM_TEMPERATURE -- and
    # a second, colder handle is built for the review pass that runs before the
    # draft is handed back.
    from app.ai.roles import resolve_role

    assert len(requested_provider) == 2
    assert requested_provider[0]["provider"] == LLMProvider.MOCK
    assert requested_provider[0]["temperature"] == resolve_role("creative_copy").temperature
    assert requested_provider[1]["temperature"] == resolve_role("critique").temperature
    assert requested_provider[1]["temperature"] < requested_provider[0]["temperature"]
    assert handle.schemas[0] is CreativeBrief
    assert "verdict" in handle.schema.model_fields  # the review pass's schema
    assert "Arabic (ar)" in handle.runnable.messages[1].content
    assert "Brand voice: warm" in handle.runnable.messages[1].content
    assert output["variations"][0]["angle"] == "Convenience"
    assert output["variations"][0]["headline"] == "Make mornings easier"
    assert output["based_on"] == "A warmer morning routine"
    # The selected UI action remains authoritative over a mismatched mock result.
    assert model_result.deliverable_type == "creative_brief"


async def test_layla_direct_api_does_not_use_tasks_or_pass_saved_visual_style(
    api_client: Any, monkeypatch: Any
) -> None:
    call: dict[str, Any] = {}

    async def fake_generate_layla_text(**kwargs: Any) -> dict[str, Any]:
        call.update(kwargs)
        return {"primary_texts": ["A clear, grounded ad line."], "summary": "Drafted."}

    monkeypatch.setattr(service, "generate_layla_text", fake_generate_layla_text)
    response = await api_client.post(
        "/api/v1/creatives/generate-text",
        json={
            "tool": "generate_ad_copy",
            "brief": "Write a restrained ad for the new coffee subscription.",
            "language": "en",
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["tool"] == "generate_ad_copy"
    assert result["output"]["primary_texts"] == ["A clear, grounded ad line."]
    assert call["tool"] == "generate_ad_copy"
    assert call["language"] == "en"
    assert "visual_style" not in call["brand_context"]


async def test_commercial_script_route_accepts_write_with_ai_request(
    api_client: Any, monkeypatch: Any
) -> None:
    call: dict[str, Any] = {}

    async def fake_generate_layla_text(**kwargs: Any) -> dict[str, Any]:
        call.update(kwargs)
        return {
            "hook": "A clear opening based on the brief.",
            "scenes": [{"order": 1, "timecode": "0–3s", "visual": "Use the supplied product details."}],
            "notes": "No unsupported claims added.",
        }

    monkeypatch.setattr(service, "generate_layla_text", fake_generate_layla_text)
    response = await api_client.post(
        "/api/v1/creatives/generate-text",
        json={
            "tool": "generate_commercial_script",
            "brief": "Write a restrained 15-second video script for the supplied product.",
            "language": "en",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["tool"] == "generate_commercial_script"
    assert response.json()["output"]["scenes"][0]["timecode"] == "0–3s"
    assert call["tool"] == "generate_commercial_script"


async def test_layla_provider_errors_are_safe_and_actionable(
    api_client: Any, monkeypatch: Any
) -> None:
    async def fail_without_provider(**kwargs: Any) -> dict[str, Any]:
        raise ProviderNotConfiguredError("OPENAI_API_KEY is missing")

    monkeypatch.setattr(service, "generate_layla_text", fail_without_provider)
    response = await api_client.post(
        "/api/v1/creatives/generate-text",
        json={
            "tool": "generate_commercial_script",
            "brief": "Write a restrained 15-second video script for the supplied product.",
            "language": "en",
        },
    )

    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "tool_unavailable"
    assert "provider configuration" in response.json()["error"]["message"]
    assert "OPENAI_API_KEY" not in response.text


async def test_image_and_video_prompts_are_directed_to_guided_layla_chat(api_client: Any) -> None:
    for tool in ("generate_image_prompt", "generate_video_prompt"):
        response = await api_client.post(
            "/api/v1/creatives/generate-text",
            json={"tool": tool, "brief": "A product launch", "language": "en"},
        )
        assert response.status_code == 403
        assert "/assistant/message" in response.text
        assert "creative_prompt" in response.text
