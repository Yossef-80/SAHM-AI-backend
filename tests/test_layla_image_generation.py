from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ai.agents.layla.graph import LAYLA_TOOLS
from app.ai.agents.orchestrator import brain
from app.ai.tools import image_generation
from app.ai.enums import LLMProvider
from app.ai.tools.image_generation import generate_image_draft_tool


@pytest.mark.asyncio
async def test_layla_image_tool_is_server_gated_and_does_not_need_tool_context(monkeypatch):
    monkeypatch.setattr(
        image_generation,
        "get_settings",
        lambda: (_ for _ in ()).throw(AssertionError("settings must not be read before opt-in")),
    )
    result = await generate_image_draft_tool.ainvoke(
        {"prompt": "An original product ad image"},
        config={"configurable": {"allow_image_generation": False}},
    )

    payload = json.loads(result)
    assert payload["status"] == "unavailable"
    assert "not explicitly requested" in payload["reason"]


@pytest.mark.asyncio
async def test_layla_image_tool_returns_labelled_mock_preview(monkeypatch):
    monkeypatch.setattr(
        image_generation,
        "get_settings",
        lambda: SimpleNamespace(
            image_provider="mock",
            image_api_key=None,
            image_model="mock-image-1",
            image_base_url=None,
        ),
    )
    result = await generate_image_draft_tool.ainvoke(
        {"prompt": "Create an original product image", "aspect_ratio": "4:5"},
        config={"configurable": {"allow_image_generation": True}},
    )

    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["asset"]["provider"] == "mock"
    assert payload["asset"]["url"].startswith("data:image/svg+xml;base64,")
    assert payload["asset"]["aspectRatio"] == "4:5"


@pytest.mark.asyncio
async def test_layla_image_tool_uses_operator_settings_without_exposing_credentials(monkeypatch):
    settings = SimpleNamespace(
        image_provider="openai",
        image_api_key="fake-secret-never-serialized",
        image_model="operator-image-model",
        image_base_url="https://images.example/v1",
    )
    generated = SimpleNamespace(
        url="https://images.example/output/preview.png",
        prompt="Create an original ad.\nSTYLE (follow exactly):\n- Palette: navy",
        provider="openai",
        model="operator-image-model",
        aspect_ratio="1:1",
    )
    request = AsyncMock(return_value=[generated])
    monkeypatch.setattr(image_generation, "get_settings", lambda: settings)
    monkeypatch.setattr(image_generation, "_generate_openai_image", request)

    result = await generate_image_draft_tool.ainvoke(
        {
            "prompt": "Create an original ad.",
            "style_spec": {"palette": ["navy"]},
        },
        config={"configurable": {"allow_image_generation": True}},
    )

    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["asset"]["url"] == generated.url
    assert "fake-secret-never-serialized" not in result
    request.assert_awaited_once()
    assert request.await_args.kwargs["api_key"] == settings.image_api_key
    assert "Palette: navy" in request.await_args.args[0]


@pytest.mark.asyncio
async def test_layla_image_tool_reads_the_workspace_connector_row(monkeypatch):
    """A workspace-configured provider must win over the process env here too.

    Regression: this tool read ``Settings`` only, so a workspace that had selected
    ``openai`` and stored its key in Settings -> Connectors still produced a
    placeholder SVG labelled "MOCK ASSET" with ``status: ok`` from the chat, while
    the creatives screen generated with the real provider.
    """
    env_settings = SimpleNamespace(
        image_provider="mock",
        image_api_key=None,
        image_model="mock-image-1",
        image_base_url=None,
        image_allow_mock_assets=False,
    )
    monkeypatch.setattr(image_generation, "get_settings", lambda: env_settings)

    # Exactly the surface the adapter reads off a resolved ConnectorSettings.
    workspace = SimpleNamespace(
        image_provider="openai",
        image_api_key="workspace-secret-key",
        image_model="gpt-image-1",
        image_base_url="https://gateway.example/v1",
    )

    async def _resolve(_ctx):
        return workspace

    monkeypatch.setattr(
        "app.core.tools.settings.resolve_tool_settings", _resolve, raising=True
    )

    generated = SimpleNamespace(
        url="https://gateway.example/out/preview.png",
        prompt="A product ad.",
        provider="openai",
        model="gpt-image-1",
        aspect_ratio="1:1",
    )
    request = AsyncMock(return_value=[generated])
    monkeypatch.setattr(image_generation, "_generate_openai_image", request)

    context = SimpleNamespace(user_supplied={}, business_id="biz_test")
    result = await generate_image_draft_tool.ainvoke(
        {"prompt": "A product ad."},
        config={
            "configurable": {
                "allow_image_generation": True,
                "tool_context": context,
            }
        },
    )

    payload = json.loads(result)
    assert payload["status"] == "ok", payload
    request.assert_awaited_once()
    assert request.await_args.kwargs["api_key"] == "workspace-secret-key"
    assert request.await_args.kwargs["model"] == "gpt-image-1"
    assert request.await_args.kwargs["base_url"] == "https://gateway.example/v1"
    assert "workspace-secret-key" not in result


@pytest.mark.asyncio
async def test_layla_image_tool_refuses_a_workspace_with_no_real_provider(monkeypatch):
    """No placeholder is returned as the requested image when a context is present."""
    env_settings = SimpleNamespace(
        image_provider="mock",
        image_api_key=None,
        image_model="mock-image-1",
        image_base_url=None,
        image_allow_mock_assets=False,
    )
    monkeypatch.setattr(image_generation, "get_settings", lambda: env_settings)

    workspace = SimpleNamespace(
        image_provider="mock",
        image_api_key=None,
        image_model="mock-image-1",
        image_base_url=None,
    )

    async def _resolve(_ctx):
        return workspace

    monkeypatch.setattr(
        "app.core.tools.settings.resolve_tool_settings", _resolve, raising=True
    )

    result = await generate_image_draft_tool.ainvoke(
        {"prompt": "A product ad."},
        config={
            "configurable": {
                "allow_image_generation": True,
                "tool_context": SimpleNamespace(user_supplied={}),
            }
        },
    )

    payload = json.loads(result)
    assert payload["status"] == "unavailable", payload
    assert "no placeholder was created" in payload["reason"]
    assert "asset" not in payload


def test_image_generation_tool_is_assigned_to_layla_only():
    assert generate_image_draft_tool in LAYLA_TOOLS
    from app.ai.agents.karim.graph import KARIM_TOOLS
    from app.ai.agents.nour.graph import NOUR_TOOLS
    from app.ai.agents.omar.graph import OMAR_TOOLS
    from app.ai.agents.salma.graph import SALMA_TOOLS

    assert generate_image_draft_tool not in NOUR_TOOLS
    assert generate_image_draft_tool not in OMAR_TOOLS
    assert generate_image_draft_tool not in KARIM_TOOLS
    assert generate_image_draft_tool not in SALMA_TOOLS


@pytest.mark.asyncio
async def test_explicit_layla_generation_is_routed_to_layla_and_returns_preview(monkeypatch):
    from langchain_core.messages import AIMessage, ToolMessage

    captured: dict[str, object] = {}
    generated_asset = {
        "url": "data:image/svg+xml;base64,PHN2Zy8+",
        "prompt": "Original product ad image prompt",
        "provider": "mock",
        "model": "mock-image-1",
        "aspectRatio": "1:1",
    }

    class _Router:
        async def ainvoke(self, _messages):
            # The deterministic explicit-image gate must still select Layla even
            # if the generic router initially suggests another agent.
            return brain.RoutingDecision(agent="nour", reason="router suggestion")

    class _Chat:
        def with_structured_output(self, _schema):
            return _Router()

    class _FakeGraph:
        async def ainvoke(self, _state, config):
            captured.update(config.get("configurable", {}))
            return {
                "messages": [
                    ToolMessage(
                        content=json.dumps({"status": "ok", "asset": generated_asset}),
                        name="generate_image_draft_tool",
                        tool_call_id="gen-1",
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "submit_creative_brief",
                                "args": {
                                    "summary": "Original campaign concept",
                                    "details": ["A clear product-led image."],
                                },
                                "id": "submit-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    ToolMessage(
                        content="REPORT_SUBMITTED_SUCCESSFULLY",
                        name="submit_creative_brief",
                        tool_call_id="submit-1",
                    ),
                ]
            }

    monkeypatch.setattr(
        brain,
        "get_ai_settings",
        lambda: SimpleNamespace(llm_provider=LLMProvider.MOCK),
    )
    monkeypatch.setattr(brain, "get_llm", lambda **_kwargs: SimpleNamespace(chat=_Chat()))
    monkeypatch.setattr(brain, "compress_history_messages", lambda _history: [])
    monkeypatch.setattr(brain, "layla_graph", _FakeGraph())

    result = await brain.orchestrate(
        "Generate an image for my product in Sahm",
        history=[],
    )

    assert captured["allow_image_generation"] is True
    assert any(block.type == "image_preview" for block in result.blocks)
    assert any(block.type == "ad_prompt" for block in result.blocks)
    assert any(block.type == "text" and "mock" in block.text for block in result.blocks)


def test_image_request_detector_does_not_treat_prompt_writing_as_generation():
    assert brain._requests_image_generation("Generate an image inside Sahm")
    assert not brain._requests_image_generation("Create an image prompt for an ad")
    assert not brain._requests_image_generation("Write a prompt to generate an image")
    assert not brain._requests_image_generation("How do I generate an image?")


def test_layla_graph_only_binds_image_tool_after_explicit_request(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from app.ai.agents.layla import graph as layla

    captured: list[list[str]] = []

    class _BoundModel:
        def invoke(self, _messages, _config):
            return AIMessage(content="ok")

    class _Chat:
        def bind_tools(self, tools):
            captured.append([tool.name for tool in tools])
            return _BoundModel()

    monkeypatch.setattr(
        layla,
        "get_ai_settings",
        lambda: SimpleNamespace(llm_provider=LLMProvider.MOCK),
    )
    monkeypatch.setattr(layla, "get_llm", lambda **_kwargs: SimpleNamespace(chat=_Chat()))

    layla.call_model(
        {"messages": [HumanMessage(content="Write an ad copy") ]},
        {"configurable": {"allow_image_generation": False}},
    )
    assert "generate_image_draft_tool" not in captured[-1]

    layla.call_model(
        {"messages": [HumanMessage(content="Generate an image") ]},
        {"configurable": {"allow_image_generation": True}},
    )
    assert "generate_image_draft_tool" in captured[-1]

    layla.call_model(
        {
            "messages": [
                HumanMessage(content="Generate an image"),
                ToolMessage(content="already attempted", name="generate_image_draft_tool", tool_call_id="gen"),
            ]
        },
        {"configurable": {"allow_image_generation": True}},
    )
    assert "generate_image_draft_tool" not in captured[-1]


def test_requests_image_generation_broadened_arabic_and_context():
    # Arabic feminine phrasing addressing Layla
    assert brain._requests_image_generation("اعملي صورة للمنتج")
    assert brain._requests_image_generation("صمميلي بوستر جديد")
    assert brain._requests_image_generation("ارسم صورة")
    assert brain._requests_image_generation("ولديلي صورة")
    assert brain._requests_image_generation("عايز اعملص صورة للبراند بتاعي")
    assert brain._requests_image_generation("طلعيلي صورة للبراند")
    assert brain._requests_image_generation("نفذيلي صورة إعلانية")
    
    # Tool mentions & design mode context
    assert brain._requests_image_generation("استخدمي /image_draft")
    assert brain._requests_image_generation("any text", context={"creative_mode": "design"})
    assert brain._requests_image_generation("any text", context={"allow_image_generation": True})

    # English phrases
    assert brain._requests_image_generation("draw me an image")
    assert brain._requests_image_generation("design a poster for our campaign")
    assert brain._requests_image_generation("create creative visual")

    # Negative / questions
    assert not brain._requests_image_generation("how to generate image")
    assert not brain._requests_image_generation("ازاي اعمل صورة")


def test_reference_analysis_message_grounding():
    context = {
        "reference_analysis_context": {
            "source_name": "product_hero.jpg",
            "summary": "Coffee cup on wooden table with warm lighting",
            "style": {
                "palette": ["#5A3825", "#F5EFEB"],
                "lighting": "Warm morning sunlight",
                "composition": "Centered close-up",
            },
        }
    }
    # Arabic
    ar_msg = brain._reference_analysis_message(context, arabic=True)
    assert ar_msg is not None
    assert "مرجع_الصورة_المعتمدة_للتصميم" in ar_msg
    assert "product_hero.jpg" in ar_msg
    assert "Coffee cup on wooden table" in ar_msg
    assert "#5A3825" in ar_msg

    # English
    en_msg = brain._reference_analysis_message(context, arabic=False)
    assert en_msg is not None
    assert "REFERENCED_CREATIVE_IMAGE_GROUNDING" in en_msg
    assert "product_hero.jpg" in en_msg
    assert "Warm morning sunlight" in en_msg

