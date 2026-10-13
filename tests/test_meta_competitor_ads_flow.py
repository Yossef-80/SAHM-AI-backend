from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage

from app.ai.agents.nour import meta_ads
from app.ai.agents.orchestrator import brain
from app.ai.enums import LLMProvider
from app.ai.tools import meta_ads as meta_ads_tools
from app.schemas.blocks import MessageWithBlocks, TextBlock
from app.schemas.tools import AdRecord, GetCompetitorAdsOutput


class _FakeStructured:
    def __init__(self, response):
        self.response = response

    async def ainvoke(self, _messages):
        return self.response


class _FakeGemini:
    def __init__(self, extractor_response, analysis_response=None):
        self.extractor_response = extractor_response
        self.analysis_response = analysis_response

    def structured(self, schema):
        if schema is meta_ads.CompetitorAdsSearchRequest:
            return _FakeStructured(self.extractor_response)
        return _FakeStructured(self.analysis_response)


class _FakeChat:
    def with_structured_output(self, schema):
        if schema is brain.RoutingDecision:
            return _FakeStructured(brain.RoutingDecision(agent="chat", reason="test"))
        return _FakeStructured(MessageWithBlocks(blocks=[TextBlock(text="ordinary response")]))

    async def ainvoke(self, _messages):
        return AIMessage(content="ordinary response")


def test_intent_gate_is_explicit_and_does_not_retrigger_on_followups():
    assert meta_ads.is_meta_competitor_ads_request("حلل إعلانات المنافسين على Meta Ad Library")
    assert meta_ads.is_meta_competitor_ads_request("Analyze competitor ads from the Meta ad library")
    assert not meta_ads.is_meta_competitor_ads_request("Hello, can you help me?")

    history = [
        {"role": "user", "content": "Analyze competitor ads on Meta Ad Library"},
        {"role": "assistant", "content": "Here are three observed ads."},
    ]
    assert not meta_ads.is_meta_competitor_ads_request("What was the third one's CTA?", history)

    clarification_history = [
        {"role": "user", "content": "حلل إعلانات المنافسين على Meta"},
        {
            "role": "assistant",
            "content": "مين المنافس أو اسم صفحة Facebook/Instagram اللي تحب أبحث عنها؟",
        },
    ]
    assert meta_ads.is_meta_competitor_ads_request("Acme Egypt", clarification_history)
    resolved = meta_ads.resolve_meta_ads_prompt("Acme Egypt", clarification_history)
    assert "الطلب الأصلي لتحليل إعلانات المنافسين" in resolved
    assert "Acme Egypt" in resolved


@pytest.mark.asyncio
async def test_ordinary_chat_never_enters_meta_route(monkeypatch):
    meta_route = AsyncMock(return_value=MessageWithBlocks(blocks=[TextBlock(text="meta")]))
    monkeypatch.setattr(brain, "analyze_meta_competitor_ads", meta_route)
    monkeypatch.setattr(brain, "get_ai_settings", lambda: SimpleNamespace(llm_provider=LLMProvider.MOCK))
    monkeypatch.setattr(brain, "get_llm", lambda **_kwargs: SimpleNamespace(chat=_FakeChat()))
    factory = Mock()

    result = await brain.orchestrate(
        "Hello, what can you help me with?",
        history=[],
        ad_tool_context_factory=factory,
    )

    assert result.blocks[0].text == "ordinary response"
    meta_route.assert_not_awaited()
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_meta_request_bypasses_general_agents(monkeypatch):
    response = MessageWithBlocks(blocks=[TextBlock(text="isolated ad report")])
    meta_route = AsyncMock(return_value=response)
    monkeypatch.setattr(brain, "analyze_meta_competitor_ads", meta_route)
    monkeypatch.setattr(
        brain,
        "get_llm",
        Mock(side_effect=AssertionError("general model/router must not run on the Meta-only path")),
    )
    factory = Mock()

    result = await brain.orchestrate(
        "حلل إعلانات المنافسين من Meta Ad Library لعلامة Acme",
        history=[],
        ad_tool_context_factory=factory,
    )

    assert result is response
    meta_route.assert_awaited_once()
    assert meta_route.await_args.kwargs["tool_context_factory"] is factory
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_search_stages_previews_without_sending_any_image_to_a_model(monkeypatch, tmp_path):
    extracted = meta_ads.CompetitorAdsSearchRequest(
        page_or_query="Acme Egypt", country="EG", max_results=2
    )
    monkeypatch.setattr(meta_ads, "_text_model", lambda: _FakeGemini(extracted))

    ad_output = GetCompetitorAdsOutput(
        ads=[
            AdRecord(
                ad_id="ad-42",
                page_name="Acme Egypt",
                headline="A useful offer",
                body="خصم على منتج جديد لفترة محدودة",
                platforms=["facebook", "instagram"],
                snapshot_url="https://www.facebook.com/ads/archive/render_ad/?id=ad-42",
            )
        ],
        page_or_query="Acme Egypt",
        country="EG",
        source="get_competitor_ads:meta_ad_library:EG",
        status="ok",
    )
    adapter = AsyncMock(return_value=ad_output)
    monkeypatch.setattr(meta_ads, "search_competitor_ads", adapter)

    async def fake_inspect(ad_id, _url):
        return meta_ads_tools.SnapshotImage(
            ad_id=ad_id,
            mime_type="image/jpeg",
            base64_data=base64.b64encode(b"one public preview").decode("ascii"),
        )

    monkeypatch.setattr(meta_ads, "inspect_ad_snapshot", fake_inspect)
    vlm = Mock(side_effect=AssertionError("the initial search must not invoke the VLM"))
    monkeypatch.setattr(meta_ads, "_ads_vlm_model", vlm)

    context = SimpleNamespace(
        business_id="biz_test",
        conversation_id="conv_test",
        require_settings=lambda: SimpleNamespace(upload_dir=str(tmp_path)),
    )
    factory = Mock(return_value=context)
    result = await meta_ads.analyze_meta_competitor_ads(
        "حلل إعلانات المنافسين على Meta لعلامة Acme Egypt",
        user_context="علامتنا تقدم منتجات يومية عملية.",
        tool_context_factory=factory,
    )

    adapter.assert_awaited_once()
    assert adapter.await_args.args[0].page_or_query == "Acme Egypt"
    assert adapter.await_args.args[0].country == "EG"
    assert adapter.await_args.args[1] is context
    vlm.assert_not_called()
    question = next(block for block in result.blocks if block.type == "question_with_options")
    assert len(question.options) == 1
    choice = question.options[0]
    assert choice.payload["kind"] == "meta_ad_preview"
    assert choice.payload["adId"] == "ad-42"
    assert "access_token" not in str(choice.payload)
    from app.ai.tools.ad_previews import read_ad_preview

    staged, mime = read_ad_preview(
        upload_dir=tmp_path,
        business_id="biz_test",
        preview_id=choice.payload["previewId"],
    )
    assert staged == b"one public preview"
    assert mime == "image/jpeg"
    from app.core.orchestrator.blocks import validate_blocks

    serialized = result.model_dump(mode="json", by_alias=True)["blocks"]
    assert validate_blocks(serialized)[-1]["type"] == "question_with_options"


@pytest.mark.asyncio
async def test_selected_ad_sends_only_the_chosen_image_to_the_vlm(monkeypatch, tmp_path):
    from app.ai.tools.ad_previews import store_ad_preview

    selected_bytes = b"selected image bytes"
    other_bytes = b"unselected image bytes"
    selected_id = store_ad_preview(
        upload_dir=tmp_path,
        business_id="biz_test",
        image_bytes=selected_bytes,
        mime_type="image/png",
    )
    store_ad_preview(
        upload_dir=tmp_path,
        business_id="biz_test",
        image_bytes=other_bytes,
        mime_type="image/png",
    )
    analysis = meta_ads.AdStyleSpec(
        palette=["#112233", "#ffffff"],
        composition="Large product focal point with open space for a headline.",
        typography="Bold sans-serif headline; exact font is not identifiable.",
        tone="Confident and practical.",
        layout_pattern="One focal image, short heading, small CTA area.",
        color_mood="Warm contrast.",
        text_density="Low.",
        summary="A simple, high-contrast product-led composition.",
    )
    captured = []

    class _CapturingStructured:
        async def ainvoke(self, messages):
            captured.extend(messages)
            return analysis

    class _FakeVlm:
        def structured(self, schema):
            assert schema is meta_ads.AdStyleSpec
            return _CapturingStructured()

    monkeypatch.setattr(meta_ads, "_ads_vlm_model", lambda: _FakeVlm())
    context = SimpleNamespace(
        business_id="biz_test",
        require_settings=lambda: SimpleNamespace(upload_dir=str(tmp_path)),
    )
    result = await meta_ads.analyze_selected_competitor_ad(
        {
            "kind": "meta_ad_preview",
            "previewId": selected_id,
            "adId": "ad-42",
            "pageName": "Acme Egypt",
            "platforms": ["facebook"],
            "publicUrl": "https://www.facebook.com/ads/library/?id=ad-42",
        },
        tool_context_factory=Mock(return_value=context),
    )

    assert len(captured) == 1
    parts = captured[0].content
    image_parts = [part for part in parts if part.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert base64.b64encode(selected_bytes).decode("ascii") in image_parts[0]["image_url"]
    assert base64.b64encode(other_bytes).decode("ascii") not in str(parts)
    assert any(block.type == "ad_style_analysis" for block in result.blocks)
    assert "ما التغييرات التي تريدها" in result.blocks[-1].text


@pytest.mark.asyncio
async def test_extractor_cannot_search_a_competitor_name_taken_from_brand_context(monkeypatch):
    extracted = meta_ads.CompetitorAdsSearchRequest(page_or_query="Cairo Coffee Co")
    monkeypatch.setattr(meta_ads, "_text_model", lambda: _FakeGemini(extracted))

    result = await meta_ads._extract_search_request(
        {
            "prompt": "حلل إعلانات المنافسين على Meta",
            "user_context": "Our own business name is Cairo Coffee Co.",
        }
    )

    assert result["search_request"].page_or_query is None


@pytest.mark.asyncio
async def test_missing_competitor_asks_for_name_without_creating_tool_context(monkeypatch):
    extracted = meta_ads.CompetitorAdsSearchRequest(page_or_query=None)
    monkeypatch.setattr(meta_ads, "_text_model", lambda: _FakeGemini(extracted))
    adapter = AsyncMock()
    monkeypatch.setattr(meta_ads, "search_competitor_ads", adapter)
    factory = Mock()

    result = await meta_ads.analyze_meta_competitor_ads(
        "حلل إعلانات المنافسين على Meta",
        tool_context_factory=factory,
    )

    assert "مين المنافس" in result.blocks[0].text
    factory.assert_not_called()
    adapter.assert_not_awaited()


@pytest.mark.asyncio
async def test_answer_selection_takes_the_isolated_vlm_path(monkeypatch):
    response = MessageWithBlocks(blocks=[TextBlock(text="selected" )])
    selected_route = AsyncMock(return_value=response)
    monkeypatch.setattr(brain, "analyze_selected_competitor_ad", selected_route)
    monkeypatch.setattr(
        brain,
        "get_llm",
        Mock(side_effect=AssertionError("general routing must not run for an ad selection")),
    )
    factory = Mock()
    selection = {"kind": "meta_ad_preview", "previewId": "1" * 32, "adId": "ad-1"}

    result = await brain.orchestrate(
        "Acme — spring ad",
        history=[],
        ad_flow_selection=selection,
        ad_tool_context_factory=factory,
    )

    assert result is response
    selected_route.assert_awaited_once_with(
        selection,
        tool_context_factory=factory,
        event_queue=None,
    )
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_ad_changes_draft_prompt_and_image_generation_is_separate_opt_in(monkeypatch):
    style_context = {"type": "ad_style_analysis", "analysis": {"palette": ["#123456"]}}
    prompt_response = MessageWithBlocks(blocks=[TextBlock(text="copyable prompt")])
    draft_route = AsyncMock(return_value=prompt_response)
    generation_route = AsyncMock(return_value=prompt_response)
    meta_route = AsyncMock(return_value=prompt_response)
    monkeypatch.setattr(brain, "draft_ad_prompt", draft_route)
    monkeypatch.setattr(brain, "generate_ad_image_draft", generation_route)
    monkeypatch.setattr(brain, "analyze_meta_competitor_ads", meta_route)

    draft = await brain.orchestrate(
        "اجعل اللون أزرق والنص أقصر",
        history=[],
        ad_style_context=style_context,
    )
    assert draft is prompt_response
    draft_route.assert_awaited_once()
    generation_route.assert_not_awaited()

    competitor_wording = await brain.orchestrate(
        "Make the competitor ad brighter",
        history=[],
        ad_style_context=style_context,
    )
    assert competitor_wording is prompt_response
    assert draft_route.await_count == 2
    meta_route.assert_not_awaited()

    generated = await brain.orchestrate(
        "Generate an image draft inside Sahm",
        history=[],
        ad_style_context=style_context,
        ad_prompt_context={"prompt": "Create an original ad"},
        ad_tool_context_factory=Mock(),
    )
    assert generated is prompt_response
    generation_route.assert_awaited_once()


@pytest.mark.asyncio
async def test_prompt_draft_uses_structured_text_model_and_returns_copyable_prompt(monkeypatch):
    captured_messages = []
    expected_prompt = (
        "Create an original product ad with a calm blue palette, a clear headline hierarchy, "
        "readable Arabic typography, generous whitespace, and placeholders for unconfirmed claims."
    )

    class _PromptStructured:
        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return meta_ads.AdPromptDraft(prompt=expected_prompt)

    class _PromptModel:
        def structured(self, schema):
            assert schema is meta_ads.AdPromptDraft
            return _PromptStructured()

    monkeypatch.setattr(meta_ads, "_text_model", lambda: _PromptModel())
    result = await meta_ads.draft_ad_prompt(
        "خلفية زرقاء ونص أقصر ومساحة واضحة للعنوان",
        {
            "sourceAdId": "ad-42",
            "sourceName": "Acme",
            "analysis": {"palette": ["navy", "white"], "composition": "one product focal point"},
        },
        user_context="منتج منزلي للاستخدام اليومي؛ لا توجد أسعار أو عروض مؤكدة.",
    )

    assert captured_messages[0].type == "system"
    prompt_block = next(block for block in result.blocks if block.type == "ad_prompt")
    assert prompt_block.prompt == expected_prompt
    assert prompt_block.source_ad_id == "ad-42"
    assert prompt_block.source_name == "Acme"


def test_snapshot_token_is_never_added_to_model_prompt_or_public_citation():
    secret_marker = "test-only-snapshot-secret"
    ad = AdRecord(
        ad_id="12345",
        page_name="Demo",
        body="A public ad",
        snapshot_url=(
            "https://www.facebook.com/ads/archive/render_ad/?id=12345&access_token="
            + secret_marker
        ),
    )
    content = meta_ads._analysis_prompt(
        meta_ads.CompetitorAdsSearchRequest(page_or_query="Demo"), [ad], "", []
    )

    assert secret_marker not in str(content)
    assert secret_marker not in meta_ads_tools.public_ad_library_url(ad.ad_id)
    assert meta_ads_tools.public_ad_library_url(ad.ad_id) == "https://www.facebook.com/ads/library/?id=12345"


def test_unsafe_snapshot_host_is_rejected_without_fetching():
    assert meta_ads_tools._validated_url(
        "https://attacker.example/ads/archive/render_ad/?id=1", initial_snapshot=True
    ) is None
    assert meta_ads_tools._validated_url(
        "http://127.0.0.1/ads/archive/render_ad/?id=1", initial_snapshot=True
    ) is None


def test_httpx_request_log_redacts_meta_access_token():
    import logging

    credential_filter = meta_ads_tools._HttpxCredentialRedactionFilter()
    record = logging.LogRecord(
        "httpx",
        logging.INFO,
        "test.py",
        1,
        "HTTP Request: GET https://graph.facebook.com/ads_archive?access_token=do-not-log&limit=2 200 OK",
        (),
        None,
    )

    assert credential_filter.filter(record)
    assert "do-not-log" not in record.getMessage()
    assert "***REDACTED***" in record.getMessage()


def test_unavailable_error_does_not_leak_access_token():
    result = GetCompetitorAdsOutput(
        status="unavailable",
        source="meta",
        unavailable_reason=(
            "Ad Library request failed: "
            "https://graph.facebook.com/?access_token=do-not-leak-this"
        ),
    )
    response = meta_ads._format_unavailable(
        {
            "search_request": meta_ads.CompetitorAdsSearchRequest(page_or_query="Acme"),
            "ads_output": result,
        }
    )["response"]

    serialized = str(response.model_dump(mode="json"))
    assert "do-not-leak-this" not in serialized
    assert "تعذر الوصول إلى Meta Ad Library الآن" in serialized
