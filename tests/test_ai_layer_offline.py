"""Offline coverage for the app.ai provider, memory, tools, and audit boundaries.

Every external boundary is replaced by a deterministic fake or static HTML.
These tests must not call a live LLM, search/reviews API, website, Meta, or DB.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from app.ai.enums import DatabaseProvider, LLMProvider, SearchProvider, StructuredOutputMethod
from app.ai.errors import ProviderNotConfiguredError, ProviderNotInstalledError
from app.ai.memory.brand_memory import BrandProfile, get_brand_profile, put_brand_profile
from app.ai.providers.db import _PROVIDER_MODULES as DB_PROVIDER_MODULES
from app.ai.providers.llm import _PROVIDER_MODULES, get_llm, get_vlm_llm
from app.ai.providers.reviews import PlaceResult, PlaceReview
from app.ai.providers.search import _PROVIDER_MODULES as SEARCH_PROVIDER_MODULES
from app.ai.settings import AISettings
from app.core.errors import ToolUnavailable


class TinySchema(BaseModel):
    answer: str


def test_provider_registries_cover_every_enum_and_mock_llm_is_offline() -> None:
    assert set(_PROVIDER_MODULES) == set(LLMProvider)
    assert set(SEARCH_PROVIDER_MODULES) == set(SearchProvider)
    assert set(DB_PROVIDER_MODULES) == set(DatabaseProvider)

    handle = get_llm(settings=AISettings(llm_provider=LLMProvider.MOCK))
    assert handle.provider == LLMProvider.MOCK
    reply = handle.chat.invoke([HumanMessage(content="offline test")])
    assert "[mock]" in str(reply.content)
    structured = handle.structured(TinySchema).invoke([HumanMessage(content="offline test")])
    assert isinstance(structured, TinySchema)


def test_dedicated_vlm_url_uses_its_model_without_changing_general_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.ai.providers import llm as llm_module

    monkeypatch.delenv("AI_ADS_VLM_PROVIDER", raising=False)
    monkeypatch.delenv("AI_ADS_VLM_MODEL", raising=False)
    monkeypatch.setenv("AI_ADS_VLM_BASE_URL", "https://dedicated-vlm.example/")
    monkeypatch.setenv("VLM_OLLAMA_MODEL", "qwen3-vl:8b")
    settings = AISettings(
        _env_file=None,
        llm_provider=LLMProvider.GROQ,
        llm_model="general-model",
        groq_model="general-groq-model",
        ollama_base_url="https://general-ollama.example",
        ollama_model="general-ollama-model-tag",
    )
    built: dict[str, Any] = {}

    def fake_get_llm(*, provider: Any, model: str, settings: AISettings) -> Any:
        built.update(provider=provider, model=model, base_url=settings.ollama_base_url)
        return SimpleNamespace(provider=provider, model_name=model)

    monkeypatch.setattr(llm_module, "get_llm", fake_get_llm)
    handle = get_vlm_llm(settings=settings)

    assert handle.provider == LLMProvider.OLLAMA
    assert handle.model_name == "qwen3-vl:8b"
    assert built == {
        "provider": LLMProvider.OLLAMA,
        "model": "qwen3-vl:8b",
        "base_url": "https://dedicated-vlm.example",
    }
    assert settings.llm_provider == LLMProvider.GROQ
    assert settings.llm_model == "general-model"
    assert settings.groq_model == "general-groq-model"
    assert settings.ollama_model == "general-ollama-model-tag"
    assert settings.ollama_base_url == "https://general-ollama.example"


def test_dedicated_vlm_requires_its_own_model_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_ADS_VLM_PROVIDER", raising=False)
    monkeypatch.delenv("AI_ADS_VLM_MODEL", raising=False)
    monkeypatch.delenv("VLM_OLLAMA_MODEL", raising=False)
    settings = AISettings(_env_file=None, ads_vlm_base_url="https://dedicated-vlm.example")
    with pytest.raises(ProviderNotConfiguredError, match="VLM_OLLAMA_MODEL"):
        get_vlm_llm(settings=settings)


def test_gemini_vlm_has_independent_provider_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.ai.providers import llm as llm_module

    monkeypatch.setenv("AI_ADS_VLM_PROVIDER", "gemini")
    monkeypatch.setenv("AI_ADS_VLM_MODEL", "gemini-vision-test")
    # An explicitly selected Gemini VLM takes precedence over legacy VLM Ollama settings.
    monkeypatch.setenv("AI_ADS_VLM_BASE_URL", "https://legacy-vlm.example/")
    monkeypatch.setenv("VLM_OLLAMA_MODEL", "legacy-qwen-vl")
    settings = AISettings(
        _env_file=None,
        llm_provider=LLMProvider.OLLAMA,
        llm_model="system-llm-model",
        ollama_model="general-ollama-model",
        ollama_base_url="https://general-ollama.example",
        gemini_model="general-gemini-model",
        structured_output_method=StructuredOutputMethod.JSON_SCHEMA,
    )
    built: dict[str, Any] = {}

    def fake_get_llm(*, provider: Any, model: str, settings: AISettings) -> Any:
        built.update(provider=provider, model=model, settings=settings)
        return SimpleNamespace(provider=provider, model_name=model)

    monkeypatch.setattr(llm_module, "get_llm", fake_get_llm)
    handle = get_vlm_llm(settings=settings)

    assert handle.provider == LLMProvider.GEMINI
    assert handle.model_name == "gemini-vision-test"
    assert built["provider"] == LLMProvider.GEMINI
    assert built["model"] == "gemini-vision-test"
    assert built["settings"].structured_output_method is None
    assert built["settings"].ollama_base_url == "https://general-ollama.example"
    assert settings.llm_provider == LLMProvider.OLLAMA
    assert settings.llm_model == "system-llm-model"
    assert settings.ollama_model == "general-ollama-model"
    assert settings.gemini_model == "general-gemini-model"


def test_explicit_vlm_provider_requires_an_independent_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_ADS_VLM_MODEL", raising=False)
    settings = AISettings(_env_file=None, ads_vlm_provider=LLMProvider.GEMINI)

    with pytest.raises(ProviderNotConfiguredError, match="AI_ADS_VLM_MODEL"):
        get_vlm_llm(settings=settings)


def test_gemini_vlm_adapter_builds_offline_without_changing_system_llm() -> None:
    settings = AISettings(
        _env_file=None,
        llm_provider=LLMProvider.OLLAMA,
        llm_model="system-ollama-model",
        ollama_base_url="http://localhost:11434",
        ads_vlm_provider=LLMProvider.GEMINI,
        ads_vlm_model="gemini-3.5-flash",
        gemini_api_key="test-key-not-used",
    )

    handle = get_vlm_llm(settings=settings)

    assert handle.provider == LLMProvider.GEMINI
    assert handle.model_name == "gemini-3.5-flash"
    assert handle.structured_method == StructuredOutputMethod.FUNCTION_CALLING
    assert settings.llm_provider == LLMProvider.OLLAMA
    assert settings.llm_model == "system-ollama-model"


@pytest.mark.parametrize(
    ("provider", "extra"),
    [
        (LLMProvider.OLLAMA, {"ollama_base_url": None}),
        (LLMProvider.OPENAI, {"openai_api_key": None, "openai_base_url": None}),
        (LLMProvider.GROQ, {"groq_api_key": None}),
        (LLMProvider.GEMINI, {"gemini_api_key": None}),
        (LLMProvider.OPENROUTER, {"openrouter_api_key": None}),
    ],
)
def test_selected_remote_llm_providers_fail_closed_without_configuration(
    provider: LLMProvider, extra: dict[str, Any]
) -> None:
    settings = AISettings(llm_provider=provider, **extra)
    with pytest.raises(ProviderNotConfiguredError):
        get_llm(settings=settings)


def test_anthropic_adapter_reports_missing_optional_package_or_key_without_calling_api() -> None:
    settings = AISettings(llm_provider=LLMProvider.ANTHROPIC, anthropic_api_key=None)
    with pytest.raises((ProviderNotConfiguredError, ProviderNotInstalledError)):
        get_llm(settings=settings)


def test_search_cost_guard_blocks_live_provider_before_search(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.ai.tools.search as search_module

    class FakeLiveSearch:
        name = "fake_live_search"
        is_placeholder = False

        def search(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            raise AssertionError("the paid search adapter must not be called without opt-in")

    monkeypatch.setattr(
        search_module,
        "get_ai_settings",
        lambda: AISettings(search_provider=SearchProvider.TAVILY, allow_paid_search_calls=False),
    )
    monkeypatch.setattr(search_module, "get_search_provider", lambda _choice: FakeLiveSearch())
    blocked = json.loads(search_module.web_search_tool.invoke({"query": "coffee in Cairo"}))
    assert blocked["status"] == "unavailable"
    assert "No search" not in blocked.get("unavailable_reason", "")
    assert "disabled" in blocked["unavailable_reason"]

    class FakeHit:
        title = "Static result"
        url = "https://example.test/source"
        snippet = "An offline fixture result."
        published_date = None
        score = 0.9

    class FakeOptedInSearch(FakeLiveSearch):
        def search(self, query: str, *, max_results: int = 5) -> list[Any]:
            return [FakeHit()][:max_results]

    monkeypatch.setattr(
        search_module,
        "get_ai_settings",
        lambda: AISettings(search_provider=SearchProvider.TAVILY, allow_paid_search_calls=True),
    )
    monkeypatch.setattr(search_module, "get_search_provider", lambda _choice: FakeOptedInSearch())
    accepted = json.loads(search_module.web_search_tool.invoke({"query": "coffee in Cairo"}))
    assert accepted["status"] == "ok"
    assert accepted["results"][0]["url"] == "https://example.test/source"


def test_reviews_are_blocked_without_opt_in_but_injected_fakes_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.ai.tools.get_reviews as review_tool_module
    from app.ai.tools.get_reviews import GetReviewsInput, run_get_reviews

    monkeypatch.setattr(
        review_tool_module,
        "get_ai_settings",
        lambda: AISettings(allow_paid_review_calls=False),
    )
    payload = GetReviewsInput(business_name="River Roast", location="Cairo, Egypt")
    blocked = json.loads(run_get_reviews(payload))
    assert blocked["status"] == "unavailable"
    assert "No provider was contacted" in blocked["unavailable_reason"]

    class FakeGoogle:
        def search(self, business_name: str, location: str | None, language: str) -> list[PlaceResult]:
            assert business_name == "River Roast"
            assert location == "Cairo, Egypt"
            assert language == "en"
            return [
                PlaceResult(
                    place_id="fixture-place",
                    name="River Roast",
                    address="Cairo",
                    match_score=0.98,
                    rating=4.6,
                    total_ratings=12,
                    reviews=[PlaceReview(rating=5, text="Fixture review only.")],
                )
            ]

    accepted = json.loads(run_get_reviews(payload, google=FakeGoogle()))
    assert accepted["status"] == "ok"
    assert accepted["reviews"][0]["text"] == "Fixture review only."
    assert accepted["total_ratings"] == 12


def test_static_html_audit_extracts_observations_and_never_predicts_performance() -> None:
    from app.ai.tools.analyze_website import _extract_audit

    fixture = Path(__file__).parent / "fixtures" / "marketing_page.html"
    html = fixture.read_text(encoding="utf-8")
    audit = _extract_audit(
        html,
        final_url="https://river-roast.example/",
        requested_url="https://river-roast.example/",
        max_chars=2000,
    )
    assert audit.status == "ok"
    assert audit.title == "River Roast | Office coffee subscriptions"
    assert any("350 EGP" in fact.value or "350 EGP" in fact.evidence for fact in audit.offers_and_prices)
    assert audit.calls_to_action
    assert audit.trust_signals
    assert any("products/office-box" in fact.value for fact in audit.important_links)
    assert audit.analysis_boundaries
    assert not any("conversion rate" in fact.value.lower() for fact in audit.observations)


async def test_brand_report_and_full_analysis_compose_only_supplied_static_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.ai.tools.brand_research as research_module

    async def fake_search(*_args: Any, **_kwargs: Any) -> str:
        return json.dumps(
            {
                "status": "partial",
                "source": "offline_mock",
                "results": [
                    {"title": "Fixture result", "url": "https://example.test/source", "snippet": "Not live."}
                ],
            }
        )

    async def fake_website(*_args: Any, **_kwargs: Any) -> str:
        return json.dumps(
            {
                "status": "ok",
                "observations": [
                    {
                        "category": "visible_price_text",
                        "label": "Price",
                        "value": "EGP 350",
                        "source_url": "https://river-roast.example/",
                    }
                ],
                "not_observed": ["No explicit customer count was observed."],
            }
        )

    monkeypatch.setattr(research_module, "web_search_tool", SimpleNamespace(ainvoke=fake_search))
    monkeypatch.setattr(research_module, "analyze_website_tool", SimpleNamespace(ainvoke=fake_website))
    report = json.loads(
        await research_module.brand_report.ainvoke(
            {"brand": "River Roast", "url": "https://river-roast.example/", "country": "Egypt"}
        )
    )
    assert report["status"] == "partial"  # search is explicitly a mock/partial source
    assert report["website"]["observations"][0]["value"] == "EGP 350"

    analysis = json.loads(
        research_module.full_analysis.invoke(
            {"brand": "River Roast", "report_data": json.dumps(report)}
        )
    )
    assert analysis["status"] == "partial"
    assert "https://river-roast.example/" in analysis["source_urls"]
    assert "No explicit customer count was observed." in analysis["gaps"]
    assert "performance" in analysis["analysis_boundary"].lower()


def test_file_routing_ignores_server_paths_and_redacts_supplied_content() -> None:
    from app.ai.tools.brand_research import route_file_to_tool

    no_content = json.loads(
        route_file_to_tool.invoke(
            {"filename": "private.json", "file_path": "/etc/passwd", "brand": "Test"}
        )
    )
    assert no_content["status"] == "unavailable"
    assert "not read" in no_content["unavailable_reason"]

    supplied = json.loads(
        route_file_to_tool.invoke(
            {
                "filename": "evidence.json",
                "file_content": '{"access_token":"do-not-store-this","impressions":12}',
                "brand": "Test",
            }
        )
    )
    assert supplied["route"] == "campaign_metrics"
    assert "do-not-store-this" not in json.dumps(supplied)


def test_memory_profile_and_learning_are_scoped_and_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.ai.tools.brand_context as brand_context_module
    import app.ai.tools.save_learning as save_learning_module

    store = InMemoryStore()
    monkeypatch.setattr(brand_context_module, "current_store", lambda: store)
    monkeypatch.setattr(save_learning_module, "current_store", lambda: store)
    monkeypatch.setattr(
        brand_context_module,
        "get_ai_settings",
        lambda: AISettings(memory_global_profile_max_tokens=100),
    )
    monkeypatch.setattr(
        save_learning_module,
        "get_ai_settings",
        lambda: AISettings(memory_item_max_tokens=100),
    )

    put_brand_profile(
        store,
        "biz-a",
        BrandProfile(name="River Roast", description="api_key=private-value"),
    )
    profile = get_brand_profile(store, "biz-a")
    assert profile is not None
    assert "private-value" not in profile.description
    assert get_brand_profile(store, "biz-b") is None

    saved = json.loads(
        save_learning_module.save_learning_tool.invoke(
            {
                "insight": "The supplied page lists EGP 350; access_token=private-value",
                "category": "market",
                "source_url": "https://example.test/?token=private-value",
            },
            config={"configurable": {"business_id": "biz-a", "agent": "omar"}},
        )
    )
    assert saved["status"] == "ok"
    assert "private-value" not in json.dumps(saved)

    context = json.loads(
        brand_context_module.get_brand_context_tool.invoke(
            {}, config={"configurable": {"business_id": "biz-a"}}
        )
    )
    assert context["status"] == "ok"
    assert context["profile"]["name"] == "River Roast"
    assert "private-value" not in json.dumps(context)

    settings_context = json.loads(
        brand_context_module.get_brand_context_tool.invoke(
            {},
            config={
                "configurable": {
                    "business_id": "biz-a",
                    "business_profile_context": {
                        "name": "Settings Roast",
                        "industry": "Coffee shop",
                        "country": "EG",
                        "target_audience": "Office workers in Cairo",
                    },
                }
            },
        )
    )
    assert settings_context["status"] == "ok"
    assert settings_context["profile_source"] == "business_settings"
    assert settings_context["profile"]["name"] == "Settings Roast"
    assert settings_context["profile"]["industry"] == "Coffee shop"

    other = json.loads(
        brand_context_module.get_brand_context_tool.invoke(
            {}, config={"configurable": {"business_id": "biz-b"}}
        )
    )
    assert other["status"] == "partial"


def test_database_provider_boundaries_do_not_contact_postgres_or_create_app_migrations(
    tmp_path: Path,
) -> None:
    from app.ai.providers.db import memory, postgres, sqlite

    settings = AISettings(db_provider=DatabaseProvider.MEMORY)
    store = memory.build_store(settings)
    store.put(("brands", "biz-a"), "profile", {"name": "A"})
    assert store.get(("brands", "biz-a"), "profile").value == {"name": "A"}
    assert store.get(("brands", "biz-b"), "profile") is None
    assert memory.build_checkpointer(settings) is not None

    with pytest.raises(ProviderNotConfiguredError):
        postgres._connect(AISettings(db_provider=DatabaseProvider.POSTGRES, postgres_url=None))

    # SQLite uses a temporary AI-only file if its optional plugin is installed;
    # otherwise the provider fails explicitly rather than pretending to persist.
    sqlite_settings = AISettings(
        db_provider=DatabaseProvider.SQLITE,
        sqlite_path=str(tmp_path / "ai-checkpoint.sqlite"),
    )
    try:
        sqlite_store = sqlite.build_store(sqlite_settings)
    except ProviderNotInstalledError:
        assert not (tmp_path / "ai-checkpoint.sqlite").exists()
    else:
        sqlite_store.put(("brands", "biz-a"), "profile", {"name": "A"})
        assert sqlite_store.get(("brands", "biz-a"), "profile").value == {"name": "A"}
        sqlite_checkpointer = sqlite.build_checkpointer(sqlite_settings)
        assert sqlite_checkpointer is not None


def test_tool_audit_callback_redacts_fields_and_output() -> None:
    from app.ai.loggers.callbacks import AIToolAuditCallback

    callback = AIToolAuditCallback(agent="nour")
    run_id = uuid.uuid4()
    callback.on_tool_start(
        {"name": "web_search"},
        "",
        run_id=run_id,
        inputs={"query": "coffee", "api_key": "private-value"},
    )
    callback.on_tool_end(
        '{"status":"ok","access_token":"private-value","count":1}',
        run_id=run_id,
    )
    rows = callback.drain()
    assert len(rows) == 1
    assert rows[0]["tool"] == "web_search"
    assert rows[0]["status"] == "ok"
    assert "private-value" not in json.dumps(rows)
    assert rows[0]["duration_ms"] >= 0


async def test_core_bridge_allowlists_fail_closed_without_tool_context() -> None:
    from app.ai.tools.core_bridge import get_core_tools_for_agent
    from app.core.llm.mock_client import sample_from_schema

    for agent in ("nour", "layla", "karim", "salma"):
        tools = get_core_tools_for_agent(agent)
        assert tools
        for tool in tools:
            sampled = sample_from_schema(tool.args_schema.model_json_schema())
            required = {
                name: sampled[name]
                for name, field in tool.args_schema.model_fields.items()
                if field.is_required() and name in sampled
            }
            args = tool.args_schema.model_validate(required).model_dump(mode="json")
            result = await tool.ainvoke(args, config={"configurable": {"business_id": "biz_test"}})
            parsed = json.loads(result)
            assert parsed["status"] == "unavailable"
            assert parsed["unavailable_reason"]


async def test_optional_nour_gui_graph_uses_the_selected_provider_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    from langchain_core.messages import HumanMessage

    from app.schemas.blocks import MessageWithBlocks, TextBlock

    module = importlib.import_module("app.ai.agents.nour.gui_graph")
    monkeypatch.setattr(module, "get_ai_settings", lambda: AISettings(llm_provider=LLMProvider.MOCK))

    class FakeRunnable:
        def invoke(self, messages: list[Any], config: Any = None) -> MessageWithBlocks:
            assert any(getattr(message, "type", None) == "human" for message in messages)
            return MessageWithBlocks(blocks=[TextBlock(text="Offline UI block.")])

    fake_handle = SimpleNamespace(structured=lambda _schema: FakeRunnable())
    monkeypatch.setattr(module, "get_llm", lambda **_kwargs: fake_handle)
    state = await module.graph.ainvoke(
        {"messages": [HumanMessage(content="Format this test result.")], "final_blocks": None}
    )
    assert state["final_blocks"].blocks[0].text == "Offline UI block."


async def test_nour_graph_uses_brand_memory_static_website_and_labeled_mock_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.messages import AIMessage

    from app.ai.agents.nour import graph as nour_module
    from app.ai.tools import analyze_website as website_module
    from app.ai.tools import brand_context as brand_context_module

    html = (Path(__file__).parent / "fixtures" / "marketing_page.html").read_text(encoding="utf-8")

    async def fetch_static(_url: str) -> tuple[str, str]:
        return html, "https://river-roast.example/"

    monkeypatch.setattr(website_module, "_fetch_public_html", fetch_static)
    memory = InMemoryStore()
    monkeypatch.setattr(brand_context_module, "current_store", lambda: memory)
    monkeypatch.setattr(nour_module, "get_ai_settings", lambda: AISettings(llm_provider=LLMProvider.MOCK))

    class ScriptedChat:
        def __init__(self) -> None:
            self.responses = [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "get_brand_context_tool", "args": {}, "id": "nour-brand", "type": "tool_call"}
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "analyze_website_tool",
                            "args": {"url": "https://river-roast.example/"},
                            "id": "nour-site",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "web_search_tool",
                            "args": {"query": "River Roast Egypt official products", "max_results": 2},
                            "id": "nour-search",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "submit_research_report",
                            "args": {
                                "market_overview": "Static fixture evidence only.",
                                "competitor_summary": [
                                    {
                                        "brand": "River Roast",
                                        "positioning": "Office coffee subscriptions.",
                                        "strengths": ["A weekly subscription offer is stated."],
                                        "weaknesses_or_gaps": [],
                                        "evidence": [
                                            {
                                                "claim": "The office coffee box is listed at EGP 350.",
                                                "source_url": "https://river-roast.example/",
                                                "observed_text": "EGP 350 per box.",
                                                "source_type": "website",
                                            }
                                        ],
                                    }
                                ],
                                "pricing_insights": ["EGP 350 per box is explicitly shown on the page."],
                                "offer_insights": [],
                                "shipping_insights": ["The page says weekly delivery in Cairo."],
                                "search_insights": [],
                                "social_insights": [],
                                "opportunities_for_target_brand": [],
                                "risks": [],
                                "evidence": [],
                                "missing_inputs": [],
                                "assumptions": [],
                            },
                            "id": "nour-submit",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
            self.inputs: list[list[Any]] = []
            self.bound_tools: list[str] = []

        def bind_tools(self, tools: list[Any], **_kwargs: Any) -> ScriptedChat:
            self.bound_tools = [tool.name for tool in tools]
            return self

        def invoke(self, messages: list[Any], config: Any = None) -> AIMessage:
            self.inputs.append(list(messages))
            return self.responses.pop(0)

    scripted = ScriptedChat()
    monkeypatch.setattr(nour_module, "get_llm", lambda **_kwargs: SimpleNamespace(chat=scripted))

    state = await nour_module.nour_graph.ainvoke(
        {"messages": [HumanMessage(content="Research this supplied website and market.")]},
        {"configurable": {"thread_id": "nour-static", "business_id": "biz-static"}},
    )
    assert not scripted.responses
    assert {"get_brand_context_tool", "analyze_website_tool", "web_search_tool"} <= set(scripted.bound_tools)
    tool_messages = [message for message in state["messages"] if getattr(message, "type", None) == "tool"]
    brand_result = json.loads(next(m.content for m in tool_messages if m.name == "get_brand_context_tool"))
    site_result = json.loads(next(m.content for m in tool_messages if m.name == "analyze_website_tool"))
    search_result = json.loads(next(m.content for m in tool_messages if m.name == "web_search_tool"))
    assert brand_result["status"] == "partial"
    assert site_result["status"] == "ok"
    assert site_result["offers_and_prices"]
    assert search_result["status"] == "partial"
    assert "placeholder" in search_result["unavailable_reason"].lower()
    assert state["messages"][-1].name == "submit_research_report"


def test_reference_upload_checks_type_size_and_path_without_calling_vlm(tmp_path: Path) -> None:
    from app.core.tools.media import _load_reference_image
    from app.schemas.tools import AnalyzeReferenceInput

    upload = tmp_path / "reference.png"
    upload.write_bytes(b"\x89PNG\r\n\x1a\n" + b"static-image-payload")
    settings = SimpleNamespace(
        upload_dir=str(tmp_path),
        website_max_bytes=1024,
        website_fetch_timeout_seconds=1,
    )
    mime, encoded, label = asyncio.run(
        _load_reference_image(AnalyzeReferenceInput(upload_id="reference.png"), settings)
    )
    assert mime == "image/png"
    assert encoded
    assert label == "upload:reference.png"

    with pytest.raises(ToolUnavailable, match="invalid"):
        asyncio.run(_load_reference_image(AnalyzeReferenceInput(upload_id="../secret.png"), settings))
