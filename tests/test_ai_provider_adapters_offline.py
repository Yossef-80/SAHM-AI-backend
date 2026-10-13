"""Offline contract tests for every app.ai provider family and its logging/memory edges.

Provider SDKs, external HTTP calls, and PostgreSQL are replaced with in-process
fakes. No model, paid search/review API, website, or database server is contacted.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import SecretStr, ValidationError

from app.ai.enums import (
    DatabaseProvider,
    LLMProvider,
    LogFormat,
    LogKind,
    SearchDepth,
    SearchProvider,
    StructuredOutputMethod,
)
from app.ai.errors import ProviderNotConfiguredError, ProviderRequestError
from app.ai.settings import AISettings


@pytest.mark.parametrize(
    ("adapter_path", "sdk_module", "sdk_symbol", "settings_values", "method", "secret"),
    [
        (
            "app.ai.providers.llm.ollama",
            "langchain_ollama",
            "ChatOllama",
            {
                "ollama_base_url": "https://ollama.example.test/",
                "ollama_model": "ollama-fixture-model",
                "ollama_num_ctx": 2048,
                "ollama_keep_alive": "10m",
                "ollama_reasoning": False,
                "ollama_timeout_seconds": 17,
            },
            StructuredOutputMethod.JSON_SCHEMA,
            False,
        ),
        (
            "app.ai.providers.llm.openai",
            "langchain_openai",
            "ChatOpenAI",
            {"openai_api_key": "offline-openai-key", "openai_model": "openai-fixture-model"},
            StructuredOutputMethod.FUNCTION_CALLING,
            False,
        ),
        (
            "app.ai.providers.llm.anthropic",
            "langchain_anthropic",
            "ChatAnthropic",
            {
                "anthropic_api_key": "offline-anthropic-key",
                "anthropic_model": "anthropic-fixture-model",
            },
            StructuredOutputMethod.FUNCTION_CALLING,
            False,
        ),
        (
            "app.ai.providers.llm.gemini",
            "langchain_google_genai",
            "ChatGoogleGenerativeAI",
            {"gemini_api_key": "offline-gemini-key", "gemini_model": "gemini-fixture-model"},
            StructuredOutputMethod.FUNCTION_CALLING,
            True,
        ),
        (
            "app.ai.providers.llm.groq",
            "langchain_groq",
            "ChatGroq",
            {"groq_api_key": "offline-groq-key", "groq_model": "groq-fixture-model"},
            StructuredOutputMethod.FUNCTION_CALLING,
            True,
        ),
        (
            "app.ai.providers.llm.openrouter",
            "langchain_openai",
            "ChatOpenAI",
            {
                "openrouter_api_key": "offline-openrouter-key",
                "openrouter_model": "openrouter-fixture-model",
            },
            StructuredOutputMethod.JSON_SCHEMA,
            True,
        ),
    ],
)
def test_remote_llm_adapters_only_construct_configured_sdk_clients(
    adapter_path: str,
    sdk_module: str,
    sdk_symbol: str,
    settings_values: dict[str, Any],
    method: StructuredOutputMethod,
    secret: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All provider builders forward configuration without invoking the SDK."""
    captured: dict[str, Any] = {}
    fake_sdk = ModuleType(sdk_module)

    def fake_constructor(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(model=kwargs.get("model"), kwargs=kwargs)

    setattr(fake_sdk, sdk_symbol, fake_constructor)
    monkeypatch.setitem(sys.modules, sdk_module, fake_sdk)

    settings = AISettings(
        _env_file=None,
        llm_temperature=0.6,
        llm_max_tokens=321,
        llm_timeout_seconds=9,
        llm_max_retries=2,
        **settings_values,
    )
    adapter = importlib.import_module(adapter_path)
    model = adapter.build_chat_model(settings, model=None, temperature=None)

    expected_model = next(value for key, value in settings_values.items() if key.endswith("_model"))
    assert model.model == expected_model
    assert adapter.DEFAULT_STRUCTURED_METHOD == method
    assert captured["model"] == expected_model
    assert captured["temperature"] == 0.6
    token_limit_key = (
        "max_tokens"
        if adapter_path.endswith(("openai", "anthropic", "gemini", "groq", "openrouter"))
        else "num_predict"
    )
    assert captured[token_limit_key] == 321

    if adapter_path.endswith("ollama"):
        assert captured["base_url"] == "https://ollama.example.test"
        assert captured["num_ctx"] == 2048
        assert captured["keep_alive"] == "10m"
        assert captured["reasoning"] is False
        assert captured["client_kwargs"]["timeout"] == 17
        assert captured["async_client_kwargs"] == captured["client_kwargs"]
        assert captured["client_kwargs"]["headers"]["ngrok-skip-browser-warning"] == "true"
    else:
        assert captured.get("timeout", captured.get("request_timeout")) == 9
        if adapter_path.endswith("openai"):
            assert captured["api_key"] == "offline-openai-key"
        else:
            value = captured["api_key"]
            assert isinstance(value, SecretStr) is secret
            expected_key = f"offline-{adapter_path.rsplit('.', 1)[-1]}-key"
            if secret:
                assert value.get_secret_value() == expected_key
            else:
                assert value == expected_key
        if adapter_path.endswith("openrouter"):
            assert captured["base_url"] == "https://openrouter.ai/api/v1"
            assert captured["default_headers"] == {"X-Title": "Sahm Marketing OS"}


def test_provider_loader_reports_unregistered_choice_without_fallback() -> None:
    from app.ai.providers._loader import load_provider_module

    with pytest.raises(ProviderNotConfiguredError, match="No llm provider module is registered"):
        load_provider_module({}, LLMProvider.MOCK, "llm")


def test_db_and_search_facades_use_the_selected_offline_provider_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.ai.providers import db as db_facade
    from app.ai.providers import search as search_facade

    db_settings = AISettings(_env_file=None, db_provider=DatabaseProvider.MEMORY)
    search_settings = AISettings(_env_file=None, search_provider=SearchProvider.MOCK)
    monkeypatch.setattr(db_facade, "get_ai_settings", lambda: db_settings)
    monkeypatch.setattr(search_facade, "get_ai_settings", lambda: search_settings)
    db_facade.reset_db_providers()
    search_facade.reset_search_providers()
    try:
        store = db_facade.get_store()
        store.put(("brands", "biz-offline"), "profile", {"name": "Fixture"})
        assert db_facade.get_store() is store
        assert db_facade.current_store().get(("brands", "biz-offline"), "profile").value == {
            "name": "Fixture"
        }
        assert db_facade.get_checkpointer() is not None

        provider = search_facade.get_search_provider()
        assert provider.name == "mock"
        assert provider.is_placeholder is True
        assert search_facade.get_search_provider() is provider
    finally:
        db_facade.reset_db_providers()
        search_facade.reset_search_providers()


def test_ai_redaction_covers_bare_tokens_without_masking_usage_counters() -> None:
    from app.ai.security import redact_sensitive_data

    raw = {
        "token": "offline-token-secret",
        "token_count": 3,
        "input_tokens": 14,
        "summary": "token=offline-inline-token-secret",
    }
    safe = redact_sensitive_data(raw)

    assert safe["token"] == "[REDACTED]"
    assert safe["token_count"] == 3
    assert safe["input_tokens"] == 14
    assert "offline-inline-token-secret" not in safe["summary"]


def test_settings_validate_provider_enums_and_normalize_legacy_ollama_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("OLLAMA_BASE_URL", "NGROK_URL", "OLLAMA_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI_LLM_PROVIDER", "mock")
    monkeypatch.setenv("AI_SEARCH_PROVIDER", "none")
    monkeypatch.setenv("AI_DB_PROVIDER", "memory")
    monkeypatch.setenv("NGROK_URL", "  https://ollama.example.test/tunnel///  ")

    settings = AISettings(_env_file=None)
    assert settings.llm_provider == LLMProvider.MOCK
    assert settings.search_provider == SearchProvider.NONE
    assert settings.db_provider == DatabaseProvider.MEMORY
    assert settings.ollama_base_url == "https://ollama.example.test/tunnel"

    with monkeypatch.context() as patch:
        patch.setenv("AI_LLM_PROVIDER", "not-a-provider")
        with pytest.raises(ValidationError):
            AISettings(_env_file=None)


def test_tavily_adapter_uses_only_a_mocked_http_client_and_bounds_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.ai.providers.search.tavily import TavilySearchProvider

    request: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "results": [
                    {
                        "title": "Fixture source",
                        "url": "https://example.test/fixture",
                        "content": "0123456789abcdefghijkl",
                        "published_date": "2026-01-02",
                        "score": 0.91,
                    },
                    "ignored_non_object_row",
                ]
            }

    class FakeClient:
        def __init__(self, *, timeout: float, trust_env: bool) -> None:
            request["timeout"] = timeout
            request["trust_env"] = trust_env

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def post(self, url: str, *, json: dict[str, Any]) -> FakeResponse:
            request["url"] = url
            request["json"] = json
            return FakeResponse()

    import app.ai.providers.search.tavily as tavily_module

    monkeypatch.setattr(tavily_module, "httpx", SimpleNamespace(Client=FakeClient))
    provider = TavilySearchProvider(
        AISettings(
            _env_file=None,
            tavily_api_key="offline-tavily-key",
            search_depth=SearchDepth.ADVANCED,
            search_timeout_seconds=4.5,
            search_snippet_chars=12,
        )
    )
    results = provider.search("coffee api_key=offline-query-secret", max_results=100)

    assert request["url"] == "https://api.tavily.com/search"
    assert request["timeout"] == 4.5 and request["trust_env"] is False
    assert request["json"]["api_key"] == "offline-tavily-key"
    assert request["json"]["query"] == "coffee api_key=[REDACTED]"
    assert request["json"]["max_results"] == 20
    assert request["json"]["search_depth"] == "advanced"
    assert len(results) == 1
    assert results[0].snippet == "0123456789ab"
    assert results[0].score == 0.91


def test_mock_and_disabled_search_adapters_are_truthful() -> None:
    from app.ai.providers.search.mock import build_search_provider
    from app.ai.providers.search.none import build_search_provider as build_disabled_search

    mock = build_search_provider(AISettings(_env_file=None, search_provider=SearchProvider.MOCK))
    assert mock.is_placeholder is True
    assert len(mock.search("coffee in Cairo", max_results=5)) == 3
    assert mock.search("coffee in Cairo")[0].url == mock.search("coffee in Cairo")[0].url

    disabled_settings = AISettings(_env_file=None, search_provider=SearchProvider.NONE)
    disabled = build_disabled_search(disabled_settings)
    with pytest.raises(ProviderNotConfiguredError, match="search is disabled"):
        disabled.search("coffee")


def test_google_places_adapter_maps_fixture_response_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.ai.providers import reviews

    captured: dict[str, Any] = {}

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "places": [
                    {
                        "id": "place-other",
                        "displayName": {"text": "River Coffee"},
                        "rating": 4.1,
                        "userRatingCount": 8,
                        "reviews": [],
                    },
                    {
                        "id": "place-river-roast",
                        "displayName": {"text": "River Roast"},
                        "formattedAddress": "Cairo, Egypt",
                        "googleMapsUri": "https://maps.example.test/place-river-roast",
                        "rating": 4.8,
                        "userRatingCount": 25,
                        "reviews": [
                            {
                                "rating": 5,
                                "text": {"text": "Fixture review text."},
                                "relativePublishTimeDescription": "2 weeks ago",
                            }
                        ],
                    },
                ]
            },
        )

    monkeypatch.setattr(reviews.httpx, "post", fake_post)
    settings = AISettings(
        _env_file=None,
        google_places_api_key="offline-places-key",
        reviews_timeout_seconds=7,
    )
    provider = reviews.GooglePlacesProvider(settings)
    places = provider.search("River Roast", "Cairo, Egypt", "en")

    assert captured["url"] == "https://places.googleapis.com/v1/places:searchText"
    assert captured["headers"]["X-Goog-Api-Key"] == "offline-places-key"
    assert "places.reviews" in captured["headers"]["X-Goog-FieldMask"]
    assert captured["json"] == {
        "textQuery": "River Roast, Cairo, Egypt",
        "languageCode": "en",
        "maxResultCount": 5,
    }
    assert captured["timeout"] == 7
    assert places[0].place_id == "place-river-roast"
    assert places[0].match_score == 1.0
    assert places[0].reviews[0].text == "Fixture review text."
    assert places[0].reviews[0].relative_time == "2 weeks ago"
    assert places[1].match_score < places[0].match_score


def test_apify_reviews_adapter_uses_fake_responses_and_maps_actor_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.ai.providers import reviews

    requests: list[dict[str, Any]] = []
    response_bodies = [
        [
            {
                "title": "River Roast",
                "placeId": "fixture-place-id",
                "address": "Cairo, Egypt",
                "url": "https://maps.example.test/river-roast",
                "totalScore": 4.6,
                "reviewsCount": 19,
                "reviewsData": [
                    {
                        "stars": "5",
                        "reviewText": "Fixture Apify review.",
                        "publishedAtDate": "2026-02-01",
                    }
                ],
            }
        ],
        [{"stars": 4, "text": "Fetched fixture review.", "timeAgo": "last month"}],
    ]

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        requests.append({"url": url, **kwargs})
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json=response_bodies.pop(0),
        )

    monkeypatch.setattr(reviews.httpx, "post", fake_post)
    provider = reviews.ApifyReviewsProvider(
        AISettings(
            _env_file=None,
            apify_api_token="offline-apify-token",
            google_maps_reviews_actor_id="compass/crawler-google-places",
            reviews_timeout_seconds=11,
        )
    )
    places = provider.search(
        "River Roast", "Cairo", max_reviews=2, sort="newest", language="ar"
    )
    fetched = provider.fetch("fixture place", max_reviews=1, sort=None, language="en")

    assert len(requests) == 2
    assert "/acts/compass~crawler-google-places/run-sync-get-dataset-items" in requests[0]["url"]
    assert requests[0]["headers"]["Authorization"] == "Bearer offline-apify-token"
    assert requests[0]["params"] == {"format": "json"}
    assert requests[0]["json"]["searchStringsArray"] == ["River Roast"]
    assert requests[0]["json"]["locationQuery"] == "Cairo"
    assert requests[0]["json"]["reviewsSort"] == "newest"
    assert requests[0]["json"]["language"] == "ar"
    assert requests[0]["json"]["scrapeReviewsPersonalData"] is False
    assert "fixture%20place" in requests[1]["json"]["startUrls"][0]["url"]
    assert places[0].place_id == "fixture-place-id"
    assert places[0].match_score == 1.0
    assert places[0].reviews[0].rating == 5.0
    assert places[0].reviews[0].text == "Fixture Apify review."
    assert fetched[0].rating == 4.0
    assert fetched[0].relative_time == "last month"


def test_postgres_provider_uses_injected_connection_and_fake_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.ai.providers.db import postgres

    connection = object()
    monkeypatch.setattr(postgres, "_connect", lambda _settings: connection)
    checkpoint_module = ModuleType("langgraph.checkpoint.postgres")
    store_module = ModuleType("langgraph.store.postgres")
    made: dict[str, list[Any]] = {"checkpoint": [], "store": []}

    class FakePostgresSaver:
        def __init__(self, conn: Any) -> None:
            self.connection = conn
            self.setup_called = False
            made["checkpoint"].append(self)

        def setup(self) -> None:
            self.setup_called = True

    class FakePostgresStore:
        def __init__(self, conn: Any) -> None:
            self.connection = conn
            self.setup_called = False
            made["store"].append(self)

        def setup(self) -> None:
            self.setup_called = True

    checkpoint_module.PostgresSaver = FakePostgresSaver  # type: ignore[attr-defined]
    store_module.PostgresStore = FakePostgresStore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", checkpoint_module)
    monkeypatch.setitem(sys.modules, "langgraph.store.postgres", store_module)

    settings = AISettings(
        _env_file=None,
        db_provider=DatabaseProvider.POSTGRES,
        postgres_url="postgresql://offline.invalid/db",
    )
    saver = postgres.build_checkpointer(settings)
    store = postgres.build_store(settings)

    assert saver.connection is connection and saver.setup_called is True
    assert store.connection is connection and store.setup_called is True
    assert made["checkpoint"] == [saver]
    assert made["store"] == [store]


@pytest.mark.parametrize("log_format", [LogFormat.JSON, LogFormat.TEXT])
def test_ai_component_logger_redacts_credentials_but_keeps_usage_fields(
    log_format: LogFormat,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.ai.loggers.base as log_base

    settings = AISettings(
        _env_file=None,
        log_dir=str(tmp_path),
        log_format=log_format,
        log_to_console=False,
    )
    monkeypatch.setattr(log_base, "get_ai_settings", lambda: settings)
    name = f"offline-contract-{log_format.value.lower()}"
    logger = log_base.get_component_logger(LogKind.PROVIDER, name)
    logger.info(
        "adapter_completed",
        token="offline-token-secret",
        api_key="offline-api-secret",
        input_tokens=23,
        nested={"access_token": "offline-nested-secret"},
    )

    text = (tmp_path / "providers" / f"{name}.log").read_text(encoding="utf-8")
    assert "offline-token-secret" not in text
    assert "offline-api-secret" not in text
    assert "offline-nested-secret" not in text
    assert "23" in text
    if log_format == LogFormat.JSON:
        row = json.loads(text)
        assert row["event"] == "adapter_completed"
        assert row["input_tokens"] == 23
        assert row["token"] == "[REDACTED]"
    else:
        assert "adapter_completed" in text
        assert "input_tokens=23" in text


def test_ai_logging_callback_routes_tool_and_llm_lifecycle_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.ai.loggers.base as log_base
    from app.ai.loggers.callbacks import AILoggingCallback

    settings = AISettings(_env_file=None, log_dir=str(tmp_path), log_format=LogFormat.JSON)
    monkeypatch.setattr(log_base, "get_ai_settings", lambda: settings)
    callback = AILoggingCallback(agent="nour")

    tool_run = uuid4()
    callback.on_tool_start(
        {"name": "fixture_tool"},
        "",
        run_id=tool_run,
        inputs={"token": "offline-tool-secret"},
    )
    callback.on_tool_end(
        '{"status":"ok","access_token":"offline-output-secret"}',
        run_id=tool_run,
    )

    llm_run = uuid4()
    callback.on_chat_model_start(
        {"name": "fixture-model"},
        [[HumanMessage(content="offline only")]],
        run_id=llm_run,
        invocation_params={"model": "mock-fixture-model"},
        metadata={"langgraph_node": "agent"},
    )
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="offline response",
                        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                    )
                )
            ]
        ]
    )
    callback.on_llm_end(response, run_id=llm_run)

    tool_log = (tmp_path / "tools" / "fixture_tool.log").read_text(encoding="utf-8")
    llm_log = (tmp_path / "providers" / "llm.log").read_text(encoding="utf-8")
    assert "tool_started" in tool_log and "tool_finished" in tool_log
    assert "offline-tool-secret" not in tool_log
    assert "offline-output-secret" not in tool_log
    assert "llm_started" in llm_log and "llm_finished" in llm_log
    assert "mock-fixture-model" in llm_log
    assert '"input_tokens": 7' in llm_log


def test_history_compressor_keeps_roles_and_compacts_only_long_assistant_turns() -> None:
    from app.ai.memory.compressor import compress_history_messages

    history = [
        {"role": "user", "content": "Keep this exact user message."},
        {
            "role": "assistant",
            "content": (
                "# Summary\n"
                + ("Long detail.\n" * 80)
                + "- One key fact\n- Another key fact"
            ),
            "persona_id": "nour",
        },
    ]
    compressed = compress_history_messages(history, max_chars=80)

    assert compressed[0] == history[0]
    assert compressed[1]["role"] == "assistant"
    assert compressed[1]["persona_id"] == "nour"
    assert "Summary of previous AI response" in compressed[1]["content"]
    assert "One key fact" in compressed[1]["content"]
    assert len(history[1]["content"]) > len(compressed[1]["content"])


async def test_session_memory_uses_mock_summary_and_redacts_before_persisting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.ai.memory.compressor as compressor
    import app.ai.settings as settings_module

    class FakeChat:
        received: list[Any] = []

        async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
            self.received = messages
            return SimpleNamespace(content="Updated summary api_key=offline-summary-secret")

    fake_chat = FakeChat()
    monkeypatch.setattr(compressor, "get_llm", lambda: SimpleNamespace(chat=fake_chat))
    monkeypatch.setattr(
        settings_module,
        "get_ai_settings",
        lambda: AISettings(_env_file=None, memory_retain_recent_turns=2),
    )
    flushes: list[bool] = []

    async def flush() -> None:
        flushes.append(True)

    db = SimpleNamespace(flush=flush)
    conversation = SimpleNamespace(id="conv-memory-fixture", summary="Earlier facts.")
    messages = [
        SimpleNamespace(
            role="user",
            content="The brand sells coffee.",
            persona_id="nour",
            blocks=[],
        ),
        SimpleNamespace(
            role="assistant",
            content="The audience is office managers.",
            persona_id="nour",
            blocks=[],
        ),
        SimpleNamespace(
            role="user",
            content="Use api_key=offline-input-secret only in a test.",
            persona_id="omar",
            blocks=[{"token": "offline-block-secret"}],
        ),
        SimpleNamespace(
            role="assistant",
            content="The CTA is a weekly subscription.",
            persona_id="omar",
            blocks=[],
        ),
    ]

    history, summary = await compressor.manage_session_memory_with_token_cap(
        db, conversation, messages, max_tokens=32
    )

    prompt = "\n".join(str(message.content) for message in fake_chat.received)
    assert "offline-input-secret" not in prompt
    assert "offline-input-secret" not in json.dumps(history)
    assert "offline-block-secret" not in json.dumps(history)
    assert "offline-summary-secret" not in summary
    assert "[REDACTED]" in summary
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert conversation.summary == summary
    assert flushes == [True]


async def test_memory_debouncer_runs_a_fake_compressor_without_sleeping_for_real_time() -> None:
    from app.ai.memory.compressor import MemoryDebouncer

    calls: list[str] = []

    class FakeCompressor:
        async def update_global_profile(self, business_id: str) -> None:
            calls.append(business_id)

    debouncer = MemoryDebouncer(delay_seconds=0)
    debouncer.compressor = FakeCompressor()  # type: ignore[assignment]
    debouncer.schedule_ltm_update("biz-offline")
    task = debouncer._timers["biz-offline"]
    await task

    assert calls == ["biz-offline"]
    assert debouncer._timers == {}
