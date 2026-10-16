"""The token/image controls, and the wiring that makes them real.

Three claims are verified here, in order of how easily they break silently:

1. A workspace's saved LLM settings *reach the chat turn*. Before this round
   ``/settings/llm`` was read by the task/agent runtime only, while the assistant
   built its model from the process-wide ``AI_LLM_PROVIDER`` -- so switching the
   model in the UI changed nothing in chat.
2. The sampling controls (``maxTokens`` / ``temperature`` / ``reasoningEffort``)
   survive the whole path: settings API -> column -> resolution -> provider call
   arguments, and ``null`` clears an override instead of writing a zero.
3. The image connector's ``quality`` is validated on write and is the value that
   reaches the provider request body.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.providers import llm as llm_factory
from app.core.llm.model_compat import AVAILABLE_REASONING_EFFORTS
from app.core.llm.router import (
    invalidate_workspace_llm_cache,
    resolve_workspace_tier,
)
from app.db.models import Business, LlmProviderSettings

pytestmark = pytest.mark.anyio

HEADERS = {"X-Workspace-Id": "biz_test"}


@pytest.fixture()
def clean_llm_cache():
    """The resolution layer caches rows for 60s; a test must not inherit one."""
    invalidate_workspace_llm_cache()
    yield
    invalidate_workspace_llm_cache()


def _recording_provider(captured: dict[str, Any], response_text: str = "hello from the workspace model"):
    """A stand-in provider module: records what ``get_llm`` asked it to build."""

    class FakeChat:
        model = "fake-chat"

        async def ainvoke(self, messages: Any, **_kwargs: Any) -> Any:
            captured["messages"] = messages
            return SimpleNamespace(content=response_text)

        def with_structured_output(self, _schema: Any) -> Any:  # pragma: no cover
            raise AssertionError("this test forces the agent, so no routing model is needed")

    from app.ai.enums import StructuredOutputMethod

    class FakeModule:
        DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.JSON_SCHEMA

        @staticmethod
        def build_chat_model(settings: Any, **kwargs: Any) -> Any:
            captured["kwargs"] = kwargs
            captured["settings"] = settings
            return FakeChat()

    return FakeModule


# ------------------------------------------------- 1. resolution returns them ----


async def test_resolve_workspace_tier_returns_the_sampling_controls(
    seeded_db: AsyncSession, clean_llm_cache: None
) -> None:
    seeded_db.add(
        LlmProviderSettings(
            id="llm_1",
            business_id="biz_test",
            strong_provider="openai",
            strong_model="gpt-5.6-terra",
            strong_max_tokens=1200,
            strong_temperature=0.55,
            strong_reasoning_effort="high",
        )
    )
    await seeded_db.commit()
    invalidate_workspace_llm_cache()

    values = await resolve_workspace_tier(seeded_db, "biz_test", "strong")

    assert values["provider"] == "openai"
    assert values["model"] == "gpt-5.6-terra"
    assert values["max_tokens"] == 1200
    assert values["temperature"] == 0.55
    assert values["reasoning_effort"] == "high"


async def test_a_workspace_without_an_override_reports_no_sampling_values(
    seeded_db: AsyncSession, clean_llm_cache: None
) -> None:
    """``None`` means "process default" -- never a zero that would be sent."""
    values = await resolve_workspace_tier(seeded_db, "biz_test", "strong")
    assert values["max_tokens"] is None
    assert values["temperature"] is None
    assert values["reasoning_effort"] is None


# ------------------------------- 2. the settings API stores and returns them ----


async def test_llm_settings_round_trip_includes_the_sampling_controls(
    api_client: Any, clean_llm_cache: None
) -> None:
    saved = await api_client.patch(
        "/api/v1/settings/llm",
        headers=HEADERS,
        json={
            "strong": {
                "provider": "openai",
                "model": "gpt-5.6-terra",
                "apiKey": "sk-secret",
                "maxTokens": 2048,
                "temperature": 0.7,
                "reasoningEffort": "LOW",
            }
        },
    )
    assert saved.status_code == 200, saved.text
    body = saved.json()
    assert body["strong"]["maxTokens"] == 2048
    assert body["strong"]["temperature"] == 0.7
    # Normalised to lower case on write, like the provider field.
    assert body["strong"]["reasoningEffort"] == "low"
    assert "sk-secret" not in saved.text

    reread = await api_client.get("/api/v1/settings/llm", headers=HEADERS)
    assert reread.json()["strong"]["reasoningEffort"] == "low"
    assert reread.json()["strong"]["maxTokens"] == 2048


async def test_clearing_a_sampling_control_sends_null_and_clears_it(
    api_client: Any, clean_llm_cache: None
) -> None:
    await api_client.patch(
        "/api/v1/settings/llm",
        headers=HEADERS,
        json={"strong": {"provider": "openai", "model": "gpt-4o-mini", "maxTokens": 999}},
    )
    cleared = await api_client.patch(
        "/api/v1/settings/llm",
        headers=HEADERS,
        json={"strong": {"maxTokens": None}},
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["strong"]["maxTokens"] is None


@pytest.mark.parametrize("bad", ["xhigh", "minimal", "HIGHER", "none "])
async def test_an_unsupported_reasoning_effort_is_rejected(
    api_client: Any, clean_llm_cache: None, bad: str
) -> None:
    """``HIGHER``/``none `` normalise; genuinely unknown values are a 422."""
    response = await api_client.patch(
        "/api/v1/settings/llm",
        headers=HEADERS,
        json={"strong": {"reasoningEffort": bad}},
    )
    if bad.strip().lower() in AVAILABLE_REASONING_EFFORTS:
        assert response.status_code == 200, response.text
        assert response.json()["strong"]["reasoningEffort"] == bad.strip().lower()
    else:
        assert response.status_code == 422, response.text


# ------------------------------------- 3. the chat turn uses the saved settings ----


async def test_chat_turn_builds_its_model_from_the_workspace_settings(
    api_client: Any,
    seeded_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    clean_llm_cache: None,
) -> None:
    """The assistant, not just the task runtime, honours Settings -> LLM.

    ``forced_agent="chat"`` skips the routing model so the only provider call is
    the answer itself, which is what makes the assertion below unambiguous.
    """
    seeded_db.add(
        LlmProviderSettings(
            id="llm_chat",
            business_id="biz_test",
            strong_provider="openai",
            strong_model="gpt-5.6-terra",
            strong_base_url="https://workspace-gateway.invalid/v1",
            strong_max_tokens=1500,
            strong_temperature=0.9,
            strong_reasoning_effort="none",
        )
    )
    await seeded_db.commit()
    invalidate_workspace_llm_cache()

    captured: dict[str, Any] = {}
    monkeypatch.setattr(llm_factory, "load_provider_module", lambda *_a, **_k: _recording_provider(captured))

    from app.ai.agents.orchestrator import brain

    result = await brain.orchestrate(
        "من أنت؟",
        history=[],
        business_id="biz_test",
        workspace_id="ws_test",
        db=seeded_db,
        forced_agent="chat",
    )

    kwargs = captured["kwargs"]
    assert kwargs["model"] == "gpt-5.6-terra"
    assert kwargs["max_tokens"] == 1500
    assert kwargs["temperature"] == 0.9
    assert kwargs["reasoning_effort"] == "none"
    assert captured["settings"].openai_base_url == "https://workspace-gateway.invalid/v1"
    # The turn itself completed with the workspace model's answer.
    assert "hello from the workspace model" in str(getattr(result, "message", result))


async def test_a_workspace_on_mock_does_not_downgrade_a_configured_process(
    seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch, clean_llm_cache: None
) -> None:
    """A ``mock`` row means "nothing configured", not "serve placeholders"."""
    seeded_db.add(
        LlmProviderSettings(
            id="llm_mock",
            business_id="biz_test",
            strong_provider="mock",
            strong_model="mock-strong-1",
        )
    )
    await seeded_db.commit()
    invalidate_workspace_llm_cache()

    monkeypatch.setenv("AI_LLM_PROVIDER", "openai")
    monkeypatch.setenv("AI_LLM_MODEL", "gpt-4o-mini")
    from app.ai.settings import get_ai_settings

    get_ai_settings.cache_clear()

    captured: dict[str, Any] = {}
    monkeypatch.setattr(llm_factory, "load_provider_module", lambda *_a, **_k: _recording_provider(captured))

    await llm_factory.get_workspace_chat_llm(db=seeded_db, business_id="biz_test", tier="strong")

    assert captured["kwargs"]["model"] == "gpt-4o-mini"
    get_ai_settings.cache_clear()


async def test_chat_falls_back_to_the_process_model_when_settings_are_unreadable(
    seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch, clean_llm_cache: None
) -> None:
    """A broken settings read must not take the turn down."""

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("settings table unavailable")

    monkeypatch.setenv("AI_LLM_PROVIDER", "openai")
    monkeypatch.setenv("AI_LLM_MODEL", "process-default-model")
    from app.ai.settings import get_ai_settings

    get_ai_settings.cache_clear()

    monkeypatch.setattr("app.core.llm.router.resolve_workspace_tier", _boom)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(llm_factory, "load_provider_module", lambda *_a, **_k: _recording_provider(captured))

    handle = await llm_factory.get_workspace_chat_llm(
        db=seeded_db, business_id="biz_test", tier="strong"
    )

    # Built from the process configuration instead of raising.
    assert handle is not None
    # The process configuration is tier-aware: AI_LLM_MODEL still names the base
    # model (and reaches the strong tier when nothing else is set), but an
    # explicit LLM_MODEL_STRONG wins. Assert against the resolved value rather
    # than a hardcoded string so the test never depends on a developer's .env.
    assert captured["kwargs"]["model"] == (get_ai_settings().model_for_tier("strong"))
    get_ai_settings.cache_clear()


# ------------------------------------------------- 4. image quality end to end ----


async def test_image_quality_is_published_as_a_closed_set(api_client: Any) -> None:
    response = await api_client.get("/api/v1/settings/connectors", headers=HEADERS)
    assert response.status_code == 200, response.text
    image = next(c for c in response.json()["connectors"] if c["connector"] == "image")
    quality = next(f for f in image["fields"] if f["key"] == "quality")
    # Published as choices, so the client renders a select and "HGIH" is untypable.
    assert quality["choices"] == ["auto", "low", "medium", "high"]
    assert quality["value"] == "auto"


async def test_image_quality_is_validated_on_write(api_client: Any) -> None:
    bad = await api_client.patch(
        "/api/v1/settings/connectors/image",
        headers=HEADERS,
        json={"provider": "openai", "quality": "HGIH"},
    )
    assert bad.status_code == 422, bad.text
    assert bad.json()["error"]["detail"]["accepted"] == ["auto", "low", "medium", "high"]

    good = await api_client.patch(
        "/api/v1/settings/connectors/image",
        headers=HEADERS,
        json={"provider": "openai", "quality": "LOW"},
    )
    assert good.status_code == 200, good.text
    stored = next(f for f in good.json()["fields"] if f["key"] == "quality")
    assert stored["value"] == "low"
    assert stored["source"] == "workspace"


async def test_the_env_quality_is_the_fallback_when_no_row_exists(
    seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No workspace row -> `IMAGE_QUALITY`, then the connector default `auto`."""
    from app.core.tools import media

    monkeypatch.setenv("IMAGE_API_KEY", "sk-env")
    monkeypatch.setenv("IMAGE_PROVIDER", "openai")
    monkeypatch.setenv("IMAGE_MODEL", "gpt-image-1")
    monkeypatch.setenv("IMAGE_QUALITY", "medium")
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    captured: dict[str, Any] = {}

    async def _fake_openai(prompt: str, ratio: str, **kwargs: Any) -> list[Any]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(media, "_generate_openai_image", _fake_openai)
    payload = SimpleNamespace(
        prompt="a coffee bag",
        aspect_ratio="1:1",
        style_spec={},
        reference_analysis_id=None,
        reference_image_url=None,
        count=1,
    )
    await media.generate_image_for_workspace(
        payload, db=seeded_db, business_id="biz_test", settings=settings
    )
    assert captured["quality"] == "medium"

    # And with nothing configured anywhere, the connector's own default applies
    # rather than an unset value reaching the provider.
    monkeypatch.delenv("IMAGE_QUALITY")
    get_settings.cache_clear()
    settings = get_settings()
    await media.generate_image_for_workspace(
        payload, db=seeded_db, business_id="biz_test", settings=settings
    )
    assert captured["quality"] == "auto"
    get_settings.cache_clear()


async def test_creatives_bridge_takes_the_quality_from_the_connector_row(
    seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch, test_settings: Any
) -> None:
    """Workspace row -> resolved connector -> provider request, for /creatives."""
    from app.core.tools import media
    from app.db.models import ToolSettings as ToolSettingsRow

    # The env has its own value: the workspace row must win over it, exactly as
    # it does for the model and the base URL.
    monkeypatch.setenv("IMAGE_API_KEY", "sk-env")
    monkeypatch.setenv("IMAGE_PROVIDER", "openai")
    monkeypatch.setenv("IMAGE_MODEL", "gpt-4o-mini-image")
    monkeypatch.setenv("IMAGE_BASE_URL", "https://env-provider.invalid/v1")
    monkeypatch.setenv("IMAGE_QUALITY", "low")
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    seeded_db.add(
        ToolSettingsRow(
            id="ts_img",
            business_id="biz_test",
            connector="image",
            image_provider="openai",
            image_model="gpt-image-1",
            base_url="https://ws-provider.invalid/v1",
            image_quality="high",
        )
    )
    await seeded_db.commit()

    captured: dict[str, Any] = {}

    async def _fake_openai(prompt: str, ratio: str, **kwargs: Any) -> list[Any]:
        captured.update({"prompt": prompt, "ratio": ratio, **kwargs})
        from app.schemas.tools import GeneratedAssetRecord

        return [
            GeneratedAssetRecord(
                id="crt_test_assets",
                url="https://cdn.example/out.png",
                prompt=prompt,
                provider="openai",
                model=str(kwargs.get("model") or ""),
                aspect_ratio=ratio,
            )
        ]

    monkeypatch.setattr(media, "_generate_openai_image", _fake_openai)

    payload = SimpleNamespace(
        prompt="a coffee bag on a marble counter",
        aspect_ratio="1:1",
        style_spec={},
        reference_analysis_id=None,
        reference_image_url=None,
        count=1,
    )
    await media.generate_image_for_workspace(
        payload, db=seeded_db, business_id="biz_test", settings=settings
    )

    assert captured["model"] == "gpt-image-1", "the workspace model must win"
    assert captured["base_url"] == "https://ws-provider.invalid/v1"
    assert captured["quality"] == "high", "the workspace quality must win over IMAGE_QUALITY"
    get_settings.cache_clear()
