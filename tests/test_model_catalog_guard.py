"""A model that does not exist must be caught when it is chosen, not at chat time.

Live sequence this pins: the model was set to ``gemini-3-flash``, which the API
answers with ``404 models/gemini-3-flash is not found`` -- the model list from the
same key has ``gemini-3.6-flash``, ``gemini-3.5-flash`` and more, but no
``gemini-3-flash``. The save succeeded, and every turn after it died.

The guard must also fail *open*: no key, no network, a provider without a listing
endpoint or a custom gateway are all "we cannot verify", never "refuse".
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.llm.model_catalog import (  # noqa: E402
    _extract,
    list_models,
    suggest_models,
)

AVAILABLE = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]


# ------------------------------------------------------------ suggestions -----


def test_the_family_first_suggestion_is_the_one_the_user_meant() -> None:
    suggestions = suggest_models("gemini-3-flash", AVAILABLE)

    assert suggestions[0] == "gemini-3.6-flash"
    assert "gemini-3.5-flash" in suggestions
    assert "gemini-2.5-flash" not in suggestions[:2]


def test_a_typo_still_gets_a_suggestion() -> None:
    assert suggest_models("gemnini-3.5-flsh", AVAILABLE)


def test_an_unrelated_name_gets_no_suggestions() -> None:
    assert suggest_models("totally-different-model", AVAILABLE) == []


def test_nothing_chosen_or_nothing_available_is_empty() -> None:
    assert suggest_models("", AVAILABLE) == []
    assert suggest_models("gemini-3-flash", []) == []


def test_suggestions_are_unique_and_bounded() -> None:
    many = [f"gemini-3.{n}-flash" for n in range(20)]

    suggestions = suggest_models("gemini-3-flash", many)

    assert len(suggestions) == len(set(suggestions))
    assert len(suggestions) <= 8


# ---------------------------------------------------------- the listing --------


def test_only_chat_models_are_listed_for_gemini() -> None:
    """An embedding or TTS model is not a chat model an agent can call tools on."""
    payload = {
        "models": [
            {"name": "models/gemini-3.6-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-2.5-flash-preview-tts", "supportedGenerationMethods": ["generateContent"]},
        ]
    }

    assert _extract("gemini", payload) == ["gemini-3.6-flash", "gemini-2.5-flash-preview-tts"]


def test_openai_shaped_listings_are_parsed() -> None:
    payload = {"data": [{"id": "meta-llama/llama-3.3-70b:free"}, {"id": "openai/gpt-4o"}]}

    assert _extract("openrouter", payload) == ["meta-llama/llama-3.3-70b:free", "openai/gpt-4o"]


@pytest.mark.parametrize(
    "args",
    [
        ("gemini", ""),                      # no credential: cannot verify
        ("unknown-provider", "key"),         # no listing endpoint
        ("gemini", "key"),                   # custom gateway below
    ],
)
def test_an_unverifiable_catalog_is_none(args: tuple[str, str]) -> None:
    provider, key = args
    base_url = "https://my-gateway.internal/v1" if provider == "gemini" else None

    assert asyncio.run(list_models(provider, key, base_url=base_url)) is None


def test_a_failed_listing_call_is_none_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Client:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, *args: object, **kwargs: object) -> object:
            raise httpx.ConnectError("no network")

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    assert asyncio.run(list_models("gemini", "key")) is None


def test_a_successful_listing_comes_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"models": [{"name": "models/gemini-3.6-flash", "supportedGenerationMethods": ["generateContent"]}]}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, *args: object, **kwargs: object) -> _Response:
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    assert asyncio.run(list_models("gemini", "key")) == ["gemini-3.6-flash"]


# ------------------------------------------------- the save-time guard --------


def test_the_settings_guard_runs_before_anything_is_written() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "api" / "settings.py").read_text(
        encoding="utf-8"
    )

    assert "model_not_found" in source
    assert "suggest_models" in source
    # The validation loop must come before the writing loop.
    assert source.index("offered = await list_models(") < source.index(
        "setattr(row, f\"{tier}_{field}\", value.strip()"
    )


# ------------------------------------------- the settings endpoint, end to end --


pytestmark = pytest.mark.anyio

HEADERS = {"X-Workspace-Id": "biz_test"}


@pytest.fixture()
def clean_llm_cache():
    from app.core.llm.router import invalidate_workspace_llm_cache

    invalidate_workspace_llm_cache()
    yield
    invalidate_workspace_llm_cache()


async def test_saving_a_model_the_provider_does_not_have_is_refused(
    api_client: object, clean_llm_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``gemini-3-flash`` looked fine and killed every turn afterwards."""
    async def _fake_list(provider: str, api_key: str, *, base_url: str | None = None) -> list[str]:
        assert provider == "gemini"
        return AVAILABLE

    monkeypatch.setattr("app.core.llm.model_catalog.list_models", _fake_list)

    response = await api_client.patch(
        "/api/v1/settings/llm",
        headers=HEADERS,
        json={"strong": {"provider": "gemini", "model": "gemini-3-flash"}},
    )

    assert response.status_code == 422, response.text
    body = response.text
    assert "gemini-3-flash" in body
    assert "gemini-3.6-flash" in body  # the suggestion, not just the complaint


async def test_the_refused_model_is_not_written(
    api_client: object, clean_llm_cache: None, monkeypatch: pytest.MonkeyPatch, seeded_db: object
) -> None:
    from sqlalchemy import select

    from app.db.models import LlmProviderSettings

    async def _fake_list(provider: str, api_key: str, *, base_url: str | None = None) -> list[str]:
        return AVAILABLE

    monkeypatch.setattr("app.core.llm.model_catalog.list_models", _fake_list)

    rejected = await api_client.patch(
        "/api/v1/settings/llm",
        headers=HEADERS,
        json={"strong": {"provider": "gemini", "model": "gemini-3-flash"}},
    )
    assert rejected.status_code == 422, rejected.text

    row = await seeded_db.scalar(
        select(LlmProviderSettings).where(LlmProviderSettings.business_id == "biz_test")
    )
    assert row is None or getattr(row, "strong_model", None) != "gemini-3-flash"


async def test_a_model_the_provider_does_have_saves(
    api_client: object, clean_llm_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fake_list(provider: str, api_key: str, *, base_url: str | None = None) -> list[str]:
        return AVAILABLE

    monkeypatch.setattr("app.core.llm.model_catalog.list_models", _fake_list)

    response = await api_client.patch(
        "/api/v1/settings/llm",
        headers=HEADERS,
        json={"strong": {"provider": "gemini", "model": "gemini-3.6-flash"}},
    )

    assert response.status_code == 200, response.text
    assert response.json()["strong"]["model"] == "gemini-3.6-flash"


async def test_an_unverifiable_catalog_still_saves(
    api_client: object, clean_llm_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No key, no network or a custom gateway must never block a save."""
    async def _unreachable(provider: str, api_key: str, *, base_url: str | None = None) -> None:
        return None

    monkeypatch.setattr("app.core.llm.model_catalog.list_models", _unreachable)

    response = await api_client.patch(
        "/api/v1/settings/llm",
        headers=HEADERS,
        json={"strong": {"provider": "custom", "model": "my-gateway-model", "baseUrl": "https://g/v1"}},
    )

    assert response.status_code == 200, response.text


async def test_a_stored_key_that_cannot_be_decrypted_still_saves(
    api_client: object, clean_llm_cache: None, monkeypatch: pytest.MonkeyPatch, seeded_db: object
) -> None:
    """A key we cannot read back is 'cannot verify', and a save must survive it."""
    from app.api.settings import TokenCipher
    from app.config import get_settings
    from app.db.models import LlmProviderSettings

    # A *valid* stored key, so the rest of the endpoint still resolves; only the
    # verification step is then handed a cipher that refuses to decrypt.
    real = TokenCipher(get_settings().token_encryption_key)
    seeded_db.add(
        LlmProviderSettings(
            id="llm_broken_key", business_id="biz_test", strong_provider="gemini",
            strong_model="gemini-3.5-flash", strong_api_key_enc=real.encrypt("stored-key"),
        )
    )
    await seeded_db.commit()
    from app.core.llm.router import invalidate_workspace_llm_cache

    invalidate_workspace_llm_cache()

    class _Boom:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @staticmethod
        def encrypt(value: str) -> str:
            return f"enc:{value}"

        @staticmethod
        def decrypt(_value: str) -> str:
            raise RuntimeError("wrong encryption key")

    monkeypatch.setattr("app.api.settings.TokenCipher", _Boom)

    seen: dict[str, str] = {}

    async def _fake_list(provider: str, api_key: str, *, base_url: str | None = None) -> list[str]:
        seen["key"] = api_key
        return AVAILABLE

    monkeypatch.setattr("app.core.llm.model_catalog.list_models", _fake_list)

    response = await api_client.patch(
        "/api/v1/settings/llm",
        headers=HEADERS,
        json={"strong": {"model": "gemini-3.6-flash"}},
    )

    assert response.status_code == 200, response.text
    # The unreadable key was not swapped for a wrong one: verification was skipped
    # rather than performed with a credential we do not actually have.
    assert seen.get("key") == ""


async def test_the_catalog_is_asked_about_the_tier_being_saved(
    api_client: object, clean_llm_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fast tier has its own provider, and must be checked on its own terms."""
    calls: list[tuple[str, str]] = []

    async def _fake_list(provider: str, api_key: str, *, base_url: str | None = None) -> list[str]:
        calls.append((provider, api_key))
        return AVAILABLE

    monkeypatch.setattr("app.core.llm.model_catalog.list_models", _fake_list)

    response = await api_client.patch(
        "/api/v1/settings/llm",
        headers=HEADERS,
        json={"fast": {"provider": "groq", "model": "qwen/qwen3.8-27b", "apiKey": "gsk_x"}},
    )

    assert response.status_code == 422, response.text
    assert calls == [("groq", "gsk_x")]
