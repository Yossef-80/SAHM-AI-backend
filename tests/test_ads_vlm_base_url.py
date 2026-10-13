from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.ai.agents.nour import meta_ads
from app.ai.enums import LLMProvider, StructuredOutputMethod
from app.ai.errors import ProviderNotConfiguredError
from app.ai.providers import llm as llm_factory
from app.ai.providers.llm import ollama as ollama_provider
from app.ai.settings import AISettings


def test_ads_vlm_base_url_is_separate_from_general_ollama_url(monkeypatch):
    monkeypatch.setenv("AI_ADS_VLM_BASE_URL", "  https://vision.example.ngrok.app/  ")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://general.example.ngrok.app/")

    settings = AISettings(_env_file=None)

    assert settings.ads_vlm_base_url == "https://vision.example.ngrok.app"
    assert settings.ollama_base_url == "https://general.example.ngrok.app"


def test_selected_ad_vlm_uses_dedicated_url_and_model_without_changing_global(monkeypatch):
    settings = SimpleNamespace(
        ads_vlm_provider="ollama",
        ads_vlm_model="qwen3-vl:8b",
        ads_vlm_base_url="https://vision.example.ngrok.app",
        ollama_base_url="https://general.example.ngrok.app",
        ollama_model="qwen3:8b",
        gemini_api_key=None,
    )
    captured = {}
    monkeypatch.setattr(meta_ads, "get_ai_settings", lambda: settings)
    monkeypatch.setattr(
        meta_ads,
        "get_llm",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    meta_ads._ads_vlm_model()

    assert captured["provider"] == LLMProvider.OLLAMA
    assert captured["model"] == "qwen3-vl:8b"
    assert captured["base_url"] == "https://vision.example.ngrok.app"
    assert settings.ollama_base_url == "https://general.example.ngrok.app"
    assert settings.ollama_model == "qwen3:8b"


def test_selected_ad_vlm_can_use_dedicated_url_when_general_ollama_url_is_unset(monkeypatch):
    settings = SimpleNamespace(
        ads_vlm_provider="ollama",
        ads_vlm_model="qwen3-vl:8b",
        ads_vlm_base_url="https://vision.example.ngrok.app",
        ollama_base_url=None,
        ollama_model="qwen3:8b",
        gemini_api_key=None,
    )
    captured = {}
    monkeypatch.setattr(meta_ads, "get_ai_settings", lambda: settings)
    monkeypatch.setattr(
        meta_ads,
        "get_llm",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    meta_ads._ads_vlm_model()

    assert captured["base_url"] == "https://vision.example.ngrok.app"


def test_get_llm_forwards_endpoint_override_only_for_ollama(monkeypatch):
    captured = {}

    class FakeChat:
        model = "qwen3-vl:8b"
        model_name = None

    class FakeOllamaModule:
        DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.JSON_SCHEMA

        @staticmethod
        def build_chat_model(settings, **kwargs):
            captured["settings"] = settings
            captured["kwargs"] = kwargs
            return FakeChat()

    monkeypatch.setattr(
        llm_factory,
        "load_provider_module",
        lambda *_args: FakeOllamaModule,
    )
    settings = SimpleNamespace(
        llm_provider=LLMProvider.OLLAMA,
        llm_model="general-text-model",
        structured_output_method=None,
    )

    llm_factory.get_llm(
        provider=LLMProvider.OLLAMA,
        model="qwen3-vl:8b",
        temperature=0.1,
        settings=settings,
        base_url="https://vision.example.ngrok.app",
    )

    assert captured["kwargs"] == {
        "model": "qwen3-vl:8b",
        "temperature": 0.1,
        "base_url": "https://vision.example.ngrok.app",
    }


def test_get_llm_rejects_endpoint_override_for_non_ollama_provider(monkeypatch):
    class FakeModule:
        DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.JSON_SCHEMA

    monkeypatch.setattr(llm_factory, "load_provider_module", lambda *_args: FakeModule)
    settings = SimpleNamespace(
        llm_provider=LLMProvider.GEMINI,
        llm_model=None,
        structured_output_method=None,
    )

    with pytest.raises(ProviderNotConfiguredError, match="only for the Ollama provider"):
        llm_factory.get_llm(
            provider=LLMProvider.GEMINI,
            settings=settings,
            base_url="https://vision.example.ngrok.app",
        )


def test_ollama_builder_uses_override_for_vlm_and_general_url_as_fallback(monkeypatch):
    captured = []

    class FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured.append(kwargs)

    fake_module = ModuleType("langchain_ollama")
    fake_module.ChatOllama = FakeChatOllama
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake_module)

    settings = SimpleNamespace(
        ollama_base_url="https://general.example.ngrok.app",
        ollama_model="general-text-model",
        llm_temperature=0.2,
        ollama_num_ctx=8192,
        llm_max_tokens=4096,
        ollama_keep_alive="30m",
        ollama_timeout_seconds=30.0,
        ollama_reasoning=None,
    )

    vision_chat = ollama_provider.build_chat_model(
        settings,
        model="qwen3-vl:8b",
        temperature=0.1,
        base_url="https://vision.example.ngrok.app/",
    )
    general_chat = ollama_provider.build_chat_model(settings, model="general-text-model")

    assert vision_chat.kwargs["base_url"] == "https://vision.example.ngrok.app"
    assert vision_chat.kwargs["model"] == "qwen3-vl:8b"
    assert general_chat.kwargs["base_url"] == "https://general.example.ngrok.app"
    assert settings.ollama_base_url == "https://general.example.ngrok.app"
    assert len(captured) == 2


def test_ollama_builder_requires_some_base_url():
    settings = SimpleNamespace(
        ollama_base_url=None,
        ollama_model="general-text-model",
        llm_temperature=0.2,
        ollama_num_ctx=8192,
        llm_max_tokens=4096,
        ollama_keep_alive="30m",
        ollama_timeout_seconds=30.0,
        ollama_reasoning=None,
    )

    with pytest.raises(ProviderNotConfiguredError, match="Ollama needs OLLAMA_BASE_URL"):
        ollama_provider.build_chat_model(settings)
