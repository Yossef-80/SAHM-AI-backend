"""LLM providers: pick a chat model by enum, import its module lazily.

    from app.ai.providers.llm import get_llm
    llm = get_llm()                                   # uses AI_LLM_PROVIDER from .env
    llm = get_llm(provider=LLMProvider.OPENAI)        # explicit override (e.g. per agent)

    llm.chat                    -> a LangChain BaseChatModel (bind_tools, invoke, stream...)
    llm.structured(MySchema)    -> Runnable that returns a validated MySchema instance

Every module in this folder exposes the same 2-item contract:

    DEFAULT_STRUCTURED_METHOD: StructuredOutputMethod
    def build_chat_model(settings, *, model: str | None, temperature: float | None) -> BaseChatModel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.ai.enums import LLMProvider, StructuredOutputMethod
from app.ai.errors import ProviderNotConfiguredError
from app.ai.loggers.providers.llm import logger
from app.ai.providers._loader import load_provider_module
from app.ai.settings import AISettings, get_ai_settings

#: enum -> module. Add a provider = add one line here + one file in this folder.
_PROVIDER_MODULES: dict[LLMProvider, str] = {
    LLMProvider.OLLAMA: "app.ai.providers.llm.ollama",
    LLMProvider.OPENAI: "app.ai.providers.llm.openai",
    LLMProvider.ANTHROPIC: "app.ai.providers.llm.anthropic",
    LLMProvider.GEMINI: "app.ai.providers.llm.gemini",
    LLMProvider.GROQ: "app.ai.providers.llm.groq",
    LLMProvider.OPENROUTER: "app.ai.providers.llm.openrouter",
    LLMProvider.MOCK: "app.ai.providers.llm.mock",
}


@dataclass(frozen=True)
class LLMHandle:
    """A chat model plus the knowledge of how to get structured output from it."""

    chat: BaseChatModel
    provider: LLMProvider
    model_name: str
    structured_method: StructuredOutputMethod

    def structured(self, schema: type[BaseModel]) -> Runnable:
        """Runnable that returns a validated ``schema`` instance."""
        return self.chat.with_structured_output(schema, method=self.structured_method.value)


def get_llm(
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    temperature: float | None = None,
    settings: AISettings | None = None,
    base_url: str | None = None,
) -> LLMHandle:
    """Build the chat model selected by the enum (``AI_LLM_PROVIDER`` by default)."""
    cfg = settings or get_ai_settings()
    choice = provider or cfg.llm_provider
    if base_url is not None and choice != LLMProvider.OLLAMA:
        raise ProviderNotConfiguredError("Endpoint overrides are supported only for the Ollama provider.")
    module = load_provider_module(_PROVIDER_MODULES, choice, "llm")

    build_kwargs: dict[str, Any] = {"model": model or cfg.llm_model, "temperature": temperature}
    if base_url is not None:
        build_kwargs["base_url"] = base_url

    chat = module.build_chat_model(cfg, **build_kwargs)
    method = cfg.structured_output_method or module.DEFAULT_STRUCTURED_METHOD
    model_name = str(getattr(chat, "model", None) or getattr(chat, "model_name", None) or choice)

    logger.info("llm_built", provider=choice.value, model=model_name, structured_method=method.value)
    return LLMHandle(chat=chat, provider=choice, model_name=model_name, structured_method=method)


def get_vlm_llm(*, settings: AISettings | None = None) -> LLMHandle:
    """Return Layla's vision-capable handle without changing global LLM selection.

    ``AI_ADS_VLM_PROVIDER`` and ``AI_ADS_VLM_MODEL`` can select an independent
    VLM (for example Gemini) for reference-image analysis. If they are unset,
    the legacy dedicated Ollama tunnel is used when configured; otherwise the
    existing general app.ai provider/model selection is reused unchanged.
    """
    cfg = settings or get_ai_settings()

    if cfg.ads_vlm_provider is not None:
        vlm_model = (cfg.ads_vlm_model or "").strip()
        if cfg.ads_vlm_provider == LLMProvider.OLLAMA:
            vlm_model = vlm_model or (cfg.vlm_ollama_model or "").strip()
        if not vlm_model:
            raise ProviderNotConfiguredError(
                f"AI_ADS_VLM_PROVIDER={cfg.ads_vlm_provider.value} is configured, "
                "but AI_ADS_VLM_MODEL is missing. Set it to the exact vision-capable "
                "model identifier for the selected provider."
            )

        # Use the selected VLM provider's structured-output default rather than
        # inheriting a global method configured for the system LLM provider.
        vlm_settings = cfg.model_copy(update={"structured_output_method": None})
        if cfg.ads_vlm_provider == LLMProvider.OLLAMA and cfg.ads_vlm_base_url:
            vlm_settings = vlm_settings.model_copy(
                update={"ollama_base_url": cfg.ads_vlm_base_url}
            )
        return get_llm(
            provider=cfg.ads_vlm_provider,
            model=vlm_model,
            settings=vlm_settings,
        )

    if not cfg.ads_vlm_base_url:
        return get_llm(settings=cfg)

    vlm_model = (cfg.vlm_ollama_model or "").strip()
    if not vlm_model:
        raise ProviderNotConfiguredError(
            "AI_ADS_VLM_BASE_URL is configured, but VLM_OLLAMA_MODEL is missing. "
            "Set it to the exact vision-capable model tag installed on the dedicated VLM server."
        )

    vlm_settings = cfg.model_copy(update={"ollama_base_url": cfg.ads_vlm_base_url})
    return get_llm(
        provider=LLMProvider.OLLAMA,
        model=vlm_model,
        settings=vlm_settings,
    )


__all__ = ["LLMHandle", "get_llm", "get_vlm_llm"]
