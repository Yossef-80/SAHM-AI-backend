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
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    settings: AISettings | None = None,
    base_url: str | None = None,
) -> LLMHandle:
    """Build the chat model selected by the enum (``AI_LLM_PROVIDER`` by default).

    ``temperature``, ``max_tokens`` and ``reasoning_effort`` override the
    process configuration for this handle. The parameters are filtered per model
    family by the provider module -- sending ``temperature`` to a reasoning
    model, or ``max_tokens`` instead of ``max_completion_tokens``, is a 400.
    """
    cfg = settings or get_ai_settings()
    choice = provider or cfg.llm_provider
    if base_url is not None and choice != LLMProvider.OLLAMA:
        raise ProviderNotConfiguredError("Endpoint overrides are supported only for the Ollama provider.")
    module = load_provider_module(_PROVIDER_MODULES, choice, "llm")

    build_kwargs: dict[str, Any] = {
        "model": model or cfg.llm_model,
        "temperature": temperature,
    }
    # Omitted rather than passed as None: a provider adapter must be able to tell
    # "the caller set nothing, use the process default" from "the caller set
    # nothing", and forwarding None used to overwrite ``settings.llm_max_tokens``.
    if max_tokens is not None:
        build_kwargs["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        build_kwargs["reasoning_effort"] = reasoning_effort
    if base_url is not None:
        build_kwargs["base_url"] = base_url

    chat = module.build_chat_model(cfg, **build_kwargs)
    method = cfg.structured_output_method or module.DEFAULT_STRUCTURED_METHOD
    model_name = str(getattr(chat, "model", None) or getattr(chat, "model_name", None) or choice)

    # Metering is attached here, and only here, so *every* consumer -- the
    # assistant turn, every agent graph, the research tab, memory distillation --
    # is accounted for without any call site opting in. Without this the usage
    # screen was reporting an empty table while the product spent real tokens.
    from app.core.ai_usage import attach_usage_metering

    chat = attach_usage_metering(chat, provider=choice.value, model_name=model_name)

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


async def get_workspace_chat_llm(
    *,
    db: Any,
    business_id: str | None,
    tier: str = "strong",
    settings: AISettings | None = None,
) -> LLMHandle:
    """The chat model this *workspace* selected, falling back to global config.

    ``/settings/llm`` has always let a workspace pick its provider, model,
    base URL and key -- and until now only the task/agent runtime read it, while
    the assistant kept using the process-wide ``AI_LLM_PROVIDER``. A workspace
    that switched to OpenAI (or to a newer model) saw no change in chat, which is
    the wiring gap this closes.

    Everything here is best-effort: any failure, or a workspace with no override
    at all, falls back to the global settings rather than breaking the turn.
    """
    cfg = settings or get_ai_settings()
    if db is None or not business_id:
        return get_llm(settings=cfg)

    try:
        from app.core.llm.router import resolve_workspace_tier

        tier_state = await resolve_workspace_tier(db, business_id, tier)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - never let routing break chat
        return get_llm(settings=cfg)

    provider_value = str(
        tier_state.get("provider") or getattr(cfg.llm_provider, "value", cfg.llm_provider)
    ).lower()
    # The settings API speaks "claude"; this layer calls the provider "anthropic".
    aliases = {"claude": "anthropic", "azure": "openai", "custom": "openai"}
    provider_value = aliases.get(provider_value, provider_value)
    try:
        choice = LLMProvider(provider_value)
    except ValueError:
        choice = cfg.llm_provider

    workspace_model = str(tier_state.get("model") or "") or None
    if choice is LLMProvider.MOCK:
        # A workspace on `mock` means "no provider configured", not "serve
        # placeholders in production": keep the process provider so a test
        # workspace cannot silently downgrade a real one. Its model, key and
        # base URL are placeholders from the same row, so they are dropped with
        # it -- keeping `mock-strong-1` and sending it to a real provider is the
        # half-applied override this rule exists to prevent.
        choice = cfg.llm_provider
        workspace_model = None
        api_key = ""
        base_url = None
    else:
        api_key = str(tier_state.get("api_key") or "")
        base_url = tier_state.get("base_url")

    overrides: dict[str, Any] = {
        "openai_api_key": api_key or cfg.openai_api_key,
        "openai_model": workspace_model or cfg.openai_model,
        "anthropic_api_key": api_key or cfg.anthropic_api_key,
        "anthropic_model": workspace_model or cfg.anthropic_model,
        "gemini_api_key": api_key or cfg.gemini_api_key,
        "gemini_model": workspace_model or cfg.gemini_model,
        "groq_api_key": api_key or cfg.groq_api_key,
        "groq_model": workspace_model or cfg.groq_model,
        "openrouter_api_key": api_key or cfg.openrouter_api_key,
        "openrouter_model": workspace_model or cfg.openrouter_model,
    }
    if base_url:
        # `openai_base_url` is the pass-through an OpenAI-compatible gateway
        # needs; other providers keep their own env-configured endpoint.
        overrides["openai_base_url"] = str(base_url)
    cfg = cfg.model_copy(update=overrides)

    return get_llm(
        provider=choice,
        model=workspace_model,
        temperature=_optional_float(tier_state.get("temperature")),
        max_tokens=_optional_int(tier_state.get("max_tokens")),
        reasoning_effort=_optional_str(tier_state.get("reasoning_effort")),
        settings=cfg,
    )


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


__all__ = ["LLMHandle", "get_llm", "get_vlm_llm", "get_workspace_chat_llm"]
