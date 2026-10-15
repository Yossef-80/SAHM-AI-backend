"""Anthropic (Claude) provider.

Selected when ``AI_LLM_PROVIDER=anthropic``. Needs ``uv add langchain-anthropic``.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.ai.enums import StructuredOutputMethod
from app.ai.errors import ProviderNotConfiguredError, ProviderNotInstalledError
from app.ai.settings import AISettings

DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.FUNCTION_CALLING


def build_chat_model(
    settings: AISettings,
    *,
    model: str | None = None,
    temperature: float | None = None,
    #: Accepted for a uniform `get_llm` signature; Claude has no separate
    #: reasoning-effort knob and its output limit comes from settings.
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ProviderNotInstalledError("anthropic", "langchain-anthropic") from exc

    if not settings.anthropic_api_key:
        raise ProviderNotConfiguredError(
            "AI_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty."
        )

    return ChatAnthropic(
        model=model or settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
