"""Groq chat-model provider."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from app.ai.enums import StructuredOutputMethod
from app.ai.errors import ProviderNotConfiguredError, ProviderNotInstalledError
from app.ai.settings import AISettings

DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.FUNCTION_CALLING


def build_chat_model(
    settings: AISettings,
    *,
    model: str | None = None,
    temperature: float | None = None,
    #: Accepted for a uniform `get_llm` signature. None of the non-OpenAI
    #: providers has a separate "reasoning effort" knob, and their own token
    #: limit already comes from settings, so both are intentionally unused.
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ProviderNotInstalledError("groq", "langchain-groq") from exc

    if not settings.groq_api_key:
        raise ProviderNotConfiguredError("AI_LLM_PROVIDER=groq but GROQ_API_KEY is empty.")

    return ChatGroq(
        model=model or settings.groq_model,
        api_key=SecretStr(settings.groq_api_key),
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
