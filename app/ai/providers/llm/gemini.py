"""Gemini chat-model provider for the general AI layer.

The separate VLM integration remains isolated; this adapter only follows the
existing ``AI_LLM_PROVIDER=gemini`` selection for LangGraph text/structured calls.
"""

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
) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ProviderNotInstalledError("gemini", "langchain-google-genai") from exc

    if not settings.gemini_api_key:
        raise ProviderNotConfiguredError(
            "AI_LLM_PROVIDER=gemini but GEMINI_API_KEY/GOOGLE_API_KEY is empty."
        )

    return ChatGoogleGenerativeAI(
        model=model or settings.gemini_model,
        api_key=SecretStr(settings.gemini_api_key),
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        request_timeout=settings.llm_timeout_seconds,
        retries=settings.llm_max_retries,
    )
