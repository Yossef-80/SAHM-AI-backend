"""OpenAI provider (also any OpenAI-compatible server via ``OPENAI_BASE_URL``).

Selected when ``AI_LLM_PROVIDER=openai``. Needs ``uv add langchain-openai``.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.ai.enums import StructuredOutputMethod
from app.ai.errors import ProviderNotConfiguredError, ProviderNotInstalledError
from app.ai.settings import AISettings

DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.FUNCTION_CALLING


def build_chat_model(
    settings: AISettings, *, model: str | None = None, temperature: float | None = None
) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ProviderNotInstalledError("openai", "langchain-openai") from exc

    if not settings.openai_api_key and not settings.openai_base_url:
        raise ProviderNotConfiguredError("AI_LLM_PROVIDER=openai but OPENAI_API_KEY is empty.")

    effective_model = (model or "").strip()
    if not effective_model or effective_model.lower() in {"openai", "default"}:
        effective_model = settings.openai_model

    return ChatOpenAI(
        model=effective_model,
        api_key=settings.openai_api_key or "not-needed",
        base_url=settings.openai_base_url,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )

