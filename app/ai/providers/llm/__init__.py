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

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.ai.enums import LLMProvider, StructuredOutputMethod
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
) -> LLMHandle:
    """Build the chat model selected by the enum (``AI_LLM_PROVIDER`` by default)."""
    cfg = settings or get_ai_settings()
    choice = provider or cfg.llm_provider
    module = load_provider_module(_PROVIDER_MODULES, choice, "llm")

    chat = module.build_chat_model(cfg, model=model or cfg.llm_model, temperature=temperature)
    method = cfg.structured_output_method or module.DEFAULT_STRUCTURED_METHOD
    model_name = str(getattr(chat, "model", None) or getattr(chat, "model_name", None) or choice)

    logger.info("llm_built", provider=choice.value, model=model_name, structured_method=method.value)
    return LLMHandle(chat=chat, provider=choice, model_name=model_name, structured_method=method)


__all__ = ["LLMHandle", "get_llm"]
