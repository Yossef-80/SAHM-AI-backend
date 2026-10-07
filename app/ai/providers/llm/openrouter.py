from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.ai.enums import StructuredOutputMethod
from app.ai.settings import AISettings

DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.JSON_SCHEMA

def build_chat_model(settings: AISettings, *, model: str | None, temperature: float | None) -> BaseChatModel:
    api_key = getattr(settings, "openrouter_api_key", None)
    
    kwargs = {}
    if api_key:
        kwargs["api_key"] = SecretStr(api_key)
        
    return ChatOpenAI(
        model=model or settings.openrouter_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:3000", # Required by OpenRouter for free models
            "X-Title": "Sahm Marketing OS"
        },
        **kwargs
    )
