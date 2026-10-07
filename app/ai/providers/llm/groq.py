from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import SecretStr

from app.ai.enums import StructuredOutputMethod
from app.ai.settings import AISettings

DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.FUNCTION_CALLING

def build_chat_model(settings: AISettings, *, model: str | None, temperature: float | None) -> BaseChatModel:
    api_key = getattr(settings, "groq_api_key", None)
    
    kwargs = {}
    if api_key:
        kwargs["api_key"] = SecretStr(api_key)
        
    return ChatGroq(
        model=model or settings.groq_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        **kwargs
    )
