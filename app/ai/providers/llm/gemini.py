from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from app.ai.enums import StructuredOutputMethod
from app.ai.settings import AISettings

DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.FUNCTION_CALLING

def build_chat_model(settings: AISettings, *, model: str | None, temperature: float | None) -> BaseChatModel:
    # We grab gemini_api_key from settings or rely on GOOGLE_API_KEY env var natively.
    # Note: ensure settings has gemini_api_key if you want to pass it explicitly.
    api_key = getattr(settings, "gemini_api_key", None)
    
    kwargs = {}
    if api_key:
        kwargs["api_key"] = SecretStr(api_key)
        
    return ChatGoogleGenerativeAI(
        model=model or settings.gemini_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_output_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        **kwargs
    )
