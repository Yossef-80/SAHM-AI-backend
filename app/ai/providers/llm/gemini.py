import warnings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from app.ai.enums import StructuredOutputMethod
from app.ai.settings import AISettings

# Suppress LangChain's fixed sampling defaults warning on flash-lite models
warnings.filterwarnings("ignore", message=r".*uses fixed sampling defaults.*")

DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.FUNCTION_CALLING

def build_chat_model(settings: AISettings, *, model: str | None, temperature: float | None) -> BaseChatModel:
    api_key = getattr(settings, "gemini_api_key", None)
    
    kwargs = {}
    if api_key:
        kwargs["api_key"] = SecretStr(api_key)

    chosen_model = model or settings.gemini_model
    temp = temperature if temperature is not None else settings.llm_temperature
    if "flash-lite" in chosen_model.lower():
        temp = None
        
    return ChatGoogleGenerativeAI(
        model=chosen_model,
        temperature=temp,
        max_output_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        **kwargs
    )
