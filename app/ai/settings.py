"""AI-layer settings: provider selection through enums, values from ``.env``.

Selecting a provider is a one-line ``.env`` change, validated against the enums
in ``app.ai.enums``. A typo (e.g. ``AI_LLM_PROVIDER=olama``) fails loudly at
startup instead of silently falling back.

    AI_LLM_PROVIDER=ollama        # ollama | openai | anthropic | mock
    AI_SEARCH_PROVIDER=tavily     # tavily | mock | none
    AI_DB_PROVIDER=sqlite         # memory | sqlite | postgres

Per-agent overrides live in each agent's ``config.py`` (e.g. ``NOUR_LLM_PROVIDER``).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.ai.enums import (
    DatabaseProvider,
    LLMProvider,
    LogFormat,
    LogLevel,
    SearchDepth,
    SearchProvider,
    StructuredOutputMethod,
)


def _env(*names: str) -> AliasChoices:
    """Accept several env-var names for one field (new name first, legacy after)."""
    return AliasChoices(*names)


class AISettings(BaseSettings):
    """Everything the providers, tools, loggers and agents are allowed to read."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # ========================================================= SELECTIONS ====
    llm_provider: LLMProvider = Field(LLMProvider.OLLAMA, validation_alias=_env("AI_LLM_PROVIDER"))
    search_provider: SearchProvider = Field(
        SearchProvider.TAVILY, validation_alias=_env("AI_SEARCH_PROVIDER")
    )
    db_provider: DatabaseProvider = Field(
        DatabaseProvider.SQLITE, validation_alias=_env("AI_DB_PROVIDER")
    )
    #: ``None`` = use the provider's own best default (see providers/llm/*.py).
    structured_output_method: StructuredOutputMethod | None = Field(
        None, validation_alias=_env("AI_STRUCTURED_OUTPUT_METHOD")
    )

    # ===================================================== LLM (generic) ====
    #: Overrides the provider-specific model below when set.
    llm_model: str | None = Field(None, validation_alias=_env("AI_LLM_MODEL"))
    llm_temperature: float = Field(0.2, validation_alias=_env("AI_LLM_TEMPERATURE"))
    llm_max_tokens: int = Field(4096, validation_alias=_env("AI_LLM_MAX_TOKENS"))
    llm_timeout_seconds: float = Field(180.0, validation_alias=_env("AI_LLM_TIMEOUT_SECONDS"))
    llm_max_retries: int = Field(2, validation_alias=_env("AI_LLM_MAX_RETRIES"))

    # ============================================================ OLLAMA ====
    ollama_base_url: str | None = Field(
        None, validation_alias=_env("OLLAMA_BASE_URL", "NGROK_URL", "OLLAMA_URL")
    )
    ollama_model: str = Field(
        "qwen3:8b", validation_alias=_env("OLLAMA_MODEL", "NGROK_MODEL_ID", "OLLAMA_MODEL_ID")
    )
    #: Context window. Big values (e.g. 256k defaults on some GGUFs) make Colab
    #: GPUs time out -- keep this as small as your prompts allow.
    ollama_num_ctx: int = Field(8192, validation_alias=_env("OLLAMA_NUM_CTX"))
    #: How long Ollama keeps the model loaded between calls (avoids reload cost).
    ollama_keep_alive: str = Field("30m", validation_alias=_env("OLLAMA_KEEP_ALIVE"))
    #: ``None`` = don't send; ``False`` disables "thinking" on Qwen3-style models (faster).
    ollama_reasoning: bool | None = Field(None, validation_alias=_env("OLLAMA_REASONING"))
    ollama_timeout_seconds: float = Field(
        300.0, validation_alias=_env("OLLAMA_TIMEOUT_SECONDS", "NGROK_TIMEOUT_SECONDS")
    )

    # ============================================================ OPENAI ====
    openai_api_key: str | None = Field(None, validation_alias=_env("OPENAI_API_KEY"))
    openai_model: str = Field("gpt-4o-mini", validation_alias=_env("OPENAI_MODEL"))
    #: Set to use any OpenAI-compatible server (vLLM, LM Studio, Groq...).
    openai_base_url: str | None = Field(None, validation_alias=_env("OPENAI_BASE_URL"))

    # ========================================================= ANTHROPIC ====
    anthropic_api_key: str | None = Field(None, validation_alias=_env("ANTHROPIC_API_KEY"))
    anthropic_model: str = Field("claude-haiku-4-5", validation_alias=_env("ANTHROPIC_MODEL"))

    # ============================================================ GEMINI ====
    gemini_api_key: str | None = Field(None, validation_alias=_env("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    gemini_model: str = Field("gemini-1.5-flash", validation_alias=_env("GEMINI_MODEL"))
    
    # ============================================================== GROQ ====
    groq_api_key: str | None = Field(None, validation_alias=_env("GROQ_API_KEY"))
    groq_model: str = Field("llama3-8b-8192", validation_alias=_env("GROQ_MODEL"))

    # ======================================================== OPENROUTER ====
    openrouter_api_key: str | None = Field(None, validation_alias=_env("OPENROUTER_API_KEY"))
    openrouter_model: str = Field("google/gemini-flash-1.5-exp", validation_alias=_env("OPENROUTER_MODEL"))

    # ============================================================ SEARCH ====
    tavily_api_key: str | None = Field(None, validation_alias=_env("TAVILY_API_KEY"))
    search_depth: SearchDepth = Field(SearchDepth.BASIC, validation_alias=_env("AI_SEARCH_DEPTH"))
    search_timeout_seconds: float = Field(20.0, validation_alias=_env("AI_SEARCH_TIMEOUT_SECONDS"))
    #: Snippets are truncated to this many characters to protect the context window.
    search_snippet_chars: int = Field(600, validation_alias=_env("AI_SEARCH_SNIPPET_CHARS"))

    # ========================================================== WEBSITES ====
    website_timeout_seconds: float = Field(15.0, validation_alias=_env("WEBSITE_FETCH_TIMEOUT_SECONDS"))
    website_max_bytes: int = Field(2_000_000, validation_alias=_env("WEBSITE_MAX_BYTES"))
    website_max_redirects: int = Field(3, validation_alias=_env("WEBSITE_MAX_REDIRECTS"))
    #: Page text handed back to the LLM is capped to this many characters.
    website_text_chars: int = Field(4000, validation_alias=_env("AI_WEBSITE_TEXT_CHARS"))

    # ================================================================ DB ====
    sqlite_path: str = Field("./var/ai_memory.sqlite", validation_alias=_env("AI_SQLITE_PATH"))
    postgres_url: str | None = Field(None, validation_alias=_env("AI_POSTGRES_URL"))

    # =========================================================== LOGGING ====
    log_dir: str = Field("./logs", validation_alias=_env("AI_LOG_DIR"))
    log_level: LogLevel = Field(LogLevel.INFO, validation_alias=_env("AI_LOG_LEVEL"))
    log_format: LogFormat = Field(LogFormat.TEXT, validation_alias=_env("AI_LOG_FORMAT"))
    log_to_console: bool = Field(False, validation_alias=_env("AI_LOG_TO_CONSOLE"))
    log_max_bytes: int = Field(5_000_000, validation_alias=_env("AI_LOG_MAX_BYTES"))
    log_backup_count: int = Field(3, validation_alias=_env("AI_LOG_BACKUP_COUNT"))

    @field_validator("ollama_base_url", mode="after")
    @classmethod
    def _strip_url(cls, v: str | None) -> str | None:
        """Ngrok URLs are often pasted with a trailing slash or spaces."""
        return v.strip().rstrip("/") if v else v


@lru_cache
def get_ai_settings() -> AISettings:
    """Cached accessor. Call ``get_ai_settings.cache_clear()`` after editing env in tests."""
    return AISettings()


__all__ = ["AISettings", "get_ai_settings"]
