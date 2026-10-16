"""AI-layer settings: provider selection through enums, values from ``.env``.

Selecting a provider is a one-line ``.env`` change, validated against the enums
in ``app.ai.enums``. A typo (e.g. ``AI_LLM_PROVIDER=olama``) fails loudly at
startup instead of silently falling back.

    AI_LLM_PROVIDER=ollama        # ollama | openai | anthropic | gemini | groq | openrouter | mock
    AI_SEARCH_PROVIDER=tavily     # tavily | mock | none
    AI_DB_PROVIDER=sqlite         # memory | sqlite | postgres

Live search/review providers are metered and require their explicit ``AI_ALLOW_PAID_*`` opt-in.
Provider-specific model defaults live in ``app.ai.config``; environment variables override them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.ai.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
)
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

    # ============================================================ TIERS ====
    # Two tiers existed in ``app/config.py`` (and in the workspace settings API
    # and table) since the beginning, but every agent graph read
    # ``AI_LLM_PROVIDER`` directly -- so a workspace could pick a strong model in
    # Settings -> LLM and Layla would keep writing on the process default. These
    # two pairs are the missing half of that wiring.
    #
    # ``None`` means "same as ``AI_LLM_PROVIDER`` / ``AI_LLM_MODEL``", so an
    # unchanged .env behaves exactly as before.
    llm_provider_strong: LLMProvider | None = Field(
        None, validation_alias=_env("LLM_PROVIDER_STRONG")
    )
    llm_model_strong: str | None = Field(None, validation_alias=_env("LLM_MODEL_STRONG"))
    llm_provider_fast: LLMProvider | None = Field(
        None, validation_alias=_env("LLM_PROVIDER_FAST")
    )
    llm_model_fast: str | None = Field(None, validation_alias=_env("LLM_MODEL_FAST"))

    # ============================================== ROLE SAMPLING ==========
    #: Per-role temperature overrides, e.g. ``LLM_TEMPERATURE_CREATIVE=0.9``.
    #: Reading the process default (0.2) for ad copy is why every "variation"
    #: came back as the same sentence with the words shuffled.
    # ``NoDecode`` keeps the raw env string intact: pydantic-settings would
    # otherwise JSON-decode any dict-typed field before the validator runs, so
    # the friendly ``creative_copy=0.9,critique=0.1`` form (and an empty value)
    # would blow up at import time instead of being parsed by our validator.
    llm_temperature_overrides: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=dict,
        validation_alias=_env("LLM_TEMPERATURE_OVERRIDES"),
    )

    #: Layla reviews a creative deliverable once before submitting it (see
    #: ``app.ai.agents.layla.critique``). One extra strong-tier call per
    #: deliverable; set ``AI_LAYLA_SELF_CRITIQUE=0`` to spend it elsewhere.
    layla_self_critique: bool = Field(
        True, validation_alias=_env("AI_LAYLA_SELF_CRITIQUE")
    )

    # ===================================================== LLM (generic) ====
    #: Overrides the provider-specific model below when set.
    llm_model: str | None = Field(None, validation_alias=_env("AI_LLM_MODEL"))
    llm_temperature: float = Field(0.2, ge=0, le=2, validation_alias=_env("AI_LLM_TEMPERATURE"))
    llm_max_tokens: int = Field(4096, gt=0, validation_alias=_env("AI_LLM_MAX_TOKENS"))
    #: Reasoning models (o-series, GPT-5, GPT-6) take ``reasoning_effort`` instead
    #: of an eager answer. Left unset, the provider layer sends ``none`` for those
    #: models -- required for tool calling on Chat Completions from GPT-5.4 on.
    #: Set it to ``low``/``medium``/``high`` for non-tool, quality-first turns.
    llm_reasoning_effort: str | None = Field(
        None, validation_alias=_env("AI_LLM_REASONING_EFFORT")
    )
    llm_timeout_seconds: float = Field(180.0, gt=0, validation_alias=_env("AI_LLM_TIMEOUT_SECONDS"))
    llm_max_retries: int = Field(2, ge=0, validation_alias=_env("AI_LLM_MAX_RETRIES"))

    # ======================================================= ADS VLM (optional) ====
    #: Optional dedicated provider/model for reference-image analysis only.
    #: When unset, get_vlm_llm preserves the existing dedicated-Ollama/general-LLM fallback.
    ads_vlm_provider: LLMProvider | None = Field(
        None, validation_alias=_env("AI_ADS_VLM_PROVIDER")
    )
    ads_vlm_model: str | None = Field(None, validation_alias=_env("AI_ADS_VLM_MODEL"))

    # ============================================================ OLLAMA ====
    ollama_base_url: str | None = Field(
        None, validation_alias=_env("OLLAMA_BASE_URL", "NGROK_URL", "OLLAMA_URL")
    )
    #: Optional dedicated Ollama-compatible endpoint for Layla reference-image
    #: analysis. It does not alter the general LLM provider or model selection.
    ads_vlm_base_url: str | None = Field(None, validation_alias=_env("AI_ADS_VLM_BASE_URL"))
    #: Model tag installed on the dedicated VLM endpoint. Keep independent from
    #: OLLAMA_MODEL, which belongs to the general Ollama/LLM server.
    vlm_ollama_model: str | None = Field(None, validation_alias=_env("VLM_OLLAMA_MODEL"))
    #: Pixel budget for the temporary image copy sent to the VLM. The original
    #: upload remains untouched; Qwen3-VL uses fewer visual patches at lower sizes.
    ads_vlm_max_pixels: int = Field(
        1_000_000,
        ge=262_144,
        le=4_000_000,
        validation_alias=_env("AI_ADS_VLM_MAX_PIXELS"),
    )
    ollama_model: str = Field(
        DEFAULT_OLLAMA_MODEL, validation_alias=_env("OLLAMA_MODEL", "NGROK_MODEL_ID", "OLLAMA_MODEL_ID")
    )
    #: Context window. Big values (e.g. 256k defaults on some GGUFs) make Colab
    #: GPUs time out -- keep this as small as your prompts allow.
    ollama_num_ctx: int = Field(8192, gt=0, validation_alias=_env("OLLAMA_NUM_CTX"))
    #: How long Ollama keeps the model loaded between calls (avoids reload cost).
    ollama_keep_alive: str = Field("30m", validation_alias=_env("OLLAMA_KEEP_ALIVE"))
    #: ``None`` = don't send; ``False`` disables "thinking" on Qwen3-style models (faster).
    ollama_reasoning: bool | None = Field(None, validation_alias=_env("OLLAMA_REASONING"))
    ollama_timeout_seconds: float = Field(
        1000.0, gt=0, validation_alias=_env("OLLAMA_TIMEOUT_SECONDS", "NGROK_TIMEOUT_SECONDS")
    )

    # ============================================================ OPENAI ====
    openai_api_key: str | None = Field(None, validation_alias=_env("OPENAI_API_KEY"))
    openai_model: str = Field(DEFAULT_OPENAI_MODEL, validation_alias=_env("OPENAI_MODEL"))
    #: Set to use any OpenAI-compatible server (vLLM, LM Studio, Groq...).
    openai_base_url: str | None = Field(None, validation_alias=_env("OPENAI_BASE_URL"))

    # ========================================================= ANTHROPIC ====
    anthropic_api_key: str | None = Field(None, validation_alias=_env("ANTHROPIC_API_KEY"))
    anthropic_model: str = Field(DEFAULT_ANTHROPIC_MODEL, validation_alias=_env("ANTHROPIC_MODEL"))

    # ============================================================ GEMINI ====
    gemini_api_key: str | None = Field(None, validation_alias=_env("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    gemini_model: str = Field(DEFAULT_GEMINI_MODEL, validation_alias=_env("GEMINI_MODEL"))
    
    # ============================================================== GROQ ====
    groq_api_key: str | None = Field(None, validation_alias=_env("GROQ_API_KEY"))
    groq_model: str = Field(DEFAULT_GROQ_MODEL, validation_alias=_env("GROQ_MODEL"))

    # ======================================================== OPENROUTER ====
    openrouter_api_key: str | None = Field(None, validation_alias=_env("OPENROUTER_API_KEY"))
    openrouter_model: str = Field(DEFAULT_OPENROUTER_MODEL, validation_alias=_env("OPENROUTER_MODEL"))

    # ============================================================ SEARCH ====
    tavily_api_key: str | None = Field(None, validation_alias=_env("TAVILY_API_KEY"))
    search_depth: SearchDepth = Field(SearchDepth.BASIC, validation_alias=_env("AI_SEARCH_DEPTH"))
    allow_paid_search_calls: bool = Field(False, validation_alias=_env("AI_ALLOW_PAID_SEARCH_CALLS"))
    search_timeout_seconds: float = Field(20.0, gt=0, validation_alias=_env("AI_SEARCH_TIMEOUT_SECONDS"))
    #: Snippets are truncated to this many characters to protect the context window.
    search_snippet_chars: int = Field(600, gt=0, validation_alias=_env("AI_SEARCH_SNIPPET_CHARS"))

    # ========================================================== WEBSITES ====
    website_timeout_seconds: float = Field(15.0, gt=0, validation_alias=_env("WEBSITE_FETCH_TIMEOUT_SECONDS"))
    website_max_bytes: int = Field(2_000_000, gt=0, validation_alias=_env("WEBSITE_MAX_BYTES"))
    website_max_redirects: int = Field(3, ge=0, validation_alias=_env("WEBSITE_MAX_REDIRECTS"))
    #: Page text handed back to the LLM is capped to this many characters.
    website_text_chars: int = Field(4000, validation_alias=_env("AI_WEBSITE_TEXT_CHARS"))

    # ================================================================ DB ====
    sqlite_path: str = Field("./var/ai_memory.sqlite", validation_alias=_env("AI_SQLITE_PATH"))
    postgres_url: str | None = Field(None, validation_alias=_env("AI_POSTGRES_URL"))

    # ============================================================ MEMORY ====
    memory_session_max_tokens: int = Field(10000, gt=0, validation_alias=_env("SESSION_MEMORY_MAX_TOKENS", "MEMORY_SESSION_MAX_TOKENS"))
    brand_memory_token_limit: int = Field(10000, gt=0, validation_alias=_env("BRAND_MEMORY_TOKEN_LIMIT", "BRAND_IDENTITY_TOKEN_LIMIT", "MEMORY_BRAND_MAX_TOKENS"))
    memory_ltm_debounce_seconds: int = Field(180, ge=0, validation_alias=_env("MEMORY_LTM_DEBOUNCE_SECONDS"))
    memory_item_max_tokens: int = Field(120, gt=0, validation_alias=_env("MEMORY_ITEM_MAX_TOKENS"))
    memory_global_profile_max_tokens: int = Field(300, gt=0, validation_alias=_env("MEMORY_GLOBAL_PROFILE_MAX_TOKENS"))
    memory_retain_recent_turns: int = Field(2, ge=1, validation_alias=_env("MEMORY_RETAIN_RECENT_TURNS"))

    # =========================================================== REVIEWS ====
    google_places_api_key: str | None = Field(None, validation_alias=_env("GOOGLE_PLACES_API_KEY"))
    apify_api_token: str | None = Field(None, validation_alias=_env("APIFY_API_TOKEN", "APIFY_TOKEN"))
    google_maps_reviews_actor_id: str = Field("compass/crawler-google-places", validation_alias=_env("APIFY_MAPS_ACTOR_ID"))
    reviews_timeout_seconds: float = Field(120.0, gt=0, validation_alias=_env("REVIEWS_TIMEOUT_SECONDS"))
    allow_paid_review_calls: bool = Field(False, validation_alias=_env("AI_ALLOW_PAID_REVIEW_CALLS"))

    # =============================================================== GA4 ====
    ga4_client_email: str | None = Field(None, validation_alias=_env("GA4_CLIENT_EMAIL"))
    ga4_private_key: str | None = Field(None, validation_alias=_env("GA4_PRIVATE_KEY"))

    @property
    def GA4_CLIENT_EMAIL(self) -> str | None:
        return self.ga4_client_email

    @property
    def GA4_PRIVATE_KEY(self) -> str | None:
        return self.ga4_private_key

    # =========================================================== LOGGING ====
    log_dir: str = Field("./logs", validation_alias=_env("AI_LOG_DIR"))
    log_level: LogLevel = Field(LogLevel.INFO, validation_alias=_env("AI_LOG_LEVEL"))
    log_format: LogFormat = Field(LogFormat.TEXT, validation_alias=_env("AI_LOG_FORMAT"))
    log_to_console: bool = Field(False, validation_alias=_env("AI_LOG_TO_CONSOLE"))
    log_max_bytes: int = Field(5_000_000, gt=0, validation_alias=_env("AI_LOG_MAX_BYTES"))
    log_backup_count: int = Field(3, ge=0, validation_alias=_env("AI_LOG_BACKUP_COUNT"))

    @field_validator("llm_temperature_overrides", mode="before")
    @classmethod
    def _parse_temperature_overrides(cls, v: object) -> object:
        """Accept ``LLM_TEMPERATURE_OVERRIDES=creative_copy=0.9,routing=0.1``."""
        from app.ai.roles import parse_temperature_overrides

        if v is None:
            return {}
        if isinstance(v, str):
            return parse_temperature_overrides(v)
        return v

    @field_validator("ollama_base_url", "ads_vlm_base_url", mode="after")
    @classmethod
    def _strip_url(cls, v: str | None) -> str | None:
        """Ngrok URLs are often pasted with a trailing slash or spaces."""
        return v.strip().rstrip("/") if v else v

    # --------------------------------------------------------- tiers ------

    def provider_for_tier(self, tier: str) -> LLMProvider:
        """The provider for ``fast``/``strong``, falling back to the global one."""
        value = getattr(self, f"llm_provider_{tier}", None)
        return value or self.llm_provider

    def model_for_tier(self, tier: str) -> str | None:
        """The model for ``fast``/``strong``, or None to keep the provider default."""
        return getattr(self, f"llm_model_{tier}", None) or self.llm_model

    def role_spec(self, role: str):
        """The tier/temperature/max-tokens profile for one kind of work."""
        from app.ai.roles import resolve_role

        return resolve_role(role, self.llm_temperature_overrides)


@lru_cache
def get_ai_settings() -> AISettings:
    """Cached accessor. Call ``get_ai_settings.cache_clear()`` after editing env in tests."""
    return AISettings()


settings = get_ai_settings()

__all__ = ["AISettings", "get_ai_settings", "settings"]
