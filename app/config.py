"""Application configuration.

Every runtime knob comes from the environment. Nothing in this file hardcodes a
model name or provider; those live in the provider->tier map below and are set
purely by env vars (see .env.example).

Rule enforced across the codebase: no module other than ``config.py`` may
contain a literal provider/model string.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Tier = Literal["fast", "strong"]

#: Providers that the LLM router knows how to construct.
LLMProviderName = Literal["mock", "claude", "openai"]

#: Image-generation providers behind the image interface.
ImageProviderName = Literal["mock", "openai", "stability"]

#: Web-search providers behind the research tools.
SearchProviderName = Literal["mock", "tavily", "serper", "brave"]

#: Keyword-data providers.
KeywordProviderName = Literal["mock", "dataforseo"]


class Settings(BaseSettings):
    """All environment-driven configuration for the Sahm backend."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app --
    app_name: str = "Sahm"
    environment: Literal["dev", "test", "staging", "prod"] = "dev"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # ------------------------------------------------------------------ db ----
    # Postgres+pgvector is the production target. SQLite is supported so the
    # app boots and the whole flow runs with zero external services.
    database_url: str = "sqlite+aiosqlite:///./sahm.db"
    db_echo: bool = False
    # Embedding dimension must match the configured embedding provider output.
    embedding_dim: int = 64
    # Vector backend: "pgvector" on Postgres, "python" (cosine in-process) on
    # SQLite or when pgvector is unavailable.
    vector_backend: Literal["auto", "pgvector", "python"] = "auto"

    # ---------------------------------------------------------------- llm ----
    # Which provider serves which tier. Switching provider or model is an env
    # change only -- no code edits anywhere else in the repo.
    llm_provider_fast: LLMProviderName = "mock"
    llm_provider_strong: LLMProviderName = "mock"
    llm_model_fast: str = "mock-fast-1"
    llm_model_strong: str = "mock-strong-1"

    # Provider API keys (never logged, never persisted).
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Per-call HTTP behaviour for the real providers.
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 0.5
    llm_max_output_tokens: int = 4096

    # Cost guard: per-workspace daily token budget.
    daily_token_budget: int = 2_000_000
    # Hard cap on tokens a single agent run may consume.
    max_tokens_per_run: int = 60_000
    # Hard cap on agent iterations (LLM<->tool round trips) per run.
    max_agent_iterations: int = 6
    # Wall-clock cap for a single agent run, seconds.
    agent_run_timeout_seconds: float = 180.0

    # ---------------------------------------------------------- embeddings ---
    embedding_provider: Literal["mock", "openai"] = "mock"
    embedding_model: str = "mock-embed-1"

    # ------------------------------------------------------------- search ----
    search_provider: SearchProviderName = "mock"
    tavily_api_key: str | None = None
    serper_api_key: str | None = None
    brave_api_key: str | None = None

    # ------------------------------------------------------------ keywords ---
    keyword_provider: KeywordProviderName = "mock"
    dataforseo_login: str | None = None
    dataforseo_password: str | None = None

    # -------------------------------------------------------------- images ---
    image_provider: ImageProviderName = "mock"
    image_api_key: str | None = None
    image_model: str = "mock-image-1"
    image_base_url: str | None = None

    # ---------------------------------------------------------------- meta ---
    # Meta Marketing API. When absent, meta_read tools return "unavailable".
    meta_access_token: str | None = None
    meta_ad_account_id: str | None = None
    meta_app_id: str | None = None
    meta_app_secret: str | None = None
    meta_graph_base_url: str = "https://graph.facebook.com"
    meta_graph_version: str = "v21.0"
    meta_api_timeout_seconds: float = 30.0

    # -------------------------------------------------------------- places ---
    google_places_api_key: str | None = None
    places_base_url: str = "https://maps.googleapis.com/maps/api/place"

    # ------------------------------------------------------------ security ---
    # Fernet key for encrypting integration tokens at rest (base64, 32 bytes).
    token_encryption_key: str | None = None
    # SSRF protection for analyze_website.
    website_fetch_timeout_seconds: float = 15.0
    website_max_bytes: int = 2_000_000
    website_max_redirects: int = 3
    website_allowed_schemes: tuple[str, ...] = ("http", "https")

    # --------------------------------------------------------------- uploads --
    upload_dir: str = "./var/uploads"
    upload_max_bytes: int = 10_000_000
    upload_allowed_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "text/csv",
        "application/pdf",
    )

    # ------------------------------------------------------------ features ---
    # Comma-separated list of origins allowed to call the API (frontend).
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)

    # ----------------------------------------------------------------- log ---
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("cors_origins", "upload_allowed_types", "website_allowed_schemes", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        """Allow comma-separated strings in env for tuple fields."""
        if isinstance(v, str):
            return tuple(part.strip() for part in v.split(",") if part.strip())
        return v

    # ------------------------------------------------------------- helpers ---
    @property
    def provider_for_tier(self) -> dict[Tier, LLMProviderName]:
        """The provider->tier map. The single source of truth for routing."""
        return {"fast": self.llm_provider_fast, "strong": self.llm_provider_strong}

    @property
    def model_for_tier(self) -> dict[Tier, str]:
        """The model->tier map. Never read a model string from anywhere else."""
        return {"fast": self.llm_model_fast, "strong": self.llm_model_strong}

    def is_mock_llm(self) -> bool:
        """True when no real provider is configured for either tier."""
        return self.llm_provider_fast == "mock" and self.llm_provider_strong == "mock"

    def has_real_llm_key(self) -> bool:
        """True when at least one configured provider has its key present."""
        providers = set(self.provider_for_tier.values())
        if "claude" in providers:
            return bool(self.anthropic_api_key)
        if "openai" in providers:
            return bool(self.openai_api_key)
        return False


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (FastAPI dependency friendly)."""
    return Settings()
