"""Application configuration.

Every runtime knob comes from the environment. Nothing in this file hardcodes a
model name or provider; those live in the provider->tier map below and are set
purely by env vars (see .env.example).

Rule enforced across the codebase: no module other than ``config.py`` may
contain a literal provider/model string.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Tier = Literal["fast", "strong"]

#: Groq model-family prefix used only to shape compatible request payloads;
#: this does not select a provider or a model.
GROQ_GPT_OSS_MODEL_PREFIX = "openai/gpt-oss"

#: Providers that the LLM router knows how to construct.
LLMProviderName = Literal[
    "mock", "claude", "openai", "groq", "gemini", "deepseek", "mistral",
    "openrouter", "together", "fireworks", "perplexity", "xai", "custom",
]

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
    # Strict tenancy. When true, a request that names no workspace (no
    # ``X-Workspace-Id`` header and no authenticated principal) is rejected with
    # ``workspace_required`` instead of falling back to the oldest business.
    # Default false so dev keeps working untouched; set it in prod once clients
    # send the header or a token.
    require_workspace_header: bool = False
    # P1 auth. When true, the sensitive surface (tool invocation, Meta
    # connect/disconnect, recommendation approve/reject, persona PATCH) requires
    # a valid ``Authorization: Bearer`` token and a sufficient role; an anonymous
    # caller gets ``401``. When false -- the default -- an anonymous caller is
    # treated exactly as it was before authentication existed, so the current
    # frontend keeps working before its login page exists. An *authenticated*
    # caller is always subject to role checks, whatever this is set to.
    require_auth: bool = False
    # How long an issued session stays valid before ``/auth/refresh`` is needed.
    session_ttl_days: int = 30
    # Failed logins per email before the account is temporarily locked. An
    # in-memory counter for now (see ``app/api/auth.py``); P4.2 replaces it with
    # a real token-bucket limiter.
    login_max_failures: int = 8
    login_lockout_seconds: int = 900

    # ------------------------------------------------------------------ db ----
    # Postgres+pgvector is the production target. SQLite is supported so the
    # app boots and the whole flow runs with zero external services.
    database_url: str = "sqlite+aiosqlite:///./sahm.db"
    db_echo: bool = False

    # --- Supabase (HTTPS API: REST / Auth) ---------------------------------
    # These are NOT database credentials. ``database_url`` above is what
    # SQLAlchemy uses; a Supabase connection string needs the database password
    # from Dashboard -> Project Settings -> Database. These four are for the
    # PostgREST / Auth endpoints only.
    supabase_url: str | None = None
    supabase_publishable_key: str | None = None
    supabase_secret_key: str | None = None
    supabase_jwks_url: str | None = None

    # --- Supabase Auth (JWT verification) ------------------------------------
    # When set, the backend verifies Supabase-issued JWTs instead of issuing
    # its own opaque sessions. The frontend uses @supabase/supabase-js for
    # login/register/OAuth and sends the access token as Authorization: Bearer.
    use_supabase_auth: bool = False
    # Embedding dimension must match the configured embedding provider output.
    embedding_dim: int = 64
    # Vector backend: "pgvector" on Postgres, "python" (cosine in-process) on
    # SQLite or when pgvector is unavailable.
    vector_backend: Literal["auto", "pgvector", "python"] = "auto"

    # ---------------------------------------------------------------- llm ----
    # Which provider serves which tier. Switching provider or model is an env
    # change only -- no code edits anywhere else in the repo.
    llm_provider_fast: LLMProviderName = Field(
        "mock", validation_alias=AliasChoices("LLM_PROVIDER_FAST", "AI_LLM_PROVIDER")
    )
    llm_provider_strong: LLMProviderName = Field(
        "mock", validation_alias=AliasChoices("LLM_PROVIDER_STRONG", "AI_LLM_PROVIDER")
    )
    llm_model_fast: str = Field(
        "mock-fast-1",
        validation_alias=AliasChoices("LLM_MODEL_FAST", "GEMINI_MODEL", "AI_LLM_MODEL"),
    )
    llm_model_strong: str = Field(
        "mock-strong-1",
        validation_alias=AliasChoices("LLM_MODEL_STRONG", "GEMINI_MODEL", "AI_LLM_MODEL"),
    )

    # Provider API keys (never logged, never persisted).
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = Field(
        None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )

    # --- OpenAI-compatible gateways (OpenRouter, Google Gemini, vLLM, ...) ---
    # The provider stays "openai"; these two env vars point the LLM layer at any
    # endpoint that speaks the OpenAI Chat Completions API -- no code change.
    #   OpenRouter: https://openrouter.ai/api/v1/chat/completions
    #   Gemini:     https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
    # LLM_BASE_URL must be the FULL endpoint including "/chat/completions":
    # the client does no path joining. Leave it unset to use the OpenAI default.
    llm_base_url: str | None = None
    # Merged into every LLM request (OpenRouter's optional HTTP-Referer /
    # X-Title, gateways that want a project header, ...). Never logged.
    # Env format: a JSON object or "name=value,name2=value2". NoDecode hands
    # the raw env string to _parse_extra_headers instead of letting
    # pydantic-settings JSON-decode (and reject) it first.
    llm_extra_headers: Annotated[dict[str, str], NoDecode] = Field(default_factory=dict)

    # Per-call HTTP behaviour for the real providers.
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 0.5
    llm_max_output_tokens: int = 4096

    # Cost guard: per-workspace daily token budget.
    daily_token_budget: int = 2_000_000
    # Billing: USD price per one million tokens, used to turn the meter into an
    # invoice-ready estimate. Both default to 0, which means "show tokens only,
    # do not quote a price" -- inventing a rate for a customer is worse than
    # showing none. Set these once the resale price is agreed.
    token_price_in_usd_per_million: float = Field(
        0.0, validation_alias=AliasChoices("AI_TOKEN_PRICE_IN_USD_PER_MILLION")
    )
    token_price_out_usd_per_million: float = Field(
        0.0, validation_alias=AliasChoices("AI_TOKEN_PRICE_OUT_USD_PER_MILLION")
    )
    # Hard cap on tokens a single agent run may consume.
    max_tokens_per_run: int = 60_000
    # Hard cap on agent iterations (LLM<->tool round trips) per run.
    max_agent_iterations: int = 6
    # Wall-clock cap for a single agent run, seconds.
    agent_run_timeout_seconds: float = 180.0

    # -------------------------------------------------------------- memory ----
    # Max tokens in active chat history before older turns are summarized into DB
    memory_session_max_tokens: int = Field(
        10000,
        validation_alias=AliasChoices("SESSION_MEMORY_MAX_TOKENS", "MEMORY_SESSION_MAX_TOKENS"),
    )
    # Token threshold for brand profile & intelligence extraction before rolling summarization
    brand_memory_token_limit: int = Field(
        10000,
        validation_alias=AliasChoices("BRAND_MEMORY_TOKEN_LIMIT", "BRAND_IDENTITY_TOKEN_LIMIT", "MEMORY_BRAND_MAX_TOKENS"),
    )
    # Inactivity delay in seconds before updating global profile LTM in background (180s = 3 min)
    memory_ltm_debounce_seconds: int = 180
    # Maximum tokens cap for distilled BrandMemoryItem.text
    memory_item_max_tokens: int = 120
    # Maximum tokens cap for global Profile.user_context_summary
    memory_global_profile_max_tokens: int = 300
    # Number of recent turns to retain raw when session is summarized
    memory_retain_recent_turns: int = 2

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
    # Placeholder SVGs are never returned unless an operator explicitly opts in.
    image_allow_mock_assets: bool = False

    # ---------------------------------------------------------------- meta ---
    # Meta Marketing API. When absent, meta_read tools return "unavailable".
    meta_access_token: str | None = None
    meta_ad_account_id: str | None = None
    meta_page_id: str | None = None
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
    website_allowed_schemes: tuple[str, ...] | str = ("http", "https")

    # --------------------------------------------------------------- uploads --
    upload_dir: str = "./var/uploads"
    # Public HTTPS origin used only for generated image bytes that the provider
    # returns as base64. Files receive unguessable asset IDs and are served by
    # the read-only public-assets route so Meta can fetch them during approval.
    public_asset_base_url: str | None = None
    upload_max_bytes: int = 10_000_000
    upload_allowed_types: tuple[str, ...] | str = (
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "text/csv",
        "application/pdf",
    )

    # ------------------------------------------------------------ features ---
    # Comma-separated list of origins allowed to call the API (frontend).
    # NoDecode: a complex (tuple) field is JSON-decoded by pydantic-settings
    # *before* any field_validator runs, so the plain CSV form shipped in
    # .env.example ("http://localhost:3000") would fail to parse. NoDecode
    # hands the raw string to _split_csv, exactly as llm_extra_headers does.
    cors_origins: Annotated[tuple[str, ...], NoDecode] = ("http://localhost:3000",)
    # P4.2 replaced the two "*" wildcards with explicit lists. A wildcard origin
    # combined with credentials is rejected at startup by ``validate_cors``
    # (app/main.py) -- browsers already refuse it, so shipping it is a lie.
    cors_allow_methods: Annotated[tuple[str, ...], NoDecode] = (
        "GET",
        "POST",
        "PATCH",
        "PUT",
        "DELETE",
        "OPTIONS",
    )
    cors_allow_headers: Annotated[tuple[str, ...], NoDecode] = (
        "Authorization",
        "Content-Type",
        "X-Request-Id",
        "X-Workspace-Id",
    )

    # ------------------------------------------------------- security headers --
    # P4.8 / P5.9. The API only ever returns JSON, so the policy below is the
    # maximally restrictive one: no scripts, no connections, no framing.
    security_headers_enabled: bool = True
    # Strict-Transport-Security is only meaningful over TLS -- a browser ignores
    # it on http:// and it can never be served correctly from a plain-http pod,
    # so it is documented here rather than being an unconditional claim.
    security_hsts_enabled: bool = True
    security_hsts_max_age: int = 31_536_000
    content_security_policy: str = (
        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    )

    # ---------------------------------------------------------- rate limits ---
    # P4.2. Token bucket per route class, in-process. See ``app/core/ratelimit.py``
    # for the honest limitation (per-worker, not shared) and the off switch.
    rate_limit_enabled: bool = True
    # Authentication: the strictest class. A login is a password check, so a
    # burst of 10 with a 3/minute refill is generous for a human and useless to
    # a script.
    rate_limit_auth_capacity: int = 10
    rate_limit_auth_refill_per_second: float = 0.05
    # Tool invocation: costs money (each call can spend LLM tokens).
    rate_limit_tool_capacity: int = 30
    rate_limit_tool_refill_per_second: float = 0.5
    # Writes (approvals, settings, memory, campaigns).
    rate_limit_write_capacity: int = 60
    rate_limit_write_refill_per_second: float = 1.0
    # Reads: generous, because polling a task list must never 429.
    rate_limit_read_capacity: int = 120
    rate_limit_read_refill_per_second: float = 2.0
    # Paths exempt from limiting (health probes must never be throttled).
    rate_limit_exempt_paths: Annotated[tuple[str, ...], NoDecode] = (
        "/health",
        "/health/ready",
        "/docs",
        "/openapi.json",
        "/redoc",
    )

    # ----------------------------------------------------------------- log ---
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator(
        "cors_origins",
        "cors_allow_methods",
        "cors_allow_headers",
        "rate_limit_exempt_paths",
        "upload_allowed_types",
        "website_allowed_schemes",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, v: object) -> object:
        """Allow comma-separated strings (or a JSON array) in env for tuple fields."""
        if isinstance(v, str):
            raw = v.strip()
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return tuple(str(item).strip() for item in parsed if str(item).strip())
            return tuple(part.strip() for part in raw.split(",") if part.strip())
        return v

    @field_validator("llm_extra_headers", mode="before")
    @classmethod
    def _parse_extra_headers(cls, v: object) -> object:
        """Allow a JSON object or ``name=value,name2=value2`` in the env."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return {str(name): str(value) for name, value in v.items()}
        if not isinstance(v, str):
            return v
        raw = v.strip()
        if not raw:
            return {}
        if raw.startswith("{"):
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("LLM_EXTRA_HEADERS must be a JSON object")
            return {str(name): str(value) for name, value in parsed.items()}
        headers: dict[str, str] = {}
        for part in raw.split(","):
            entry = part.strip()
            if not entry:
                continue
            name, sep, value = entry.partition("=")
            if not sep:
                raise ValueError(
                    f"LLM_EXTRA_HEADERS entry '{entry}' is not in 'name=value' form"
                )
            headers[name.strip()] = value.strip()
        return headers

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
        if "claude" in providers and self.anthropic_api_key:
            return True
        return "openai" in providers and bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (FastAPI dependency friendly)."""
    return Settings()
