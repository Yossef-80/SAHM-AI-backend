"""Every selectable option in the AI layer, as an Enum.

This is the ONLY place where the set of supported providers / methods / backends
is declared. To switch provider you change a value in ``.env`` (validated
against these enums by ``app.ai.settings``) -- you never edit code.

To ADD a new provider:
    1. Add a member here (e.g. ``LLMProvider.GROQ = "groq"``).
    2. Create its module (e.g. ``app/ai/providers/llm/groq.py``) exposing the
       same contract as the other modules in that folder.
    3. Map the enum member to the module path in that folder's ``__init__.py``.
"""

from __future__ import annotations

from enum import StrEnum


# ----------------------------------------------------------------- providers --


class LLMProvider(StrEnum):
    """Which chat-model backend serves the agents."""

    OLLAMA = "ollama"  # local or remote (ngrok / Colab) Ollama server
    OPENAI = "openai"  # OpenAI or any OpenAI-compatible endpoint
    ANTHROPIC = "anthropic"  # Claude
    GEMINI = "gemini"  # Google Gemini
    GROQ = "groq"  # Groq (Llama3, Mixtral, etc.)
    OPENROUTER = "openrouter"  # OpenRouter (Aggregator)
    MOCK = "mock"  # deterministic, offline -- for tests and dry runs


class SearchProvider(StrEnum):
    """Which web-search API backs the ``web_search`` tool."""

    TAVILY = "tavily"
    MOCK = "mock"  # deterministic placeholder results, clearly labelled
    NONE = "none"  # search disabled: the tool reports ``unavailable``


class DatabaseProvider(StrEnum):
    """Where LangGraph persists agent state (checkpointer) and long-term memory (store)."""

    MEMORY = "memory"  # in-process, lost on exit -- tests / quick demos
    SQLITE = "sqlite"  # single file on disk -- local development
    POSTGRES = "postgres"  # production


# ------------------------------------------------------------------- methods --


class StructuredOutputMethod(StrEnum):
    """How a chat model is forced to return a Pydantic object.

    ``JSON_SCHEMA`` is the most reliable for Ollama (the schema is sent as the
    ``format`` constraint). ``FUNCTION_CALLING`` is the safest for OpenAI /
    Anthropic. ``JSON_MODE`` only guarantees valid JSON, not the schema.
    """

    JSON_SCHEMA = "json_schema"
    FUNCTION_CALLING = "function_calling"
    JSON_MODE = "json_mode"


class SearchDepth(StrEnum):
    """Search thoroughness (Tavily: ``basic`` = 1 credit, ``advanced`` = 2 credits)."""

    BASIC = "basic"
    ADVANCED = "advanced"


# ------------------------------------------------------------------- logging --


class LogKind(StrEnum):
    """Top-level log folder. Each kind gets ``logs/<kind>s/<name>.log``."""

    AGENT = "agent"
    TOOL = "tool"
    PROVIDER = "provider"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LogFormat(StrEnum):
    TEXT = "text"  # human-readable, one line per event
    JSON = "json"  # one JSON object per line (log shippers / jq)


# --------------------------------------------------------------- identifiers --


class AgentName(StrEnum):
    """Stable agent identifiers (used for logger names and memory metadata)."""

    NOUR = "nour"
    OMAR = "omar"
    LAYLA = "layla"
    KARIM = "karim"
    SALMA = "salma"


class ToolName(StrEnum):
    """Stable tool identifiers. Must match the ``name`` of each LangChain tool."""

    WEB_SEARCH = "web_search"
    ANALYZE_WEBSITE = "analyze_website"
    GET_BRAND_CONTEXT = "get_brand_context"
    SAVE_LEARNING = "save_learning"


class LearningCategory(StrEnum):
    """Buckets for findings stored by ``save_learning``."""

    MARKET = "market"
    COMPETITOR = "competitor"
    AUDIENCE = "audience"
    CREATIVE = "creative"
    PERFORMANCE = "performance"


__all__ = [
    "AgentName",
    "DatabaseProvider",
    "LLMProvider",
    "LearningCategory",
    "LogFormat",
    "LogKind",
    "LogLevel",
    "SearchDepth",
    "SearchProvider",
    "StructuredOutputMethod",
    "ToolName",
]
