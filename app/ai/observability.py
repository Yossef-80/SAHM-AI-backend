"""Unified observability and monitoring integration for Sahm AI.

Connects:
1. Local Diagnostic Loggers (LLM calls & Agent execution)
2. LangSmith Tracing & Observability (100% Free Developer tier)
3. Langfuse Tracing, Metrics & Cost Tracking (100% Free Cloud or Self-hosted)
4. Local Real-Time Token Cost Estimation

Fails gracefully: if network fails, or keys are missing or invalid, the
application continues unaffected with local structured logs.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from app.ai.loggers.callbacks import AILoggingCallback, AIToolAuditCallback
from app.ai.loggers.cost import _PRICING_TABLE, estimate_token_cost, format_cost
from app.ai.settings import get_ai_settings

_logger = logging.getLogger("sahm.ai.observability")


def is_langsmith_enabled() -> bool:
    """True when LangSmith tracing is active via environment or settings."""
    cfg = get_ai_settings()
    if getattr(cfg, "langchain_tracing_v2", False):
        return True
    env_tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes")
    has_key = bool(
        getattr(cfg, "langchain_api_key", None)
        or os.getenv("LANGCHAIN_API_KEY")
        or os.getenv("LANGSMITH_API_KEY")
    )
    return env_tracing or has_key


def is_langfuse_enabled() -> bool:
    """True when Langfuse public and secret keys are available."""
    cfg = get_ai_settings()
    pk = getattr(cfg, "langfuse_public_key", None) or os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = getattr(cfg, "langfuse_secret_key", None) or os.getenv("LANGFUSE_SECRET_KEY")
    return bool(pk and sk)


def get_langfuse_callback() -> BaseCallbackHandler | None:
    """Safely initialize the Langfuse LangChain callback handler.

    Returns None if Langfuse is not enabled or if initialization encounters an error.
    """
    if not is_langfuse_enabled():
        return None

    try:
        from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

        cfg = get_ai_settings()
        # Ensure environment variables are populated for Langfuse v5 client
        pk = getattr(cfg, "langfuse_public_key", None) or os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = getattr(cfg, "langfuse_secret_key", None) or os.getenv("LANGFUSE_SECRET_KEY")
        host = (
            getattr(cfg, "langfuse_host", None)
            or os.getenv("LANGFUSE_HOST")
            or "https://cloud.langfuse.com"
        )

        if pk:
            os.environ["LANGFUSE_PUBLIC_KEY"] = pk
        if sk:
            os.environ["LANGFUSE_SECRET_KEY"] = sk
        if host:
            os.environ["LANGFUSE_HOST"] = host

        return LangfuseCallbackHandler(public_key=pk)
    except Exception as exc:
        _logger.warning("langfuse_init_failed", extra={"error": str(exc)})
        return None


def get_observability_callbacks(
    agent: str | None = None,
    audit: AIToolAuditCallback | None = None,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
) -> list[BaseCallbackHandler]:
    """Assemble all active callbacks for an agent or graph run.

    Always includes:
    1. AILoggingCallback (detailed local LLM & Agent diagnostics)
    2. audit (DB tool persistence, if supplied)

    Optionally includes:
    3. Langfuse callback handler (if configured in .env)
    Note: LangSmith requires no manual callback injection when LANGCHAIN_TRACING_V2=true,
    as LangChain natively picks it up from the process environment.
    """
    callbacks: list[BaseCallbackHandler] = [AILoggingCallback(agent=agent)]
    if audit is not None:
        callbacks.append(audit)

    # Langfuse Integration
    lf_handler = get_langfuse_callback()
    if lf_handler is not None:
        callbacks.append(lf_handler)

    return callbacks


def get_observability_status() -> dict[str, Any]:
    """Diagnostic status report for logging and observability integrations."""
    cfg = get_ai_settings()
    pk = getattr(cfg, "langfuse_public_key", None) or os.getenv("LANGFUSE_PUBLIC_KEY")
    return {
        "local_logging": {
            "status": "active",
            "log_dir": cfg.log_dir,
            "log_level": cfg.log_level.value,
            "format": cfg.log_format.value,
            "llm_log": f"{cfg.log_dir}/providers/llm.log",
            "agent_logs": f"{cfg.log_dir}/agents/",
            "diagnostic_log": f"{cfg.log_dir}/agents/diagnostic.log",
        },
        "langsmith": {
            "enabled": is_langsmith_enabled(),
            "project": getattr(cfg, "langchain_project", "sahm-ai")
            or os.getenv("LANGCHAIN_PROJECT", "sahm-ai"),
            "endpoint": getattr(cfg, "langchain_endpoint", None)
            or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        },
        "langfuse": {
            "enabled": is_langfuse_enabled(),
            "host": getattr(cfg, "langfuse_host", "https://cloud.langfuse.com")
            or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            "public_key_configured": bool(pk),
            "public_key_preview": f"{pk[:8]}..." if pk and len(pk) > 8 else None,
        },
        "cost_estimator": {
            "status": "active",
            "supported_models_count": len(_PRICING_TABLE),
            "free_models": ["mock", "ollama", "local"],
        },
    }


__all__ = [
    "estimate_token_cost",
    "format_cost",
    "get_langfuse_callback",
    "get_observability_callbacks",
    "get_observability_status",
    "is_langfuse_enabled",
    "is_langsmith_enabled",
]
