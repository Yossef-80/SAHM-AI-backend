"""Evidence-preserving web search for the ``app.ai`` LangGraphs."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.ai.errors import ProviderError
from app.ai.loggers.tools.web_search import logger
from app.ai.providers.search import get_search_provider
from app.ai.security import redact_credential_text, redact_sensitive_data
from app.ai.settings import get_ai_settings


class WebSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    max_results: int = Field(default=3, ge=1, le=5)


def _search_result(
    *,
    status: str,
    query: str,
    source: str,
    results: list[dict[str, Any]] | None = None,
    reason: str | None = None,
) -> str:
    return json.dumps(
        redact_sensitive_data(
            {
                "status": status,
                "query": query,
                "source": source,
                "results": results or [],
                "unavailable_reason": reason,
            }
        ),
        ensure_ascii=False,
        default=str,
    )


@tool(args_schema=WebSearchInput)
def web_search_tool(
    query: str,
    max_results: int = 3,
    config: RunnableConfig | None = None,  # noqa: ARG001 - reserved for LangGraph config injection
) -> str:
    """Search the configured public-web provider and return source URLs/snippets.

    Use focused queries for market or competitor research. Treat every hit as a
    source lead, not proof: cite its URL and verify important claims. Search is
    metered, so live providers require ``AI_ALLOW_PAID_SEARCH_CALLS=true``;
    placeholder/mock results are always labelled and must never be presented as
    real sources. The tool does not make a second LLM call to summarize results.
    """
    safe_query = redact_credential_text(query).strip()
    settings = get_ai_settings()
    try:
        provider = get_search_provider(settings.search_provider)
        source = str(getattr(provider, "name", settings.search_provider.value))
        if not getattr(provider, "is_placeholder", False) and not settings.allow_paid_search_calls:
            reason = (
                "Live web search is metered and is disabled until an administrator explicitly "
                "enables AI_ALLOW_PAID_SEARCH_CALLS."
            )
            logger.warning("web_search_blocked_by_cost_guard", provider=source)
            return _search_result(
                status="unavailable", query=safe_query, source=source, reason=reason
            )
        hits = provider.search(safe_query, max_results=max_results)
    except ProviderError as exc:
        logger.warning(
            "web_search_unavailable",
            provider=settings.search_provider.value,
            error_type=type(exc).__name__,
        )
        return _search_result(
            status="unavailable",
            query=safe_query,
            source=settings.search_provider.value,
            reason=str(exc),
        )
    except Exception as exc:  # provider contract errors must not crash the agent graph
        logger.error(
            "web_search_failed",
            provider=settings.search_provider.value,
            error_type=type(exc).__name__,
        )
        return _search_result(
            status="unavailable",
            query=safe_query,
            source=settings.search_provider.value,
            reason="The configured search provider failed; no search findings are available.",
        )

    rows = [
        {
            "title": hit.title,
            "url": hit.url,
            "snippet": hit.snippet,
            "published_date": hit.published_date,
            "score": hit.score,
        }
        for hit in hits[:max_results]
    ]
    placeholder = bool(getattr(provider, "is_placeholder", False))
    status = "partial" if placeholder or not rows else "ok"
    reason = None
    if placeholder:
        reason = "These are deterministic mock placeholders, not live search results."
    elif not rows:
        reason = "The provider returned no results for this query."
    logger.info(
        "web_search_completed",
        provider=source,
        result_count=len(rows),
        placeholder=placeholder,
    )
    return _search_result(
        status=status,
        query=safe_query,
        source=source,
        results=rows,
        reason=reason,
    )


__all__ = ["WebSearchInput", "web_search_tool"]
