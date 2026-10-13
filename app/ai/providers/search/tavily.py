"""Tavily search provider, guarded by the app.ai search tool's cost policy.

Reads ``TAVILY_API_KEY``, ``AI_SEARCH_DEPTH``, ``AI_SEARCH_TIMEOUT_SECONDS``
and ``AI_SEARCH_SNIPPET_CHARS`` from the AI settings object.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.ai.errors import ProviderNotConfiguredError, ProviderRequestError
from app.ai.loggers.providers.search import logger
from app.ai.providers.search.base import BaseSearchProvider, SearchResult
from app.ai.security import redact_credential_text
from app.ai.settings import AISettings

_TAVILY_URL = "https://api.tavily.com/search"


class TavilySearchProvider(BaseSearchProvider):
    name = "tavily"

    def __init__(self, settings: AISettings) -> None:
        if not settings.tavily_api_key:
            raise ProviderNotConfiguredError(
                "AI_SEARCH_PROVIDER=tavily but TAVILY_API_KEY is empty."
            )
        self._key = settings.tavily_api_key
        self._depth = settings.search_depth.value
        self._snippet_chars = settings.search_snippet_chars
        self._timeout = settings.search_timeout_seconds

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        safe_query = redact_credential_text(query).strip()
        if not safe_query:
            raise ProviderRequestError("Tavily query must not be empty.")
        result_limit = max(1, min(int(max_results), 20))
        try:
            # A short-lived client avoids leaking an unclosed connection pool
            # from the process-wide cached provider instance.
            with httpx.Client(timeout=self._timeout, trust_env=False) as client:
                resp = client.post(
                    _TAVILY_URL,
                    json={
                        "api_key": self._key,
                        "query": safe_query,
                        "max_results": result_limit,
                        "search_depth": self._depth,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "tavily_http_error",
                status=exc.response.status_code,
                query=safe_query,
            )
            raise ProviderRequestError(
                f"Tavily returned HTTP {exc.response.status_code}"
            ) from exc
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.error(
                "tavily_request_failed",
                error_type=type(exc).__name__,
                query=safe_query,
            )
            raise ProviderRequestError(
                f"Tavily request failed: {type(exc).__name__}"
            ) from exc

        rows: Any = data.get("results", []) if isinstance(data, dict) else []
        if not isinstance(rows, list):
            raise ProviderRequestError("Tavily returned an invalid results payload.")
        logger.info("tavily_ok", query=safe_query, results=len(rows), depth=self._depth)
        return [
            SearchResult(
                title=str(row.get("title", "")),
                url=str(row.get("url", "")),
                snippet=str(row.get("content") or "")[: self._snippet_chars],
                published_date=row.get("published_date"),
                score=row.get("score"),
            )
            for row in rows[:result_limit]
            if isinstance(row, dict)
        ]


def build_search_provider(settings: AISettings) -> BaseSearchProvider:
    return TavilySearchProvider(settings)


__all__ = ["TavilySearchProvider", "build_search_provider"]
