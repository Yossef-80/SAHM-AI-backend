"""Tavily search provider. Selected when ``AI_SEARCH_PROVIDER=tavily``.

Reads ``TAVILY_API_KEY``, ``AI_SEARCH_DEPTH`` (basic|advanced),
``AI_SEARCH_TIMEOUT_SECONDS`` and ``AI_SEARCH_SNIPPET_CHARS`` from ``.env``.
"""

from __future__ import annotations

import httpx

from app.ai.errors import ProviderNotConfiguredError, ProviderRequestError
from app.ai.loggers.providers.search import logger
from app.ai.providers.search.base import BaseSearchProvider, SearchResult
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
        self._client = httpx.Client(timeout=settings.search_timeout_seconds)

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        try:
            resp = self._client.post(
                _TAVILY_URL,
                json={
                    "api_key": self._key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": self._depth,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("tavily_http_error", status=exc.response.status_code, query=query)
            raise ProviderRequestError(f"Tavily returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            logger.error("tavily_request_failed", error=str(exc), query=query)
            raise ProviderRequestError(f"Tavily request failed: {type(exc).__name__}") from exc

        rows = resp.json().get("results", [])
        logger.info("tavily_ok", query=query, results=len(rows), depth=self._depth)
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=(r.get("content") or "")[: self._snippet_chars],
                published_date=r.get("published_date"),
                score=r.get("score"),
            )
            for r in rows
        ]


def build_search_provider(settings: AISettings) -> BaseSearchProvider:
    return TavilySearchProvider(settings)
