"""Search providers: pick one by enum, import its module lazily.

    from app.ai.providers.search import get_search_provider
    provider = get_search_provider()        # uses AI_SEARCH_PROVIDER from .env
    hits = provider.search("coffee brands in Egypt", max_results=5)

Every module in this folder exposes:

    def build_search_provider(settings) -> BaseSearchProvider
"""

from __future__ import annotations

from functools import lru_cache

from app.ai.enums import SearchProvider
from app.ai.loggers.providers.search import logger
from app.ai.providers._loader import load_provider_module
from app.ai.providers.search.base import BaseSearchProvider, SearchResult
from app.ai.settings import get_ai_settings

_PROVIDER_MODULES: dict[SearchProvider, str] = {
    SearchProvider.TAVILY: "app.ai.providers.search.tavily",
    SearchProvider.MOCK: "app.ai.providers.search.mock",
    SearchProvider.NONE: "app.ai.providers.search.none",
}


@lru_cache(maxsize=8)
def _build(choice: SearchProvider) -> BaseSearchProvider:
    module = load_provider_module(_PROVIDER_MODULES, choice, "search")
    provider = module.build_search_provider(get_ai_settings())
    logger.info("search_provider_built", provider=choice.value)
    return provider


def get_search_provider(provider: SearchProvider | None = None) -> BaseSearchProvider:
    """Cached provider instance (reuses its HTTP connection pool)."""
    return _build(provider or get_ai_settings().search_provider)


def reset_search_providers() -> None:
    _build.cache_clear()


__all__ = ["BaseSearchProvider", "SearchResult", "get_search_provider", "reset_search_providers"]
