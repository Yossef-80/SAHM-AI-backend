"""Disabled search. Selected when ``AI_SEARCH_PROVIDER=none``.

Raises ``ProviderNotConfiguredError`` so ``web_search`` reports
``status="unavailable"`` and the agent tells the user instead of guessing.
"""

from __future__ import annotations

from app.ai.errors import ProviderNotConfiguredError
from app.ai.providers.search.base import BaseSearchProvider, SearchResult
from app.ai.settings import AISettings


class DisabledSearchProvider(BaseSearchProvider):
    name = "none"

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        raise ProviderNotConfiguredError(
            "Web search is disabled (AI_SEARCH_PROVIDER=none). Ask the user to paste sources."
        )


def build_search_provider(settings: AISettings) -> BaseSearchProvider:  # noqa: ARG001
    return DisabledSearchProvider()
