"""Mock search provider. Selected when ``AI_SEARCH_PROVIDER=mock``.

Returns deterministic placeholder hits flagged ``is_placeholder=True`` so the
``web_search`` tool labels them as such -- never presented as real evidence.
"""

from __future__ import annotations

import hashlib

from app.ai.providers.search.base import BaseSearchProvider, SearchResult
from app.ai.settings import AISettings


class MockSearchProvider(BaseSearchProvider):
    name = "mock"
    is_placeholder = True

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        digest = hashlib.sha256(query.encode()).hexdigest()[:8]
        return [
            SearchResult(
                title=f"[mock] Result {i + 1} for '{query}'",
                url=f"https://example.com/mock/{digest}/{i + 1}",
                snippet="Placeholder snippet -- no real search provider is configured.",
            )
            for i in range(min(max_results, 3))
        ]


def build_search_provider(settings: AISettings) -> BaseSearchProvider:  # noqa: ARG001
    return MockSearchProvider()
