"""Contract shared by every search provider."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class SearchResult(BaseModel):
    """One normalised web hit, identical whatever the provider."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    published_date: str | None = None
    score: float | None = None


class BaseSearchProvider(ABC):
    """Every provider module exposes ``build_search_provider(settings) -> BaseSearchProvider``."""

    #: Short id shown in tool output (``source``) and logs.
    name: str = "base"
    #: True when results are placeholders (mock) and must never be presented as real.
    is_placeholder: bool = False

    @abstractmethod
    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """Return up to ``max_results`` hits. Raise ``ProviderError`` on failure."""


__all__ = ["BaseSearchProvider", "SearchResult"]
