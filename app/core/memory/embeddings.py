"""Embedding providers behind one interface.

``mock`` produces deterministic, *semantically meaningful* vectors (hashed
bag-of-words) so similarity retrieval can be tested offline. ``openai`` calls
the embeddings API over httpx.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

import httpx

from app.config import Settings, get_settings
from app.core.errors import LLMNotConfiguredError, LLMProviderError
from app.core.llm.base import is_retryable_exception, is_retryable_status
from app.core.logging import get_logger

_logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@runtime_checkable
class EmbeddingProvider(Protocol):
    dim: int

    async def embed(self, text: str) -> list[float]:
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class MockEmbeddingProvider:
    """Deterministic hashed bag-of-words embeddings.

    Similar texts (shared tokens) land near each other, which is enough to test
    retrieval ordering. Not suitable for production semantic search.
    """

    def __init__(self, dim: int = 64, model: str = "mock-embed-1") -> None:
        self.dim = dim
        self.model = model

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    async def embed(self, text: str) -> list[float]:
        return self._vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class OpenAIEmbeddingProvider:
    """OpenAI text-embedding API over httpx."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        dim: int,
        timeout: float = 30.0,
        base_url: str = "https://api.openai.com/v1/embeddings",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.timeout = timeout
        self.base_url = base_url
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise LLMNotConfiguredError("OPENAI_API_KEY is not set for embeddings")
        try:
            response = await self.client.post(
                self.base_url,
                headers={"authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
            )
        except Exception as exc:
            if is_retryable_exception(exc):
                raise LLMProviderError(f"embeddings request failed: {type(exc).__name__}") from exc
            raise LLMProviderError(f"embeddings request failed: {exc}", retryable=False) from exc
        if response.status_code >= 400:
            raise LLMProviderError(
                f"embeddings returned {response.status_code}",
                retryable=is_retryable_status(response.status_code),
                status_code=response.status_code,
            )
        data = response.json().get("data", [])
        return [list(item["embedding"]) for item in data]

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0] if vectors else [0.0] * self.dim


_provider: EmbeddingProvider | None = None


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Cached provider chosen by config."""
    global _provider
    if _provider is not None:
        return _provider
    cfg = settings or get_settings()
    if cfg.embedding_provider == "openai":
        _provider = OpenAIEmbeddingProvider(
            api_key=cfg.openai_api_key,
            model=cfg.embedding_model,
            dim=cfg.embedding_dim,
        )
    else:
        _provider = MockEmbeddingProvider(dim=cfg.embedding_dim, model=cfg.embedding_model)
    _logger.info(
        "embedding_provider_built",
        extra={"provider": cfg.embedding_provider, "dim": cfg.embedding_dim},
    )
    return _provider


def reset_embedding_provider() -> None:
    global _provider
    _provider = None
