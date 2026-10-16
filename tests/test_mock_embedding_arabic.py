"""The offline embedding provider must give Arabic a direction.

This product's memory is written mostly in Arabic, and the mock provider's
tokenizer was ``[a-z0-9']+`` -- Latin only. Every Arabic memory therefore
embedded to a zero vector, which has no direction to compare: pgvector returns
NaN for its cosine distance, so retrieval was ordered arbitrarily (and, before
the store learned to detect it, surfaced a null score).

Only the *mock* provider is affected -- a real embedding model handles Arabic
natively -- but the mock is what the test suite and every offline install run on,
so its behaviour decides whether retrieval tests mean anything.
"""

from __future__ import annotations

import pytest

from app.core.memory.embeddings import MockEmbeddingProvider
from app.db.models import cosine

pytestmark = pytest.mark.anyio


async def test_arabic_text_is_no_longer_a_zero_vector() -> None:
    provider = MockEmbeddingProvider()
    vector = await provider.embed("قهوة مختصة للمكاتب")
    assert any(value != 0.0 for value in vector)
    assert len(vector) == provider.dim


async def test_arabic_similarity_is_meaningful() -> None:
    """Shared Arabic words must land nearer than unrelated Arabic text."""
    provider = MockEmbeddingProvider()
    query = await provider.embed("قهوة مختصة للمكاتب")
    related = await provider.embed("بن مختصة للمكاتب")
    unrelated = await provider.embed("إطارات سيارات وقطع غيار")

    assert cosine(query, related) > cosine(query, unrelated)
    assert cosine(query, unrelated) == pytest.approx(0.0, abs=1e-9)


async def test_arabic_and_english_still_differ() -> None:
    """The fallback must not make every script collide into one bag of words."""
    provider = MockEmbeddingProvider()
    arabic = await provider.embed("قهوة")
    english = await provider.embed("coffee")
    assert cosine(arabic, english) == pytest.approx(0.0, abs=1e-9)


async def test_english_tokens_are_unchanged() -> None:
    """The Latin path is the one the existing suite was written against."""
    provider = MockEmbeddingProvider()
    assert provider._tokens("Coffee Subscription 12%") == ["coffee", "subscription", "12"]
    assert provider._tokens("don't stop") == ["don't", "stop"]


async def test_empty_text_still_has_no_direction() -> None:
    """Nothing to embed stays nothing: the store's guard covers this case."""
    provider = MockEmbeddingProvider()
    assert set(await provider.embed("")) == {0.0}
    assert set(await provider.embed("   \n\t ")) == {0.0}


async def test_a_non_letter_string_gets_a_direction() -> None:
    """Digits and punctuation are not words, but they are still information."""
    provider = MockEmbeddingProvider()
    vector = await provider.embed("12% !!!")
    assert any(value != 0.0 for value in vector)


async def test_the_vector_is_deterministic() -> None:
    provider = MockEmbeddingProvider()
    assert await provider.embed("قهوة مختصة") == await provider.embed("قهوة مختصة")
