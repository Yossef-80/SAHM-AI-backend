"""Layered brand memory: **no embeddings, no vector store**.

The package is deliberately small and deterministic:

``kinds``      the single registry of memory kinds (layer, TTL, importance,
               token cap, which task groups may read it, whether the public API
               may create it).
``tokens``     one token counter for the whole repo (Arabic-aware).
``lexical``    BM25 over an Arabic/English-normalised token stream.
``policy``     every threshold, read from settings; one frozen object per access.
``lifecycle``  dedupe keys, TTL/expiry, pruning order.
``ranking``    the scoring function (affinity + lexical + recency + importance).
``store``      the only writer and the only reader of ``brand_memory_items``.

Why no embeddings
-----------------
The previous implementation "embedded" text into 64 hashed bags-of-words and
stored the result in a ``Vector`` column; the only provider that ever ran was
``mock``. It cost an extra column, an extra migration, an extra config surface
(``EMBEDDING_DIM``, ``EMBEDDING_PROVIDER``, ``VECTOR_BACKEND``), a reindex
endpoint and a class of "provider changed, everything must be re-embedded"
failures -- while ranking no better than keyword overlap. Marketing memory is
short, dense, bilingual and heavily tagged by kind; BM25 + recency + kind
affinity retrieves it deterministically and needs no external service.

Layout of the memory tiers (see ``docs/MEMORY_ARCHITECTURE.md``):

    working    in-flight turn, not persisted
    session    conversation turns + rolling summary (``Conversation.summary``)
    profile    one distilled row per business (``Profile.user_context_summary``)
    brand      ``brand_memory_items`` -- identity, plans, research, learnings
    scratch    attachments, reference analyses, competitor notes (TTL-bounded)
"""

from app.core.memory.kinds import (
    API_WRITABLE_KINDS,
    MEMORY_KINDS,
    MEMORY_KIND_SPECS,
    MemoryKindSpec,
    is_known_kind,
    label_for,
    max_tokens_for,
    spec_for,
    task_group,
)
from app.core.memory.lifecycle import (
    dedupe_digest,
    expiry_for,
    is_expired,
    memory_defaults,
    prune_rank,
    utcnow,
)
from app.core.memory.policy import MemoryPolicy, get_memory_policy
from app.core.memory.ranking import RankedMemory, rank_items, take_top
from app.core.memory.store import (
    BRAND_CONTEXT_TOKEN_CAP,
    BrandMemoryStore,
    format_context_for_prompt,
    get_brand_context,
)
from app.core.memory.tokens import count_tokens, fits, truncate_to_tokens

__all__ = [
    "API_WRITABLE_KINDS",
    "BRAND_CONTEXT_TOKEN_CAP",
    "BrandMemoryStore",
    "MEMORY_KINDS",
    "MEMORY_KIND_SPECS",
    "MemoryKindSpec",
    "MemoryPolicy",
    "RankedMemory",
    "count_tokens",
    "dedupe_digest",
    "expiry_for",
    "fits",
    "format_context_for_prompt",
    "get_brand_context",
    "get_memory_policy",
    "is_expired",
    "is_known_kind",
    "label_for",
    "max_tokens_for",
    "memory_defaults",
    "prune_rank",
    "rank_items",
    "spec_for",
    "take_top",
    "task_group",
    "truncate_to_tokens",
    "utcnow",
]
