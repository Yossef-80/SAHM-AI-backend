"""Schemas for ``/api/v1/memory/*`` (P2.2).

The five kinds that had a read path and no write path -- ``voice_example``,
``visual_style``, ``offer``, ``audience``, ``competitor_note`` -- are writable
through ``POST /memory/items``, and a brand-voice example is a ``voice_example``
memory item by design rather than a separate resource.

Bounds are enforced here, not in the handler: a memory row is embedded into a
prompt on every agent run, so an unbounded ``text`` is a prompt-injection
surface and an unbounded ``metadata`` is a way to write a 10 MB blob through a
form.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import Field, field_validator

from app.core.memory.store import MEMORY_KINDS
from app.schemas.common import ApiModel

#: A single memory item is embedded verbatim into the brand context
#: (``app/core/memory/store.py::build_context``), which is capped at 900 tokens.
#: 4 000 characters is comfortably more than one item needs.
MAX_MEMORY_TEXT = 4_000
#: Metadata is a tag bag, not a document.
MAX_METADATA_KEYS = 32
MAX_METADATA_VALUE = 500
MAX_METADATA_JSON = 8_000

#: The literal 7 kinds, exposed to the client so a form can render the picker
#: without hardcoding them.
MEMORY_KIND_VALUES: tuple[str, ...] = tuple(MEMORY_KINDS)

MemoryKind = Annotated[str, Field(pattern="^(voice_example|visual_style|offer|audience|competitor_note|learning|reference_analysis)$")]


def _check_metadata(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Bound metadata by key count, value length and total serialised size."""
    if value is None:
        return None
    if len(value) > MAX_METADATA_KEYS:
        raise ValueError(f"metadata may hold at most {MAX_METADATA_KEYS} keys")
    for key, item in value.items():
        if len(key) > 100:
            raise ValueError("metadata key names are capped at 100 characters")
        if isinstance(item, str) and len(item) > MAX_METADATA_VALUE:
            raise ValueError(
                f"metadata['{key}'] is longer than {MAX_METADATA_VALUE} characters"
            )
    import json

    if len(json.dumps(value, default=str)) > MAX_METADATA_JSON:
        raise ValueError(f"metadata serialises to more than {MAX_METADATA_JSON} bytes")
    return value


class MemoryItemCreate(ApiModel):
    """``POST /memory/items`` body. One item of any of the 7 kinds."""

    kind: MemoryKind
    text: str = Field(min_length=1, max_length=MAX_MEMORY_TEXT)
    metadata: dict[str, Any] = Field(default_factory=dict)
    campaign_id: str | None = Field(default=None, max_length=64)

    @field_validator("metadata")
    @classmethod
    def _bound_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        _check_metadata(v)
        return v


class MemoryItemUpdate(ApiModel):
    """``PATCH /memory/items/{id}`` body.

    ``kind`` is intentionally **not** editable: it decides which retrieval slice
    the row belongs to and which prompt section consumes it. Re-kind a row by
    deleting and re-creating it, so the embedding is rebuilt from scratch.
    """

    text: str | None = Field(default=None, min_length=1, max_length=MAX_MEMORY_TEXT)
    metadata: dict[str, Any] | None = None

    @field_validator("metadata")
    @classmethod
    def _bound_metadata(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        return _check_metadata(v)


class MemoryItemOut(ApiModel):
    """One memory item. No embedding: it is large, useless to a client, and the
    only consumer of similarity is the search box."""

    id: str
    business_id: str
    kind: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    campaign_id: str | None = None
    created_at: datetime | None = None
    #: Only present on ``GET /memory/items?q=``: the cosine similarity to the
    #: query, 0..1.
    score: float | None = None


class MemoryItemList(ApiModel):
    """``GET /memory/items`` response.

    ``mode`` tells the client which ordering rule produced ``items``, because the
    two are genuinely different and a UI must not assume substring matching:
    ``"recent"`` is newest-first, ``"similar"`` is embedding-similarity to ``q``.
    """

    items: list[MemoryItemOut] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    kind: str | None = None
    query: str | None = None
    mode: str = "recent"


class MemoryReindexRequest(ApiModel):
    """``POST /memory/reindex`` body. ``kind`` omitted means "every kind"."""

    kind: str | None = None


class MemoryReindexResponse(ApiModel):
    reindexed: int
    kind: str | None = None
    item_ids: list[str] = Field(default_factory=list)


__all__ = [
    "MAX_MEMORY_TEXT",
    "MEMORY_KIND_VALUES",
    "MEMORY_KINDS",
    "MemoryItemCreate",
    "MemoryItemList",
    "MemoryItemOut",
    "MemoryItemUpdate",
    "MemoryReindexRequest",
    "MemoryReindexResponse",
]