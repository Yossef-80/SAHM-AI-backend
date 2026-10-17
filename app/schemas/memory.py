"""Schemas for ``/api/v1/memory/*``.

The kind list is no longer a literal regex in this file: it is derived from
``app.core.memory.kinds`` so a new kind cannot be creatable through one layer and
invisible to another (that is exactly how six kinds were written but never
retrieved). Service-owned kinds -- ``brand_identity``, ``strategy_blueprint``,
the research artifacts, ``chat_attachment`` -- are deliberately *not* writable
through the public API; their producers own their shape.

Bounds are enforced here, not in the handler: a memory row is injected into a
prompt on every agent run, so an unbounded ``text`` is a prompt-injection surface
and an unbounded ``metadata`` is a way to write a 10 MB blob through a form.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import Field, field_validator

from app.core.memory.kinds import API_WRITABLE_KINDS, MEMORY_KINDS
from app.schemas.common import ApiModel

#: A single memory item is injected verbatim into the brand context, which is
#: capped at ~900 tokens. 4 000 characters is comfortably more than one item
#: needs; the store additionally truncates per kind.
MAX_MEMORY_TEXT = 4_000
#: Metadata is a tag bag, not a document.
MAX_METADATA_KEYS = 32
MAX_METADATA_VALUE = 500
MAX_METADATA_JSON = 8_000

#: Every kind the system knows about (for filters and documentation).
MEMORY_KIND_VALUES: tuple[str, ...] = tuple(MEMORY_KINDS)
#: The subset a client may create directly.
MEMORY_WRITABLE_KIND_VALUES: tuple[str, ...] = tuple(API_WRITABLE_KINDS)

MemoryKind = Annotated[
    str,
    Field(pattern="^(" + "|".join(API_WRITABLE_KINDS) + ")$"),
]
AnyMemoryKind = Annotated[
    str,
    Field(pattern="^(" + "|".join(MEMORY_KINDS) + ")$"),
]


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

    if len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")) > MAX_METADATA_JSON:
        raise ValueError(f"metadata serialises to more than {MAX_METADATA_JSON} bytes")
    return value


class MemoryItemCreate(ApiModel):
    """``POST /memory/items`` body.

    ``importance`` and ``pinned`` are ranking inputs the author may set: a
    pinned offer must stay in the prompt even when a query matches it weakly.
    """

    # Any known kind passes validation; the handler rejects the six service-owned
    # ones with an explanation. Rejecting them in the pattern would make the
    # response a field-level "does not match ^(..)$" instead of saying *why*
    # ``brand_identity`` cannot be hand-written.
    kind: AnyMemoryKind
    text: str = Field(min_length=1, max_length=MAX_MEMORY_TEXT)
    metadata: dict[str, Any] = Field(default_factory=dict)
    campaign_id: str | None = Field(default=None, max_length=64)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    pinned: bool = False

    @field_validator("metadata")
    @classmethod
    def _bound_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        _check_metadata(v)
        return v


class MemoryItemUpdate(ApiModel):
    """``PATCH /memory/items/{id}`` body.

    ``kind`` is intentionally **not** editable: it decides the item's lifetime,
    prompt section and retrieval slice. Re-kind a row by deleting and re-creating
    it.
    """

    text: str | None = Field(default=None, min_length=1, max_length=MAX_MEMORY_TEXT)
    metadata: dict[str, Any] | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    pinned: bool | None = None

    @field_validator("metadata")
    @classmethod
    def _bound_metadata(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        return _check_metadata(v)


class MemoryItemOut(ApiModel):
    """One memory item. Never includes internals the client cannot act on."""

    id: str
    business_id: str
    kind: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    campaign_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    importance: float = 0.5
    pinned: bool = False
    expires_at: datetime | None = None
    archived_at: datetime | None = None
    #: Retrieval signals (only on ``GET /memory/items?q=``).
    score: float | None = None
    reason: str | None = None


class MemoryItemList(ApiModel):
    """``GET /memory/items`` response.

    ``mode`` names the ordering rule, because the two are genuinely different:
    ``"recent"`` is newest-first (the editor list), ``"lexical"`` is
    BM25 + recency + affinity relevance to ``q`` (never substring matching --
    ``reason`` on each item says which terms matched).
    """

    items: list[MemoryItemOut] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    kind: str | None = None
    query: str | None = None
    mode: str = "recent"


class MemoryPruneRequest(ApiModel):
    """``POST /memory/prune`` body. ``kind`` omitted means "every kind"."""

    kind: AnyMemoryKind | None = None
    dry_run: bool = True


class MemoryPruneResponse(ApiModel):
    """How many rows would be (or were) archived, per kind."""

    dry_run: bool = True
    archived_by_kind: dict[str, int] = Field(default_factory=dict)
    total_archived: int = 0


__all__ = [
    "MAX_MEMORY_TEXT",
    "MEMORY_KINDS",
    "MEMORY_KIND_VALUES",
    "MEMORY_WRITABLE_KIND_VALUES",
    "MemoryItemCreate",
    "MemoryItemList",
    "MemoryItemOut",
    "MemoryItemUpdate",
    "MemoryPruneRequest",
    "MemoryPruneResponse",
]
