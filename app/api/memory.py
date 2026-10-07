"""Brand-memory CRUD: ``/api/v1/memory/*`` (P2.2).

Why this file exists
--------------------
``MEMORY_KINDS`` declares seven kinds, but before P2.2 only two of them were
ever written anywhere: ``learning`` (by the agent's ``save_learning`` tool) and
``reference_analysis`` (by ``analyze_reference_creative``). The other five --
``voice_example``, ``visual_style``, ``offer``, ``audience``,
``competitor_note`` -- had a read path and no write path at all, which is why the
vector store, the most valuable asset in the product, was ~90% empty. This
router is what makes them writable.

Isolation
---------
Every statement filters on ``business_id``, and an id-addressed route answers
**404**, never 403, for a row belonging to another workspace -- a 403 confirms
the row exists, which turns a guessed id into an enumeration oracle. That is the
same rule the rest of the API follows (``app/api/deps.py::assert_owned``).

Ordering, honestly
------------------
``GET /memory/items`` has two modes and says which one produced the result:

* no ``q`` -> ``mode="recent"``, newest first. This is the editor list.
* ``q`` given -> ``mode="similar"``, ordered by **embedding similarity** to ``q``
  through ``BrandMemoryStore.search()``. It is *not* substring matching: a row
  whose text does not contain the query can legitimately come first if it means
  something similar, and a row that contains it may not. ``score`` is returned so
  the UI can show why.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.api.deps import (
    AuthDep,
    SessionDep,
    SettingsDep,
    WorkspaceDep,
    assert_campaign_owned,
    require_role,
)
from app.core.errors import NotFoundError, ValidationFailed
from app.core.memory.store import MEMORY_KINDS, BrandMemoryStore
from app.db.models import BrandMemoryItem
from app.schemas.memory import (
    MemoryItemCreate,
    MemoryItemList,
    MemoryItemOut,
    MemoryItemUpdate,
    MemoryReindexRequest,
    MemoryReindexResponse,
)

router = APIRouter(prefix="/memory", tags=["memory"])

#: Writing brand memory is a write: owner or member, never viewer (P1.7 matrix).
WRITE_MEMORY = "member"
#: Re-embedding is an operator action, not an authoring one.
REINDEX_MEMORY = "owner"

#: Hard ceiling on one page. A UI asking for 10 000 rows gets 200 and a truthful
#: ``total`` rather than a silent 422 it cannot handle.
MAX_LIMIT = 200
DEFAULT_LIMIT = 50


def _out(item: BrandMemoryItem, score: float | None = None) -> MemoryItemOut:
    return MemoryItemOut(
        id=item.id,
        business_id=item.business_id,
        kind=item.kind,
        text=item.text,
        metadata=dict(item.metadata_ or {}),
        campaign_id=item.campaign_id,
        created_at=item.created_at,
        score=score,
    )


def _validate_kind(kind: str | None) -> None:
    """Unknown kind is a 422, not a silent empty list.

    An empty result would make a typo in a filter look like "you have no
    memories", which is exactly the kind of quiet wrong answer an admin panel
    should not produce.
    """
    if kind is not None and kind not in MEMORY_KINDS:
        raise ValidationFailed(
            f"unknown memory kind '{kind}'",
            detail={"allowed": list(MEMORY_KINDS)},
        )


@router.get("/items", response_model=MemoryItemList)
async def list_memory_items(
    ws: WorkspaceDep,
    db: SessionDep,
    kind: str | None = Query(default=None, description="One of MEMORY_KINDS"),
    q: str | None = Query(
        default=None,
        max_length=500,
        description="Semantic query. Results are similarity-ordered, not substring-matched.",
    ),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
) -> MemoryItemList:
    """List this workspace's memory items, newest first or by similarity."""
    _validate_kind(kind)
    store = BrandMemoryStore(db)

    if q:
        hits = await store.search(
            business_id=ws.business_id,
            query=q,
            kinds=[kind] if kind else [],
            top_k=limit,
        )
        items = [_out(item, score=score) for item, score in hits]
        mode = "similar"
    else:
        rows = await store.list_items(
            business_id=ws.business_id, kind=kind, limit=limit
        )
        items = [_out(row) for row in rows]
        mode = "recent"

    total = await store.count_items(
        business_id=ws.business_id,
    )
    return MemoryItemList(
        items=items, total=total, limit=limit, kind=kind, query=q, mode=mode
    )


@router.post("/items", response_model=MemoryItemOut, status_code=201)
async def create_memory_item(
    payload: MemoryItemCreate,
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    db: SessionDep,
) -> MemoryItemOut:
    """Create one item of any of the seven kinds. This is the writer.

    The body-supplied ``campaign_id`` is ownership-checked before use
    (``assert_campaign_owned``), because it is the one field that could point a
    row at another tenant's campaign.
    """
    await require_role(principal, settings, ws, minimum=WRITE_MEMORY)
    await assert_campaign_owned(db, payload.campaign_id, ws)

    store = BrandMemoryStore(db)
    item = await store.write_item(
        business_id=ws.business_id,
        kind=payload.kind,
        text=payload.text,
        metadata=payload.metadata,
        campaign_id=payload.campaign_id,
    )
    await db.commit()
    await db.refresh(item)
    return _out(item)


@router.patch("/items/{item_id}", response_model=MemoryItemOut)
async def update_memory_item(
    item_id: str,
    payload: MemoryItemUpdate,
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    db: SessionDep,
) -> MemoryItemOut:
    """Edit the text and/or metadata of one item. Re-embeds on a text change.

    404 for another workspace's row, not 403.
    """
    await require_role(principal, settings, ws, minimum=WRITE_MEMORY)
    if not ({"text", "metadata"} & set(payload.model_fields_set)):
        raise ValidationFailed(
            "no fields to update; send 'text' and/or 'metadata'",
            detail={"accepted_fields": ["text", "metadata"]},
        )

    store = BrandMemoryStore(db)
    item = await store.update_item(
        business_id=ws.business_id,
        item_id=item_id,
        text=payload.text,
        metadata=payload.metadata if "metadata" in payload.model_fields_set else None,
    )
    if item is None:
        raise NotFoundError(f"memory item '{item_id}' not found")
    await db.commit()
    return _out(item)


@router.delete("/items/{item_id}", status_code=204)
async def delete_memory_item(
    item_id: str,
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    db: SessionDep,
) -> None:
    """Remove one memory item. 204, or 404 for another workspace's row."""
    await require_role(principal, settings, ws, minimum=WRITE_MEMORY)
    store = BrandMemoryStore(db)
    if not await store.delete_item(business_id=ws.business_id, item_id=item_id):
        raise NotFoundError(f"memory item '{item_id}' not found")
    await db.commit()


@router.post("/reindex", response_model=MemoryReindexResponse)
async def reindex_memory(
    payload: MemoryReindexRequest,
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    db: SessionDep,
) -> MemoryReindexResponse:
    """Recompute every embedding for this workspace. Owner only.

    Needed after ``EMBEDDING_DIM`` or the embedding provider changes: a row
    embedded under the old dimension either raises on insert or, worse, scores
    meaningfully against nothing. Restricted to ``owner`` because it is a bulk
    write proportional to the size of the table, not an authoring action.
    """
    await require_role(principal, settings, ws, minimum=REINDEX_MEMORY)
    _validate_kind(payload.kind)

    store = BrandMemoryStore(db)
    touched = await store.reindex_items(
        business_id=ws.business_id, kind=payload.kind
    )
    await db.commit()
    return MemoryReindexResponse(
        reindexed=len(touched),
        kind=payload.kind,
        item_ids=[item.id for item in touched],
    )


__all__ = ["router"]