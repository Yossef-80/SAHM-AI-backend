"""Brand-memory CRUD and inspection: ``/api/v1/memory/*``.

What changed when embeddings were removed
-----------------------------------------
``GET /memory/items?q=`` used to mean "cosine similarity of a 64-dimension
hashed bag-of-words vector". It now means BM25 + recency + kind/task affinity
over the same rows -- a deterministic ranking that needs no provider, no
reindex step and no vector column. ``mode`` reports which ordering was used and
each item carries a human-readable ``reason`` ("lex_bm25=0.81 rec=0.42 ...").

The former ``POST /memory/reindex`` endpoint is gone: nothing is embedded, so
there is nothing to rebuild. Its operational replacement is ``POST
/memory/prune`` -- archive expired rows and enforce the per-kind ceiling.

Isolation
---------
Every statement filters on ``business_id``, and an id-addressed route answers
**404**, never 403, for a row belonging to another workspace -- a 403 confirms
the row exists, which turns a guessed id into an enumeration oracle.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import (
    AuthDep,
    SessionDep,
    SettingsDep,
    WorkspaceDep,
    assert_campaign_owned,
    require_role,
)
from app.core.errors import NotFoundError, ValidationFailed
from app.core.memory.kinds import API_WRITABLE_KINDS, MEMORY_KINDS, is_known_kind
from app.core.memory.store import BrandMemoryStore
from app.db.models import BrandMemoryItem
from app.schemas.memory import (
    MemoryItemCreate,
    MemoryItemList,
    MemoryItemOut,
    MemoryItemUpdate,
    MemoryPruneRequest,
    MemoryPruneResponse,
)

router = APIRouter(prefix="/memory", tags=["memory"])

#: Writing brand memory is a write: owner or member, never viewer.
WRITE_MEMORY = "member"
#: Pruning is an operator action, not an authoring one.
PRUNE_MEMORY = "owner"

#: Hard ceiling on one page.
MAX_LIMIT = 200
DEFAULT_LIMIT = 50


def _out(
    item: BrandMemoryItem,
    score: float | None = None,
    reason: str | None = None,
) -> MemoryItemOut:
    return MemoryItemOut(
        id=item.id,
        business_id=item.business_id,
        kind=item.kind,
        text=item.text,
        metadata=dict(item.metadata_ or {}),
        campaign_id=item.campaign_id,
        created_at=item.created_at,
        updated_at=getattr(item, "updated_at", None),
        importance=float(getattr(item, "importance", 0.5) or 0.5),
        pinned=bool(getattr(item, "pinned", False)),
        expires_at=getattr(item, "expires_at", None),
        archived_at=getattr(item, "archived_at", None),
        score=score,
        reason=reason,
    )


def _validate_kind(kind: str | None) -> None:
    """Unknown kind is a 422, not a silent empty list.

    An empty result would make a typo in a filter look like "you have no
    memories", which is exactly the kind of quiet wrong answer an admin panel
    should not produce.
    """
    if kind is not None and not is_known_kind(kind):
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
        description="Relevance query: BM25 + recency + kind affinity, not substring matching.",
    ),
    include_archived: bool = Query(
        default=False, description="Include archived (expired/pruned) rows."
    ),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
) -> MemoryItemList:
    """List this workspace's memory items: newest first, or ranked by relevance."""
    _validate_kind(kind)
    store = BrandMemoryStore(db)

    if q:
        hits = await store.search_ranked(
            business_id=ws.business_id,
            query=q,
            kinds=[kind] if kind else None,
            top_k=limit,
            include_archived=include_archived,
        )
        items = [_out(hit.item, score=hit.score, reason=hit.reason()) for hit in hits]
        mode = "lexical"
    else:
        rows = await store.list_items(
            business_id=ws.business_id,
            kind=kind,
            limit=limit,
            include_archived=include_archived,
        )
        items = [_out(row) for row in rows]
        mode = "recent"

    total = await store.count_items(business_id=ws.business_id)
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
    """Create one item of a client-writable kind.

    Service-owned kinds (``brand_identity``, ``strategy_blueprint``, research
    artifacts, chat attachments) are rejected: their producers own their shape
    and their lifecycle.
    """
    await require_role(principal, settings, ws, minimum=WRITE_MEMORY)
    if payload.kind not in API_WRITABLE_KINDS:
        raise ValidationFailed(
            f"kind '{payload.kind}' is written by its own pipeline and cannot be created here",
            detail={"writable": list(API_WRITABLE_KINDS)},
        )
    await assert_campaign_owned(db, payload.campaign_id, ws)

    store = BrandMemoryStore(db)
    item = await store.write_item(
        business_id=ws.business_id,
        kind=payload.kind,
        text=payload.text,
        metadata=payload.metadata,
        campaign_id=payload.campaign_id,
        importance=payload.importance,
        pinned=payload.pinned,
        source="api",
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
    """Edit the text, metadata, importance or pin state of one item."""
    await require_role(principal, settings, ws, minimum=WRITE_MEMORY)
    if not ({"text", "metadata", "importance", "pinned"} & set(payload.model_fields_set)):
        raise ValidationFailed(
            "no fields to update",
            detail={"accepted_fields": ["text", "metadata", "importance", "pinned"]},
        )

    store = BrandMemoryStore(db)
    item = await store.update_item(
        business_id=ws.business_id,
        item_id=item_id,
        text=payload.text,
        metadata=payload.metadata if "metadata" in payload.model_fields_set else None,
        importance=payload.importance,
        pinned=payload.pinned,
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


@router.post("/items/{item_id}/archive", status_code=204)
async def archive_memory_item(
    item_id: str,
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    db: SessionDep,
) -> None:
    """Soft-delete: drop the row out of retrieval but keep it auditable."""
    await require_role(principal, settings, ws, minimum=WRITE_MEMORY)
    store = BrandMemoryStore(db)
    if not await store.archive_item(
        business_id=ws.business_id, item_id=item_id, reason="manual"
    ):
        raise NotFoundError(f"memory item '{item_id}' not found")
    await db.commit()


@router.post("/prune", response_model=MemoryPruneResponse)
async def prune_memory(
    payload: MemoryPruneRequest,
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    db: SessionDep,
) -> MemoryPruneResponse:
    """Archive expired rows and enforce the per-kind ceiling.

    ``dry_run`` defaults to true: an operator can see what would move before
    anything moves. Owner only.
    """
    await require_role(principal, settings, ws, minimum=PRUNE_MEMORY)
    _validate_kind(payload.kind)

    store = BrandMemoryStore(db)
    report = await store.prune(
        business_id=ws.business_id, kind=payload.kind, dry_run=payload.dry_run
    )
    if not payload.dry_run:
        await db.commit()
    return MemoryPruneResponse(
        dry_run=payload.dry_run,
        archived_by_kind={kind: count for kind, count in report.items() if count},
        total_archived=sum(report.values()),
    )


__all__ = ["router"]
