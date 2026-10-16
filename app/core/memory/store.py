"""Brand memory: read/write plus token-bounded context assembly.

Layer 8 of the brief. Two backends behind one interface:

* PostgreSQL + pgvector -- similarity search runs in the database with the
  ``<=>`` operator.
* SQLite (dev/test) -- vectors are stored as JSON and cosine similarity is
  computed in Python. Correct, just slower; documented in DECISIONS.md.

``get_brand_context`` is deliberately *compact*: it never dumps the whole
memory. It returns the stable profile fields, the top-k most similar memory
items for the current task, and the most recent learnings, all under a hard
token cap.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Float, cast, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.core.memory.embeddings import get_embedding_provider
from app.db.models import BrandMemoryItem, BrandProfile, Business, cosine
from app.schemas.tools import (
    BrandContext,
    BrandContextItem,
    GetBrandContextInput,
    Learning,
    MemoryScope,
    SaveLearningInput,
)

_logger = get_logger(__name__)

#: Token cap for the assembled BrandContext. Roughly 4 chars per token.
#:
#: Two caps, because the right budget depends on who is reading. A chat turn
#: shares its window with the conversation, the tool schemas and the routing
#: history; a one-shot creative request is a single deliverable and can afford a
#: real brand brief. Both used to be one number, so "write me ad copy that sounds
#: like us" was written from roughly four profile fields plus one or two memory
#: items -- the reason strong-tier output still read as generic.
BRAND_CONTEXT_TOKEN_CAP = 900
#: Cap used by creative generation (``/creatives/generate-text``), where the
#: brand context *is* the brief rather than background for a conversation.
CREATIVE_BRAND_CONTEXT_TOKEN_CAP = 2600

#: Rough chars-per-token used for the estimate. Deliberately conservative.
_CHARS_PER_TOKEN = 4

#: Memory kinds that are always eligible for retrieval.
MEMORY_KINDS = (
    "voice_example",
    "visual_style",
    "offer",
    "audience",
    "competitor_note",
    "learning",
    "reference_analysis",
)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _row_to_item(row: Any) -> BrandMemoryItem:
    """Rehydrate a mapping row into an ORM instance.

    The store reads through ``__table__`` (not the ORM) because the similarity
    queries need the raw column set; this keeps the three read paths in
    ``store.py`` consistent instead of making each caller rebuild the object.
    The embedding is deliberately not loaded -- it is large, unused by the API
    layer, and there is a dedicated ``search`` path for anything semantic.
    """
    return BrandMemoryItem(
        id=row["id"],
        business_id=row["business_id"],
        kind=row["kind"],
        text=row["text"],
        campaign_id=row["campaign_id"],
        metadata_=row["metadata"] or {},
        created_at=row["created_at"],
    )


# ------------------------------------------------------------------- store ----


class BrandMemoryStore:
    """All brand-memory reads and writes go through here."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self._embeddings = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embedding_provider()
        return self._embeddings

    # ------------------------------------------------------------- writes --

    async def write_item(
        self,
        *,
        business_id: str,
        kind: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        campaign_id: str | None = None,
        conversation_id: str | None = None,
        persona_id: str | None = None,
        agent: str | None = None,
        source: str = "brand_memory",
    ) -> BrandMemoryItem:
        """Persist one memory item with its embedding.

        Provenance travels with the item. ``campaign_id`` is a real column;
        ``conversation_id`` / ``persona_id`` / ``source`` go into ``metadata``
        because they are already indexed by the caller when they matter, and
        because a memory that cannot say where it came from is a memory the next
        agent cannot weigh -- research written by Nour, a learning from Salma's
        insight and a line of brand voice all used to look identical downstream.
        """
        if kind not in MEMORY_KINDS:
            raise ValueError(
                f"unknown memory kind '{kind}'; expected one of {MEMORY_KINDS}"
            )
        vector = await self.embeddings.embed(text)
        enriched = dict(metadata or {})
        enriched.setdefault("source", source)
        if conversation_id:
            enriched.setdefault("conversation_id", conversation_id)
        if persona_id or agent:
            enriched.setdefault("persona_id", persona_id or agent)
        if campaign_id:
            enriched.setdefault("campaign_id", campaign_id)
        item = BrandMemoryItem(
            id=_new_id("mem"),
            business_id=business_id,
            kind=kind,
            text=text,
            embedding=vector,
            metadata_=enriched,
            campaign_id=campaign_id,
        )
        self.db.add(item)
        await self.db.flush()
        _logger.info(
            "memory_written",
            extra={"kind": kind, "business_id": business_id, "chars": len(text)},
        )
        return item

    async def save_learning(
        self,
        *,
        business_id: str,
        payload: SaveLearningInput,
        conversation_id: str | None = None,
        persona_id: str | None = None,
    ) -> Learning:
        """Write a learning back to brand memory."""
        text_parts = [f"What worked: {payload.what_worked}"]
        if payload.what_didnt:
            text_parts.append(f"What did not: {payload.what_didnt}")
        text_parts.append(
            "Applies to future campaigns." if payload.applies_to_future
            else "Specific to this campaign."
        )
        item = await self.write_item(
            business_id=business_id,
            kind="learning",
            text="\n".join(text_parts),
            metadata={
                "what_worked": payload.what_worked,
                "what_didnt": payload.what_didnt,
                "applies_to_future": payload.applies_to_future,
                "tags": payload.tags,
            },
            campaign_id=payload.campaign_id,
            conversation_id=conversation_id,
            persona_id=persona_id,
            source="save_learning",
        )
        return Learning(
            id=item.id,
            business_id=business_id,
            campaign_id=payload.campaign_id,
            what_worked=payload.what_worked,
            what_didnt=payload.what_didnt,
            applies_to_future=payload.applies_to_future,
            tags=payload.tags,
            created_at=item.created_at,
            status="ok",
            source="brand_memory",
        )

    # -------------------------------------------------------------- reads --

    async def _search_pgvector(
        self, *, business_id: str, vector: list[float], kinds: list[str], top_k: int
    ) -> list[tuple[BrandMemoryItem, float]]:
        """Similarity search in Postgres using pgvector's cosine distance."""
        from pgvector.sqlalchemy import Vector

        # ``cast(..., Float)`` is load-bearing, not decoration. Without it, this
        # stack infers the JSON type for the ``<=>`` expression, and the driver
        # then hands the resulting *float* to the JSON deserializer, which raises
        # "'float' object is not subscriptable". That exception was swallowed by
        # the caller's fallback, so every semantic search silently degraded to
        # "newest 500 rows, scored in Python" -- correct-looking results in the
        # wrong order, with score 0.0 on the wire. Declaring the type once here
        # keeps the real ordering and keeps the fallback for what it is for.
        distance = cast(BrandMemoryItem.embedding.cosine_distance(vector), Float)
        stmt = (
            BrandMemoryItem.__table__.select()
            .add_columns(distance.label("distance"))
            .where(BrandMemoryItem.business_id == business_id)
            .order_by(distance)
            .limit(top_k)
        )
        if kinds:
            stmt = stmt.where(BrandMemoryItem.kind.in_(kinds))
        rows = (await self.db.execute(stmt)).mappings().all()
        return [
            (
                BrandMemoryItem(
                    id=r["id"],
                    business_id=r["business_id"],
                    kind=r["kind"],
                    text=r["text"],
                    campaign_id=r["campaign_id"],
                    metadata_=r["metadata"] or {},
                    created_at=r["created_at"],
                ),
                1.0 - float(r.get("distance", 0.0) or 0.0),
            )
            for r in rows
        ]

    #: How many rows the in-Python fallback is allowed to score. The fallback
    #: used to load *every* embedding of the business into memory and score it in
    #: Python, which turns a slow search into a memory spike as the workspace
    #: accumulates research. Newest-first is the useful half of that set.
    PYTHON_SEARCH_CANDIDATES = 500

    async def _search_python(
        self,
        *,
        business_id: str,
        vector: list[float],
        kinds: list[str],
        top_k: int,
        campaign_id: str | None = None,
    ) -> list[tuple[BrandMemoryItem, float]]:
        """Similarity search in Python (SQLite / no pgvector)."""
        stmt = BrandMemoryItem.__table__.select().where(
            BrandMemoryItem.business_id == business_id
        )
        if kinds:
            stmt = stmt.where(BrandMemoryItem.kind.in_(kinds))
        stmt = stmt.order_by(BrandMemoryItem.created_at.desc()).limit(
            max(self.PYTHON_SEARCH_CANDIDATES, top_k)
        )
        rows = (await self.db.execute(stmt)).mappings().all()
        scored: list[tuple[BrandMemoryItem, float]] = []
        for r in rows:
            stored = r.get("embedding")
            if not stored:
                continue
            if isinstance(stored, str):
                import json

                try:
                    stored = json.loads(stored)
                except json.JSONDecodeError:
                    continue
            scored.append(
                (
                    BrandMemoryItem(
                        id=r["id"],
                        business_id=r["business_id"],
                        kind=r["kind"],
                        text=r["text"],
                        campaign_id=r["campaign_id"],
                        metadata_=r["metadata"] or {},
                        created_at=r["created_at"],
                    ),
                    cosine([float(x) for x in stored], vector),
                )
            )
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    async def _search_backend(
        self, *, business_id: str, vector: list[float], kinds: list[str], top_k: int
    ) -> list[tuple[BrandMemoryItem, float]]:
        if self.db.bind.dialect.name == "postgresql":
            pairs: list[tuple[BrandMemoryItem, float]] | None = None
            try:
                pairs = await self._search_pgvector(
                    business_id=business_id, vector=vector, kinds=kinds, top_k=top_k
                )
            except Exception as exc:  # pragma: no cover - fall back, never fail a run
                _logger.warning(
                    "pgvector_search_failed_falling_back",
                    extra={"error": str(exc)},
                )
            if pairs is not None:
                # Cosine distance is *undefined* against a zero-magnitude vector
                # and pgvector returns NaN rather than erroring. NaN then orders
                # arbitrarily in SQL and serialises as a null score on the wire,
                # so a degenerate query (an embedding provider that returns zeros,
                # e.g. the offline mock, or a row stored before embeddings were
                # configured) is detected here and served by the Python scan
                # instead: same ordering rules, a finite score, no nulls.
                finite = [(item, score) for item, score in pairs if math.isfinite(score)]
                if finite:
                    return finite
                _logger.warning(
                    "pgvector_search_scores_not_finite_falling_back",
                    extra={
                        "rows": len(pairs),
                        "reason": "cosine distance is undefined for a zero vector",
                    },
                )
        return await self._search_python(
            business_id=business_id, vector=vector, kinds=kinds, top_k=top_k
        )

    async def search(
        self,
        *,
        business_id: str,
        query: str,
        kinds: list[str],
        top_k: int,
        campaign_id: str | None = None,
    ) -> list[tuple[BrandMemoryItem, float]]:
        """Similarity search, with campaign-scoped items pulled in beside the rest.

        A campaign's own research and learnings used to be indistinguishable from
        workspace-wide memory, so an agent working on campaign X could miss the
        brief that was written for campaign X. When a campaign is in scope the
        results are the merge of that campaign's top matches and the business-wide
        top matches -- campaign knowledge first on ties, since it is the more
        specific context -- each capped so the token budget in ``build_context``
        is unaffected.
        """
        vector = await self.embeddings.embed(query)
        if not campaign_id:
            return await self._search_backend(
                business_id=business_id, vector=vector, kinds=kinds, top_k=top_k
            )

        scoped = await self._search_backend(
            business_id=business_id,
            vector=vector,
            kinds=kinds,
            top_k=top_k,
        )
        campaign_hits = [pair for pair in scoped if getattr(pair[0], "campaign_id", None) == campaign_id]
        merged: list[tuple[BrandMemoryItem, float]] = []
        seen: set[str] = set()
        for item, score in campaign_hits + scoped:
            if item.id in seen:
                continue
            seen.add(item.id)
            merged.append((item, score))
            if len(merged) >= max(top_k, 1):
                break
        return merged

    async def recent_learnings(
        self, *, business_id: str, limit: int, campaign_id: str | None = None
    ) -> list[BrandMemoryItem]:
        """Newest learnings. With ``campaign_id``, that campaign's learnings win.

        The point of a learning is that the next campaign starts smarter, so
        workspace-wide learnings are still returned -- they are simply ranked
        behind the ones written for the campaign actually in play.
        """
        stmt = (
            BrandMemoryItem.__table__.select()
            .where(
                BrandMemoryItem.business_id == business_id,
                BrandMemoryItem.kind == "learning",
            )
            .order_by(BrandMemoryItem.created_at.desc())
            .limit(max(limit, 1) * (3 if campaign_id else 1))
        )
        rows = (await self.db.execute(stmt)).mappings().all()
        items = [_row_to_item(r) for r in rows]
        if campaign_id:
            items.sort(
                key=lambda item: (
                    0 if (getattr(item, "campaign_id", None) == campaign_id) else 1,
                    -(item.created_at.timestamp() if item.created_at else 0.0),
                )
            )
        return items[:max(limit, 1)]

    async def get_business(self, business_id: str) -> Business:
        business = await self.db.get(Business, business_id)
        if business is None:
            raise NotFoundError(f"business '{business_id}' not found")
        return business

    async def get_profile(self, business_id: str) -> BrandProfile | None:
        stmt = BrandProfile.__table__.select().where(
            BrandProfile.business_id == business_id
        )
        row = (await self.db.execute(stmt)).mappings().first()
        if row is None:
            return None
        return BrandProfile(
            id=row["id"],
            business_id=row["business_id"],
            brand_voice=row["brand_voice"],
            visual_style=row["visual_style"],
            target_audience=row["target_audience"],
            key_offers=row["key_offers"] or [],
            goals=row["goals"] or [],
            competitors=row["competitors"] or [],
        )

    async def count_items(self, business_id: str) -> int:
        from sqlalchemy import func

        stmt = (
            BrandMemoryItem.__table__.select()
            .with_only_columns(func.count())
            .where(BrandMemoryItem.business_id == business_id)
        )
        return int((await self.db.execute(stmt)).scalar() or 0)

    # ------------------------------------------------- CRUD (P2.2) --

    async def list_items(
        self, *, business_id: str, kind: str | None = None, limit: int = 50
    ) -> list[BrandMemoryItem]:
        """Newest first, optionally filtered to one kind.

        Plain newest-first rather than similarity-ordered: an editor screen wants
        "what did I write last week", which is what this returns. The
        similarity ordering lives on :meth:`search`, which is what the
        ``?q=`` box uses.
        """
        stmt = (
            BrandMemoryItem.__table__.select()
            .where(BrandMemoryItem.business_id == business_id)
            .order_by(BrandMemoryItem.created_at.desc())
            .limit(limit)
        )
        if kind:
            stmt = stmt.where(BrandMemoryItem.kind == kind)
        return [_row_to_item(row) for row in (await self.db.execute(stmt)).mappings().all()]

    async def get_item(
        self, *, business_id: str, item_id: str
    ) -> BrandMemoryItem | None:
        """One row, ownership-filtered. ``None`` for another workspace's row.

        An ORM ``select`` rather than the ``__table__`` form the read paths use:
        the CRUD methods modify and delete the result, which requires a real
        session-managed instance -- a hand-built transient one raises
        "Instance is not persisted" on delete.
        """
        return await self.db.scalar(
            select(BrandMemoryItem).where(
                BrandMemoryItem.id == item_id,
                BrandMemoryItem.business_id == business_id,
            )
        )

    async def update_item(
        self,
        *,
        business_id: str,
        item_id: str,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BrandMemoryItem | None:
        """Edit text and/or metadata. Re-embeds only when the text changed.

        Re-embedding on a metadata-only edit would be waste; re-embedding is not
        skipped when the text changes, because a stale vector returns the row
        under the wrong search results forever.
        """
        row = await self.get_item(business_id=business_id, item_id=item_id)
        if row is None:
            return None
        if text is not None and text != row.text:
            row.text = text
            row.embedding = await self.embeddings.embed(text)
        if metadata is not None:
            row.metadata_ = metadata
        await self.db.flush()
        _logger.info(
            "memory_updated",
            extra={"kind": row.kind, "business_id": business_id, "id": item_id},
        )
        return row

    async def delete_item(self, *, business_id: str, item_id: str) -> bool:
        """Remove one row. False when it does not exist in this workspace."""
        row = await self.get_item(business_id=business_id, item_id=item_id)
        if row is None:
            return False
        await self.db.delete(row)
        await self.db.flush()
        _logger.info(
            "memory_deleted",
            extra={"business_id": business_id, "id": item_id},
        )
        return True

    async def reindex_items(
        self, *, business_id: str, kind: str | None = None
    ) -> list[BrandMemoryItem]:
        """Recompute every embedding for this workspace (optionally one kind).

        Needed after ``EMBEDDING_DIM`` or the embedding provider changes: rows
        written under the old dimension produce either a dimension error or,
        worse, silently meaningless similarity scores.
        """
        stmt = select(BrandMemoryItem).where(
            BrandMemoryItem.business_id == business_id
        )
        if kind:
            stmt = stmt.where(BrandMemoryItem.kind == kind)
        items = list((await self.db.execute(stmt)).scalars().all())
        for item in items:
            item.embedding = await self.embeddings.embed(item.text)
        touched = items
        await self.db.flush()
        _logger.info(
            "memory_reindexed",
            extra={
                "business_id": business_id,
                "kind": kind or "all",
                "count": len(touched),
            },
        )
        return touched

    # ------------------------------------------------------ context assembly --

    async def build_context(
        self,
        *,
        business_id: str,
        scope: MemoryScope,
        task_name: str | None = None,
        campaign_id: str | None = None,
        token_cap: int | None = None,
    ) -> BrandContext:
        """Assemble a compact, token-bounded BrandContext.

        Order of precedence under the cap:
        1. stable profile fields (always included)
        2. top-k similar memory items for the current task
        3. most recent learnings

        Anything that does not fit is dropped and ``truncated`` is set.

        ``token_cap`` overrides the default budget; creative generation passes
        ``CREATIVE_BRAND_CONTEXT_TOKEN_CAP`` because a single deliverable has no
        conversation competing for the window.
        """
        cap = token_cap or BRAND_CONTEXT_TOKEN_CAP
        business = await self.get_business(business_id)
        profile = await self.get_profile(business_id)

        query = scope.query or task_name or business.name
        kinds = scope.kinds or [k for k in MEMORY_KINDS if k != "learning"]

        ctx = BrandContext(
            business_id=business_id,
            business_name=business.name,
            industry=business.industry,
            country=business.country,
            language=business.language,
            currency=business.currency,
            monthly_budget=business.monthly_budget,
            brand_voice=profile.brand_voice if profile else None,
            visual_style=profile.visual_style if profile else None,
            target_audience=profile.target_audience if profile else None,
            key_offers=list(profile.key_offers) if profile else [],
            goals=list(profile.goals) if profile else [],
            competitors=list(profile.competitors) if profile else [],
            status="ok",
            source="brand_memory",
        )

        used = _estimate_tokens(
            " ".join(
                filter(
                    None,
                    [
                        business.name,
                        business.industry,
                        ctx.brand_voice,
                        ctx.visual_style,
                        ctx.target_audience,
                        " ".join(ctx.key_offers),
                        " ".join(ctx.goals),
                        " ".join(ctx.competitors),
                    ],
                )
            )
        )

        # 2. similar items
        similar = await self.search(
            business_id=business_id,
            query=query,
            kinds=kinds,
            top_k=scope.top_k,
            campaign_id=campaign_id,
        )
        for item, score in similar:
            label = _label_for(item)
            candidate = _estimate_tokens(f"{label} {item.text}")
            if used + candidate > cap:
                ctx.truncated = True
                break
            used += candidate
            ctx.items.append(
                BrandContextItem(
                    kind=item.kind,
                    label=label,
                    detail=item.text,
                    similarity=round(score, 4),
                    memory_id=item.id,
                )
            )

        # 3. recent learnings
        if scope.include_learnings and scope.max_learnings > 0:
            for item in await self.recent_learnings(
                business_id=business_id, limit=scope.max_learnings, campaign_id=campaign_id
            ):
                candidate = _estimate_tokens(item.text)
                if used + candidate > cap:
                    ctx.truncated = True
                    break
                used += candidate
                ctx.learnings.append(
                    BrandContextItem(
                        kind="learning",
                        label="Recent learning",
                        detail=item.text,
                        memory_id=item.id,
                    )
                )

        ctx.token_estimate = used
        return ctx


def _label_for(item: BrandMemoryItem) -> str:
    labels = {
        "voice_example": "Brand voice example",
        "visual_style": "Visual style",
        "offer": "Offer",
        "audience": "Audience",
        "competitor_note": "Competitor note",
        "learning": "Learning",
        "reference_analysis": "Reference analysis",
    }
    return labels.get(item.kind, item.kind.replace("_", " ").title())


async def get_brand_context(payload: GetBrandContextInput, ctx: Any) -> BrandContext:
    """Tool handler: get_brand_context."""
    from app.core.tools.base import ToolContext

    assert isinstance(ctx, ToolContext)
    store = BrandMemoryStore(ctx.db)
    scope = payload.scope
    if payload.campaign_id and not scope.query:
        scope = scope.model_copy(update={"query": f"campaign {payload.campaign_id}"})
    return await store.build_context(
        business_id=ctx.business_id,
        scope=scope,
        task_name=payload.task_name,
        campaign_id=payload.campaign_id or getattr(ctx, "campaign_id", None),
    )


def format_context_for_prompt(
    ctx_brand: BrandContext, *, token_cap: int | None = None
) -> str:
    """Render a BrandContext as the brand-context block of a prompt.

    ``token_cap`` must be the cap the context was assembled with, so the note
    tells the model how much of the brand memory it is actually missing.
    """
    lines: list[str] = ["BRAND CONTEXT (use this; do not invent anything beyond it)"]
    lines.append(f"Business: {ctx_brand.business_name}")
    if ctx_brand.industry:
        lines.append(f"Industry: {ctx_brand.industry}")
    if ctx_brand.country:
        lines.append(f"Market: {ctx_brand.country}")
    if ctx_brand.currency:
        lines.append(f"Currency: {ctx_brand.currency}")
    if ctx_brand.monthly_budget is not None:
        lines.append(f"Monthly budget: {ctx_brand.monthly_budget}")
    if ctx_brand.brand_voice:
        lines.append(f"Brand voice: {ctx_brand.brand_voice}")
    if ctx_brand.visual_style:
        lines.append(f"Visual style: {ctx_brand.visual_style}")
    if ctx_brand.target_audience:
        lines.append(f"Target audience: {ctx_brand.target_audience}")
    if ctx_brand.key_offers:
        lines.append("Key offers: " + "; ".join(ctx_brand.key_offers))
    if ctx_brand.goals:
        lines.append("Goals: " + "; ".join(ctx_brand.goals))
    if ctx_brand.competitors:
        lines.append("Competitors: " + "; ".join(ctx_brand.competitors))
    if ctx_brand.items:
        lines.append("")
        lines.append("Relevant brand memory:")
        for item in ctx_brand.items:
            sim = f" (similarity {item.similarity:.2f})" if item.similarity is not None else ""
            lines.append(f"- [{item.kind}] {item.label}: {item.detail}{sim}")
    if ctx_brand.learnings:
        lines.append("")
        lines.append("Recent learnings:")
        for item in ctx_brand.learnings:
            lines.append(f"- {item.detail}")
    if ctx_brand.truncated:
        lines.append("")
        lines.append(
            "(brand context was truncated to stay under "
            f"{(token_cap or BRAND_CONTEXT_TOKEN_CAP)} tokens)"
        )
    return "\n".join(lines)


__all__ = [
    "BRAND_CONTEXT_TOKEN_CAP",
    "CREATIVE_BRAND_CONTEXT_TOKEN_CAP",
    "BrandMemoryStore",
    "MEMORY_KINDS",
    "format_context_for_prompt",
    "get_brand_context",
]
