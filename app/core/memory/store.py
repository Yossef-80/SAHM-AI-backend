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

from sqlalchemy import select
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
#: Documented in docs/API_CONTRACT.md and DECISIONS.md.
BRAND_CONTEXT_TOKEN_CAP = 900

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
    ) -> BrandMemoryItem:
        """Persist one memory item with its embedding."""
        if kind not in MEMORY_KINDS:
            raise ValueError(
                f"unknown memory kind '{kind}'; expected one of {MEMORY_KINDS}"
            )
        vector = await self.embeddings.embed(text)
        item = BrandMemoryItem(
            id=_new_id("mem"),
            business_id=business_id,
            kind=kind,
            text=text,
            embedding=vector,
            metadata_=metadata or {},
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
        self, *, business_id: str, payload: SaveLearningInput
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

        distance = BrandMemoryItem.embedding.cosine_distance(vector)
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

    async def _search_python(
        self, *, business_id: str, vector: list[float], kinds: list[str], top_k: int
    ) -> list[tuple[BrandMemoryItem, float]]:
        """Similarity search in Python (SQLite / no pgvector)."""
        stmt = BrandMemoryItem.__table__.select().where(
            BrandMemoryItem.business_id == business_id
        )
        if kinds:
            stmt = stmt.where(BrandMemoryItem.kind.in_(kinds))
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

    async def search(
        self, *, business_id: str, query: str, kinds: list[str], top_k: int
    ) -> list[tuple[BrandMemoryItem, float]]:
        vector = await self.embeddings.embed(query)
        if self.db.bind.dialect.name == "postgresql":
            try:
                return await self._search_pgvector(
                    business_id=business_id, vector=vector, kinds=kinds, top_k=top_k
                )
            except Exception as exc:  # pragma: no cover - fall back, never fail a run
                _logger.warning(
                    "pgvector_search_failed_falling_back",
                    extra={"error": str(exc)},
                )
        return await self._search_python(
            business_id=business_id, vector=vector, kinds=kinds, top_k=top_k
        )

    async def recent_learnings(
        self, *, business_id: str, limit: int
    ) -> list[BrandMemoryItem]:
        stmt = (
            BrandMemoryItem.__table__.select()
            .where(
                BrandMemoryItem.business_id == business_id,
                BrandMemoryItem.kind == "learning",
            )
            .order_by(BrandMemoryItem.created_at.desc())
            .limit(limit)
        )
        rows = (await self.db.execute(stmt)).mappings().all()
        return [
            BrandMemoryItem(
                id=r["id"],
                business_id=r["business_id"],
                kind=r["kind"],
                text=r["text"],
                campaign_id=r["campaign_id"],
                metadata_=r["metadata"] or {},
                created_at=r["created_at"],
            )
            for r in rows
        ]

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

    # ------------------------------------------------------ context assembly --

    async def build_context(
        self, *, business_id: str, scope: MemoryScope, task_name: str | None = None
    ) -> BrandContext:
        """Assemble a compact, token-bounded BrandContext.

        Order of precedence under the cap:
        1. stable profile fields (always included)
        2. top-k similar memory items for the current task
        3. most recent learnings

        Anything that does not fit is dropped and ``truncated`` is set.
        """
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
            business_id=business_id, query=query, kinds=kinds, top_k=scope.top_k
        )
        for item, score in similar:
            label = _label_for(item)
            candidate = _estimate_tokens(f"{label} {item.text}")
            if used + candidate > BRAND_CONTEXT_TOKEN_CAP:
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
                business_id=business_id, limit=scope.max_learnings
            ):
                candidate = _estimate_tokens(item.text)
                if used + candidate > BRAND_CONTEXT_TOKEN_CAP:
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
        business_id=ctx.business_id, scope=scope, task_name=payload.task_name
    )


def format_context_for_prompt(ctx_brand: BrandContext) -> str:
    """Render a BrandContext as the brand-context block of a prompt."""
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
            f"(brand context was truncated to stay under {BRAND_CONTEXT_TOKEN_CAP} tokens)"
        )
    return "\n".join(lines)


__all__ = [
    "BRAND_CONTEXT_TOKEN_CAP",
    "BrandMemoryStore",
    "MEMORY_KINDS",
    "format_context_for_prompt",
    "get_brand_context",
]
