"""Brand memory: layered writes, lexical retrieval, token-bounded assembly.

Design in one paragraph
----------------------
Brand memory is **not** a vector database. It is a small, curated set of short
facts (voice, offers, audience, identity), one active plan, a handful of
episodic notes (research, competitor notes, reference analyses) and durable
learnings. At that size the honest retrieval problem is "which of these few
hundred short rows is relevant to *this* task, and fresh enough to trust" --
which is a ranking problem, not a similarity-search problem.

The layers, and where each is stored:

===========  ==========================================  ===========================
layer        storage                                     written by
===========  ==========================================  ===========================
working      ``messages`` (raw turns, token-capped)       the assistant bridge
session      ``conversations.summary`` + watermark       ``manage_session_memory``
identity     ``businesses`` + ``brand_profiles`` (+       Settings, brand identity
             ``brand_memory_items`` kinds of layer
             ``identity``)
plan         ``brand_memory_items`` (``strategy_blueprint``) background LTM
episodic     ``brand_memory_items`` (research, notes,     research tools, Layla
             references)
learning     ``brand_memory_items`` (``learning``)         ``save_learning``
===========  ==========================================  ===========================

Every read here is deterministic and explainable: candidates are bounded by a
recency window, ranked by BM25 + recency decay + kind/task affinity + importance
(+ pinned/campaign boosts), then packed into a hard token budget in layer order.
No embedding provider, no reindex step, no model call.

The public surface is deliberately the same as the old vector store
(``write_item`` / ``search`` / ``build_context`` / CRUD) so callers did not have
to change when the retrieval engine did.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Iterable, Sequence

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.core.memory.kinds import (
    EPISODIC_KINDS,
    IDENTITY_KINDS,
    LEARNING_KINDS,
    MEMORY_KINDS,
    PLAN_KINDS,
    label_for,
    max_tokens_for,
    spec_for,
)
from app.core.memory.lifecycle import (
    dedupe_digest,
    ensure_aware,
    expiry_for,
    is_expired,
    prune_rank,
    utcnow,
)
from app.core.memory.policy import MemoryPolicy
from app.core.memory.ranking import RankedMemory, rank_items
from app.core.memory.tokens import count_tokens, truncate_to_tokens
from app.db.models import BrandMemoryItem, BrandProfile, Business, Profile
from app.schemas.tools import (
    BrandContext,
    BrandContextItem,
    GetBrandContextInput,
    Learning,
    MemoryScope,
    SaveLearningInput,
)

_logger = get_logger(__name__)

#: Default token cap for the assembled BrandContext. Kept as a module constant
#: because the API contract and docs quote it; :class:`MemoryPolicy` can override
#: it through ``MEMORY_CONTEXT_TOKEN_CAP``.
BRAND_CONTEXT_TOKEN_CAP = 900


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class _Budget:
    """Tiny token budget used by :meth:`BrandMemoryStore.build_context`."""

    def __init__(self, total: int) -> None:
        self.total = max(0, total)
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used)

    def try_add(self, tokens: int) -> bool:
        if tokens > self.remaining:
            return False
        self.used += tokens
        return True


class BrandMemoryStore:
    """All brand-memory reads and writes go through here."""

    def __init__(self, db: AsyncSession, policy: MemoryPolicy | None = None) -> None:
        self.db = db
        self.policy = policy or MemoryPolicy.from_settings()

    # ------------------------------------------------------------------ writes --

    async def write_item(
        self,
        *,
        business_id: str,
        kind: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        campaign_id: str | None = None,
        importance: float | None = None,
        pinned: bool = False,
        source: str = "manual",
        dedupe: bool | None = None,
    ) -> BrandMemoryItem:
        """Persist one memory item, de-duplicated and TTL-stamped.

        ``kind`` must exist in the registry: accepting unknown kinds silently is
        how six kinds ended up written-but-never-retrievable.
        """
        spec = spec_for(kind)
        clean = truncate_to_tokens(
            (text or "").strip(),
            max_tokens_for(kind, self.policy.item_max_tokens),
        )
        if not clean:
            raise ValueError("memory text must not be empty")

        now = utcnow()
        use_dedupe = self.policy.dedupe_enabled if dedupe is None else dedupe
        key = (
            dedupe_digest(
                kind=kind,
                business_id=business_id,
                text=clean,
                campaign_id=campaign_id,
            )
            if use_dedupe
            else None
        )

        if key is not None:
            existing = await self.db.scalar(
                select(BrandMemoryItem).where(
                    BrandMemoryItem.business_id == business_id,
                    BrandMemoryItem.dedupe_digest == key,
                )
            )
            if existing is not None:
                existing.text = clean
                existing.metadata_ = {**(existing.metadata_ or {}), **(metadata or {})}
                existing.updated_at = now
                existing.archived_at = None
                existing.expires_at = expiry_for(
                    kind, now=now, default_ttl_days=self.policy.default_ttl_days
                )
                existing.importance = max(
                    float(existing.importance or 0.0),
                    float(spec.importance if importance is None else importance),
                )
                existing.pinned = bool(existing.pinned or pinned)
                await self.db.flush()
                _logger.info(
                    "memory_deduped",
                    extra={"kind": kind, "business_id": business_id, "id": existing.id},
                )
                return existing

        item = BrandMemoryItem(
            id=_new_id("mem"),
            business_id=business_id,
            kind=kind,
            text=clean,
            metadata_=metadata or {},
            campaign_id=campaign_id,
            importance=float(spec.importance if importance is None else max(0.0, min(1.0, importance))),
            pinned=bool(pinned),
            source=str(source or "manual")[:40],
            dedupe_digest=key,
            created_at=now,
            updated_at=now,
            expires_at=expiry_for(
                kind, now=now, default_ttl_days=self.policy.default_ttl_days
            ),
        )
        self.db.add(item)
        await self.db.flush()

        if self.policy.sweep_on_write:
            await self._enforce_limits(business_id, kind=kind)

        _logger.info(
            "memory_written",
            extra={"kind": kind, "business_id": business_id, "chars": len(clean)},
        )
        return item

    async def upsert_singleton(
        self,
        *,
        business_id: str,
        kind: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        source: str = "",
        importance: float | None = None,
    ) -> BrandMemoryItem:
        """Keep exactly ONE active row for ``(business, kind)``.

        Some kinds are state, not events: the brand identity, the active plan.
        Appending a new row per update makes the newest one compete with every
        previous one in retrieval (and the older ones stay "fresh" for months
        because recency decays slowly). Those kinds are overwritten in place.
        """
        spec = spec_for(kind)
        clean = truncate_to_tokens(
            (text or "").strip(),
            max_tokens_for(kind, self.policy.item_max_tokens),
        )
        now = utcnow()
        row = await self.db.scalar(
            select(BrandMemoryItem)
            .where(
                BrandMemoryItem.business_id == business_id,
                BrandMemoryItem.kind == kind,
                BrandMemoryItem.archived_at.is_(None),
            )
            .order_by(BrandMemoryItem.created_at.desc())
            .limit(1)
        )
        if row is None:
            return await self.write_item(
                business_id=business_id,
                kind=kind,
                text=clean,
                metadata=metadata,
                source=source or "singleton",
                importance=importance,
                dedupe=False,
            )

        row.text = clean
        row.metadata_ = metadata or {}
        row.updated_at = now
        row.importance = float(
            spec.importance if importance is None else max(0.0, min(1.0, importance))
        )
        row.expires_at = expiry_for(
            kind, now=now, default_ttl_days=self.policy.default_ttl_days
        )
        row.dedupe_digest = (
            dedupe_digest(kind=kind, business_id=business_id, text=clean)
            if self.policy.dedupe_enabled
            else None
        )
        row.source = str(source or row.source or "singleton")[:40]
        await self.db.flush()
        _logger.info(
            "memory_singleton_updated",
            extra={"kind": kind, "business_id": business_id, "id": row.id},
        )
        return row

    async def save_learning(
        self, *, business_id: str, payload: SaveLearningInput
    ) -> Learning:
        """Write a learning back to brand memory."""
        text_parts = [f"What worked: {payload.what_worked}"]
        if payload.what_didnt:
            text_parts.append(f"What did not: {payload.what_didnt}")
        text_parts.append(
            "Applies to future campaigns."
            if payload.applies_to_future
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

    async def update_item(
        self,
        *,
        business_id: str,
        item_id: str,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        pinned: bool | None = None,
    ) -> BrandMemoryItem | None:
        """Edit an item. Recomputes the dedupe key and the TTL when text changes."""
        row = await self.get_item(business_id=business_id, item_id=item_id)
        if row is None:
            return None
        now = utcnow()
        if text is not None and text != row.text:
            row.text = truncate_to_tokens(
                text.strip(),
                max_tokens_for(row.kind, self.policy.item_max_tokens),
            )
            row.dedupe_digest = (
                dedupe_digest(
                    kind=row.kind,
                    business_id=business_id,
                    text=row.text,
                    campaign_id=row.campaign_id,
                )
                if self.policy.dedupe_enabled
                else None
            )
            row.expires_at = expiry_for(
                row.kind, now=now, default_ttl_days=self.policy.default_ttl_days
            )
        if metadata is not None:
            row.metadata_ = metadata
        if importance is not None:
            row.importance = max(0.0, min(1.0, float(importance)))
        if pinned is not None:
            row.pinned = bool(pinned)
        row.updated_at = now
        await self.db.flush()
        _logger.info(
            "memory_updated",
            extra={"kind": row.kind, "business_id": business_id, "id": item_id},
        )
        return row

    async def delete_item(self, *, business_id: str, item_id: str) -> bool:
        """Hard-delete one row. False when it does not exist in this workspace."""
        row = await self.get_item(business_id=business_id, item_id=item_id)
        if row is None:
            return False
        await self.db.delete(row)
        await self.db.flush()
        _logger.info("memory_deleted", extra={"business_id": business_id, "id": item_id})
        return True

    async def archive_item(
        self, *, business_id: str, item_id: str, reason: str = "manual"
    ) -> bool:
        """Soft-delete: out of retrieval, still auditable and reversible."""
        row = await self.get_item(business_id=business_id, item_id=item_id)
        if row is None:
            return False
        row.archived_at = utcnow()
        row.metadata_ = {**(row.metadata_ or {}), "archived_reason": reason}
        await self.db.flush()
        return True

    async def touch_items(
        self, item_ids: Sequence[str], *, business_id: str | None = None
    ) -> None:
        """Record that these items were injected into a prompt.

        Usage is the only signal that distinguishes a genuinely useful note from
        one that has sat untouched for a year; pruning reads it. When
        ``business_id`` is given the update is tenant-scoped, so a stray id can
        never bump a counter in another workspace.
        """
        ids = [item_id for item_id in item_ids if item_id]
        if not ids:
            return
        stmt = update(BrandMemoryItem).where(BrandMemoryItem.id.in_(ids))
        if business_id is not None:
            stmt = stmt.where(BrandMemoryItem.business_id == business_id)
        await self.db.execute(
            stmt.values(use_count=BrandMemoryItem.use_count + 1, last_used_at=utcnow())
        )

    # --------------------------------------------------------------- lifecycle --

    def _active_filter(self, business_id: str):
        now = utcnow()
        return (
            BrandMemoryItem.business_id == business_id,
            BrandMemoryItem.archived_at.is_(None),
            ((BrandMemoryItem.expires_at.is_(None)) | (BrandMemoryItem.expires_at > now)),
        )

    async def sweep_expired(self, business_id: str, *, kind: str | None = None) -> int:
        """Archive expired rows. Returns how many moved.

        Deliberately *not* built on ``_active_filter``: that filter already
        excludes expired rows, so filtering by it would make the sweep a no-op
        forever (the rows it is meant to catch would never be selected).
        """
        stmt = select(BrandMemoryItem).where(
            BrandMemoryItem.business_id == business_id,
            BrandMemoryItem.archived_at.is_(None),
            BrandMemoryItem.expires_at.is_not(None),
        )
        if kind:
            stmt = stmt.where(BrandMemoryItem.kind == kind)
        rows = list((await self.db.execute(stmt)).scalars().all())
        now = utcnow()
        moved = 0
        for row in rows:
            if is_expired(row, now=now):
                row.archived_at = now
                row.metadata_ = {**(row.metadata_ or {}), "archived_reason": "expired"}
                moved += 1
        if moved:
            await self.db.flush()
            _logger.info(
                "memory_swept_expired",
                extra={"business_id": business_id, "kind": kind or "all", "count": moved},
            )
        return moved

    async def prune_over_cap(self, business_id: str, *, kind: str) -> int:
        """Archive the worst rows above ``MEMORY_MAX_ITEMS_PER_KIND``."""
        rows = list(
            (
                await self.db.execute(
                    select(BrandMemoryItem)
                    .where(
                        *self._active_filter(business_id),
                        BrandMemoryItem.kind == kind,
                    )
                    .order_by(BrandMemoryItem.created_at.desc())
                )
            )
            .scalars()
            .all()
        )
        overflow = len(rows) - self.policy.max_items_per_kind
        if overflow <= 0:
            return 0
        candidates = sorted(
            [row for row in rows if not row.pinned],
            key=prune_rank,
        )[:overflow]
        now = utcnow()
        for row in candidates:
            row.archived_at = now
            row.metadata_ = {**(row.metadata_ or {}), "archived_reason": "over_cap"}
        await self.db.flush()
        _logger.info(
            "memory_pruned_over_cap",
            extra={"business_id": business_id, "kind": kind, "archived": len(candidates)},
        )
        return len(candidates)

    async def prune(
        self, *, business_id: str, kind: str | None = None, dry_run: bool = True
    ) -> dict[str, int]:
        """Expired-row sweep + per-kind ceiling. ``dry_run`` counts only."""
        kinds = [kind] if kind else list(MEMORY_KINDS)
        report: dict[str, int] = {}
        for name in kinds:
            rows = list(
                (
                    await self.db.execute(
                        select(BrandMemoryItem).where(
                            BrandMemoryItem.business_id == business_id,
                            BrandMemoryItem.kind == name,
                            BrandMemoryItem.archived_at.is_(None),
                        )
                    )
                )
                .scalars()
                .all()
            )
            expired = sum(1 for row in rows if is_expired(row))
            overflow = max(0, len(rows) - expired - self.policy.max_items_per_kind)
            report[name] = expired + overflow
            if dry_run:
                continue
            await self.sweep_expired(business_id, kind=name)
            await self.prune_over_cap(business_id, kind=name)
        return report

    async def _enforce_limits(self, business_id: str, *, kind: str) -> None:
        await self.sweep_expired(business_id, kind=kind)
        await self.prune_over_cap(business_id, kind=kind)

    # ------------------------------------------------------------------- reads --

    async def search_ranked(
        self,
        *,
        business_id: str,
        query: str = "",
        kinds: Sequence[str] | None = None,
        top_k: int = 6,
        task_name: str | None = None,
        campaign_id: str | None = None,
        include_archived: bool = False,
    ) -> list[RankedMemory]:
        """Rank active rows by lexical + recency + affinity + importance."""
        stmt = select(BrandMemoryItem).where(
            BrandMemoryItem.business_id == business_id
        )
        if not include_archived:
            stmt = stmt.where(*self._active_filter(business_id)[1:])
        if kinds:
            stmt = stmt.where(BrandMemoryItem.kind.in_(list(kinds)))
        # Recency-bounded candidate window: ranking a 50 000-row table in Python
        # on every chat turn would be wasteful, and a note older than the window
        # is better served by the recency term than by a full scan.
        stmt = stmt.order_by(BrandMemoryItem.created_at.desc()).limit(
            self.policy.candidate_window
        )
        rows = list((await self.db.execute(stmt)).scalars().all())
        ranked = rank_items(
            rows,
            query=query,
            task_name=task_name,
            policy=self.policy,
            campaign_id=campaign_id,
        )
        return ranked[: max(0, top_k)]

    async def search(
        self,
        *,
        business_id: str,
        query: str,
        kinds: Sequence[str] | None = None,
        top_k: int = 6,
        task_name: str | None = None,
        campaign_id: str | None = None,
    ) -> list[tuple[BrandMemoryItem, float]]:
        """Backwards-compatible shape: ``[(item, score)]`` best-first."""
        return [
            (hit.item, hit.score)
            for hit in await self.search_ranked(
                business_id=business_id,
                query=query,
                kinds=kinds,
                top_k=top_k,
                task_name=task_name,
                campaign_id=campaign_id,
            )
        ]

    async def recent_learnings(
        self, *, business_id: str, limit: int
    ) -> list[BrandMemoryItem]:
        stmt = (
            select(BrandMemoryItem)
            .where(
                *self._active_filter(business_id),
                BrandMemoryItem.kind.in_(list(LEARNING_KINDS)),
            )
            .order_by(BrandMemoryItem.created_at.desc())
            .limit(limit)
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def get_business(self, business_id: str) -> Business:
        business = await self.db.get(Business, business_id)
        if business is None:
            raise NotFoundError(f"business '{business_id}' not found")
        return business

    async def get_profile(self, business_id: str) -> BrandProfile | None:
        return await self.db.scalar(
            select(BrandProfile).where(BrandProfile.business_id == business_id).limit(1)
        )

    async def get_distilled_profile(self, business_id: str) -> str:
        """The capped brand-facts paragraph maintained by the background LTM."""
        value = await self.db.scalar(
            select(Profile.user_context_summary)
            .where(Profile.business_id == business_id)
            .limit(1)
        )
        return str(value or "").strip()

    async def count_items(self, business_id: str, *, kind: str | None = None) -> int:
        """Count active rows, optionally for one kind."""
        stmt = (
            select(func.count())
            .select_from(BrandMemoryItem)
            .where(*self._active_filter(business_id))
        )
        if kind:
            stmt = stmt.where(BrandMemoryItem.kind == kind)
        return int((await self.db.execute(stmt)).scalar() or 0)

    async def list_items(
        self,
        *,
        business_id: str,
        kind: str | None = None,
        limit: int = 50,
        include_archived: bool = False,
    ) -> list[BrandMemoryItem]:
        """Newest first, optionally filtered to one kind.

        Ranking lives on :meth:`search_ranked`; an editor screen wants
        "what did I write last week", which is exactly newest-first.
        """
        stmt = select(BrandMemoryItem).where(
            BrandMemoryItem.business_id == business_id
        )
        if not include_archived:
            stmt = stmt.where(*self._active_filter(business_id)[1:])
        if kind:
            stmt = stmt.where(BrandMemoryItem.kind == kind)
        stmt = stmt.order_by(BrandMemoryItem.created_at.desc()).limit(limit)
        return list((await self.db.execute(stmt)).scalars().all())

    async def get_item(
        self, *, business_id: str, item_id: str
    ) -> BrandMemoryItem | None:
        """One row, ownership-filtered. ``None`` for another workspace's row."""
        return await self.db.scalar(
            select(BrandMemoryItem).where(
                BrandMemoryItem.id == item_id,
                BrandMemoryItem.business_id == business_id,
            )
        )

    # ------------------------------------------------------- context assembly --

    async def build_context(
        self,
        *,
        business_id: str,
        scope: MemoryScope,
        task_name: str | None = None,
        campaign_id: str | None = None,
    ) -> BrandContext:
        """Assemble the token-bounded, layer-ordered brand context.

        Order (each section is skipped when its budget is exhausted):

        1. standing facts -- Business + BrandProfile + the distilled
           ``Profile.user_context_summary`` paragraph;
        2. identity memories -- ``brand_identity``, voice, visual style, offers,
           audience (always preferred over episodic notes);
        3. the active plan -- the newest ``strategy_blueprint``;
        4. episodic evidence -- ranked research/competitor/reference notes;
        5. recent learnings.

        ``truncated`` is set when anything was dropped, so the caller can say so
        instead of implying the context is complete.
        """
        business = await self.get_business(business_id)
        profile = await self.get_profile(business_id)
        distilled = await self.get_distilled_profile(business_id)

        cap = scope.token_budget or self.policy.context_token_cap or BRAND_CONTEXT_TOKEN_CAP
        budget = _Budget(cap)

        ctx = BrandContext(
            business_id=business_id,
            business_name=business.name,
            industry=business.industry,
            country=business.country,
            language=business.language or "en",
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

        # -- 1. standing facts -------------------------------------------------
        facts_text = " ".join(
            filter(
                None,
                [
                    business.name,
                    business.industry,
                    business.country,
                    ctx.brand_voice,
                    ctx.visual_style,
                    ctx.target_audience,
                    " ".join(ctx.key_offers),
                    " ".join(ctx.goals),
                    " ".join(ctx.competitors),
                ],
            )
        )
        budget.try_add(count_tokens(facts_text))

        if distilled:
            distilled = truncate_to_tokens(
                distilled, min(self.policy.profile_max_tokens, budget.remaining)
            )
            if distilled:
                budget.try_add(count_tokens(distilled))
                ctx.items.append(
                    BrandContextItem(
                        kind="profile_summary",
                        label="Distilled brand facts",
                        detail=distilled,
                    )
                )
            else:
                ctx.truncated = True

        query = scope.query.strip()
        campaign = campaign_id or scope.campaign_id
        included_ids: list[str] = []

        # -- 2. identity -------------------------------------------------------
        identity_budget = min(budget.remaining, self.policy.identity_budget())
        if identity_budget > 0:
            identity_hits = await self.search_ranked(
                business_id=business_id,
                # No query filter here: identity facts belong in every prompt.
                # The query only affects which *episodic* evidence is relevant.
                query="",
                kinds=IDENTITY_KINDS,
                top_k=scope.top_k,
                task_name=task_name,
                campaign_id=campaign,
            )
            identity_spent = 0
            for hit in identity_hits:
                item = hit.item
                label = label_for(item.kind)
                tokens = count_tokens(f"{label} {item.text}")
                if identity_spent + tokens > identity_budget:
                    ctx.truncated = True
                    break
                identity_spent += tokens
                budget.try_add(tokens)
                ctx.items.append(
                    BrandContextItem(
                        kind=item.kind,
                        label=label,
                        detail=item.text,
                        similarity=hit.score,
                        memory_id=item.id,
                        reason=hit.reason(),
                    )
                )
                included_ids.append(item.id)

        # -- 3. active plan ----------------------------------------------------
        plan_budget = min(budget.remaining, self.policy.plan_budget())
        if plan_budget > 0:
            plan_hits = await self.search_ranked(
                business_id=business_id,
                kinds=PLAN_KINDS,
                top_k=1,
                task_name=task_name,
                campaign_id=campaign,
                query="",
            )
            for hit in plan_hits:
                item = hit.item
                tokens = count_tokens(item.text)
                if not budget.try_add(tokens) or tokens > plan_budget:
                    ctx.truncated = True
                    break
                ctx.items.append(
                    BrandContextItem(
                        kind=item.kind,
                        label=label_for(item.kind),
                        detail=item.text,
                        similarity=hit.score,
                        memory_id=item.id,
                        reason=hit.reason(),
                    )
                )
                included_ids.append(item.id)

        # -- 4. episodic evidence ---------------------------------------------
        # Default to the episodic layer. Passing ``None`` (any kind) let the
        # identity rows and the active plan be retrieved a *second* time here,
        # so a plan could be injected twice in one prompt and pay for itself
        # twice. Only an explicit ``scope.kinds`` may widen this.
        requested = [kind for kind in (scope.kinds or []) if kind]
        episodic_kinds = requested or EPISODIC_KINDS
        if budget.remaining > 0:
            episodic = await self.search_ranked(
                business_id=business_id,
                query=query,
                kinds=episodic_kinds,
                top_k=max(scope.top_k, self.policy.context_top_k),
                task_name=task_name,
                campaign_id=campaign,
            )
            for hit in episodic:
                item = hit.item
                if item.kind in LEARNING_KINDS:
                    continue  # handled below, so learnings keep their own section
                if item.id in included_ids:
                    continue  # already injected by an earlier section
                label = label_for(item.kind)
                tokens = count_tokens(f"{label} {item.text}")
                if not budget.try_add(tokens):
                    ctx.truncated = True
                    break
                ctx.items.append(
                    BrandContextItem(
                        kind=item.kind,
                        label=label,
                        detail=item.text,
                        similarity=hit.score,
                        memory_id=item.id,
                        reason=hit.reason(),
                    )
                )
                included_ids.append(item.id)

        # -- 5. learnings ------------------------------------------------------
        if scope.include_learnings and scope.max_learnings > 0 and budget.remaining > 0:
            learnings = await self.search_ranked(
                business_id=business_id,
                query=query,
                kinds=LEARNING_KINDS,
                top_k=min(scope.max_learnings, self.policy.recent_learnings),
                task_name=task_name,
                campaign_id=campaign,
            )
            for hit in learnings:
                item = hit.item
                tokens = count_tokens(item.text)
                if not budget.try_add(tokens):
                    ctx.truncated = True
                    break
                ctx.learnings.append(
                    BrandContextItem(
                        kind="learning",
                        label="Learning",
                        detail=item.text,
                        similarity=hit.score,
                        memory_id=item.id,
                        reason=hit.reason(),
                    )
                )
                included_ids.append(item.id)

        ctx.token_estimate = budget.used
        if included_ids:
            await self.touch_items(included_ids, business_id=business_id)
        return ctx


# --------------------------------------------------------------------- tools --


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
            relevance = (
                f" (relevance {item.similarity:.2f})"
                if item.similarity is not None
                else ""
            )
            lines.append(f"- [{item.kind}] {item.label}: {item.detail}{relevance}")
    if ctx_brand.learnings:
        lines.append("")
        lines.append("Recent learnings:")
        for item in ctx_brand.learnings:
            lines.append(f"- {item.detail}")
    if ctx_brand.truncated:
        lines.append("")
        lines.append(
            f"(brand context was truncated to stay under {ctx_brand.token_estimate} "
            f"of {BRAND_CONTEXT_TOKEN_CAP} tokens)"
        )
    return "\n".join(lines)


__all__ = [
    "BRAND_CONTEXT_TOKEN_CAP",
    "BrandMemoryStore",
    "MEMORY_KINDS",
    "format_context_for_prompt",
    "get_brand_context",
]
