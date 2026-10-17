"""Deterministic memory ranking -- no vectors, no model call.

Every signal is something the system already knows:

======================  ==================================================
signal                  meaning
======================  ==================================================
``lexical``             BM25 + exact-phrase hit against the query
``recency``             ``0.5 ** (age_days / half_life)`` -- decays smoothly
``importance``          per-item 0..1, seeded from the kind registry
``affinity``            1 when the item's kind serves the current task group
``pinned``              user/model explicitly pinned it (flat boost)
``campaign``            item belongs to the campaign in focus (flat boost)
======================  ==================================================

The result is explainable: :class:`RankedMemory.reason` names the signals that
fired, which is what the admin search box and the debug logs print instead of a
bare cosine number.

Why no learning-to-rank, embeddings, or LLM re-scoring here: this runs inside
every agent turn and inside every ``GET /memory/items?q=``. It must be
sub-millisecond, deterministic, auditable, and work with no API key. At the
corpus sizes this product has (hundreds of rows per workspace), lexical+recency
beats a 64-dimension hashed bag-of-words by a wide margin -- and does not
require re-embedding the table when the provider changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Sequence

from app.core.memory.kinds import spec_for, task_group
from app.core.memory.lexical import LexicalIndex, phrase_bonus
from app.core.memory.lifecycle import ensure_aware, utcnow
from app.core.memory.policy import MemoryPolicy


@dataclass(frozen=True)
class RankedMemory:
    """One scored memory with its per-signal breakdown."""

    item: Any
    score: float
    lexical: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    affinity: float = 0.0
    pinned: bool = False
    campaign: bool = False
    matched: tuple[str, ...] = ()

    def reason(self) -> str:
        parts = [
            f"score={self.score:.2f}",
            f"lex={self.lexical:.2f}",
            f"rec={self.recency:.2f}",
            f"imp={self.importance:.2f}",
            f"aff={self.affinity:.1f}",
        ]
        if self.pinned:
            parts.append("pinned")
        if self.campaign:
            parts.append("campaign")
        if self.matched:
            parts.append("terms=" + ",".join(self.matched[:4]))
        return " ".join(parts)


def recency_factor(
    created_at: datetime | None,
    *,
    now: datetime | None = None,
    half_life_days: float,
) -> float:
    """Exponential decay, 1.0 at creation, 0.5 after one half-life."""
    created = ensure_aware(created_at)
    if created is None:
        return 0.5
    reference = ensure_aware(now) or utcnow()
    age_days = max(0.0, (reference - created).total_seconds() / 86_400.0)
    return round(0.5 ** (age_days / max(0.1, half_life_days)), 4)


def rank_items(
    items: Sequence[Any],
    *,
    query: str = "",
    task_name: str | None = None,
    policy: MemoryPolicy,
    now: datetime | None = None,
    campaign_id: str | None = None,
) -> list[RankedMemory]:
    """Score and order ``items`` best-first.

    ``items`` is the candidate window the store fetched (already filtered by
    kind, expiry and archive state). The function never touches the database.
    """
    if not items:
        return []

    group = task_group(task_name)
    reference = ensure_aware(now) or utcnow()

    docs = [str(getattr(item, "text", "") or "") for item in items]
    index = LexicalIndex(docs) if query.strip() else None
    lexical_hits = index.score_map(query) if index is not None else {}

    # Renormalise by the weights that can actually fire: with no query, a
    # perfect recency+affinity item must still be able to reach ~1.0.
    active_total = (
        policy.weight_affinity
        + policy.weight_importance
        + policy.weight_recency
        + (policy.weight_lexical if query.strip() else 0.0)
    ) or 1.0

    ranked: list[RankedMemory] = []
    for position, item in enumerate(items):
        kind = str(getattr(item, "kind", "") or "")
        try:
            spec = spec_for(kind)
            affinity = 1.0 if spec.serves(group) else 0.0
        except ValueError:  # unknown legacy kind: keep it eligible, no bonus
            affinity = 0.0

        lexical = 0.0
        matched: tuple[str, ...] = ()
        if index is not None:
            hit = lexical_hits.get(position)
            if hit is not None:
                lexical = hit.score
                matched = hit.matched
            bonus = phrase_bonus(query, docs[position])
            if bonus:
                lexical = min(1.0, lexical + policy.phrase_boost)

        recency = recency_factor(
            getattr(item, "created_at", None),
            now=reference,
            half_life_days=policy.recency_half_life_days,
        )
        importance = float(getattr(item, "importance", 0.5) or 0.5)
        pinned = bool(getattr(item, "pinned", False))
        item_campaign = getattr(item, "campaign_id", None)
        campaign = bool(campaign_id and item_campaign and item_campaign == campaign_id)

        base = (
            policy.weight_affinity * affinity
            + policy.weight_importance * importance
            + policy.weight_recency * recency
            + policy.weight_lexical * lexical
        ) / active_total

        score = base
        if pinned:
            score += policy.pinned_boost
        if campaign:
            score += policy.campaign_boost
        score = round(min(1.0, max(0.0, score)), 4)

        # A query that matches nothing should not return unrelated rows just
        # because they are fresh; the assembler relies on this floor.
        if query.strip() and lexical < policy.min_relevance and not pinned:
            continue

        ranked.append(
            RankedMemory(
                item=item,
                score=score,
                lexical=round(lexical, 4),
                recency=recency,
                importance=round(importance, 4),
                affinity=affinity,
                pinned=pinned,
                campaign=campaign,
                matched=matched,
            )
        )

    ranked.sort(
        key=lambda row: (
            row.score,
            row.pinned,
            ensure_aware(getattr(row.item, "created_at", None)) or utcnow(),
        ),
        reverse=True,
    )
    return ranked


def take_top(ranked: Iterable[RankedMemory], limit: int) -> list[RankedMemory]:
    return list(ranked)[: max(0, limit)]


__all__ = ["RankedMemory", "rank_items", "recency_factor", "take_top"]
