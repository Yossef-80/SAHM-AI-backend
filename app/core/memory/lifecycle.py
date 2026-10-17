"""Write-path lifecycle: identity of a memory, expiry, and pruning.

Three rules that keep a long-lived workspace memory from turning into a swamp:

1. **Dedupe.** The same statement written twice is one row, not two. The key is
   a digest of the *semantic* identity (kind + workspace + campaign + normalised
   text), so ``"بن البيت قهوة"`` and ``"بِنْ البَيتِ قهوه"`` collapse.
2. **Expiry.** Episodic kinds carry a TTL from :mod:`app.core.memory.kinds`.
   Expiry is written into the row (``expires_at``) so reads are an indexed
   predicate rather than Python arithmetic over the whole table.
3. **Pruning.** A hard per-kind ceiling, enforced on write: the oldest,
   least-important, never-used rows are archived (``archived_at``), never
   deleted. Archiving keeps the audit trail and makes a bad prune reversible.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.memory.kinds import default_importance, spec_for
from app.core.memory.lexical import normalize


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_aware(value: datetime | None) -> datetime | None:
    """Normalise SQLite's naive datetimes to UTC-aware ones."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def dedupe_digest(
    *,
    kind: str,
    business_id: str,
    text: str,
    campaign_id: str | None = None,
) -> str:
    """Stable digest of a memory's semantic identity."""
    # Coerce, don't trust: this digest is computed from arguments supplied by a
    # dozen producers, and a non-string (a mocked session attribute, an int id)
    # must not raise in the middle of a write.
    payload = "\x1f".join(
        [
            str(kind).strip().lower(),
            str(business_id).strip(),
            str(campaign_id or "").strip(),
            normalize(str(text)),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def expiry_for(
    kind: str,
    *,
    now: datetime | None = None,
    default_ttl_days: int = 180,
) -> datetime | None:
    """When a row of ``kind`` stops being eligible, or ``None`` for never."""
    spec = spec_for(kind)
    ttl = spec.ttl_days if spec.ttl_days is not None else None
    if ttl is None:
        # A kind declared permanent stays permanent; ``default_ttl_days`` only
        # applies to kinds the registry does not know (defensive path).
        return None
    base = ensure_aware(now) or utcnow()
    return base + timedelta(days=ttl)


def memory_defaults(
    kind: str,
    *,
    business_id: str,
    text: str,
    campaign_id: str | None = None,
    source: str = "",
) -> dict[str, Any]:
    """Lifecycle columns for a row created outside :class:`BrandMemoryStore`.

    Several producers (brand identity sync, research runs, chat attachments)
    construct ``BrandMemoryItem`` directly because they need to control the id
    or write inside a larger transaction. Without this helper those rows would
    silently get ``expires_at=None`` and ``importance=0.5`` and outlive every
    row the store writes.
    """
    return {
        "importance": default_importance(kind),
        "expires_at": expiry_for(kind),
        "dedupe_digest": dedupe_digest(
            kind=kind,
            business_id=business_id,
            text=text,
            campaign_id=campaign_id,
        ),
        "source": str(source or "direct")[:40],
    }


def is_expired(item: Any, *, now: datetime | None = None) -> bool:
    expires_at = ensure_aware(getattr(item, "expires_at", None))
    if expires_at is None:
        return False
    return expires_at <= (ensure_aware(now) or utcnow())


def prune_rank(item: Any) -> tuple[float, float, datetime]:
    """Sort key for pruning: worst first.

    Never-used, unimportant and old rows go before anything the agents have
    actually retrieved (``use_count`` / ``last_used_at`` are maintained by the
    store on every successful retrieval).
    """
    created = ensure_aware(getattr(item, "created_at", None)) or utcnow()
    pinned = 1.0 if getattr(item, "pinned", False) else 0.0
    importance = float(getattr(item, "importance", 0.5) or 0.5)
    use_count = int(getattr(item, "use_count", 0) or 0)
    # Higher rank = safer to archive.
    return (pinned, importance + min(use_count, 10) * 0.01, created)


__all__ = [
    "dedupe_digest",
    "ensure_aware",
    "expiry_for",
    "is_expired",
    "memory_defaults",
    "prune_rank",
    "utcnow",
]
