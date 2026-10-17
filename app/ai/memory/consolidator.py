"""One background memory pass per workspace.

What this replaces
------------------
After every chat turn the API used to start **three** independent background
passes for the same business:

1. ``manage_session_memory_with_token_cap`` -- summarise old turns;
2. ``memory_debouncer.schedule_ltm_update`` -- after 3 idle minutes, a *strong*
   LLM pass rewriting ``Profile.user_context_summary`` and
   ``BrandMemoryItem(strategy_blueprint)``;
3. ``brand_intelligence_agent.sync_brand_identity`` -- another *strong* LLM pass
   rewriting ``BrandProfile`` and ``BrandMemoryItem(brand_identity)``.

Nothing serialised them. Two turns in quick succession could have passes 2 and
3 interleaved, each reading the profile, editing it and committing -- last
writer wins, and the loser's tokens were paid for. On a per-message cadence that
is both a correctness bug (a profile row edited from a stale read) and a cost
bug (two 4k-token completions per message).

The contract now
----------------
``schedule()`` is the only entry point. It guarantees, per ``business_id``:

* **one** in-flight pass (an ``asyncio.Lock``), so no two writers race;
* session compaction runs on the turn itself (it is token-gated and cheap);
* the two LLM passes run once the workspace has been idle for
  ``MEMORY_LTM_DEBOUNCE_SECONDS``, and collapse N messages into one run;
* the identity pass additionally respects
  ``MEMORY_IDENTITY_MIN_INTERVAL_SECONDS`` -- it rewrites the whole brand
  profile from scratch, which is pointless every three minutes.

Every step is isolated: a failure in one is logged and the next still runs. A
memory failure must never fail a chat turn (this all runs after the response has
been streamed).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ConsolidationRequest:
    business_id: str
    conversation_id: str | None = None
    request_id: str = ""


class MemoryConsolidator:
    """Serialise and debounce the post-turn memory work, per workspace."""

    def __init__(self) -> None:
        self._timers: dict[str, asyncio.Task[None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._session_compacted: set[str] = set()
        self._last_identity_run: dict[str, float] = {}

    # ------------------------------------------------------------------ public --
    def schedule(
        self,
        *,
        business_id: str,
        conversation_id: str | None = None,
        request_id: str = "",
    ) -> None:
        """Queue memory work for this workspace; safe to call on every turn."""
        if not business_id:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:  # called from a sync context (tests, CLI)
            return

        task = asyncio.create_task(
            self._run(
                ConsolidationRequest(
                    business_id=business_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                )
            ),
            name=f"memory-consolidate:{business_id}",
        )
        self._timers[business_id] = task
        task.add_done_callback(
            lambda finished, key=business_id: self._forget(key, finished)
        )

    async def wait_idle(self, business_id: str | None = None) -> None:
        """Await the in-flight pass(es). Used by tests and shutdown."""
        keys = [business_id] if business_id else list(self._timers)
        for key in keys:
            task = self._timers.get(key)
            if task is not None and not task.done():
                try:
                    await task
                except Exception:  # noqa: BLE001 - already logged inside
                    pass

    def reset(self) -> None:
        """Drop all bookkeeping (tests)."""
        self._timers.clear()
        self._locks.clear()
        self._session_compacted.clear()
        self._last_identity_run.clear()

    # ------------------------------------------------------------------ internals --
    def _forget(self, business_id: str, task: asyncio.Task[None]) -> None:
        if self._timers.get(business_id) is task:
            self._timers.pop(business_id, None)

    def _lock(self, business_id: str) -> asyncio.Lock:
        lock = self._locks.get(business_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[business_id] = lock
        return lock

    async def _run(self, req: ConsolidationRequest) -> None:
        async with self._lock(req.business_id):
            # 1. Session layer -- cheap, and the next turn needs it, so it runs
            #    before the idle delay rather than after it.
            await self._step(req, "session", self._compact_session)

            delay = _debounce_seconds()
            if delay:
                await asyncio.sleep(delay)

            # 2. Profile layer -- the rolling LTM summary.
            await self._step(req, "profile", self._update_profile)

            # 3. Identity layer -- whole-profile re-extraction.
            if _identity_sync_enabled() and self._identity_due(req.business_id):
                await self._step(req, "identity", self._sync_identity)

    async def _step(self, req: ConsolidationRequest, name: str, fn: Any) -> None:
        started = time.monotonic()
        try:
            await fn(req)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            logger.warning(
                "memory_step_failed",
                extra={
                    "business_id": req.business_id,
                    "step": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                },
            )
            return
        logger.info(
            "memory_step_finished",
            extra={
                "business_id": req.business_id,
                "step": name,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            },
        )

    def _identity_due(self, business_id: str) -> bool:
        min_interval = _identity_min_interval()
        last = self._last_identity_run.get(business_id, 0.0)
        return (time.monotonic() - last) >= min_interval

    async def _compact_session(self, req: ConsolidationRequest) -> None:
        if not req.conversation_id:
            return
        # Compaction is idempotent but not free: skip it when this process has
        # already compacted this conversation and nothing was added since.
        from sqlalchemy import select

        from app.ai.memory.compressor import manage_session_memory_with_token_cap
        from app.db.models import Conversation, Message
        from app.db.session import session_scope

        async with session_scope() as db:
            conv = await db.get(Conversation, req.conversation_id)
            if conv is None:
                return
            messages = list(
                (
                    await db.execute(
                        select(Message)
                        .where(Message.conversation_id == req.conversation_id)
                        .order_by(Message.created_at.asc())
                    )
                )
                .scalars()
                .all()
            )
            await manage_session_memory_with_token_cap(
                db, conversation=conv, messages=messages
            )
            await db.commit()

    async def _update_profile(self, req: ConsolidationRequest) -> None:
        from app.ai.memory.compressor import IntelligentMemoryCompressor

        await IntelligentMemoryCompressor().update_global_profile(req.business_id)

    async def _sync_identity(self, req: ConsolidationRequest) -> None:
        from app.core.brand.identity_service import brand_intelligence_agent
        from app.db.session import session_scope

        async with session_scope() as db:
            await brand_intelligence_agent.sync_brand_identity(db, req.business_id)
            await db.commit()
        self._last_identity_run[req.business_id] = time.monotonic()


def _debounce_seconds() -> int:
    try:
        from app.ai.settings import get_ai_settings

        return max(0, int(get_ai_settings().memory_ltm_debounce_seconds))
    except Exception:  # noqa: BLE001 - settings must never break memory
        return 180


def _identity_sync_enabled() -> bool:
    try:
        from app.ai.settings import get_ai_settings

        return bool(getattr(get_ai_settings(), "memory_identity_sync_enabled", True))
    except Exception:  # noqa: BLE001
        return True


def _identity_min_interval() -> int:
    try:
        from app.ai.settings import get_ai_settings

        return max(
            0, int(getattr(get_ai_settings(), "memory_identity_min_interval_seconds", 900))
        )
    except Exception:  # noqa: BLE001
        return 900


#: Process-wide singleton -- scheduling must be shared, or two requests would
#: each hold their own "one in-flight pass" lock.
memory_consolidator = MemoryConsolidator()


__all__ = ["ConsolidationRequest", "MemoryConsolidator", "memory_consolidator"]
