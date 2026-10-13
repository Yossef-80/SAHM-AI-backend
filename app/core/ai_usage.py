"""Global token metering for **every** LLM call the product makes.

Why this file exists
--------------------
``app/core/usage.py`` already knew how to *store* and *enforce* usage, but the
only thing that ever called :func:`record_usage` was the legacy
:mod:`app.core.agents.runtime`. Everything that runs on the new LangGraph layer
(``app/ai/**``) -- the assistant turn, Nour/Omar/Layla/Karim/Salma graphs, the
research tab, Layla's text generation, the reference-image VLM pass and every
memory distillation -- spent tokens that were **never recorded**. That is why the
usage screen looked broken: it was telling the truth about an empty table.

How it works
------------
1. A LangChain callback (:class:`TokenUsageCallback`) is attached to every chat
   model built by :func:`app.ai.providers.llm.get_llm`, so *any* invocation --
   including ones hidden inside a LangGraph node or a ``asyncio.to_thread`` --
   reports its usage without the call site knowing anything about accounting.
2. The callback resolves the *scope* (workspace, business, persona, action) from
   a :class:`contextvars.ContextVar` that the HTTP layer sets once per request in
   :func:`app.api.deps.resolve_workspace`. Context variables are copied into
   ``asyncio.to_thread`` workers, so the attribution survives the thread hop.
3. Deltas are appended to a process-local, thread-safe queue. A single writer
   task (started from the app lifespan) drains the queue into ``usage_records``
   on its own session, so accounting never touches -- or rolls back with -- the
   request's transaction. ``flush_ai_usage`` is exposed for tests and shutdown.

Everything here is best-effort by design: a metering failure must never fail a
user's turn. Failures are logged as ``usage_meter_*`` and swallowed.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import uuid
from collections import deque
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any, Iterator

from app.core.logging import get_logger

_logger = get_logger(__name__)

#: How often the writer drains the queue even when nothing woke it up.
FLUSH_INTERVAL_SECONDS: float = 5.0

#: Hard cap on the in-memory queue. Beyond this the oldest deltas are dropped
#: (with one warning per overflow) rather than growing without bound in a
#: process whose database is unreachable.
MAX_PENDING_DELTAS: int = 5_000


@dataclass(frozen=True, slots=True)
class UsageScope:
    """Who is spending, and on what, for the LLM calls made inside this context."""

    workspace_id: str
    business_id: str
    action: str = "unknown"
    persona_id: str | None = None
    task_id: str | None = None
    conversation_id: str | None = None


@dataclass(slots=True)
class UsageDelta:
    """One recorded LLM call, waiting to be written."""

    workspace_id: str
    business_id: str
    action: str
    persona_id: str | None
    task_id: str | None
    model: str | None
    provider: str | None
    tokens_in: int
    tokens_out: int
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return int(self.tokens_in) + int(self.tokens_out)


_scope: ContextVar[UsageScope | None] = ContextVar("sahm_usage_scope", default=None)

_pending: deque[UsageDelta] = deque()
_pending_lock = threading.Lock()
_overflow_warned = False

_loop: asyncio.AbstractEventLoop | None = None
_wakeup: asyncio.Event | None = None
_writer_task: asyncio.Task[None] | None = None


# --------------------------------------------------------------------- scope ---


def set_usage_scope(scope: UsageScope) -> Token:
    """Bind ``scope`` to the current task. Callers may ignore the token."""
    return _scope.set(scope)


def reset_usage_scope(token: Token | None) -> None:
    if token is None:
        return
    with contextlib.suppress(ValueError, LookupError):
        _scope.reset(token)


def current_usage_scope() -> UsageScope | None:
    return _scope.get()


@contextlib.contextmanager
def usage_scope(
    *,
    workspace_id: str,
    business_id: str,
    action: str = "unknown",
    persona_id: str | None = None,
    task_id: str | None = None,
    conversation_id: str | None = None,
) -> Iterator[UsageScope]:
    """Temporarily attribute LLM calls to ``action`` (e.g. ``research.website``)."""
    scope = UsageScope(
        workspace_id=workspace_id,
        business_id=business_id,
        action=action,
        persona_id=persona_id,
        task_id=task_id,
        conversation_id=conversation_id,
    )
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        reset_usage_scope(token)


@contextlib.contextmanager
def usage_action(action: str, *, persona_id: str | None = None) -> Iterator[None]:
    """Re-label the *current* scope for the duration of a block.

    No-op when no scope is bound (unit tests, CLI scripts): the deltas are then
    dropped by the writer with a debug log rather than inventing a tenant.
    """
    existing = _scope.get()
    if existing is None:
        yield
        return
    token = _scope.set(
        replace(
            existing,
            action=action,
            persona_id=persona_id or existing.persona_id,
        )
    )
    try:
        yield
    finally:
        reset_usage_scope(token)


# ------------------------------------------------------------------ recording --


def record_ai_usage(
    *,
    model: str | None,
    tokens_in: int,
    tokens_out: int,
    provider: str | None = None,
    estimated: bool = False,
    persona_id: str | None = None,
    action: str | None = None,
    workspace_id: str | None = None,
    business_id: str | None = None,
) -> None:
    """Queue one LLM call. Safe to call from any thread; never raises.

    Zero/zero deltas are dropped: a provider that reports nothing would otherwise
    inflate the call count with rows that carry no billable information.
    """
    tokens_in = max(0, int(tokens_in or 0))
    tokens_out = max(0, int(tokens_out or 0))
    if tokens_in == 0 and tokens_out == 0:
        return

    scope = current_usage_scope()
    resolved_workspace = workspace_id or (scope.workspace_id if scope else "")
    resolved_business = business_id or (scope.business_id if scope else "")
    if not resolved_workspace or not resolved_business:
        # No tenant in context (a script, a test, a background job without a
        # workspace): dropping is correct. Attributing it to "the oldest
        # business" would silently bill one customer for another's work.
        _logger.debug(
            "usage_meter_without_scope",
            extra={"model": model, "tokens_in": tokens_in, "tokens_out": tokens_out},
        )
        return

    delta = UsageDelta(
        workspace_id=resolved_workspace,
        business_id=resolved_business,
        action=(action or (scope.action if scope else "unknown")) or "unknown",
        persona_id=persona_id or (scope.persona_id if scope else None),
        task_id=scope.task_id if scope else None,
        model=model,
        provider=provider,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        estimated=estimated,
    )

    global _overflow_warned
    with _pending_lock:
        if len(_pending) >= MAX_PENDING_DELTAS:
            _pending.popleft()
            if not _overflow_warned:
                _overflow_warned = True
                _logger.warning(
                    "usage_meter_queue_overflow",
                    extra={"max_pending": MAX_PENDING_DELTAS},
                )
        _pending.append(delta)

    _wake_writer()


def pending_usage_count() -> int:
    with _pending_lock:
        return len(_pending)


def drain_pending_usage() -> list[UsageDelta]:
    with _pending_lock:
        drained = list(_pending)
        _pending.clear()
    return drained


def _wake_writer() -> None:
    loop = _loop
    event = _wakeup
    if loop is None or event is None or loop.is_closed():
        return
    try:
        loop.call_soon_threadsafe(event.set)
    except RuntimeError:  # pragma: no cover - loop torn down mid-flight
        pass


# --------------------------------------------------------------------- writer --


async def _write_deltas(deltas: list[UsageDelta]) -> int:
    """Insert ``deltas`` on a dedicated session. Returns the number written."""
    if not deltas:
        return 0
    from app.db.models import UsageRecord
    from app.db.session import session_scope

    written = 0
    async with session_scope() as session:
        for delta in deltas:
            session.add(
                UsageRecord(
                    id=f"ur_{uuid.uuid4().hex[:24]}",
                    business_id=delta.business_id,
                    workspace_id=delta.workspace_id,
                    task_id=delta.task_id,
                    persona_id=delta.persona_id,
                    tier=None,
                    action=delta.action,
                    model=delta.model or delta.provider,
                    tokens_in=int(delta.tokens_in),
                    tokens_out=int(delta.tokens_out),
                    estimated=bool(delta.estimated),
                )
            )
            written += 1
    return written


async def flush_ai_usage() -> int:
    """Drain the queue into ``usage_records``. Never raises."""
    deltas = drain_pending_usage()
    if not deltas:
        return 0
    try:
        return await _write_deltas(deltas)
    except Exception as exc:  # pragma: no cover - depends on live database
        _logger.warning(
            "usage_meter_flush_failed",
            extra={"error": str(exc), "dropped": len(deltas)},
        )
        return 0


async def usage_writer_loop() -> None:
    """Drain the queue forever; started from the app lifespan."""
    global _loop, _wakeup
    _loop = asyncio.get_running_loop()
    _wakeup = asyncio.Event()
    interval = _flush_interval_seconds()
    while True:
        try:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(_wakeup.wait(), timeout=interval)
            _wakeup.clear()
            await flush_ai_usage()
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await flush_ai_usage()
            raise
        except Exception as exc:  # pragma: no cover
            _logger.warning("usage_meter_loop_error", extra={"error": str(exc)})
            await asyncio.sleep(interval)


def start_usage_writer() -> None:
    """Start the background writer once per process."""
    global _writer_task
    if _writer_task is not None and not _writer_task.done():
        return
    _writer_task = asyncio.create_task(usage_writer_loop(), name="ai-usage-writer")


async def stop_usage_writer() -> None:
    """Cancel the writer and flush whatever is left."""
    global _writer_task, _loop, _wakeup
    task, _writer_task = _writer_task, None
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    await flush_ai_usage()
    _wakeup = None


def _flush_interval_seconds() -> float:
    raw = os.getenv("AI_USAGE_FLUSH_INTERVAL_SECONDS", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return FLUSH_INTERVAL_SECONDS
    return value if value > 0 else FLUSH_INTERVAL_SECONDS


# ---------------------------------------------------------------- the callback --


def _estimate_tokens(text: Any) -> int:
    """Rough ``len/4`` estimate used only when a provider reports no usage."""
    if not text:
        return 0
    if isinstance(text, list):
        joined = " ".join(_message_text(item) for item in text)
    else:
        joined = str(text)
    return max(0, len(joined) // 4)


def _message_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        return str(part.get("text") or "")
    return str(part or "")


@dataclass(slots=True)
class _Extracted:
    tokens_in: int
    tokens_out: int
    estimated: bool


def extract_token_usage(response: Any, *, prompts: Any = None) -> _Extracted:
    """Pull ``(tokens_in, tokens_out)`` out of a LangChain ``LLMResult``.

    Order matters: the standardised ``usage_metadata`` block first, then the
    provider-shaped ``llm_output``/``generation_info``, and only then a
    character-count estimate -- flagged as estimated so a billing report can tell
    a measured number from a guess.
    """
    llm_output = getattr(response, "llm_output", None) or {}
    generations = getattr(response, "generations", None) or []

    message_usage: Any = None
    generation_usage: Any = None
    prompt_text = ""
    completion_text = ""

    for group in generations:
        for generation in group or []:
            message = getattr(generation, "message", None)
            if message is not None:
                if message_usage is None:
                    message_usage = getattr(message, "usage_metadata", None)
                if not completion_text:
                    completion_text = _message_text(getattr(message, "content", ""))
            info = getattr(generation, "generation_info", None)
            if isinstance(info, dict) and generation_usage is None:
                generation_usage = info
            if not completion_text:
                completion_text = _message_text(getattr(generation, "text", ""))

    if isinstance(message_usage, dict):
        tokens_in = message_usage.get("input_tokens")
        tokens_out = message_usage.get("output_tokens")
        if tokens_in is not None or tokens_out is not None:
            return _Extracted(
                int(tokens_in or 0), int(tokens_out or 0), estimated=False
            )

    for key in ("token_usage", "usage"):
        blob = llm_output.get(key)
        if isinstance(blob, dict):
            tokens_in = blob.get("prompt_tokens", blob.get("input_tokens"))
            tokens_out = blob.get("completion_tokens", blob.get("output_tokens"))
            if tokens_in is not None or tokens_out is not None:
                return _Extracted(
                    int(tokens_in or 0), int(tokens_out or 0), estimated=False
                )

    if isinstance(generation_usage, dict):
        tokens_in = generation_usage.get("prompt_tokens", generation_usage.get("input_tokens"))
        tokens_out = generation_usage.get(
            "completion_tokens", generation_usage.get("output_tokens")
        )
        if tokens_in is not None or tokens_out is not None:
            return _Extracted(
                int(tokens_in or 0), int(tokens_out or 0), estimated=False
            )

    # Some wrappers surface the same numbers under a different key.
    for key in ("token_usage", "usage_metadata"):
        blob = llm_output.get(key)
        if isinstance(blob, dict):
            total = blob.get("total_tokens")
            if isinstance(total, int) and total > 0:
                return _Extracted(total, 0, estimated=False)

    prompt_source = prompt_text or prompts or llm_output.get("prompts") or []
    estimated_in = _estimate_tokens(prompt_source)
    estimated_out = _estimate_tokens(completion_text)
    return _Extracted(estimated_in, estimated_out, estimated=True)


def _callback_base() -> type:
    """LangChain's callback base when the AI stack is installed, else ``object``.

    Resolved once at import so this module -- imported from ``app.api.deps`` on
    every request -- never hard-depends on LangChain being present.
    """
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except Exception:  # pragma: no cover - AI stack not installed
        return object
    return BaseCallbackHandler


class TokenUsageCallback(_callback_base()):  # type: ignore[misc, valid-type]
    """Record one usage row for every LLM call a model makes.

    ``raise_error = False`` is deliberate: a metering bug must never turn a
    customer's answer into a 500.
    """

    raise_error = False

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        model_name: str | None = None,
    ) -> None:
        try:
            super().__init__()
        except TypeError:  # pragma: no cover - base is ``object``
            pass
        # ``model_name`` is an alias: the attach helper speaks of
        # ``provider/model_name`` while direct callers say ``model``.
        self.provider = provider
        self.model = model or model_name
        self._prompts: dict[str, Any] = {}

    # -- request/response hooks -------------------------------------------------
    # The prompts are kept so an un-reporting provider can still be billed with
    # an estimate instead of being recorded as free.

    def _remember(self, kwargs: dict[str, Any], prompts: Any) -> None:
        run_id = str(kwargs.get("run_id") or "")
        if run_id:
            self._prompts[run_id] = prompts

    def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        self._remember(kwargs, prompts)

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self._remember(kwargs, messages)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            prompts = self._prompts.pop(str(kwargs.get("run_id") or ""), None)
            extracted = extract_token_usage(response, prompts=prompts)
            record_ai_usage(
                model=self.model,
                provider=self.provider,
                tokens_in=extracted.tokens_in,
                tokens_out=extracted.tokens_out,
                estimated=extracted.estimated,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _logger.debug("usage_meter_callback_failed", extra={"error": str(exc)})


def build_usage_callback(*, provider: str | None, model: str | None) -> Any:
    return TokenUsageCallback(provider=provider, model=model)


def attach_usage_metering(model: Any, *, provider: str | None, model_name: str | None) -> Any:
    """Attach the metering callback to ``model`` and return it unchanged.

    Instance-level ``callbacks`` is used (rather than ``with_config``) so the
    object identity -- and therefore ``bind_tools`` /
    ``with_structured_output`` -- behaves exactly as before.
    """
    try:
        handler = build_usage_callback(provider=provider, model=model_name)
        existing = getattr(model, "callbacks", None)
        if existing is None:
            model.callbacks = [handler]
        elif isinstance(existing, (list, tuple)):
            model.callbacks = [*existing, handler]
        else:
            # A BaseCallbackManager is configured; merging into it would need
            # private APIs, so leave the provider's configuration alone.
            _logger.debug("usage_meter_skipped_managed_callbacks")
    except Exception as exc:  # pragma: no cover - defensive
        _logger.debug("usage_meter_attach_failed", extra={"error": str(exc)})
    return model


__all__ = [
    "MAX_PENDING_DELTAS",
    "TokenUsageCallback",
    "UsageDelta",
    "UsageScope",
    "attach_usage_metering",
    "build_usage_callback",
    "current_usage_scope",
    "drain_pending_usage",
    "extract_token_usage",
    "flush_ai_usage",
    "pending_usage_count",
    "record_ai_usage",
    "reset_usage_scope",
    "set_usage_scope",
    "start_usage_writer",
    "stop_usage_writer",
    "usage_action",
    "usage_scope",
    "usage_writer_loop",
]
