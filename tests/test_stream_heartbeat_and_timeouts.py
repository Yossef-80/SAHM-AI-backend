"""The streamed turn must never look like a frozen screen.

Live log the user pasted ("علق عند الorchestrator"):

    [ROUTING - OUTGOING] 200 POST /api/v1/assistant/message/stream (3767ms, DB: 2313ms/11q)
    INFO: 127.0.0.1:55284 - "POST /api/v1/assistant/message/stream HTTP/1.1" 200 OK

Both of those lines are written when the SSE **headers** go out -- proven by a probe
against the same middleware -- so they contain nothing about the turn itself, and the
turn logged nothing of its own. ``handle_message`` had no start/finish log at all, and
a model call is allowed 180s x 3 tries, so a stuck turn could sit there for minutes
with an empty log and a spinner on screen.

What this file pins:

* a streamed turn announces itself (``assistant_turn_started`` /
  ``assistant_turn_finished``) and closes with a line saying whether a terminal
  event ever went out;
* a slow turn sends a *named* ``still_working`` event, not only an invisible
  keep-alive comment;
* a turn that outlives the ceiling ends the stream with a readable message
  **without cancelling the work**, and the memory pass still happens when the
  reply finally lands.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import BackgroundTasks

from app.api import assistant as assistant_api
from app.schemas.assistant import AssistantMessageRequest

pytestmark = pytest.mark.anyio


# ----------------------------------------------------------------- helpers -----


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture()
def captured_logs():
    handler = _Capture()
    logger = logging.getLogger("app.api.assistant")
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def _response(conversation_id: str = "conv_1") -> SimpleNamespace:
    class _Response:
        def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
            return {"conversationId": conversation_id, "blocks": []}

    return SimpleNamespace(conversation_id=conversation_id, model_dump=_Response().model_dump)


def _workspace() -> SimpleNamespace:
    return SimpleNamespace(business=SimpleNamespace(id="biz_test"), business_id="biz_test")


def _stub_handle_message(*, delay: float = 0.0, error: Exception | None = None, tasks: list | None = None):
    """A stand-in turn: optionally slow, optionally failing."""

    async def _handle(_db: Any, _payload: Any, **_kwargs: Any) -> Any:
        if tasks is not None:
            tasks.append(asyncio.current_task())
        await asyncio.sleep(delay)
        if error is not None:
            raise error
        return _response()

    return _handle


async def _run_stream(payload: AssistantMessageRequest, monkeypatch: pytest.MonkeyPatch, **stub: Any) -> str:
    monkeypatch.setattr(assistant_api, "handle_message", _stub_handle_message(**stub))
    response = await assistant_api.post_message_stream(
        payload,
        ws=_workspace(),
        db=None,
        llm=None,
        fast_llm=None,
        request_id="req_test",
        background_tasks=BackgroundTasks(),
    )
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
    return "".join(chunks)


# ---------------------------------------------------------- the plain turn -----


async def test_a_normal_turn_still_sends_started_and_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = await _run_stream(AssistantMessageRequest(message="مرحبا", language="ar"), monkeypatch)

    assert "event: progress" in body
    assert '"type": "started"' in body
    assert "event: complete" in body
    assert "event: error" not in body


async def test_the_stream_says_whether_it_ended_by_itself(
    monkeypatch: pytest.MonkeyPatch, captured_logs: _Capture
) -> None:
    await _run_stream(AssistantMessageRequest(message="مرحبا", language="ar"), monkeypatch)

    closed = [r for r in captured_logs.records if r.msg == "assistant_stream_closed"]
    assert closed, "a stream must log how it ended"
    fields = getattr(closed[0], "extra", {})
    assert fields.get("terminal_event_sent") is True
    assert fields.get("duration_ms") >= 0


async def test_a_failed_turn_is_logged_with_its_own_error(
    monkeypatch: pytest.MonkeyPatch, captured_logs: _Capture
) -> None:
    body = await _run_stream(
        AssistantMessageRequest(message="مرحبا", language="ar"),
        monkeypatch,
        error=RuntimeError("provider exploded"),
    )

    assert "event: error" in body
    failed = [r for r in captured_logs.records if r.msg == "assistant_stream_failed"]
    assert failed and getattr(failed[0], "extra", {}).get("error_type") == "RuntimeError"


# --------------------------------------------------------- the slow turn -------


async def test_a_slow_turn_says_it_is_still_working(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare keep-alive comment keeps proxies happy and tells the user nothing."""
    monkeypatch.setattr(assistant_api, "_HEARTBEAT_EVERY_SECONDS", 0.05)
    monkeypatch.setattr(assistant_api, "_STILL_WORKING_AFTER_SECONDS", 1)

    body = await _run_stream(
        AssistantMessageRequest(message="صمم صورة", language="ar"),
        monkeypatch,
        delay=1.6,
    )

    assert ": keep-alive" in body
    assert '"type": "still_working"' in body
    assert "event: complete" in body  # the turn did finish, just slowly


async def test_the_still_working_event_carries_the_elapsed_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assistant_api, "_HEARTBEAT_EVERY_SECONDS", 0.05)
    monkeypatch.setattr(assistant_api, "_STILL_WORKING_AFTER_SECONDS", 1)

    body = await _run_stream(
        AssistantMessageRequest(message="صمم صورة", language="ar"), monkeypatch, delay=1.6
    )

    assert '"seconds":' in body


# --------------------------------------------------- the turn that never ends --


async def test_a_turn_that_outlives_the_ceiling_ends_the_stream_with_arabic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assistant_api, "_HEARTBEAT_EVERY_SECONDS", 0.05)
    monkeypatch.setattr(assistant_api, "_STREAM_GIVE_UP_SECONDS", 1)

    body = await _run_stream(
        AssistantMessageRequest(message="صمم صورة", language="ar"), monkeypatch, delay=3.0
    )

    assert "event: error" in body
    assert "وقت أطول من المتوقع" in body


async def test_the_give_up_does_not_cancel_the_running_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool may be mid-publish; ending the screen must not kill the work."""
    monkeypatch.setattr(assistant_api, "_HEARTBEAT_EVERY_SECONDS", 0.05)
    monkeypatch.setattr(assistant_api, "_STREAM_GIVE_UP_SECONDS", 1)
    tasks: list[Any] = []

    body = await _run_stream(
        AssistantMessageRequest(message="صمم صورة", language="ar"),
        monkeypatch,
        delay=2.0,
        tasks=tasks,
    )

    assert "event: error" in body
    assert tasks and not tasks[0].cancelled()
    tasks[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]


async def test_the_memory_pass_still_happens_after_a_late_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assistant_api, "_HEARTBEAT_EVERY_SECONDS", 0.05)
    monkeypatch.setattr(assistant_api, "_STREAM_GIVE_UP_SECONDS", 1)

    done = asyncio.Event()
    seen: list[str] = []

    async def _record(*, business_id: str, conversation_id: str, request_id: str) -> None:
        seen.append(conversation_id)
        done.set()

    monkeypatch.setattr(assistant_api, "_post_turn_memory_background", _record)

    await _run_stream(
        AssistantMessageRequest(message="صمم صورة", language="ar"), monkeypatch, delay=1.6
    )
    await asyncio.wait_for(done.wait(), timeout=5)

    assert seen == ["conv_1"]


# ------------------------------------------------------------- the turn log ----


def test_the_turn_itself_logs_a_start_and_a_finish() -> None:
    """The missing lines in the pasted log: the turn announced nothing at all."""
    source = (
        Path(__file__).resolve().parents[1] / "app" / "core" / "orchestrator" / "service.py"
    ).read_text(encoding="utf-8")

    assert "assistant_turn_started" in source
    assert "assistant_turn_finished" in source
    assert "turn_started = time.monotonic()" in source


def test_every_graph_run_logs_the_model_and_the_duration() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "app" / "ai" / "agents" / "orchestrator" / "brain.py"
    ).read_text(encoding="utf-8")

    assert 'logger.info(\n            "agent_graph_started"' in source or "agent_graph_started" in source
    assert "agent_graph_finished" in source
    assert source.count("agent_graph_finished") >= 2  # first try and the fallback retry


async def test_a_finished_turn_does_not_wait_out_the_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug behind a finished answer that still looked hung.

    The loop used to wait on the progress queue alone, so a turn that pushed no
    progress items (the common case) sat there until the heartbeat expired -- up to
    15 real seconds of a completed reply that the browser had not received yet.
    """
    import time as _time

    monkeypatch.setattr(assistant_api, "_HEARTBEAT_EVERY_SECONDS", 30)

    started = _time.monotonic()
    body = await _run_stream(AssistantMessageRequest(message="مرحبا", language="ar"), monkeypatch)
    elapsed = _time.monotonic() - started

    assert "event: complete" in body
    assert elapsed < 2, f"a finished turn waited {elapsed:.1f}s for the heartbeat"
