"""Every LLM call must be billed to the workspace that caused it.

The regression these tests lock down is the one that made the usage screen look
broken: ``usage_records`` was only written by the legacy agent runtime, so the
assistant, the research tab and every LangGraph agent spent tokens the meter
never saw.

Covered here:

* the LangChain callback turns an ``LLMResult`` into exactly one queued delta;
* a delta with no workspace in context is dropped rather than attributed to
  "the oldest business" (billing one customer for another's work);
* the writer persists queued deltas to ``usage_records`` with the action label;
* ``GET /usage`` reports the new per-action breakdown and refuses to quote a
  price when no rate is configured.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select

from app.core.ai_usage import (
    UsageScope,
    TokenUsageCallback,
    drain_pending_usage,
    flush_ai_usage,
    pending_usage_count,
    record_ai_usage,
    set_usage_scope,
    usage_action,
)
from app.db.models import UsageRecord


class _Message:
    def __init__(self, content: str, usage: dict[str, int] | None = None) -> None:
        self.content = content
        self.usage_metadata = usage


class _Generation:
    def __init__(self, message: _Message) -> None:
        self.message = message
        self.text = None
        self.generation_info = None


class _Result:
    def __init__(self, message: _Message, llm_output: dict[str, Any] | None = None) -> None:
        self.generations = [[_Generation(message)]]
        self.llm_output = llm_output


@pytest.fixture(autouse=True)
def _clean_queue() -> Any:
    drain_pending_usage()
    yield
    drain_pending_usage()


def test_callback_queues_one_delta_per_call() -> None:
    set_usage_scope(UsageScope(workspace_id="ws_1", business_id="biz_1", action="chat"))
    callback = TokenUsageCallback(provider="mock", model_name="mock-chat")

    callback.on_llm_end(_Result(_Message("answer", {"input_tokens": 120, "output_tokens": 30})))

    deltas = drain_pending_usage()
    assert len(deltas) == 1
    delta = deltas[0]
    assert (delta.tokens_in, delta.tokens_out) == (120, 30)
    assert delta.workspace_id == "ws_1"
    assert delta.action == "chat"
    assert delta.estimated is False


def test_provider_without_usage_is_estimated_not_zero() -> None:
    """A provider that reports nothing must still be billed -- approximately.

    Recording zero would hand the customer free tokens; the ``estimated`` flag is
    what keeps that honest rather than a silent under-count.
    """
    set_usage_scope(UsageScope(workspace_id="ws_1", business_id="biz_1", action="chat"))
    callback = TokenUsageCallback(provider="ollama", model_name="qwen")

    callback.on_llm_end(_Result(_Message("a" * 400)))
    deltas = drain_pending_usage()
    assert len(deltas) == 1
    assert deltas[0].tokens_out > 0
    assert deltas[0].estimated is True


def test_unscoped_call_is_dropped_not_attributed() -> None:
    import app.core.ai_usage as ai_usage

    ai_usage._scope.set(None)
    record_ai_usage(model="m", tokens_in=500, tokens_out=500)
    assert pending_usage_count() == 0


def test_usage_action_relabels_only_its_block() -> None:
    set_usage_scope(UsageScope(workspace_id="ws_1", business_id="biz_1", action="chat"))
    record_ai_usage(model="m", tokens_in=10, tokens_out=1)
    with usage_action("research.analyze_website"):
        record_ai_usage(model="m", tokens_in=20, tokens_out=2)
    record_ai_usage(model="m", tokens_in=30, tokens_out=3)

    actions = [delta.action for delta in drain_pending_usage()]
    assert actions == ["chat", "research.analyze_website", "chat"]


@pytest.mark.anyio
async def test_flush_writes_rows_with_action_and_estimated(api_client: Any, seeded_db: Any) -> None:
    from app.db.models import Business

    business = (await seeded_db.execute(select(Business))).scalars().first()
    assert business is not None

    set_usage_scope(
        UsageScope(
            workspace_id=business.workspace_id or "",
            business_id=business.id,
            action="chat",
        )
    )
    record_ai_usage(model="mock-chat", provider="mock", tokens_in=100, tokens_out=25)
    record_ai_usage(model="mock-chat", provider="mock", tokens_in=7, tokens_out=3, estimated=True)

    assert await flush_ai_usage() == 2

    rows = (
        (
            await seeded_db.execute(
                select(UsageRecord).where(UsageRecord.business_id == business.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2
    assert {row.action for row in rows} == {"chat"}
    assert sum(row.tokens_in + row.tokens_out for row in rows) == 135
    assert sum(1 for row in rows if row.estimated) == 1


@pytest.mark.anyio
async def test_usage_endpoint_reports_actions_and_withholds_unset_price(api_client: Any) -> None:
    resp = await api_client.get("/api/v1/usage?period=today")
    assert resp.status_code == 200
    body = resp.json()

    # New fields the billing screen reads.
    assert "byAction" in body
    assert "estimatedCalls" in body
    # No price configured in tests -> tokens only, never a fabricated figure.
    assert body["estimatedCostUsd"] is None
    assert body["costPerMillionInUsd"] == 0.0
