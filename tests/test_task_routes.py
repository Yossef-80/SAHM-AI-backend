"""Task/chat separation, durable task results, and response-contract regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.llm.mock_client import MockClient
from app.core.orchestrator.blocks import validate_block, validate_blocks
from app.core.orchestrator.service import create_task, run_task_in_background
from app.db.models import AgentTask, AgentToolCall, Conversation, Message
from app.schemas.blocks import Message as MessageSchema
from app.schemas.tasks import AgentTask as AgentTaskSchema

PERSONA_TASKS = [
    ("researcher", "competitor_research"),
    ("omar", "marketing_strategy"),
    ("karim", "performance_analysis"),
    ("salma", "optimization_proposal"),
]


def test_shared_task_wire_fixture_matches_backend_pydantic_contract() -> None:
    fixture = Path(__file__).parent / "fixtures" / "task_response.json"
    wire_task = json.loads(fixture.read_text(encoding="utf-8"))
    parsed = AgentTaskSchema.model_validate(wire_task)
    assert parsed.id == "task_contract"
    assert parsed.task_name == "ad_variations"
    assert parsed.result["output"]["variations"]


@pytest.mark.parametrize(("persona_id", "task_name"), PERSONA_TASKS)
async def test_tasks_route_accepts_the_supported_agent_task_for_each_persona(
    persona_id: str, task_name: str, api_client, monkeypatch
) -> None:
    async def do_not_run_in_background(*_args, **_kwargs) -> None:
        return None

    import app.api.tasks as tasks_api

    monkeypatch.setattr(tasks_api, "run_task_in_background", do_not_run_in_background)
    response = await api_client.post(
        "/api/v1/tasks",
        params={"persona_id": persona_id, "task": task_name},
        json={"inputs": {"offline_fixture": True}},
    )
    assert response.status_code == 200, response.text
    task = AgentTaskSchema.model_validate(response.json())
    assert task.persona_id == persona_id
    assert task.task_name == task_name
    assert task.status.value == "queued"


async def test_layla_direct_text_tasks_are_redirected_away_from_legacy_core_tasks(api_client) -> None:
    response = await api_client.post(
        "/api/v1/tasks",
        params={"persona_id": "layla", "task": "ad_variations"},
        json={"inputs": {"brief": "Write three distinct ad angles"}},
    )
    assert response.status_code == 403
    assert "/creatives/generate-text" in response.text


async def test_tasks_post_is_separate_and_persists_conversation_progress(
    api_client, seeded_db, monkeypatch
) -> None:
    conversation = Conversation(
        id="conv_action_task",
        business_id="biz_test",
        title="Action-card task",
        context={},
    )
    seeded_db.add(conversation)
    await seeded_db.commit()

    async def do_not_run_in_background(*_args, **_kwargs) -> None:
        return None

    import app.api.tasks as tasks_api

    monkeypatch.setattr(tasks_api, "run_task_in_background", do_not_run_in_background)

    response = await api_client.post(
        "/api/v1/tasks",
        params={"persona_id": "omar", "task": "marketing_strategy"},
        json={
            "inputs": {
                "brief": "A strategy for a coffee subscription launch",
                "count": 3,
                "campaignId": "cmp_action",
                "creativeId": "cr_action",
                "conversationId": "must-not-be-task-input",
                "apiKey": "not-a-real-secret",
                "nested": {"note": "keep this", "metaAccessToken": "not-a-real-token"},
            },
            "conversationId": conversation.id,
        },
    )
    assert response.status_code == 200, response.text
    wire_task = response.json()
    task_schema = AgentTaskSchema.model_validate(wire_task)
    assert task_schema.status.value == "queued"
    assert wire_task["conversationId"] == conversation.id
    assert wire_task["personaId"] == "omar"
    assert wire_task["taskName"] == "marketing_strategy"
    assert wire_task["result"] is None

    task_row = await seeded_db.get(AgentTask, wire_task["id"])
    assert task_row is not None
    assert task_row.inputs == {
        "brief": "A strategy for a coffee subscription launch",
        "count": 3,
        "campaign_id": "cmp_action",
        "creative_id": "cr_action",
        "nested": {"note": "keep this"},
    }

    messages = (
        await seeded_db.execute(
            select(Message).where(Message.conversation_id == conversation.id)
        )
    ).scalars().all()
    assert len(messages) == 1  # /tasks did not create a chat user turn
    assert messages[0].role == "assistant"
    assert messages[0].task_id == task_row.id
    assert messages[0].persona_id == "omar"
    assert messages[0].blocks[0]["type"] == "task_progress"
    assert validate_block(messages[0].blocks[0]).task_id == task_row.id

    conversation_response = await api_client.get(
        f"/api/v1/assistant/conversations/{conversation.id}"
    )
    assert conversation_response.status_code == 200
    detail = conversation_response.json()
    assert len(detail["messages"]) == 1
    persisted_message = MessageSchema.model_validate(detail["messages"][0])
    assert persisted_message.id == messages[0].id
    assert persisted_message.task_id == task_row.id
    assert persisted_message.persona_id == "omar"


async def test_legacy_queued_layla_task_fails_closed_without_core_llm_or_tool_context(
    api_client, db_engine, seeded_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.agents import persona, runtime
    from app.core.tools import base as tool_base

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy Layla tasks must not enter the core agent runtime")

    monkeypatch.setattr(persona, "load_persona", forbidden)
    monkeypatch.setattr(runtime, "run_agent", forbidden)
    monkeypatch.setattr(tool_base, "ToolContext", forbidden)

    conversation = Conversation(
        id="conv_task_worker",
        business_id="biz_test",
        title="Legacy Layla task",
        context={},
    )
    seeded_db.add(conversation)
    await seeded_db.commit()

    inputs = {"brief": "Write three warm coffee subscription ad variations", "count": 3}
    task = await create_task(
        seeded_db,
        task_name="ad_variations",
        persona_id="layla",
        conversation_id=conversation.id,
        business_id="biz_test",
        inputs=inputs,
        request_id="req-task-worker",
    )
    await seeded_db.commit()

    maker = async_sessionmaker(db_engine, expire_on_commit=False)
    await run_task_in_background(
        task.id,
        db_factory=maker,
        llm=MockClient(model="mock-strong-1"),
        persona_id="layla",
        task_name="ad_variations",
        inputs=inputs,
        user_message="Create variations from the action-card brief.",
        request_id="req-task-worker",
    )

    task_row = await seeded_db.get(AgentTask, task.id)
    assert task_row is not None
    await seeded_db.refresh(task_row)
    assert task_row.status == "failed"
    assert "/creatives/generate-text" in (task_row.error or "")
    assert task_row.progress == 1.0
    assert task_row.finished_at is not None
    assert task_row.tokens_in == 0 and task_row.tokens_out == 0
    assert task_row.result is None
    assert any(step["status"] == "failed" for step in task_row.steps)
    assert task_row.blocks
    assert validate_blocks(task_row.blocks) == task_row.blocks

    calls = (
        await seeded_db.execute(
            select(AgentToolCall).where(AgentToolCall.task_id == task.id)
        )
    ).scalars().all()
    assert calls == []

    response = await api_client.get(f"/api/v1/tasks/{task.id}")
    assert response.status_code == 200, response.text
    wire_task = response.json()
    accepted = AgentTaskSchema.model_validate(wire_task)
    assert accepted.status.value == "failed"
    assert "/creatives/generate-text" in wire_task["error"]
    assert wire_task["usage"]["tokensIn"] == 0
    assert wire_task["blocks"]
    assert validate_blocks(wire_task["blocks"]) == wire_task["blocks"]


async def test_assistant_message_returns_the_persisted_user_and_assistant_ids(
    api_client, monkeypatch
) -> None:
    from app.ai.agents.orchestrator import brain
    from app.schemas.blocks import TextBlock

    async def fake_orchestrate(*_args, **_kwargs):
        return brain.OrchestratorResult(
            blocks=[TextBlock(text="Mocked assistant response.")],
            persona_id="omar",
        )

    monkeypatch.setattr(brain, "orchestrate", fake_orchestrate)
    response = await api_client.post(
        "/api/v1/assistant/message",
        json={"message": "A short saved chat turn"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["userMessage"]["id"]
    assert body["message"]["id"]
    assert body["userMessage"]["id"] != body["message"]["id"]
    assert body["userMessage"]["role"] == "user"
    assert body["message"]["role"] == "assistant"
    assert body["message"]["personaId"] == "omar"

    conversation_id = body["conversationId"]
    detail = await api_client.get(f"/api/v1/assistant/conversations/{conversation_id}")
    assert detail.status_code == 200
    ids = [message["id"] for message in detail.json()["messages"]]
    assert body["userMessage"]["id"] in ids
    assert body["message"]["id"] in ids


async def test_persona_choice_stays_sticky_for_the_whole_conversation(
    api_client, monkeypatch
) -> None:
    """Picking an agent once must pin the conversation to it on later turns.

    Regression for "asking something from the agents list should open a chat that
    keeps talking to the same agent, like the analysis agent does": the composer
    only sends an explicit personaId, and the backend has to remember it so the
    next message does not silently fall back to automatic routing.
    """
    from app.ai.agents.orchestrator import brain
    from app.schemas.blocks import TextBlock

    forced: list[str | None] = []

    async def fake_orchestrate(*_args, **kwargs):
        forced.append(kwargs.get("forced_agent"))
        return brain.OrchestratorResult(
            blocks=[TextBlock(text="Mocked assistant response.")],
            persona_id=kwargs.get("forced_agent") or "omar",
        )

    monkeypatch.setattr(brain, "orchestrate", fake_orchestrate)

    first = await api_client.post(
        "/api/v1/assistant/message",
        json={"message": "Talk to Layla about the creative", "personaId": "layla"},
    )
    assert first.status_code == 200, first.text
    conversation_id = first.json()["conversationId"]

    detail = await api_client.get(f"/api/v1/assistant/conversations/{conversation_id}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["context"]["personaId"] == "layla"

    # Second turn with no explicit persona: it must still be forced to Layla.
    second = await api_client.post(
        "/api/v1/assistant/message",
        json={"conversationId": conversation_id, "message": "And make it night time"},
    )
    assert second.status_code == 200, second.text
    assert forced == ["layla", "layla"]
