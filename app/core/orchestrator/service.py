"""Conversation and long-task orchestration.

Routers call this; it holds the business logic. Keeps ``app/api/`` thin, as the
folder contract requires.

Responsibilities:
- persist conversations and messages
- run one assistant turn through the LangGraph orchestrator
- create/poll/cancel AgentTask records (queued|running|succeeded|failed|cancelled)
- stream a run's events
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.core.orchestrator.graph import OrchestratorDeps, build_graph
from app.core.orchestrator.state import initial_state
from app.core.orchestrator.blocks import validate_blocks
from app.db.models import AgentTask, Conversation, Message
from app.schemas.assistant import (
    AssistantMessageRequest,
    AssistantMessageResponse,
    ConversationDetail,
    ConversationSummary,
    MessageContext,
)
from app.schemas.blocks import Message as MessageSchema
from app.schemas.common import StepStatus, TaskStatus

_logger = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


# ------------------------------------------------------------- conversations --


async def _default_business(db: AsyncSession) -> Any:
    from app.db.models import Business

    stmt = select(Business).order_by(Business.created_at).limit(1)
    business = (await db.execute(stmt)).scalars().first()
    if business is None:
        business = Business(id=_new_id("biz"), name="My Business", workspace_id="default")
        db.add(business)
        await db.flush()
    return business


async def get_or_create_conversation(
    db: AsyncSession, conversation_id: str | None, business_id: str
) -> Conversation:
    if conversation_id:
        existing = await db.get(Conversation, conversation_id)
        if existing is not None:
            return existing
        raise NotFoundError(f"conversation '{conversation_id}' not found")
    conversation = Conversation(
        id=_new_id("conv"),
        business_id=business_id,
        title="",
        context={},
    )
    db.add(conversation)
    await db.flush()
    return conversation


async def list_conversations(db: AsyncSession, business_id: str) -> list[ConversationSummary]:
    stmt = (
        select(Conversation)
        .where(Conversation.business_id == business_id)
        .order_by(Conversation.updated_at.desc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    out: list[ConversationSummary] = []
    for row in rows:
        preview = ""
        if row.messages:
            last = row.messages[-1]
            preview = (last.content or "")[:120]
        out.append(
            ConversationSummary(
                id=row.id,
                title=row.title,
                created_at=row.created_at,
                updated_at=row.updated_at,
                message_count=len(row.messages),
                last_message_preview=preview,
            )
        )
    return out


async def get_conversation(db: AsyncSession, conversation_id: str) -> ConversationDetail:
    conversation = await db.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError(f"conversation '{conversation_id}' not found")
    messages = [
        MessageSchema(
            id=m.id,
            role=m.role,
            content=m.content or "",
            blocks=[b for b in (m.blocks or [])],
            created_at=m.created_at,
            task_id=m.task_id,
            persona_id=m.persona_id,
        )
        for m in conversation.messages
    ]
    ctx = conversation.context or {}
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=messages,
        context=MessageContext(**{k: v for k, v in ctx.items()
                                  if k in MessageContext.model_fields}),
    )


async def _persist_message(
    db: AsyncSession,
    *,
    conversation_id: str,
    role: str,
    content: str,
    blocks: list[dict[str, Any]],
    task_id: str | None = None,
    persona_id: str | None = None,
) -> Message:
    message = Message(
        id=_new_id("msg"),
        conversation_id=conversation_id,
        role=role,
        content=content,
        blocks=blocks,
        task_id=task_id,
        persona_id=persona_id,
    )
    db.add(message)
    await db.flush()
    return message


# ---------------------------------------------------------------- the turn ----


async def handle_message(
    db: AsyncSession,
    request: AssistantMessageRequest,
    *,
    llm: Any,
    request_id: str,
) -> AssistantMessageResponse:
    """Run one assistant turn synchronously."""
    business = await _default_business(db)
    conversation = await get_or_create_conversation(
        db, request.conversation_id, business.id
    )

    # Merge sticky context.
    context = dict(conversation.context or {})
    if request.context:
        context.update(
            {k: v for k, v in request.context.model_dump().items() if v is not None}
        )
    conversation.context = context

    await _persist_message(
        db,
        conversation_id=conversation.id,
        role="user",
        content=request.message,
        blocks=[{"type": "text", "text": request.message}],
    )
    if not conversation.title:
        conversation.title = request.message[:80]

    deps = OrchestratorDeps(
        db=db,
        llm=llm,
        business_id=business.id,
        workspace_id=business.workspace_id,
        request_id=request_id,
        forced_persona_id=request.persona_id,
    )
    graph = build_graph(deps)
    state = initial_state(
        conversation_id=conversation.id,
        business_id=business.id,
        workspace_id=business.workspace_id,
        user_message=request.message,
        request_id=request_id,
        context=context,
    )
    final = await graph.ainvoke(state)
    blocks = validate_blocks(list(final.get("blocks") or []))
    agent_result = final.get("agent_result") or {}

    content = ""
    for block in blocks:
        if block.get("type") == "text":
            content = str(block.get("text") or "")
            break
    if not content:
        content = str(final.get("error") or "")

    await _persist_message(
        db,
        conversation_id=conversation.id,
        role="assistant",
        content=content,
        blocks=blocks,
        task_id=final.get("task_id"),
        persona_id=str(agent_result.get("persona_id") or ""),
    )
    await db.commit()

    message = MessageSchema(
        id=_new_id("msg"),
        role="assistant",
        content=content,
        blocks=blocks,
        created_at=_now(),
        task_id=final.get("task_id"),
        persona_id=str(agent_result.get("persona_id") or ""),
    )
    return AssistantMessageResponse(message=message, task_id=final.get("task_id"), blocks=blocks)


# -------------------------------------------------------------- agent tasks ----


async def create_task(
    db: AsyncSession,
    *,
    task_name: str,
    persona_id: str,
    conversation_id: str | None,
    business_id: str,
    inputs: dict[str, Any],
    request_id: str,
) -> AgentTask:
    task = AgentTask(
        id=_new_id("task"),
        business_id=business_id,
        conversation_id=conversation_id,
        task_name=task_name,
        persona_id=persona_id,
        status=TaskStatus.QUEUED.value,
        steps=[
            {"name": "queued", "status": StepStatus.PENDING.value, "detail": "waiting to start"}
        ],
        inputs=inputs,
        request_id=request_id,
    )
    db.add(task)
    await db.flush()
    return task


async def get_task(db: AsyncSession, task_id: str) -> AgentTask:
    task = await db.get(AgentTask, task_id)
    if task is None:
        raise NotFoundError(f"task '{task_id}' not found")
    return task


async def cancel_task(db: AsyncSession, task_id: str) -> AgentTask:
    task = await get_task(db, task_id)
    if task.status in (TaskStatus.SUCCEEDED.value, TaskStatus.FAILED.value,
                       TaskStatus.CANCELLED.value):
        # Already terminal: cancelling is a no-op, not an error.
        return task
    task.status = TaskStatus.CANCELLED.value
    task.finished_at = _now()
    task.error = task.error or "cancelled by user"
    await db.commit()
    return task


async def run_task_in_background(
    task_id: str,
    *,
    db_factory: Any,
    llm: Any,
    persona_id: str,
    task_name: str,
    inputs: dict[str, Any],
    user_message: str,
    request_id: str,
) -> None:
    """Execute a queued AgentTask. Called from a background asyncio task."""
    async with db_factory() as db:
        task = await db.get(AgentTask, task_id)
        if task is None or task.status == TaskStatus.CANCELLED.value:
            return
        task.status = TaskStatus.RUNNING.value
        task.started_at = _now()
        task.steps = [{"name": "running", "status": StepStatus.RUNNING.value}]
        await db.flush()
        await db.commit()

        try:
            from app.core.agents.persona import load_persona
            from app.core.agents.runtime import run_agent
            from app.core.tools.base import ToolContext

            persona = await load_persona(persona_id, db, task.workspace_id or "default")
            ctx = ToolContext(
                workspace_id="default",
                business_id=task.business_id or "",
                db=db,
                llm=llm,
                request_id=request_id,
                task_id=task.id,
                persona_id=persona.id,
            )
            result = await run_agent(
                persona, task_name, inputs, ctx, llm=llm, user_message=user_message
            )
            task.status = (
                TaskStatus.SUCCEEDED.value
                if not result.error and not result.incomplete
                else TaskStatus.FAILED.value
            )
            task.result = result.model_dump(mode="json")
            task.incomplete = result.incomplete
            task.error = result.error or result.incomplete_reason
            task.tokens_in = result.usage.tokens_in
            task.tokens_out = result.usage.tokens_out
            task.model = result.usage.model
            task.progress = 1.0
            task.finished_at = _now()
            await db.commit()
        except Exception as exc:
            _logger.warning("background_task_failed", extra={"task_id": task_id, "error": str(exc)})
            task.status = TaskStatus.FAILED.value
            task.error = str(exc)
            task.finished_at = _now()
            await db.commit()


async def stream_task(
    db: AsyncSession,
    task_id: str,
    *,
    llm: Any,
    request_id: str,
) -> Any:
    """Yield SSE-formatted events for a running task.

    Same interface the polling endpoint uses, so switching the frontend from
    polling to SSE is a transport change only.
    """
    task = await get_task(db, task_id)
    yield _sse("task_progress", {"task_id": task.id, "status": task.status,
                                 "progress": task.progress})
    yield _sse("done", {"task_id": task.id, "status": task.status})


def _sse(event: str, data: dict[str, Any]) -> str:
    import json

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


__all__ = [
    "cancel_task",
    "create_task",
    "get_conversation",
    "get_task",
    "handle_message",
    "list_conversations",
    "run_task_in_background",
    "stream_task",
]
