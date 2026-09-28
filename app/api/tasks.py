"""Task endpoints: GET /tasks/{id}, POST /tasks/{id}/cancel."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.deps import RequestIdDep, SessionDep
from app.core.agents.persona import load_persona
from app.core.orchestrator.service import (
    cancel_task,
    create_task,
    get_task,
    run_task_in_background,
    stream_task,
)
from app.db.models import AgentTask, Business
from app.schemas.tasks import AgentTask as AgentTaskSchema
from app.schemas.tasks import CancelTaskResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])


async def _business(db) -> Business:
    stmt = select(Business).order_by(Business.created_at).limit(1)
    business = (await db.execute(stmt)).scalars().first()
    if business is None:
        import uuid

        business = Business(id=f"biz_{uuid.uuid4().hex[:20]}", name="My Business")
        db.add(business)
        await db.flush()
    return business


def _to_schema(row: AgentTask) -> AgentTaskSchema:
    return AgentTaskSchema(
        id=row.id,
        conversation_id=row.conversation_id,
        task_name=row.task_name,
        persona_id=row.persona_id,
        status=row.status,
        steps=row.steps or [],
        progress=row.progress,
        incomplete=bool(row.incomplete),
        error=row.error,
        result=row.result,
        blocks=row.blocks or [],
        usage={
            "tokens_in": row.tokens_in,
            "tokens_out": row.tokens_out,
            "model": row.model or "",
            "latency_ms": 0,
        },
        request_id=row.request_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


@router.get("/{task_id}", response_model=AgentTaskSchema)
async def read_task(task_id: str, db: SessionDep) -> AgentTaskSchema:
    """Poll a long-running task. SSE uses the same shape behind /tasks/{id}/stream."""
    return _to_schema(await get_task(db, task_id))


@router.post("/{task_id}/cancel", response_model=CancelTaskResponse)
async def cancel(task_id: str, db: SessionDep) -> CancelTaskResponse:
    row = await cancel_task(db, task_id)
    return CancelTaskResponse(
        task_id=row.id, status=row.status, message="Task cancelled."
    )


@router.get("/{task_id}/stream")
async def stream(task_id: str, db: SessionDep) -> StreamingResponse:
    """SSE endpoint. Same interface as polling, so the frontend can switch."""
    generator = stream_task(db, task_id, llm=None, request_id="")
    return StreamingResponse(generator, media_type="text/event-stream")


@router.post("", response_model=AgentTaskSchema)
async def start_task(
    persona_id: str,
    task: str,
    db: SessionDep,
    request_id: RequestIdDep,
) -> AgentTaskSchema:
    """Queue a long-running task and return immediately."""
    business = await _business(db)
    persona = await load_persona(persona_id, db)
    if not persona.can_run_task(task):
        from app.core.errors import TaskNotAllowedError

        raise TaskNotAllowedError(
            f"persona '{persona_id}' may not run task '{task}'",
            detail={"allowed_tasks": persona.allowed_tasks},
        )
    row = await create_task(
        db,
        task_name=task,
        persona_id=persona_id,
        conversation_id=None,
        business_id=business.id,
        inputs={},
        request_id=request_id,
    )
    await db.commit()

    import asyncio

    from app.core.llm.router import get_llm
    from app.db.session import get_sessionmaker

    asyncio.create_task(
        run_task_in_background(
            row.id,
            db_factory=get_sessionmaker(),
            llm=get_llm("strong"),
            persona_id=persona_id,
            task_name=task,
            inputs={},
            user_message=f"Run the {task} task.",
            request_id=request_id,
        )
    )
    return _to_schema(row)


__all__ = ["cancel", "read_task", "start_task", "stream"]
