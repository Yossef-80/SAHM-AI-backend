"""Task endpoints: GET /tasks/{id}, POST /tasks/{id}/cancel."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import RequestIdDep, SessionDep, SettingsDep, WorkspaceDep
from app.core.agents.persona import load_persona
from app.core.orchestrator.service import (
    cancel_task,
    create_task,
    get_task,
    run_task_in_background,
    stream_task,
)
from app.db.models import AgentTask
from app.schemas.tasks import AgentTask as AgentTaskSchema
from app.schemas.tasks import CancelTaskResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])


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
async def read_task(
    task_id: str, ws: WorkspaceDep, db: SessionDep
) -> AgentTaskSchema:
    """Poll a long-running task. SSE uses the same shape behind /tasks/{id}/stream."""
    return _to_schema(await get_task(db, task_id, business_id=ws.business_id))


@router.post("/{task_id}/cancel", response_model=CancelTaskResponse)
async def cancel(
    task_id: str, ws: WorkspaceDep, db: SessionDep
) -> CancelTaskResponse:
    row = await cancel_task(db, task_id, business_id=ws.business_id)
    return CancelTaskResponse(
        task_id=row.id, status=row.status, message="Task cancelled."
    )


@router.get("/{task_id}/stream")
async def stream(
    task_id: str, ws: WorkspaceDep, db: SessionDep
) -> StreamingResponse:
    """SSE endpoint. Same interface as polling, so the frontend can switch.

    The task is loaded (and ownership-checked) here, not inside the generator:
    once the first chunk is sent the status line is fixed, so a 404 raised
    mid-stream would surface as a broken connection instead of a clean answer.
    """
    task = await get_task(db, task_id, business_id=ws.business_id)
    return StreamingResponse(stream_task(task), media_type="text/event-stream")


@router.post("", response_model=AgentTaskSchema)
async def start_task(
    persona_id: str,
    task: str,
    ws: WorkspaceDep,
    db: SessionDep,
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> AgentTaskSchema:
    """Queue a long-running task and return immediately."""
    persona = await load_persona(persona_id, db, ws.workspace_id)
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
        business_id=ws.business_id,
        inputs={},
        request_id=request_id,
    )
    await db.commit()

    import asyncio

    from app.core.llm.router import get_workspace_llm
    from app.db.session import get_sessionmaker

    llm = await get_workspace_llm(
        "strong", db=db, business_id=ws.business_id, settings=settings
    )
    asyncio.create_task(
        run_task_in_background(
            row.id,
            db_factory=get_sessionmaker(),
            llm=llm,
            persona_id=persona_id,
            task_name=task,
            inputs={},
            user_message=f"Run the {task} task.",
            request_id=request_id,
        )
    )
    return _to_schema(row)


__all__ = ["cancel", "read_task", "start_task", "stream"]
