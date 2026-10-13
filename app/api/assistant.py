"""Assistant endpoints: POST /assistant/message, GET conversations."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.api.deps import FastLLMDep, LLMDep, RequestIdDep, SessionDep, WorkspaceDep
from app.core.logging import get_logger
from app.core.orchestrator.service import (
    get_conversation,
    handle_message,
    list_conversations,
)
from app.db.session import session_scope
from app.schemas.assistant import (
    AssistantMessageRequest,
    AssistantMessageResponse,
    ConversationDetail,
    ConversationSummary,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/assistant", tags=["assistant"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


async def _post_turn_memory_background(
    business_id: str,
    conversation_id: str,
    request_id: str,
) -> None:
    """Run post-turn memory compaction and LTM debounce in background outside HTTP turn lifecycle.

    Uses a dedicated session factory (session_scope()) to never compete with request connections,
    and isolates telemetry under bg_mem_<request_id>.
    """
    bg_req_id = f"bg_mem_{request_id}"
    logger.info(
        "post_turn_memory_background_started",
        extra={
            "business_id": business_id,
            "conversation_id": conversation_id,
            "request_id": bg_req_id,
        },
    )
    # 1. Background session memory compaction (if needed)
    try:
        from app.ai.memory.compressor import manage_session_memory_with_token_cap
        from app.db.models import Conversation, Message
        from sqlalchemy import select

        async with session_scope() as bg_db:
            conv = await bg_db.get(Conversation, conversation_id)
            if conv:
                msg_rows = (
                    await bg_db.execute(
                        select(Message)
                        .where(Message.conversation_id == conversation_id)
                        .order_by(Message.created_at.asc())
                    )
                ).scalars().all()
                await manage_session_memory_with_token_cap(
                    bg_db,
                    conversation=conv,
                    messages=list(msg_rows),
                )
                await bg_db.commit()
    except Exception as exc:
        logger.warning(
            "post_turn_memory_compaction_failed",
            extra={
                "business_id": business_id,
                "conversation_id": conversation_id,
                "request_id": bg_req_id,
                "error": str(exc),
            },
        )

    # 2. Debounced LTM update
    try:
        from app.ai.memory.compressor import memory_debouncer
        memory_debouncer.schedule_ltm_update(business_id)
    except Exception as exc:
        logger.warning(
            "post_turn_ltm_schedule_failed",
            extra={
                "business_id": business_id,
                "request_id": bg_req_id,
                "error": str(exc),
            },
        )

    # 3. Dedicated Brand & Dashboard Intelligence Agent (updates brand identity & profile in background on every chat)
    try:
        from app.core.brand.identity_service import brand_intelligence_agent
        async with session_scope() as bg_db:
            await brand_intelligence_agent.sync_brand_identity(bg_db, business_id)
        logger.info(
            "post_turn_brand_intelligence_synced",
            extra={
                "business_id": business_id,
                "request_id": bg_req_id,
            },
        )
    except Exception as exc:
        logger.warning(
            "post_turn_brand_intelligence_failed",
            extra={
                "business_id": business_id,
                "request_id": bg_req_id,
                "error": str(exc),
            },
        )


@router.post("/message", response_model=AssistantMessageResponse)
async def post_message(
    payload: AssistantMessageRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    llm: LLMDep,
    fast_llm: FastLLMDep,
    request_id: RequestIdDep,
    background_tasks: BackgroundTasks,
) -> AssistantMessageResponse:
    """Send a message and get the assistant's blocks back."""
    resp = await handle_message(
        db,
        payload,
        llm=llm,
        fast_llm=fast_llm,
        request_id=request_id,
        business=ws.business,
        defer_post_memory=True,
    )
    background_tasks.add_task(
        _post_turn_memory_background,
        business_id=ws.business_id,
        conversation_id=resp.conversation_id,
        request_id=request_id,
    )
    return resp


@router.post("/message/stream")
async def post_message_stream(
    payload: AssistantMessageRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    llm: LLMDep,
    fast_llm: FastLLMDep,
    request_id: RequestIdDep,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    """Send real agent/tool progress as SSE, followed by the normal final reply."""
    async def events():
        queue: asyncio.Queue[dict] = asyncio.Queue()
        work = asyncio.create_task(
            handle_message(
                db,
                payload,
                llm=llm,
                fast_llm=fast_llm,
                request_id=request_id,
                business=ws.business,
                event_queue=queue,
                defer_post_memory=True,
            )
        )
        yield _sse("progress", {"type": "started"})
        try:
            while not work.done() or not queue.empty():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    if work.done():
                        break
                    yield ": keep-alive\n\n"
                    continue
                yield _sse("progress", item)

            response = await work
            background_tasks.add_task(
                _post_turn_memory_background,
                business_id=ws.business_id,
                conversation_id=response.conversation_id,
                request_id=request_id,
            )
            yield _sse("complete", response.model_dump(mode="json", by_alias=True))
        except asyncio.CancelledError:
            work.cancel()
            try:
                await work
            except (asyncio.CancelledError, Exception):
                pass
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _sse("error", {"message": "The assistant could not complete this request."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        background=background_tasks,
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def get_conversations(
    ws: WorkspaceDep, db: SessionDep
) -> list[ConversationSummary]:
    """This workspace's conversations only."""
    return await list_conversations(db, ws.business_id)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_detail(
    conversation_id: str, ws: WorkspaceDep, db: SessionDep
) -> ConversationDetail:
    """One conversation. Another workspace's id is a 404, never a 403."""
    return await get_conversation(db, conversation_id, business_id=ws.business_id)


__all__ = ["get_conversation_detail", "get_conversations", "post_message", "post_message_stream"]
