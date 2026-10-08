"""Assistant endpoints: POST /assistant/message, GET conversations."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import FastLLMDep, LLMDep, RequestIdDep, SessionDep, WorkspaceDep
from app.core.orchestrator.service import (
    get_conversation,
    handle_message,
    list_conversations,
)
from app.schemas.assistant import (
    AssistantMessageRequest,
    AssistantMessageResponse,
    ConversationDetail,
    ConversationSummary,
)

router = APIRouter(prefix="/assistant", tags=["assistant"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post("/message", response_model=AssistantMessageResponse)
async def post_message(
    payload: AssistantMessageRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    llm: LLMDep,
    fast_llm: FastLLMDep,
    request_id: RequestIdDep,
) -> AssistantMessageResponse:
    """Send a message and get the assistant's blocks back."""
    return await handle_message(
        db,
        payload,
        llm=llm,
        fast_llm=fast_llm,
        request_id=request_id,
        business=ws.business,
    )


@router.post("/message/stream")
async def post_message_stream(
    payload: AssistantMessageRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    llm: LLMDep,
    fast_llm: FastLLMDep,
    request_id: RequestIdDep,
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
