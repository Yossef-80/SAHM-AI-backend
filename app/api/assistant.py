"""Assistant endpoints: POST /assistant/message, GET conversations."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, Response
from fastapi.responses import StreamingResponse
from pydantic import Field

from app.ai.tools.ad_previews import AdPreviewNotFoundError, read_ad_preview
from app.api.deps import (
    CurrentUserDep,
    FastLLMDep,
    LLMDep,
    RequestIdDep,
    SessionDep,
    SettingsDep,
    WorkspaceDep,
)
from app.core.chat_attachments import store_chat_attachment
from app.core.errors import NotFoundError, ValidationFailed
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
from app.schemas.common import ApiModel

logger = get_logger(__name__)

router = APIRouter(prefix="/assistant", tags=["assistant"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


async def _post_turn_memory_background(
    business_id: str,
    conversation_id: str,
    request_id: str,
) -> None:
    """Hand the turn's memory work to the single per-workspace consolidator.

    Before P2 this function started three independent background passes
    (session compaction, the debounced LTM rewrite and the brand-identity
    extraction) that could interleave and write the same profile row from
    stale reads -- see ``app/ai/memory/consolidator.py``. The consolidator
    serialises them per business, debounces the two LLM passes, and skips the
    identity pass when it ran recently.
    """
    bg_req_id = f"bg_mem_{request_id}"
    try:
        from app.ai.memory.consolidator import memory_consolidator

        memory_consolidator.schedule(
            business_id=business_id,
            conversation_id=conversation_id,
            request_id=bg_req_id,
        )
    except Exception as exc:  # noqa: BLE001 - memory must never fail a turn
        logger.warning(
            "post_turn_memory_schedule_failed",
            extra={
                "business_id": business_id,
                "conversation_id": conversation_id,
                "request_id": bg_req_id,
                "error_type": type(exc).__name__,
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


class ChatAttachmentRequest(ApiModel):
    """One image the user attached to the message they are about to send."""

    image_data: str = Field(min_length=16, max_length=12_000_000)
    source_name: str | None = Field(default=None, max_length=160)


class ChatAttachmentResponse(ApiModel):
    id: str
    name: str
    mime: str
    created_at: str | None = None


@router.post("/attachments", response_model=ChatAttachmentResponse)
async def upload_chat_attachment(
    payload: ChatAttachmentRequest,
    ws: WorkspaceDep,
    db: SessionDep,
) -> ChatAttachmentResponse:
    """Store a chat image and return the id to send with the message.

    Deliberately *not* ``/creatives/analyze-reference``: attaching a photo to a
    chat is not a request to run Layla's creative workflow, and the old behaviour
    (analyse on upload, then switch the composer into creative mode) is exactly
    what users complained about. Analysis stays available as an explicit action.
    """
    if not payload.image_data.strip().startswith("data:image/"):
        raise ValidationFailed("attachment must be an inline image data URL")
    stored = await store_chat_attachment(
        db,
        business_id=ws.business_id,
        image_data=payload.image_data,
        source_name=payload.source_name,
    )
    return ChatAttachmentResponse(**stored)


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


@router.get("/ad-previews/{preview_id}")
async def get_ad_preview(
    preview_id: str,
    ws: WorkspaceDep,
    settings: SettingsDep,
    _user: CurrentUserDep,
) -> Response:
    """Return stored image bytes for an ad preview within the workspace scope."""
    try:
        data, mime = read_ad_preview(
            upload_dir=settings.upload_dir,
            business_id=ws.business_id,
            preview_id=preview_id,
        )
    except AdPreviewNotFoundError as exc:
        raise NotFoundError("ad preview not found") from exc
    return Response(
        content=data,
        media_type=mime,
        headers={
            "Cache-Control": "private, max-age=60",
            "X-Content-Type-Options": "nosniff",
        },
    )


__all__ = [
    "get_ad_preview",
    "get_conversation_detail",
    "get_conversations",
    "post_message",
    "post_message_stream",
]
