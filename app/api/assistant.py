"""Assistant endpoints: POST /assistant/message, GET conversations."""

from __future__ import annotations

import asyncio
import json
import time
from functools import partial

from fastapi import APIRouter, BackgroundTasks, Query, Response
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
    DEFAULT_MESSAGE_PAGE_SIZE,
    MAX_CONVERSATIONS_LISTED,
    MAX_MESSAGE_PAGE_SIZE,
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


def _slow_turn_message(language: str | None) -> str:
    """The sentence the user reads when a turn outlives the stream."""
    if (language or "").lower().startswith("ar"):
        return (
            "الرد واخد وقت أطول من المتوقع، فسِبت الشاشة تكمّل شغلها في الخلفية. "
            "افتح المحادثة تاني بعد شوية وهتلاقي الرد محفوظ."
        )
    return (
        "This reply is taking longer than expected, so the turn continues in the "
        "background. Reopen the conversation in a moment and the answer will be there."
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


#: Conversations and businesses with a background memory job already running.
#: Chatty usage used to enqueue one compaction and one brand-identity sync per
#: turn, and each of them re-read the whole conversation or re-ran the same
#: analysis as the turn before. One job at a time per key is the fix.
_memory_jobs_in_flight: set[str] = set()
_brand_sync_in_flight: set[str] = set()

#: Messages the background compactor is willing to read per run.
_MAX_BACKGROUND_MESSAGES = 400

#: How often the SSE stream proves it is still alive while the turn runs.
_HEARTBEAT_EVERY_SECONDS = 15

#: After this long, the browser gets a *named* event saying the turn is still
#: running. A bare ``: keep-alive`` comment keeps proxies happy but is invisible
#: in the UI, so a slow model call looks like a frozen chat -- which is exactly
#: how it gets reported ("علق عند الorchestrator").
_STILL_WORKING_AFTER_SECONDS = 45

#: A turn is never allowed to hold the stream open forever. The work is
#: deliberately NOT cancelled (a tool may be mid-publish); the reply still lands
#: in the conversation, so a refresh shows it.
_STREAM_GIVE_UP_SECONDS = 600


def _late_completion_memory(task: "asyncio.Task", business_id: str, request_id: str) -> None:
    """A turn that outlived its stream still deserves its memory pass."""
    try:
        response = task.result()
    except (asyncio.CancelledError, Exception):  # noqa: BLE001 - a dead turn has nothing to save
        return
    conversation_id = getattr(response, "conversation_id", None)
    if not conversation_id:
        return
    logger.info(
        "assistant_stream_late_completion",
        extra={"conversation_id": conversation_id, "request_id": request_id},
    )
    asyncio.ensure_future(
        _post_turn_memory_background(
            business_id=business_id,
            conversation_id=conversation_id,
            request_id=request_id,
        )
    )


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
    job_key = f"conv:{conversation_id}"
    if job_key in _memory_jobs_in_flight:
        logger.info(
            "post_turn_memory_compaction_skipped",
            extra={"conversation_id": conversation_id, "request_id": bg_req_id, "reason": "already_running"},
        )
    else:
        _memory_jobs_in_flight.add(job_key)
        try:
            from app.ai.memory.compressor import manage_session_memory_with_token_cap
            from app.db.models import Conversation, Message
            from sqlalchemy import select

            async with session_scope() as bg_db:
                conv = await bg_db.get(Conversation, conversation_id)
                if conv:
                    # Newest N, then restored to chronological order: the
                    # compactor only needs the tail plus its own summary.
                    msg_rows = (
                        await bg_db.execute(
                            select(Message)
                            .where(Message.conversation_id == conversation_id)
                            .order_by(Message.created_at.desc())
                            .limit(_MAX_BACKGROUND_MESSAGES)
                        )
                    ).scalars().all()
                    await manage_session_memory_with_token_cap(
                        bg_db,
                        conversation=conv,
                        messages=list(reversed(msg_rows)),
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
        finally:
            _memory_jobs_in_flight.discard(job_key)

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
    sync_key = f"biz:{business_id}"
    if sync_key in _brand_sync_in_flight:
        logger.info(
            "post_turn_brand_intelligence_skipped",
            extra={"business_id": business_id, "request_id": bg_req_id, "reason": "already_running"},
        )
        return
    _brand_sync_in_flight.add(sync_key)
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
    finally:
        _brand_sync_in_flight.discard(sync_key)


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
        started = time.monotonic()
        finished = False
        try:
            while not work.done() or not queue.empty():
                # Wait on the turn *and* the queue together. Waiting on the queue
                # alone made every reply that did not push a progress item wait out
                # the whole 15s heartbeat before "complete" could be sent -- a
                # finished answer that still looked hung.
                getter: asyncio.Future = asyncio.ensure_future(queue.get())
                try:
                    await asyncio.wait(
                        {getter, work},
                        timeout=_HEARTBEAT_EVERY_SECONDS,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except asyncio.CancelledError:
                    getter.cancel()
                    raise
                if getter.done():
                    item = getter.result()
                    yield _sse("progress", item)
                    continue
                getter.cancel()
                if work.done() and queue.empty():
                    break
                if not queue.empty():
                    continue
                elapsed = int(time.monotonic() - started)
                yield ": keep-alive\n\n"
                if elapsed >= _STREAM_GIVE_UP_SECONDS:
                    logger.warning(
                        "assistant_stream_gave_up",
                        extra={
                            "elapsed_seconds": elapsed,
                            "request_id": request_id,
                        },
                    )
                    # Keep the memory pass for whenever the turn really ends.
                    work.add_done_callback(
                        partial(
                            _late_completion_memory,
                            business_id=ws.business_id,
                            request_id=request_id,
                        )
                    )
                    yield _sse("error", {"message": _slow_turn_message(payload.language)})
                    return
                if elapsed >= _STILL_WORKING_AFTER_SECONDS:
                    yield _sse("progress", {"type": "still_working", "seconds": elapsed})

            response = await work
            background_tasks.add_task(
                _post_turn_memory_background,
                business_id=ws.business_id,
                conversation_id=response.conversation_id,
                request_id=request_id,
            )
            yield _sse("complete", response.model_dump(mode="json", by_alias=True))
            finished = True
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
            logger.warning(
                "assistant_stream_failed",
                extra={
                    "request_id": request_id,
                    "error_type": type(e).__name__,
                    "error": str(e)[:400],
                },
            )
            yield _sse("error", {"message": "The assistant could not complete this request."})
        finally:
            # One line that answers "did the stream end, or did the browser give
            # up?" -- the access log cannot, because it is written at headers time.
            logger.info(
                "assistant_stream_closed",
                extra={
                    "request_id": request_id,
                    "terminal_event_sent": finished,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        background=background_tasks,
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def get_conversations(
    ws: WorkspaceDep,
    db: SessionDep,
    limit: int = Query(
        MAX_CONVERSATIONS_LISTED,
        ge=1,
        le=MAX_CONVERSATIONS_LISTED,
        description="Maximum conversations to return, newest first.",
    ),
) -> list[ConversationSummary]:
    """This workspace's conversations only."""
    return await list_conversations(db, ws.business_id, limit=limit)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_detail(
    conversation_id: str,
    ws: WorkspaceDep,
    db: SessionDep,
    limit: int = Query(
        DEFAULT_MESSAGE_PAGE_SIZE,
        ge=1,
        le=MAX_MESSAGE_PAGE_SIZE,
        description="Newest N messages to return (chronological order).",
    ),
    before: str | None = Query(
        None,
        description="Return messages older than this message id (next page cursor).",
    ),
) -> ConversationDetail:
    """One conversation. Another workspace's id is a 404, never a 403."""
    return await get_conversation(
        db,
        conversation_id,
        business_id=ws.business_id,
        limit=limit,
        before=before,
    )


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
