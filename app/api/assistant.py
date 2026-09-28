"""Assistant endpoints: POST /assistant/message, GET conversations."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import LLMDep, RequestIdDep, SessionDep
from app.core.orchestrator.service import (
    get_conversation,
    handle_message,
    list_conversations,
)
from app.db.models import Business
from app.schemas.assistant import (
    AssistantMessageRequest,
    AssistantMessageResponse,
    ConversationDetail,
    ConversationSummary,
)
from sqlalchemy import select

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/message", response_model=AssistantMessageResponse)
async def post_message(
    payload: AssistantMessageRequest,
    db: SessionDep,
    llm: LLMDep,
    request_id: RequestIdDep,
) -> AssistantMessageResponse:
    """Send a message and get the assistant's blocks back."""
    return await handle_message(db, payload, llm=llm, request_id=request_id)


@router.get("/conversations", response_model=list[ConversationSummary])
async def get_conversations(db: SessionDep) -> list[ConversationSummary]:
    stmt = select(Business).order_by(Business.created_at).limit(1)
    business = (await db.execute(stmt)).scalars().first()
    if business is None:
        return []
    return await list_conversations(db, business.id)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_detail(
    conversation_id: str, db: SessionDep
) -> ConversationDetail:
    return await get_conversation(db, conversation_id)
