"""Schemas for the assistant endpoints (POST /assistant/message, ...)."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.blocks import Block, Message
from app.schemas.common import ApiModel


class MessageContext(ApiModel):
    """Sticky context carried with a conversation turn."""

    campaign_id: str | None = None
    creative_id: str | None = None
    insight_id: str | None = None
    business_id: str | None = None


class AssistantMessageRequest(ApiModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=8000)
    context: MessageContext | None = None
    # Force a persona instead of letting the orchestrator classify.
    persona_id: str | None = None
    # Opt into streaming-ish long-task handling: returns a task_id immediately.
    async_mode: bool = False


class AssistantMessageResponse(ApiModel):
    message: Message
    # Present when async_mode=True or the run was long enough to be queued.
    task_id: str | None = None
    blocks: list[Block] = Field(default_factory=list)


class ConversationSummary(ApiModel):
    id: str
    title: str = ""
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    last_message_preview: str = ""


class ConversationDetail(ApiModel):
    id: str
    title: str = ""
    created_at: datetime
    updated_at: datetime
    messages: list[Message] = Field(default_factory=list)
    context: MessageContext | None = None
