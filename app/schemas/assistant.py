"""Schemas for the assistant endpoints (POST /assistant/message, ...)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.blocks import Block, Message
from app.schemas.common import ApiModel


class MessageContext(ApiModel):
    """Sticky context carried with a conversation turn."""

    campaign_id: str | None = None
    creative_id: str | None = None
    insight_id: str | None = None
    reference_analysis_id: str | None = None
    workflow: Literal["creative_prompt"] | None = None
    prompt_type: Literal["image", "video"] | None = None
    # Explicit UI choice of what this turn should produce. When present it
    # overrides the keyword guess in core_bridge._validate_explicit_intent, so
    # the user no longer has to phrase a message to match a regex to get an
    # image. `chat` is a real value, not an absence: it keeps the regex fallback.
    creative_mode: Literal["chat", "prompt", "design"] | None = None
    persona_id: str | None = None
    business_id: str | None = None


class AssistantAnswer(ApiModel):
    """A validated selection from one of this conversation's question blocks."""

    message_id: str
    option_ids: list[str] = Field(default_factory=list, min_length=1, max_length=12)


class AssistantMessageRequest(ApiModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=8000)
    context: MessageContext | None = None
    # Force a persona instead of letting the orchestrator classify.
    persona_id: str | None = None
    # Selection from a question_with_options block (e.g. Layla plan approval).
    answer: AssistantAnswer | None = None
    # Opt into streaming-ish long-task handling: returns a task_id immediately.
    async_mode: bool = False


class AssistantMessageResponse(ApiModel):
    message: Message
    # Persisted user row, additive for clients that need stable IDs across reloads.
    user_message: Message | None = None
    # Present when async_mode=True or the run was long enough to be queued.
    task_id: str | None = None
    blocks: list[Block] = Field(default_factory=list)
    #: Always set. The client needs it to navigate to /assistant/{id} when the
    #: turn started a brand-new conversation.
    conversation_id: str | None = None


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
