"""The ``Message.blocks`` discriminated union.

This mirrors the frontend's Zod schema for assistant message blocks. Every
block the orchestrator emits is validated against this union before it leaves
the API, so an unknown or malformed result degrades to a ``text`` or ``error``
block instead of a 500.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator, Field, Tag, model_validator

from app.schemas.common import (
    ApiModel,
    Autonomy,
    BlockKind,
    Evidence,
    Option,
    RecommendationStatus,
    TaskStatus,
    TaskStep,
    Tone,
)


class BaseBlock(ApiModel):
    """Common fields. ``type`` is the discriminator on the wire."""

    id: str | None = None


# ------------------------------------------------------------------- text ----


class TextBlock(BaseBlock):
    type: Literal[BlockKind.TEXT] = BlockKind.TEXT
    text: str = Field(min_length=1)


# --------------------------------------------------------------- findings ----


class Finding(ApiModel):
    title: str
    detail: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class FindingsBlock(BaseBlock):
    type: Literal[BlockKind.FINDINGS] = BlockKind.FINDINGS
    title: str = "Findings"
    findings: list[Finding] = Field(default_factory=list)


# ---------------------------------------------------------- recommendation ----


class RecommendationBlock(BaseBlock):
    type: Literal[BlockKind.RECOMMENDATION] = BlockKind.RECOMMENDATION
    recommendation_id: str
    action_type: str
    title: str
    rationale: str = ""
    payload: dict[str, object] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    status: RecommendationStatus = RecommendationStatus.PENDING
    # Set when execution is gated on human approval.
    requires_approval: bool = False
    estimated_impact: str | None = None


# ----------------------------------------------------------------- metric ----


class MetricBlock(BaseBlock):
    type: Literal[BlockKind.METRIC] = BlockKind.METRIC
    label: str
    value: float
    unit: str | None = None
    delta: float | None = None
    delta_label: str | None = None
    trend: Literal["up", "down", "flat", "unknown"] = "unknown"
    comparison_period: str | None = None


# ------------------------------------------------------------ action card ----


class ActionCard(BaseBlock):
    """A clickable follow-up. Agents never call each other; the user clicks."""

    type: Literal[BlockKind.ACTION_CARD] = BlockKind.ACTION_CARD
    action_type: str = Field(description="e.g. 'run_task', 'open_url', 'approve'")
    label: str
    description: str | None = None
    target_persona: str | None = None
    task: str | None = None
    inputs: dict[str, object] = Field(default_factory=dict)
    payload: dict[str, object] = Field(default_factory=dict)


# ----------------------------------------------------------- task progress ----


class TaskProgressBlock(BaseBlock):
    type: Literal[BlockKind.TASK_PROGRESS] = BlockKind.TASK_PROGRESS
    task_id: str
    task_name: str
    persona_id: str
    status: TaskStatus = TaskStatus.QUEUED
    steps: list[TaskStep] = Field(default_factory=list)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    incomplete: bool = False
    message: str | None = None


# ------------------------------------------------------------ context used ----


class ContextItem(ApiModel):
    kind: str = Field(description="memory item kind or 'brand_profile'")
    label: str
    detail: str = ""
    similarity: float | None = None
    memory_id: str | None = None


class ContextUsedBlock(BaseBlock):
    type: Literal[BlockKind.CONTEXT_USED] = BlockKind.CONTEXT_USED
    items: list[ContextItem] = Field(default_factory=list)
    token_estimate: int = 0
    truncated: bool = False


# ---------------------------------------------------- question with options --


class QuestionWithOptionsBlock(BaseBlock):
    type: Literal[BlockKind.QUESTION_WITH_OPTIONS] = BlockKind.QUESTION_WITH_OPTIONS
    question: str
    options: list[Option] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    allow_free_text: bool = True


# ------------------------------------------------------------------ error ----


class ErrorBlock(BaseBlock):
    type: Literal[BlockKind.ERROR] = BlockKind.ERROR
    message: str
    code: str | None = None
    retryable: bool = False
    unavailable_sources: list[str] = Field(default_factory=list)


# ------------------------------------------------------- streaming events ----


class AgentEventData(ApiModel):
    """Discriminated union of streaming events for TaskProgress + text."""

    type: Literal[
        "step_started",
        "tool_called",
        "tool_result",
        "token",
        "completed",
        "failed",
    ]
    step: str | None = None
    detail: str | None = None
    token: str | None = None
    data: dict[str, object] | None = None


# ------------------------------------------------------------ message model --


class Message(ApiModel):
    id: str
    role: str = Field(description="user | assistant | system")
    content: str = ""
    #: The discriminated union. Must be ``Block`` (not BaseBlock) or the
    #: discriminator is lost on serialisation and the frontend cannot render.
    blocks: list[Block] = Field(default_factory=list)
    created_at: datetime
    task_id: str | None = None
    persona_id: str | None = None


def _block_discriminator(v: object) -> str:
    if isinstance(v, BaseBlock):
        return str(v.type.value)
    if isinstance(v, dict):
        return str(v.get("type", ""))
    return ""


Block = Annotated[
    Union[
        Annotated[TextBlock, Tag(BlockKind.TEXT.value)],
        Annotated[FindingsBlock, Tag(BlockKind.FINDINGS.value)],
        Annotated[RecommendationBlock, Tag(BlockKind.RECOMMENDATION.value)],
        Annotated[MetricBlock, Tag(BlockKind.METRIC.value)],
        Annotated[ActionCard, Tag(BlockKind.ACTION_CARD.value)],
        Annotated[TaskProgressBlock, Tag(BlockKind.TASK_PROGRESS.value)],
        Annotated[ContextUsedBlock, Tag(BlockKind.CONTEXT_USED.value)],
        Annotated[QuestionWithOptionsBlock, Tag(BlockKind.QUESTION_WITH_OPTIONS.value)],
        Annotated[ErrorBlock, Tag(BlockKind.ERROR.value)],
    ],
    Discriminator(_block_discriminator),
]


class MessageWithBlocks(ApiModel):
    """Helper wrapper so a list of blocks can be validated in one call."""

    blocks: list[Block]

    @model_validator(mode="before")
    @classmethod
    def _accept_plain_list(cls, data: object) -> object:
        if isinstance(data, list):
            return {"blocks": data}
        return data


# Re-export so routers can build blocks without importing every class.
BLOCK_TYPES: tuple[type[BaseBlock], ...] = (
    TextBlock,
    FindingsBlock,
    RecommendationBlock,
    MetricBlock,
    ActionCard,
    TaskProgressBlock,
    ContextUsedBlock,
    QuestionWithOptionsBlock,
    ErrorBlock,
)

__all__ = [
    "ActionCard",
    "AgentEventData",
    "BaseBlock",
    "Block",
    "BlockKind",
    "BLOCK_TYPES",
    "ContextItem",
    "ContextUsedBlock",
    "ErrorBlock",
    "Finding",
    "FindingsBlock",
    "Message",
    "MessageWithBlocks",
    "MetricBlock",
    "Option",
    "QuestionWithOptionsBlock",
    "RecommendationBlock",
    "TaskProgressBlock",
    "TextBlock",
    "Autonomy",
    "Tone",
    "Language",
]
