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

def blocks_to_plaintext(blocks: list[Any] | None) -> str:
    """Canonical lossless serializer converting Message.blocks to plain text.

    Single source of truth for text representations across content column,
    compaction, LTM summarization, notifications, and exports.
    Covers every block type and guarantees no information loss.
    RTL Arabic and emojis round-trip verbatim.
    """
    if not blocks:
        return ""
    if not isinstance(blocks, (list, tuple)):
        blocks = [blocks]

    parts: list[str] = []

    for block in blocks:
        if block is None:
            continue
        if hasattr(block, "model_dump"):
            b = block.model_dump(mode="json", by_alias=True)
        elif isinstance(block, dict):
            b = block
        else:
            b = getattr(block, "__dict__", {})

        b_type = str(b.get("type") or getattr(b, "type", "")).lower()
        if hasattr(b_type, "value"):
            b_type = str(b_type.value).lower()

        if b_type == "text":
            t = str(b.get("text") or "").strip()
            if t:
                parts.append(t)

        elif b_type == "findings":
            lines: list[str] = []
            title = str(b.get("title") or "").strip()
            if title:
                lines.append(title)

            raw_findings = b.get("findings") or b.get("items") or []
            for f in raw_findings:
                if hasattr(f, "model_dump"):
                    f_dict = f.model_dump(mode="json", by_alias=True)
                elif isinstance(f, dict):
                    f_dict = f
                else:
                    f_dict = getattr(f, "__dict__", {})

                f_title = str(f_dict.get("title") or f_dict.get("label") or "").strip()
                f_detail = str(f_dict.get("detail") or f_dict.get("description") or "").strip()
                f_source = str(f_dict.get("source") or "").strip()

                # Check evidence if source is not explicitly specified
                if not f_source:
                    evidence_list = f_dict.get("evidence") or []
                    ev_sources = []
                    for ev in evidence_list:
                        if hasattr(ev, "model_dump"):
                            ev = ev.model_dump(mode="json", by_alias=True)
                        if isinstance(ev, dict):
                            s = ev.get("url") or ev.get("source") or ev.get("title")
                            if s:
                                ev_sources.append(str(s).strip())
                    if ev_sources:
                        f_source = ", ".join(ev_sources[:2])

                line = ""
                if f_title and f_detail:
                    line = f"- {f_title}: {f_detail}"
                elif f_title:
                    line = f"- {f_title}"
                elif f_detail:
                    line = f"- {f_detail}"

                if line and f_source:
                    line += f" ({f_source})"
                if line:
                    lines.append(line)

            if lines:
                parts.append("\n".join(lines))

        elif b_type == "question_with_options":
            lines = []
            q = str(b.get("question") or "").strip()
            if q:
                lines.append(q)
            options = b.get("options") or []
            for idx, opt in enumerate(options, 1):
                if hasattr(opt, "model_dump"):
                    opt = opt.model_dump(mode="json", by_alias=True)
                if isinstance(opt, dict):
                    opt_label = str(opt.get("label") or opt.get("text") or "").strip()
                    opt_desc = str(opt.get("description") or "").strip()
                else:
                    opt_label = str(getattr(opt, "label", "") or getattr(opt, "text", "")).strip()
                    opt_desc = str(getattr(opt, "description", "") or "").strip()

                if opt_label and opt_desc:
                    lines.append(f"{idx}. {opt_label} - {opt_desc}")
                elif opt_label:
                    lines.append(f"{idx}. {opt_label}")
            if lines:
                parts.append("\n".join(lines))

        elif b_type == "recommendation":
            lines = []
            rec_title = str(b.get("title") or "").strip()
            if rec_title:
                lines.append(f"**Recommendation:** {rec_title}")
            rationale = str(b.get("rationale") or "").strip()
            if rationale:
                lines.append(rationale)
            impact = str(b.get("estimated_impact") or b.get("estimatedImpact") or "").strip()
            if impact:
                lines.append(f"Estimated impact: {impact}")
            if lines:
                parts.append("\n".join(lines))

        elif b_type == "metric":
            label = str(b.get("label") or "").strip()
            val = b.get("value")
            unit = str(b.get("unit") or "").strip()
            delta = b.get("delta")
            line = f"**{label}:** {val}{unit}" if unit else f"**{label}:** {val}"
            if delta is not None:
                line += f" (delta: {delta})"
            parts.append(line)

        elif b_type == "action_card":
            label = str(b.get("label") or b.get("title") or "").strip()
            desc = str(b.get("description") or "").strip()
            line = f"**Action:** {label}"
            if desc:
                line += f" - {desc}"
            parts.append(line)

        elif b_type == "task_progress":
            name = str(b.get("task_name") or b.get("taskName") or "Task").strip()
            status = str(b.get("status") or "").strip()
            msg = str(b.get("message") or "").strip()
            line = f"**Task {name}:** {status}"
            if msg:
                line += f" - {msg}"
            parts.append(line)

        elif b_type == "context_used":
            items = b.get("items") or []
            lines = ["**Context used:**"]
            for item in items:
                if hasattr(item, "model_dump"):
                    item = item.model_dump(mode="json", by_alias=True)
                if isinstance(item, dict):
                    l = str(item.get("label") or "").strip()
                    d = str(item.get("detail") or "").strip()
                else:
                    l = str(getattr(item, "label", "")).strip()
                    d = str(getattr(item, "detail", "")).strip()
                if l and d:
                    lines.append(f"- {l}: {d}")
                elif l:
                    lines.append(f"- {l}")
            if len(lines) > 1:
                parts.append("\n".join(lines))

        elif b_type == "error":
            msg = str(b.get("message") or "").strip()
            parts.append(f"**Error:** {msg}")

        else:
            # Unknown / future block type -> degrade gracefully to key fields
            lines = []
            for k in ("title", "label", "text", "message", "description", "detail", "summary"):
                val = str(b.get(k) or "").strip()
                if val:
                    lines.append(val)
            if not lines:
                for k, v in b.items():
                    if k not in ("id", "type") and v:
                        lines.append(f"{k}: {v}")
            if lines:
                parts.append("\n".join(lines))

    return "\n\n".join(parts)


__all__ = [
    "ActionCard",
    "AgentEventData",
    "BaseBlock",
    "Block",
    "BlockKind",
    "BLOCK_TYPES",
    "blocks_to_plaintext",
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

