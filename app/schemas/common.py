"""Shared primitives and enums mirrored from the frontend contract.

Wire format is camelCase (what a Next.js + Zod frontend sends and expects).
Every schema uses ``populate_by_name=True`` so Python-side snake_case also
works, which keeps internal code readable.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    """Base for every schema that crosses the HTTP boundary."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        use_enum_values=False,
        str_strip_whitespace=True,
    )


# ------------------------------------------------------------------ enums ----


class Tone(str, Enum):
    BOLD = "bold"
    PLAYFUL = "playful"
    MINIMAL = "minimal"
    PROFESSIONAL = "professional"


class Autonomy(str, Enum):
    SUGGEST = "suggest"
    AUTO_DRAFT = "auto_draft"
    ASK_FIRST = "ask_first"


class Language(str, Enum):
    EN = "en"
    AR = "ar"


class Tier(str, Enum):
    FAST = "fast"
    STRONG = "strong"


class ToolSideEffect(str, Enum):
    NONE = "none"
    DRAFT_WRITE = "draft_write"
    EXTERNAL_WRITE = "external_write"


class ToolStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class RecommendationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class BlockKind(str, Enum):
    TEXT = "text"
    FINDINGS = "findings"
    RECOMMENDATION = "recommendation"
    METRIC = "metric"
    ACTION_CARD = "action_card"
    TASK_PROGRESS = "task_progress"
    CONTEXT_USED = "context_used"
    QUESTION_WITH_OPTIONS = "question_with_options"
    ERROR = "error"


class IntentKind(str, Enum):
    RESEARCH = "research"
    STRATEGY = "strategy"
    CREATIVE = "creative"
    ANALYZE = "analyze"
    OPTIMIZE = "optimize"
    ASK_CLARIFICATION = "ask_clarification"


# ------------------------------------------------------------ shared types ---


class Evidence(ApiModel):
    """A pointer to where a claim came from. Never invented."""

    source: str = Field(description="e.g. 'web_search', 'get_campaign_insights', 'user_input'")
    detail: str = ""
    url: str | None = None
    retrieved_at: datetime | None = None


class Option(ApiModel):
    """One selectable answer in a question_with_options block."""

    id: str
    label: str
    description: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class Usage(ApiModel):
    """Token accounting for a single LLM call or a whole run."""

    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    latency_ms: int = 0

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            model=self.model or other.model,
            latency_ms=self.latency_ms + other.latency_ms,
        )


class TaskStep(ApiModel):
    """One step inside an AgentTask's step list."""

    name: str
    status: StepStatus = StepStatus.PENDING
    detail: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None


class ApiError(ApiModel):
    """Consistent JSON error body for every failure."""

    code: str
    message: str
    detail: dict[str, object] = Field(default_factory=dict)
    request_id: str | None = None


class ErrorResponse(ApiModel):
    error: ApiError


IdStr = Annotated[str, Field(min_length=1, max_length=64)]
