"""Schemas for the agent runtime's own results and events."""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import ApiModel, Usage


class ToolCallRecord(ApiModel):
    """One tool invocation made during a run, for the UI and for debugging."""

    tool: str
    input: dict[str, object] = Field(default_factory=dict)
    output_summary: str = ""
    status: str = "ok"
    duration_ms: int = 0
    error: str | None = None


class AgentResult(ApiModel):
    """The runtime's return value for run_agent()."""

    task: str
    persona_id: str
    output: dict[str, object] = Field(default_factory=dict)
    tool_calls_made: list[ToolCallRecord] = Field(default_factory=list)
    unavailable_sources: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    request_id: str = ""
    # True when a hard cap (iterations/tokens/timeout) was hit.
    incomplete: bool = False
    incomplete_reason: str | None = None
    # What the autonomy policy decided to persist, for auditability.
    persisted: list[str] = Field(default_factory=list)
    error: str | None = None


class AgentEvent(ApiModel):
    """Streaming event yielded by run_agent_stream()."""

    type: str = Field(
        description="step_started | tool_called | tool_result | token | completed | failed"
    )
    step: str | None = None
    detail: str | None = None
    token: str | None = None
    data: dict[str, object] | None = None


class RunAgentRequest(ApiModel):
    persona_id: str
    task: str
    inputs: dict[str, object] = Field(default_factory=dict)
    campaign_id: str | None = None
    creative_id: str | None = None
    insight_id: str | None = None
    conversation_id: str | None = None
