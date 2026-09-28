"""Schemas for long-running agent tasks (AgentTask records)."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import ApiModel, TaskStatus, TaskStep, Usage


class AgentTask(ApiModel):
    id: str
    conversation_id: str | None = None
    task_name: str
    persona_id: str
    status: TaskStatus = TaskStatus.QUEUED
    steps: list[TaskStep] = Field(default_factory=list)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    incomplete: bool = False
    error: str | None = None
    result: dict[str, object] | None = None
    blocks: list[dict[str, object]] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    request_id: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class CancelTaskResponse(ApiModel):
    task_id: str
    status: TaskStatus
    message: str = ""
