"""Structured creative output schema for Layla."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ScriptBeat(BaseModel):
    timecode: str = ""
    duration_seconds: int | None = None
    visual: str = ""
    camera: str = ""
    audio_or_dialogue: str = ""
    on_screen_text: str = ""


class CreativeVariant(BaseModel):
    headline: str = ""
    primary_text: str = ""
    description: str = ""
    call_to_action: str = ""
    angle: str = ""


CreativePromptField = Literal[
    "subject",
    "objective",
    "audience",
    "specific_platform",
    "broad_platform",
    "format",
    "copy",
    "offer",
    "cta",
    "style",
    "palette",
    "avoid",
    "reference",
    "duration",
    "shots",
    "audio",
]


class CreativePromptFactUpdate(BaseModel):
    """One user-grounded semantic update to a guided prompt interview."""

    field: CreativePromptField = Field(
        description="The canonical prompt-interview slot being updated."
    )
    value: str = Field(
        default="",
        max_length=500,
        description="A concise normalized value; never add facts beyond the cited user wording.",
    )
    status: Literal["confirmed", "correction", "negated", "uncertain"] = Field(
        default="confirmed",
        description="Whether the user confirmed, corrected, declined, or expressed uncertainty about this value.",
    )
    evidence: str = Field(
        min_length=1,
        max_length=500,
        description="An exact quote copied from the latest user message that supports this update.",
    )


class CreativeBrief(BaseModel):
    """Call this tool to submit copy, a brief, script, or generation prompt."""

    deliverable_type: Literal[
        "ad_copy",
        "creative_brief",
        "commercial_script",
        "image_prompt",
        "video_prompt",
        "ad_variations",
        "reference_adaptation",
    ] = "creative_brief"
    summary: str = ""
    details: list[str] = Field(default_factory=list)
    headline: str | None = None
    primary_text: str | None = None
    description: str | None = None
    call_to_action: str | None = None
    variants: list[CreativeVariant] = Field(default_factory=list)
    script: list[ScriptBeat] = Field(default_factory=list)
    image_prompt: str | None = None
    video_prompt: str | None = None
    negative_prompt: str | None = None
    aspect_ratio: str | None = None
    music_direction: str | None = None
    reference_style_spec: dict[str, Any] | None = None
    preserved_elements: list[str] = Field(default_factory=list)
    proposed_changes: list[str] = Field(default_factory=list)
    ready_for_approval: bool = Field(
        default=False,
        description="True only when the task-relevant details are sufficient to present a concrete plan for user approval; false while any essential input is missing.",
    )
    missing_inputs: list[str] = Field(default_factory=list)
    questions_for_user: list[str] = Field(default_factory=list)
    prompt_fact_updates: list[CreativePromptFactUpdate] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "For guided prompt interviews, include semantic updates from the latest user turn only. "
            "Every update must cite an exact user quote in evidence; leave empty when none."
        ),
    )
    generation_requested: bool = False
    generated_assets: list[dict[str, Any]] = Field(default_factory=list)
