from typing import Literal

from pydantic import BaseModel, Field


class StrategyPhase(BaseModel):
    name: str = ""
    duration_days: int | None = Field(default=None, ge=1)
    focus: str = ""
    actions: list[str] = Field(default_factory=list)


class KPITarget(BaseModel):
    metric: str = ""
    target: str | None = Field(
        default=None,
        description="Only provide a target when it is supported by user input or evidence; otherwise leave null and note the missing input.",
    )


class ChannelPlan(BaseModel):
    channel: str = ""
    role: str = ""
    budget_share_pct: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Optional suggested share only when supported by the brief. Never invent a currency amount or total budget.",
    )
    placements: list[str] = Field(default_factory=list)
    rationale: str = ""


class MarketingStrategyOutput(BaseModel):
    """Structured marketing strategy. Missing facts remain explicit, not guessed."""

    summary: str = ""
    positioning: str = ""
    value_propositions: list[str] = Field(default_factory=list)
    target_segments: list[str] = Field(default_factory=list)
    phases: list[StrategyPhase] = Field(default_factory=list)
    channels: list[ChannelPlan] = Field(default_factory=list)
    kpis: list[KPITarget] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)


class CampaignStrategyOutput(BaseModel):
    """Campaign-level brief that can be handed to Layla for creative work."""

    summary: str = ""
    objective: str | None = Field(
        default=None,
        description="Use only an objective stated by the user or present in the supplied evidence; otherwise leave null.",
    )
    audience: str | None = None
    positioning: str | None = None
    key_messages: list[str] = Field(default_factory=list)
    channels: list[ChannelPlan] = Field(default_factory=list)
    placements: list[str] = Field(default_factory=list)
    creative_direction: str | None = None
    cta_direction: str | None = None
    creatives_needed: list[str] = Field(default_factory=list)
    success_metrics: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)


class ContentItem(BaseModel):
    """One proposed calendar entry. The API later assigns its exact date."""

    week: int | None = Field(
        default=None,
        ge=1,
        description="1-based seven-day bucket within the requested month",
    )
    channel: Literal["instagram", "facebook", "tiktok", "email", "blog"] = "instagram"
    format: Literal["post", "reel", "story", "ad", "email", "article"] = "post"
    topic: str = ""
    title: str = ""
    hook: str = ""
    caption: str = ""
    hashtags: list[str] = Field(default_factory=list)
    pillar: str = ""
    goal: str = ""
    cta: str = ""


class ContentCalendarOutput(BaseModel):
    summary: str = ""
    cadence: str = ""
    items: list[ContentItem] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)


class StrategyReport(BaseModel):
    """Call this tool to submit Omar's final output for the current run."""

    summary: str = Field(description="Concise summary of the strategy completed in this run")
    details: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(
        default_factory=list,
        description="Facts or settings that must be supplied before they can be relied on; do not guess them.",
    )
