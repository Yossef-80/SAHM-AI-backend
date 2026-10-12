"""Pydantic schemas for strategic planning tools (Omar)."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ==============================================================================
# Marketing Strategy Schemas
# ==============================================================================

class GenerateMarketingStrategyInput(BaseModel):
    focus: str = Field(
        description="Any specific strategic focus or angle (e.g. 'premium positioning vs. Toyota', 'youth demographic')."
    )


class TargetSegment(BaseModel):
    name: str = Field(description="Segment name or persona title")
    demographics: str = Field(description="Demographics and psychographics summary")
    pain_points: list[str] = Field(default_factory=list, description="Primary customer pain points")
    buying_triggers: list[str] = Field(default_factory=list, description="Motivators that trigger purchase decisions")


class StrategyPhase(BaseModel):
    phase: int = Field(description="Phase number: 1, 2, or 3")
    name: str = Field(description="Phase name (e.g. 'Testing & Validation', 'Optimization', 'Scaling')")
    duration_days: int = Field(default=30, description="Duration in days")
    focus_goals: list[str] = Field(default_factory=list, description="Core objectives for this phase")
    key_actions: list[str] = Field(default_factory=list, description="High-priority execution steps")


class MarketingStrategyOutput(BaseModel):
    positioning_statement: str = Field(
        description="Crisp, evidence-based brand positioning statement."
    )
    target_segments: list[TargetSegment] = Field(
        default_factory=list, description="Target customer segments with distinct needs."
    )
    value_propositions: list[str] = Field(
        default_factory=list, description="Core value propositions / USPs."
    )
    messaging_pillars: list[str] = Field(
        default_factory=list, description="Strategic pillars defining all brand communication."
    )
    competitive_advantage: str = Field(
        default="", description="Defensible differentiators against named competitors."
    )
    ninety_day_roadmap: list[StrategyPhase] = Field(
        default_factory=list, description="3-phase 90-day execution roadmap."
    )
    primary_kpis: list[str] = Field(
        default_factory=list, description="Top-line business and marketing KPIs."
    )


# ==============================================================================
# Campaign Strategy Schemas
# ==============================================================================

class GenerateCampaignStrategyInput(BaseModel):
    objective: str = Field(description="Primary campaign objective, e.g. 'Conversions', 'Lead Generation'.")
    budget: float | None = Field(default=None, description="Total budget amount, or None if unconstrained.")


class ChannelAllocation(BaseModel):
    channel: str = Field(description="Channel name, e.g. 'Facebook', 'Instagram', 'TikTok', 'Google Search'.")
    budget_share_pct: float = Field(
        description="Percentage share of total budget (0 to 100). All channels MUST sum to exactly 100."
    )
    role: str = Field(default="", description="Strategic function of this channel in the funnel.")
    recommended_formats: list[str] = Field(
        default_factory=list, description="Recommended creative formats (e.g. 'Reels', 'Carousels')."
    )
    kpis: list[str] = Field(default_factory=list, description="Target channel-level metrics/KPIs.")


class CampaignStrategyOutput(BaseModel):
    big_idea: str = Field(description="Core creative concept / big idea powering the campaign.")
    objective: str = Field(description="Campaign objective.")
    target_audiences: list[str] = Field(default_factory=list, description="Target audience groups.")
    channels: list[ChannelAllocation] = Field(
        default_factory=list, description="Channel mix with budget_share_pct values summing to 100."
    )
    timeline_weeks: int = Field(default=4, description="Recommended campaign duration in weeks.")
    key_messages: list[str] = Field(default_factory=list, description="Core messaging angles and copy hooks.")
    budget_allocation_summary: str = Field(
        default="", description="Rationale explaining budget distribution across channels."
    )


# ==============================================================================
# Content Calendar Schemas
# ==============================================================================

class GenerateContentCalendarInput(BaseModel):
    duration_days: int = Field(description="Number of days for the content calendar (e.g. 7, 14, 30).")


class CalendarPost(BaseModel):
    day: int = Field(description="Day number in schedule (1 to duration_days).")
    channel: str = Field(description="Publishing channel (e.g. 'Instagram', 'Facebook', 'TikTok', 'LinkedIn').")
    content_pillar: str = Field(description="Content pillar/theme this post reinforces.")
    format: str = Field(description="Post format (e.g. 'Reel', 'Carousel', 'Single Image', 'Story').")
    concept: str = Field(description="Creative visual hook and concept description.")
    caption_hook: str = Field(description="Suggested opening headline or caption hook.")
    call_to_action: str = Field(description="Specific conversion CTA.")
    best_time: str = Field(default="18:00", description="Optimal posting time.")


class ContentCalendarOutput(BaseModel):
    duration_days: int = Field(description="Total calendar duration in days.")
    themes: list[str] = Field(default_factory=list, description="Weekly or thematic focus themes.")
    posts: list[CalendarPost] = Field(default_factory=list, description="Day-by-day scheduled posts.")
    cadence_summary: str = Field(default="", description="Summary of posting frequency and channel balance.")
