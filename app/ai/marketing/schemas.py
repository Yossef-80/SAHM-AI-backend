"""Internal structured outputs for marketing generation and vision."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class MarketingModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class StyleSpec(MarketingModel):
    palette: list[str] = Field(default_factory=list, description="Observed colours; do not guess.")
    composition: str = ""
    typography: str = ""
    tone: str = ""
    layout_pattern: str = ""
    hook_type: str = ""
    color_mood: str = ""
    text_density: str = ""
    brand_elements: list[str] = Field(default_factory=list)
    notes: str = ""


class ReferenceVisionOutput(MarketingModel):
    summary: str = ""
    style: StyleSpec = Field(default_factory=StyleSpec)
    observed_offers: list[str] = Field(default_factory=list)
    observed_cta: list[str] = Field(default_factory=list)


class StrategyPhase(MarketingModel):
    name: str
    duration_days: int = Field(gt=0)
    focus: str


class StrategyKPI(MarketingModel):
    metric: str
    target: float


class CampaignStrategy(MarketingModel):
    summary: str
    key_messages: list[str] = Field(default_factory=list)
    channels: list[Literal["facebook", "instagram", "messenger", "audience_network"]] = Field(default_factory=list)
    placements: list[str] = Field(default_factory=list)
    bidding_strategy: Literal["lowest_cost", "cost_cap", "bid_cap", "roas_goal"] = "lowest_cost"
    phases: list[StrategyPhase] = Field(default_factory=list)
    kpis: list[StrategyKPI] = Field(default_factory=list)


class TextSection(MarketingModel):
    heading: str
    body: str


class GeneratedTextResult(MarketingModel):
    title: str
    sections: list[TextSection] = Field(min_length=1)
    plain_text: str
    artifact_type: Literal["text", "image_prompt", "video_prompt"] = "text"


class GenericTextDraft(MarketingModel):
    title: str
    sections: list[TextSection] = Field(min_length=1)


class ImagePromptDraft(MarketingModel):
    title: str
    image_prompt: str = Field(min_length=1)
    composition: str
    lighting: str
    palette: list[str] = Field(default_factory=list)
    text_overlay: str = ""
    negative_prompt: str = ""
    aspect_ratio: str


class VideoShotDraft(MarketingModel):
    visual: str
    camera_motion: str
    voiceover: str = ""
    on_screen_text: str = ""
    duration_weight: float = Field(default=1.0, gt=0)


class VideoPromptDraft(MarketingModel):
    title: str
    opening_hook: str
    shots: list[VideoShotDraft] = Field(min_length=2, max_length=8)
    voiceover_direction: str = ""
    sound_direction: str = ""
    cta: str = ""
    negative_prompt: str = ""


__all__ = [
    "CampaignStrategy",
    "GeneratedTextResult",
    "GenericTextDraft",
    "ImagePromptDraft",
    "ReferenceVisionOutput",
    "StrategyKPI",
    "StrategyPhase",
    "StyleSpec",
    "TextSection",
    "VideoPromptDraft",
    "VideoShotDraft",
]
