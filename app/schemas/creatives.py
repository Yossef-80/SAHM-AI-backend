"""Schemas for creative endpoints.

The reference-inspired flow is a fixed two-step chain:
analyze_reference_creative -> ReferenceAnalysis (structured StyleSpec) -> injected
into image_prompt / generate_image. A prose description of the reference is never
passed through.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.core.tools.base import ToolOutput
from app.schemas.common import ApiModel


class AnalyzeReferenceRequest(ApiModel):
    # Exactly one of image_url / upload_id must be provided.
    image_url: str | None = None
    upload_id: str | None = None
    campaign_id: str | None = None
    # Extra instructions for the vision model (e.g. "focus on the hook").
    focus: str | None = None

    def model_post_init(self, __context: object) -> None:
        if not self.image_url and not self.upload_id:
            raise ValueError("either imageUrl or uploadId is required")


class StyleSpec(ApiModel):
    """Structured style extracted from a reference creative."""

    palette: list[str] = Field(default_factory=list, description="hex colors, ordered")
    composition: str = ""
    typography: str = ""
    tone: str = ""
    layout_pattern: str = ""
    hook_type: str = ""
    color_mood: str = ""
    text_density: str = ""
    brand_elements: list[str] = Field(default_factory=list)
    notes: str = ""


class ReferenceAnalysis(ToolOutput):
    id: str = ""
    source: str = Field(default="", description="image_url or upload id")
    summary: str = ""
    style: StyleSpec = Field(default_factory=StyleSpec)
    observed_offers: list[str] = Field(default_factory=list)
    observed_cta: list[str] = Field(default_factory=list)
    model: str = ""
    created_at: datetime | None = None


class GenerateCreativeRequest(ApiModel):
    prompt: str = Field(min_length=1)
    style_spec: StyleSpec | None = None
    # When set, the reference is analysed first and the resulting StyleSpec is
    # injected -- a prose description of the reference is never passed through.
    reference_analysis_id: str | None = None
    reference_image_url: str | None = None
    aspect_ratio: str = Field(default="1:1", description="e.g. 1:1, 4:5, 9:16, 16:9")
    campaign_id: str | None = None
    persona_id: str | None = None
    num_images: int = Field(default=1, ge=1, le=4)


class GeneratedAsset(ToolOutput):
    id: str
    url: str
    prompt: str
    aspect_ratio: str
    provider: str = ""
    model: str = ""
    style_spec: StyleSpec | None = None
    reference_analysis_id: str | None = None
    created_at: datetime | None = None


class GenerateCreativeResponse(ApiModel):
    assets: list[GeneratedAsset] = Field(default_factory=list)
    reference_analysis: ReferenceAnalysis | None = None
