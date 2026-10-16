"""Schemas for creative endpoints.

The reference-inspired flow is a fixed two-step chain:
analyze_reference_creative -> ReferenceAnalysis (structured StyleSpec) -> injected
into image_prompt / generate_image. A prose description of the reference is never
passed through.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.core.tools.base import ToolOutput
from app.schemas.common import ApiModel


class AnalyzeReferenceRequest(ApiModel):
    # Exactly one of image_url / upload_id / image_data must be provided.
    image_url: str | None = None
    upload_id: str | None = None
    # Small browser uploads arrive as a base64 data URL; bytes are validated and
    # moved to private storage before the vision tool is invoked.
    image_data: str | None = Field(default=None, max_length=3_000_000)
    campaign_id: str | None = None
    source_name: str = Field(default="", max_length=160)
    # Extra instructions for the vision model (e.g. "focus on the hook").
    focus: str | None = None

    def model_post_init(self, __context: object) -> None:
        supplied = sum(bool(value) for value in (self.image_url, self.upload_id, self.image_data))
        if supplied != 1:
            raise ValueError("provide exactly one of imageUrl, uploadId, or imageData")


class GenerateLaylaTextRequest(ApiModel):
    """A focused Layla writing request from the creative workspace."""

    tool: Literal[
        "generate_creative_brief",
        "generate_commercial_script",
        "generate_image_prompt",
        "generate_video_prompt",
        "generate_ad_copy",
        "generate_ad_variations",
    ]
    brief: str = Field(min_length=3, max_length=12000)
    language: Literal["en", "ar"] = "en"
    campaign_id: str | None = None
    creative_id: str | None = None


class GenerateLaylaTextResponse(ApiModel):
    id: str
    tool: Literal[
        "generate_creative_brief",
        "generate_commercial_script",
        "generate_image_prompt",
        "generate_video_prompt",
        "generate_ad_copy",
        "generate_ad_variations",
    ]
    output: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    # Which role/tier produced this text, and how much brand memory it was given.
    # Optional so older clients and stored payloads keep validating; the UI shows
    # them in the creative's detail panel.
    role: str | None = None
    model: str | None = None
    brand_context_tokens: int | None = None


class StyleElement(ApiModel):
    """One observed reference element with approximate relative layout."""

    element: str = ""
    relative_position: str = ""
    relative_size: str = ""
    visual_attributes: str = ""
    exact_text: str = ""
    layer_order: int | None = None


class StyleSpec(ApiModel):
    """Structured style/layout extracted from a reference creative."""

    canvas_ratio: str = ""
    background: str = ""
    lighting: str = ""
    visible_text: list[str] = Field(default_factory=list)
    elements: list[StyleElement] = Field(default_factory=list)
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
    source: str = Field(default="", description="private upload id or sanitized source label")
    source_name: str = ""
    summary: str = ""
    style: StyleSpec = Field(default_factory=StyleSpec)
    observed_offers: list[str] = Field(default_factory=list)
    observed_cta: list[str] = Field(default_factory=list)
    model: str = ""
    created_at: datetime | None = None


class CreativeReferenceItem(ApiModel):
    """A persisted VLM analysis that can be reused in later creative chats."""

    id: str
    source_name: str = "Saved reference"
    summary: str = ""
    style: StyleSpec = Field(default_factory=StyleSpec)
    favorite: bool = False
    created_at: datetime | None = None


class CreativeReferenceUpdate(ApiModel):
    favorite: bool


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


class SaveTextCreativeRequest(ApiModel):
    """Persist a piece of Layla text as a real creative.

    ``original_text`` is what Layla produced; ``text`` is what the user is
    keeping. When the two differ, the pair is a style signal (hook c of the
    learning loop) rather than a plain approval (hook b).
    """

    tool: Literal[
        "generate_creative_brief",
        "generate_commercial_script",
        "generate_image_prompt",
        "generate_video_prompt",
        "generate_ad_copy",
        "generate_ad_variations",
    ] = "generate_ad_copy"
    text: str = Field(min_length=1, max_length=20000)
    original_text: str | None = Field(default=None, max_length=20000)
    brief: str = Field(default="", max_length=12000)
    title: str = Field(default="", max_length=160)
    campaign_id: str | None = None
    creative_id: str | None = None
    model: str | None = Field(default=None, max_length=120)


class UpdateTextCreativeRequest(ApiModel):
    """Edit the copy of a saved text creative.

    Saving the edit records the before/after pair as brand-voice memory, which is
    the whole point: the user's own rewrite teaches Layla more than any adjective
    in their profile.
    """

    text: str = Field(min_length=1, max_length=20000)


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
    # Without these the endpoint returned HTTP 200 + `{"assets": []}` whenever the
    # image provider was missing or refused, which the client rendered as a green
    # "0 creatives generated" toast. The reason now reaches the caller.
    status: str = "ok"
    unavailable_reason: str | None = None
