"""media tools (11-12): reference analysis and image generation.

The reference-inspired creative flow is a fixed two-step chain and it is
structural, not advisory:

    analyze_reference_creative -> ReferenceAnalysis (structured StyleSpec)
        -> injected into image_prompt / generate_image

A prose description of the reference is never passed to the image model. The
``style_spec`` dict is the only thing that crosses the boundary, so the palette,
composition, typography, tone, layout pattern and hook type survive intact.
"""

from __future__ import annotations

import base64
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.core.errors import BlockedTargetError, ToolUnavailable
from app.core.logging import get_logger
from app.core.security import resolve_and_check
from app.core.tools.base import Tool, ToolContext
from app.db.models import Creative
from app.schemas.common import ToolSideEffect
from app.schemas.creatives import (
    GeneratedAsset,
    ReferenceAnalysis,
    StyleSpec,
)
from app.schemas.tools import (
    AnalyzeReferenceInput,
    GenerateImageInput,
    GenerateImageOutput,
    GeneratedAssetRecord,
)

_logger = get_logger(__name__)


# ================================================ 11. analyze_reference_creative


class _StyleExtraction(BaseModel):
    """What the vision model returns. Always structured, never prose."""

    palette: list[str] = Field(default_factory=list, description="hex colours, ordered")
    composition: str = ""
    typography: str = ""
    tone: str = ""
    layout_pattern: str = ""
    hook_type: str = ""
    color_mood: str = ""
    text_density: str = ""
    brand_elements: list[str] = Field(default_factory=list)
    summary: str = ""
    observed_offers: list[str] = Field(default_factory=list)
    observed_cta: list[str] = Field(default_factory=list)


async def _load_reference_image(
    payload: AnalyzeReferenceInput, ctx: ToolContext
) -> tuple[str, str, str]:
    """Return (mime_type, base64_data, source_label)."""
    cfg = ctx.require_settings()
    if payload.upload_id:
        path = Path(cfg.upload_dir) / f"{payload.upload_id}"
        if not path.is_file():
            raise ToolUnavailable(f"upload '{payload.upload_id}' was not found")
        data = path.read_bytes()
        mime = _guess_mime(path.name)
        return mime, base64.b64encode(data).decode(), f"upload:{payload.upload_id}"

    if payload.image_url:
        safe = resolve_and_check(payload.image_url, allowed_schemes=("http", "https"))
        async with httpx.AsyncClient(timeout=cfg.website_fetch_timeout_seconds) as client:
            resp = await client.get(safe.url)
            resp.raise_for_status()
            if len(resp.content) > cfg.website_max_bytes:
                raise ToolUnavailable("reference image exceeds the size cap")
            mime = resp.headers.get("content-type", "image/jpeg").split(";")[0]
            if not mime.startswith("image/"):
                raise ToolUnavailable(f"url did not return an image (got {mime})")
        return mime, base64.b64encode(resp.content).decode(), f"url:{payload.image_url}"

    raise ToolUnavailable("either imageUrl or uploadId must be provided")


def _guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/jpeg")


async def _vision_extract(
    image_b64: str, mime: str, focus: str | None, ctx: ToolContext
) -> _StyleExtraction:
    """Ask the vision-capable LLM for a structured style analysis."""
    from app.core.llm.base import Msg

    instruction = (
        "Analyse this reference creative and describe its VISUAL STYLE in structured "
        "form. Report only what you can actually see. Never invent a palette colour, "
        "a font name you cannot read, or a claim about performance. If you cannot "
        "determine something, leave the field empty."
    )
    if focus:
        instruction += f" Pay particular attention to: {focus}."

    result = await ctx.llm.complete(
        system=instruction,
        messages=[
            Msg(
                role="user",
                content=(
                    "Return the structured style analysis for the attached image."
                ),
            )
        ],
        response_schema=_StyleExtraction,
        tier="strong",
        max_tokens=1500,
        temperature=0.2,
    )
    if not isinstance(result.parsed, _StyleExtraction):
        raise ToolUnavailable("the vision model did not return a usable style analysis")
    return result.parsed


async def _handle_analyze_reference(
    payload: AnalyzeReferenceInput, ctx: ToolContext
) -> ReferenceAnalysis:
    source = "analyze_reference_creative:vision_llm"
    try:
        mime, image_b64, label = await _load_reference_image(payload, ctx)
    except ToolUnavailable as exc:
        return ReferenceAnalysis.unavailable(source=source, reason=exc.message)
    except BlockedTargetError as exc:
        return ReferenceAnalysis.unavailable(source=source, reason=exc.message)
    except httpx.HTTPError as exc:
        return ReferenceAnalysis.unavailable(source=source, reason=f"could not fetch image: {exc}")

    try:
        extraction = await _vision_extract(image_b64, mime, payload.focus, ctx)
    except Exception as exc:
        return ReferenceAnalysis.unavailable(
            source=source,
            reason=(
                f"The vision model could not analyse the image: {exc}. Describe the "
                "reference in your own words and I will build the style spec from "
                "that instead."
            ),
        )

    style = StyleSpec(
        palette=extraction.palette,
        composition=extraction.composition,
        typography=extraction.typography,
        tone=extraction.tone,
        layout_pattern=extraction.layout_pattern,
        hook_type=extraction.hook_type,
        color_mood=extraction.color_mood,
        text_density=extraction.text_density,
        brand_elements=extraction.brand_elements,
        notes=extraction.summary,
    )

    # Persist as a reference_analysis memory item so future tasks can reuse it.
    memory_id: str | None = None
    try:
        from app.core.memory.store import BrandMemoryStore

        store = BrandMemoryStore(ctx.db)
        item = await store.write_item(
            business_id=ctx.business_id,
            kind="reference_analysis",
            text=(
                f"Reference creative style: {extraction.summary}. "
                f"Palette {', '.join(extraction.palette) or 'not available'}. "
                f"Layout {extraction.layout_pattern or 'not available'}. "
                f"Hook type {extraction.hook_type or 'not available'}."
            ),
            metadata={
                "style": style.model_dump(),
                "source": label,
                "campaign_id": payload.campaign_id,
            },
            campaign_id=payload.campaign_id,
        )
        memory_id = item.id
    except Exception as exc:  # pragma: no cover - memory write must not fail the tool
        _logger.warning("reference_memory_write_failed", extra={"error": str(exc)})

    return ReferenceAnalysis(
        id=memory_id or f"ref_{uuid.uuid4().hex[:16]}",
        source=label,
        status="ok",
        summary=extraction.summary,
        style=style,
        observed_offers=extraction.observed_offers,
        observed_cta=extraction.observed_cta,
        created_at=datetime.now(timezone.utc),
    )


analyze_reference_creative_tool = Tool(
    name="analyze_reference_creative",
    description=(
        "Analyse a reference image and return a STRUCTURED style spec (palette, "
        "composition, typography, tone, layout pattern, hook type). Always call this "
        "before generating an image that should follow a reference. Never describe "
        "the reference in prose to the image model -- pass the structured StyleSpec "
        "from this tool instead."
    ),
    input_model=AnalyzeReferenceInput,
    output_model=ReferenceAnalysis,
    side_effect=ToolSideEffect.DRAFT_WRITE,
    handler=_handle_analyze_reference,
)


# ================================================== 12. generate_image


def _mock_asset(prompt: str, aspect_ratio: str, index: int) -> GeneratedAssetRecord:
    """Deterministic offline asset: a labelled SVG data URL."""
    digest = hashlib.sha256(f"{prompt}|{index}".encode()).hexdigest()[:10]
    palette = ["#0f172a", "#22d3ee", "#f472b6", "#facc15"]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1080" '
        f'viewBox="0 0 1080 1080"><rect width="1080" height="1080" fill="{palette[0]}"/>'
        f'<circle cx="540" cy="430" r="230" fill="{palette[1]}" opacity="0.85"/>'
        f'<rect x="140" y="720" width="800" height="40" rx="20" fill="{palette[2]}"/>'
        f'<text x="540" y="400" font-family="sans-serif" font-size="64" fill="#ffffff" '
        f'text-anchor="middle">MOCK ASSET</text>'
        f'<text x="540" y="480" font-family="sans-serif" font-size="28" fill="#cbd5e1" '
        f'text-anchor="middle">{digest}</text></svg>'
    )
    encoded = base64.b64encode(svg.encode()).decode()
    return GeneratedAssetRecord(
        id=f"asset_{uuid.uuid4().hex[:16]}",
        url=f"data:image/svg+xml;base64,{encoded}",
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        provider="mock",
        model="mock-image-1",
        style_spec={},
    )


async def _generate_openai_image(
    prompt: str, aspect_ratio: str, cfg: Any
) -> list[GeneratedAssetRecord]:
    size = {"1:1": "1024x1024", "4:5": "1024x1280", "9:16": "1024x1792", "16:9": "1792x1024"}.get(
        aspect_ratio, "1024x1024"
    )
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"authorization": f"Bearer {cfg.image_api_key}"},
            json={"model": cfg.image_model, "prompt": prompt, "size": size, "n": 1},
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
    return [
        GeneratedAssetRecord(
            id=f"asset_{uuid.uuid4().hex[:16]}",
            url=item.get("url", ""),
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            provider="openai",
            model=cfg.image_model,
        )
        for item in data
    ]


async def _handle_generate_image(
    payload: GenerateImageInput, ctx: ToolContext
) -> GenerateImageOutput:
    cfg = ctx.require_settings()
    source = f"generate_image:{cfg.image_provider}"

    # The style spec is the ONLY thing that may carry reference styling.
    style = dict(payload.style_spec or {})
    if payload.reference_analysis_id:
        resolved = await _resolve_reference(ctx, payload.reference_analysis_id)
        if resolved is not None:
            style = {**style, **resolved}

    effective_prompt = _compose_image_prompt(payload.prompt, style)

    try:
        if cfg.image_provider == "openai" and cfg.image_api_key:
            assets = await _generate_openai_image(effective_prompt, payload.aspect_ratio, cfg)
        elif cfg.image_provider == "stability" and cfg.image_api_key:
            return GenerateImageOutput.unavailable(
                source=source,
                reason=(
                    "The stability image provider is configured but not wired up in "
                    "this build. Switch IMAGE_PROVIDER=openai with IMAGE_API_KEY set, "
                    "or use IMAGE_PROVIDER=mock for offline placeholder assets."
                ),
            )
        else:
            assets = [
                _mock_asset(effective_prompt, payload.aspect_ratio, i)
                for i in range(payload.num_images)
            ]
    except httpx.HTTPError as exc:
        return GenerateImageOutput.unavailable(
            source=source, reason=f"image generation failed: {exc}"
        )

    # Persist each asset so it shows up under the campaign.
    persisted: list[GeneratedAssetRecord] = []
    for asset in assets:
        asset.style_spec = style
        try:
            ctx.db.add(
                Creative(
                    id=asset.id,
                    business_id=ctx.business_id,
                    campaign_id=payload.campaign_id or ctx.campaign_id,
                    persona_id=ctx.persona_id,
                    task_name="generate_image",
                    kind="image",
                    url=asset.url,
                    prompt=asset.prompt,
                    aspect_ratio=asset.aspect_ratio,
                    provider=asset.provider,
                    model=asset.model,
                    style_spec=style or None,
                    reference_analysis_id=payload.reference_analysis_id,
                    status="ok",
                )
            )
            await ctx.db.flush()
        except Exception as exc:  # pragma: no cover
            _logger.warning("creative_persist_failed", extra={"error": str(exc)})
        persisted.append(asset)

    return GenerateImageOutput(assets=persisted, status="ok", source=source)


def _compose_image_prompt(prompt: str, style: dict[str, Any]) -> str:
    """Fold the structured style spec into the image prompt.

    This is the only sanctioned path from a reference analysis to a generation
    call: structured fields in, structured text out. No prose paraphrase of the
    reference is ever involved.
    """
    if not style:
        return prompt
    parts = [prompt, "", "STYLE (follow exactly):"]
    mapping = [
        ("palette", "Palette"),
        ("composition", "Composition"),
        ("typography", "Typography"),
        ("tone", "Tone"),
        ("layout_pattern", "Layout pattern"),
        ("hook_type", "Hook type"),
        ("color_mood", "Colour mood"),
        ("text_density", "Text density"),
    ]
    for key, label in mapping:
        value = style.get(key)
        if isinstance(value, list) and value:
            parts.append(f"- {label}: {', '.join(str(v) for v in value)}")
        elif isinstance(value, str) and value:
            parts.append(f"- {label}: {value}")
    if isinstance(style.get("brand_elements"), list) and style["brand_elements"]:
        parts.append(f"- Brand elements: {', '.join(str(v) for v in style['brand_elements'])}")
    return "\n".join(parts)


async def _resolve_reference(ctx: ToolContext, reference_id: str) -> dict[str, Any] | None:
    """Look up a stored reference_analysis memory item and return its style."""
    from sqlalchemy import select

    from app.db.models import BrandMemoryItem

    stmt = select(BrandMemoryItem).where(
        BrandMemoryItem.id == reference_id,
        BrandMemoryItem.business_id == ctx.business_id,
        BrandMemoryItem.kind == "reference_analysis",
    )
    item = (await ctx.db.execute(stmt)).scalars().first()
    if item is None:
        return None
    style = (item.metadata_ or {}).get("style")
    return style if isinstance(style, dict) else None


generate_image_tool = Tool(
    name="generate_image",
    description=(
        "Generate image creatives from a prompt and an optional structured style "
        "spec. When the user wants something that looks like a reference image, "
        "first call analyze_reference_creative and pass the resulting style_spec "
        "(or its reference_analysis_id) here. Do not describe the reference in prose."
    ),
    input_model=GenerateImageInput,
    output_model=GenerateImageOutput,
    side_effect=ToolSideEffect.DRAFT_WRITE,
    handler=_handle_generate_image,
)


MEDIA_TOOLS: list[Tool] = [analyze_reference_creative_tool, generate_image_tool]

__all__ = [
    "MEDIA_TOOLS",
    "analyze_reference_creative_tool",
    "generate_image_tool",
]
