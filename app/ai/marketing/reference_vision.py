"""Multimodal reference-image analysis for Layla.

The image is passed as an actual multimodal content block to an app.ai chat
model. ``AI_ADS_VLM_PROVIDER`` and ``AI_ADS_VLM_MODEL`` can select a separate
vision provider/model (such as Gemini) without changing the system LLM. The
legacy ``AI_ADS_VLM_BASE_URL`` plus ``VLM_OLLAMA_MODEL`` dedicated-tunnel path
continues to work when no explicit VLM provider is selected.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import math
import warnings

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

from app.ai.config import TEXT_ONLY_VISION_MODEL_PREFIXES
from app.ai.enums import LLMProvider
from app.ai.providers.llm import LLMHandle, get_vlm_llm
from app.ai.settings import get_ai_settings

logger = logging.getLogger(__name__)


class VisionAnalysisUnavailable(RuntimeError):
    """The selected model cannot honestly inspect this reference image."""


def _prepare_reference_image_for_vlm(
    image_b64: str,
    mime: str,
    *,
    max_pixels: int,
) -> tuple[str, str]:
    """Downscale only the analysis copy, keeping aspect ratio and source bytes intact.

    Images at or below the budget are passed through byte-for-byte. Larger images
    are resized with a high-quality filter, then re-encoded in their original
    supported format. If decoding/re-encoding fails, preserve the prior behavior
    and send the validated source rather than blocking analysis on preprocessing.
    """
    try:
        source_bytes = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VisionAnalysisUnavailable("reference image data is not valid base64") from exc

    try:
        # Pillow raises a warning for unusually large decoded images; treat that
        # as a hard safety limit rather than decoding an unbounded image.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(source_bytes)) as source:
                image = ImageOps.exif_transpose(source)
                width, height = image.size
                source_pixels = width * height
                if source_pixels <= max_pixels:
                    return image_b64, mime

                scale = math.sqrt(max_pixels / source_pixels)
                target_size = (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                )
                image = _image_mode_for_resize(image, mime)
                resized = image.resize(target_size, Image.Resampling.LANCZOS)
                output = io.BytesIO()
                _save_resized_image(resized, mime, output)

        prepared_b64 = base64.b64encode(output.getvalue()).decode("ascii")
        logger.info(
            "Reference image downscaled for VLM: %dx%d -> %dx%d pixels (budget=%d)",
            width,
            height,
            target_size[0],
            target_size[1],
            max_pixels,
        )
        return prepared_b64, mime
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise VisionAnalysisUnavailable(
            "reference image exceeds Pillow's safe decode limit"
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Could not resize reference image for VLM; using source bytes: %s", exc)
        return image_b64, mime


def _image_mode_for_resize(image: Image.Image, mime: str) -> Image.Image:
    """Normalize uncommon Pillow modes while retaining transparency where relevant."""
    if mime == "image/jpeg":
        return image.convert("RGB")
    if mime == "image/png":
        if image.mode in {"L", "LA", "RGB", "RGBA"}:
            return image
        if "transparency" in image.info or "A" in image.getbands():
            return image.convert("RGBA")
        return image.convert("L" if image.mode == "1" else "RGB")
    if "A" in image.getbands():
        return image.convert("RGBA")
    return image if image.mode in {"L", "RGB"} else image.convert("RGB")


def _save_resized_image(image: Image.Image, mime: str, output: io.BytesIO) -> None:
    """Encode a compact, OCR-friendly image without changing its declared MIME."""
    if mime == "image/jpeg":
        image.save(output, format="JPEG", quality=90, optimize=True, subsampling=0)
    elif mime == "image/png":
        image.save(output, format="PNG", compress_level=4)
    else:  # image/webp
        image.save(output, format="WEBP", quality=92, method=4)


class StyleElementExtraction(BaseModel):
    element: str = ""
    relative_position: str = ""
    relative_size: str = ""
    visual_attributes: str = ""
    exact_text: str = ""
    layer_order: int | None = None


class ReferenceVisualExtraction(BaseModel):
    """Structured observations returned for a reference image."""

    canvas_ratio: str = ""
    background: str = ""
    lighting: str = ""
    visible_text: list[str] = Field(default_factory=list)
    elements: list[StyleElementExtraction] = Field(default_factory=list)
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


def _reject_known_text_only_model(llm: LLMHandle) -> None:
    """Fail honestly for known mock/text-only models before sending image data."""
    model = str(llm.model_name or "").strip().lower()
    if llm.provider == LLMProvider.MOCK or model.startswith("mock-"):
        raise VisionAnalysisUnavailable(
            "The mock LLM does not inspect image pixels, so no reference analysis was fabricated. "
            "Use a vision-capable model to analyse references."
        )
    if model.startswith(TEXT_ONLY_VISION_MODEL_PREFIXES):
        raise VisionAnalysisUnavailable(
            "The selected GPT-OSS model is text-only and cannot inspect image pixels. "
            "The general provider/model selection was left unchanged."
        )


async def analyze_reference_image(
    *,
    image_b64: str,
    mime: str,
    focus: str | None,
) -> ReferenceVisualExtraction:
    """Return image-grounded visual observations using the app.ai VLM provider."""
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise VisionAnalysisUnavailable("reference image must be PNG, JPEG, or WebP")
    if not image_b64:
        raise VisionAnalysisUnavailable("reference image data is empty")

    settings = get_ai_settings()
    model = get_vlm_llm(settings=settings)
    _reject_known_text_only_model(model)
    vision_image_b64, vision_mime = _prepare_reference_image_for_vlm(
        image_b64,
        mime,
        max_pixels=settings.ads_vlm_max_pixels,
    )
    instruction = (
        "Analyse the attached reference image for a high-fidelity visual recreation. "
        "Return only structured observations, not a prose summary. Record the canvas "
        "ratio, background and lighting; transcribe every legible word exactly (same "
        "language, capitalization and line order); and list up to 24 visible elements "
        "in reading order with approximate relative position, size, appearance, exact "
        "visible text, and layer order. Preserve composition and object count. Report "
        "only what is actually visible: never guess an unreadable word, exact font name, "
        "palette colour, offer, CTA, or performance claim. Leave uncertain fields empty."
    )
    if focus and focus.strip():
        instruction += f" Pay particular attention to: {focus.strip()}."

    messages = [
        SystemMessage(content=instruction),
        HumanMessage(
            content=[
                {"type": "text", "text": "Return the structured style analysis for the attached image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{vision_mime};base64,{vision_image_b64}",
                    },
                },
            ]
        ),
    ]
    result = await model.structured(ReferenceVisualExtraction).ainvoke(messages)
    try:
        parsed = (
            result
            if isinstance(result, ReferenceVisualExtraction)
            else ReferenceVisualExtraction.model_validate(result)
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise VisionAnalysisUnavailable(
            "the vision model did not return a usable structured style analysis"
        ) from exc
    return parsed


__all__ = [
    "ReferenceVisualExtraction",
    "StyleElementExtraction",
    "VisionAnalysisUnavailable",
    "analyze_reference_image",
]
