"""Isolated image-reference loading and visual-style extraction.

The VLM client is deliberately independent from ``AI_LLM_PROVIDER`` and from the
text provider. It receives the bytes and MIME type of an image; this module
never hands an image URL, upload identifier, or data URL to a text-only model.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.marketing.schemas import ReferenceVisionOutput
from app.ai.settings import AISettings, get_ai_settings
from app.config import Settings, get_settings
from app.core.errors import BlockedTargetError, ToolUnavailable
from app.core.memory.lifecycle import memory_defaults
from app.core.security import resolve_and_check, sniff_mime
from app.db.models import BrandMemoryItem
from app.schemas.creatives import ReferenceAnalysis, StyleSpec

_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_DATA_URL_RE = re.compile(r"^data:(image/(?:jpeg|png|webp|gif));base64,([A-Za-z0-9+/=\r\n]+)$")


class VisionModelUnavailable(RuntimeError):
    """The isolated VLM is disabled, unconfigured or returned unusable output."""


def _clean_source_name(name: str | None, fallback: str = "Uploaded image") -> str:
    candidate = Path(name or "").name.strip()[:160]
    return candidate or fallback


def _source_label(value: str | None) -> str:
    if not value:
        return "uploaded image"
    if value.startswith("data:"):
        return "uploaded image"
    if value.startswith("http://") or value.startswith("https://"):
        parts = urlsplit(value)
        return urlunsplit((parts.scheme, parts.netloc, parts.path[:300], "", ""))
    return value[:200]


def _verify_image(data: bytes, declared_mime: str, max_bytes: int) -> str:
    if not data:
        raise ToolUnavailable("image is empty")
    if len(data) > max_bytes:
        raise ToolUnavailable(f"image exceeds the {max_bytes}-byte upload limit")
    actual = sniff_mime(data[:32])
    if actual not in _IMAGE_MIMES:
        raise ToolUnavailable("file bytes are not a supported image")
    if declared_mime not in _IMAGE_MIMES or actual != declared_mime:
        raise ToolUnavailable("image MIME type does not match its file bytes")
    return actual


async def load_reference_image(
    *,
    image_url: str | None,
    upload_id: str | None,
    source_name: str | None,
    core_settings: Settings | None = None,
) -> tuple[bytes, str, str, str]:
    """Load a bounded image and return bytes, MIME, safe source label, name."""
    cfg = core_settings or get_settings()
    max_bytes = cfg.upload_max_bytes
    image_types = set(cfg.upload_allowed_types) & _IMAGE_MIMES
    if not image_types:
        image_types = _IMAGE_MIMES

    if upload_id:
        # Legacy upload identifiers are plain file names only; no traversal.
        if Path(upload_id).name != upload_id or upload_id in {".", ".."}:
            raise ToolUnavailable("invalid upload identifier")
        root = Path(cfg.upload_dir).resolve()
        path = (root / upload_id).resolve()
        if path.parent != root or not path.is_file():
            raise ToolUnavailable("uploaded image was not found")
        data = path.read_bytes()
        suffix = path.suffix.lower()
        by_suffix = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
        mime = by_suffix.get(suffix, "")
        mime = _verify_image(data, mime, max_bytes)
        if mime not in image_types:
            raise ToolUnavailable("this image type is disabled on the server")
        return data, mime, f"upload:{upload_id}", _clean_source_name(source_name or upload_id)

    if not image_url:
        raise ToolUnavailable("either imageUrl or uploadId is required")

    if image_url.startswith("data:"):
        if len(image_url) > int(max_bytes * 4 / 3) + 512:
            raise ToolUnavailable("image exceeds the upload limit")
        match = _DATA_URL_RE.fullmatch(image_url)
        if not match:
            raise ToolUnavailable("only base64 PNG, JPEG, WebP or GIF data URLs are accepted")
        declared_mime = match.group(1)
        try:
            data = base64.b64decode(match.group(2), validate=True)
        except (ValueError, TypeError) as exc:
            raise ToolUnavailable("image data URL is malformed") from exc
        mime = _verify_image(data, declared_mime, max_bytes)
        if mime not in image_types:
            raise ToolUnavailable("this image type is disabled on the server")
        digest = hashlib.sha256(data).hexdigest()[:12]
        return data, mime, f"inline:{digest}", _clean_source_name(source_name)

    try:
        safe = resolve_and_check(image_url, allowed_schemes=("http", "https"))
    except BlockedTargetError:
        raise
    try:
        async with httpx.AsyncClient(timeout=cfg.website_fetch_timeout_seconds) as client:
            response = await client.get(safe.url)
            response.raise_for_status()
            if len(response.content) > max_bytes:
                raise ToolUnavailable("reference image exceeds the upload limit")
            declared_mime = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            data = response.content
    except httpx.HTTPError as exc:
        raise ToolUnavailable("could not fetch the reference image") from exc
    mime = _verify_image(data, declared_mime, max_bytes)
    if mime not in image_types:
        raise ToolUnavailable("this image type is disabled on the server")
    parts = urlsplit(safe.url)
    fallback = Path(parts.path).name or (parts.hostname or "Reference image")
    return data, mime, _source_label(safe.url), _clean_source_name(source_name, fallback)


def _parse_json_text(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("VLM response must be a JSON object")
    return parsed


async def extract_visual_style(
    image_bytes: bytes,
    mime_type: str,
    *,
    focus: str | None = None,
    ai_settings: AISettings | None = None,
) -> tuple[ReferenceVisionOutput, str]:
    """Send actual image bytes to the configured isolated Gemini/Ollama VLM."""
    cfg = ai_settings or get_ai_settings()
    provider = cfg.ads_vlm_provider
    if provider == "disabled":
        raise VisionModelUnavailable("Visual reference analysis is disabled (AI_ADS_VLM_PROVIDER=disabled).")

    instruction = (
        "Inspect the attached image itself. Return JSON matching the supplied schema. "
        "Describe only visible visual design: palette, composition, typography, tone, "
        "layout, hook, text density and visible brand elements. Quote observed offer "
        "or CTA text only when legible. Never infer performance or guess unreadable text; "
        "leave unknown fields empty. Do not recommend copying a distinctive brand or ad."
    )
    if focus:
        instruction += f" User focus: {focus[:300]}"
    b64 = base64.b64encode(image_bytes).decode("ascii")

    if provider == "gemini":
        api_key = cfg.gemini_api_key
        if not api_key:
            raise VisionModelUnavailable(
                "The isolated Gemini visual model is not configured; set AI_ADS_VLM_PROVIDER and GEMINI_API_KEY."
            )
        model = cfg.ads_vlm_model or cfg.gemini_model
        base = (cfg.ads_vlm_base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        body = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": instruction + "\nSchema:\n" + json.dumps(ReferenceVisionOutput.model_json_schema(), ensure_ascii=False)},
                    {"inline_data": {"mime_type": mime_type, "data": b64}},
                ],
            }],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
        }
        try:
            async with httpx.AsyncClient(timeout=cfg.ads_vlm_timeout_seconds) as client:
                response = await client.post(
                    f"{base}/models/{model}:generateContent",
                    params={"key": api_key},
                    json=body,
                )
            if response.is_error:
                raise VisionModelUnavailable(f"Gemini VLM request failed (HTTP {response.status_code}).")
            payload = response.json()
            candidates = payload.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            text = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        except VisionModelUnavailable:
            raise
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise VisionModelUnavailable("Gemini VLM request failed or returned an unreadable response.") from exc
        model_label = f"gemini:{model}"
    elif provider == "ollama":
        base = (cfg.ads_vlm_base_url or cfg.ollama_base_url or "").rstrip("/")
        model = cfg.ads_vlm_model
        if not base or not model:
            raise VisionModelUnavailable(
                "The isolated Ollama VLM needs AI_ADS_VLM_BASE_URL (or OLLAMA_BASE_URL) and a vision-capable AI_ADS_VLM_MODEL."
            )
        body = {
            "model": model,
            "messages": [{"role": "user", "content": instruction, "images": [b64]}],
            "format": ReferenceVisionOutput.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            async with httpx.AsyncClient(timeout=cfg.ads_vlm_timeout_seconds) as client:
                response = await client.post(
                    f"{base}/api/chat",
                    headers={"ngrok-skip-browser-warning": "true", "User-Agent": "sahm-ai/1.0"},
                    json=body,
                )
            if response.is_error:
                raise VisionModelUnavailable(f"Ollama VLM request failed (HTTP {response.status_code}).")
            text = str(response.json().get("message", {}).get("content", ""))
        except VisionModelUnavailable:
            raise
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise VisionModelUnavailable("Ollama VLM request failed or returned an unreadable response.") from exc
        model_label = f"ollama:{model}"
    else:  # guarded by AISettings, retained for defensive failure
        raise VisionModelUnavailable("Unsupported isolated VLM provider.")

    try:
        result = ReferenceVisionOutput.model_validate(_parse_json_text(text))
    except Exception as exc:
        raise VisionModelUnavailable("The visual model did not return a usable structured style analysis.") from exc
    return result, model_label


async def analyze_reference_source(
    db: AsyncSession,
    *,
    business_id: str,
    image_url: str | None,
    upload_id: str | None,
    source_name: str | None = None,
    campaign_id: str | None = None,
    focus: str | None = None,
    core_settings: Settings | None = None,
    ai_settings: AISettings | None = None,
) -> ReferenceAnalysis:
    """Read/analyse a reference image and persist only the reusable StyleSpec."""
    try:
        image, mime, label, name = await load_reference_image(
            image_url=image_url,
            upload_id=upload_id,
            source_name=source_name,
            core_settings=core_settings,
        )
        vision, model = await extract_visual_style(image, mime, focus=focus, ai_settings=ai_settings)
    except (ToolUnavailable, BlockedTargetError, VisionModelUnavailable) as exc:
        reason = getattr(exc, "message", None) or str(exc)
        return ReferenceAnalysis.unavailable(source="isolated_visual_model", reason=reason).model_copy(
            update={"source_name": _clean_source_name(source_name), "kind": "image"}
        )

    reference_id = f"ref_{uuid.uuid4().hex[:20]}"
    style_dict = vision.style.model_dump(mode="json")
    summary = vision.summary.strip()[:2000]
    text = " ".join(
        part for part in (
            f"Observed visual reference style: {summary}" if summary else "Observed visual reference style.",
            f"Palette: {', '.join(vision.style.palette)}" if vision.style.palette else "",
            f"Composition: {vision.style.composition}" if vision.style.composition else "",
            f"Tone: {vision.style.tone}" if vision.style.tone else "",
        ) if part
    )[:4000]
    row = BrandMemoryItem(
        id=reference_id,
        business_id=business_id,
        kind="reference_analysis",
        text=text,
        **memory_defaults(
            "reference_analysis",
            business_id=business_id,
            text=text,
            campaign_id=campaign_id,
            source="reference_analysis",
        ),
        metadata_={
            "schema_version": 1,
            "style": style_dict,
            "summary": summary,
            "source_label": _source_label(label),
            "source_name": name,
            "source_digest": hashlib.sha256(image).hexdigest()[:20],
            "favorite": False,
            "campaign_id": campaign_id,
            "kind": "image",
            "model": model,
            "observed_offers": vision.observed_offers,
            "observed_cta": vision.observed_cta,
        },
        campaign_id=campaign_id,
    )
    db.add(row)
    await db.flush()
    return ReferenceAnalysis(
        id=reference_id,
        source=_source_label(label),
        source_name=name,
        kind="image",
        status="ok",
        summary=summary,
        style=StyleSpec.model_validate(style_dict),
        observed_offers=vision.observed_offers,
        observed_cta=vision.observed_cta,
        model=model,
        created_at=datetime.now(UTC),
    )


__all__ = [
    "VisionModelUnavailable",
    "analyze_reference_source",
    "extract_visual_style",
    "load_reference_image",
]
