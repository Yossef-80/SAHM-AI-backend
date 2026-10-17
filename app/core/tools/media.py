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
import binascii
import hashlib
import io
import math
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from PIL import Image, UnidentifiedImageError

from app.ai.marketing.reference_vision import (
    ReferenceVisualExtraction as _StyleExtraction,
)
from app.ai.marketing.reference_vision import (
    analyze_reference_image,
)
from app.core.errors import (
    BlockedTargetError,
    ToolUnavailable,
    ToolValidationError,
)
from app.core.logging import get_logger
from app.core.security import resolve_and_check
from app.core.tools.base import Tool, ToolContext
from app.db.models import Creative
from app.schemas.common import ToolSideEffect
from app.schemas.creatives import (
    ReferenceAnalysis,
    StyleSpec,
)
from app.schemas.tools import (
    AnalyzeReferenceInput,
    GeneratedAssetRecord,
    GenerateImageInput,
    GenerateImageOutput,
)

_logger = get_logger(__name__)


# ================================================ 11. analyze_reference_creative


async def _load_reference_image(
    payload: AnalyzeReferenceInput, settings: Any
) -> tuple[str, str, str]:
    """Return (mime_type, base64_data, source_label) from a trusted upload or public URL."""
    cfg = settings
    if payload.upload_id:
        upload_root = Path(cfg.upload_dir).resolve()
        path = (upload_root / str(payload.upload_id)).resolve()
        if upload_root not in path.parents:
            raise ToolUnavailable("reference upload id is invalid")
        if not path.is_file():
            raise ToolUnavailable(f"upload '{payload.upload_id}' was not found")
        if path.stat().st_size > cfg.website_max_bytes:
            raise ToolUnavailable("reference image exceeds the size cap")
        declared_mime = _guess_mime(path.name)
        if declared_mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ToolUnavailable("reference upload must be PNG, JPEG, or WebP")
        data = path.read_bytes()
        mime = _sniff_image_mime(data)
        if mime != declared_mime:
            raise ToolUnavailable("reference upload content does not match its image file type")
        return mime, base64.b64encode(data).decode(), f"upload:{payload.upload_id}"

    if payload.image_url:
        safe = resolve_and_check(payload.image_url, allowed_schemes=("http", "https"))
        # Redirects are deliberately not followed: a second URL would need a
        # fresh SSRF/DNS check before any request could be made to it.
        async with httpx.AsyncClient(
            timeout=cfg.website_fetch_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            async with client.stream("GET", safe.url) as resp:
                resp.raise_for_status()
                mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].lower()
                if mime not in {"image/png", "image/jpeg", "image/webp"}:
                    raise ToolUnavailable(f"url did not return a supported image (got {mime})")
                content_length = resp.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > cfg.website_max_bytes:
                            raise ToolUnavailable("reference image exceeds the size cap")
                    except ValueError as exc:
                        raise ToolUnavailable("reference image returned an invalid size header") from exc
                chunks: list[bytes] = []
                size = 0
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > cfg.website_max_bytes:
                        raise ToolUnavailable("reference image exceeds the size cap")
                    chunks.append(chunk)
                data = b"".join(chunks)
        if _sniff_image_mime(data) != mime:
            raise ToolUnavailable("reference image content does not match its content type")
        # Keep only a private random file ID for later image-conditioned edits;
        # never persist a user URL that could contain signed query credentials.
        private_id = _persist_private_reference_image(data, mime, cfg)
        return mime, base64.b64encode(data).decode(), f"upload:{private_id}"

    raise ToolUnavailable("either imageUrl or uploadId must be provided")


def _persist_private_reference_image(data: bytes, mime: str, cfg: Any) -> str:
    extension = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(mime)
    if extension is None or not data or len(data) > cfg.website_max_bytes:
        raise ToolUnavailable("reference image bytes are not safe to retain")
    upload_root = Path(cfg.upload_dir).resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    file_id = f"reference_{uuid.uuid4().hex}.{extension}"
    target = (upload_root / file_id).resolve()
    if upload_root not in target.parents:
        raise ToolUnavailable("reference file destination is invalid")
    try:
        with target.open("xb") as handle:
            handle.write(data)
    except OSError as exc:
        raise ToolUnavailable("reference image could not be stored safely") from exc
    return file_id


def persist_inline_reference_image(data_url: str, cfg: Any) -> str:
    """Validate and retain a small browser-uploaded data URL as a private file.

    The browser sends only a data URL, never a user-controlled filesystem path.
    The returned opaque filename is later consumed by ``_load_reference_image``.
    """
    match = re.fullmatch(
        r"data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/]+={0,2})",
        data_url.strip(),
        flags=re.IGNORECASE | re.ASCII,
    )
    if not match:
        raise ToolUnavailable("reference upload must be a base64 PNG, JPEG, or WebP image")

    declared_mime = match.group(1).lower()
    encoded = match.group(2)
    max_encoded_chars = ((int(cfg.website_max_bytes) + 2) // 3) * 4
    if len(encoded) > max_encoded_chars:
        raise ToolUnavailable("reference image exceeds the size cap")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ToolUnavailable("reference image data is not valid base64") from exc
    if not data or len(data) > int(cfg.website_max_bytes):
        raise ToolUnavailable("reference image is empty or exceeds the size cap")
    if _sniff_image_mime(data) != declared_mime:
        raise ToolUnavailable("reference image content does not match its declared file type")
    return _persist_private_reference_image(data, declared_mime, cfg)


MAX_UPLOADED_CREATIVE_BYTES = 10 * 1024 * 1024


def encode_uploaded_creative_image(
    data: bytes, *, max_bytes: int = MAX_UPLOADED_CREATIVE_BYTES
) -> tuple[str, int, int]:
    """Validate raw uploaded bytes and return ``(data_url, width, height)``.

    Trusts the file signature, never the extension or the declared content type:
    a text file renamed to ``.png`` is rejected. The image is then decoded once
    through Pillow, which is what actually proves it is an image rather than
    something that merely starts with a PNG header.
    """
    if not data:
        raise ToolValidationError("uploaded creative is empty")
    if len(data) > max_bytes:
        raise ToolValidationError(
            f"uploaded creative exceeds the {max_bytes // (1024 * 1024)} MB size cap"
        )
    if not _sniff_image_mime(data):
        raise ToolValidationError("uploaded creative must be a PNG, JPEG, or WebP image")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            width, height = int(img.size[0]), int(img.size[1])
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ToolValidationError("uploaded file is not a readable image") from exc

    mime = _sniff_image_mime(data)
    return f"data:{mime};base64,{base64.b64encode(data).decode()}", width, height


#: The ratios the product (and Meta's placements) actually work in. Snapping to
#: these rather than reducing the raw gcd matters: the UI validates this field
#: against exactly this enum, so a literal "1234:567" would fail to parse.
CANONICAL_ASPECT_RATIOS: tuple[tuple[str, float], ...] = (
    ("1:1", 1.0),
    ("4:5", 4 / 5),
    ("9:16", 9 / 16),
    ("16:9", 16 / 9),
)


def aspect_ratio_label(width: int, height: int) -> str:
    """Snap real pixel dimensions to the nearest canonical ``W:H`` label."""
    if width <= 0 or height <= 0:
        return "1:1"
    ratio = width / height
    return min(CANONICAL_ASPECT_RATIOS, key=lambda item: abs(item[1] - ratio))[0]


def _guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "")


def _sniff_image_mime(data: bytes) -> str:
    """Recognize supported raster formats from their file signatures."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return ""


async def _vision_extract(
    image_b64: str, mime: str, focus: str | None
) -> _StyleExtraction:
    """Pass the validated image to app.ai's independently configured VLM."""
    return await analyze_reference_image(image_b64=image_b64, mime=mime, focus=focus)


async def analyze_reference_for_workspace(
    payload: AnalyzeReferenceInput,
    *,
    db: Any,
    business_id: str,
    settings: Any,
) -> ReferenceAnalysis:
    """API/storage bridge for Layla's image analysis; model selection stays in app.ai."""
    source = "analyze_reference_creative:vision_llm"
    try:
        mime, image_b64, label = await _load_reference_image(payload, settings)
    except ToolUnavailable as exc:
        return ReferenceAnalysis.unavailable(source=source, reason=exc.message)
    except BlockedTargetError as exc:
        return ReferenceAnalysis.unavailable(source=source, reason=exc.message)
    except httpx.HTTPError:
        return ReferenceAnalysis.unavailable(
            source=source,
            reason="could not fetch the reference image; verify that the public image URL is reachable",
        )

    try:
        extraction = await _vision_extract(image_b64, mime, payload.focus)
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
        canvas_ratio=extraction.canvas_ratio,
        background=extraction.background,
        lighting=extraction.lighting,
        visible_text=extraction.visible_text,
        elements=[element.model_dump() for element in extraction.elements],
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

    # Persistence is kept on this narrow DB bridge; image interpretation and
    # provider/model selection live in app.ai.marketing.reference_vision.
    memory_id: str | None = None
    try:
        from app.core.memory.store import BrandMemoryStore

        store = BrandMemoryStore(db)
        item = await store.write_item(
            business_id=business_id,
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
                "source_name": (payload.source_name or "Saved reference")[:160],
                "summary": extraction.summary[:2000],
                "favorite": False,
                "campaign_id": payload.campaign_id,
            },
            campaign_id=payload.campaign_id,
        )
        memory_id = item.id
    except Exception as exc:  # pragma: no cover - memory write must not fail the tool
        _logger.warning("reference_memory_write_failed", extra={"error": str(exc)})

    return ReferenceAnalysis(
        id=memory_id or "",
        source=label,
        source_name=(payload.source_name or "Saved reference")[:160],
        status="ok" if memory_id else "partial",
        unavailable_reason=(
            None if memory_id else "The image was analyzed, but the result could not be saved for later reuse."
        ),
        summary=extraction.summary,
        style=style,
        observed_offers=extraction.observed_offers,
        observed_cta=extraction.observed_cta,
        created_at=datetime.now(UTC),
    )


async def _handle_analyze_reference(
    payload: AnalyzeReferenceInput, ctx: ToolContext
) -> ReferenceAnalysis:
    return await analyze_reference_for_workspace(
        payload,
        db=ctx.db,
        business_id=ctx.business_id,
        settings=ctx.require_settings(),
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
    prompt: str,
    aspect_ratio: str,
    *,
    api_key: str,
    model: str,
    base_url: str | None = None,
    reference_image: tuple[str, bytes] | None = None,
) -> list[GeneratedAssetRecord]:
    """Call the configured OpenAI-compatible generation or image-edit endpoint.

    The same configured provider, API key and model are retained. If a reference
    is present, its bytes are sent as image-edit input; it is never reused as the
    final asset. Image-edit models that return base64 are persisted separately by
    the local asset store below.
    """
    endpoint = "/images/edits" if reference_image else "/images/generations"
    url = (base_url or "https://api.openai.com/v1").rstrip("/") + endpoint
    is_real_openai = "api.openai.com" in url
    model_lower = str(model).lower()

    if is_real_openai and "gpt-image" in model_lower:
        size = {"1:1": "1024x1024", "4:5": "1024x1536", "9:16": "1024x1536", "16:9": "1536x1024"}.get(
            aspect_ratio, "1024x1024"
        )
    elif is_real_openai and "dall-e-3" in model_lower:
        size = {"1:1": "1024x1024", "4:5": "1024x1792", "9:16": "1024x1792", "16:9": "1792x1024"}.get(
            aspect_ratio, "1024x1024"
        )
    else:
        size = {"1:1": "1024x1024", "4:5": "1024x1280", "9:16": "1024x1792", "16:9": "1792x1024"}.get(
            aspect_ratio, "1024x1024"
        )
    headers = {"authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        if reference_image:
            mime, image_bytes = reference_image
            extension = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(mime, "png")
            resp = await client.post(
                url,
                headers=headers,
                data={"model": model, "prompt": prompt, "size": size, "n": "1"},
                files={"image": (f"reference.{extension}", image_bytes, mime)},
            )
        else:
            resp = await client.post(
                url,
                headers=headers,
                json={"model": model, "prompt": prompt, "size": size, "n": 1},
            )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            _logger.error(
                "openai_image_call_failed",
                extra={
                    "status_code": resp.status_code,
                    "error_text": resp.text[:1000],
                    "model": model,
                    "size": size,
                    "endpoint": endpoint,
                },
            )
            raise
        data = resp.json().get("data", [])
    assets: list[GeneratedAssetRecord] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        asset_url = item.get("url") or ""
        if not asset_url and isinstance(item.get("b64_json"), str):
            output_mime = str(item.get("mime_type") or "image/png").lower()
            asset_url = f"data:{output_mime};base64,{item['b64_json']}"
        assets.append(
            GeneratedAssetRecord(
                id=f"asset_{uuid.uuid4().hex[:16]}",
                url=asset_url,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                provider="openai",
                model=model,
            )
        )
    return assets


def _persist_generated_data_url(data_url: str, cfg: Any) -> str:
    """Validate file bytes and save a provider-returned image without AI analysis."""
    from app.ai.marketing.meta_ads import validate_creative_image_url

    header, separator, encoded = data_url.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise ToolUnavailable("The image provider returned an unsupported image format.")
    declared_mime = header[5:].split(";", 1)[0].lower()
    if declared_mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ToolUnavailable("Only PNG, JPEG, or WebP final images can be saved for Meta.")
    try:
        content = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ToolUnavailable("The image provider returned invalid image bytes.") from exc
    if not content or len(content) > cfg.upload_max_bytes:
        raise ToolUnavailable("The generated final image is empty or exceeds the configured size limit.")
    sniffed = _sniff_image_mime(content)
    if sniffed != declared_mime:
        raise ToolUnavailable("Generated image bytes do not match their declared file type.")

    base = str(cfg.public_asset_base_url or "").strip()
    parsed = urlparse(base)
    if (
        not validate_creative_image_url(base)
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ToolUnavailable(
            "A public HTTPS asset origin is not configured. Set PUBLIC_ASSET_BASE_URL; "
            "no final image was persisted."
        )

    extension = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[declared_mime]
    asset_id = f"asset_{uuid.uuid4().hex}"
    root = (Path(cfg.upload_dir).resolve() / "public_assets").resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root / f"{asset_id}.{extension}").resolve()
    if root not in target.parents:
        raise ToolUnavailable("The generated image destination is invalid.")
    try:
        with target.open("xb") as handle:
            handle.write(content)
    except OSError as exc:
        raise ToolUnavailable("The generated image could not be saved safely.") from exc
    return f"{base.rstrip('/')}{cfg.api_prefix}/media/public-assets/{asset_id}.{extension}"


async def generate_image_for_workspace(
    payload: GenerateImageInput,
    *,
    db: Any,
    business_id: str,
    settings: Any,
    campaign_id: str | None = None,
    persona_id: str | None = None,
    allow_mock_assets: bool = False,
) -> GenerateImageOutput:
    """Generate/persist a creative using explicit API/storage dependencies.

    This is the HTTP bridge used by /creatives; unlike the tool adapter below it
    does not construct or require a core ToolContext.
    """
    from app.core.tools.settings import load_connector_settings

    cfg = settings
    tools_cfg = await load_connector_settings(db, business_id, cfg)
    image_provider = tools_cfg.image_provider or cfg.image_provider
    image_key = tools_cfg.image_api_key or cfg.image_api_key
    image_model = tools_cfg.image_model or cfg.image_model
    image_base = tools_cfg.image_base_url or cfg.image_base_url
    source = f"generate_image:{image_provider}"

    style = dict(payload.style_spec or {})
    reference_image: tuple[str, bytes] | None = None
    if payload.reference_analysis_id or payload.reference_image_url:
        try:
            style, reference_image = await _reference_material(
                payload,
                db=db,
                business_id=business_id,
                settings=cfg,
                initial_style=style,
            )
        except (ToolUnavailable, BlockedTargetError) as exc:
            return GenerateImageOutput.unavailable(source=source, reason=exc.message)
        except httpx.HTTPError:
            return GenerateImageOutput.unavailable(
                source=source,
                reason="The reference image could not be fetched safely; no final image was generated.",
            )
        if reference_image is None:
            return GenerateImageOutput.unavailable(
                source=source,
                reason="The reference image is unavailable; no text-only substitute was generated.",
            )

    effective_prompt = _compose_image_prompt(
        payload.prompt, style, has_reference=reference_image is not None
    )

    try:
        if image_provider == "openai" and image_key:
            assets = await _generate_openai_image(
                effective_prompt,
                payload.aspect_ratio,
                api_key=image_key,
                model=image_model,
                base_url=image_base,
                reference_image=reference_image,
            )
        elif reference_image is not None:
            return GenerateImageOutput.unavailable(
                source=source,
                reason=(
                    "The configured image provider does not support reference-conditioned edits "
                    "in this build. No text-only substitute was generated."
                ),
            )
        elif image_provider == "stability":
            return GenerateImageOutput.unavailable(
                source=source,
                reason=(
                    "The stability image provider is configured but not wired up in "
                    "this build. Select a supported image provider; no placeholder was created."
                ),
            )
        elif image_provider == "mock":
            if not cfg.image_allow_mock_assets and not allow_mock_assets:
                return GenerateImageOutput.unavailable(
                    source=source,
                    reason=(
                        "Mock placeholder images are disabled by default. Configure a real "
                        "image provider, or explicitly enable mock assets for a test workspace."
                    ),
                )
            assets = [
                _mock_asset(effective_prompt, payload.aspect_ratio, i)
                for i in range(payload.num_images)
            ]
        else:
            return GenerateImageOutput.unavailable(
                source=source,
                reason="No supported image provider with credentials is configured; no image was created.",
            )
    except httpx.HTTPError as exc:
        return GenerateImageOutput.unavailable(
            source=source, reason=f"image generation failed: {exc}"
        )
    except ToolUnavailable as exc:
        return GenerateImageOutput.unavailable(source=source, reason=exc.message)

    if not assets:
        return GenerateImageOutput.unavailable(
            source=source, reason="The image provider returned no final asset."
        )

    from app.ai.marketing.meta_ads import validate_creative_image_url

    for asset in assets:
        if asset.provider == "mock":
            continue
        if asset.url.startswith("data:image/"):
            try:
                asset.url = _persist_generated_data_url(asset.url, cfg)
            except ToolUnavailable as exc:
                return GenerateImageOutput.unavailable(source=source, reason=exc.message)
        if not validate_creative_image_url(asset.url):
            return GenerateImageOutput.unavailable(
                source=source,
                reason="The final image is not available at a public HTTPS URL; it cannot be safely attached to a Meta ad.",
            )

    persisted: list[GeneratedAssetRecord] = []
    for asset in assets:
        asset.style_spec = style
        try:
            db.add(
                Creative(
                    id=asset.id,
                    business_id=business_id,
                    campaign_id=payload.campaign_id or campaign_id,
                    persona_id=persona_id,
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
            await db.flush()
        except Exception as exc:  # pragma: no cover
            _logger.warning("creative_persist_failed", extra={"error": str(exc)})
        persisted.append(asset)

    return GenerateImageOutput(assets=persisted, status="ok", source=source)


async def _handle_generate_image(
    payload: GenerateImageInput, ctx: ToolContext
) -> GenerateImageOutput:
    """Compatibility adapter for the typed core tool registry."""
    return await generate_image_for_workspace(
        payload,
        db=ctx.db,
        business_id=ctx.business_id,
        settings=ctx.require_settings(),
        campaign_id=ctx.campaign_id,
        persona_id=ctx.persona_id,
        allow_mock_assets=bool((ctx.user_supplied or {}).get("allow_mock_assets")),
    )


def _compose_image_prompt(
    prompt: str, style: dict[str, Any], *, has_reference: bool = False
) -> str:
    """Use the app.ai reference-faithful prompt when the actual source is attached."""
    if has_reference:
        from app.ai.marketing.creative_prompt import compose_reference_faithful_prompt

        return compose_reference_faithful_prompt(prompt, style)
    if not style:
        return prompt
    parts = [prompt, "", "STYLE OBSERVATIONS (follow as closely as possible):"]
    mapping = [
        ("canvas_ratio", "Canvas ratio"),
        ("background", "Background"),
        ("lighting", "Lighting"),
        ("palette", "Palette"),
        ("composition", "Composition"),
        ("typography", "Typography"),
        ("tone", "Tone"),
        ("layout_pattern", "Layout pattern"),
        ("hook_type", "Hook type"),
        ("color_mood", "Colour mood"),
        ("text_density", "Text density"),
        ("visible_text", "Visible text (verbatim if legible)"),
        ("elements", "Observed element layout"),
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


async def _reference_material(
    payload: GenerateImageInput,
    *,
    db: Any,
    business_id: str,
    settings: Any,
    initial_style: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, bytes] | None]:
    """Resolve a workspace-owned reference using explicit storage dependencies."""
    source_label: str | None = None
    style = dict(initial_style)
    if payload.reference_analysis_id:
        from sqlalchemy import select

        from app.db.models import BrandMemoryItem

        item = await db.scalar(
            select(BrandMemoryItem).where(
                BrandMemoryItem.id == payload.reference_analysis_id,
                BrandMemoryItem.business_id == business_id,
                BrandMemoryItem.kind == "reference_analysis",
            )
        )
        if item is None:
            raise ToolUnavailable("The selected reference analysis is not available in this workspace.")
        metadata = item.metadata_ if isinstance(item.metadata_, dict) else {}
        stored_style = metadata.get("style")
        if isinstance(stored_style, dict):
            style = {**style, **stored_style}
        source_label = metadata.get("source") if isinstance(metadata.get("source"), str) else None
        if not source_label:
            raise ToolUnavailable("The original reference image source was not retained; re-analyse the reference.")
        if payload.reference_image_url:
            if not source_label.startswith("url:") or payload.reference_image_url != source_label[4:]:
                raise ToolUnavailable("The supplied reference URL does not match the selected reference analysis.")

    if source_label and source_label.startswith("url:"):
        reference_request = AnalyzeReferenceInput(
            image_url=source_label[4:], campaign_id=payload.campaign_id
        )
    elif source_label and source_label.startswith("upload:"):
        reference_request = AnalyzeReferenceInput(
            upload_id=source_label[7:], campaign_id=payload.campaign_id
        )
    elif payload.reference_image_url:
        reference_request = AnalyzeReferenceInput(
            image_url=payload.reference_image_url, campaign_id=payload.campaign_id
        )
    else:
        return style, None

    mime, image_b64, _label = await _load_reference_image(reference_request, settings)
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except Exception as exc:
        raise ToolUnavailable("The stored reference image is invalid.") from exc
    return style, (mime, image_bytes)


generate_image_tool = Tool(
    name="generate_image",
    description=(
        "Generate a new final image from a prompt. For a close match to a reference, "
        "first call analyze_reference_creative and pass its reference_analysis_id; "
        "the original reference image is then supplied separately to the configured "
        "image-edit endpoint together with structured layout/text observations. The "
        "output is a different asset. Never analyze the final image after generation."
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
    "analyze_reference_for_workspace",
    "generate_image_for_workspace",
    "generate_image_tool",
]
