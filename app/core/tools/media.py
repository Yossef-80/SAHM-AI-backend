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

import asyncio
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
            # Local disk is wiped on every deploy; the mirrored copy is what
            # keeps an attachment usable without asking the user to re-upload.
            from app.core.storage import fetch_private_asset

            mirrored = await fetch_private_asset(cfg, str(payload.upload_id))
            if mirrored is None:
                raise ToolUnavailable(f"upload '{payload.upload_id}' was not found")
            mime = _sniff_image_mime(mirrored)
            if mime not in {"image/png", "image/jpeg", "image/webp"}:
                raise ToolUnavailable("reference upload is not a supported image")
            return mime, base64.b64encode(mirrored).decode(), f"upload:{payload.upload_id}"
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


async def mirror_uploaded_reference(file_id: str, data: bytes, mime: str, cfg: Any) -> None:
    """Mirror a locally stored upload into Supabase Storage (best effort).

    Never raises: the file is already on local disk at this point, so a storage
    outage must not turn a successful upload into an error for the user.
    """
    try:
        from app.core.storage import mirror_private_asset

        await mirror_private_asset(cfg, file_id, data, mime)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("upload_mirror_failed", extra={"error": str(exc)})


def _decode_inline_image(data_url: str, cfg: Any) -> tuple[bytes, str]:
    """Validate a browser data URL and return ``(bytes, mime)``."""
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
    return data, declared_mime


def persist_inline_reference_image(data_url: str, cfg: Any) -> str:
    """Validate and retain a small browser-uploaded data URL as a private file.

    The browser sends only a data URL, never a user-controlled filesystem path.
    The returned opaque filename is later consumed by ``_load_reference_image``.
    """
    data, declared_mime = _decode_inline_image(data_url, cfg)
    return _persist_private_reference_image(data, declared_mime, cfg)


async def persist_inline_reference_image_async(data_url: str, cfg: Any) -> str:
    """``persist_inline_reference_image`` plus the Supabase Storage mirror.

    Prefer this from request handlers: it is the version that survives a deploy,
    because the local copy under ``upload_dir`` does not.
    """
    data, declared_mime = _decode_inline_image(data_url, cfg)
    file_id = _persist_private_reference_image(data, declared_mime, cfg)
    await mirror_uploaded_reference(file_id, data, declared_mime, cfg)
    return file_id


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


#: The placeholder model id ``config.py`` ships as a default. Sending it to a
#: real provider is always a configuration mistake, never an intent.
_MOCK_MODEL = "mock-image-1"


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


#: The render ceiling for one provider call. 120s was *below* the provider's
#: observed worst case for a large, high-quality edit (over two minutes), so a
#: render that was about to succeed was cut off -- and still billed.
_DEFAULT_IMAGE_TIMEOUT_SECONDS = 300.0

#: A retry is only paid for when the request provably did not run: a rate limit,
#: a 5xx, or a connection that never opened. A *timeout* is deliberately not
#: retried -- the render may still be completing and billing -- so it is
#: surfaced instead of repeated.
_IMAGE_RETRY_ATTEMPTS = 3


def _image_timeout_seconds(cfg: Any) -> float:
    """How long one provider call may take, from ``IMAGE_TIMEOUT_SECONDS``."""
    try:
        value = float(getattr(cfg, "image_timeout_seconds", None))
    except (TypeError, ValueError):
        value = 0.0
    return value if value >= 30.0 else _DEFAULT_IMAGE_TIMEOUT_SECONDS


def _image_retry_delay(response: "httpx.Response | None", attempt: int) -> float:
    """Seconds to wait before the next attempt: the provider's own hint first."""
    header = ""
    if response is not None:
        header = response.headers.get("retry-after", "") or ""
    try:
        hinted = float(header)
    except ValueError:
        hinted = 0.0
    if hinted > 0:
        return min(hinted, 30.0)
    return float(min(2 ** (attempt - 1), 8))


def _provider_error_reason(response: "httpx.Response") -> str:
    """One actionable sentence for a failed provider call.

    The only useful explanation lives in the provider's JSON body
    (``error.code`` / ``error.message``): a safety-filter rejection and a
    misspelled ``size`` are both HTTP 400. ``raise_for_status`` threw that away
    and surfaced ``Client error '400 Bad Request'``, which told the user nothing
    and the operator less.
    """
    status = response.status_code
    try:
        body = response.json()
    except ValueError:
        body = None
    error = (body or {}).get("error") if isinstance(body, dict) else None
    error = error if isinstance(error, dict) else {}
    code = str(error.get("code") or "").strip()
    message = " ".join(str(error.get("message") or "").split())[:300]
    haystack = f"{code} {message}".lower()

    if status in (401, 403):
        return (
            "The image provider rejected the configured API key; check the image "
            "credentials in Settings -> Connectors."
        )
    if status == 429:
        return (
            "The image provider is rate-limiting this key; the request was not "
            "charged. Try again in a moment."
        )
    if "moderation" in haystack or "content_policy" in haystack or "safety" in haystack:
        return "The image provider's safety filter rejected this prompt; rephrase it and try again."
    if status == 404:
        suffix = f": {message}" if message else ""
        return f"The image provider does not recognise the configured model{suffix}; check IMAGE_MODEL."
    if 500 <= status <= 599:
        return f"The image provider had a temporary failure (HTTP {status}); try again shortly."
    if message:
        return f"The image provider rejected the request: {message}"
    return f"The image provider rejected the request (HTTP {status})."


def _usage_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Token accounting for one image call, when the provider reports it.

    OpenAI's image models report ``usage`` (text tokens, image tokens, output
    tokens) and this was dropped on the floor, so the usage screen could only
    ever account for text calls. Flattened, because this feeds a log line.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}
    summary: dict[str, Any] = {}
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            summary[str(key)] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                    summary[f"{key}_{sub_key}"] = sub_value
    return summary


async def _generate_openai_image(
    prompt: str,
    aspect_ratio: str,
    *,
    api_key: str,
    model: str,
    base_url: str | None = None,
    reference_image: tuple[str, bytes] | None = None,
    quality: str | None = None,
    count: int = 1,
    timeout: float | None = None,
    attempts: int = _IMAGE_RETRY_ATTEMPTS,
    sleep: Any | None = None,
) -> list[GeneratedAssetRecord]:
    """Call the configured OpenAI-compatible generation or image-edit endpoint.

    The same configured provider, API key and model are retained. If a reference
    is present, its bytes are sent as image-edit input; it is never reused as the
    final asset. Image-edit models that return base64 are persisted separately by
    the local asset store below.

    ``size`` and ``quality`` are resolved per model family, because the two
    generations of OpenAI image models disagree about both. The single mapping
    this used to send (``1024x1280``, ``1024x1792``, ``1792x1024``) is DALL-E's,
    and a GPT Image model rejects it outright: it accepts only ``1024x1024``,
    ``1536x1024`` and ``1024x1536``. ``quality`` was never sent at all, which
    silently bought the provider default -- ``high`` on GPT Image, the expensive
    one -- for every workspace.

    ``n`` is the resolved ``count``, clamped to what the model accepts
    (``dall-e-3`` is ``n=1``). It previously sent a hardcoded ``1`` while the
    schema and the UI offered up to four, so a request for four renders returned
    one and reported nothing. The product now renders a single image, and that
    decision is enforced in :func:`generate_creative_assets` -- the one place
    that can spend money -- so the number sent and the number promised come from
    the same value.

    ``input_fidelity`` is deliberately **not** sent. The provider default is
    ``low``; ``high`` adds a per-image charge for copying faces and logo detail,
    which has to be an explicit product decision rather than a silent default.

    Failures that provably never ran (429, 5xx, refused connection) are retried;
    anything else is raised as :class:`ToolUnavailable` carrying the provider's
    own explanation.
    """
    from app.core.llm.base import is_retryable_status
    from app.core.llm.model_compat import (
        bare_model_name,
        image_quality_for_model,
        image_size_for_model,
    )

    size = image_size_for_model(aspect_ratio, model)
    resolved_quality = image_quality_for_model(quality, model)
    endpoint = "/images/edits" if reference_image else "/images/generations"
    url = (base_url or "https://api.openai.com/v1").rstrip("/") + endpoint
    headers = {"authorization": f"Bearer {api_key}"}
    optional: dict[str, object] = {"quality": resolved_quality} if resolved_quality else {}
    wanted = 1 if bare_model_name(model).startswith("dall-e-3") else max(1, min(int(count or 1), 10))
    pause = sleep or asyncio.sleep
    # A dead host fails in seconds; only the render itself gets the long ceiling.
    client_timeout = httpx.Timeout(float(timeout or _DEFAULT_IMAGE_TIMEOUT_SECONDS), connect=15.0)
    tries = max(1, int(attempts))

    async with httpx.AsyncClient(timeout=client_timeout, trust_env=False) as client:
        response: httpx.Response | None = None
        for attempt in range(1, tries + 1):
            try:
                if reference_image:
                    mime, image_bytes = reference_image
                    extension = {
                        "image/png": "png",
                        "image/jpeg": "jpg",
                        "image/webp": "webp",
                    }.get(mime, "png")
                    response = await client.post(
                        url,
                        headers=headers,
                        data={
                            "model": model,
                            "prompt": prompt,
                            "size": size,
                            "n": str(wanted),
                            **optional,
                        },
                        files={"image": (f"reference.{extension}", image_bytes, mime)},
                    )
                else:
                    response = await client.post(
                        url,
                        headers=headers,
                        json={"model": model, "prompt": prompt, "size": size, "n": wanted, **optional},
                    )
            except httpx.TimeoutException as exc:
                # Not retried on purpose: the render may still be completing and
                # billing, and a second request would be paid for twice.
                raise ToolUnavailable(
                    "The image provider did not answer in time; the image may still be "
                    "rendering on their side, so no second attempt was made."
                ) from exc
            except httpx.ConnectError as exc:
                if attempt >= tries:
                    raise ToolUnavailable(
                        "Could not reach the image provider; check the network and the "
                        "configured image base URL."
                    ) from exc
                delay = _image_retry_delay(None, attempt)
                _logger.warning(
                    "image_generation_retry",
                    extra={"attempt": attempt, "reason": type(exc).__name__, "delay": delay},
                )
                await pause(delay)
                continue
            except httpx.HTTPError as exc:
                raise ToolUnavailable(f"The image provider call failed: {type(exc).__name__}.") from exc

            if response.status_code < 400:
                break
            if is_retryable_status(response.status_code) and attempt < tries:
                delay = _image_retry_delay(response, attempt)
                _logger.warning(
                    "image_generation_retry",
                    extra={
                        "attempt": attempt,
                        "status": response.status_code,
                        "delay": delay,
                    },
                )
                await pause(delay)
                continue
            raise ToolUnavailable(_provider_error_reason(response))

    if response is None:  # pragma: no cover - the loop always assigns or raises
        raise ToolUnavailable("The image provider was never called.")

    try:
        payload = response.json()
    except ValueError as exc:
        # A gateway or proxy answering with HTML instead of JSON used to escape
        # as an unhandled ValueError and become a 500 on the success path.
        raise ToolUnavailable(
            "The image provider returned a response that is not JSON; no image was created."
        ) from exc
    if not isinstance(payload, dict):
        raise ToolUnavailable("The image provider returned an unexpected payload; no image was created.")

    data = payload.get("data", [])
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

    summary = _usage_summary(payload)
    if summary:
        _logger.info(
            "image_generation_usage",
            extra={
                "provider": "openai",
                "model": model,
                "size": size,
                "quality": resolved_quality,
                "assets": len(assets),
                **summary,
            },
        )
    return assets


def _is_durable_asset_url(url: str, cfg: Any) -> bool:
    """True when this URL is already hosted on our own storage (no re-upload needed)."""
    from app.core.storage import DEFAULT_BUCKET, _supabase_url

    base = _supabase_url(cfg)
    if base and f"/storage/v1/object/public/{DEFAULT_BUCKET}/" in url:
        return True
    asset_base = str(getattr(cfg, "public_asset_base_url", "") or "").strip().rstrip("/")
    return bool(asset_base) and url.startswith(f"{asset_base}/")


async def _persist_generated_asset(asset: GeneratedAssetRecord, cfg: Any) -> str:
    """Move a generated image out of the provider response and into our storage.

    Provider URLs are temporary (OpenAI's expire in about an hour) and base64
    data URLs are megabytes of text: neither belongs in a database row. This
    hands the bytes to ``app.core.storage`` (Supabase Storage, local fallback)
    and returns the durable public URL to store instead.
    """
    from app.ai.marketing.meta_ads import validate_creative_image_url
    from app.core.storage import DEFAULT_BUCKET, _supabase_url, persist_image_asset

    if asset.provider != "mock" and asset.url.startswith("https://"):
        base = _supabase_url(cfg)
        if base and asset.url.startswith(base):
            return asset.url
        if not validate_creative_image_url(asset.url):
            raise ToolUnavailable(
                "The image provider returned a non-HTTPS asset URL; it cannot be used for a Meta ad."
            )

    url, error = await persist_image_asset(
        cfg,
        asset_url=asset.url,
        path_prefix=f"generated/{asset.provider or 'unknown'}",
        bucket=DEFAULT_BUCKET,
    )
    if not url:
        raise ToolUnavailable(error or "The generated image could not be persisted.")
    return url


async def generate_creative_assets(
    *,
    cfg: Any,
    prompt: str,
    aspect_ratio: str,
    num_images: int = 1,
    style: dict[str, Any] | None = None,
    reference_image: tuple[str, bytes] | None = None,
    provider: str,
    api_key: str | None,
    model: str | None,
    base_url: str | None = None,
    quality: str | None = None,
    allow_mock_assets: bool = False,
    image_call: Any | None = None,
) -> tuple[list[GeneratedAssetRecord], str | None]:
    """The single image-generation entry point for the whole app.

    Every caller -- the ``/creatives`` HTTP route, Layla's chat tool and the
    marketing pipeline -- goes through here so that:

    * the provider call happens exactly once, in one place;
    * every returned ``url`` is already durable (Supabase Storage), because
      provider URLs expire (OpenAI's in about an hour) and base64 data URLs are
      megabytes of text that must never reach a database row;
    * ``mock`` assets are only produced when a workspace explicitly allows them;
    * **one** real image is rendered per request: ``num_images`` still describes
      what a caller may ask for (and drives the offline mock, which is free), but
      a paid render is capped to a single asset here, with a log line when a
      larger number was requested. The count is enforced at the only point that
      can spend money rather than in each caller.

    Returns ``(assets, reason)``; ``reason`` is set and ``assets`` empty when the
    request could not be fulfilled. Provider failures never raise out of here: a
    surprise from the provider becomes a reason, because a 500 after the user has
    waited a minute is the least useful outcome.
    """
    from app.ai.marketing.meta_ads import validate_creative_image_url

    style = dict(style or {})
    effective_prompt = _compose_image_prompt(
        prompt, style, has_reference=reference_image is not None
    )

    assets: list[GeneratedAssetRecord] = []
    try:
        if provider == "openai" and api_key:
            if not model or model == _MOCK_MODEL:
                return [], "The configuration matched a placeholder image model; set IMAGE_MODEL."
            if int(num_images or 1) > 1:
                _logger.info(
                    "image_count_capped",
                    extra={"requested": int(num_images), "generated": 1, "provider": provider},
                )
            # ``image_call`` is the provider seam: the chat tool passes its own
            # module-level reference so the call can be stubbed in tests without
            # a second copy of this logic existing anywhere.
            call = image_call or _generate_openai_image
            assets = await call(
                effective_prompt,
                aspect_ratio,
                api_key=api_key,
                model=model,
                base_url=base_url,
                reference_image=reference_image,
                quality=quality,
                count=1,
                timeout=_image_timeout_seconds(cfg),
            )
            # A gateway (or a provider that ignores ``n``) can return more items
            # than were asked for. One paid render must not become four persisted
            # creatives, so the response is capped to what this call requested.
            if len(assets) > 1:
                _logger.info(
                    "image_assets_truncated",
                    extra={"returned": len(assets), "kept": 1, "model": model},
                )
                assets = assets[:1]
        elif reference_image is not None:
            return [], (
                "The configured image provider does not support reference-conditioned edits "
                "in this build. No text-only substitute was generated."
            )
        elif provider == "stability":
            return [], (
                "The stability image provider is configured but not wired up in this "
                "build. Select a supported image provider; no placeholder was created."
            )
        elif provider == "mock":
            if not (getattr(cfg, "image_allow_mock_assets", False) or allow_mock_assets):
                return [], (
                    "Mock placeholder images are disabled by default. Configure a real "
                    "image provider, or explicitly enable mock assets for a test workspace."
                )
            assets = [
                _mock_asset(effective_prompt, aspect_ratio, i)
                for i in range(max(1, num_images))
            ]
        else:
            return [], "No supported image provider with credentials is configured; no image was created."
    except httpx.HTTPError as exc:
        return [], f"image generation failed: {exc}"
    except ToolUnavailable as exc:
        return [], exc.message
    except Exception as exc:  # noqa: BLE001 - a provider surprise must not become a 500
        # Anything the provider layer did not anticipate (a malformed body, a
        # library-level error) still leaves the request answered, with a reason
        # the UI can show, instead of a stack trace after a minute of waiting.
        _logger.warning(
            "image_generation_unhandled",
            extra={"provider": provider, "error": type(exc).__name__},
        )
        return [], f"image generation failed ({type(exc).__name__}); no image was created."

    if not assets:
        return [], "The image provider returned no final asset."

    for asset in assets:
        asset.style_spec = style
        if _is_durable_asset_url(asset.url, cfg) and not asset.url.startswith("data:"):
            continue
        try:
            asset.url = await _persist_generated_asset(asset, cfg)
        except ToolUnavailable as exc:
            # Storage is a deployment concern, not a reason to lose the image the
            # user just paid for. The asset keeps whatever URL the provider gave
            # us and is flagged as not storage-hosted, so the write path can
            # refuse to put a megabyte of base64 into a database row while a
            # plain URL (or a dev-only mock placeholder) still gets stored.
            asset.durable = False
            _logger.warning(
                "generated_asset_not_persisted",
                extra={"provider": asset.provider, "reason": exc.message},
            )
            if asset.provider != "mock" and not asset.url.startswith("https://"):
                # An inline payload we cannot re-host is not a usable creative:
                # it cannot be attached to a Meta ad and must not reach a row.
                return [], exc.message
            continue
        if asset.provider != "mock" and not (
            validate_creative_image_url(asset.url) or _is_durable_asset_url(asset.url, cfg)
        ):
            # ``_is_durable_asset_url`` covers the local-disk fallback: on a
            # developer machine the asset lives on this very server over
            # ``http://localhost``, which is perfectly displayable in the app even
            # though Meta would refuse it -- and Meta's own gate still runs when
            # an ad is actually submitted.
            return [], (
                "The final image is not available at a public HTTPS URL; "
                "it cannot be safely attached to a Meta ad."
            )
    return assets, None


async def persist_creative_assets(
    db: Any,
    *,
    business_id: str,
    assets: list[GeneratedAssetRecord],
    campaign_id: str | None = None,
    persona_id: str | None = None,
    style: dict[str, Any] | None = None,
    reference_analysis_id: str | None = None,
    task_name: str = "generate_image",
    status: str = "ok",
) -> list[GeneratedAssetRecord]:
    """Write one ``Creative`` row per asset. Only the durable URL is stored.

    An asset whose bytes never reached storage keeps its place in the gallery but
    is written with an empty ``url`` and ``status="unpersisted"``: storing the
    provider's temporary link would rot, and storing its base64 is what made this
    table slow in the first place.
    """
    for asset in assets:
        asset.style_spec = style or {}
        # Rule: never write an inline payload into a row. A plain URL is fine
        # even when it is not storage-hosted (it is a short string, and the row
        # still points at the image); a data URL is what made this table slow.
        stored_url = "" if asset.url.startswith("data:") else asset.url
        stored_status = status if (stored_url or not asset.url) else "unpersisted"
        try:
            db.add(
                Creative(
                    id=asset.id,
                    business_id=business_id,
                    campaign_id=campaign_id,
                    persona_id=persona_id,
                    task_name=task_name,
                    kind="image",
                    url=stored_url,
                    prompt=asset.prompt,
                    aspect_ratio=asset.aspect_ratio,
                    provider=asset.provider,
                    model=asset.model,
                    style_spec=style or None,
                    reference_analysis_id=reference_analysis_id,
                    status=stored_status,
                )
            )
            await db.flush()
        except Exception as exc:  # pragma: no cover - never fail the generation on a row write
            _logger.warning("creative_persist_failed", extra={"error": str(exc)})
    return assets


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

    This is the HTTP bridge used by /creatives. Provider access and durable
    storage live in :func:`generate_creative_assets`, which the chat path uses
    too -- this function only resolves workspace settings and references.
    """
    from app.core.tools.settings import load_connector_settings

    cfg = settings
    tools_cfg = await load_connector_settings(db, business_id, cfg)
    image_provider = str(tools_cfg.image_provider or cfg.image_provider or "mock").strip().lower()
    image_key = tools_cfg.image_api_key or cfg.image_api_key
    image_model = tools_cfg.image_model or cfg.image_model
    image_base = tools_cfg.image_base_url or cfg.image_base_url
    # Resolved workspace -> env -> default ("auto") by the connector layer.
    # getattr: several call sites pass a hand-built settings object (tests, and
    # the VLM bridge), and a missing attribute must not take image generation
    # down when the answer is simply "no preference".
    image_quality = getattr(tools_cfg, "image_quality", None)
    source = f"generate_image:{image_provider}"

    style = dict(getattr(payload, "style_spec", None) or {})
    reference_image: tuple[str, bytes] | None = None
    if getattr(payload, "reference_analysis_id", None) or getattr(payload, "reference_image_url", None):
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

    assets, reason = await generate_creative_assets(
        cfg=cfg,
        prompt=payload.prompt,
        aspect_ratio=payload.aspect_ratio,
        # getattr: hand-built payloads (tests, the VLM bridge) carry only the
        # fields they care about; a missing count means "one image".
        num_images=int(getattr(payload, "num_images", 1) or 1),
        style=style,
        reference_image=reference_image,
        provider=image_provider,
        api_key=image_key,
        model=image_model,
        base_url=image_base,
        quality=image_quality,
        allow_mock_assets=allow_mock_assets,
    )
    if reason:
        return GenerateImageOutput.unavailable(source=source, reason=reason)

    persisted = await persist_creative_assets(
        db,
        business_id=business_id,
        assets=assets,
        campaign_id=getattr(payload, "campaign_id", None) or campaign_id,
        persona_id=persona_id,
        style=style,
        reference_analysis_id=getattr(payload, "reference_analysis_id", None),
    )
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
