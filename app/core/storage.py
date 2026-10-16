"""Storage adapter for generated/uploaded binary assets.

Preference order:
1. Supabase Storage (when ``SUPABASE_URL`` + a key are configured) -- the
   production target. Returns a public HTTPS URL.
2. Local disk under ``cfg.upload_dir`` (dev/test fallback) -- keeps the app
   working when Supabase is not configured.

The adapter is intentionally thin: it only moves bytes in/out and returns a URL.
Callers decide what to persist in the database (always just the URL).
"""

from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path
from typing import Any

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BUCKET = "creatives"


def _supabase_url(cfg: Any) -> str | None:
    value = getattr(cfg, "supabase_url", None) or os.environ.get("SUPABASE_URL")
    if not value:
        # Fall back to the known project URL derived from DATABASE_URL host.
        return None
    return str(value).rstrip("/")


def _supabase_key(cfg: Any) -> str | None:
    for attr in ("supabase_secret_key", "supabase_service_role_key", "supabase_publishable_key"):
        v = getattr(cfg, attr, None)
        if v:
            return str(v)
    for env in ("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_PUBLISHABLE_KEY", "SUPABASE_ANON_KEY"):
        v = os.environ.get(env)
        if v:
            return v
    return None


def is_supabase_configured(cfg: Any) -> bool:
    return bool(_supabase_url(cfg) and _supabase_key(cfg))


async def ensure_bucket(cfg: Any, bucket: str = DEFAULT_BUCKET) -> bool:
    """Best-effort bucket creation (requires service role). Returns True if usable."""
    url = _supabase_url(cfg)
    key = _supabase_key(cfg)
    if not (url and key):
        return False
    try:
        async with httpx.AsyncClient(timeout=20.0, trust_env=False) as client:
            resp = await client.post(
                f"{url}/storage/v1/bucket",
                headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"id": bucket, "name": bucket, "public": True},
            )
            # 200/201 created, 409 already exists -> both fine
            if resp.status_code in (200, 201, 409):
                return True
            logger.warning("supabase_bucket_create_failed", extra={"status": resp.status_code, "body": resp.text[:200]})
            return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("supabase_bucket_create_error", extra={"error": str(exc)})
        return False


async def upload_bytes(
    cfg: Any,
    data: bytes,
    mime: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    path: str,
) -> str | None:
    """Upload bytes to Supabase Storage; return public URL or None on failure."""
    url = _supabase_url(cfg)
    key = _supabase_key(cfg)
    if not (url and key and data):
        return None
    try:
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            resp = await client.post(
                f"{url}/storage/v1/object/{bucket}/{path}",
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                    "Content-Type": mime,
                    "x-upsert": "true",
                },
                content=data,
            )
            if resp.status_code not in (200, 201):
                logger.warning(
                    "supabase_upload_failed",
                    extra={"status": resp.status_code, "body": resp.text[:300], "path": path},
                )
                return None
        return f"{url}/storage/v1/object/public/{bucket}/{path}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("supabase_upload_error", extra={"error": str(exc), "path": path})
        return None


def local_public_url(cfg: Any, relative_path: str) -> str:
    """Build a public URL for a locally persisted asset (legacy fallback)."""
    base = str(getattr(cfg, "public_asset_base_url", None) or "").strip()
    api_prefix = str(getattr(cfg, "api_prefix", "/api/v1") or "/api/v1")
    return f"{base.rstrip('/')}{api_prefix}/media/public-assets/{relative_path}"


_EXTENSION_BY_MIME = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/svg+xml": "svg"}


def sniff_image_mime(data: bytes) -> str | None:
    """Identify PNG/JPEG/WebP from magic bytes; None when unrecognised."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:5] == b"<?xml" or data.lstrip()[:4] == b"<svg":
        return "image/svg+xml"
    return None


def _usable_local_origin(cfg: Any) -> str | None:
    """The usable origin for locally persisted assets, or None.

    A *production* creative URL has to be readable by Meta's servers, so a bare
    host, an ``http://`` origin, or a base URL carrying its own path/query is not
    usable as an asset origin -- and silently accepting one produced URLs that
    404'd only once an ad was submitted.

    A **loopback** ``http://`` origin is the one documented exception, and it is
    what makes image generation work on a developer machine with no Supabase
    project: ``PUBLIC_ASSET_BASE_URL=http://localhost:8000`` keeps the asset
    reachable by the browser that is already talking to this server (the file is
    served by ``/media/public-assets/`` with a 128-bit random name), instead of
    discarding an image the user has already paid the provider for. The ad path
    still enforces its own HTTPS/data-URL rule -- ``validate_creative_image_url``
    -- so a loopback URL can never be submitted to Meta.
    """
    base = str(getattr(cfg, "public_asset_base_url", "") or "").strip()
    if not base:
        return None
    from urllib.parse import urlsplit

    parsed = urlsplit(base)
    if not parsed.netloc:
        return None
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        return None
    if parsed.scheme == "https":
        return base
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return base
    return None


def _write_local(cfg: Any, data: bytes, extension: str, asset_id: str) -> str | None:
    """Fallback: write the asset under ``cfg.upload_dir`` and return its public URL."""
    upload_dir = getattr(cfg, "upload_dir", None)
    if not upload_dir:
        # Hand-built settings objects (tests, the VLM bridge) may not carry the
        # upload configuration at all; there is simply nowhere to write.
        return None
    try:
        root = (Path(upload_dir).resolve() / "public_assets").resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = (root / f"{asset_id}.{extension}").resolve()
        if root not in target.parents:
            return None
        with target.open("wb") as handle:
            handle.write(data)
    except OSError as exc:
        logger.warning("local_asset_write_failed", extra={"error": str(exc)})
        return None
    return local_public_url(cfg, f"{asset_id}.{extension}")


async def _download(url: str) -> tuple[bytes, str] | None:
    """Download a provider-hosted image so it can be re-hosted before its URL expires."""
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, trust_env=False) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content, resp.headers.get("content-type", "").split(";")[0].strip().lower()
    except Exception as exc:  # noqa: BLE001
        logger.warning("provider_asset_download_failed", extra={"error": str(exc)})
        return None


async def persist_image_asset(
    cfg: Any,
    *,
    asset_url: str,
    path_prefix: str = "generated",
    bucket: str = DEFAULT_BUCKET,
) -> tuple[str | None, str | None]:
    """Move a freshly generated image into durable storage.

    Accepts either a ``data:`` URL (base64 payload) or an ``http(s)`` URL from the
    provider (those expire -- OpenAI's last about an hour), and returns
    ``(public_url, error)``.

    Order: Supabase Storage first, local disk second. The caller must store only
    the returned URL in the database; raw bytes never belong in a row.
    """
    if not asset_url:
        return None, "The image provider returned an empty asset."

    declared_mime = ""
    data: bytes | None = None
    if asset_url.startswith("data:"):
        header, separator, encoded = asset_url.partition(",")
        if not separator or ";base64" not in header:
            return None, "The image provider returned an unsupported image format."
        declared_mime = header[5:].split(";", 1)[0].lower()
        try:
            data = base64.b64decode(encoded, validate=True)
        except Exception:  # noqa: BLE001
            return None, "The image provider returned invalid image bytes."
    else:
        downloaded = await _download(asset_url)
        if downloaded is None:
            return None, "The generated image could not be downloaded before its URL expired."
        data, declared_mime = downloaded

    if not data:
        return None, "The generated image is empty."
    max_bytes = int(getattr(cfg, "upload_max_bytes", 0) or 0)
    if max_bytes and len(data) > max_bytes:
        return None, "The generated final image exceeds the configured size limit."

    detected = sniff_image_mime(data)
    if detected is None:
        return None, "The generated image bytes are not a supported image format."
    mime = declared_mime if declared_mime in _EXTENSION_BY_MIME else detected
    if declared_mime in _EXTENSION_BY_MIME and declared_mime != detected and detected != "image/svg+xml":
        return None, "Generated image bytes do not match their declared file type."

    extension = _EXTENSION_BY_MIME.get(mime, "png")
    asset_id = f"asset_{uuid.uuid4().hex}"
    object_path = f"{path_prefix.strip('/')}/{asset_id}.{extension}"

    uploaded = await upload_bytes(cfg, data, mime, bucket=bucket, path=object_path)
    if uploaded:
        return uploaded, None

    # Local fallback. Everything below applies only to this route.
    origin = _usable_local_origin(cfg)
    if origin is None:
        return None, (
            "Supabase Storage is not configured and PUBLIC_ASSET_BASE_URL is not a usable "
            "public HTTPS asset origin; the generated image has no public HTTPS URL."
        )
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        return None, "Only PNG, JPEG, or WebP final images can be saved as a public asset."
    local = _write_local(cfg, data, extension, asset_id)
    if local:
        return local, None
    return None, "The generated image could not be persisted."


# ------------------------------------------------------------------ uploads --
#: User-supplied images (chat attachments, reference creatives). Kept in a
#: separate bucket from generated assets so the two can be curated apart.
UPLOAD_BUCKET = "uploads"


def private_object_path(file_id: str) -> str:
    return f"references/{file_id}"


async def mirror_private_asset(cfg: Any, file_id: str, data: bytes, mime: str) -> bool:
    """Best-effort mirror of a locally stored upload into Supabase Storage.

    Local disk is what ``upload_dir`` is for and stays the primary read path, but
    a container filesystem is ephemeral: without the mirror, every chat
    attachment and reference image disappears on the next deploy while the row
    that points at it survives.
    """
    if not is_supabase_configured(cfg) or not data:
        return False
    url = await upload_bytes(
        cfg, data, mime, bucket=UPLOAD_BUCKET, path=private_object_path(file_id)
    )
    return url is not None


async def fetch_private_asset(cfg: Any, file_id: str) -> bytes | None:
    """Fetch a mirrored upload from Supabase Storage (local disk missing)."""
    if not is_supabase_configured(cfg) or not file_id:
        return None
    url = _supabase_url(cfg)
    key = _supabase_key(cfg)
    try:
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            resp = await client.get(
                f"{url}/storage/v1/object/{UPLOAD_BUCKET}/{private_object_path(file_id)}",
                headers={"apikey": key, "Authorization": f"Bearer {key}"},
            )
            if resp.status_code != 200:
                return None
            return resp.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("private_asset_fetch_failed", extra={"error": str(exc)})
        return None


__all__ = [
    "DEFAULT_BUCKET",
    "UPLOAD_BUCKET",
    "ensure_bucket",
    "fetch_private_asset",
    "is_supabase_configured",
    "local_public_url",
    "mirror_private_asset",
    "persist_image_asset",
    "private_object_path",
    "sniff_image_mime",
    "upload_bytes",
]
