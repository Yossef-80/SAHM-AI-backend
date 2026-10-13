"""Workspace-scoped, short-lived image previews for the ad-selection flow.

Preview bytes are never put in assistant blocks or sent to the browser as an
unprotected third-party URL. A random opaque id is stored in the message block;
the UI fetches the bytes through the authenticated, workspace-scoped API route.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from pathlib import Path

PREVIEW_TTL_SECONDS = 24 * 60 * 60
MAX_PREVIEW_BYTES = 2_000_000
MIME_TO_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
SUFFIX_TO_MIME = {suffix: mime for mime, suffix in MIME_TO_SUFFIX.items()}
_PREVIEW_ID = re.compile(r"^[0-9a-f]{32}$")


class AdPreviewNotFoundError(FileNotFoundError):
    """The preview is absent, expired, invalid, or belongs to another tenant."""


def _tenant_dir(upload_dir: str | Path, business_id: str) -> Path:
    tenant_key = hashlib.sha256(business_id.encode("utf-8")).hexdigest()[:32]
    return (Path(upload_dir).expanduser() / "ad_previews" / tenant_key).resolve()


def store_ad_preview(
    *,
    upload_dir: str | Path,
    business_id: str,
    image_bytes: bytes,
    mime_type: str,
    now: float | None = None,
) -> str:
    """Store one validated image under an opaque id and return that id."""
    suffix = MIME_TO_SUFFIX.get(mime_type.lower().split(";", 1)[0].strip())
    if suffix is None:
        raise ValueError("unsupported preview image type")
    if not image_bytes or len(image_bytes) > MAX_PREVIEW_BYTES:
        raise ValueError("preview image is empty or exceeds the size cap")

    directory = _tenant_dir(upload_dir, business_id)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.time() if now is None else now
    _prune_expired(directory, now=timestamp)

    preview_id = uuid.uuid4().hex
    destination = directory / f"{preview_id}{suffix}"
    temporary = directory / f".{preview_id}.tmp"
    temporary.write_bytes(image_bytes)
    os.replace(temporary, destination)
    return preview_id


def read_ad_preview(
    *,
    upload_dir: str | Path,
    business_id: str,
    preview_id: str,
    now: float | None = None,
) -> tuple[bytes, str]:
    """Read an unexpired preview belonging to exactly this business."""
    if not _PREVIEW_ID.fullmatch(preview_id):
        raise AdPreviewNotFoundError("preview not found")

    directory = _tenant_dir(upload_dir, business_id)
    timestamp = time.time() if now is None else now
    for suffix, mime_type in SUFFIX_TO_MIME.items():
        candidate = directory / f"{preview_id}{suffix}"
        try:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(directory):
                continue
            stat = resolved.stat()
            if timestamp - stat.st_mtime > PREVIEW_TTL_SECONDS:
                resolved.unlink(missing_ok=True)
                continue
            data = resolved.read_bytes()
            if not data or len(data) > MAX_PREVIEW_BYTES:
                resolved.unlink(missing_ok=True)
                continue
            return data, mime_type
        except FileNotFoundError:
            continue
    raise AdPreviewNotFoundError("preview not found")


def _prune_expired(directory: Path, *, now: float) -> None:
    """Opportunistically remove old regular files in this tenant's cache."""
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for candidate in entries:
        if candidate.is_symlink() or candidate.suffix not in (*SUFFIX_TO_MIME, ".tmp"):
            continue
        try:
            if candidate.is_file() and now - candidate.stat().st_mtime > PREVIEW_TTL_SECONDS:
                candidate.unlink(missing_ok=True)
        except OSError:
            continue


__all__ = [
    "MAX_PREVIEW_BYTES",
    "PREVIEW_TTL_SECONDS",
    "AdPreviewNotFoundError",
    "read_ad_preview",
    "store_ad_preview",
]
