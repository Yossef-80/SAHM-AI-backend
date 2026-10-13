"""Read-only serving for unguessable generated image assets sent to Meta.

The asset URL is a 128-bit random capability. Only validated raster files written
by the image-generation path are served; this router has no upload or listing
endpoint and never analyses image content.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import get_settings

router = APIRouter(prefix="/media", tags=["media"])
_ASSET_NAME = re.compile(r"^asset_[a-f0-9]{32}\.(png|jpg|webp)$")
_MIME = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}


def _sniff(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return ""


@router.get("/public-assets/{asset_name}", include_in_schema=False)
async def get_public_asset(asset_name: str) -> FileResponse:
    """Serve one generated raster image by its opaque, non-credential URL."""
    match = _ASSET_NAME.fullmatch(asset_name)
    if not match:
        raise HTTPException(status_code=404, detail="Asset not found")
    cfg = get_settings()
    root = (Path(cfg.upload_dir).resolve() / "public_assets").resolve()
    target = (root / asset_name).resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    if target.stat().st_size <= 0 or target.stat().st_size > cfg.upload_max_bytes:
        raise HTTPException(status_code=404, detail="Asset not found")
    try:
        with target.open("rb") as handle:
            signature = handle.read(16)
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Asset not found") from exc
    extension = match.group(1)
    if _sniff(signature) != extension:
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(
        target,
        media_type=_MIME[extension],
        filename=asset_name,
        content_disposition_type="inline",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


__all__ = ["get_public_asset", "router"]
