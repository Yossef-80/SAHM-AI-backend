"""Images a user attaches to a chat turn.

Why this is not the creative-reference path
-------------------------------------------
Uploading a photo in the assistant used to run the *creative* reference
analysis immediately, which dropped the user into Layla's creative workflow for
a picture they only wanted to talk about. An attachment here is a plain image
the chat model can look at: it is validated, stored, announced to the model as a
multimodal content block, and otherwise stays out of the way.

Storage reuses :func:`app.core.tools.media.persist_inline_reference_image` --
the same signature sniffing, size cap and opaque filename the reference upload
uses -- and reuses ``BrandMemoryItem`` as the row (kind ``chat_attachment``)
rather than adding a table for four columns.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.errors import ValidationFailed
from app.core.logging import get_logger
from app.core.tools.media import persist_inline_reference_image_async
from app.db.models import BrandMemoryItem

_logger = get_logger(__name__)

KIND = "chat_attachment"

#: A chat turn may carry at most this many images.
MAX_ATTACHMENTS_PER_TURN = 4

#: Largest encoded payload handed to the model per image (characters of base64).
MAX_INLINE_CHARS = 4_000_000


async def _decode(data_url: str) -> tuple[str, str]:
    """Validate the browser payload, store it (disk + storage mirror), return ids."""
    mime = ""
    head = data_url.strip()[:64].lower()
    for candidate in ("image/png", "image/jpeg", "image/webp"):
        if head.startswith(f"data:{candidate};base64,"):
            mime = candidate
            break
    if not mime:
        raise ValidationFailed(
            "attachments must be a base64 PNG, JPEG or WebP data URL",
            detail={"supported": ["image/png", "image/jpeg", "image/webp"]},
        )
    try:
        file_id = await persist_inline_reference_image_async(data_url, get_settings())
    except Exception as exc:
        raise ValidationFailed(f"attachment could not be stored: {exc}") from exc
    return file_id, mime


async def store_chat_attachment(
    db: AsyncSession,
    *,
    business_id: str,
    image_data: str,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Validate, store and register one chat attachment."""
    file_id, mime = await _decode(image_data or "")
    name = (source_name or "attachment").strip()[:160] or "attachment"
    attachment_id = f"att_{uuid4().hex[:20]}"
    row = BrandMemoryItem(
        id=attachment_id,
        business_id=business_id,
        kind=KIND,
        text=name,
        metadata_={
            "source_name": name,
            "file_id": file_id,
            "mime": mime,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        created_at=datetime.now(timezone.utc),
    )
    db.add(row)
    await db.flush()
    await db.commit()
    return {
        "id": attachment_id,
        "name": name,
        "mime": mime,
        "created_at": row.metadata_.get("created_at"),
    }


async def _read_file(file_id: str) -> str | None:
    """Read the stored private file back as a data URL, or None.

    Local disk first, then the Supabase Storage mirror: attachments uploaded
    before a redeploy only exist in the mirror.
    """
    settings = get_settings()
    root = Path(settings.upload_dir).resolve()
    target = (root / file_id).resolve()
    raw: bytes | None = None
    if root in target.parents and target.is_file():
        try:
            raw = target.read_bytes()
        except OSError:
            raw = None
    if raw is None:
        from app.core.storage import fetch_private_asset

        raw = await fetch_private_asset(settings, file_id)
    if raw is None:
        return None
    if not raw or len(raw) > int(settings.website_max_bytes):
        return None
    suffix = target.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(
        suffix
    )
    if mime is None:
        return None
    encoded = base64.b64encode(raw).decode()
    if len(encoded) > MAX_INLINE_CHARS:
        return None
    return f"data:{mime};base64,{encoded}"


async def load_chat_attachments(
    db: AsyncSession, *, business_id: str, ids: list[str]
) -> list[dict[str, Any]]:
    """Resolve this turn's attachment ids into model-ready payloads.

    Ids that do not belong to this business, or whose file is gone, are skipped
    silently: an attachment is context, never an error the user has to fix.
    """
    wanted = [str(item) for item in (ids or []) if str(item).strip()]
    if not wanted:
        return []
    wanted = wanted[:MAX_ATTACHMENTS_PER_TURN]
    rows = (
        (
            await db.execute(
                select(BrandMemoryItem).where(
                    BrandMemoryItem.id.in_(wanted),
                    BrandMemoryItem.business_id == business_id,
                    BrandMemoryItem.kind == KIND,
                )
            )
        )
        .scalars()
        .all()
    )
    by_id = {row.id: row for row in rows}

    payloads: list[dict[str, Any]] = []
    for attachment_id in wanted:
        row = by_id.get(attachment_id)
        if row is None:
            continue
        metadata = row.metadata_ or {}
        data_url = await _read_file(str(metadata.get("file_id") or ""))
        if data_url is None:
            _logger.warning(
                "chat_attachment_unreadable", extra={"attachment_id": attachment_id}
            )
            continue
        payloads.append(
            {
                "id": row.id,
                "name": str(metadata.get("source_name") or row.text or "attachment")[:160],
                "mime": str(metadata.get("mime") or "image/png"),
                "data_url": data_url,
                # The stored private file id, which is exactly the `upload_id`
                # `analyze_reference_creative` accepts (the tool resolves it under
                # `upload_dir`). Without it the model can *see* the attached picture
                # but cannot act on it: it has no id to hand to the analysis tool,
                # so it asks the user for one the user cannot know -- which is how
                # "make a design like this image" ended in "send me the upload_id".
                "upload_id": str(metadata.get("file_id") or ""),
            }
        )
    return payloads


__all__ = [
    "KIND",
    "MAX_ATTACHMENTS_PER_TURN",
    "load_chat_attachments",
    "store_chat_attachment",
]
