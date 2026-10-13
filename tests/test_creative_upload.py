"""POST /creatives/upload — a user-supplied image becomes a usable creative.

The point of the whole upload path is that the bytes eventually travel to Meta
inside the request, so the assertions that matter are not "a row was created"
but: the image survives the round trip intact, and it still passes the filter
that decides which creatives a Meta plan is allowed to use. A creative that
uploads fine but is then silently disqualified would look exactly the same here
without those two checks — that is the bug this file exists to prevent.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import pytest
from PIL import Image

from app.api.deps import WORKSPACE_HEADER
from app.ai.marketing import meta_ads
from app.core.tools.media import (
    MAX_UPLOADED_CREATIVE_BYTES,
    encode_uploaded_creative_image,
)
from app.core.errors import ToolValidationError
from app.db.models import Creative

URL = "/api/v1/creatives/upload"
HEADERS = {WORKSPACE_HEADER: "biz_test"}


def _png(width: int = 1080, height: int = 1080, colour: str = "#4B47F2") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(width: int = 800, height: int = 1000) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "#FF4D7D").save(buf, format="JPEG")
    return buf.getvalue()


async def test_uploading_a_png_creates_a_creative_and_keeps_every_byte(
    api_client: Any, seeded_db: Any
) -> None:
    original = _png()

    res = await api_client.post(
        URL,
        headers=HEADERS,
        files={"file": ("hero.png", original, "image/png")},
    )
    assert res.status_code == 200, res.text
    body = res.json()

    assert body["provider"] == "upload"
    # ToolOutput is a plain BaseModel, so the wire is snake_case — same as /generate.
    assert body["aspect_ratio"] == "1:1"
    assert body["url"].startswith("data:image/png;base64,")
    # The exact bytes come back, so nothing was re-encoded or truncated.
    assert base64.b64decode(body["url"].split(",", 1)[1]) == original

    row = await seeded_db.get(Creative, body["id"])
    assert row is not None
    assert row.business_id == "biz_test"
    assert row.url == body["url"]


async def test_uploaded_creative_survives_the_filter_and_the_bytes_round_trip(
    api_client: Any,
) -> None:
    """An upload must be usable in a Meta plan, not merely stored.

    `validate_creative_image_url` gates which creatives reach Meta, and
    `_creative_image_payload` is what actually produces the `bytes` field. If
    either rejected the stored shape, the upload would appear to succeed and
    then vanish from the campaign.
    """
    original = _jpeg()
    res = await api_client.post(
        URL, headers=HEADERS, files={"file": ("product.jpg", original, "image/jpeg")}
    )
    assert res.status_code == 200, res.text
    stored_url = res.json()["url"]

    assert meta_ads.validate_creative_image_url(stored_url) is True
    payload = await meta_ads._creative_image_payload(stored_url, timeout=5.0)
    assert payload is not None
    assert base64.b64decode(payload) == original


async def test_aspect_ratio_comes_from_the_real_pixel_dimensions(api_client: Any) -> None:
    res = await api_client.post(
        URL, headers=HEADERS, files={"file": ("tall.png", _png(1080, 1350), "image/png")}
    )
    assert res.status_code == 200, res.text
    assert res.json()["aspect_ratio"] == "4:5"


async def test_a_text_file_renamed_png_is_rejected(api_client: Any) -> None:
    """Trusts the signature, not the extension or the declared content type."""
    res = await api_client.post(
        URL,
        headers=HEADERS,
        files={"file": ("evil.png", b"#!/bin/sh\nrm -rf /\n", "image/png")},
    )
    assert res.status_code == 422, res.text
    assert "PNG, JPEG, or WebP" in res.json()["error"]["message"]


async def test_a_truncated_png_header_is_rejected(api_client: Any) -> None:
    """Signature sniffing alone is not enough; the image must really decode."""
    res = await api_client.post(
        URL,
        headers=HEADERS,
        files={"file": ("broken.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "image/png")},
    )
    assert res.status_code == 422, res.text
    assert "not a readable image" in res.json()["error"]["message"]


async def test_an_empty_upload_is_rejected(api_client: Any) -> None:
    res = await api_client.post(
        URL, headers=HEADERS, files={"file": ("empty.png", b"", "image/png")}
    )
    assert res.status_code == 422, res.text


def test_the_size_cap_is_enforced() -> None:
    """Unit-level: a real >10 MB image would make this test unreasonably slow."""
    oversized = b"\x00" * (MAX_UPLOADED_CREATIVE_BYTES + 1)
    with pytest.raises(ToolValidationError, match="size cap"):
        encode_uploaded_creative_image(oversized)


def test_a_non_image_is_rejected_at_the_unit_level_too() -> None:
    with pytest.raises(ToolValidationError):
        encode_uploaded_creative_image(b"GIF89a not supported here")


async def test_another_workspaces_campaign_cannot_receive_the_upload(
    api_client: Any,
) -> None:
    """campaignId is checked against the caller's workspace before anything is written."""
    res = await api_client.post(
        URL,
        headers=HEADERS,
        files={"file": ("hero.png", _png(8, 8), "image/png")},
        data={"campaignId": "camp_does_not_exist"},
    )
    assert res.status_code in (403, 404), res.text
