"""Attaching a photo to a chat must not run Layla's creative analysis.

The behaviour these tests lock down is the one the user asked to remove:
uploading an image in the assistant used to call
``POST /creatives/analyze-reference`` (spending a vision call and dropping the
composer into the creative workflow) and the picture ended up in the reference
library instead of the conversation.

What must hold now:

* ``POST /assistant/attachments`` stores the bytes and returns an id — and it
  must *not* create a ``reference_analysis`` memory row;
* the id travels with the next message through ``context.attachmentIds``;
* an id that belongs to another business is ignored, never leaked into the turn.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from sqlalchemy import select

from app.core.chat_attachments import KIND, load_chat_attachments, store_chat_attachment
from app.db.models import BrandMemoryItem

#: 1x1 transparent PNG — the smallest real image a browser can produce.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


@pytest.mark.anyio
async def test_store_and_load_roundtrip(seeded_db: Any) -> None:
    from app.db.models import Business

    business = (await seeded_db.execute(select(Business))).scalars().first()
    assert business is not None

    stored = await store_chat_attachment(
        seeded_db,
        business_id=business.id,
        image_data=PNG_DATA_URL,
        source_name="product.png",
    )
    assert stored["id"].startswith("att_")
    assert stored["mime"] == "image/png"
    assert stored["name"] == "product.png"

    payloads = await load_chat_attachments(
        seeded_db, business_id=business.id, ids=[stored["id"]]
    )
    assert len(payloads) == 1
    assert payloads[0]["data_url"].startswith("data:image/png;base64,")
    assert payloads[0]["name"] == "product.png"

    # Stored as a chat attachment, never as an analyzed creative reference.
    rows = (
        (
            await seeded_db.execute(
                select(BrandMemoryItem).where(BrandMemoryItem.business_id == business.id)
            )
        )
        .scalars()
        .all()
    )
    kinds = {row.kind for row in rows}
    assert KIND in kinds
    assert "reference_analysis" not in kinds


@pytest.mark.anyio
async def test_other_business_id_is_ignored(seeded_db: Any) -> None:
    from app.db.models import Business

    # A second tenant: an id from one workspace must never resolve in another.
    other = Business(id="biz_other", workspace_id="ws_other", name="Other Co")
    seeded_db.add(other)
    await seeded_db.commit()

    stored = await store_chat_attachment(
        seeded_db, business_id="biz_test", image_data=PNG_DATA_URL, source_name="mine.png"
    )
    assert await load_chat_attachments(seeded_db, business_id=other.id, ids=[stored["id"]]) == []


@pytest.mark.anyio
async def test_non_image_payload_is_rejected(seeded_db: Any) -> None:
    from app.core.errors import ValidationFailed
    from app.db.models import Business

    business = (await seeded_db.execute(select(Business))).scalars().first()
    assert business is not None

    with pytest.raises(ValidationFailed):
        await store_chat_attachment(
            seeded_db,
            business_id=business.id,
            image_data="data:text/plain;base64,aGVsbG8gd29ybGQ=",
        )


@pytest.mark.anyio
async def test_attachment_endpoint_accepts_a_photo(api_client: Any) -> None:
    resp = await api_client.post(
        "/api/v1/assistant/attachments",
        json={"imageData": PNG_DATA_URL, "sourceName": "photo.png"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"].startswith("att_")
    assert body["name"] == "photo.png"
