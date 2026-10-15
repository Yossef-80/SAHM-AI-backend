"""The saved-references list must not be taken down by one unreadable row.

``GET /creatives/references`` read every row's ``style`` with
``StyleSpec.model_validate`` and no guard, so a single stored analysis whose
JSON no longer matches the schema made the *whole* list a 500 -- for every
picker in the workspace. The row is now skipped with a log line, and favouriting
a row (which does not need its style at all) keeps working.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import BrandMemoryItem

HEADERS = {"X-Workspace-Id": "biz_test", "Origin": "http://localhost:3000"}

GOOD_STYLE = {"palette": ["#0f172a"], "composition": "centered"}


async def _seed(db: AsyncSession, reference_id: str, style: Any, **metadata: Any) -> None:
    db.add(
        BrandMemoryItem(
            id=reference_id,
            business_id="biz_test",
            kind="reference_analysis",
            text="A saved reference analysis.",
            metadata_={
                "source_name": f"{reference_id}.png",
                "summary": "saved",
                "favorite": False,
                "style": style,
                **metadata,
            },
        )
    )
    await db.commit()


async def test_unreadable_style_skips_its_row_only(
    api_client: object, seeded_db: AsyncSession
) -> None:
    await _seed(seeded_db, "mem_ref_good", GOOD_STYLE)
    await _seed(seeded_db, "mem_ref_broken", {"palette": "not-a-list"})
    await _seed(seeded_db, "mem_ref_prose", "earthy tones, natural light")

    resp = await api_client.get("/api/v1/creatives/references", headers=HEADERS)  # type: ignore[attr-defined]

    assert resp.status_code == 200, resp.text  # type: ignore[attr-defined]
    ids = [row["id"] for row in resp.json()]  # type: ignore[attr-defined]
    assert ids == ["mem_ref_good"]
    assert resp.json()[0]["style"]["palette"] == ["#0f172a"]  # type: ignore[attr-defined]


async def test_list_is_empty_rather_than_broken_without_readable_rows(
    api_client: object, seeded_db: AsyncSession
) -> None:
    """Unknown keys are ignored by the schema; a wrong *type* is not."""
    await _seed(seeded_db, "mem_ref_broken", {"palette": {"a": 1}})

    resp = await api_client.get("/api/v1/creatives/references", headers=HEADERS)  # type: ignore[attr-defined]

    assert resp.status_code == 200, resp.text  # type: ignore[attr-defined]
    assert resp.json() == []  # type: ignore[attr-defined]


async def test_favouriting_an_unreadable_reference_still_works(
    api_client: object, seeded_db: AsyncSession
) -> None:
    """The flag is on the row; reading its style is not required to set it."""
    await _seed(seeded_db, "mem_ref_broken", {"palette": "not-a-list"})

    resp = await api_client.patch(  # type: ignore[attr-defined]
        "/api/v1/creatives/references/mem_ref_broken",
        headers=HEADERS,
        json={"favorite": True},
    )

    assert resp.status_code == 200, resp.text  # type: ignore[attr-defined]
    body = resp.json()  # type: ignore[attr-defined]
    assert body["favorite"] is True
    # An unreadable style comes back as an empty one, never as an exception.
    assert body["style"]["palette"] == []
