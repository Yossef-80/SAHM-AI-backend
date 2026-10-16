"""The creative learning loop: user signals become brand-voice memory.

Three hooks, all of which used to be dead ends because generated copy was never
persisted anywhere:

* saving copy the way Layla wrote it        -> a positive example;
* saving copy the user rewrote              -> a before/after pair;
* editing a saved creative later            -> a before/after pair;
* saying "أقصر" in chat                     -> a correction pair.

Each one lands in brand memory as a ``voice_example`` item, and the top matches
are injected back into Layla's prompts as few-shot examples. These tests assert
the *learning*, not just the HTTP 200: a save that persists a row but teaches
nothing would look identical from the outside.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select

from app.ai.agents.layla.learning import (
    correction_intent,
    format_examples_block,
    is_meaningful_edit,
    last_layla_draft,
    learn_from_chat_correction,
    learn_from_edit,
    select_voice_examples,
)
from app.api.deps import WORKSPACE_HEADER
from app.core.memory.store import BrandMemoryStore
from app.db.models import BrandMemoryItem, Creative

URL = "/api/v1/creatives/text"
HEADERS = {WORKSPACE_HEADER: "biz_test"}

DRAFT = "Wake up to fresh coffee delivered to your desk every Monday morning."
EDITED = "Never run out of coffee again. Delivered Monday, before you run dry."


async def _memory_texts(db: Any, kind: str = "voice_example") -> list[str]:
    rows = (
        await db.execute(select(BrandMemoryItem).where(BrandMemoryItem.kind == kind))
    ).scalars().all()
    return [row.text for row in rows]


# --------------------------------------------------------------------------- #
# 1. Intent detection                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("أقصر", "shorter"),
        ("خليها أقصر شوية", "shorter"),
        ("make it shorter please", "shorter"),
        ("طولها شوية", "longer"),
        ("مش كده، عيدها", "redirection"),
        ("خليها أكثر رسمية", "more_formal"),
        ("this is too salesy", "less_salesy"),
    ],
)
def test_correction_intent_is_recognised_in_both_languages(message: str, expected: str) -> None:
    assert correction_intent(message) == expected


def test_a_new_brief_is_not_a_correction() -> None:
    """A long request that happens to contain a trigger word is a new task."""

    brief = (
        "اعمل حملة جديدة لقهوة المختصة في القاهرة، الجمهور موظفين مكاتب، "
        "الميزانية 5000 جنيه، وعايز نسخة إعلانية أقصر من اللي فاتت لكل مجموعة"
    )
    assert correction_intent(brief) is None


def test_trivial_edits_are_not_style_signals() -> None:
    assert is_meaningful_edit("Fresh coffee daily.", "Fresh coffee daily!") is False
    assert is_meaningful_edit("Fresh coffee daily.", "Never run out of coffee again.") is True
    assert is_meaningful_edit("", "anything") is False
    assert is_meaningful_edit("same", "same") is False


def test_last_draft_skips_tool_calls_and_acknowledgements() -> None:
    from types import SimpleNamespace

    messages = [
        SimpleNamespace(type="ai", content="Sure!"),
        SimpleNamespace(type="ai", content="x" * 200, tool_calls=[{"name": "t"}]),
        SimpleNamespace(type="ai", content="A" * 120, tool_calls=[]),
        SimpleNamespace(type="human", content="B" * 300),
    ]
    assert last_layla_draft(messages) == "A" * 120


def test_examples_block_is_empty_without_examples() -> None:
    assert format_examples_block([]) == ""
    assert format_examples_block(["   "]) == ""
    block = format_examples_block(["Rejected: A\nAccepted: B"])
    assert "Example 1:" in block and "Rejected: A" in block


# --------------------------------------------------------------------------- #
# 2. The store-level hooks                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_manual_edit_is_stored_as_a_pair(seeded_db: Any) -> None:
    store = BrandMemoryStore(seeded_db)
    memory_id = await learn_from_edit(
        store,
        business_id="biz_test",
        original=DRAFT,
        revised=EDITED,
        context="coffee subscription ad",
    )
    await seeded_db.commit()

    assert memory_id is not None
    texts = await _memory_texts(seeded_db)
    stored = [t for t in texts if EDITED in t]
    assert stored, texts
    assert DRAFT in stored[0]
    assert "Rejected:" in stored[0] and "Accepted:" in stored[0]


@pytest.mark.asyncio
async def test_a_trivial_edit_stores_nothing(seeded_db: Any) -> None:
    store = BrandMemoryStore(seeded_db)
    assert (
        await learn_from_edit(
            store, business_id="biz_test", original="Fresh coffee.", revised="Fresh coffee!"
        )
        is None
    )


@pytest.mark.asyncio
async def test_a_chat_correction_needs_a_previous_draft(seeded_db: Any) -> None:
    store = BrandMemoryStore(seeded_db)
    assert (
        await learn_from_chat_correction(
            store, business_id="biz_test", user_message="أقصر", previous_draft=""
        )
        is None
    )
    stored = await learn_from_chat_correction(
        store, business_id="biz_test", user_message="أقصر", previous_draft=DRAFT
    )
    await seeded_db.commit()
    assert stored is not None


@pytest.mark.asyncio
async def test_relevant_examples_come_back_for_a_query(seeded_db: Any) -> None:
    store = BrandMemoryStore(seeded_db)
    await learn_from_edit(
        store,
        business_id="biz_test",
        original=DRAFT,
        revised=EDITED,
        context="coffee subscription ad copy",
    )
    await seeded_db.commit()

    examples = await select_voice_examples(
        store, business_id="biz_test", query="coffee subscription ad copy", limit=3
    )
    assert examples, "a stored example must be retrievable"
    assert any(EDITED in example for example in examples)
    # The seeded fixture already has one voice_example; the limit is respected.
    assert len(examples) <= 3


# --------------------------------------------------------------------------- #
# 3. The endpoints                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_saving_copy_unchanged_records_an_approval(api_client: Any, seeded_db: Any) -> None:
    response = await api_client.post(
        URL,
        headers=HEADERS,
        json={
            "tool": "generate_ad_copy",
            "text": DRAFT,
            "originalText": DRAFT,
            "brief": "coffee subscription ad",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prompt"] == DRAFT

    creative = await seeded_db.get(Creative, body["id"])
    assert creative is not None and creative.kind == "copy"
    # The row holds exactly the approved copy -- no title prefix, so a later edit
    # diffs against the real text.
    assert creative.prompt == DRAFT

    texts = await _memory_texts(seeded_db)
    assert any(DRAFT in text and "Approved copy:" in text for text in texts), texts


@pytest.mark.asyncio
async def test_saving_copy_the_user_rewrote_records_a_pair(api_client: Any, seeded_db: Any) -> None:
    response = await api_client.post(
        URL,
        headers=HEADERS,
        json={
            "tool": "generate_ad_copy",
            "text": EDITED,
            "originalText": DRAFT,
            "brief": "coffee subscription ad",
        },
    )
    assert response.status_code == 200, response.text

    texts = await _memory_texts(seeded_db)
    pair = [text for text in texts if DRAFT in text and EDITED in text]
    assert pair, texts
    assert "Rejected:" in pair[0]


@pytest.mark.asyncio
async def test_editing_a_saved_creative_teaches_from_the_diff(api_client: Any, seeded_db: Any) -> None:
    created = await api_client.post(
        URL,
        headers=HEADERS,
        json={"tool": "generate_ad_copy", "text": DRAFT, "originalText": DRAFT, "brief": ""},
    )
    creative_id = created.json()["id"]

    response = await api_client.patch(
        f"/api/v1/creatives/{creative_id}/text", headers=HEADERS, json={"text": EDITED}
    )
    assert response.status_code == 200, response.text
    assert response.json()["prompt"] == EDITED

    row = await seeded_db.get(Creative, creative_id)
    await seeded_db.refresh(row)
    assert row.prompt == EDITED

    texts = await _memory_texts(seeded_db)
    assert any(DRAFT in text and EDITED in text for text in texts), texts


@pytest.mark.asyncio
async def test_the_creative_list_publishes_the_kind(api_client: Any) -> None:
    """Without `kind` the client renders a saved copy hit as a broken image."""

    await api_client.post(
        URL,
        headers=HEADERS,
        json={"tool": "generate_ad_variations", "text": DRAFT, "originalText": DRAFT, "brief": ""},
    )
    listed = await api_client.get("/api/v1/creatives", headers=HEADERS)
    assert listed.status_code == 200
    rows = listed.json()
    assert any(row.get("kind") == "copy" and row.get("prompt") == DRAFT for row in rows)


@pytest.mark.asyncio
async def test_another_workspace_cannot_edit_this_creative(api_client: Any, seeded_db: Any) -> None:
    created = await api_client.post(
        URL,
        headers=HEADERS,
        json={"tool": "generate_ad_copy", "text": DRAFT, "originalText": DRAFT, "brief": ""},
    )
    creative_id = created.json()["id"]

    response = await api_client.patch(
        f"/api/v1/creatives/{creative_id}/text",
        headers={WORKSPACE_HEADER: "biz_other"},
        json={"text": EDITED},
    )
    # 404, never 403: a guessed id must not be an existence oracle.
    assert response.status_code == 404
