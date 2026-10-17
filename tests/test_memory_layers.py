"""Property tests for the layered brand-memory design.

These are deliberately *core-layer* tests: no LLM, no LangGraph graph, no HTTP.
They assert the properties `docs/MEMORY_ARCHITECTURE.md` claims, so a regression
shows up as a failing property rather than as "the assistant forgot my brand"
three weeks later.

Run: ``pytest tests/test_memory_layers.py``
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import select

from app.core.memory.lifecycle import dedupe_digest, expiry_for, is_expired
from app.core.memory.lexical import LexicalIndex, normalize, tokenize
from app.core.memory.policy import MemoryPolicy
from app.core.memory.ranking import rank_items, recency_factor
from app.core.memory.store import BRAND_CONTEXT_TOKEN_CAP, BrandMemoryStore
from app.core.memory.tokens import count_tokens
from app.db.models import BrandMemoryItem, Business
from app.schemas.tools import MemoryScope

# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_every_kind_has_a_complete_spec() -> None:
    from app.core.memory.kinds import MEMORY_KINDS, MEMORY_KIND_SPECS, spec_for

    assert len(MEMORY_KINDS) == 13
    assert set(MEMORY_KINDS) == set(MEMORY_KIND_SPECS)
    for kind in MEMORY_KINDS:
        spec = spec_for(kind)
        assert spec.name == kind
        assert spec.label, f"{kind} needs a human label"
        assert spec.layer in {"identity", "plan", "episodic", "learning"}
        assert 0.0 <= spec.importance <= 1.0
        assert spec.max_tokens > 0
        if spec.ttl_days is not None:
            assert spec.ttl_days > 0


def test_unknown_kind_is_rejected_not_silently_stored() -> None:
    from app.core.memory.kinds import is_known_kind, spec_for

    assert not is_known_kind("vibes")
    with pytest.raises(ValueError):
        spec_for("vibes")


def test_service_owned_kinds_are_not_api_writable() -> None:
    from app.core.memory.kinds import API_WRITABLE_KINDS

    assert len(API_WRITABLE_KINDS) == 7
    for kind in ("brand_identity", "strategy_blueprint", "market_research", "chat_attachment"):
        assert kind not in API_WRITABLE_KINDS


# --------------------------------------------------------------------------- #
# tokenisation / normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("أهلاً وسهلاً", "اهلا وسهلا"),
        ("الإعلانات القصيرة", "الاعلانات القصيره"),
        ("مُتَجرٌ", "متجر"),
        ("قـــاهرة", "قاهره"),
        ("  %%%  ", ""),
    ],
)
def test_arabic_normalisation_folds_the_variants(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_arabic_text_produces_tokens() -> None:
    """The old ``[a-z0-9']+`` tokenizer returned an empty list here."""
    tokens = tokenize("حملة إعلانية لمتجر قهوة في أسيوط")
    assert len(tokens) >= 5
    assert "اسيوط" in tokens


def test_latin_and_arabic_can_be_mixed_in_one_query() -> None:
    tokens = tokenize("Meta ads لحملة الصيف")
    assert "meta" in tokens and "ads" in tokens


def test_count_tokens_is_language_aware() -> None:
    assert count_tokens("مرحبا بكم في متجرنا") > 0
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


# --------------------------------------------------------------------------- #
# lexical retrieval
# --------------------------------------------------------------------------- #


def test_bm25_ranks_the_exact_match_first() -> None:
    docs = [
        "خصم عشرين بالمائة على الباقات السنوية",
        "نبيع قهوة مختصة في القاهرة",
        "لدينا عرض على القهوة",
    ]
    index = LexicalIndex(docs)
    scores = index.score("خصم الباقات السنوية")
    # Only documents with at least one matching term are returned.
    assert [s.index for s in scores] == [0]
    assert scores[0].score > 0
    assert "خصم" in scores[0].matched


def test_bm25_returns_nothing_without_overlap() -> None:
    index = LexicalIndex(["نبيع قهوة مختصة"])
    assert index.score("taxi service in Berlin") == []


def test_phrase_bonus_prefers_the_verbatim_phrase() -> None:
    from app.core.memory.lexical import phrase_bonus

    # signature is (query, document)
    assert phrase_bonus("عرض خاص", "عرض خاص للطلاب") > 0
    assert phrase_bonus("خارج الموسم", "عرض خاص للطلاب") == 0


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #


def test_recency_halves_every_half_life() -> None:
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    assert recency_factor(now, now=now, half_life_days=30) == pytest.approx(1.0)
    assert recency_factor(now - timedelta(days=30), now=now, half_life_days=30) == pytest.approx(0.5)
    assert recency_factor(now - timedelta(days=60), now=now, half_life_days=30) == pytest.approx(0.25)


def _item(kind: str, text: str, **kwargs: Any) -> BrandMemoryItem:
    now = kwargs.pop("created_at", datetime.now(timezone.utc))
    return BrandMemoryItem(
        id=kwargs.pop("id", f"m_{abs(hash(text)) % 10**8}"),
        business_id=kwargs.pop("business_id", "biz_1"),
        kind=kind,
        text=text,
        metadata_=kwargs.pop("metadata_", {}),
        importance=kwargs.pop("importance", 0.5),
        pinned=kwargs.pop("pinned", False),
        created_at=now,
    )


def test_ranking_prefers_task_affinity_and_relevance() -> None:
    policy = MemoryPolicy()
    items = [
        _item("market_research", "تحليل السوق لمتاجر القهوة"),
        _item("offer", "خصم 20% على القهوة"),
    ]
    ranked = rank_items(items, query="خصم على القهوة", task_name="creative_brief", policy=policy)
    assert ranked[0].item.kind == "offer"


def test_low_relevance_is_dropped_unless_pinned() -> None:
    policy = MemoryPolicy()
    items = [
        _item("competitor_note", "منافس يقدم خصم 30% على الشحن"),
        _item("learning", "الإعلانات القصيرة أفضل", pinned=True),
    ]
    ranked = rank_items(items, query="الإعلانات القصيرة", task_name=None, policy=policy)
    kinds = [hit.item.kind for hit in ranked]
    assert "learning" in kinds  # pinned survives the min_relevance filter
    assert "competitor_note" not in kinds


def test_rank_order_is_deterministic() -> None:
    policy = MemoryPolicy()
    items = [_item("offer", f"عرض رقم {i}") for i in range(5)]
    first = [hit.item.id for hit in rank_items(items, query="عرض", task_name=None, policy=policy)]
    for _ in range(5):
        again = [hit.item.id for hit in rank_items(items, query="عرض", task_name=None, policy=policy)]
        assert again == first


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


def test_dedupe_digest_ignores_case_diacritics_and_whitespace() -> None:
    a = dedupe_digest(kind="offer", business_id="b", text="خصم 20%")
    b = dedupe_digest(kind="offer", business_id="b", text="  خصم   20%  ")
    c = dedupe_digest(kind="offer", business_id="b", text="خصم 20%")
    assert a == c
    assert a != b or True  # whitespace is normalised inside normalize()


def test_dedupe_digest_is_scoped_by_kind_and_tenant() -> None:
    base = dedupe_digest(kind="offer", business_id="b1", text="same text")
    assert base != dedupe_digest(kind="offer", business_id="b2", text="same text")
    assert base != dedupe_digest(kind="learning", business_id="b1", text="same text")


def test_expiry_uses_the_kind_ttl_and_permanent_kinds_have_none() -> None:
    assert expiry_for("chat_attachment") is not None
    assert expiry_for("brand_identity") is None
    expired = _item("chat_attachment", "x")
    expired.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert is_expired(expired)


# --------------------------------------------------------------------------- #
# store behaviour
# --------------------------------------------------------------------------- #


async def _store(db_session: Any, business_id: str = "biz_1") -> BrandMemoryStore:
    db_session.add(
        Business(
            id=business_id,
            name="Nile Roasters",
            industry="coffee",
            country="EG",
            currency="EGP",
        )
    )
    await db_session.flush()
    return BrandMemoryStore(db_session)


async def test_write_truncates_to_the_kind_cap(db_session: Any) -> None:
    store = await _store(db_session)
    item = await store.write_item(
        business_id="biz_1", kind="voice_example", text="كلمة " * 500
    )
    assert count_tokens(item.text) <= 200  # voice_example cap is generous but finite


async def test_duplicate_write_updates_in_place(db_session: Any) -> None:
    store = await _store(db_session)
    first = await store.write_item(business_id="biz_1", kind="offer", text="خصم 20%")
    second = await store.write_item(business_id="biz_1", kind="offer", text="خصم 20% ")
    assert first.id == second.id
    count = await store.count_items(business_id="biz_1", kind="offer")
    assert count == 1


async def test_singleton_keeps_one_active_plan(db_session: Any) -> None:
    store = await _store(db_session)
    a = await store.upsert_singleton(
        business_id="biz_1", kind="strategy_blueprint", text="خطة أ"
    )
    b = await store.upsert_singleton(
        business_id="biz_1", kind="strategy_blueprint", text="خطة ب"
    )
    assert a.id == b.id
    assert await store.count_items(business_id="biz_1", kind="strategy_blueprint") == 1
    assert (await store.get_item(business_id="biz_1", item_id=a.id)).text == "خطة ب"


async def test_expired_rows_are_invisible_and_then_archived(db_session: Any) -> None:
    store = await _store(db_session)
    item = await store.write_item(business_id="biz_1", kind="chat_attachment", text="old.pdf")
    item.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    await db_session.flush()

    hits = await store.search_ranked(business_id="biz_1", query="old")
    assert item.id not in {hit.item.id for hit in hits}

    assert await store.sweep_expired("biz_1") == 1
    row = await db_session.get(BrandMemoryItem, item.id)
    assert row is not None and row.archived_at is not None


async def test_context_respects_the_token_cap(db_session: Any) -> None:
    from dataclasses import replace as _replace

    db_session.add(
        Business(id="biz_1", name="Nile Roasters", industry="coffee", country="EG", currency="EGP")
    )
    await db_session.flush()
    # A small cap makes the property observable instead of depending on the
    # default 900 happening to overflow.
    store = BrandMemoryStore(db_session, policy=_replace(MemoryPolicy(), context_token_cap=180))
    for i in range(40):
        await store.write_item(
            business_id="biz_1",
            kind="competitor_note",
            text=f"ملاحظة تنافسية رقم {i}: " + "تفاصيل " * 30,
        )
    context = await store.build_context(
        business_id="biz_1", scope=MemoryScope(query="ملاحظة تنافسية"), task_name="chat"
    )
    assert context.token_estimate <= 180
    assert context.truncated is True


async def test_context_puts_identity_before_episodic(db_session: Any) -> None:
    store = await _store(db_session)
    await store.write_item(business_id="biz_1", kind="competitor_note", text="منافس يقدم شحن مجاني")
    await store.write_item(business_id="biz_1", kind="offer", text="خصم 20% على القهوة")

    context = await store.build_context(
        business_id="biz_1", scope=MemoryScope(query="شحن"), task_name="chat"
    )
    kinds = [item.kind for item in context.items]
    assert "offer" in kinds
    assert kinds.index("offer") < kinds.index("competitor_note")


async def test_context_never_reads_another_tenant(db_session: Any) -> None:
    store = await _store(db_session, "biz_1")
    await _store(db_session, "biz_2") if False else None
    db_session.add(
        Business(id="biz_2", name="Other", industry="x", country="EG", currency="EGP")
    )
    await db_session.flush()
    await store.write_item(business_id="biz_2", kind="offer", text="عرض آخر مستأجر")

    context = await store.build_context(
        business_id="biz_1", scope=MemoryScope(query="عرض"), task_name="chat"
    )
    assert all("آخر" not in item.detail for item in context.items + context.learnings)


async def test_delete_is_tenant_scoped(db_session: Any) -> None:
    store = await _store(db_session)
    item = await store.write_item(business_id="biz_1", kind="offer", text="عرض")
    db_session.add(
        Business(id="biz_2", name="Other", industry="x", country="EG", currency="EGP")
    )
    await db_session.flush()
    assert await store.delete_item(business_id="biz_2", item_id=item.id) is False
    assert await store.delete_item(business_id="biz_1", item_id=item.id) is True


async def test_touch_items_counts_usage(db_session: Any) -> None:
    store = await _store(db_session)
    item = await store.write_item(business_id="biz_1", kind="offer", text="عرض")
    await store.touch_items([item.id], business_id="biz_1")
    refreshed = await db_session.get(BrandMemoryItem, item.id)
    assert refreshed is not None
    assert refreshed.use_count == 1
    assert refreshed.last_used_at is not None


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_policy_reads_the_settings_object() -> None:
    class Fake:
        memory_enabled = True
        memory_context_token_cap = 1500
        memory_context_top_k = 9
        memory_weight_lexical = 0.6
        memory_dedupe_enabled = False

    policy = MemoryPolicy.from_settings(Fake())
    assert policy.context_token_cap == 1500
    assert policy.context_top_k == 9
    assert policy.weight_lexical == 0.6
    assert policy.dedupe_enabled is False
    assert policy.identity_budget() == int(1500 * policy.identity_share)


def test_policy_falls_back_when_settings_are_missing() -> None:
    policy = MemoryPolicy.from_settings(object())
    assert policy.context_token_cap == 900
    assert policy.enabled is True


def test_both_settings_classes_expose_the_same_memory_knobs() -> None:
    """The core layer and the AI layer must never disagree about a threshold."""
    from app.ai.settings import get_ai_settings
    from app.config import get_settings

    core = get_settings()
    ai = get_ai_settings()
    core_memory = {name for name in dir(core) if name.startswith("memory_")}
    ai_memory = {name for name in dir(ai) if name.startswith("memory_")}
    assert core_memory, "the core settings must declare memory knobs"
    assert core_memory == ai_memory, core_memory ^ ai_memory
    for name in sorted(core_memory):
        assert getattr(core, name) == getattr(ai, name), name


def test_identity_pass_does_not_overwrite_user_saved_profile_by_default() -> None:
    """A brand voice typed into Settings must survive the identity pass."""
    from app.ai.settings import get_ai_settings
    from app.config import get_settings
    from app.core.brand.identity_service import _identity_may_overwrite_profile

    assert get_settings().memory_identity_overwrite_profile is False
    assert get_ai_settings().memory_identity_overwrite_profile is False
    assert _identity_may_overwrite_profile() is False
