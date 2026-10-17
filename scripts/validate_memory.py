"""Validation harness for the brand-memory architecture.

Simulates a realistic bilingual (Arabic/English) marketing workspace for one
month, then measures the things a PM actually cares about:

  1. prompt cost per turn (is the 900-token cap real?)
  2. retrieval quality (does the right memory come back?)
  3. cross-language recall (ar query -> en fact and vice versa)
  4. latency at 50 / 500 / 2 000 rows
  5. dedupe behaviour (exact vs paraphrase)
  6. lifecycle (expiry, sweep, prune)
  7. multi-worker safety of the consolidator lock
  8. write amplification (use_count touching)
  9. metadata payload size

Run:  PYTHONPATH=. python3 scripts/validate_memory.py

Writes ``memory_validation_results.json`` into the repository root, next to
the file the numbers in ``docs/MEMORY_VALIDATION_REPORT.md`` were taken from.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.memory.kinds import IDENTITY_KINDS, EPISODIC_KINDS
from app.core.memory.store import BRAND_CONTEXT_TOKEN_CAP, BrandMemoryStore
from app.db.models import Base, BrandMemoryItem, BrandProfile, Business
from app.schemas.tools import MemoryScope

NOW = datetime.now(timezone.utc)
BIZ = "biz_nile"
OTHER = "biz_other"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    name: str
    passed: bool
    detail: str
    data: dict = field(default_factory=dict)


RESULTS: list[Result] = []


def record(name: str, passed: bool, detail: str, **data) -> None:
    RESULTS.append(Result(name, passed, detail, data))
    print(("PASS " if passed else "FAIL ") + name + " -- " + detail)


async def make_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    return engine, Session


async def seed_business(db, business_id: str, name: str) -> None:
    db.add(Business(id=business_id, name=name, industry="specialty coffee",
                    country="EG", language="ar", currency="EGP", monthly_budget=60000))
    db.add(BrandProfile(id=f"bp_{business_id}", business_id=business_id,
                        brand_voice="ودود ومباشر، بدون مبالغة",
                        visual_style="earthy tones, natural light",
                        target_audience="موظفو مكاتب في القاهرة وأسيوط، 25-40",
                        key_offers=["اشتراك قهوة شهري", "توصيل مجاني داخل القاهرة"],
                        goals=["زيادة الاشتراكات"], competitors=["Bean There", "Kaffa"]))
    await db.flush()


# --------------------------------------------------------------------------- #
# 1-4. month-1 simulation
# --------------------------------------------------------------------------- #
MONTH_WRITES = [
    # (week, kind, text, language)
    (0, "brand_identity", "هوية العلامة: محمصة النيل، قهوة مختصة، نبرة ودودة ومباشرة.", "ar"),
    (0, "offer", "خصم 20% على الاشتراك السنوي، وتوصيل مجاني داخل القاهرة.", "ar"),
    (0, "offer", "Free delivery inside Cairo on every subscription.", "en"),
    (0, "audience", "موظفو المكاتب الذين يطلبون فرق 10-50 شخص.", "ar"),
    (0, "voice_example", "We talk like a friend who knows coffee, not a barista snob.", "en"),
    (1, "website_analysis", "Checkout has 4 steps and no wallet payment; trust badges missing.", "en"),
    (1, "competitor_note", "Bean There leads on price: 3 bags for 999 EGP.", "en"),
    (1, "competitor_note", "منافس كافا يتفوق على إنستجرام بصورة المنتج.", "ar"),
    (2, "market_research", "Search demand for 'cold brew cairo' grew 38% YoY.", "en"),
    (2, "market_gaps", "No premium sugar-free iced coffee between 50-70 EGP.", "en"),
    (2, "strategy_blueprint", "الشهر الأول: اختبار UGC ريلز. الميزانية 60% ميتا، 30% تيك توك.", "ar"),
    (3, "learning", "الإعلانات القصيرة (15 ثانية) تفوقت على الكاروسيل بنسبة 22%.", "ar"),
    (3, "learning", "Static carousels underperformed short UGC reels.", "en"),
    (3, "chat_attachment", "عرض-السعر-يناير.pdf", "ar"),
    (3, "reference_analysis", "Competitor ad: warm tones, product close-up, price anchor 349 EGP.", "en"),
]

QUERIES = [
    # (query, expected kind, language-of-query)
    ("ما هو العرض الحالي؟", "offer", "ar"),
    ("What is our current offer?", "offer", "en"),
    ("من هو جمهورنا المستهدف؟", "audience", "ar"),
    ("من هم المنافسون؟", "competitor_note", "en"),
    ("ما خطة الشهر الأول؟", "strategy_blueprint", "ar"),
    ("ماذا تعلمنا من الإعلانات القصيرة؟", "learning", "ar"),
    ("What did we learn about short ads?", "learning", "en"),
    ("ما هي فجوات السوق؟", "market_gaps", "ar"),
    ("cold brew demand", "market_research", "en"),
    ("مشاكل صفحة الدفع", "website_analysis", "ar"),
]


async def month_one(db) -> None:
    import random
    random.seed(7)
    store = BrandMemoryStore(db)
    for week, kind, text, lang in MONTH_WRITES:
        item = await store.write_item(business_id=BIZ, kind=kind, text=text, source="sim")
        item.created_at = NOW - timedelta(days=28 - week * 7)
        await db.flush()
    await db.commit()
    print(f"\n[sim] {await store.count_items(BIZ)} memory rows after month 1\n")


async def measure_context_cost(db) -> None:
    store = BrandMemoryStore(db)
    tokens, dupes = [], 0
    for task in ("chat", "competitor_research", "creative_brief", "marketing_strategy"):
        for q, _, _ in QUERIES:
            ctx = await store.build_context(
                business_id=BIZ, scope=MemoryScope(query=q), task_name=task
            )
            tokens.append(ctx.token_estimate)
            ids = [i.memory_id for i in ctx.items]
            dupes += len(ids) - len(set(ids))
    tokens.sort()
    p50, p95, mx = statistics.median(tokens), tokens[int(len(tokens) * 0.95)], max(tokens)
    record("prompt cost stays under the cap", mx <= BRAND_CONTEXT_TOKEN_CAP,
           f"memory tokens per turn: p50={p50:.0f} p95={p95:.0f} max={mx:.0f} (cap {BRAND_CONTEXT_TOKEN_CAP})",
           p50=p50, p95=p95, max=mx)
    record("no memory item injected twice in one prompt", dupes == 0,
           f"{dupes} duplicate item ids across 40 assembled contexts")


async def measure_retrieval(db) -> None:
    store = BrandMemoryStore(db)
    hits, misses, cross_lang_misses = 0, [], []
    for q, expected, lang in QUERIES:
        ctx = await store.build_context(
            business_id=BIZ, scope=MemoryScope(query=q), task_name="chat"
        )
        kinds = [i.kind for i in ctx.items] + [i.kind for i in ctx.learnings]
        if expected in kinds:
            hits += 1
        else:
            misses.append((q, expected, kinds))
            if (lang == "ar") != any("\u0600" <= c <= "\u06FF" for c in "".join(kinds)):
                cross_lang_misses.append((q, expected))
    record("retrieval hit rate", hits >= len(QUERIES) * 0.8,
           f"{hits}/{len(QUERIES)} queries returned the expected kind ({hits/len(QUERIES):.0%})",
           hits=hits, misses=misses)
    for q, expected, kinds in misses:
        print(f"      missed: {q!r} -> expected {expected}, got {kinds}")
    record("cross-language recall", len(cross_lang_misses) == 0,
           f"{len(cross_lang_misses)} misses look like language mismatch: {cross_lang_misses}")


async def measure_latency(db) -> None:
    store = BrandMemoryStore(db)
    async def timed(n=25):
        samples = []
        for i in range(n):
            q = QUERIES[i % len(QUERIES)][0]
            t0 = time.perf_counter()
            await store.build_context(business_id=BIZ, scope=MemoryScope(query=q), task_name="chat")
            samples.append((time.perf_counter() - t0) * 1000)
        samples.sort()
        return statistics.median(samples), samples[int(len(samples) * 0.95)]

    p50, p95 = await timed()
    rows_small = await store.count_items(BIZ)
    record("context assembly latency (small workspace)", p95 < 250,
           f"{rows_small} rows -> p50={p50:.1f} ms p95={p95:.1f} ms", p50=p50, p95=p95)

    # scale to 2 000 rows
    for i in range(2000):
        item = await store.write_item(
            business_id=BIZ,
            kind="competitor_note",
            text=f"ملاحظة تنافسية رقم {i} عن السوق المصري والمنافسين والأسعار {i*3} جنيه.",
            source="bulk",
        )
        item.created_at = NOW - timedelta(days=i % 180)
    await db.commit()
    rows_big = await store.count_items(BIZ)
    p50, p95 = await timed()
    record("context assembly latency at scale", p95 < 400,
           f"{rows_big} rows -> p50={p50:.1f} ms p95={p95:.1f} ms", p50=p50, p95=p95)


# --------------------------------------------------------------------------- #
# 5. dedupe
# --------------------------------------------------------------------------- #
async def measure_dedupe(db) -> None:
    store = BrandMemoryStore(db)
    before = await store.count_items(BIZ, kind="offer")
    await store.write_item(business_id=BIZ, kind="offer",
                           text="خصم 20% على الاشتراك السنوي، وتوصيل مجاني داخل القاهرة.")
    after_exact = await store.count_items(BIZ, kind="offer")
    await store.write_item(business_id=BIZ, kind="offer",
                           text="خصم 20 في المية على الاشتراك السنوي وتوصيل مجاني في القاهرة")
    after_paraphrase = await store.count_items(BIZ, kind="offer")
    record("exact duplicate is collapsed", after_exact == before,
           f"{before} -> {after_exact} rows after rewriting the same fact verbatim")
    record("paraphrase is collapsed", after_paraphrase == after_exact,
           f"{after_exact} -> {after_paraphrase} rows after a paraphrased rewrite "
           f"(0 = semantic dedupe, expected 1 extra row)")


# --------------------------------------------------------------------------- #
# 6. lifecycle
# --------------------------------------------------------------------------- #
async def measure_lifecycle(db) -> None:
    store = BrandMemoryStore(db)
    att = await store.write_item(business_id=BIZ, kind="chat_attachment", text="spec-2025.pdf")
    assert att.expires_at is not None
    att.expires_at = NOW - timedelta(days=1)
    await db.flush()
    ctx = await store.build_context(business_id=BIZ, scope=MemoryScope(query="spec"), task_name="chat")
    visible = any(i.memory_id == att.id for i in ctx.items)
    swept = await store.sweep_expired(BIZ)
    await db.commit()
    row = await db.get(BrandMemoryItem, att.id)
    record("expired attachment is not injected", visible is False, "expired row absent from context")
    record("sweep archives expired rows", swept >= 1 and row.archived_at is not None,
           f"swept={swept}, archived_at set={row.archived_at is not None}")

    # permanent kinds must never expire; the plan is a 90-day kind by design
    identity_text = "هوية مؤقتة للاختبار"
    identity = await store.write_item(business_id=BIZ, kind="brand_identity", text=identity_text)
    plan = await store.write_item(business_id=BIZ, kind="strategy_blueprint", text="خطة")
    record("permanent kinds have no expiry", identity.expires_at is None,
           f"brand_identity.expires_at={identity.expires_at}, "
           f"strategy_blueprint.expires_at={plan.expires_at} (90-day kind)")


# --------------------------------------------------------------------------- #
# 7. multi-worker race
# --------------------------------------------------------------------------- #
async def measure_multiworker() -> None:
    from app.ai.memory.consolidator import MemoryConsolidator

    running = peak = 0
    a, b = MemoryConsolidator(), MemoryConsolidator()  # two worker processes

    async def work(_req):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1

    import app.ai.memory.consolidator as mod
    orig = mod._debounce_seconds
    mod._debounce_seconds = lambda: 0
    try:
        for c in (a, b):
            c._compact_session = work
            c._update_profile = work
            c._identity_due = lambda business_id: False
        a.schedule(business_id="biz_x")
        b.schedule(business_id="biz_x")
        await asyncio.sleep(0.2)
    finally:
        mod._debounce_seconds = orig

    record("one writer per workspace across worker processes", peak == 1,
           f"peak concurrent passes with two workers = {peak} "
           f"(1 = safe, 2 = the in-process lock does not span processes)")


# --------------------------------------------------------------------------- #
# 8. write amplification
# --------------------------------------------------------------------------- #
async def measure_write_amplification(db) -> None:
    from sqlalchemy import event

    store = BrandMemoryStore(db)
    updates = {"n": 0}

    def before_cursor(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("UPDATE BRAND_MEMORY_ITEMS"):
            updates["n"] += 1

    engine = db.get_bind()
    target = getattr(engine, "sync_engine", engine)
    event.listen(target, "before_cursor_execute", before_cursor)
    try:
        for q, _, _ in QUERIES[:5]:
            await store.build_context(business_id=BIZ, scope=MemoryScope(query=q), task_name="chat")
    finally:
        event.remove(target, "before_cursor_execute", before_cursor)
    record("write amplification per turn", updates["n"] <= 10,
           f"{updates['n']} UPDATE statements on brand_memory_items for 5 turns "
           f"({updates['n']/5:.1f} per turn, one per touched row per turn)")


# --------------------------------------------------------------------------- #
# 9. metadata size
# --------------------------------------------------------------------------- #
async def measure_payloads(db) -> None:
    ads = {
        "ads_fetched": 30,
        "ads": [{"ad_id": str(1000 + i), "caption": "خصم 30% على القهوة المختصة لفترة محدودة " * 3,
                 "image_url": f"https://cdn.example.com/ads/{i}.jpg", "start": "2026-09-01",
                 "status": "ACTIVE"} for i in range(30)],
    }
    size = len(json.dumps(ads, ensure_ascii=False).encode())
    record("internal payload size stays sane", size < 200_000,
           f"one competitor-ads memory row carries {size/1024:.0f} KB in metadata")


# --------------------------------------------------------------------------- #
# 10. tenant isolation under a shared query
# --------------------------------------------------------------------------- #
async def measure_isolation(db) -> None:
    store = BrandMemoryStore(db)
    await store.write_item(business_id=OTHER, kind="offer", text="عرض سري للمستأجر الآخر")
    ctx = await store.build_context(business_id=BIZ, scope=MemoryScope(query="عرض سري"),
                                    task_name="chat")
    leaked = any("الآخر" in (i.detail or "") for i in ctx.items + ctx.learnings)
    hits = await store.search_ranked(business_id=BIZ, query="عرض سري", top_k=10)
    leaked = leaked or any(h.item.business_id != BIZ for h in hits)
    record("tenant isolation", not leaked, "no cross-workspace row surfaced")


# --------------------------------------------------------------------------- #
async def main() -> None:
    engine, Session = await make_db()
    async with Session() as db:
        await seed_business(db, BIZ, "محمصة النيل")
        await seed_business(db, OTHER, "Tenant B")
        await month_one(db)
        await measure_context_cost(db)
        await measure_retrieval(db)
        await measure_dedupe(db)
        await measure_lifecycle(db)
        await measure_latency(db)
        await measure_write_amplification(db)
        await measure_payloads(db)
        await measure_isolation(db)
    await engine.dispose()
    await measure_multiworker()

    print("\n================ SUMMARY ================")
    passed = sum(1 for r in RESULTS if r.passed)
    for r in RESULTS:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    print(f"\n{passed}/{len(RESULTS)} checks passed")

    out = {"generated_at": NOW.isoformat(), "results": [
        {"name": r.name, "passed": r.passed, "detail": r.detail, "data": r.data}
        for r in RESULTS
    ]}
    out_path = Path(__file__).resolve().parent.parent / "memory_validation_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
