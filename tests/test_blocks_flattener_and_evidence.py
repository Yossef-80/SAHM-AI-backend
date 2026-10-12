"""Tests for canonical blocks flattener, JSONB UTF-8 serialization, evidence wiring, and backfill migration."""

from __future__ import annotations

import json
import logging
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Message
from app.db.session import _make_engine
from app.schemas.blocks import (
    ActionCard,
    ContextItem,
    ContextUsedBlock,
    ErrorBlock,
    Finding,
    FindingsBlock,
    MetricBlock,
    QuestionWithOptionsBlock,
    RecommendationBlock,
    TaskProgressBlock,
    TextBlock,
    blocks_to_plaintext,
)
from app.schemas.common import Evidence
from app.core.orchestrator.blocks import _research_blocks
from scripts.backfill_message_content import run_backfill


# ------------------------------------------------------------------- §2 Fixture --
COMPETITOR_NAMING_FIXTURE = [
    {
        "type": "text",
        "text": "تحليل أنماط تسمية المنافسين في سوق الشموع المصري ومقترحات لعلامتك التجارية 🕯️✨:",
    },
    {
        "type": "findings",
        "title": "النمط الأول: الأسماء المستوحاة من اللاتينية واللغات الأوروبية (European / Latin Roots)",
        "findings": [
            {
                "title": "طبيعة المنافسين",
                "detail": "براندات تركز على الفخامة والشموع العلاجية المعطرة برائحة الصويا الطبيعية.",
                "source": "https://example.com/candelaria-eg",
                "evidence": [
                    {
                        "title": "Candelaria Egypt",
                        "url": "https://example.com/candelaria-eg?utm_source=test&srsltid=123",
                        "snippet": "Hand poured luxury soy candles in Cairo",
                    }
                ],
            },
            {
                "title": "أمثلة لمنافسين",
                "detail": "Candelaria, Aromatica, Pia Scents",
                "source": "https://example.com/aromatica",
            },
            {
                "title": "سر الإلهام",
                "detail": "استخدام كلمات تدل على النور أو العطر باللغات الإيطالية والإسبانية.",
            },
            {
                "title": "مقترحات أسماء ليك (اختار منها)",
                "detail": "Aura Soy, Lumina Home, Vela & Co, Serenita Scents",
            },
        ],
    },
    {
        "type": "findings",
        "title": "النمط الثاني: الأسماء الوظيفية المباشرة (Direct / Descriptive)",
        "findings": [
            {
                "title": "طبيعة المنافسين",
                "detail": "براندات تبرز المكون الأساسي (شمع الصويا أو العسل الطبيعي).",
            },
            {
                "title": "أمثلة لمنافسين",
                "detail": "The Candle Co, Pure Wax Studio",
            },
        ],
    },
]


# ------------------------------------------------------------------- Flattener Unit Tests --

def test_flattener_text_block():
    blocks = [TextBlock(text="مرحبًا بك في سهم! 👋")]
    text = blocks_to_plaintext(blocks)
    assert text == "مرحبًا بك في سهم! 👋"


def test_flattener_findings_with_arabic_emoji_and_evidence():
    findings = [
        Finding(
            title="طبيعة المنافسين",
            detail="براندات تركز على الفخامة والشموع العلاجية 🌿🕯️",
            source="https://example.com/candelaria",
            confidence=0.9,
            evidence=[
                Evidence(
                    title="Candelaria Egypt",
                    url="https://example.com/candelaria",
                    snippet="Hand poured soy candles",
                )
            ],
        ),
        Finding(
            title="أمثلة لمنافسين",
            detail="Candelaria, Aromatica, Pia Scents",
        ),
    ]
    block = FindingsBlock(title="النمط الأول: الأسماء اللاتينية", findings=findings)
    text = blocks_to_plaintext([block])

    assert "النمط الأول: الأسماء اللاتينية" in text
    assert "- طبيعة المنافسين: براندات تركز على الفخامة والشموع العلاجية 🌿🕯️ (https://example.com/candelaria)" in text
    assert "- أمثلة لمنافسين: Candelaria, Aromatica, Pia Scents" in text
    assert not text.endswith(": ")


def test_flattener_empty_detail_never_emits_dangling_colon():
    findings = [
        Finding(title="نقطة بدون تفاصيل", detail=""),
        Finding(title="نقطة عادية", detail="تفاصيل كاملة"),
    ]
    block = FindingsBlock(title="قائمة", findings=findings)
    text = blocks_to_plaintext([block])

    assert "- نقطة بدون تفاصيل\n" in text or text.endswith("- نقطة بدون تفاصيل")
    assert "- نقطة بدون تفاصيل: " not in text
    assert "- نقطة عادية: تفاصيل كاملة" in text


def test_flattener_all_block_types():
    blocks = [
        TextBlock(text="مقدمة"),
        FindingsBlock(
            title="النتائج",
            findings=[Finding(title="ميزة", detail="سرعة عالية", source="https://sahm.app")],
        ),
        QuestionWithOptionsBlock(
            question="ما هو مجالك المفضل؟",
            options=[{"id": "opt1", "label": "التجارة الإلكترونية"}, {"id": "opt2", "label": "الخدمات"}],
        ),
        RecommendationBlock(
            recommendation_id="rec_1",
            action_type="budget",
            title="زيادة الميزانية",
            rationale="تحسين العائد",
        ),
        MetricBlock(label="العائد الإعلاني (ROAS)", value=4.5),
        ActionCard(action_type="approve", label="تأكيد الخطة", description="اعتماد خطة الشهر القادم"),
        TaskProgressBlock(task_id="t_1", task_name="تحليل المنافسين", persona_id="nour", progress=0.8),
        ContextUsedBlock(items=[ContextItem(kind="brand", label="اسم البراند", detail="شمعة القاهرة")]),
        ErrorBlock(message="حدث خطأ مؤقت في الاتصال"),
        {"type": "future_block_type", "title": "ميزة مستقبلية", "content": "بيانات تجريبية"},
    ]
    text = blocks_to_plaintext(blocks)

    assert "مقدمة" in text
    assert "النتائج" in text
    assert "- ميزة: سرعة عالية (https://sahm.app)" in text
    assert "ما هو مجالك المفضل؟" in text
    assert "1. التجارة الإلكترونية" in text
    assert "2. الخدمات" in text
    assert "**Recommendation:** زيادة الميزانية" in text
    assert "تحسين العائد" in text
    assert "**العائد الإعلاني (ROAS):** 4.5" in text
    assert "**Action:** تأكيد الخطة" in text
    assert "اعتماد خطة الشهر القادم" in text
    assert "**Task تحليل المنافسين:**" in text
    assert "**Context used:**" in text
    assert "شمعة القاهرة" in text
    assert "**Error:** حدث خطأ مؤقت في الاتصال" in text
    assert "ميزة مستقبلية" in text


# ------------------------------------------------------------------- §2 Regression Fixture --

def test_regression_fixture_lossless_roundtrip():
    text = blocks_to_plaintext(COMPETITOR_NAMING_FIXTURE)

    # Prove that every single competitor brand name and suggestion exists in content
    assert "Candelaria" in text
    assert "Aromatica" in text
    assert "Pia Scents" in text
    assert "Aura Soy" in text
    assert "Lumina Home" in text
    assert "Vela & Co" in text
    assert "Serenita Scents" in text
    assert "The Candle Co" in text
    assert "Pure Wax Studio" in text

    # Assert no colon-starvation occurred
    assert "- طبيعة المنافسين:\n" not in text
    assert "- أمثلة لمنافسين:\n" not in text
    assert "- سر الإلهام:\n" not in text
    assert "- مقترحات أسماء ليك (اختار منها):\n" not in text


# ------------------------------------------------------------------- JSONB UTF-8 Serialization --

@pytest.mark.anyio
async def test_jsonb_serialization_utf8_no_escape_soup(db_session: AsyncSession):
    conversation = Conversation(
        id="conv_test_utf8",
        business_id="biz_test",
        title="محادثة عربية تجريبية 🇪🇬",
        context={},
    )
    db_session.add(conversation)
    await db_session.flush()

    blocks_payload = [
        {
            "id": None,
            "type": "text",
            "text": "خطوة ممتازة! إليك قائمة المنافسين في السوق المصري 🕯️✨",
        },
        {
            "type": "findings",
            "title": "المنافسين المباشرين",
            "findings": [
                {
                    "title": "كاندلاريا",
                    "detail": "متخصص في شموع الصويا الطبيعية",
                    "source": "https://example.com/candelaria",
                    "evidence": [],
                }
            ],
        },
    ]

    message = Message(
        id="msg_test_utf8",
        conversation_id="conv_test_utf8",
        role="assistant",
        content=blocks_to_plaintext(blocks_payload),
        blocks=blocks_payload,
    )
    db_session.add(message)
    await db_session.commit()

    # Query back using raw SQL / ORM
    res = await db_session.execute(select(Message).where(Message.id == "msg_test_utf8"))
    loaded_msg = res.scalar_one()

    # Assert native data matches exactly
    assert loaded_msg.blocks == blocks_payload
    assert loaded_msg.blocks[0]["text"] == "خطوة ممتازة! إليك قائمة المنافسين في السوق المصري 🕯️✨"

    # Verify JSON serialization has NO \uXXXX escape sequences
    serialized = json.dumps(loaded_msg.blocks, ensure_ascii=False)
    assert "\\u062e" not in serialized
    assert "\\u" not in serialized
    assert "خطوة ممتازة" in serialized
    assert "كاندلاريا" in serialized


# ------------------------------------------------------------------- Evidence Wiring Tests --

def test_research_blocks_evidence_wiring(caplog):
    output = {
        "competitors": [
            {
                "name": "Candelaria",
                "positioning": "Luxury soy candles",
                "strengths": ["Brand loyalty"],
                "source": "https://candelaria.eg",
            }
        ],
        "evidence": [
            {
                "title": "Candelaria Cairo",
                "url": "https://candelaria.eg/?utm_medium=social&srsltid=999",
                "snippet": "Handmade candles in Cairo",
            }
        ],
    }

    blocks = _research_blocks("competitor_research", output)
    assert len(blocks) >= 1
    finding_block = blocks[0]
    assert finding_block["type"] == "findings"
    finding = finding_block["findings"][0]

    # Verify evidence was attached
    assert len(finding["evidence"]) == 1
    assert finding["evidence"][0]["title"] == "Candelaria Cairo"
    # Verify tracking params stripped
    assert "utm_medium" not in finding["evidence"][0]["url"]
    assert "srsltid" not in finding["evidence"][0]["url"]
    assert finding["evidence"][0]["url"] == "https://candelaria.eg/"


def test_research_blocks_missing_evidence_warning(caplog):
    caplog.set_level(logging.WARNING)
    output = {
        "competitors": [
            {
                "name": "Ghost Brand",
                "positioning": "Unknown",
            }
        ],
        "evidence": [],
    }

    _research_blocks("competitor_research", output)
    warning_logged = any("research_finding_missing_evidence" in rec.message for rec in caplog.records)
    assert warning_logged


# ------------------------------------------------------------------- Backfill Migration Tests --

@pytest.mark.anyio
async def test_backfill_migration_idempotent(db_session: AsyncSession):
    conv = Conversation(
        id="conv_backfill_test",
        business_id="biz_test",
        title="محادثة لاختبار الترحيل",
        summary="ملخص قديم غير دقيق يعتمد على نصوص مفقودة التفاصيل",
    )
    db_session.add(conv)
    await db_session.flush()

    # Message 1: Starved content with empty colons (Problem B reproduction)
    starved_content = "النمط الأول: الأسماء المستوحاة من اللاتينية\n- طبيعة المنافسين: \n- أمثلة لمنافسين: \n"
    msg1 = Message(
        id="msg_starved",
        conversation_id="conv_backfill_test",
        role="assistant",
        content=starved_content,
        blocks=COMPETITOR_NAMING_FIXTURE,
    )

    # Message 2: Message lacking blocks
    msg2 = Message(
        id="msg_no_blocks",
        conversation_id="conv_backfill_test",
        role="assistant",
        content="نص عادي بدون بلوكات",
        blocks=[],
    )

    db_session.add_all([msg1, msg2])
    await db_session.commit()

    # Run 1: Dry run
    dry_counts = await run_backfill(dry_run=True, session=db_session)
    assert dry_counts["total_inspected"] == 2
    assert dry_counts["updated"] == 1
    assert dry_counts["lacking_blocks"] == 1
    assert dry_counts["summaries_invalidated"] == 1

    # Verify content was NOT committed during dry run
    res = await db_session.execute(select(Message).where(Message.id == "msg_starved"))
    assert res.scalar_one().content == starved_content

    # Run 2: Real execution
    real_counts = await run_backfill(dry_run=False, session=db_session)
    assert real_counts["updated"] == 1
    assert real_counts["summaries_invalidated"] == 1

    # Verify content is now lossless
    res = await db_session.execute(select(Message).where(Message.id == "msg_starved"))
    updated_msg = res.scalar_one()
    assert "Candelaria" in updated_msg.content
    assert "Aura Soy" in updated_msg.content
    assert "- طبيعة المنافسين:\n" not in updated_msg.content

    # Verify conversation summary was invalidated
    res_conv = await db_session.execute(select(Conversation).where(Conversation.id == "conv_backfill_test"))
    assert res_conv.scalar_one().summary is None

    # Run 3: Second run must be completely idempotent (0 updated, 1 unchanged)
    idempotent_counts = await run_backfill(dry_run=False, session=db_session)
    assert idempotent_counts["updated"] == 0
    assert idempotent_counts["unchanged"] == 1
    assert idempotent_counts["lacking_blocks"] == 1
    assert idempotent_counts["summaries_invalidated"] == 0
