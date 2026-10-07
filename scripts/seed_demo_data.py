"""Seed a realistic demo dataset into whatever ``DATABASE_URL`` points at.

Why this exists
---------------
Six of the seven ``BrandProfile`` columns (``brand_voice``, ``visual_style``,
``target_audience``, ``key_offers``, ``goals``, ``competitors``) have no endpoint
that writes them, and only two of the seven ``MEMORY_KINDS`` (``learning`` and
``reference_analysis``) are ever written by production code. A fresh database is
therefore permanently NULL/empty and the assistant has nothing to work with. This
script fills the gap so the frontend has something to render.

It targets the configured database and nothing else: the URL is read from
``app.config.get_settings()`` (env / ``.env``), never hardcoded, and it is never
printed with credentials. Use ``DATABASE_URL`` to aim it somewhere safe::

    DATABASE_URL=sqlite+aiosqlite:///./demo.db python scripts/seed_demo_data.py

Usage (from the ``sahm-ai backend`` folder)::

    python scripts/seed_demo_data.py                # idempotent insert
    python scripts/seed_demo_data.py --reset        # delete demo rows first
    python scripts/seed_demo_data.py --json         # summary as JSON

``Base.metadata.create_all`` runs first, so it works on an empty database with no
Alembic migration applied.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Allow `python scripts/seed_demo_data.py` from the backend root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, func, select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.core.memory.store import MEMORY_KINDS, BrandMemoryStore  # noqa: E402
from app.core.orchestrator.blocks import (  # noqa: E402
    findings_block,
    metric_block,
    text_block,
)
from app.db.models import (  # noqa: E402
    Base,
    BrandMemoryItem,
    BrandProfile,
    Business,
    Campaign,
    Conversation,
    Creative,
    Insight,
    Integration,
    Message,
    Recommendation,
    UsageRecord,
)
from app.db.session import create_all, dispose_engine, get_sessionmaker  # noqa: E402
from app.schemas.blocks import Finding  # noqa: E402
from app.schemas.common import Evidence  # noqa: E402

# --------------------------------------------------------------------- ids ---

#: Fixed ids so the script is idempotent: a second run finds the same rows.
DEMO_BUSINESS_ID = "biz_demo"
DEMO_WORKSPACE_ID = "ws_demo"
DEMO_PROFILE_ID = "bp_demo"

CAMPAIGN_IDS = {
    "draft": "cmp_demo_subscription",
    "active": "cmp_demo_ramadan",
    "paused": "cmp_demo_office",
    "completed": "cmp_demo_coldbrew",
}

#: An inline SVG data URL -- renders offline, no external host, and comfortably
#: inside the ``creatives.url`` String(1000) limit.
_PLACEHOLDER_SVG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='400' height='400'>"
    "<rect width='400' height='400' fill='%231f2937'/>"
    "<circle cx='200' cy='200' r='88' fill='%23d9a066'/>"
    "</svg>"
)


def _day(offset: int) -> datetime:
    """Midnight today +/- ``offset`` days, so demo dates always look current."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return today + timedelta(days=offset)


# ------------------------------------------------------------------- data ---

_BUSINESS = {
    "id": DEMO_BUSINESS_ID,
    "workspace_id": DEMO_WORKSPACE_ID,
    "name": "Cairo Coffee Co",
    "industry": "F&B",
    "website_url": "https://example.com",
    "country": "EG",
    "language": "en",
    "currency": "EGP",
    "monthly_budget": 120_000.0,
}

#: Every column here is currently NULL in production: no endpoint writes them.
_PROFILE = {
    "id": DEMO_PROFILE_ID,
    "business_id": DEMO_BUSINESS_ID,
    "brand_voice": (
        "Warm, direct and a little cheeky. We talk like a friend who really knows "
        "coffee -- never like a barista snob, never corporate."
    ),
    "visual_style": (
        "Warm neutrals with a single amber accent (#d9a066 on #1f2937). Natural "
        "window light, matte ceramics, real hands, no stock-smiling models."
    ),
    "target_audience": (
        "Urban Cairo professionals aged 25-40 who buy coffee on the way to work "
        "and subscribe so they stop thinking about it. Secondary: office managers "
        "ordering for teams of 10-50."
    ),
    "key_offers": [
        "Subscription beans: 12% off every delivery, free delivery in Cairo, cancel anytime",
        "Same-day delivery within Greater Cairo when ordered before 2pm",
        "Office catering: 5kg boxes for teams of 10+, weekly or one-off",
        "First-order discount code HELLO12",
    ],
    "goals": [
        "Reach 1,000 active subscriptions within two quarters",
        "Lift repeat purchase rate from 24% to 35%",
        "Win 12 office catering accounts in Zamalek and Downtown",
    ],
    "competitors": [
        "Bean There -- leads on price and app-based loyalty",
        "Kaffa Roasters -- leads on Instagram aesthetics and cold-brew range",
        "Blue Bottle Cairo -- premium positioning, weak local delivery",
    ],
}

#: (kind, text, metadata). One entry per MEMORY_KINDS value -- all seven.
#: Only ``learning`` and ``reference_analysis`` are written by production code.
_MEMORY_ITEMS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "voice_example",
        "Your first bag is on us -- use code HELLO12 before Friday and we'll "
        "deliver it before your Monday standup.",
        {"channel": "instagram_reel", "tone": "playful", "do": "lead with the offer"},
    ),
    (
        "visual_style",
        "Matte ceramic cup, amber liquid, morning window light, one flat lay "
        "angle, generous negative space for copy.",
        {"palette": ["#1f2937", "#d9a066", "#f5f5f4"], "lighting": "natural"},
    ),
    (
        "offer",
        "Subscription beans: 12% off every delivery, free delivery in Cairo, "
        "cancel anytime. No minimum term.",
        {"discount_pct": 12, "cancellation": "anytime", "price_band": "mid"},
    ),
    (
        "audience",
        "Office managers in Downtown and Zamalek ordering for teams of 10-50; "
        "they care about reliable delivery windows, not about origin stories.",
        {"segment": "b2b_office", "size": "10-50", "top_jobs": ["order", "reorder"]},
    ),
    (
        "competitor_note",
        "Bean There undercut us by 15% on the entry bag and won the discount "
        "search terms; Kaffa owns the cold-brew visual language.",
        {"competitor": "Bean There", "gap": "no free delivery", "watch": "Kaffa"},
    ),
    (
        "learning",
        "What worked: the first-order discount creative lifted subscription "
        "signups 18% at a 12% cost increase. What did not: broad-interest "
        "targeting above 40, which doubled CPM with flat conversion.",
        {
            "what_worked": "first-order discount creative",
            "lift_pct": 18,
            "cpm_increase_pct": 12,
            "applies_to_future": True,
        },
    ),
    (
        "reference_analysis",
        "Reference: competitor reel with a slow pour and on-screen price. Style "
        "spec: tight close-up, amber rim light, single text card, 9:16.",
        {
            "style": {
                "palette": ["#1f2937", "#d9a066"],
                "shot": "tight_close_up",
                "text_cards": 1,
                "aspect_ratio": "9:16",
            },
            "source_url": "https://example.com/reference-reel",
        },
    ),
]

_CAMPAIGNS: list[dict[str, Any]] = [
    {
        "id": CAMPAIGN_IDS["draft"],
        "business_id": DEMO_BUSINESS_ID,
        "name": "Subscription Drive Q1",
        "status": "draft",
        "objective": "conversions",
        "channel": "meta",
        "budget": 45_000.0,
        "start_date": _day(7),
        "end_date": _day(52),
        "target_audience": "Lookalike 1% of subscribers, Cairo, 25-40",
        "notes": "Waiting on two new creative variations before it can launch.",
    },
    {
        "id": CAMPAIGN_IDS["active"],
        "business_id": DEMO_BUSINESS_ID,
        "name": "Ramadan Night Market",
        "status": "active",
        "objective": "awareness",
        "channel": "meta",
        "budget": 60_000.0,
        "start_date": _day(-21),
        "end_date": _day(14),
        "target_audience": "Cairo 18-35, broad with a 25-40 core",
        "meta_campaign_id": "238100000000001",
        "notes": "Live. Reels are carrying it; static is being cut.",
    },
    {
        "id": CAMPAIGN_IDS["paused"],
        "business_id": DEMO_BUSINESS_ID,
        "name": "Office Catering Retargeting",
        "status": "paused",
        "objective": "leads",
        "channel": "meta",
        "budget": 18_000.0,
        "start_date": _day(-60),
        "end_date": _day(-5),
        "target_audience": "Office managers, Downtown and Zamalek",
        "meta_campaign_id": "238100000000002",
        "notes": "Paused by Salma: CPL doubled after the audience widened.",
    },
    {
        "id": CAMPAIGN_IDS["completed"],
        "business_id": DEMO_BUSINESS_ID,
        "name": "Cold Brew Launch",
        "status": "completed",
        "objective": "conversions",
        "channel": "meta",
        "budget": 32_000.0,
        "start_date": _day(-95),
        "end_date": _day(-40),
        "target_audience": "Cairo 20-40, interest in cold brew",
        "meta_campaign_id": "238100000000003",
        "notes": "Beat the CAC target. Learning written to brand memory.",
    },
]

_CREATIVES: list[dict[str, Any]] = [
    {
        "id": "crv_demo_reel_1",
        "business_id": DEMO_BUSINESS_ID,
        "campaign_id": CAMPAIGN_IDS["active"],
        "kind": "video",
        "url": "https://example.com/demo/creatives/ramadan-reel.mp4",
        "prompt": "Slow pour of cold brew over ice, amber rim light, 9:16.",
        "aspect_ratio": "9:16",
        "persona_id": "layla",
        "task_name": "video_prompt",
        "style_spec": {"palette": ["#1f2937", "#d9a066"], "shot": "tight_close_up"},
    },
    {
        "id": "crv_demo_static_1",
        "business_id": DEMO_BUSINESS_ID,
        "campaign_id": CAMPAIGN_IDS["active"],
        "kind": "image",
        "url": _PLACEHOLDER_SVG,
        "prompt": "Flat lay of two iced coffees on a marble counter, natural light.",
        "aspect_ratio": "1:1",
        "persona_id": "layla",
        "task_name": "image_prompt",
        "status": "draft",
    },
    {
        "id": "crv_demo_static_2",
        "business_id": DEMO_BUSINESS_ID,
        "campaign_id": CAMPAIGN_IDS["completed"],
        "kind": "image",
        "url": _PLACEHOLDER_SVG,
        "prompt": "Cold brew bottle on a sunlit balcony, condensation, 4:5 crop.",
        "aspect_ratio": "4:5",
        "persona_id": "layla",
        "task_name": "image_prompt",
    },
    {
        "id": "crv_demo_office_1",
        "business_id": DEMO_BUSINESS_ID,
        "campaign_id": CAMPAIGN_IDS["paused"],
        "kind": "image",
        "url": "https://example.com/demo/creatives/office-catering.png",
        "prompt": "Five kraft catering boxes stacked beside a desk calendar.",
        "aspect_ratio": "1:1",
        "persona_id": "omar",
        "task_name": "creative_brief",
    },
]

_INSIGHTS: list[dict[str, Any]] = [
    {
        "id": "ins_demo_roas",
        "campaign_id": CAMPAIGN_IDS["active"],
        "kind": "performance",
        "title": "Reels are carrying the campaign at 2.4x ROAS",
        "detail": "Reel placements return 2.4x while static sits at 0.9x over the "
        "same 21 days.",
        "severity": "info",
        "metrics": {"roas_reels": 2.4, "roas_static": 0.9, "spend": 41200.0},
        "persona_id": "karim",
    },
    {
        "id": "ins_demo_cpl",
        "campaign_id": CAMPAIGN_IDS["paused"],
        "kind": "efficiency",
        "title": "Cost per lead doubled after the audience widened",
        "detail": "CPL moved from 180 EGP to 371 EGP between week 2 and week 6 "
        "after the targeting change.",
        "severity": "warning",
        "metrics": {"cpl_before": 180.0, "cpl_after": 371.0, "change_pct": 106.0},
        "persona_id": "karim",
    },
    {
        "id": "ins_demo_fatigue",
        "campaign_id": CAMPAIGN_IDS["active"],
        "kind": "creative_fatigue",
        "title": "Frequency above 4.5 on the same reel for 9 days",
        "detail": "Frequency has crossed the fatigue threshold; CTR is down 31% "
        "week over week on the leading creative.",
        "severity": "critical",
        "metrics": {"frequency": 4.6, "ctr_change_pct": -31.0},
        "persona_id": "salma",
    },
    {
        "id": "ins_demo_winner",
        "campaign_id": CAMPAIGN_IDS["completed"],
        "kind": "result",
        "title": "Cold Brew launch beat the CAC target by 18%",
        "detail": "Final CAC was 214 EGP against a 260 EGP target.",
        "severity": "info",
        "metrics": {"cac": 214.0, "cac_target": 260.0, "variance_pct": -17.7},
        "persona_id": "karim",
    },
]


def _evidence(source: str, detail: str, url: str | None = None) -> dict[str, Any]:
    """One evidence pointer, in the frontend's camelCase wire shape."""
    return Evidence(source=source, detail=detail, url=url).model_dump(
        mode="json", by_alias=True
    )


_RECOMMENDATIONS: list[dict[str, Any]] = [
    {
        "id": "rec_demo_pause",
        "campaign_id": CAMPAIGN_IDS["active"],
        "insight_id": "ins_demo_fatigue",
        "action_type": "pause_ad",
        "title": "Pause the fatigued Ramadan reel",
        "rationale": "Frequency is 4.6 with CTR down 31%; pausing costs reach but "
        "protects the campaign's remaining budget.",
        "payload": {"ad_id": "238100000000001_1", "reason": "creative_fatigue"},
        "evidence": [_evidence("get_campaign_insights", "frequency 4.6, CTR -31%")],
        "status": "pending",
        "persona_id": "salma",
        "requires_approval": True,
        "estimated_impact": "+18% CTR on remaining budget",
    },
    {
        "id": "rec_demo_budget",
        "campaign_id": CAMPAIGN_IDS["active"],
        "insight_id": "ins_demo_roas",
        "action_type": "adjust_budget",
        "title": "Shift 15% of budget from static to reels",
        "rationale": "Reels return 2.4x against 0.9x on static in the same window.",
        "payload": {"from": "STATIC", "to": "REELS", "shift_pct": 15},
        "evidence": [_evidence("get_campaign_insights", "ROAS 2.4 vs 0.9")],
        "status": "approved",
        "persona_id": "salma",
        "requires_approval": True,
        "estimated_impact": "+0.2 blended ROAS",
        "approved_by": "demo.user@example.com",
        "approved_at": _day(-2),
    },
    {
        "id": "rec_demo_creative",
        "campaign_id": CAMPAIGN_IDS["draft"],
        "insight_id": None,
        "action_type": "create_creative",
        "title": "Produce two new subscription-drive creatives",
        "rationale": "The campaign cannot launch with a single creative; Layla's "
        "brief is ready.",
        "payload": {"count": 2, "brief": "Subscription Drive Q1"},
        "evidence": [_evidence("user_input", "Campaign notes flag missing creative")],
        "status": "rejected",
        "persona_id": "omar",
        "requires_approval": True,
        "estimated_impact": "unblocks launch",
        "approved_by": "demo.user@example.com",
        "approved_at": _day(-3),
    },
    {
        "id": "rec_demo_targeting",
        "campaign_id": CAMPAIGN_IDS["paused"],
        "insight_id": "ins_demo_cpl",
        "action_type": "change_targeting",
        "title": "Return office retargeting to the original audience",
        "rationale": "The widened audience doubled CPL with no lead-quality gain.",
        "payload": {"audience": "original_office_managers"},
        "evidence": [_evidence("get_breakdowns", "CPL 180 -> 371 EGP")],
        "status": "executed",
        "persona_id": "salma",
        "requires_approval": True,
        "estimated_impact": "CPL back to ~200 EGP",
        "approved_by": "demo.user@example.com",
        "approved_at": _day(-6),
        "executed_at": _day(-6),
        "execution_result": {"status": "ok", "detail": "audience restored"},
    },
]

_CONVERSATIONS: list[dict[str, Any]] = [
    {
        "id": "cnv_demo_strategy",
        "title": "Should we push subscriptions or office catering?",
        "context": {"campaign_id": CAMPAIGN_IDS["draft"]},
        "messages": [
            {
                "id": "msg_demo_strategy_1",
                "role": "user",
                "content": "Should we push subscriptions or office catering next quarter?",
                "blocks": [text_block("Should we push subscriptions or office catering?")],
            },
            {
                "id": "msg_demo_strategy_2",
                "role": "assistant",
                "persona_id": "omar",
                "content": "Subscriptions first, catering second.",
                "blocks": [
                    text_block(
                        "Subscriptions first. They need no new supply chain, and the "
                        "retention maths is already proven."
                    ),
                    findings_block(
                        "What the numbers say",
                        [
                            Finding(
                                title="Subscription CAC is 2.1x lower",
                                detail="214 EGP vs 450 EGP blended on catering.",
                                confidence=0.8,
                                source="get_campaign_insights",
                            ),
                            Finding(
                                title="Catering has a 90-day sales cycle",
                                detail="Office managers decide in budget cycles, so a "
                                "Q1 push lands after the budget is already set.",
                                confidence=0.6,
                                source="user_input",
                            ),
                        ],
                    ),
                    metric_block("Subscription CAC", 214.0, unit="EGP", trend="down"),
                ],
            },
        ],
    },
    {
        "id": "cnv_demo_ramadan",
        "title": "How is Ramadan Night Market doing?",
        "context": {"campaign_id": CAMPAIGN_IDS["active"]},
        "messages": [
            {
                "id": "msg_demo_ramadan_1",
                "role": "user",
                "content": "How is Ramadan Night Market doing?",
                "blocks": [text_block("How is Ramadan Night Market doing?")],
            },
            {
                "id": "msg_demo_ramadan_2",
                "role": "assistant",
                "persona_id": "karim",
                "content": "Reels are winning; static is not.",
                "blocks": [
                    findings_block(
                        "Performance",
                        [
                            Finding(
                                title="Reels 2.4x ROAS, static 0.9x",
                                detail="Same 21-day window, same creative family.",
                                confidence=0.9,
                                source="get_campaign_insights",
                            ),
                            Finding(
                                title="Frequency is past the fatigue threshold",
                                detail="The leading reel has been running 9 days.",
                                confidence=0.7,
                                source="get_breakdowns",
                            ),
                        ],
                    ),
                    metric_block("Blended ROAS", 1.8, unit="x", trend="up", delta=0.2),
                ],
            },
        ],
    },
]

#: One disconnected provider so the integrations screen has something real to show.
_INTEGRATIONS: list[dict[str, Any]] = [
    {
        "id": "int_demo_meta",
        "business_id": DEMO_BUSINESS_ID,
        "provider": "meta",
        "access_token_enc": None,
        "refresh_token_enc": None,
        "account_id": None,
        "account_name": None,
        "scopes": ["ads_read", "business_management"],
        "status": "disconnected",
    },
]

_USAGE: list[dict[str, Any]] = [
    {"persona_id": "omar", "tier": "strong", "offset_days": -1, "tokens_in": 3120, "tokens_out": 1840},
    {"persona_id": "layla", "tier": "strong", "offset_days": -1, "tokens_in": 2480, "tokens_out": 2210},
    {"persona_id": "karim", "tier": "fast", "offset_days": -1, "tokens_in": 960, "tokens_out": 640},
    {"persona_id": "researcher", "tier": "strong", "offset_days": 0, "tokens_in": 4310, "tokens_out": 2760},
    {"persona_id": "salma", "tier": "strong", "offset_days": 0, "tokens_in": 2050, "tokens_out": 1180},
]


# ----------------------------------------------------------------- helpers ---


async def _count(db: AsyncSession, model: Any, **where: Any) -> int:
    """Rows matching ``where`` (empty filter counts the whole table)."""
    stmt = select(func.count()).select_from(model)
    for column, value in where.items():
        stmt = stmt.where(getattr(model, column) == value)
    return int((await db.execute(stmt)).scalar_one())


async def _add_if_absent(db: AsyncSession, model: Any, obj: Any) -> bool:
    """Insert ``obj`` unless its primary key already exists. Returns True if added."""
    if await db.get(model, obj.id) is not None:
        return False
    db.add(obj)
    await db.flush()
    return True


async def _reset(db: AsyncSession) -> dict[str, int]:
    """Delete every demo row, children before parents so FKs stay satisfied."""
    campaign_ids = list(
        (
            await db.execute(
                select(Campaign.id).where(Campaign.business_id == DEMO_BUSINESS_ID)
            )
        )
        .scalars()
        .all()
    )
    conversation_ids = list(
        (
            await db.execute(
                select(Conversation.id).where(Conversation.business_id == DEMO_BUSINESS_ID)
            )
        )
        .scalars()
        .all()
    )
    no_campaigns = campaign_ids or [""]
    no_conversations = conversation_ids or [""]

    async def _delete(stmt: Any) -> int:
        result = await db.execute(stmt)
        await db.flush()
        return int(result.rowcount or 0)

    removed: dict[str, int] = {}
    removed["messages"] = await _delete(
        delete(Message).where(Message.conversation_id.in_(no_conversations))
    )
    removed["conversations"] = await _delete(
        delete(Conversation).where(Conversation.id.in_(no_conversations))
    )
    removed["recommendations"] = await _delete(
        delete(Recommendation).where(Recommendation.campaign_id.in_(no_campaigns))
    )
    removed["insights"] = await _delete(
        delete(Insight).where(Insight.campaign_id.in_(no_campaigns))
    )
    removed["creatives"] = await _delete(
        delete(Creative).where(Creative.business_id == DEMO_BUSINESS_ID)
    )
    removed["campaigns"] = await _delete(
        delete(Campaign).where(Campaign.id.in_(no_campaigns))
    )
    removed["brand_memory_items"] = await _delete(
        delete(BrandMemoryItem).where(BrandMemoryItem.business_id == DEMO_BUSINESS_ID)
    )
    removed["brand_profiles"] = await _delete(
        delete(BrandProfile).where(BrandProfile.business_id == DEMO_BUSINESS_ID)
    )
    removed["integrations"] = await _delete(
        delete(Integration).where(Integration.business_id == DEMO_BUSINESS_ID)
    )
    removed["usage_records"] = await _delete(
        delete(UsageRecord).where(UsageRecord.business_id == DEMO_BUSINESS_ID)
    )
    removed["businesses"] = await _delete(
        delete(Business).where(Business.id == DEMO_BUSINESS_ID)
    )
    await db.commit()
    return removed


# ------------------------------------------------------------------- seed ---


async def seed(db: AsyncSession, *, reset: bool) -> dict[str, Any]:
    """Insert (or verify) the whole demo dataset. Returns the run summary."""
    warnings: list[str] = []
    removed: dict[str, int] = {}
    if reset:
        removed = await _reset(db)

    added: dict[str, int] = {}

    # -- business + brand profile (the six columns nothing else writes) -------
    added["businesses"] = int(
        await _add_if_absent(db, Business, Business(**_BUSINESS))
    )
    profile = await db.get(BrandProfile, DEMO_PROFILE_ID)
    if profile is None:
        db.add(BrandProfile(**_PROFILE))
        await db.flush()
        added["brand_profiles"] = 1
    else:
        added["brand_profiles"] = 0
        # Keep the profile authoritative on a re-run: these fields are otherwise
        # only ever written by this script.
        for key, value in _PROFILE.items():
            setattr(profile, key, value)

    # -- brand memory, all seven kinds, via the production store -------------
    store = BrandMemoryStore(db)
    memory_added = 0
    for kind, text, metadata in _MEMORY_ITEMS:
        existing = await db.scalar(
            select(func.count())
            .select_from(BrandMemoryItem)
            .where(
                BrandMemoryItem.business_id == DEMO_BUSINESS_ID,
                BrandMemoryItem.kind == kind,
                BrandMemoryItem.text == text,
            )
        )
        if existing:
            continue
        try:
            await store.write_item(
                business_id=DEMO_BUSINESS_ID,
                kind=kind,
                text=text,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - a demo seed must not die on one row
            warnings.append(f"memory item '{kind}' not written: {exc}")
            continue
        memory_added += 1
    added["brand_memory_items"] = memory_added

    # -- campaigns -----------------------------------------------------------
    campaign_added = 0
    for spec in _CAMPAIGNS:
        campaign_added += int(await _add_if_absent(db, Campaign, Campaign(**spec)))
    added["campaigns"] = campaign_added

    # -- creatives -----------------------------------------------------------
    creative_added = 0
    for spec in _CREATIVES:
        creative_added += int(await _add_if_absent(db, Creative, Creative(**spec)))
    added["creatives"] = creative_added

    # -- insights ------------------------------------------------------------
    insight_added = 0
    for spec in _INSIGHTS:
        insight_added += int(await _add_if_absent(db, Insight, Insight(**spec)))
    added["insights"] = insight_added

    # -- recommendations -----------------------------------------------------
    recommendation_added = 0
    for spec in _RECOMMENDATIONS:
        recommendation_added += int(
            await _add_if_absent(db, Recommendation, Recommendation(**spec))
        )
    added["recommendations"] = recommendation_added

    # -- conversations + messages -------------------------------------------
    conversation_added = message_added = 0
    for spec in _CONVERSATIONS:
        conversation_added += int(
            await _add_if_absent(
                db,
                Conversation,
                Conversation(
                    id=spec["id"],
                    business_id=DEMO_BUSINESS_ID,
                    title=spec["title"],
                    context=spec["context"],
                ),
            )
        )
        for message in spec["messages"]:
            message_added += int(
                await _add_if_absent(
                    db, Message, Message(conversation_id=spec["id"], **message)
                )
            )
    added["conversations"] = conversation_added
    added["messages"] = message_added

    # -- integrations + usage ------------------------------------------------
    integration_added = 0
    for spec in _INTEGRATIONS:
        integration_added += int(
            await _add_if_absent(db, Integration, Integration(**spec))
        )
    added["integrations"] = integration_added

    usage_added = 0
    for index, spec in enumerate(_USAGE):
        row = UsageRecord(
            id=f"use_demo_{index + 1}",
            business_id=DEMO_BUSINESS_ID,
            workspace_id=DEMO_WORKSPACE_ID,
            persona_id=spec["persona_id"],
            tier=spec["tier"],
            model=f"mock-{spec['tier']}-1",
            tokens_in=spec["tokens_in"],
            tokens_out=spec["tokens_out"],
            created_at=_day(spec["offset_days"]) + timedelta(hours=index),
        )
        usage_added += int(await _add_if_absent(db, UsageRecord, row))
    added["usage_records"] = usage_added

    await db.commit()

    # -- final state, counted against the database ---------------------------
    counts = {
        "businesses": await _count(db, Business),
        "brand_profiles": await _count(db, BrandProfile),
        "brand_memory_items": await _count(db, BrandMemoryItem, business_id=DEMO_BUSINESS_ID),
        "campaigns": await _count(db, Campaign, business_id=DEMO_BUSINESS_ID),
        "creatives": await _count(db, Creative, business_id=DEMO_BUSINESS_ID),
        "insights": await _count(db, Insight),
        "recommendations": await _count(db, Recommendation),
        "conversations": await _count(db, Conversation, business_id=DEMO_BUSINESS_ID),
        "messages": await _count(db, Message),
        "integrations": await _count(db, Integration, business_id=DEMO_BUSINESS_ID),
        "usage_records": await _count(db, UsageRecord, business_id=DEMO_BUSINESS_ID),
    }

    memory_kinds = list(
        (
            await db.execute(
                select(BrandMemoryItem.kind)
                .where(BrandMemoryItem.business_id == DEMO_BUSINESS_ID)
                .distinct()
            )
        )
        .scalars()
        .all()
    )

    return {
        "business_id": DEMO_BUSINESS_ID,
        "workspace_id": DEMO_WORKSPACE_ID,
        "profile_id": DEMO_PROFILE_ID,
        "reset": reset,
        "removed": removed,
        "inserted": added,
        "counts": counts,
        "campaign_ids": [spec["id"] for spec in _CAMPAIGNS],
        "recommendation_ids": [spec["id"] for spec in _RECOMMENDATIONS],
        "memory_kinds": sorted(memory_kinds),
        "memory_kinds_covered": sorted(set(MEMORY_KINDS) - set(memory_kinds)),
        "warnings": warnings,
    }


# ------------------------------------------------------------------ output ---


def _safe_url(url: str) -> str:
    """The database target with any password masked."""
    scheme, sep, rest = url.partition("://")
    if not sep:
        return scheme
    credentials, at, host = rest.rpartition("@")
    if at and ":" in credentials:
        user, _, _ = credentials.partition(":")
        return f"{scheme}://{user}:***@{host}"
    return f"{scheme}://{rest}"


def _print_summary(summary: dict[str, Any], database: str) -> None:
    print("Demo data seeded.")
    print(f"  database        : {database}")
    print(f"  business_id     : {summary['business_id']}")
    print(f"  workspace_id    : {summary['workspace_id']}")
    print(f"  profile_id      : {summary['profile_id']}")
    if summary["reset"]:
        removed = ", ".join(f"{k}={v}" for k, v in summary["removed"].items())
        print(f"  removed (reset) : {removed or 'nothing to remove'}")
    inserted = ", ".join(
        f"{k}={v}" for k, v in summary["inserted"].items() if v
    )
    print(f"  inserted        : {inserted or 'nothing (already seeded)'}")
    print("  counts:")
    for table, count in summary["counts"].items():
        print(f"    {table:<20}{count}")
    missing = summary["memory_kinds_covered"]
    print(f"  memory kinds    : {', '.join(summary['memory_kinds'])}")
    if missing:
        print(f"  MISSING kinds   : {', '.join(missing)}")
    print("  campaign ids    :")
    for campaign_id in summary["campaign_ids"]:
        print(f"    {campaign_id}")
    print("  recommendation ids:")
    for recommendation_id in summary["recommendation_ids"]:
        print(f"    {recommendation_id}")
    for warning in summary["warnings"]:
        print(f"  ! {warning}")


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Seed demo data into the configured DATABASE_URL."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete existing demo rows (children first) before seeding",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the summary as JSON"
    )
    args = parser.parse_args(argv)

    # Everything comes from Settings: no URL, no key, no secret in this file.
    settings = get_settings()
    # Works on an empty database with no Alembic migration applied.
    await create_all()

    maker = get_sessionmaker()
    try:
        async with maker() as db:
            summary = await seed(db, reset=args.reset)
    finally:
        await dispose_engine()

    summary["database"] = _safe_url(settings.database_url)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary, summary["database"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main(sys.argv[1:])))
    except Exception as exc:  # noqa: BLE001 - a CLI must print, not traceback
        print(f"seed_demo_data failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc