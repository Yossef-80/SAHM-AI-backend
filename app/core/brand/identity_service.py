"""Brand Identity synthesis, extraction, and synchronization service.

Extracts a rich, living Brand Identity profile (name, color palette, voice,
target audience, SWOT analysis, offers, competitors, and market gaps) from
conversations, research findings, and memory items.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.memory.lifecycle import memory_defaults
from app.db.models import (
    BrandMemoryItem,
    BrandProfile as BrandProfileRow,
    Business,
    Conversation,
    Message,
    Profile,
)
from app.schemas.business import (
    BrandIdentityResponse,
    BrandIdentityUpdateRequest,
    ColorPalette,
    IdentityEssence,
    SWOTAnalysis,
    TargetAudienceProfile,
)

logger = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


DEFAULT_PALETTES: dict[str, dict[str, Any]] = {
    "food": {
        "primary": "#DC2626",
        "secondary": "#1E293B",
        "accent": "#F59E0B",
        "background": "#FFFBEB",
        "text": "#0F172A",
        "palette": ["#DC2626", "#F59E0B", "#1E293B", "#FEF3C7", "#10B981"],
    },
    "coffee": {
        "primary": "#B45309",
        "secondary": "#1F2937",
        "accent": "#D97706",
        "background": "#FFFDF7",
        "text": "#1F2937",
        "palette": ["#78350F", "#B45309", "#D97706", "#FDE68A", "#1F2937"],
    },
    "candles": {
        "primary": "#C084FC",
        "secondary": "#312E81",
        "accent": "#F59E0B",
        "background": "#FAF5FF",
        "text": "#1E1B4B",
        "palette": ["#9333EA", "#C084FC", "#F59E0B", "#F3E8FF", "#312E81"],
    },
    "fashion": {
        "primary": "#0F172A",
        "secondary": "#334155",
        "accent": "#059669",
        "background": "#F8FAFC",
        "text": "#0F172A",
        "palette": ["#0F172A", "#334155", "#059669", "#E2E8F0", "#D97706"],
    },
    "auto": {
        "primary": "#EA580C",
        "secondary": "#0F172A",
        "accent": "#38BDF8",
        "background": "#F8FAFC",
        "text": "#020617",
        "palette": ["#EA580C", "#0F172A", "#38BDF8", "#E2E8F0", "#16A34A"],
    },
    "default": {
        "primary": "#2563EB",
        "secondary": "#1E293B",
        "accent": "#F59E0B",
        "background": "#F8FAFC",
        "text": "#0F172A",
        "palette": ["#2563EB", "#1E293B", "#F59E0B", "#E0E7FF", "#10B981"],
    },
}


def _infer_default_palette(industry: str | None, name: str | None) -> ColorPalette:
    text = f"{industry or ''} {name or ''}".lower()
    if any(k in text for k in ["coffee", "بن", "قهوة", "cafe", "roaster"]):
        data = DEFAULT_PALETTES["coffee"]
    elif any(k in text for k in ["candle", "شمع", "شموع", "fragrance", "scent"]):
        data = DEFAULT_PALETTES["candles"]
    elif any(k in text for k in ["food", "نودلز", "اندومي", "أكل", "مطعم", "restaurant", "noodle"]):
        data = DEFAULT_PALETTES["food"]
    elif any(k in text for k in ["fashion", "suit", "ملابس", "سترة", "بدل", "apparel", "wear", "قميص"]):
        data = DEFAULT_PALETTES["fashion"]
    elif any(k in text for k in ["auto", "car", "سيارات", "صيانة", "موتور", "motor"]):
        data = DEFAULT_PALETTES["auto"]
    else:
        data = DEFAULT_PALETTES["default"]
    return ColorPalette(**data)


def _clean_competitor_name(raw: str) -> str:
    cleaned = raw.strip().lstrip("@#*").rstrip("*.,;:!؟?").strip()
    cleaned = cleaned.strip('"\'«»“”')
    return cleaned


async def record_competitors(db: AsyncSession, business_id: str, competitor_names: list[str]) -> list[str]:
    """Persist newly discovered or researched competitors into BrandProfileRow, BrandMemoryItem, and BrandIdentity cache."""
    if not competitor_names or not business_id:
        return []

    biz = await db.get(Business, business_id)
    if not biz:
        return []

    profile_stmt = select(BrandProfileRow).where(BrandProfileRow.business_id == business_id)
    bp = (await db.execute(profile_stmt)).scalars().first()
    if not bp:
        bp = BrandProfileRow(id=f"bp_{business_id}", business_id=business_id, competitors=[])
        db.add(bp)

    existing_comps = list(bp.competitors or [])
    existing_lower = {c.strip().lower() for c in existing_comps}

    ignored_names = {
        "nour", "omar", "salma", "layla", "karim", "sahm", "ai",
        "web_analyzer", "gap_analysis", "reviews", "researcher",
        "competitor", "competitors", "منافس", "المنافس", "منافسين", "المنافسين",
        "undefined", "null", "none", "n/a", "unknown", "brand", "market",
    }

    added: list[str] = []
    for raw in competitor_names:
        cleaned = _clean_competitor_name(raw)
        if not cleaned or len(cleaned) < 2 or len(cleaned) > 60:
            continue
        if cleaned.lower() in ignored_names or cleaned.lower() in existing_lower:
            continue
        if biz.name and cleaned.lower() == biz.name.lower():
            continue

        existing_comps.append(cleaned)
        existing_lower.add(cleaned.lower())
        added.append(cleaned)

    if not added:
        return existing_comps

    from sqlalchemy.orm.attributes import flag_modified
    bp.competitors = list(existing_comps)
    flag_modified(bp, "competitors")

    now_dt = _now()
    for comp in added:
        mem = BrandMemoryItem(
            id=f"mem_comp_{uuid.uuid4().hex[:16]}",
            business_id=business_id,
            kind="competitor_note",
            text=f"Competitor researched and added to Brand Identity: {comp}",
            metadata_={"competitor": comp, "source": "chat_research"},
            created_at=now_dt,
        )
        db.add(mem)

    # Also update any existing BrandIdentity cache in BrandMemoryItem
    mem_stmt = (
        select(BrandMemoryItem)
        .where(BrandMemoryItem.business_id == business_id, BrandMemoryItem.kind == "brand_identity")
        .order_by(BrandMemoryItem.created_at.desc())
        .limit(1)
    )
    mem_item = (await db.execute(mem_stmt)).scalars().first()
    if mem_item and mem_item.metadata_ and "brand_identity" in mem_item.metadata_:
        cached_dict = dict(mem_item.metadata_["brand_identity"])
        cached_dict["competitors"] = list(existing_comps)
        mem_item.metadata_ = {"brand_identity": cached_dict}
        flag_modified(mem_item, "metadata_")

    await db.flush()
    logger.info("recorded_new_competitors", extra={"business_id": business_id, "added": added, "total": len(existing_comps)})
    return existing_comps


async def get_brand_identity(db: AsyncSession, business_id: str) -> BrandIdentityResponse:
    """Retrieve the unified Brand Identity profile for a business."""
    biz = await db.get(Business, business_id)
    if not biz:
        raise ValueError(f"Business not found: {business_id}")

    profile_stmt = select(BrandProfileRow).where(BrandProfileRow.business_id == business_id)
    bp = (await db.execute(profile_stmt)).scalars().first()

    # Check for stored brand_identity memory item
    mem_stmt = (
        select(BrandMemoryItem)
        .where(BrandMemoryItem.business_id == business_id, BrandMemoryItem.kind == "brand_identity")
        .order_by(BrandMemoryItem.created_at.desc())
        .limit(1)
    )
    mem_item = (await db.execute(mem_stmt)).scalars().first()

    if mem_item and mem_item.metadata_ and "brand_identity" in mem_item.metadata_:
        try:
            cached_dict = mem_item.metadata_["brand_identity"]
            cached_dict["business_id"] = business_id
            cached_dict["name"] = biz.name or cached_dict.get("name", "My Brand")
            cached_dict["currency"] = biz.currency or cached_dict.get("currency", "EGP")
            cached_dict["monthly_budget"] = biz.monthly_budget
            # Ensure any competitors in BrandProfileRow are merged
            if bp and bp.competitors:
                cur_comps = list(cached_dict.get("competitors") or [])
                for c in bp.competitors:
                    if c and c not in cur_comps:
                        cur_comps.append(c)
                cached_dict["competitors"] = cur_comps
            return BrandIdentityResponse.model_validate(cached_dict)
        except Exception as exc:
            logger.warning("failed_to_parse_cached_brand_identity", extra={"error": str(exc)})

    # Fallback: construct from BrandProfile and user context
    palette = _infer_default_palette(biz.industry, biz.name)
    if bp and bp.visual_style:
        try:
            if "{" in bp.visual_style and "primary" in bp.visual_style:
                palette = ColorPalette.model_validate(json.loads(bp.visual_style))
        except Exception:
            pass

    prof_stmt = select(Profile).where(Profile.business_id == business_id).limit(1)
    user_prof = (await db.execute(prof_stmt)).scalars().first()
    summary = user_prof.user_context_summary if user_prof else ""

    essence = IdentityEssence(
        tagline=f"Premium {biz.industry or 'Brand'} Experience" if biz.industry else "Excellence in Quality",
        brand_voice=bp.brand_voice if bp and bp.brand_voice else "Professional, authentic, customer-centric",
        tone_keywords=["Authentic", "Trustworthy", "Distinctive", "Approachable"],
    )

    audience = TargetAudienceProfile(
        summary=bp.target_audience if bp and bp.target_audience else "Discerning customers looking for quality and reliability.",
        demographics="Ages 22-45, urban residents, active digital buyers",
        psychographics="Values quality, transparency, and seamless customer service",
        pain_points=["Inconsistent quality from mass market options", "Lack of specialized, tailored solutions"],
        core_segments=["Direct Consumers", "Loyal Repeat Buyers"],
    )

    swot = SWOTAnalysis(
        strengths=["Unique product positioning", "Personalized customer attention", "Agile market adaptation"],
        weaknesses=["Early-stage brand awareness", "Limited marketing distribution scale"],
        opportunities=["Untapped digital social channels", "Growing demand for authentic local offerings"],
        threats=["Aggressive pricing from legacy players", "Rising customer acquisition costs"],
    )

    key_offers = bp.key_offers if bp and bp.key_offers else ["Signature Product Line", "Special Welcome Offer"]
    competitors = bp.competitors if bp and bp.competitors else []

    return BrandIdentityResponse(
        business_id=biz.id,
        name=biz.name,
        industry=biz.industry,
        website_url=biz.website_url,
        country=biz.country,
        language=biz.language or "en",
        currency=biz.currency or "EGP",
        monthly_budget=biz.monthly_budget,
        colors=palette,
        essence=essence,
        audience=audience,
        swot=swot,
        key_offers=key_offers,
        competitors=competitors,
        market_gaps=["High-touch experience missing in mass alternatives", "Authentic storytelling"],
        last_synced_at=bp.updated_at if bp else _now(),
        sources_count=1,
    )


async def _rolling_compress_older_chats(
    older_snippets: list[str],
    current_summary: str,
    target_tokens: int = 3000,
) -> str:
    """Intelligently summarize older conversation history into high-density brand memory.

    Preserves all user decisions, brand choices, colors, target market, products, and competitor insights
    without any arbitrary character truncation.
    """
    if not older_snippets:
        return current_summary

    from app.ai.providers.llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    fast_llm = get_llm().chat
    sys_prompt = SystemMessage(content=(
        "You are the Executive Brand Memory Assistant for SAHM OS.\n"
        "Your role is to distill historical conversations and research into a dense, permanent factual memory "
        "for the Brand Profile and Dashboard.\n\n"
        "RULES:\n"
        "1. Capture ALL explicit user decisions (brand colors, names, taglines, tones, pricing, target audiences).\n"
        "2. Capture key product offerings, customer pain points, competitors researched, and market opportunities.\n"
        "3. Drop transient chit-chat, greetings, and repetitive clarifications.\n"
        "4. Seamlessly merge into the existing brand summary in the dominant language (Arabic/English)."
    ))

    older_text = "\n".join(older_snippets)
    usr_prompt = HumanMessage(content=(
        f"=== EXISTING BRAND MEMORY ===\n{current_summary or 'None'}\n\n"
        f"=== HISTORICAL CONVERSATIONS TO COMPRESS ===\n{older_text}\n\n"
        "Produce the updated, unified historical brand memory:"
    ))
    try:
        res = await fast_llm.ainvoke([sys_prompt, usr_prompt])
        content = str(res.content).strip()
        from app.ai.memory.compressor import enforce_token_cap
        return await enforce_token_cap(content, target_tokens, context_label="brand_rolling_summary")
    except Exception as exc:
        logger.warning("rolling_brand_compression_fallback", extra={"error": str(exc)})
        return current_summary or older_text


def _identity_may_overwrite_profile() -> bool:
    """Whether the identity pass may replace human-saved brand profile fields."""
    try:
        from app.ai.settings import get_ai_settings

        return bool(
            getattr(get_ai_settings(), "memory_identity_overwrite_profile", False)
        )
    except Exception:  # noqa: BLE001 - settings must never break the pass
        return False


class BrandIntelligenceAgent:
    """Dedicated background agent responsible for continuous brand intelligence extraction,
    token-aware rolling memory compression, and live dashboard profile synchronization."""

    async def sync_brand_identity(
        self,
        db: AsyncSession,
        business_id: str,
        llm: Any = None,
    ) -> BrandIdentityResponse:
        """Analyze business conversations and memories to extract and persist Brand Identity."""
        from app.ai.memory.compressor import count_tokens
        from app.ai.settings import get_ai_settings

        ai_settings = get_ai_settings()
        token_limit = ai_settings.brand_memory_token_limit

        biz = await db.get(Business, business_id)
        if not biz:
            raise ValueError(f"Business not found: {business_id}")

        profile_stmt = select(BrandProfileRow).where(BrandProfileRow.business_id == business_id)
        bp = (await db.execute(profile_stmt)).scalars().first()
        if not bp:
            bp = BrandProfileRow(id=f"bp_{business_id}", business_id=business_id)
            db.add(bp)

        # Gather user context summary (LTM)
        prof_stmt = select(Profile).where(Profile.business_id == business_id).limit(1)
        user_prof = (await db.execute(prof_stmt)).scalars().first()
        ltm_summary = user_prof.user_context_summary if user_prof else ""

        # Gather all recent conversations
        conv_stmt = select(Conversation).where(Conversation.business_id == business_id).order_by(Conversation.updated_at.desc()).limit(20)
        conv_rows = (await db.execute(conv_stmt)).scalars().all()
        conv_ids = [c.id for c in conv_rows]

        all_msgs: list[Message] = []
        if conv_ids:
            msg_stmt = (
                select(Message)
                .where(Message.conversation_id.in_(conv_ids))
                .order_by(Message.created_at.asc())
            )
            all_msgs = list((await db.execute(msg_stmt)).scalars().all())

        # Gather existing memory items in full without character truncation
        mem_stmt = (
            select(BrandMemoryItem)
            .where(BrandMemoryItem.business_id == business_id, BrandMemoryItem.kind != "brand_identity")
            .limit(30)
        )
        mem_rows = (await db.execute(mem_stmt)).scalars().all()
        memory_snippets = [f"[{m.kind}] {m.text.strip()}" for m in mem_rows if m.text and m.text.strip()]

        # Prepare messages in full without arbitrary character slicing
        valid_chat_snippets: list[tuple[str, int]] = []
        for m in all_msgs:
            c = (m.content or "").strip()
            if len(c) > 3:
                snippet = f"{m.role.upper()}: {c}"
                valid_chat_snippets.append((snippet, count_tokens(snippet)))

        total_chat_tokens = sum(t for _, t in valid_chat_snippets)
        memory_tokens = sum(count_tokens(s) for s in memory_snippets)
        total_tokens = total_chat_tokens + memory_tokens

        # Check against environment token limit (default 10,000 tokens)
        if total_tokens <= token_limit:
            # Everything fits within token limit: keep all chats and facts in FULL without truncation
            historical_memory = ltm_summary
            active_chats = [snippet for snippet, _ in valid_chat_snippets]
        else:
            # Token limit exceeded: retain recent conversational turns in full (~4,000 tokens),
            # and compress older history into rolling distilled memory
            recent_budget = min(4000, token_limit // 2)
            accumulated_recent = 0
            split_idx = len(valid_chat_snippets)

            for idx in range(len(valid_chat_snippets) - 1, -1, -1):
                snippet_tokens = valid_chat_snippets[idx][1]
                if accumulated_recent + snippet_tokens <= recent_budget:
                    accumulated_recent += snippet_tokens
                    split_idx = idx
                else:
                    break

            older_snippets = [s for s, _ in valid_chat_snippets[:split_idx]]
            active_chats = [s for s, _ in valid_chat_snippets[split_idx:]]

            # Compress older conversation turns into rolling distilled memory
            historical_memory = await _rolling_compress_older_chats(
                older_snippets,
                ltm_summary,
                target_tokens=3000,
            )

            # Persist updated rolling summary to Profile LTM
            if user_prof:
                user_prof.user_context_summary = historical_memory
                await db.flush()

        corpus_parts = [
            f"BUSINESS NAME: {biz.name}",
            f"INDUSTRY: {biz.industry or 'Not specified'}",
        ]
        if historical_memory and historical_memory.strip():
            corpus_parts.append(f"--- ROLLING BRAND MEMORY (COMPRESSED HISTORICAL DECISIONS) ---\n{historical_memory}")
        if active_chats:
            corpus_parts.append(f"--- ACTIVE CHATS & RECENT DECISIONS (IN FULL) ---\n" + "\n".join(active_chats))
        if memory_snippets:
            corpus_parts.append(f"--- MEMORY FACTS (IN FULL) ---\n" + "\n".join(memory_snippets))

        corpus = "\n\n".join(corpus_parts)

        extracted_palette = _infer_default_palette(biz.industry, biz.name)
        brand_name = biz.name
        industry = biz.industry or "Specialized Business"
        tagline = f"The Premier Destination for {industry}"
        mission = f"Delivering excellence and distinctive value in {industry}."
        vision = f"Becoming the benchmark brand for quality and customer loyalty in {industry}."
        brand_voice = bp.brand_voice or "Warm, expert, transparent, and approachable."
        tone_keywords = ["Authentic", "Trustworthy", "Premium", "Engaging"]

        strengths = [
            "Strong focus on product quality and ingredient/material integrity",
            "Direct connection and tailored service for discerning customers",
            "Deep understanding of localized market preferences",
        ]
        weaknesses = [
            "Need to scale digital presence across Meta and TikTok channels",
            "Operational scaling during peak demand periods",
        ]
        opportunities = [
            "High demand for personalized, artisanal alternatives to commercial brands",
            "Leveraging educational content marketing and video storytelling",
            "Expanding subscription or loyalty bundle offerings",
        ]
        threats = [
            "Price sensitivity in competitive market tiers",
            "Established competitors with high ad spend budgets",
        ]
        target_summary = bp.target_audience or "Urban customers who prioritize taste, craft, and reliable quality over generic mass products."
        demographics = "Men and women aged 22-45 with discretionary spending power in metropolitan areas."
        pain_points = [
            "Frustrated with generic commercial alternatives lacking personality",
            "Difficulty finding consistent high-grade quality delivered reliably",
        ]
        offers = bp.key_offers or ["Signature Flagship Selection", "First-Order Welcome Package", "Custom Curated Bundles"]
        competitors = bp.competitors or []
        market_gaps = [
            "Premium quality with unpretentious, friendly customer service",
            "Transparency in sourcing, freshness, and packaging",
        ]

        # Try structured extraction via LLM if available
        try:
            from app.ai.providers.llm import get_llm
            from langchain_core.messages import HumanMessage, SystemMessage
            from pydantic import BaseModel, Field

            class ExtractedBrandData(BaseModel):
                brand_name: str = Field(description="The actual brand or project name clarified by the user in chats")
                industry: str = Field(description="The specific business sector / industry")
                tagline: str = Field(description="A punchy brand slogan or tagline in the same language as chats")
                mission: str = Field(description="Brand mission statement")
                vision: str = Field(description="Brand long-term vision")
                brand_voice: str = Field(description="Tone of voice and communication personality")
                tone_keywords: list[str] = Field(description="3-5 descriptive words of the brand tone")
                primary_color_hex: str = Field(
                    description="Hex color code for the brand. CRITICAL: If the user explicitly stated, chose, or confirmed colors in chats (e.g. yellow, orange, blue), extract their exact chosen color as primary (e.g. #FACC15 or #F59E0B for yellow). Priority: User's explicit choice > industry default."
                )
                secondary_color_hex: str = Field(
                    description="Secondary hex color code complementing the primary or specified by user, e.g. #1E293B."
                )
                accent_color_hex: str = Field(
                    description="Accent hex color code. CRITICAL: If the user chose an accent or mix color (e.g. strong orange #EA580C or #F97316), extract their chosen accent color."
                )
                palette_colors: list[str] = Field(
                    description="5 cohesive hex colors for the brand palette featuring the user's chosen primary and accent colors."
                )
                audience_summary: str = Field(description="Who the ideal target customer is")
                demographics: str = Field(description="Age, location, profession demographics")
                pain_points: list[str] = Field(description="Customer frustrations and needs")
                strengths: list[str] = Field(description="Top internal strengths and USPs")
                weaknesses: list[str] = Field(description="Real challenges or areas to strengthen")
                opportunities: list[str] = Field(description="Untapped growth opportunities in the market")
                threats: list[str] = Field(description="External competitor threats and risks")
                key_offers: list[str] = Field(description="Top products, packages, or services")
                competitors: list[str] = Field(description="Known direct or indirect competitors mentioned")
                market_gaps: list[str] = Field(description="Gaps in the market this brand solves")

            langchain_llm = getattr(llm, "chat", None) or get_llm(provider=ai_settings.llm_provider).chat
            extractor = langchain_llm.with_structured_output(ExtractedBrandData)

            sys_prompt = (
                "You are the Chief Brand Officer and Brand Intelligence Agent. Extract the definitive Brand Identity from the user's "
                "conversations, search data, and profile facts. Match the language (Arabic or English) "
                "predominantly used in the conversations. Output realistic, accurate brand information.\n\n"
                "CRITICAL COMPETITOR RETENTION RULE:\n"
                "Extract ALL competitor brands and businesses mentioned, researched, or compared in chats and research "
                "(e.g. 'بن النجار', 'عبد المعبود', 'شاهين', 'Indomie Egypt', 'Abu Auf', etc.). "
                "NEVER omit or drop any competitor researched by the user or agents.\n\n"
                "CRITICAL COLOR & USER PREFERENCE RULE:\n"
                "Examine the conversation history carefully. When the user explicitly states, selects, or requests "
                "brand colors or visual identity (for example 'اصفر وبرتقالي', 'yellow and orange', 'أصفر وبرتقالي قوي'), "
                "you MUST STRICTLY honor the user's explicit choice over any industry defaults:\n"
                "- primary_color_hex: MUST be the user's chosen main color (e.g. #FACC15 or #F59E0B for yellow/gold)\n"
                "- accent_color_hex: MUST be the user's chosen secondary/accent color (e.g. #EA580C or #F97316 for strong orange)\n"
                "- palette_colors: 5 cohesive hex codes including the primary, accent, dark neutral, and tints.\n"
                "Do NOT use red just because the brand sells food/noodles when the user explicitly chose yellow and orange!"
            )
            res: ExtractedBrandData = await extractor.ainvoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=f"Analyze this business data and extract the Brand Identity:\n\n{corpus}"),
            ])

            if res.brand_name and len(res.brand_name.strip()) > 1 and "Not specified" not in res.brand_name:
                brand_name = res.brand_name.strip()
            if res.industry:
                industry = res.industry.strip()
            if res.tagline:
                tagline = res.tagline.strip()
            if res.mission:
                mission = res.mission.strip()
            if res.vision:
                vision = res.vision.strip()
            if res.brand_voice:
                brand_voice = res.brand_voice.strip()
            if res.tone_keywords:
                tone_keywords = res.tone_keywords
            if res.strengths:
                strengths = res.strengths
            if res.weaknesses:
                weaknesses = res.weaknesses
            if res.opportunities:
                opportunities = res.opportunities
            if res.threats:
                threats = res.threats
            if res.audience_summary:
                target_summary = res.audience_summary
            if res.demographics:
                demographics = res.demographics
            if res.pain_points:
                pain_points = res.pain_points
            if res.key_offers:
                offers = res.key_offers
            
            # Merge all competitors: existing bp.competitors + competitor_note memory items + LLM output
            all_comps: list[str] = list(bp.competitors or [])
            for m in mem_rows:
                if m.kind == "competitor_note":
                    c_name = (m.metadata_ or {}).get("competitor")
                    if c_name and isinstance(c_name, str):
                        all_comps.append(c_name)
                    elif m.text and "Competitor" in m.text:
                        txt_after = m.text.split(":")[-1].strip()
                        if txt_after:
                            all_comps.append(txt_after)
            if res.competitors:
                all_comps.extend(res.competitors)

            deduped_comps: list[str] = []
            seen_set = set()
            for c in all_comps:
                cleaned_c = _clean_competitor_name(c) if isinstance(c, str) else ""
                if cleaned_c and len(cleaned_c) >= 2 and cleaned_c.lower() not in seen_set:
                    deduped_comps.append(cleaned_c)
                    seen_set.add(cleaned_c.lower())
            competitors = deduped_comps

            if res.market_gaps:
                market_gaps = res.market_gaps

            def _clean_hex(c: str | None, default: str) -> str:
                if not c:
                    return default
                c = c.strip()
                if not c.startswith("#") and len(c) in (6, 8):
                    c = f"#{c}"
                return c if c.startswith("#") else default

            p_col = _clean_hex(res.primary_color_hex, extracted_palette.primary)
            s_col = _clean_hex(res.secondary_color_hex, extracted_palette.secondary)
            a_col = _clean_hex(res.accent_color_hex, extracted_palette.accent)
            clean_palette = [_clean_hex(c, "") for c in res.palette_colors]
            clean_palette = [c for c in clean_palette if c.startswith("#")]
            if len(clean_palette) < 3:
                clean_palette = [p_col, a_col, s_col, "#FFF7ED", "#0F172A"]

            extracted_palette = ColorPalette(
                primary=p_col,
                secondary=s_col,
                accent=a_col,
                background="#F8FAFC",
                text="#0F172A",
                palette=clean_palette,
            )
        except Exception as exc:
            logger.warning("llm_brand_identity_extraction_fallback", extra={"error": str(exc)})

        # Update Business & BrandProfile rows
        if brand_name and brand_name != biz.name:
            biz.name = brand_name
        if industry and (not biz.industry or biz.industry == "None"):
            biz.industry = industry

        # Fill the blank fields; never overwrite what a human saved in Settings.
        #
        # This pass reads chats and rewrites the brand profile. It used to assign
        # every field unconditionally, which meant a user who typed their brand
        # voice into Settings could have it replaced minutes later by an LLM
        # inference from a chat. ``MEMORY_IDENTITY_OVERWRITE_PROFILE=true``
        # restores the old behaviour for teams that want the agent to own the
        # profile outright.
        from sqlalchemy.orm.attributes import flag_modified

        overwrite = _identity_may_overwrite_profile()

        def _assign(field_name: str, value: Any) -> None:
            current = getattr(bp, field_name, None)
            is_empty = current in (None, "", [], {}) or (
                isinstance(current, str) and not current.strip()
            )
            if overwrite or is_empty:
                setattr(bp, field_name, value)
                if field_name in {"competitors", "key_offers", "goals"}:
                    flag_modified(bp, field_name)

        _assign("brand_voice", brand_voice)
        _assign("target_audience", target_summary)
        _assign("key_offers", offers)
        _assign("competitors", competitors)
        if not bp.goals:
            bp.goals = [
                f"Establish market leadership in {industry}",
                "Achieve high brand recall and customer loyalty",
            ]
        if not bp.visual_style:
            bp.visual_style = json.dumps(extracted_palette.model_dump(), ensure_ascii=False)

        now_dt = _now()

        response = BrandIdentityResponse(
            business_id=biz.id,
            name=biz.name,
            industry=biz.industry,
            website_url=biz.website_url,
            country=biz.country,
            language=biz.language or "en",
            currency=biz.currency or "EGP",
            monthly_budget=biz.monthly_budget,
            colors=extracted_palette,
            essence=IdentityEssence(
                tagline=tagline,
                mission=mission,
                vision=vision,
                brand_voice=brand_voice,
                tone_keywords=tone_keywords,
            ),
            audience=TargetAudienceProfile(
                summary=target_summary,
                demographics=demographics,
                pain_points=pain_points,
                core_segments=["Core Enthusiasts", "Quality-Seeking Buyers"],
            ),
            swot=SWOTAnalysis(
                strengths=strengths,
                weaknesses=weaknesses,
                opportunities=opportunities,
                threats=threats,
            ),
            key_offers=offers,
            competitors=competitors,
            market_gaps=market_gaps,
            last_synced_at=now_dt,
            sources_count=len(valid_chat_snippets) + len(memory_snippets) + 1,
        )

        # Save to BrandMemoryItem for persistence and vector context
        mem_stmt = (
            select(BrandMemoryItem)
            .where(BrandMemoryItem.business_id == business_id, BrandMemoryItem.kind == "brand_identity")
            .limit(1)
        )
        existing_mem = (await db.execute(mem_stmt)).scalars().first()
        if not existing_mem:
            _identity_text = f"Brand Identity: {response.name} ({response.industry}). Tagline: {tagline}. Voice: {brand_voice}"
            existing_mem = BrandMemoryItem(
                id=f"mem_identity_{uuid.uuid4().hex[:16]}",
                business_id=business_id,
                kind="brand_identity",
                text=_identity_text,
                metadata_={"brand_identity": response.model_dump(mode="json")},
                created_at=now_dt,
                **memory_defaults(
                    "brand_identity",
                    business_id=business_id,
                    text=_identity_text,
                    source="identity_service",
                ),
            )
            db.add(existing_mem)
        else:
            existing_mem.text = f"Brand Identity: {response.name} ({response.industry}). Tagline: {tagline}. Voice: {brand_voice}"
            existing_mem.metadata_ = {"brand_identity": response.model_dump(mode="json")}

        await db.commit()
        await db.refresh(biz)
        return response


# Dedicated Singleton Brand Intelligence Agent
brand_intelligence_agent = BrandIntelligenceAgent()


async def sync_brand_identity_from_chats(
    db: AsyncSession,
    business_id: str,
    llm: Any = None,
) -> BrandIdentityResponse:
    return await brand_intelligence_agent.sync_brand_identity(db, business_id, llm=llm)


async def update_brand_identity(
    db: AsyncSession,
    business_id: str,
    update: BrandIdentityUpdateRequest,
) -> BrandIdentityResponse:
    """Manually update specific fields of the Brand Identity."""
    current = await get_brand_identity(db, business_id)
    biz = await db.get(Business, business_id)
    if not biz:
        raise ValueError(f"Business not found: {business_id}")

    profile_stmt = select(BrandProfileRow).where(BrandProfileRow.business_id == business_id)
    bp = (await db.execute(profile_stmt)).scalars().first()
    if not bp:
        bp = BrandProfileRow(id=f"bp_{business_id}", business_id=business_id)
        db.add(bp)

    if update.name is not None:
        biz.name = update.name
        current.name = update.name
    if update.industry is not None:
        biz.industry = update.industry
        current.industry = update.industry
    if update.website_url is not None:
        biz.website_url = update.website_url
        current.website_url = update.website_url
    if update.currency is not None:
        biz.currency = update.currency
        current.currency = update.currency
    if update.monthly_budget is not None:
        biz.monthly_budget = update.monthly_budget
        current.monthly_budget = update.monthly_budget
    if update.colors is not None:
        current.colors = update.colors
        bp.visual_style = json.dumps(update.colors.model_dump(), ensure_ascii=False)
    if update.essence is not None:
        current.essence = update.essence
        if update.essence.brand_voice:
            bp.brand_voice = update.essence.brand_voice
    if update.audience is not None:
        current.audience = update.audience
        if update.audience.summary:
            bp.target_audience = update.audience.summary
    if update.swot is not None:
        current.swot = update.swot
    if update.key_offers is not None:
        current.key_offers = update.key_offers
        bp.key_offers = update.key_offers
    if update.competitors is not None:
        current.competitors = update.competitors
        bp.competitors = update.competitors
    if update.market_gaps is not None:
        current.market_gaps = update.market_gaps

    current.last_synced_at = _now()

    # Persist back to memory
    mem_stmt = (
        select(BrandMemoryItem)
        .where(BrandMemoryItem.business_id == business_id, BrandMemoryItem.kind == "brand_identity")
        .limit(1)
    )
    existing_mem = (await db.execute(mem_stmt)).scalars().first()
    if not existing_mem:
        _identity_text = f"Brand Identity: {current.name} ({current.industry}). Voice: {current.essence.brand_voice}"
        existing_mem = BrandMemoryItem(
            id=f"mem_identity_{uuid.uuid4().hex[:16]}",
            business_id=business_id,
            kind="brand_identity",
            text=_identity_text,
            metadata_={"brand_identity": current.model_dump(mode="json")},
            created_at=_now(),
            **memory_defaults(
                "brand_identity",
                business_id=business_id,
                text=_identity_text,
                source="identity_service",
            ),
        )
        db.add(existing_mem)
    else:
        existing_mem.metadata_ = {"brand_identity": current.model_dump(mode="json")}

    await db.commit()
    await db.refresh(biz)
    return current
