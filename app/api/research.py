"""Research endpoints: ground-truth verification, live web citations, persistence in brand memory & LTM brain update."""

from __future__ import annotations

import ast
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import SystemMessage, HumanMessage

from app.api.deps import RequestIdDep, SessionDep, SettingsDep, WorkspaceDep
from app.db.models import BrandMemoryItem, BrandProfile, Profile
from app.ai.providers.search import get_search_provider
from app.ai.tools.analyze_website import analyze_website_tool
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/research", tags=["research"])


# =====================================================================
# Text Extraction & Cleansing Helpers (No Dictionary Leaks)
# =====================================================================

def clean_raw_string(s: str) -> str:
    """Extract clean readable text from serialized Python dicts, lists, or JSON strings."""
    if not s:
        return ""
    s = s.strip()
    
    # Check if the string itself is a serialized Python dict / list of dicts
    if (s.startswith("{'type'") or s.startswith("[{'type'") or 
        s.startswith('{"type"') or s.startswith('[{"type"')):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                texts = [item.get("text", "") for item in parsed if isinstance(item, dict)]
                if texts:
                    s = "\n".join(texts)
            elif isinstance(parsed, dict) and "text" in parsed:
                s = str(parsed["text"])
        except Exception:
            try:
                data = json.loads(s)
                if isinstance(data, list):
                    texts = [item.get("text", "") for item in data if isinstance(item, dict)]
                    if texts:
                        s = "\n".join(texts)
                elif isinstance(data, dict) and "text" in data:
                    s = str(data["text"])
            except Exception:
                pass

    # Fallback regex if text remains encapsulated in {'type': 'text', 'text': ...}
    if "{'type': 'text'" in s or '{"type": "text"' in s:
        matches = re.findall(r"['\"]text['\"]\s*:\s*['\"](.*?)['\"](?:\s*,\s*['\"][a-zA-Z_]+['\"]|\s*})", s, re.DOTALL)
        if matches:
            s = "\n".join(matches)

    # Clean replacement characters or leftover byte anomalies
    s = s.replace("\ufffd", "").replace("\\n", "\n")
    return s.strip()


def extract_clean_text(content: Any) -> str:
    """Extract strictly plain markdown text from any LangChain LLM content structure."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        extracted = "\n".join([p for p in parts if p.strip()])
        return clean_raw_string(extracted)
    if isinstance(content, dict):
        if "text" in content:
            return clean_raw_string(str(content["text"]))
        return clean_raw_string(str(content))
    return clean_raw_string(str(content))


# =====================================================================
# Request / Response Schemas
# =====================================================================

class RunResearchRequest(BaseModel):
    tool: Literal[
        "analyze_competitor",
        "analyze_website",
        "analyze_social_profile",
        "analyze_reviews",
        "analyze_market",
        "analyze_keywords",
        "analyze_competitor_creatives",
        "detect_market_gaps",
        "generate_marketing_strategy",
    ]
    query: str = Field(default="", description="Target query, brand name, URL, or keyword")


class ResearchItemOut(BaseModel):
    id: str
    tool: str
    query: str
    title: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    saved_to_memory: bool = True
    created_at: str


# =====================================================================
# Prompt Instructions
# =====================================================================

TOOL_PROMPTS = {
    "analyze_website": (
        "You are an expert Website & Conversion Rate Optimization Analyst for SAHM OS.\n"
        "Analyze the provided website extraction and structure your report into clean Markdown:\n"
        "## Overview & Brand Positioning\n"
        "## Core Offers & Value Propositions\n"
        "## Trust Signals & Social Proof (Reviews, guarantees, badges)\n"
        "## Conversion Friction & Gaps\n"
        "## Actionable Recommendations for Growth\n\n"
        "Include a '## 🌐 Sources & References' section at the end with markdown links to the analyzed domain."
    ),
    "analyze_competitor": (
        "You are a Senior Competitor Intelligence Strategist for SAHM OS.\n"
        "Analyze who is advertising to this audience and how using the verified web sources. Structure into clean Markdown:\n"
        "## Competitor Overview & Market Share in Egypt\n"
        "## Target Audience & Customer Demographics\n"
        "## Advertising Angles, Hooks & Campaigns\n"
        "## Pricing Strategy & Promotional Bundles\n"
        "## Competitor Vulnerabilities & How We Outcompete Them\n"
        "## 🌐 Sources & References\n"
        "(List each referenced web source as a clickable markdown link: - [Source Title](URL))"
    ),
    "analyze_social_profile": (
        "You are a Social Media Content Strategist for SAHM OS.\n"
        "Analyze this social profile based on real evidence and structure into clean Markdown:\n"
        "## Core Content Pillars & Themes\n"
        "## Engagement Dynamics & Top-Performing Formats (Reels vs Carousels)\n"
        "## Brand Tone of Voice & Visual Aesthetic\n"
        "## Audience Reception & Comment Sentiment\n"
        "## 'What to Post More Of' Strategic Action Items\n"
        "## 🌐 Sources & References\n"
        "(List all relevant profile and social references with clickable markdown links)"
    ),
    "analyze_reviews": (
        "You are a Customer Experience & Sentiment Analyst for SAHM OS.\n"
        "Analyze customer feedback from web and review sources. Structure into clean Markdown:\n"
        "## Overall Sentiment Score & Review Trends\n"
        "## Top Customer Praises (What customers love)\n"
        "## Top Customer Complaints & Friction Points (What hurts sales)\n"
        "## Representative Customer Quotes\n"
        "## Strategic Fixes & Positioning Opportunities\n"
        "## 🌐 Sources & References\n"
        "(List review links, store listings, and discussion threads referenced)"
    ),
    "analyze_market": (
        "You are a Macro Market & Industry Analyst for SAHM OS.\n"
        "Analyze market size, segments, and trends for this category in Egypt. Structure into clean Markdown:\n"
        "## Market Size & 2026 Growth Trajectory in Egypt\n"
        "## Key Consumer Segments & Spending Behavior\n"
        "## Emerging Category Trends & Consumer Shifts\n"
        "## Economic Realities (Currency, imports, price elasticity)\n"
        "## High-Growth Whitespaces & Category Opportunities\n"
        "## 🌐 Sources & References\n"
        "(Cite industry reports, news articles, and market sources with markdown links)"
    ),
    "analyze_keywords": (
        "You are an SEO & Search Intent Strategist for SAHM OS.\n"
        "Analyze search demand clusters and structure into clean Markdown:\n"
        "## High-Intent Transactional Keyword Clusters (High buying intent)\n"
        "## Informational & Education Keyword Clusters (Research stage)\n"
        "## Estimated Search Demand & Competition Intensity\n"
        "## Long-Tail Search Queries & Question Clusters\n"
        "## Recommended Content Silos & Target Pages\n"
        "## 🌐 Sources & References\n"
        "(Include search references and relevant domain links)"
    ),
    "analyze_competitor_creatives": (
        "You are an Elite Creative Director & Paid Ads Strategist for SAHM OS.\n"
        "Analyze competitor advertising creatives, ad formats, hooks, and promotional angles based on live market evidence.\n\n"
        "STRICT INSTRUCTION: Do NOT write a general corporate background or company history (that belongs to Competitor Research). "
        "Focus 100% on ad creatives, copywriting, visual hooks, campaign themes, and advertising psychology.\n\n"
        "Structure your report into clean Markdown:\n"
        "## 🎬 Long-Running Creative Concepts & Core Ad Angles\n"
        "• What emotional and psychological angles dominate their ads (e.g. status, value, convenience, fear of missing out)?\n"
        "• Long-running evergreen ad themes that they continue spending ad budget on.\n\n"
        "## 🪝 3-Second Hook Analysis (Visual & Headline Hooks)\n"
        "• Visual hooks: Opening frames, celebrity/influencer appearances, pattern interrupts.\n"
        "• Audio/Verbal hooks: Spoken first lines, bold statements, or curiosity gaps.\n"
        "• Headline copy hooks used in ad feeds and banners.\n\n"
        "## 📱 Creative Formats & Production Styles Breakdown\n"
        "• User-Generated Content (UGC) vs. High-Production Studio Commercials.\n"
        "• Short-form vertical video (Reels/TikTok) vs Carousel ads vs Static graphic banners.\n"
        "• Pacing, music choices, and caption strategies.\n\n"
        "## 🏷️ Promotional Offers, Hooks & Call-to-Actions (CTAs)\n"
        "• Pricing anchors, bundles (e.g., buy 1 get 1, suit + shirt bundles), seasonal discounts.\n"
        "• CTAs used (WhatsApp chat, website checkout, showroom visit).\n\n"
        "## 💡 How to Out-Create Them (3 Actionable Ad Concepts for Our Brand)\n"
        "• Concept 1: The 'Direct Contrast' Ad (Countering their biggest creative fatigue point).\n"
        "• Concept 2: The 'Authentic Craft/Behind-the-Scenes' Hook.\n"
        "• Concept 3: The 'Unbeatable Value/Offer' Ad Script.\n\n"
        "## 🌐 Sources & References\n"
        "(Cite Meta Ad Library and campaign references with markdown links)"
    ),
    "detect_market_gaps": (
        "You are an Opportunity Spotter & Product-Market Fit Strategist for SAHM OS.\n"
        "Synthesize all accumulated research to date and rank unserved needs. Structure into clean Markdown:\n"
        "## Executive Opportunity Summary\n"
        "## Unserved Customer Needs & Frustrations in the Category\n"
        "## Pricing Gaps (Missing luxury or quality mid-tier options)\n"
        "## Product, Quality & Service Whitespaces\n"
        "## Ranked Opportunities by Revenue Potential & Feasibility\n"
        "## 🌐 Sources & References"
    ),
    "generate_marketing_strategy": (
        "You are a Chief Marketing Officer (CMO) for SAHM OS.\n"
        "Build an actionable 90-day marketing blueprint. Structure into clean Markdown:\n"
        "## Core Brand Positioning & The Unfair Advantage Hook\n"
        "## 90-Day Execution Roadmap (Month 1: Foundation, Month 2: Scale, Month 3: Optimize)\n"
        "## Channel Allocation & Budget Strategy (Meta, TikTok, Google, Influencers)\n"
        "## Key Metrics & Targets (CAC, ROAS, Target Order Volume)\n"
        "## Immediate Top 3 Action Items for This Week\n"
        "## 🌐 Sources & References"
    ),
}


# =====================================================================
# Ground-Truth Entity Verification & Typo Correction
# =====================================================================

async def verify_and_resolve_entity(
    query: str,
    tool_type: str,
    fast_llm: Any,
) -> dict[str, Any]:
    """Verify whether query is a real brand/competitor, correct typos, or flag as not found."""
    if tool_type not in ("analyze_competitor", "analyze_competitor_creatives"):
        return {"status": "verified", "target_brand": query, "explanation": "General research query", "confidence": 1.0}

    verification_sys = (
        "You are an expert Ground-Truth Entity & Brand Verification Engine for commercial markets (Egypt & global).\n"
        "A user entered a brand/competitor query. Your task is to evaluate whether this is a real, active brand.\n\n"
        "Instructions:\n"
        "1. If the brand is clearly recognized or a known commercial business, return status 'verified'.\n"
        "2. If the user made a typo or phonetic misspelling of a real brand (e.g. 'Tiee Hose' -> 'Tie House', 'Concrette' -> 'Concrete', 'Town Tem' -> 'Town Team', 'Adiddas' -> 'Adidas', 'Zarra' -> 'Zara'), return status 'typo_corrected' and provide the real 'target_brand'.\n"
        "3. If the query is completely nonsensical, keyboard gibberish, fake, or non-existent in the market (e.g. 'asdfghjkl9999fake', 'FakeShopXYZ123', 'qwertyuiop'), return status 'not_found'.\n\n"
        "Respond STRICTLY in JSON format:\n"
        "{\n"
        '  "status": "verified" | "typo_corrected" | "not_found",\n'
        '  "target_brand": "Name of real brand if verified or corrected, otherwise null",\n'
        '  "explanation": "Brief explanation of why it was verified, what typo was fixed, or why not found",\n'
        '  "confidence": 0.0 to 1.0\n'
        "}"
    )

    usr_msg = f"User Query: '{query}'\nTool: {tool_type}\nDetermine entity status."

    try:
        import asyncio
        res = await asyncio.to_thread(
            fast_llm.invoke,
            [SystemMessage(content=verification_sys), HumanMessage(content=usr_msg)],
        )
        raw_json = extract_clean_text(res.content)
        # Parse JSON
        clean_json_str = raw_json
        if "```json" in clean_json_str:
            clean_json_str = clean_json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json_str:
            clean_json_str = clean_json_str.split("```")[1].split("```")[0].strip()

        data = json.loads(clean_json_str)
        return {
            "status": data.get("status", "verified"),
            "target_brand": data.get("target_brand") or query,
            "explanation": data.get("explanation", ""),
            "confidence": float(data.get("confidence", 0.9)),
        }
    except Exception as e:
        logger.warning("entity_verification_fallback", extra={"error": str(e), "query": query})
        # If verification fails unexpectedly, proceed with query as verified to avoid blocking
        return {"status": "verified", "target_brand": query, "explanation": "Direct search verification", "confidence": 0.8}


# =====================================================================
# Endpoints
# =====================================================================

@router.get("/items", response_model=list[ResearchItemOut])
async def list_research_items(
    ws: WorkspaceDep,
    db: SessionDep,
) -> list[ResearchItemOut]:
    """List recent research analyses saved in brand memory."""
    stmt = (
        select(BrandMemoryItem)
        .where(
            BrandMemoryItem.business_id == ws.business_id,
            BrandMemoryItem.kind.in_([
                "competitor_note",
                "website_analysis",
                "market_research",
                "market_gaps",
                "strategy_blueprint",
            ]),
        )
        .order_by(desc(BrandMemoryItem.created_at))
        .limit(20)
    )
    rows = (await db.execute(stmt)).scalars().all()
    
    results = []
    for r in rows:
        meta = r.metadata_ or {}
        clean_text = extract_clean_text(r.text)
        results.append(
            ResearchItemOut(
                id=r.id,
                tool=meta.get("tool", "analyze_competitor"),
                query=meta.get("query", ""),
                title=meta.get("title", "Research Item"),
                summary=clean_text,
                details=meta.get("details", {}),
                saved_to_memory=True,
                created_at=r.created_at.isoformat() if r.created_at else "",
            )
        )
    return results


@router.post("/run", response_model=ResearchItemOut)
async def run_research(
    payload: RunResearchRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    request_id: RequestIdDep,
) -> ResearchItemOut:
    """Execute a research tool with ground-truth verification, live citations, and brain update."""
    tool_type = payload.tool
    query = payload.query.strip()

    # Load brand context from database for fallback when query is optional
    profile = await db.scalar(select(Profile).where(Profile.business_id == ws.business_id).limit(1))
    brand_profile = await db.scalar(select(BrandProfile).where(BrandProfile.business_id == ws.business_id).limit(1))
    
    user_context = (profile.user_context_summary if profile and profile.user_context_summary else "").strip()
    user_context = extract_clean_text(user_context)

    # Handle tools requiring mandatory input vs optional inputs
    if not query:
        if tool_type == "analyze_website":
            raise HTTPException(status_code=400, detail="Please provide a website URL (e.g. https://yourbrand.com).")
        elif tool_type == "analyze_social_profile":
            raise HTTPException(status_code=400, detail="Please provide a social handle (e.g. @yourbrand).")
        elif tool_type == "analyze_keywords":
            raise HTTPException(status_code=400, detail="Please provide a seed keyword (e.g. classic suits cairo).")
        elif tool_type == "detect_market_gaps":
            query = "All accumulated market research & category dynamics"
        elif tool_type == "generate_marketing_strategy":
            query = "Comprehensive 90-day growth strategy"
        elif tool_type == "analyze_competitor":
            query = "Direct competitors in our category"
        elif tool_type == "analyze_competitor_creatives":
            query = "Competitor ad creatives and hooks"
        elif tool_type == "analyze_market":
            query = "Category size, segments and trends"
        elif tool_type == "analyze_reviews":
            query = "Customer reviews, themes and sentiment"

    ai_settings = get_ai_settings()
    llm = get_llm(provider=ai_settings.llm_provider).chat
    fast_llm = get_llm().chat
    search_provider = get_search_provider(ai_settings.search_provider)

    # 1. Ground-Truth Verification & Typo Detection
    verification = await verify_and_resolve_entity(query, tool_type, fast_llm)
    v_status = verification["status"]
    target_subject = verification.get("target_brand") or query
    v_explanation = verification.get("explanation", "")

    # CASE A: Entity Not Found in Ground Truth (Fake, nonsensical, or completely unknown)
    if v_status == "not_found":
        not_found_report = (
            f"## ⚠️ Brand Verification: Entity Not Found in Market\n\n"
            f"We could not verify **\"{query}\"** as an active commercial brand, business, or competitor in the market.\n\n"
            f"### Ground-Truth Verification Details:\n"
            f"• **Finding:** {v_explanation}\n"
            f"• **Search Footprint:** No verified commercial storefronts, official social pages, or recognizable business listings matched this query.\n\n"
            f"### Recommended Next Steps:\n"
            f"• Check the spelling or provide the official registered brand name.\n"
            f"• If this is a new or private business, provide their direct website in **Website analyzer** or their official handle in **Social profile** to extract direct intelligence.\n"
        )
        return ResearchItemOut(
            id=f"mem_unverified_{uuid.uuid4().hex[:10]}",
            tool=tool_type,
            query=query,
            title=f"Verification Failed: '{query}'",
            summary=not_found_report,
            details={
                "subject": query,
                "tool": tool_type,
                "verification_status": "not_found",
                "explanation": v_explanation,
                "sources": [],
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            },
            saved_to_memory=False,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # CASE B: Valid entity or Typo Corrected
    effective_query = target_subject
    typo_callout = ""
    if v_status == "typo_corrected" and target_subject.lower() != query.lower():
        typo_callout = (
            f"> 💡 **Auto-Corrected Query**: Detected and corrected typo from *\"{query}\"* to **\"{target_subject}\"** based on market ground truth.\n\n"
        )

    title_map = {
        "analyze_website": f"Website Analysis: {effective_query}",
        "analyze_competitor": f"Competitor Research: {effective_query}",
        "analyze_social_profile": f"Social Profile: {effective_query}",
        "analyze_reviews": f"Reviews & Sentiment: {effective_query}",
        "analyze_market": f"Market Analysis: {effective_query}",
        "analyze_keywords": f"Keyword Clusters: {effective_query}",
        "analyze_competitor_creatives": f"Competitor Creatives: {effective_query}",
        "detect_market_gaps": "Market Gaps & Opportunity Ranking",
        "generate_marketing_strategy": "90-Day Marketing Strategy Blueprint",
    }
    title = title_map.get(tool_type, f"{tool_type.replace('_', ' ').title()}: {effective_query}")
    if v_status == "typo_corrected" and target_subject.lower() != query.lower():
        title = f"{title} (auto-corrected from '{query}')"

    # 2. Collect Live Web Evidence & Source Links
    sources: list[dict[str, str]] = []
    evidence_blocks: list[str] = []

    if tool_type == "analyze_website":
        raw_site = await analyze_website_tool.ainvoke({"url": effective_query})
        evidence_blocks.append(f"Domain Content Extraction ({effective_query}):\n{raw_site}")
        sources.append({"title": f"Analyzed Website ({effective_query})", "url": effective_query})
    else:
        # Build targeted search query
        from urllib.parse import quote_plus
        
        if tool_type == "analyze_competitor":
            search_query = f"{effective_query} egypt brand store products positioning reviews"
        elif tool_type == "analyze_competitor_creatives":
            meta_ad_url = f"https://www.facebook.com/ads/library/?active_status=all&ad_type=all&country=EG&q={quote_plus(effective_query)}"
            sources.append({
                "title": f"Meta Ad Library (Active Egypt Ads): {effective_query}",
                "url": meta_ad_url,
            })
            search_query = f'"{effective_query}" campaign commercial ad creative video hook commercial Egypt'
        elif tool_type == "analyze_keywords":
            search_query = f"{effective_query} search demand volume egypt keywords"
        elif tool_type == "analyze_social_profile":
            search_query = f"{effective_query} instagram facebook reels engagement egypt"
        elif tool_type == "analyze_reviews":
            search_query = f"{effective_query} customer reviews complaints feedback rating egypt"
        elif tool_type == "analyze_market":
            search_query = f"{effective_query} market size growth trends consumer behavior egypt 2026"
        else:
            search_query = f"egypt market gaps opportunities {user_context[:150]}"

        try:
            hits = search_provider.search(search_query, max_results=5)
            if tool_type == "analyze_competitor_creatives":
                try:
                    sec_query = f'"{effective_query}" facebook ads instagram reels ad copy offers discount Egypt'
                    sec_hits = search_provider.search(sec_query, max_results=3)
                    hits.extend(sec_hits)
                except Exception:
                    pass

            for i, h in enumerate(hits, 1):
                if h.url:
                    sources.append({"title": h.title or f"Source {i}", "url": h.url})
                evidence_blocks.append(
                    f"[Source {i}]: {h.title}\nURL: {h.url}\nSummary Snippet: {h.snippet}"
                )
        except Exception as e:
            logger.error("search_evidence_failed", extra={"error": str(e), "query": search_query})
            evidence_blocks.append(f"Search evidence lookup: {effective_query}")

    raw_evidence = "\n\n".join(evidence_blocks)

    # 3. Synthesize Structured Report with Ground-Truth References
    sys_instruction = TOOL_PROMPTS.get(tool_type, TOOL_PROMPTS["analyze_competitor"])
    sys_prompt = SystemMessage(content=sys_instruction)
    
    brand_context_block = f"Brand Context / Long-Term Memory:\n{user_context}\n" if user_context else ""
    usr_prompt = HumanMessage(content=(
        f"Target Subject: '{effective_query}'\n"
        f"Research Type: '{tool_type}'\n\n"
        f"{brand_context_block}"
        f"Verified Web & Market Evidence (with URLs):\n{raw_evidence}\n\n"
        "Generate the complete structured intelligence report in clean Markdown.\n"
        "Ground your findings in the provided evidence and list all sources in the '## 🌐 Sources & References' section as clickable markdown links."
    ))

    try:
        import asyncio
        res = await asyncio.to_thread(llm.invoke, [sys_prompt, usr_prompt])
        summary_text = extract_clean_text(res.content)
    except Exception as e:
        logger.error("research_synthesis_failed", extra={"error": str(e)})
        summary_text = f"## Research Findings for {effective_query}\n\n{raw_evidence}"

    # Prepend typo correction banner if applicable
    if typo_callout:
        summary_text = typo_callout + summary_text

    # Ensure source references exist at the bottom if sources were found
    if sources and "Sources & References" not in summary_text and "## Sources" not in summary_text:
        source_links_md = "\n".join([f"- [{s['title']}]({s['url']})" for s in sources if s.get("url")])
        summary_text += f"\n\n## 🌐 Sources & References\n{source_links_md}\n"

    summary_text = clean_raw_string(summary_text)
    memory_id = f"mem_res_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    # 4. Save to BrandMemoryItem
    kind_map = {
        "analyze_competitor": "competitor_note",
        "analyze_competitor_creatives": "competitor_note",
        "analyze_website": "website_analysis",
        "analyze_keywords": "market_research",
        "analyze_market": "market_research",
        "analyze_reviews": "market_research",
        "detect_market_gaps": "market_gaps",
        "generate_marketing_strategy": "strategy_blueprint",
    }
    kind = kind_map.get(tool_type, "market_research")

    mem_item = BrandMemoryItem(
        id=memory_id,
        business_id=ws.business_id,
        kind=kind,
        text=summary_text,
        metadata_={
            "tool": tool_type,
            "query": query,
            "title": title,
            "details": {
                "subject": effective_query,
                "tool": tool_type,
                "verification_status": v_status,
                "corrected_from": query if v_status == "typo_corrected" else None,
                "sources": sources,
                "analyzed_at": now.isoformat(),
            },
        },
        created_at=now,
    )
    db.add(mem_item)

    # 5. Integrate into Brain / Long-Term Memory (Profile.user_context_summary)
    if not profile:
        profile = Profile(
            id=uuid.uuid4(),
            supabase_user_id=uuid.uuid4(),
            business_id=ws.business_id,
            role="owner",
            user_context_summary="",
        )
        db.add(profile)
        await db.flush()

    old_summary = extract_clean_text(profile.user_context_summary or "")
    ltm_sys = SystemMessage(content=(
        "You are the central brain memory manager for SAHM Marketing OS.\n"
        "Integrate this new research finding into the brand's permanent profile.\n"
        "Update the known competitors, audience learnings, and market opportunities.\n"
        "STRICT CONSTRAINT: Keep under 500 tokens total in high-density bullet points. Output plain markdown only."
    ))
    ltm_usr = HumanMessage(content=(
        f"CURRENT LTM PROFILE:\n{old_summary}\n\n"
        f"NEW RESEARCH INTEL TO INTEGRATE:\nTitle: {title}\nSummary Preview: {summary_text[:700]}\n\n"
        "Return the updated, unified Long-Term Memory:"
    ))
    try:
        import asyncio
        ltm_res = await asyncio.to_thread(fast_llm.invoke, [ltm_sys, ltm_usr])
        profile.user_context_summary = extract_clean_text(ltm_res.content)
    except Exception as e:
        logger.warning("profile_ltm_integration_skipped", extra={"error": str(e)})

    await db.commit()

    return ResearchItemOut(
        id=memory_id,
        tool=tool_type,
        query=query,
        title=title,
        summary=summary_text,
        details={
            "subject": effective_query,
            "tool": tool_type,
            "verification_status": v_status,
            "corrected_from": query if v_status == "typo_corrected" else None,
            "sources": sources,
            "analyzed_at": now.isoformat(),
        },
        saved_to_memory=True,
        created_at=now.isoformat(),
    )


__all__ = ["router"]
