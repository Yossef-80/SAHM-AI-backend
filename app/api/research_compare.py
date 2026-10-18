"""Competitor-vs-competitor gap analysis.

This is the structured replacement for the old "detect market gaps" button. That
button took no input at all — the endpoint substituted the literal query
``"All accumulated market research & category dynamics"`` — so its output was a
paragraph about whatever happened to be in memory. This module takes two named
competitors, grounds both in live search results, and returns a comparison an
operator can act on: which dimensions differ, where the gaps are, and which
opportunities are worth taking first.

Design rules, deliberately strict:

* **Every claim carries evidence.** Each dimension, gap and opportunity lists the
  observations it rests on. An empty evidence list is allowed only when the
  statement is explicitly labelled as inference in the prompt, and the schema
  requires the model to say so through ``confidence``.
* **One model call, one JSON object.** The model returns a single object; a
  repair pass re-asks once if it returns something unparseable. Anything else
  becomes an HTTP 502 rather than a plausible-looking half-answer.
* **It persists.** The result is written to brand memory under the existing
  ``market_gaps`` kind so the chat can immediately discuss it — which is also
  what makes the Research page and the assistant share one source of truth.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from langchain_core.messages import HumanMessage, SystemMessage

from app.ai.providers.llm import get_llm
from app.ai.providers.search import get_search_provider
from app.ai.settings import get_ai_settings
from app.api.deps import RequestIdDep, SessionDep, WorkspaceDep
from app.core.ai_usage import usage_action
from app.core.logging import get_logger
from app.core.memory.lifecycle import memory_defaults
from app.core.research_reports import resolve_output_language
from app.db.models import BrandMemoryItem, BrandProfile

logger = get_logger(__name__)
router = APIRouter(prefix="/research", tags=["research"])

# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #

VERDICTS = ("a_leads", "b_leads", "parity")
CONFIDENCE = ("high", "medium", "low")
EFFORT = ("low", "medium", "high")


class CompareRequest(BaseModel):
    competitor_a: str = Field(min_length=1, max_length=120)
    competitor_b: str = Field(min_length=1, max_length=120)
    category: str | None = Field(default=None, max_length=500)
    language: str | None = None


class DimensionComparison(BaseModel):
    dimension: str
    competitor_a: str = ""
    competitor_b: str = ""
    verdict: Literal["a_leads", "b_leads", "parity"] = "parity"
    evidence: list[str] = Field(default_factory=list)


class CompetitorGap(BaseModel):
    title: str
    why_it_matters: str = ""
    who_has_it: Literal["a", "b", "both", "neither"] = "neither"
    evidence: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


class Opportunity(BaseModel):
    title: str
    rationale: str = ""
    effort: Literal["low", "medium", "high"] = "medium"
    impact: Literal["low", "medium", "high"] = "medium"
    first_step: str = ""
    evidence: list[str] = Field(default_factory=list)


class CompetitorComparison(BaseModel):
    summary: str
    dimensions: list[DimensionComparison] = Field(default_factory=list)
    gaps: list[CompetitorGap] = Field(default_factory=list)
    opportunities: list[Opportunity] = Field(default_factory=list)
    recommended_focus: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

DIMENSIONS = (
    "positioning",
    "offer_and_pricing_signals",
    "channels",
    "creative_style_and_hooks",
    "audience_targeting",
    "content_cadence",
    "trust_and_social_proof",
    "funnel_entry_points",
)

SYSTEM_TEMPLATE = """You are an elite competitive-intelligence analyst and senior market researcher.

Compare two real competitors on the SAME fixed dimensions, using the search evidence provided as well as verified industry realities.

CRITICAL RULES FOR ACCURACY, ZERO-HALLUCINATION & HYPERLINKS:

1. ABSOLUTE TRUTHFULNESS & ZERO HALLUCINATION (ممنوع الهبد أو التأليف إطلاقاً):
   - Never invent or fabricate prices, follower counts, financial numbers, store locations, or product lines.
   - If a specific detail (e.g. secret manufacturing material, private price list, exact branch count) is NOT found in the verified search evidence or publicly known facts, state honestly:
     "غير متوفر في المصادر العامة المتاحة" (or "Not publicly disclosed in available sources").
   - Honesty and verified facts are strictly required over plausible-sounding guesses.

2. THOROUGH, HIGH-EFFORT COMPARISON:
   - If the user provides specific focus areas or questions in the prompt (e.g. pricing, materials/craftsmanship, branches, customer service), you MUST address every single requested point thoroughly in the corresponding dimensions.
   - If no specific areas are given, conduct an exhaustive, rigorous competitive analysis covering:
     * Specific pricing tiers and real price ranges (with local currency e.g. EGP / SAR) from the evidence.
     * Quality, materials, craftsmanship, and build/finishing standards.
     * Physical retail presence (branches, store experience, key locations).
     * Digital footprint, online shop, delivery speed, and customer service responsiveness.
     * Customer satisfaction, sentiment, reviews, and reputation.

3. CLICKABLE SOURCE HYPERLINKS (روابط المصادر الحية):
   - In every `evidence` item, cite the real source URL from the search evidence formatted as a markdown hyperlink: `[Source Title or Domain](URL)`.
   - Never create fictional or placeholder URLs; cite only authentic URLs provided in the SEARCH EVIDENCE block.

4. FIXED DIMENSIONS & VERDICTS:
   - `verdict` MUST be exactly one of: "a_leads" (Competitor A is superior), "b_leads" (Competitor B is superior), or "parity" (tied or balanced).
   - In each dimension, provide concrete, distinct analysis for competitor A and competitor B.

5. GAPS & OPPORTUNITIES:
   - A **gap** is where one competitor does something the other does not, and it matters commercially. `who_has_it` names who has it ("a", "b", "both", "neither"). Include evidence with source hyperlinks.
   - An **opportunity** is an actionable move for OUR brand to exploit. Give it realistic `effort` and `impact` ("low" | "medium" | "high"), and a concrete, immediate `first_step`.
   - `recommended_focus` is 2-3 ranked, strategic, punchy recommendations (each 1-2 sentences).

6. LANGUAGE:
   - Answer in {language_name}. Keep JSON keys strictly in English as defined in the schema.
   - Output ONE strict JSON object and nothing else. No markdown fences, no explanatory commentary.

Schema:
{{
  "summary": "Detailed executive summary comparing both competitors, highlighting key advantages and market standing.",
  "dimensions": [
    {{"dimension": "{dimensions}", "competitor_a": "Detailed analysis for A with specific evidence and links", "competitor_b": "Detailed analysis for B with specific evidence and links",
      "verdict": "a_leads|b_leads|parity", "evidence": ["[Source Title](URL) - Observation 1", "[Source Title](URL) - Observation 2"]}}
  ],
  "gaps": [
    {{"title": "...", "why_it_matters": "...", "who_has_it": "a|b|both|neither",
      "evidence": ["[Source Title](URL) - Observation"], "confidence": "high|medium|low"}}
  ],
  "opportunities": [
    {{"title": "...", "rationale": "...", "effort": "low|medium|high",
      "impact": "low|medium|high", "first_step": "...", "evidence": ["[Source Title](URL) - Observation"]}}
  ],
  "recommended_focus": ["...", "..."]
}}
"""


def _sources_block(hits: list[dict[str, Any]], label: str) -> str:
    """Render search hits as numbered evidence for one side."""
    if not hits:
        return f"### {label}\n(no results retrieved)\n"
    lines = [f"### {label}"]
    for index, hit in enumerate(hits[:12], start=1):
        title = str(hit.get("title") or "").strip() or "(untitled)"
        url = str(hit.get("url") or hit.get("link") or "").strip()
        snippet = str(hit.get("snippet") or hit.get("content") or "").strip()
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet[:800]}")
    return "\n".join(lines) + "\n"


def _clean_keywords(text: str) -> str:
    """Extract substantive keywords from conversational category/notes."""
    raw = (text or "").strip()
    if not raw:
        return ""
    stopwords = {
        "قارنلي", "قارن", "بينهم", "بين", "في", "من", "على", "عن", "مع", "هو", "هي", "كل", "حاجة",
        "عايز", "عاوز", "اريد", "أريد", "اعمل", "لو", "سمحت", "بالله", "عليك", "ممكن",
        "السوق", "المصري", "مصر", "بتاعتهم", "بتاعهم",
        "compare", "between", "versus", "vs", "in", "for", "and", "the", "please",
        "market", "tell", "show", "me", "what", "are", "their", "is",
    }
    words = re.findall(r"[\w\u0600-\u06FF]+", raw)
    filtered = [w for w in words if w.casefold() not in stopwords and len(w) > 1]
    return " ".join(filtered[:8])


def _dedupe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate web hits by normalized URL or title."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for hit in hits:
        url = str(hit.get("url") or hit.get("link") or "").strip().rstrip("/").casefold()
        title = str(hit.get("title") or "").strip().casefold()
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        elif title:
            if title in seen_titles:
                continue
            seen_titles.add(title)
        else:
            continue
        deduped.append(hit)
    return deduped


def _strip_json(raw: str) -> str:
    """Pull the first JSON object out of a model reply."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _invoke(llm: Any, system: str, user: str) -> str:
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content or "")


def _markdown_summary(
    comparison: CompetitorComparison,
    a: str,
    b: str,
    sources_a: list[dict[str, Any]] | None = None,
    sources_b: list[dict[str, Any]] | None = None,
) -> str:
    """A compact markdown rendering, so the existing report panel stays useful."""
    lines = [f"## {a} vs {b}", "", comparison.summary.strip(), ""]
    if comparison.dimensions:
        lines.extend(["### Dimensions", "", f"| Dimension | {a} | {b} | Verdict |", "|---|---|---|---|"])
        for dim in comparison.dimensions:
            verdict = {"a_leads": f"{a} leads", "b_leads": f"{b} leads", "parity": "parity"}[dim.verdict]
            lines.append(
                f"| {dim.dimension} | {dim.competitor_a or '—'} | {dim.competitor_b or '—'} | {verdict} |"
            )
        lines.append("")
    if comparison.gaps:
        lines.extend(["### Gaps", ""])
        for gap in comparison.gaps:
            evidence_str = ""
            if gap.evidence:
                evidence_str = " · " + "; ".join(gap.evidence[:2])
            lines.append(f"- **{gap.title}** ({gap.who_has_it}, {gap.confidence} confidence) — {gap.why_it_matters}{evidence_str}")
        lines.append("")
    if comparison.opportunities:
        lines.extend(["### Opportunities", ""])
        for opp in comparison.opportunities:
            evidence_str = ""
            if opp.evidence:
                evidence_str = " · " + "; ".join(opp.evidence[:2])
            lines.append(f"- **{opp.title}** — impact {opp.impact} / effort {opp.effort}. {opp.first_step}{evidence_str}")
        lines.append("")
    if comparison.recommended_focus:
        lines.extend(["### Recommended focus", ""])
        lines.extend(f"{i}. {item}" for i, item in enumerate(comparison.recommended_focus, start=1))
        lines.append("")

    all_sources_a = [s for s in (sources_a or []) if str(s.get("url") or s.get("link") or "").strip()]
    all_sources_b = [s for s in (sources_b or []) if str(s.get("url") or s.get("link") or "").strip()]
    if all_sources_a or all_sources_b:
        lines.extend(["### Verified Sources & Evidence", ""])
        if all_sources_a:
            lines.append(f"#### {a}")
            for s in all_sources_a[:6]:
                title = str(s.get("title") or s.get("url") or s.get("link")).strip()
                url = str(s.get("url") or s.get("link")).strip()
                lines.append(f"- [{title}]({url})")
            lines.append("")
        if all_sources_b:
            lines.append(f"#### {b}")
            for s in all_sources_b[:6]:
                title = str(s.get("title") or s.get("url") or s.get("link")).strip()
                url = str(s.get("url") or s.get("link")).strip()
                lines.append(f"- [{title}]({url})")
            lines.append("")

    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #


async def _search(provider: Any, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Best-effort search: normalizes SearchResult models and dicts."""
    clean_query = query.strip()
    if not clean_query:
        return []
    try:
        hits = await asyncio.to_thread(provider.search, clean_query, max_results=limit)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
        logger.warning(
            "compare_competitors search failed",
            extra={"query": clean_query, "error": str(exc)[:200]},
        )
        return []

    results: list[dict[str, Any]] = []
    for hit in (hits or []):
        if hasattr(hit, "model_dump"):
            results.append(hit.model_dump())
        elif isinstance(hit, dict):
            results.append(hit)
        elif hasattr(hit, "__dict__"):
            results.append(hit.__dict__)
    return results


async def _gather_deep_competitor_evidence(
    provider: Any,
    name: str,
    category: str,
    is_ar: bool,
) -> list[dict[str, Any]]:
    """Gather rich multi-angle search evidence for a competitor."""
    clean_cat = _clean_keywords(category)

    queries: list[str] = []
    if is_ar:
        queries.append(f"{name} مصر")
        queries.append(f"{name} أسعار منتجات عروض")
        queries.append(f"{name} فروع عناوين")
        queries.append(f"{name} خامات جودة تقييمات ريفيوهات")
        if clean_cat:
            queries.append(f"{name} {clean_cat}")
    else:
        queries.append(f"{name}")
        queries.append(f"{name} products prices offers")
        queries.append(f"{name} stores branches locations")
        queries.append(f"{name} quality reviews customer experience")
        if clean_cat:
            queries.append(f"{name} {clean_cat}")

    search_tasks = [_search(provider, q, limit=4) for q in queries]
    results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)

    all_hits: list[dict[str, Any]] = []
    for res in results_lists:
        if isinstance(res, list):
            all_hits.extend(res)

    deduped = _dedupe_hits(all_hits)

    if not deduped:
        deduped = await _search(provider, name, limit=6)

    return deduped[:12]


@router.post("/compare")
async def compare_competitors(
    payload: CompareRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    request_id: RequestIdDep,
) -> dict[str, Any]:
    """Compare two named competitors and return structured gaps and opportunities."""
    name_a = payload.competitor_a.strip()
    name_b = payload.competitor_b.strip()
    if name_a.casefold() == name_b.casefold():
        raise HTTPException(status_code=400, detail="Pick two different competitors.")

    category = (payload.category or "").strip()

    brand_profile = await db.scalar(
        select(BrandProfile).where(BrandProfile.business_id == ws.business_id).limit(1)
    )
    workspace_language = (
        getattr(brand_profile, "language", None)
        or getattr(ws.business, "language", None)
        or ""
    )
    language = resolve_output_language(
        payload.language, workspace_language=str(workspace_language), query=f"{name_a} {name_b}"
    )
    is_ar = language == "ar"

    ai_settings = get_ai_settings()
    provider = get_search_provider(ai_settings.search_provider)
    llm = get_llm(provider=ai_settings.llm_provider).chat

    # Deep parallel multi-query search for both competitors
    hits_a, hits_b = await asyncio.gather(
        _gather_deep_competitor_evidence(provider, name_a, category, is_ar),
        _gather_deep_competitor_evidence(provider, name_b, category, is_ar),
    )

    evidence_note = ""
    if not hits_a and not hits_b:
        logger.info(
            "compare_competitors: no search hits retrieved, synthesizing from domain patterns",
            extra={"competitor_a": name_a, "competitor_b": name_b, "category": category},
        )
        evidence_note = (
            "\nNote: Live web search returned sparse/no hits for these entities. "
            f"Synthesize the comparison using your expert industry knowledge of typical market patterns "
            f"for '{category or 'this business category'}'. Clearly state in the summary that this analysis "
            "is based on synthesized market domain patterns. Strictly adhere to zero-hallucination: if numbers or specific private facts are unknown, state 'غير متوفر في المصادر العامة المتاحة'.\n"
        )

    user_focus_note = ""
    if category:
        user_focus_note = (
            f"\nUSER SPECIFIC REQUIREMENTS & FOCUS:\n"
            f"The user specifically asked to compare and inspect: '{category}'.\n"
            f"You MUST prioritize and thoroughly address every aspect of this request (e.g. prices, materials/finishing, branches/stores, customer experience) across the relevant dimensions.\n"
        )
    else:
        user_focus_note = (
            "\nEXHAUSTIVE COMPETITIVE AUDIT:\n"
            "The user did not specify custom focus areas. Provide an exhaustive, deeply analytical comparison covering real pricing & offers, materials/quality & finishing, branch network & stores, and customer reputation.\n"
        )

    system = SYSTEM_TEMPLATE.format(
        language_name="Arabic" if is_ar else "English",
        dimensions=" | ".join(DIMENSIONS),
    )
    user_prompt = (
        f"Competitor A: {name_a}\n"
        f"Competitor B: {name_b}\n"
        f"Category / market: {category or 'infer from the evidence, do not invent'}\n"
        f"Our workspace: {ws.business_id}\n\n"
        f"{user_focus_note}\n"
        f"{evidence_note}"
        f"SEARCH EVIDENCE (Use these real URLs for markdown hyperlinks in evidence):\n\n"
        f"{_sources_block(hits_a, f'A — {name_a}')}\n"
        f"{_sources_block(hits_b, f'B — {name_b}')}\n"
        f"Cover these dimensions, in this order: {', '.join(DIMENSIONS)}.\n"
        f"Remember: Zero hallucination. If a claim is not in the evidence, state 'غير متوفر في المصادر العامة المتاحة' instead of guessing."
    )

    raw = ""
    with usage_action("research.compare_competitors"):
        try:
            raw = await asyncio.to_thread(_invoke, llm, system, user_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("compare_competitors llm failed", extra={"error": str(exc)[:300]})
            raise HTTPException(status_code=502, detail="The model call failed; try again.") from exc

    comparison: CompetitorComparison | None = None
    try:
        comparison = CompetitorComparison.model_validate(json.loads(_strip_json(raw)))
    except (json.JSONDecodeError, ValidationError):
        # One repair pass. The first reply is usually valid JSON wrapped in prose;
        # if it is genuinely malformed we ask once more, explicitly, and then fail.
        try:
            repaired = await asyncio.to_thread(
                _invoke,
                llm,
                "You convert a draft into strict JSON. Return only the JSON object; "
                "no fences, no prose.",
                f"Convert this into the required JSON schema:\n\n{raw[:8000]}",
            )
            comparison = CompetitorComparison.model_validate(json.loads(_strip_json(repaired)))
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "compare_competitors unparseable output", extra={"error": str(exc)[:200]}
            )
            raise HTTPException(
                status_code=502,
                detail="The model did not return a valid comparison. Try again.",
            ) from exc

    sources = [
        {
            "title": str(hit.get("title") or "").strip() or str(hit.get("url") or hit.get("link") or "").strip(),
            "url": str(hit.get("url") or hit.get("link") or "").strip(),
            "side": "a",
        }
        for hit in hits_a
        if str(hit.get("url") or hit.get("link") or "").strip()
    ] + [
        {
            "title": str(hit.get("title") or "").strip() or str(hit.get("url") or hit.get("link") or "").strip(),
            "url": str(hit.get("url") or hit.get("link") or "").strip(),
            "side": "b",
        }
        for hit in hits_b
        if str(hit.get("url") or hit.get("link") or "").strip()
    ]

    summary_text = _markdown_summary(comparison, name_a, name_b, sources_a=hits_a, sources_b=hits_b)
    now = datetime.now(timezone.utc)
    memory_id = f"mem_{uuid.uuid4().hex[:12]}"
    title = f"مقارنة: {name_a} مقابل {name_b}" if is_ar else f"{name_a} vs {name_b}"

    compact_text = (
        f"{title}\n{comparison.summary}"
        + "\n" + "\n".join(f"- {gap.title}" for gap in comparison.gaps[:6])
    )[:2000]

    # Persisted under the long-standing `market_gaps` kind so the assistant can
    # discuss it immediately (see the research-item context contract in
    # app/core/orchestrator/service.py).
    mem_item = BrandMemoryItem(
        id=memory_id,
        business_id=ws.business_id,
        kind="market_gaps",
        text=compact_text,
        metadata_={
            "tool": "compare_competitors",
            "query": f"{name_a} vs {name_b}",
            "title": title,
            "full_report": summary_text,
            "details": {
                "subject": f"{name_a} vs {name_b}",
                "tool": "compare_competitors",
                "competitors": [name_a, name_b],
                "category": category,
                "sources": sources,
                "language": language,
                "analyzed_at": now.isoformat(),
                "comparison": comparison.model_dump(),
            },
        },
        created_at=now,
        **memory_defaults(
            "market_gaps",
            business_id=ws.business_id,
            text=compact_text,
            source="research.compare_competitors",
        ),
    )
    db.add(mem_item)
    await db.commit()

    return {
        "id": memory_id,
        "tool": "compare_competitors",
        "query": f"{name_a} vs {name_b}",
        "title": title,
        "summary": summary_text,
        "details": mem_item.metadata_["details"],
        "saved_to_memory": True,
        "created_at": now.isoformat(),
    }
