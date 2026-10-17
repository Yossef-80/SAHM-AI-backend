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
    category: str | None = Field(default=None, max_length=160)
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

SYSTEM_TEMPLATE = """You are a competitive-intelligence analyst for a marketing platform.

Compare two real competitors on the SAME fixed dimensions, using ONLY the search
evidence provided. Rules:

1. Ground every statement in the evidence. Put the observations you used in
   `evidence` as short factual phrases (you may name the source domain).
2. If the evidence does not cover a dimension, say so in that dimension's text
   ("no data in the retrieved sources") instead of inventing a plausible answer.
   Never invent prices, follower counts, revenue, or campaign names.
3. `verdict` is one of: a_leads (competitor A is stronger), b_leads, parity.
4. A **gap** is where one competitor does something the other does not, and it
   matters commercially. `who_has_it` names the side that has it; use "neither"
   when the gap is an unmet market need neither addresses.
5. An **opportunity** is an actionable move for OUR brand — not a restatement of
   a gap. Give it effort and impact, and one concrete first step.
6. `recommended_focus` is 2-3 items, ranked, each a single sentence.
7. Answer in {language_name}. Keep field names in English.

Return ONE JSON object and nothing else. No markdown fences, no commentary.

Schema:
{{
  "summary": "one paragraph, plain language",
  "dimensions": [
    {{"dimension": "{dimensions}", "competitor_a": "...", "competitor_b": "...",
      "verdict": "a_leads|b_leads|parity", "evidence": ["...", "..."]}}
  ],
  "gaps": [
    {{"title": "...", "why_it_matters": "...", "who_has_it": "a|b|both|neither",
      "evidence": ["..."], "confidence": "high|medium|low"}}
  ],
  "opportunities": [
    {{"title": "...", "rationale": "...", "effort": "low|medium|high",
      "impact": "low|medium|high", "first_step": "...", "evidence": ["..."]}}
  ],
  "recommended_focus": ["...", "..."]
}}
"""


def _sources_block(hits: list[dict[str, Any]], label: str) -> str:
    """Render search hits as numbered evidence for one side."""
    if not hits:
        return f"### {label}\n(no results retrieved)\n"
    lines = [f"### {label}"]
    for index, hit in enumerate(hits[:6], start=1):
        title = str(hit.get("title") or "").strip() or "(untitled)"
        url = str(hit.get("url") or hit.get("link") or "").strip()
        snippet = str(hit.get("snippet") or hit.get("content") or "").strip()
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet[:600]}")
    return "\n".join(lines) + "\n"


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


def _markdown_summary(comparison: CompetitorComparison, a: str, b: str) -> str:
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
            lines.append(f"- **{gap.title}** ({gap.who_has_it}, {gap.confidence} confidence) — {gap.why_it_matters}")
        lines.append("")
    if comparison.opportunities:
        lines.extend(["### Opportunities", ""])
        for opp in comparison.opportunities:
            lines.append(f"- **{opp.title}** — impact {opp.impact} / effort {opp.effort}. {opp.first_step}")
        lines.append("")
    if comparison.recommended_focus:
        lines.extend(["### Recommended focus", ""])
        lines.extend(f"{i}. {item}" for i, item in enumerate(comparison.recommended_focus, start=1))
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #


async def _search(provider: Any, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Best-effort search: a dead provider must not silently produce an empty comparison."""
    try:
        hits = await asyncio.to_thread(provider.search, query, max_results=limit)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
        logger.warning(
            "compare_competitors search failed",
            extra={"query": query, "error": str(exc)[:200]},
        )
        return []
    return [hit for hit in (hits or []) if isinstance(hit, dict)]


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
        (brand_profile.language if brand_profile is not None else None)
        or getattr(getattr(ws, "business", None), "language", None)
        or ""
    )
    language = resolve_output_language(
        payload.language, workspace_language=str(workspace_language), query=f"{name_a} {name_b}"
    )
    is_ar = language == "ar"

    ai_settings = get_ai_settings()
    provider = get_search_provider(ai_settings.search_provider)
    llm = get_llm(provider=ai_settings.llm_provider).chat

    query_a = f"{name_a} {category}".strip()
    query_b = f"{name_b} {category}".strip()
    hits_a, hits_b = await asyncio.gather(_search(provider, query_a), _search(provider, query_b))
    if not hits_a and not hits_b:
        raise HTTPException(
            status_code=502,
            detail=(
                "The search provider returned nothing for both competitors, so there is no "
                "evidence to compare. Check the search provider configuration and try again."
            ),
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
        f"SEARCH EVIDENCE\n\n{_sources_block(hits_a, f'A — {name_a}')}\n"
        f"{_sources_block(hits_b, f'B — {name_b}')}\n"
        f"Cover these dimensions, in this order: {', '.join(DIMENSIONS)}."
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
            "title": str(hit.get("title") or "").strip(),
            "url": str(hit.get("url") or hit.get("link") or "").strip(),
            "side": "a",
        }
        for hit in hits_a[:6]
    ] + [
        {
            "title": str(hit.get("title") or "").strip(),
            "url": str(hit.get("url") or hit.get("link") or "").strip(),
            "side": "b",
        }
        for hit in hits_b[:6]
    ]

    summary_text = _markdown_summary(comparison, name_a, name_b)
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
