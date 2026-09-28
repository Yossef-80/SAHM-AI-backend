"""Intent classification.

``classify_intent`` decides which persona handles a turn and which task it runs.

Design note (recorded in docs/DECISIONS.md): routing starts with a deterministic
keyword scorer. It is free, instant, and unit-testable -- eight sample messages
route identically on every machine with no LLM involved. The fast-tier LLM is
only consulted when the rules are not confident, and it must return the same
Pydantic model. That keeps the LLM on the path the brief asks for without making
routing non-deterministic or expensive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from app.core.llm.base import LLMClient, Msg
from app.core.llm.router import get_llm
from app.core.logging import get_logger
from app.core.agents.personas import PERSONAS
from app.schemas.common import IntentKind

_logger = get_logger(__name__)


class IntentClassification(BaseModel):
    """The structured output of the classifier."""

    intent: IntentKind
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    missing_info: list[str] = Field(default_factory=list)
    task_name: str | None = None
    persona_id: str | None = None
    rationale: str = ""


#: intent -> (persona, default task)
INTENT_ROUTES: dict[IntentKind, tuple[str, str]] = {
    IntentKind.RESEARCH: ("researcher", "competitor_research"),
    IntentKind.STRATEGY: ("omar", "marketing_strategy"),
    IntentKind.CREATIVE: ("layla", "ad_copy"),
    IntentKind.ANALYZE: ("karim", "performance_analysis"),
    IntentKind.OPTIMIZE: ("salma", "optimization_proposal"),
    IntentKind.ASK_CLARIFICATION: ("omar", "marketing_strategy"),
}

#: Keyword -> intent, with weights. Highest weight wins.
_RULES: list[tuple[str, IntentKind, float]] = [
    # Ordered most-specific-first. On a tie the earlier rule wins, so the broad
    # research catch-all must stay last.
    (
        r"\b(optimis|optimiz|improve|pause|scale|budget|bid|fix|should we (stop|keep))\w*",
        IntentKind.OPTIMIZE,
        0.85,
    ),
    (r"\b(recommend\w*|proposal|what should (we|i) change)\b", IntentKind.OPTIMIZE, 0.8),
    (
        r"\b(how (is|are|did)\b.*\b(perform|doing|campaign)|performance|ctr|cpc|cpm|roas|"
        r"conversions?|spend|impressions?|clicks|fatigue|breakdown|analys|analyz)\w*",
        IntentKind.ANALYZE,
        0.85,
    ),
    (r"\b(insight|report|metrics|numbers|stats)\b", IntentKind.ANALYZE, 0.6),
    (
        r"\b(write|create|make|generate|draft|design|brief)\b.*\b(ad|copy|creative|image|"
        r"video|script|poster|banner|headline|reel|commercial)\w*",
        IntentKind.CREATIVE,
        0.9,
    ),
    (
        r"\b(ad copy|headlines?|creative brief|image prompt|video prompt|commercial|script|"
        r"variations?|tagline|slogan)\b",
        IntentKind.CREATIVE,
        0.85,
    ),
    (
        r"\b(strategy|strategic|plan|positioning|roadmap|go.to.market|channels?|"
        r"content calendar|calendar)\b",
        IntentKind.STRATEGY,
        0.85,
    ),
    (r"\b(how should we (grow|approach)|what should our)\b", IntentKind.STRATEGY, 0.7),
    (
        r"\b(research\w*|competitors?|market\w*|keywords?|seo|reviews?|trends?|"
        r"audience insight|what (are|do) (people|competitors)\w*)\b",
        IntentKind.RESEARCH,
        0.85,
    ),
    (r"\b(find out|look up|analyse the market|analyze the market)\b", IntentKind.RESEARCH, 0.75),
]

#: Signals that a specific task is meant, overriding the intent default.
_TASK_HINTS: list[tuple[str, str]] = [
    (r"\bcompetitors?", "competitor_research"),
    (r"\bmarket gap|whitespace|opportunity gap", "market_gaps"),
    (r"\bmarket (analysis|research|overview)", "market_analysis"),
    (r"\bcontent calendar|posting schedule|calendar", "content_calendar"),
    (r"\bcampaign (strategy|plan|structure)", "campaign_strategy"),
    (r"\bmarketing strategy|overall strategy|growth strategy", "marketing_strategy"),
    (r"\bcreative brief|brief for", "creative_brief"),
    (r"\bcommercial|video script|tv script", "commercial_script"),
    (r"\bimage prompt|prompt for (an|the) image", "image_prompt"),
    (r"\bvideo prompt|prompt for (a|the) video", "video_prompt"),
    (r"\bvariations?|a/b|split test", "ad_variations"),
    (r"\bad copy|copy for|headlines?", "ad_copy"),
    (r"\bperformance|how (is|are|did).*perform|analys|analyz", "performance_analysis"),
    (r"\boptimis|optimiz|recommend|what should (we|i) change", "optimization_proposal"),
]

#: Things a task needs before it can run well.
_REQUIRED_BY_TASK: dict[str, list[str]] = {
    "performance_analysis": ["campaign_id"],
    "optimization_proposal": ["campaign_id"],
    "campaign_strategy": ["budget"],
    "creative_brief": ["campaign_id"],
    "ad_variations": ["campaign_id"],
    "image_prompt": [],
    "ad_copy": [],
    "competitor_research": [],
    "market_analysis": [],
    "market_gaps": [],
    "marketing_strategy": [],
    "content_calendar": [],
    "commercial_script": [],
    "video_prompt": [],
}

#: The only keys a classifier may report as missing. Anything else is dropped
#: so a stray model output cannot produce a question the UI cannot answer.
MISSING_INFO_KEYS = ("campaign_id", "budget", "audience", "goal")

_HUMAN_LABELS = {
    "campaign_id": "which campaign this is about",
    "budget": "the budget for this campaign",
    "audience": "who the target audience is",
    "goal": "what the campaign is trying to achieve",
}


@dataclass
class RuleGuess:
    intent: IntentKind
    confidence: float
    task_name: str | None = None
    missing_info: list[str] = field(default_factory=list)
    rationale: str = ""


def _score(text: str) -> tuple[IntentKind, float, str]:
    lowered = text.lower()
    best: tuple[IntentKind, float, str] = (IntentKind.RESEARCH, 0.0, "no rule matched")
    for pattern, intent, weight in _RULES:
        if re.search(pattern, lowered) and weight > best[1]:
            best = (intent, weight, f"matched {pattern}")
    return best


def _pick_task(text: str, intent: IntentKind, default_task: str) -> str:
    """Pick a task from the intent's persona's own allowlist.

    Scoped to the persona, so a keyword belonging to another persona (e.g.
    "performance" in "what should I change to improve performance?") cannot
    pull the turn to karim when salma should own it.
    """
    lowered = text.lower()
    persona_id = INTENT_ROUTES[intent][0]
    runnable = set(PERSONAS[persona_id].allowed_tasks)
    for pattern, task in _TASK_HINTS:
        if task in runnable and re.search(pattern, lowered):
            return task
    return default_task


#: An explicit campaign reference. Deliberately strict: the plain word
#: "campaign" must NOT count, otherwise every campaign question looks answered.
_CAMPAIGN_REF = re.compile(
    r"\b(?:cmp|camp)[_\-][a-z0-9]{4,}\b|\bcampaign\s*(?:id|#)\s*[:=]?\s*[a-z0-9]{4,}\b"
)


def _has_campaign_reference(lowered: str) -> bool:
    return bool(_CAMPAIGN_REF.search(lowered))


def _detect_missing_info(text: str, task_name: str, context: dict[str, Any]) -> list[str]:
    """Info the task needs that is not in the message or the context."""
    missing: list[str] = []
    lowered = text.lower()

    for requirement in _REQUIRED_BY_TASK.get(task_name, []):
        if requirement == "campaign_id":
            if not context.get("campaign_id") and not _has_campaign_reference(lowered):
                missing.append("campaign_id")
        elif requirement == "budget":
            has_budget = re.search(
                r"(\d[\d,\.]*\s?(egp|usd|gbp|sar|aed|le|pounds|dollars))|"
                r"\bbudget (of|is)?\s*\d| monthly budget",
                lowered,
            )
            if not has_budget and not context.get("budget"):
                missing.append("budget")
    return missing


def classify_with_rules(user_message: str, context: dict[str, Any]) -> RuleGuess:
    """Deterministic pre-classifier. No LLM, no cost, fully testable."""
    intent, confidence, rationale = _score(user_message)
    default_task = INTENT_ROUTES[intent][1]
    task_name = _pick_task(user_message, intent, default_task)
    missing = _detect_missing_info(user_message, task_name, context)
    return RuleGuess(
        intent=intent,
        confidence=confidence,
        task_name=task_name,
        missing_info=missing,
        rationale=rationale,
    )


async def classify_intent(
    user_message: str,
    *,
    context: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
    brand_context: Any = None,
    confidence_threshold: float = 0.7,
) -> IntentClassification:
    """Classify a turn into an intent, a task and a persona.

    Rules first; the fast-tier LLM only when the rules are unsure. If required
    info is missing and cannot be found in brand memory, the caller routes to
    ask_clarification.
    """
    ctx = context or {}
    guess = classify_with_rules(user_message, ctx)

    if guess.confidence >= confidence_threshold:
        return IntentClassification(
            intent=guess.intent,
            confidence=guess.confidence,
            missing_info=guess.missing_info,
            task_name=guess.task_name,
            persona_id=INTENT_ROUTES[guess.intent][0],
            rationale=guess.rationale,
        )

    # Not confident: ask the fast tier.
    client = llm or get_llm("fast")
    system = (
        "You classify a marketing-assistant user message into exactly one intent "
        "and pick the task and persona that should handle it.\n\n"
        "Intents: research, strategy, creative, analyze, optimize, ask_clarification.\n"
        "Personas: researcher (research tasks), omar (strategy), layla (creative), "
        "karim (analysis), salma (optimization).\n"
        "Tasks: competitor_research, market_analysis, market_gaps, marketing_strategy, "
        "campaign_strategy, content_calendar, creative_brief, commercial_script, "
        "image_prompt, video_prompt, ad_copy, ad_variations, performance_analysis, "
        "optimization_proposal.\n\n"
        "If the message does not contain enough information to pick a task, set "
        "intent='ask_clarification' and list exactly what is missing in "
        "missing_info, using these keys where they apply: campaign_id, budget, "
        "audience, goal.\n"
        f"Deterministic pre-classifier guess: intent={guess.intent.value}, "
        f"task={guess.task_name}, confidence={guess.confidence:.2f}. Override it if "
        "it is wrong."
    )
    brand_note = ""
    if brand_context is not None:
        brand_note = (
            f"\nBrand memory says the business is '{getattr(brand_context, 'business_name', '')}' "
            f"in {getattr(brand_context, 'country', 'an unknown market')}."
        )

    try:
        result = await client.complete(
            system=system,
            messages=[Msg(role="user", content=f"User message: {user_message}{brand_note}")],
            response_schema=IntentClassification,
            tier="fast",
            max_tokens=400,
            temperature=0.0,
        )
        if result.parsed is not None:
            classification = result.parsed
            # Keep only documented keys, and fall back to the rule guess when the
            # model returns nothing usable.
            clean = [k for k in classification.missing_info if k in MISSING_INFO_KEYS]
            if not clean:
                clean = [k for k in guess.missing_info if k in MISSING_INFO_KEYS]
            classification = classification.model_copy(update={"missing_info": clean})
            _logger.info("intent_classified_by_llm", extra={"intent": classification.intent.value})
            return classification
    except Exception as exc:  # pragma: no cover - fall back to rules
        _logger.warning("intent_llm_failed_using_rules", extra={"error": str(exc)})

    return IntentClassification(
        intent=guess.intent,
        confidence=guess.confidence,
        missing_info=guess.missing_info,
        task_name=guess.task_name,
        persona_id=INTENT_ROUTES[guess.intent][0],
        rationale=guess.rationale,
    )


def clarification_question(missing_info: list[str]) -> dict[str, Any]:
    """Build the payload for a question_with_options block."""
    options: list[dict[str, Any]] = []
    for key in missing_info:
        label = _HUMAN_LABELS.get(key, key)
        if key == "campaign_id":
            options.append(
                {
                    "id": "pick_campaign",
                    "label": "Choose a campaign",
                    "description": "Pick from your existing campaigns.",
                    "payload": {"action": "list_campaigns"},
                }
            )
        elif key == "budget":
            options.append(
                {
                    "id": "give_budget",
                    "label": "Tell me the budget",
                    "description": "e.g. 50,000 EGP for the month.",
                    "payload": {"action": "free_text"},
                }
            )
        else:
            options.append(
                {
                    "id": f"provide_{key}",
                    "label": f"Tell me {label}",
                    "payload": {"action": "free_text"},
                }
            )
    question = (
        "I need a bit more before I can do this properly: "
        + ", ".join(_HUMAN_LABELS.get(m, m) for m in missing_info)
        + "."
    )
    return {
        "question": question,
        "options": options,
        "missing_info": list(missing_info),
        "allow_free_text": True,
    }


__all__ = [
    "INTENT_ROUTES",
    "IntentClassification",
    "RuleGuess",
    "classify_intent",
    "classify_with_rules",
    "clarification_question",
]
