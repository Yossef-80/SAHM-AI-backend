"""One review pass before Layla submits anything.

A creative deliverable is judged by the user, usually seconds after it appears.
Asking a second, cooled-down pass to re-read the draft against the user's own
brief catches the failures that are obvious on re-reading and invisible while
writing:

* a claim, price, guarantee or warranty the user never supplied;
* variations that are the same line twice with the words moved around;
* the wrong output language (a real failure mode for a bilingual product);
* a deliverable so generic it would fit any business.

The review is deliberately narrow. It may *remove* an unsupported claim or *fix*
a duplicate angle; it may not add facts it was not given, because a critic with
an imagination is just a second hallucination. Everything it changes is reported
in ``notes`` so the user can see what was corrected and why.

Cost: exactly one extra call, at ``critique`` temperature (cold) on the strong
tier. ``AI_LAYLA_SELF_CRITIQUE=0`` turns it off for a deployment that would
rather spend the tokens elsewhere.
"""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger

_logger = get_logger(__name__)

CRITIQUE_SYSTEM = """You are the review step in an advertising copy pipeline. You are given \
the user's brief, the brand facts that are allowed to appear, and a drafted \
deliverable. Your job is to catch defects, not to rewrite for taste.

Return a verdict:
- "ok": the draft is supported, distinct and in the requested language.
- "revise": at least one defect below is present, and you supply a corrected \
deliverable.

Defects, in priority order:
1. UNSUPPORTED: an offer, discount, price, guarantee, warranty, testimonial, \
award, availability, delivery promise, performance claim or brand fact that does \
not appear in the brief or the brand facts. Remove it or replace it with \
something the supplied facts support.
2. DUPLICATE: two or more variants that express the same idea in nearly the same \
words. Re-angle them at genuinely different motives (price, convenience, \
quality, identity, urgency, social proof) using only supplied facts.
3. LANGUAGE: the deliverable is not written in the requested language.
4. EMPTY: a required field (headline, primary text, scene visual) is blank or is \
generic filler such as "quality you can trust" with no supplied fact behind it.
5. CLAIM_DRIFT: the draft silently strengthens a supplied fact ("free delivery in \
Cairo" becoming "free delivery everywhere").

Rules you must follow:
- Never introduce a fact, number, offer, name or claim that is not in the brief \
or the brand facts.
- Keep every field that is already correct exactly as it is; change only what \
the defect requires.
- If the only problem is style that you have no brief to judge, return "ok".
- When you revise, put the complete corrected deliverable in `revised_json` as a \
JSON object with the same field names as the draft.

Return the verdict object."""


class CritiqueVerdict(BaseModel):
    """What one review pass concluded.

    ``revised_json`` is a *string* on purpose. Google's function-calling converter
    logs "Key '$defs' is not supported in schema, ignoring" and drops nested
    object references, so a verdict whose repair is a nested model silently loses
    the repair on the provider this product runs on -- the review then looks like
    it ran and changed nothing. A JSON string survives every provider's tool
    schema; it is parsed and validated here, and anything that does not parse is
    treated exactly like a failed review (keep the draft).
    """

    verdict: str = Field(description='"ok" or "revise"')
    defects: list[str] = Field(
        default_factory=list,
        description="One entry per defect found: UNSUPPORTED, DUPLICATE, LANGUAGE, EMPTY, CLAIM_DRIFT.",
    )
    notes: str = Field(
        default="",
        description="One short user-facing sentence per correction, in the deliverable's language.",
    )
    revised_json: str = Field(
        default="",
        description=(
            "The complete corrected deliverable as a JSON object with the same field names as "
            'the draft, serialized as a string. Empty string when verdict is "ok".'
        ),
    )
    #: An already-parsed repair (callers in this codebase and tests use it; it
    #: never travels to a provider).
    revised: dict[str, Any] | None = None


@lru_cache(maxsize=None)
def critique_schema(deliverable: type[BaseModel]) -> type[BaseModel]:
    """The verdict schema for a specific deliverable type.

    Built rather than hardcoded because the fix has to be judged against the same
    contract the draft came from: a plausible-looking rewrite with a renamed field
    is rejected, not shipped. ``deliverable`` is also what ``apply_verdict``
    validates the returned JSON against.

    Cached per deliverable class: a schema class rebuilt on every call would make
    ``isinstance(result, schema)`` fail against an instance of the *previous*
    build, which turns a valid verdict into a dropped one.
    """
    from pydantic import create_model

    return create_model(
        "CreativeCritiqueVerdict",
        verdict=(str, Field(description='"ok" or "revise"')),
        defects=(list[str], Field(default_factory=list)),
        notes=(str, Field(default="")),
        revised_json=(
            str,
            Field(
                default="",
                description=(
                    "The complete corrected deliverable as a JSON object with the same field "
                    'names as the draft, serialized as a string. Empty string when verdict is "ok".'
                ),
            ),
        ),
        __doc__=f"Verdict for a {deliverable.__name__} review.",
    )


def _payload_text(payload: Any) -> str:
    """Serialize the draft compactly, tolerating models and plain dicts."""
    if isinstance(payload, BaseModel):
        data = payload.model_dump()
    elif isinstance(payload, dict):
        data = payload
    else:  # pragma: no cover - defensive
        return str(payload)
    return json.dumps(data, ensure_ascii=False, indent=2)


def build_critique_request(
    *,
    brief: str,
    draft: Any,
    brand_context_text: str = "",
    language: str = "en",
    extra_rules: str = "",
) -> str:
    """The review prompt: the brief, the permitted facts, and the draft."""
    parts = [
        f"Requested output language: {'Arabic' if language == 'ar' else 'English'} ({language})",
        "",
        "USER BRIEF:",
        (brief or "").strip(),
        "",
        "BRAND FACTS THAT MAY APPEAR (nothing outside this list is allowed to become a claim):",
        (brand_context_text or "").strip() or "(none supplied -- then no specific claim may appear)",
    ]
    if extra_rules.strip():
        parts += ["", "ADDITIONAL CONSTRAINTS FROM THE PRODUCT:", extra_rules.strip()]
    parts += ["", "DRAFT DELIVERABLE:", _payload_text(draft)]
    return "\n".join(parts)


def _repair_payload(verdict: Any) -> Any:
    """The repair a verdict carries, from either shape. None when there is none.

    ``revised`` is an already-parsed dict; ``revised_json`` is the string that
    travels through a tool schema. Both are accepted, neither is trusted.
    """
    parsed = getattr(verdict, "revised", None)
    if isinstance(parsed, BaseModel):
        return parsed.model_dump()
    if isinstance(parsed, dict) and parsed:
        return parsed

    text = (getattr(verdict, "revised_json", "") or "").strip()
    if not text:
        return None
    if text.startswith("```") or text.endswith("```"):
        # Models fence JSON with or without a language tag, with or without
        # newlines. Losing a repair over formatting would be silly.
        text = text.strip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        # The deliverable is the outermost JSON object in whatever survived.
        text = text[start : end + 1]
    try:
        loaded = json.loads(text)
    except (TypeError, ValueError):
        _logger.warning("self-critique repair was not valid JSON; keeping the draft")
        return None
    return loaded if isinstance(loaded, dict) and loaded else None


def apply_verdict(
    draft: Any,
    verdict: Any,
    *,
    schema: type[BaseModel],
) -> tuple[Any, list[str]]:
    """Return ``(deliverable, notes)`` after applying a verdict.

    ``notes`` is a list of user-facing correction lines. The revised payload is
    validated against ``schema`` before it is accepted: a critique that returns a
    malformed deliverable is discarded, because the draft the user is waiting for
    must not be lost to a failed repair.
    """
    if verdict is None or (getattr(verdict, "verdict", "") or "").strip().lower() != "revise":
        return draft, []

    payload = _repair_payload(verdict)
    if payload is None:
        return draft, []

    try:
        candidate = schema.model_validate(payload)
    except Exception as exc:
        _logger.warning(
            "self-critique returned an invalid revised deliverable; keeping the draft",
            extra={"reason": type(exc).__name__},
        )
        return draft, []

    notes = [line.strip() for line in (getattr(verdict, "notes", "") or "").splitlines() if line.strip()]
    if not notes and getattr(verdict, "defects", None):
        notes = [f"Self-review corrected: {', '.join(verdict.defects)}"]
    return candidate, notes


#: The review runs on the same provider as the writing, so it meets the same
#: overload. A single 503 used to end the review silently -- the user then got an
#: unreviewed draft with nothing saying so. Two extra attempts cost nothing when
#: the provider is healthy (only the first is ever sent) and save the guarantee
#: when it is not. Deliberately *not* retried: a validation error or a malformed
#: verdict -- those are deterministic, and repeating them just burns tokens.
_CRITIQUE_ATTEMPTS = 3

#: Markers of a *transient* provider condition. The exception type differs per
#: provider (``GoogleAPIError``, ``openai.APIError``, an httpx error wrapped by a
#: client library), so the status is read defensively from whatever the raised
#: exception exposes, and only then from its message.
_RETRYABLE_STATUS_MARKERS = ("429", "500", "501", "502", "503", "504")


def _retryable_review_error(exc: BaseException) -> bool:
    """True when re-sending the same review request could plausibly succeed."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attr in ("status_code", "status", "code", "http_status"):
            value = getattr(current, attr, None)
            if isinstance(value, int) and (value == 429 or 500 <= value <= 599):
                return True
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and (status == 429 or 500 <= status <= 599):
            return True
        text = f"{type(current).__name__}: {current}".upper()
        if any(marker in text for marker in _RETRYABLE_STATUS_MARKERS):
            return True
        for marker in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "OVERLOADED", "RATE LIMIT", "HIGH DEMAND", "TIMEOUT"):
            if marker in text:
                return True
        current = current.__cause__ or current.__context__
    return False


def _review_retry_delay(attempt: int) -> float:
    """Short, bounded backoff: the provider is busy, not gone."""
    return float(min(2 ** attempt, 8))


async def critique_deliverable(
    llm: Any,
    *,
    brief: str,
    draft: Any,
    brand_context_text: str = "",
    language: str = "en",
    extra_rules: str = "",
    schema: type[BaseModel] = CritiqueVerdict,
    sleep: Any | None = None,
) -> CritiqueVerdict | None:
    """Run the review pass. Returns None when the review itself failed.

    A failed review must never block a deliverable: the caller keeps the draft.
    """
    if llm is None:  # pragma: no cover - caller opted out
        return None
    request = build_critique_request(
        brief=brief,
        draft=draft,
        brand_context_text=brand_context_text,
        language=language,
        extra_rules=extra_rules,
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    pause = sleep or asyncio.sleep
    messages = [SystemMessage(content=CRITIQUE_SYSTEM), HumanMessage(content=request)]
    result: Any = None
    for attempt in range(1, _CRITIQUE_ATTEMPTS + 1):
        try:
            runnable = llm.structured(schema)
            result = await runnable.ainvoke(messages)
            break
        except Exception as exc:  # provider/validation error
            retryable = _retryable_review_error(exc)
            if retryable and attempt < _CRITIQUE_ATTEMPTS:
                delay = _review_retry_delay(attempt)
                _logger.warning(
                    "creative self-critique hit a transient provider error; retrying",
                    extra={"attempt": attempt, "reason": type(exc).__name__, "delay": delay},
                )
                await pause(delay)
                continue
            _logger.warning(
                "creative self-critique failed; the draft is delivered as written",
                extra={
                    "reason": type(exc).__name__,
                    "attempts": attempt,
                    "retryable": retryable,
                },
            )
            return None

    try:
        # Re-validated from its fields rather than trusted as an instance: a
        # verdict that arrives as any model of the right shape is accepted, and a
        # malformed one is caught here instead of halfway through a deliverable.
        payload = result.model_dump() if isinstance(result, BaseModel) else result
        verdict = result if isinstance(result, schema) else schema.model_validate(payload)
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning(
            "creative self-critique returned an unusable verdict",
            extra={"reason": type(exc).__name__},
        )
        return None

    if (getattr(verdict, "verdict", "") or "").strip().lower() != "revise":
        return verdict
    if _repair_payload(verdict) is None:
        # A "revise" with nothing to revise is not a licence to invent a fix.
        _logger.warning("self-critique asked for a revision but returned no repair; keeping the draft")
        return verdict

    _logger.info(
        "creative_self_critique_revised",
        extra={"defects": getattr(verdict, "defects", []), "language": language},
    )
    return verdict


def critique_enabled(settings: Any) -> bool:
    """Whether the review pass is switched on for this deployment."""
    return bool(getattr(settings, "layla_self_critique", True))


__all__ = [
    "CRITIQUE_SYSTEM",
    "CritiqueVerdict",
    "apply_verdict",
    "build_critique_request",
    "critique_deliverable",
    "critique_enabled",
    "critique_schema",
]
