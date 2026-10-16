"""Turning what the user does into what Layla remembers.

A brand-voice description is an adjective ("warm but professional"); an approved
piece of copy is an example. Models follow examples far better than adjectives --
especially small fast models -- so every signal the user already produces is
worth more than another paragraph of instructions:

* they edit a headline we wrote (manual edit) -> the before/after pair *is* the
  style rule, no interpretation needed;
* they tell Layla "أقصر" (a correction in chat) -> the previous draft plus the
  instruction is a negative/positive pair;
* they save or approve a piece of copy -> that text becomes a positive example.

All three are stored as ``voice_example`` memory items -- the highest-value kind
the brand-memory store already supports -- and the top few are injected back into
Layla's prompts as few-shot examples.

Nothing here raises: a learning whose write fails must never cost the user the
deliverable they just approved.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from app.core.logging import get_logger

_logger = get_logger(__name__)

#: Longest example text stored. Examples are prompt material, not transcripts:
#: a 2000-character pair crowds out the brand context it is supposed to sharpen.
MAX_EXAMPLE_CHARS = 600

#: Minimum change ratio for an edit to count as a style signal. Guards against
#: recording "added a comma" as brand voice.
MIN_EDIT_RATIO = 0.12


# --------------------------------------------------------------------------- #
# Correction intent                                                           #
# --------------------------------------------------------------------------- #

#: Arabic and English phrasings a user reaches for when they are redirecting the
#: writing rather than asking something new. Kept as word/stem patterns because
#: both languages inflect heavily ("قصّرها", "خليها أقصر", "make it shorter").
_CORRECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?:^|\s)(?:أ?قصر|قص[ّ]?ر(?:ها|ه)?|أختصر|اختصر|مختصر|shorten|shorter|less\s+wordy|concise)", "shorter"),
    (r"(?:طو[ّ]?ل|أطول|تفصيل\s+أكتر|أوسع|expand|longer|more\s+detail|elaborate)", "longer"),
    (r"(?:مش\s+كده|مش\s+مناسب|مش\s+حلو|وحش|ضعيف|مش\s+عاجبني|بدّل|غيرها|غيّرها|redo|try\s+again|not\s+(?:quite|right)|doesn.t\s+(?:sound|feel))", "redirection"),
    (r"(?:رسمي\s+أكتر|أكثر\s+رسمية|formal|more\s+professional)", "more_formal"),
    (r"(?:بسيط|أبسط|عامي|simple|plain|casual|less\s+formal)", "simpler"),
    (r"(?:حماس|متحمس|energetic|exciting|punchier|بشكل\s+أقوى)", "more_energetic"),
    (r"(?:أهدى|هادئ|ناعم|calm|softer|gentler|أقل\s+مبالغة)", "calmer"),
    (r"(?:مبالغة|مبالغ|over\s*the\s*top|salesy|too\s+salesy|لا\s+تبالغ)", "less_salesy"),
    (r"(?:عربي|بالعربي|in\s+arabic|english|بالانجليزي|بالإنجليزي|ترجم)", "language_switch"),
)

_CORRECTION_RE = tuple((re.compile(pattern, re.IGNORECASE), label) for pattern, label in _CORRECTION_PATTERNS)

#: Above this length a message is doing more than reacting to the last draft.
#: Arabic says "أقصر" where English says "make it a bit shorter" -- measured in
#: characters, not words, so the Arabic budget is the stricter one.
MAX_CORRECTION_CHARS = 160

#: Verbs that open a *new* request. A long message that also asks for a new
#: deliverable ("اعمل حملة جديدة ... وعايز نسخة أقصر") is a brief, not a
#: correction; a short one ("اعمل نسخة أقصر") is still a correction.
_NEW_TASK_MARKERS = re.compile(
    r"(?:اعمل|أعمل|اكتب|أكتب|صمم|جه[ّ]?ز|عايز|عاوز|محتاج|اقترح|generate|write|create|"
    r"make|draft|build|produce|design|plan)",
    re.IGNORECASE,
)

#: Above this length, a message that names a new task is a brief.
NEW_TASK_MIN_CHARS = 60


def correction_intent(message: str) -> str | None:
    """The kind of correction a short user message is, or None.

    Deliberately conservative: it only fires on short messages that are clearly
    instructions about the previous output. A long brief that happens to contain
    "shorter" is a new request, not a correction.
    """
    text = (message or "").strip()
    if not text or len(text) > MAX_CORRECTION_CHARS:
        return None
    if len(text) > NEW_TASK_MIN_CHARS and _NEW_TASK_MARKERS.search(text):
        return None
    for pattern, label in _CORRECTION_RE:
        if pattern.search(text):
            return label
    return None


def is_meaningful_edit(original: str, revised: str, *, threshold: float = MIN_EDIT_RATIO) -> bool:
    """True when ``revised`` differs from ``original`` in more than punctuation.

    ``SequenceMatcher.ratio`` is 1.0 for identical strings, so the change ratio
    is its complement.
    """
    a, b = (original or "").strip(), (revised or "").strip()
    if not a or not b or a == b:
        return False
    return (1.0 - SequenceMatcher(None, a, b).ratio()) >= threshold


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #


def _clip(text: str, limit: int = MAX_EXAMPLE_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render_edit_example(*, original: str, revised: str, context: str = "") -> str:
    """The stored text for a manual-edit signal.

    Rendered as an instruction-shaped pair on purpose: retrieved later, it reads
    as "given a draft like this, this is what they wanted instead".
    """
    lines = []
    if context.strip():
        lines.append(f"Brief: {_clip(context, 200)}")
    lines.append(f"Rejected: {_clip(original)}")
    lines.append(f"Accepted: {_clip(revised)}")
    return "\n".join(lines)


def render_correction_example(*, draft: str, instruction: str, label: str) -> str:
    """The stored text for a correction given in chat."""
    return "\n".join(
        [
            f"User correction ({label}): {_clip(instruction, 200)}",
            f"Draft they were reacting to: {_clip(draft)}",
            "Preference: write the next version this way instead.",
        ]
    )


def render_approval_example(*, text: str, brief: str = "") -> str:
    """The stored text for copy the user kept."""
    lines = []
    if brief.strip():
        lines.append(f"Brief: {_clip(brief, 200)}")
    lines.append(f"Approved copy: {_clip(text)}")
    return "\n".join(lines)


def format_examples_block(examples: list[str]) -> str:
    """Render retrieved examples as the few-shot block of a prompt."""
    cleaned = [example.strip() for example in examples if (example or "").strip()]
    if not cleaned:
        return ""
    lines = [
        "HOW THIS BRAND HAS BEEN WRITTEN BEFORE (real examples from this workspace)",
        "Match their register, rhythm and level of specificity. These outrank any",
        "generic style advice; do not copy their content, only their manner.",
        "",
    ]
    for index, example in enumerate(cleaned, start=1):
        lines.append(f"Example {index}:")
        lines.append(example)
        lines.append("")
    return "\n".join(lines).rstrip()


# --------------------------------------------------------------------------- #
# Selection                                                                   #
# --------------------------------------------------------------------------- #

#: Below this similarity an example is noise rather than style guidance. The
#: store's embeddings are provider-dependent, so the bar is low and the ranking
#: does the real filtering.
MIN_EXAMPLE_SIMILARITY = 0.05


async def select_voice_examples(
    store: Any,
    *,
    business_id: str,
    query: str,
    limit: int = 3,
    campaign_id: str | None = None,
) -> list[str]:
    """The most relevant past examples for this request, best first.

    Uses the same similarity search as the rest of brand memory, so a workspace
    with no examples yet simply gets an empty list -- no special-casing.
    """
    if limit <= 0:
        return []
    try:
        matches = await store.search(
            business_id=business_id,
            query=(query or "advertising copy").strip()[:500],
            kinds=["voice_example"],
            top_k=limit,
            campaign_id=campaign_id,
        )
    except Exception as exc:  # pragma: no cover - degraded, never fatal
        _logger.warning(
            "voice example retrieval failed",
            extra={"business_id": business_id, "reason": type(exc).__name__},
        )
        return []

    examples: list[str] = []
    for item, score in matches:
        if score is not None and score < MIN_EXAMPLE_SIMILARITY:
            continue
        examples.append(item.text)
    return examples[:limit]


# --------------------------------------------------------------------------- #
# Recording                                                                   #
# --------------------------------------------------------------------------- #


async def record_voice_example(
    store: Any,
    *,
    business_id: str,
    text: str,
    source: str,
    metadata: dict[str, Any] | None = None,
    campaign_id: str | None = None,
    conversation_id: str | None = None,
    agent: str = "layla",
) -> str | None:
    """Write one ``voice_example`` item. Returns its id, or None on failure."""
    body = (text or "").strip()
    if not body:
        return None
    try:
        item = await store.write_item(
            business_id=business_id,
            kind="voice_example",
            text=body,
            metadata={**(metadata or {}), "signal": source},
            campaign_id=campaign_id,
            conversation_id=conversation_id,
            agent=agent,
            source=source,
        )
    except Exception as exc:  # pragma: no cover - degraded, never fatal
        _logger.warning(
            "voice example write failed",
            extra={"business_id": business_id, "signal": source, "reason": type(exc).__name__},
        )
        return None
    _logger.info(
        "voice_example_recorded",
        extra={"business_id": business_id, "signal": source, "memory_id": item.id},
    )
    return item.id


async def learn_from_edit(
    store: Any,
    *,
    business_id: str,
    original: str,
    revised: str,
    context: str = "",
    campaign_id: str | None = None,
    conversation_id: str | None = None,
) -> str | None:
    """Hook (c): a manual edit becomes a before/after style example."""
    if not is_meaningful_edit(original, revised):
        return None
    return await record_voice_example(
        store,
        business_id=business_id,
        text=render_edit_example(original=original, revised=revised, context=context),
        source="manual_edit",
        metadata={"has_previous": bool(original.strip()), "context": _clip(context, 200)},
        campaign_id=campaign_id,
        conversation_id=conversation_id,
    )


async def learn_from_approval(
    store: Any,
    *,
    business_id: str,
    text: str,
    brief: str = "",
    campaign_id: str | None = None,
    conversation_id: str | None = None,
    source: str = "approval",
) -> str | None:
    """Hook (b): copy the user saved or approved is a positive example."""
    return await record_voice_example(
        store,
        business_id=business_id,
        text=render_approval_example(text=text, brief=brief),
        source=source,
        metadata={"kind": "approved_copy"},
        campaign_id=campaign_id,
        conversation_id=conversation_id,
    )


async def learn_from_chat_correction(
    store: Any,
    *,
    business_id: str,
    user_message: str,
    previous_draft: str,
    campaign_id: str | None = None,
    conversation_id: str | None = None,
) -> str | None:
    """Hook (a): "أقصر" / "shorter" / "more formal" in chat.

    Only fires when a *previous Layla draft* exists to react to -- otherwise the
    message is a new request with a modifier in it, and storing it would teach the
    wrong lesson.
    """
    label = correction_intent(user_message)
    if label is None or not (previous_draft or "").strip():
        return None
    return await record_voice_example(
        store,
        business_id=business_id,
        text=render_correction_example(
            draft=previous_draft, instruction=user_message.strip(), label=label
        ),
        source="chat_correction",
        metadata={"correction": label},
        campaign_id=campaign_id,
        conversation_id=conversation_id,
    )


def last_layla_draft(messages: list[Any], *, max_chars: int = 1200) -> str:
    """The most recent assistant message that looks like a deliverable.

    Tool-call-only messages and one-line acknowledgements are skipped: teaching
    from "Sure, here you go" would be worse than teaching from nothing.
    """
    for message in reversed(messages or []):
        role = getattr(message, "type", None) or getattr(message, "role", None)
        if role not in {"ai", "assistant"}:
            continue
        if getattr(message, "tool_calls", None):
            continue
        content = getattr(message, "content", "")
        text = content if isinstance(content, str) else str(content or "")
        text = text.strip()
        if len(text) >= 80:
            return text[:max_chars]
    return ""


__all__ = [
    "MAX_EXAMPLE_CHARS",
    "MIN_EDIT_RATIO",
    "correction_intent",
    "format_examples_block",
    "is_meaningful_edit",
    "last_layla_draft",
    "learn_from_approval",
    "learn_from_chat_correction",
    "learn_from_edit",
    "record_voice_example",
    "render_approval_example",
    "render_correction_example",
    "render_edit_example",
    "select_voice_examples",
]
