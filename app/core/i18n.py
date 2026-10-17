"""The language of a user-facing reply — one resolver, used everywhere.

The rule we ship
----------------
**The application language the user picked wins.** If the UI says Arabic, the
assistant answers in Arabic, even when the user pastes an English article. If the
UI says English, it answers in English, even when the user types Arabic. The
choice is re-read on **every request**, so changing the UI language mid
conversation changes the very next answer.

Fallbacks, in order (only used when the client sends no language at all):

1. the language stored on the conversation (the last explicit choice),
2. **an Arabic project language** — Arabic is never the default value, so a
   workspace configured as Arabic is a deliberate choice and outranks the text,
3. the language the message is written in,
4. an English project language / ``en``.

The asymmetry in (2)/(4) is deliberate. ``businesses.language`` defaults to
``"en"`` on every row, so "English" is indistinguishable from "never chosen";
letting it outrank detection would give Arabic replies-in-English to every tenant
whose client does not send a language. ``"ar"`` can only have been chosen.

Why detection is *not* first
----------------------------
Before this module the reply language was inferred from the user's text
(``_contains_arabic``). That produced the classic bilingual bug: an Arabic-first
user pastes one English link, gets an English answer, and the assistant "forgets"
the language it was speaking. It also meant a tenant whose project language is
Arabic got English replies whenever the first message was short and Latin-script.
Explicit request > conversation > content, never the other way around.

Two helpers exist because there are genuinely two different questions:

* :func:`normalise_language` — "is this a language we support?" (input hygiene;
  accepts ``ar-EG``, ``AR``, ``العربية``, ``English``...);
* :func:`detect_language` — "what is this text written in?" (only used when the
  client told us nothing).

Nothing here is model-based: it runs in microseconds, on every turn.
"""

from __future__ import annotations

import re
from typing import Any, Literal

LanguageCode = Literal["ar", "en"]

SUPPORTED_LANGUAGES: tuple[LanguageCode, ...] = ("ar", "en")
DEFAULT_LANGUAGE: LanguageCode = "en"

#: Spelling/region aliases an API client or a settings form may send.
_ALIASES: dict[str, LanguageCode] = {
    "ar": "ar",
    "ara": "ar",
    "arabic": "ar",
    "ar-eg": "ar",
    "ar_eg": "ar",
    "ar-sa": "ar",
    "ar_sa": "ar",
    "ar-ae": "ar",
    "ar_ae": "ar",
    "العربية": "ar",
    "عربي": "ar",
    "عربى": "ar",
    "en": "en",
    "eng": "en",
    "english": "en",
    "en-us": "en",
    "en_us": "en",
    "en-gb": "en",
    "en_gb": "en",
    "إنجليزية": "en",
    "انجليزية": "en",
}

#: Arabic block (0600–06FF) plus Arabic presentation forms and the
#: Arabic-Indic digits. Deliberately broad: any strong Arabic signal is enough.
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def normalise_language(value: Any) -> LanguageCode | None:
    """Return a supported code for ``value`` or ``None`` when it is unknown.

    ``"ar-EG"``, ``"AR"``, ``"Arabic"`` and ``"العربية"`` all become ``"ar"``.
    An empty string, ``"auto"`` or ``None`` become ``None`` (meaning "no explicit
    choice"), which is different from "English".
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"auto", "default", "system", "none", "null"}:
        return None
    if text in _ALIASES:
        return _ALIASES[text]
    # ``ar-EG`` / ``ar_EG`` / ``ar-anything``
    base = re.split(r"[-_]", text, maxsplit=1)[0]
    return _ALIASES.get(base)


def is_arabic_text(value: str) -> bool:
    """True when the text carries at least one Arabic character."""
    return bool(_ARABIC_RE.search(value or ""))


def detect_language(value: str) -> LanguageCode | None:
    """Guess the language of ``value``; ``None`` when it is inconclusive.

    Arabic wins on a single Arabic letter even inside a mostly-English sentence:
    an Arabic-first user typing a mixed message is asking in Arabic.
    """
    text = value or ""
    if is_arabic_text(text):
        return "ar"
    if _LATIN_RE.search(text):
        return "en"
    return None


def resolve_reply_language(
    *,
    requested: Any = None,
    conversation: Any = None,
    project: Any = None,
    prompt: str = "",
    default: LanguageCode = DEFAULT_LANGUAGE,
) -> LanguageCode:
    """Decide the language of this turn's user-facing reply.

    ``requested`` is the language the UI currently shows (per request),
    ``conversation`` the last explicit choice stored with the conversation, and
    ``project`` the workspace's configured language. Only ``requested`` forces a
    language against the content of the message.
    """
    explicit = normalise_language(requested)
    if explicit:
        return explicit

    remembered = normalise_language(conversation)
    if remembered:
        return remembered

    configured = normalise_language(project)
    if configured == "ar":
        # "ar" is never a default value: the workspace chose Arabic.
        return "ar"

    detected = detect_language(prompt)
    if detected:
        return detected

    if configured:
        return configured

    return default


def language_name(code: Any) -> str:
    """``"ar"`` -> ``"Arabic"`` (for prompts); unknown values default to English."""
    return "Arabic" if normalise_language(code) == "ar" else "English"


def reply_language_directive(code: Any, *, source: str = "the application") -> str:
    """The system instruction that pins the language of every user-facing string.

    Written to survive a model that likes to mirror the user's language, which is
    the behaviour that caused the bug this module exists to fix.
    """
    target = language_name(code)
    other = "English" if target == "Arabic" else "Arabic"
    return (
        f"[RESPONSE_LANGUAGE]\n"
        f"The user chose {target} as {source} language. Write every user-facing "
        f"sentence in {target}. This holds even if the user writes in {other}, "
        f"pastes {other} text, or if earlier turns in this conversation were in "
        f"{other}. Keep brand names, product names, code, numbers, URLs and quoted "
        f"evidence exactly as they are; do not translate them. Structured JSON keys "
        f"stay in English exactly as the schema requires."
    )


#: Drop-in sentence for specialist prompts. They used to say "respond in the
#: user's language", which made each sub-agent decide for itself -- and a model
#: reading an English message from an Arabic-UI user happily switched to English.
SUBAGENT_LANGUAGE_RULE = (
    "Write every user-facing sentence in the language named by the "
    "[RESPONSE_LANGUAGE] block in this conversation. That block is authoritative "
    "even when the user writes in, or pastes text in, a different language. Only "
    "when no such block is present, answer in the language of the user's message. "
    "Keep brand names, product names, code, numbers, URLs and quoted evidence in "
    "their original form, and keep JSON keys in English."
)


__all__ = [
    "DEFAULT_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "LanguageCode",
    "detect_language",
    "is_arabic_text",
    "language_name",
    "normalise_language",
    "SUBAGENT_LANGUAGE_RULE",
    "reply_language_directive",
    "resolve_reply_language",
]
