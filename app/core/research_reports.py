"""Report shaping for the research tools: one language, one skeleton.

The research tab asks a model to write a Markdown intelligence report. Two
things went wrong in practice, and both are handled here rather than hoped for
in a prompt:

**Language.** The prompt said "answer in Arabic" and the model -- especially a
small local one -- answered in English anyway, so an Arabic-speaking customer
got an English report. Prompts are advisory; this module *checks* the output and
rewrites it when the check fails, which is the only way to guarantee the
contract.

**Structure.** Each tool has a fixed set of ``##`` sections. Models like to
invent their own headings, drop the sources section, or return one wall of text,
which makes the output render inconsistently in the UI. :func:`enforce_structure`
compares the report against the tool's own prompt headings (single source of
truth), asks once for a repair, and -- if the model still refuses -- reports the
gap in ``warnings`` instead of pretending the report is well-formed.

Nothing here invents content: the language pass only translates, and the
structure pass only moves/re-titles what the model already wrote.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger

_logger = get_logger(__name__)

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_HEADING_RE = re.compile(r"^\s{0,3}(#{2,4})\s+(.+?)\s*$", re.MULTILINE)
#: A heading in the prompt template, e.g. ``## Core Offers & Value Propositions``.
_PROMPT_HEADING_RE = re.compile(r"^\s*#{2,3}\s+(.+?)\s*$", re.MULTILINE)

#: Placeholder headings in the prompt templates that are instructions, not sections.
_NON_SECTION_HEADINGS = {
    "🌐 sources & references",
    "🌐 المصادر والمراجع",
}

SOURCES_HEADING = {"ar": "🌐 المصادر والمراجع", "en": "🌐 Sources & References"}


# ------------------------------------------------------------------ language --


def resolve_output_language(
    requested: str | None,
    *,
    workspace_language: str | None = None,
    query: str = "",
    default: str = "en",
) -> str:
    """Decide the report language: explicit request, then the workspace, then the text.

    An Arabic query is a strong signal even when the client sent no language, and
    a workspace whose saved profile says ``ar`` should not silently get English
    reports just because the browser cookie said ``en``.
    """
    candidate = (requested or "").strip().lower()[:2]
    if candidate in {"ar", "en"}:
        return candidate
    workspace_lang = (workspace_language or "").strip().lower()[:2]
    if workspace_lang in {"ar", "en"}:
        return workspace_lang
    if query and _ARABIC_RE.search(query):
        return "ar"
    return default


def arabic_ratio(text: str) -> float:
    """Share of letters in ``text`` that are Arabic. 0.0 for empty text."""
    arabic = len(_ARABIC_RE.findall(text or ""))
    latin = len(_LATIN_RE.findall(text or ""))
    total = arabic + latin
    if total == 0:
        return 0.0
    return arabic / total


def language_matches(text: str, language: str) -> bool:
    """True when ``text`` is (mostly) in ``language``.

    The thresholds are deliberately loose: a technical Arabic report legitimately
    carries Latin brand names, URLs and numbers, so anything above 15% Arabic
    letters counts as an Arabic report.
    """
    ratio = arabic_ratio(text)
    if language == "ar":
        return ratio >= 0.15
    return ratio <= 0.60


# ----------------------------------------------------------------- structure --


def required_headings(prompt_template: str) -> list[str]:
    """The section titles a tool's own prompt demands, in order.

    Derived from the prompt so the prompt stays the single source of truth --
    editing a heading there changes the contract here with no second list to
    keep in sync.
    """
    headings: list[str] = []
    for raw in _PROMPT_HEADING_RE.findall(prompt_template or ""):
        title = raw.strip().rstrip(":").strip()
        if not title:
            continue
        if title.lower() in _NON_SECTION_HEADINGS or title in _NON_SECTION_HEADINGS:
            continue
        # Skip prose that merely starts with '#' inside an instruction line.
        if len(title) > 90 or title.endswith("?"):
            continue
        headings.append(title)
    return headings


def _normalise_heading(text: str) -> str:
    """Fold a heading to a comparable key (case, punctuation, emoji insensitive)."""
    stripped = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text or "", flags=re.UNICODE)
    return re.sub(r"\s+", " ", stripped).strip().casefold()


def extract_headings(text: str) -> list[str]:
    return [match.group(2).strip() for match in _HEADING_RE.finditer(text or "")]


def _heading_present(heading: str, present: Iterable[str]) -> bool:
    """Fuzzy match, so a model that re-titles "Core Offers & Value Propositions"
    as "Core Offers and Value Propositions" is not asked to rewrite the report."""
    key = _normalise_heading(heading)
    if not key:
        return True
    tokens = [token for token in key.split(" ") if len(token) > 3]
    for candidate in present:
        candidate_key = _normalise_heading(candidate)
        if key in candidate_key or candidate_key in key:
            return True
        if tokens and sum(1 for token in tokens if token in candidate_key) >= max(
            1, len(tokens) - 1
        ):
            return True
    return False


def missing_headings(text: str, headings: list[str]) -> list[str]:
    present = extract_headings(text)
    return [heading for heading in headings if not _heading_present(heading, present)]


def split_sections(text: str) -> list[dict[str, str]]:
    """Markdown -> ``[{title, body}]`` so a structured client can render sections.

    Text before the first heading is returned as an untitled lead section, which
    is what the typo-correction banner is.
    """
    sections: list[dict[str, str]] = []
    matches = list(_HEADING_RE.finditer(text or ""))
    if not matches:
        body = (text or "").strip()
        return [{"title": "", "body": body}] if body else []

    lead = (text or "")[: matches[0].start()].strip()
    if lead:
        sections.append({"title": "", "body": lead})
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(
            {"title": match.group(2).strip(), "body": (text or "")[start:end].strip()}
        )
    return sections


# ------------------------------------------------------------------- repairs --


def _plain_text(content: Any) -> str:
    """Flatten a LangChain message content payload to plain text."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "\n".join(part for part in parts if part.strip())
    if isinstance(content, dict):
        return str(content.get("text") or "")
    return str(content)


async def _invoke(llm: Any, system: str, user: str) -> str:
    result = await asyncio.to_thread(
        llm.invoke, [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return _plain_text(getattr(result, "content", result))


async def enforce_language(text: str, *, language: str, llm: Any) -> tuple[str, bool]:
    """Rewrite ``text`` in ``language`` when the model ignored the instruction.

    Returns ``(text, rewritten)``. A failed rewrite returns the original text
    rather than an empty report -- a wrong-language report still beats no report.
    """
    if not text.strip() or language_matches(text, language):
        return text, False

    if language == "ar":
        system = (
            "أنت محرر عربي محترف. أعد كتابة التقرير التالي بالكامل باللغة العربية الفصحى.\n"
            "قواعد صارمة:\n"
            "1) احتفظ بكل الأرقام، الأسماء التجارية، الروابط، وبنية العناوين (##) كما هي تمامًا.\n"
            "2) لا تضف أي معلومة أو تحليل جديد، ولا تحذف أي معلومة موجودة.\n"
            "3) لا تكتب أي جملة افتتاحية أو خاتمة إضافية، ولا أي نص بالإنجليزية عدا الأسماء والروابط.\n"
            "4) أعد التقرير فقط بدون أي تعليق."
        )
    else:
        system = (
            "You are a professional editor. Rewrite the following report entirely in English.\n"
            "Strict rules:\n"
            "1) Keep every number, brand name, URL and heading structure (##) exactly as-is.\n"
            "2) Do not add or remove any information.\n"
            "3) No preamble, no closing remarks, no commentary -- return the report only."
        )

    try:
        rewritten = await _invoke(llm, system, text)
    except Exception as exc:  # pragma: no cover - provider dependent
        _logger.warning("research_language_repair_failed", extra={"error": str(exc)})
        return text, False

    if rewritten.strip() and language_matches(rewritten, language):
        return rewritten.strip(), True
    _logger.warning(
        "research_language_repair_ineffective",
        extra={"language": language, "ratio": arabic_ratio(rewritten)},
    )
    return text, False


async def enforce_structure(
    text: str,
    *,
    headings: list[str],
    language: str,
    llm: Any,
    attempts: int = 1,
) -> tuple[str, list[str]]:
    """Make ``text`` carry the tool's required sections, or say which are missing.

    Returns ``(text, missing)``. A repair is attempted ``attempts`` times; the
    remaining gap is reported so the UI can flag an incomplete report instead of
    rendering something that merely looks structured.
    """
    if not headings:
        return text, []

    missing = missing_headings(text, headings)
    if not missing:
        return text, []

    sources_heading = SOURCES_HEADING.get(language, SOURCES_HEADING["en"])
    skeleton = "\n".join(f"## {heading}" for heading in headings)
    if language == "ar":
        system = (
            "أنت منسّق تقارير صارم. أعد ترتيب التقرير التالي داخل الأقسام المطلوبة بالضبط.\n"
            "قواعد صارمة:\n"
            f"1) استخدم هذه العناوين بالحرف وبالترتيب: \n{skeleton}\n"
            "2) انقل نص التقرير كما هو إلى القسم المناسب؛ لا تكتب معلومة جديدة ولا تحذف معلومة.\n"
            "3) إن كان قسم بلا محتوى فأبقِ العنوان واكتب تحته «لا توجد بيانات كافية».\n"
            "4) أبقِ قسم المصادر في النهاية بعنوان "
            f"«{sources_heading}» مع روابط ماركداون.\n"
            "5) لا تكتب أي شيء قبل العنوان الأول، ولا أي تعليق خارج التقرير."
        )
    else:
        system = (
            "You are a strict report formatter. Re-organise the following report into the exact "
            "required sections.\n"
            "Strict rules:\n"
            f"1) Use these headings verbatim, in this order:\n{skeleton}\n"
            "2) Move the existing text into the right section; never add or delete information.\n"
            "3) If a section has no content, keep the heading and write 'Insufficient data' "
            "underneath it.\n"
            f"4) Keep the sources section last, titled '{sources_heading}', with markdown links.\n"
            "5) Output the report only -- no preamble, no commentary."
        )

    for _ in range(max(1, attempts)):
        try:
            repaired = await _invoke(llm, system, text)
        except Exception as exc:  # pragma: no cover - provider dependent
            _logger.warning("research_structure_repair_failed", extra={"error": str(exc)})
            break
        if repaired.strip() and len(missing_headings(repaired, headings)) < len(missing):
            text = repaired.strip()
            missing = missing_headings(text, headings)
            if not missing:
                return text, []

    return text, missing


async def shape_report(
    text: str,
    *,
    language: str,
    headings: list[str],
    llm: Any,
    sources: list[dict[str, str]] | None = None,
) -> tuple[str, list[str]]:
    """Apply both guarantees, language first, then structure.

    Language runs first because the structure repair is itself a prompt: asking
    for Arabic sections while the body is English produced a half-translated
    report. Once the body is Arabic, the section pass is a pure re-ordering.
    """
    text, _ = await enforce_language(text, language=language, llm=llm)
    text, missing = await enforce_structure(
        text, headings=headings, language=language, llm=llm, attempts=2
    )
    text = ensure_sources_section(text, language=language, sources=sources or [])
    return text, missing


def ensure_sources_section(
    text: str, *, language: str, sources: list[dict[str, str]]
) -> str:
    """Guarantee the report ends with a sources section when sources exist.

    Deterministic (no model involved): a report without its citations is not
    usable as evidence, and the list is already known server-side.
    """
    links = [s for s in sources if s.get("url")]
    if not links:
        return text
    markdown = "\n".join(f"- [{s.get('title') or s['url']}]({s['url']})" for s in links)
    lowered = (text or "").casefold()
    if "المصادر" in text or "sources" in lowered:
        return text
    heading = SOURCES_HEADING.get(language, SOURCES_HEADING["en"])
    return f"{text.rstrip()}\n\n## {heading}\n{markdown}\n"


__all__ = [
    "SOURCES_HEADING",
    "arabic_ratio",
    "enforce_language",
    "enforce_structure",
    "ensure_sources_section",
    "extract_headings",
    "language_matches",
    "missing_headings",
    "required_headings",
    "resolve_output_language",
    "shape_report",
    "split_sections",
]
