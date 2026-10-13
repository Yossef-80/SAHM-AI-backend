"""Research reports leave the backend in one language and one shape.

Two complaints from the Arabic workspace, both encoded here:

* the report came back in English even though the tab and the query were Arabic
  (a prompt asking for Arabic is advisory — the model ignored it);
* different tools returned differently-shaped reports, so the reading pane
  changed structure between runs.

These are the pure, model-free guarantees: language resolution, heading
extraction from the tool's own prompt, gap detection and the deterministic
sources section. The LLM-backed repair passes are exercised through the API in
tests that use the mock provider, but the contract is asserted here so a
regression cannot hide behind "the model happened to comply".
"""

from __future__ import annotations

from app.core.research_reports import (
    SOURCES_HEADING,
    ensure_sources_section,
    extract_headings,
    missing_headings,
    required_headings,
    resolve_output_language,
    split_sections,
)

PROMPT_AR = """اكتب تقريرًا استخباراتيًا عن المنافس.

## الملخص التنفيذي
فقرة قصيرة.

## التموضع والرسالة
نقاط.

## العروض والأسعار
نقاط.

## المصادر
روابط.
"""

REPORT_AR = """## الملخص التنفيذي
المنافس يستهدف القاهرة.

## التموضع والرسالة
- رسالة مباشرة
- تركيز على السعر
"""


def test_requested_language_wins() -> None:
    assert resolve_output_language("ar", workspace_language="en", query="analytics") == "ar"
    assert resolve_output_language("en", workspace_language="ar", query="تحليل") == "en"


def test_workspace_language_beats_the_default() -> None:
    """A workspace saved as Arabic must not get English because a cookie said so."""
    assert resolve_output_language(None, workspace_language="ar", query="") == "ar"


def test_arabic_query_is_a_signal() -> None:
    assert resolve_output_language(None, query="حلل منافسي في السوق") == "ar"
    assert resolve_output_language(None, query="analyse my competitor") == "en"


def test_required_headings_come_from_the_tool_prompt() -> None:
    headings = required_headings(PROMPT_AR)
    assert headings == ["الملخص التنفيذي", "التموضع والرسالة", "العروض والأسعار", "المصادر"]


def test_missing_headings_reports_the_gap_without_inventing() -> None:
    headings = required_headings(PROMPT_AR)
    assert missing_headings(REPORT_AR, headings) == ["العروض والأسعار", "المصادر"]
    assert extract_headings(REPORT_AR) == ["الملخص التنفيذي", "التموضع والرسالة"]


def test_split_sections_gives_the_client_stable_blocks() -> None:
    sections = split_sections(REPORT_AR)
    assert [section["title"] for section in sections] == ["الملخص التنفيذي", "التموضع والرسالة"]
    assert "القاهرة" in sections[0]["body"]

    # A wall of text still yields one renderable block rather than nothing.
    assert split_sections("نص بدون عناوين") == [{"title": "", "body": "نص بدون عناوين"}]


def test_sources_section_is_appended_deterministically() -> None:
    sources = [
        {"title": "تقرير السوق", "url": "https://example.com/market"},
        {"title": "", "url": "https://example.com/second"},
    ]
    shaped = ensure_sources_section(REPORT_AR, language="ar", sources=sources)
    assert SOURCES_HEADING["ar"] in shaped
    assert "https://example.com/market" in shaped

    # Already-cited report is left alone; the function never duplicates a section.
    already = shaped
    assert ensure_sources_section(already, language="ar", sources=sources) == already

    # No links -> nothing to add, and no empty heading either.
    assert ensure_sources_section(REPORT_AR, language="ar", sources=[{"title": "x", "url": ""}]) == REPORT_AR
