"""Unit tests for the competitor-comparison endpoint's pure pieces.

These cover the parts that fail silently in production: the model reply that
arrives wrapped in prose or fences, the schema that decides whether we accept
it, and the rendering that the Research page and the prompt both depend on.
The HTTP path itself needs a live search provider, so it is covered by
`tests/test_contract.py`-style checks elsewhere rather than here.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.api.research_compare import (
    DIMENSIONS,
    CompetitorComparison,
    _clean_keywords,
    _dedupe_hits,
    _markdown_summary,
    _sources_block,
    _strip_json,
)
from app.ai.agents.orchestrator.brain import _research_item_message

VALID = {
    "summary": "Both compete on price; A is stronger on retention.",
    "dimensions": [
        {
            "dimension": "positioning",
            "competitor_a": "premium sports",
            "competitor_b": "value sports",
            "verdict": "a_leads",
            "evidence": ["a.com home page"],
        }
    ],
    "gaps": [
        {
            "title": "No loyalty programme at B",
            "why_it_matters": "Repeat purchase drives margin",
            "who_has_it": "a",
            "evidence": ["a.com/rewards"],
            "confidence": "high",
        }
    ],
    "opportunities": [
        {
            "title": "Launch a referral loop",
            "rationale": "Neither brand rewards advocacy",
            "effort": "medium",
            "impact": "high",
            "first_step": "Measure current referral share",
            "evidence": [],
        }
    ],
    "recommended_focus": ["Own retention", "Keep price parity"],
}


class TestStripJson:
    def test_plain_object(self):
        assert json.loads(_strip_json(json.dumps(VALID))) == VALID

    def test_fenced(self):
        raw = "```json\n" + json.dumps(VALID) + "\n```"
        assert json.loads(_strip_json(raw)) == VALID

    def test_prose_around_object(self):
        raw = "Sure! Here is the comparison:\n" + json.dumps(VALID) + "\nHope that helps."
        assert json.loads(_strip_json(raw)) == VALID

    def test_object_with_nested_braces(self):
        payload = {"summary": "x", "meta": {"a": {"b": 1}}}
        assert json.loads(_strip_json(json.dumps(payload))) == payload

    def test_garbage_is_returned_unchanged(self):
        assert _strip_json("no json here") == "no json here"


class TestSchema:
    def test_valid_payload_is_accepted(self):
        comparison = CompetitorComparison.model_validate(VALID)
        assert comparison.dimensions[0].verdict == "a_leads"
        assert comparison.gaps[0].confidence == "high"

    def test_unknown_verdict_is_rejected(self):
        broken = json.loads(json.dumps(VALID))
        broken["dimensions"][0]["verdict"] = "definitely"
        with pytest.raises(ValidationError):
            CompetitorComparison.model_validate(broken)

    def test_missing_summary_is_rejected(self):
        broken = json.loads(json.dumps(VALID))
        broken.pop("summary")
        with pytest.raises(ValidationError):
            CompetitorComparison.model_validate(broken)

    def test_empty_lists_are_allowed(self):
        comparison = CompetitorComparison.model_validate({"summary": "only a summary"})
        assert comparison.gaps == []
        assert comparison.dimensions == []


class TestRendering:
    def test_markdown_contains_every_section(self):
        comparison = CompetitorComparison.model_validate(VALID)
        text = _markdown_summary(comparison, "Acme", "Bolt")
        assert "## Acme vs Bolt" in text
        assert "### Dimensions" in text
        assert "### Gaps" in text
        assert "### Opportunities" in text
        assert "### Recommended focus" in text
        # The verdict is rendered as a human sentence, not the raw enum.
        assert "Acme leads" in text
        assert "a_leads" not in text

    def test_markdown_includes_sources_when_provided(self):
        comparison = CompetitorComparison.model_validate(VALID)
        sources_a = [{"title": "Acme Store", "url": "https://acme.com/store"}]
        sources_b = [{"title": "Bolt Shoes", "url": "https://bolt.com/shop"}]
        text = _markdown_summary(comparison, "Acme", "Bolt", sources_a=sources_a, sources_b=sources_b)
        assert "### Verified Sources & Evidence" in text
        assert "[Acme Store](https://acme.com/store)" in text
        assert "[Bolt Shoes](https://bolt.com/shop)" in text

    def test_sources_block_lists_urls_and_handles_empty(self):
        block = _sources_block(
            [{"title": "Home", "url": "https://a.com", "snippet": "shoes"}], "A — Acme"
        )
        assert "https://a.com" in block
        assert "### A — Acme" in block
        assert "no results retrieved" in _sources_block([], "B — Bolt")

    def test_dimensions_are_the_documented_eight(self):
        assert len(DIMENSIONS) == 8
        assert "positioning" in DIMENSIONS
        assert "funnel_entry_points" in DIMENSIONS


class TestSearchHelpers:
    def test_clean_keywords_filters_arabic_and_english_stopwords(self):
        arabic_raw = "السوق المصري قارنلي الاسعار و الخامات و التقفيل و الاستور في مصر"
        cleaned = _clean_keywords(arabic_raw)
        assert "الاسعار" in cleaned
        assert "الخامات" in cleaned
        assert "قارنلي" not in cleaned
        assert "السوق" not in cleaned

        english_raw = "compare between the prices and store locations please"
        cleaned_en = _clean_keywords(english_raw)
        assert "prices" in cleaned_en
        assert "locations" in cleaned_en
        assert "compare" not in cleaned_en

    def test_dedupe_hits_removes_duplicate_urls_and_titles(self):
        hits = [
            {"title": "Page 1", "url": "https://example.com/item/"},
            {"title": "Page 1 duplicate", "url": "https://example.com/item"},
            {"title": "Page 2", "url": "https://example.com/item2"},
            {"title": "Unique Title", "url": ""},
            {"title": "Unique Title", "url": ""},
        ]
        deduped = _dedupe_hits(hits)
        assert len(deduped) == 3
        assert deduped[0]["url"] == "https://example.com/item/"
        assert deduped[1]["url"] == "https://example.com/item2"
        assert deduped[2]["title"] == "Unique Title"


class TestResearchItemMessage:
    def test_block_quotes_the_report_and_sources(self):
        context = {
            "research_item_context": {
                "id": "mem_1",
                "kind": "competitor_note",
                "title": "Acme competitive teardown",
                "query": "Acme",
                "tool": "analyze_competitor",
                "report": "Acme leads on price but has no loyalty programme.",
                "details": {
                    "subject": "Acme",
                    "verification_status": "verified",
                    "sources": [{"title": "Acme", "url": "https://acme.example"}],
                },
            }
        }
        message = _research_item_message(context)
        assert message is not None
        assert "[RESEARCH_RESULT_FROM_THE_RESEARCH_PAGE]" in message
        assert "no loyalty programme" in message
        assert "https://acme.example" in message
        # The instruction that stops the model from re-running the research.
        assert "do not re-run" in message.lower()

    def test_empty_report_produces_no_block(self):
        assert _research_item_message({"research_item_context": {"report": "   "}}) is None
        assert _research_item_message({}) is None
        assert _research_item_message({"research_item_context": "not a dict"}) is None


class TestSearchNormalization:
    @pytest.mark.asyncio
    async def test_normalizes_pydantic_search_results(self):
        from app.ai.providers.search.base import SearchResult
        from app.api.research_compare import _search

        class FakeProvider:
            def search(self, query: str, max_results: int = 5):
                return [
                    SearchResult(title="Acme Coffee", url="https://acme.com", snippet="Best beans in Cairo"),
                    {"title": "Dict Coffee", "url": "https://dict.com", "snippet": "Dict snippet"},
                ]

        hits = await _search(FakeProvider(), "Acme Coffee")
        assert len(hits) == 2
        assert isinstance(hits[0], dict)
        assert hits[0]["title"] == "Acme Coffee"
        assert hits[0]["url"] == "https://acme.com"
        assert isinstance(hits[1], dict)
        assert hits[1]["title"] == "Dict Coffee"

