from __future__ import annotations

from app.ai.agents.orchestrator.presentation import (
    format_agent_report,
    format_campaign_response,
)


def test_campaign_response_is_human_readable_and_flags_unverified_claims() -> None:
    omar = {
        "summary": "Strategy draft",
        "missing_inputs": ["Monthly bookings target"],
        "marketing_strategy": {
            "summary": "Attract MG owners in Egypt.",
            "positioning": "A specialist service center.",
            "target_segments": ["MG owners"],
            "channels": [
                {
                    "channel": "Meta Ads",
                    "budget_share_pct": 60,
                    "role": "Lead generation",
                    "placements": ["Facebook Feed"],
                }
            ],
            "kpis": [{"metric": "CPL", "target": None}],
            "evidence": ["Name, website, market, and monthly budget from Settings."],
        },
        "campaign_strategy": {
            "objective": "Increase service inquiries.",
            "key_messages": ["Warranty on services"],
            "missing_inputs": ["CPL target"],
        },
    }
    layla = {
        "headline": "MG needs service?",
        "primary_text": "Get a real guarantee and genuine parts, cheaper than the dealer.",
        "call_to_action": "Book now",
        "aspect_ratio": "1:1",
    }

    text = format_campaign_response(omar, layla, arabic=True)

    assert "## خطة الحملة الإعلانية" in text
    assert "### الاستراتيجية التسويقية — عمر" in text
    assert "### مسودة الإعلان — ليلى" in text
    assert "حصة مقترحة: 60%" in text
    assert "المستهدف غير محدد" in text
    assert "CPL target" in text
    assert "تحقق من ادعاء الضمان" in text
    assert "تحقق من ادعاء أصالة قطع الغيار" in text
    assert "لم يتم إنشاء أو نشر حملة" in text
    assert '"summary"' not in text
    assert text.count("Name, website, market, and monthly budget from Settings.") == 1


def test_research_report_includes_competitor_findings_and_source_link() -> None:
    report = {
        "market_overview": "The category is competitive.",
        "competitor_summary": [
            {
                "brand": "Example Garage",
                "positioning": "Specializes in MG repair.",
                "strengths": ["Publishes service details."],
                "weaknesses_or_gaps": ["No warranty details were visible on the reviewed page."],
                "evidence": [
                    {
                        "claim": "Service list",
                        "source_url": "https://example.test/services",
                        "observed_text": "MG repairs",
                    }
                ],
            }
        ],
        "missing_inputs": ["Review data was unavailable."],
    }

    text = format_agent_report("nour", report, arabic=True)

    assert "## تحليل السوق والمنافسين" in text
    assert "Example Garage" in text
    assert "نقاط الضعف أو الفجوات المرصودة" in text
    assert "[المصدر](https://example.test/services)" in text
    assert "Review data was unavailable." in text
    assert '"competitor_summary"' not in text


def test_plain_text_agent_output_is_preserved() -> None:
    assert format_agent_report("omar", "تعذر إكمال الخطة الآن.", arabic=True) == "تعذر إكمال الخطة الآن."
