from types import SimpleNamespace

from app.ai.agents.omar.schema import CampaignStrategyOutput, MarketingStrategyOutput
from app.ai.agents.orchestrator.handoff import (
    creative_handoff_payload,
    extract_tool_args,
    render_output,
    strategy_handoff_payload,
)


def test_extracts_named_submit_tool_when_other_tool_calls_precede_it():
    message = SimpleNamespace(
        tool_calls=[
            {"name": "web_search_tool", "args": {"query": "market"}},
            {"name": "submit_strategy_report", "args": {"summary": "Plan", "details": ["A"]}},
        ]
    )

    assert extract_tool_args([message], "submit_strategy_report") == {
        "summary": "Plan",
        "details": ["A"],
    }


def test_campaign_handoff_carries_strategy_but_not_content_calendar():
    state = {
        "marketing_strategy": {"positioning": "Local premium"},
        "campaign_strategy": {"objective": "traffic", "key_messages": ["Visit us"]},
        "content_calendar": {"items": [{"week": 1, "topic": "Launch"}]},
        "final_report": {"summary": "A focused plan", "details": ["Use proof"]},
    }

    payload = creative_handoff_payload(state)

    assert payload["campaign_strategy"]["objective"] == "traffic"
    assert payload["marketing_strategy"]["positioning"] == "Local premium"
    assert payload["summary"] == "A focused plan"
    assert "content_calendar" not in payload
    assert "missing_inputs" not in payload


def test_handoff_marks_missing_campaign_plan_instead_of_inventing_it():
    payload = creative_handoff_payload(
        {"marketing_strategy": {"positioning": "Known"}, "final_report": {"summary": "Strategy"}}
    )

    assert "campaign_strategy" not in payload
    assert any("did not return a structured campaign strategy" in item for item in payload["missing_inputs"])


def test_drive_strategy_models_keep_unknown_objective_and_budget_optional():
    marketing = MarketingStrategyOutput(summary="Evidence-led")
    campaign = CampaignStrategyOutput(
        summary="Draft",
        objective=None,
        missing_inputs=["Campaign objective"],
    )

    assert marketing.channels == []
    assert campaign.objective is None
    assert campaign.channels == []
    assert campaign.missing_inputs == ["Campaign objective"]


def test_render_output_handles_arabic_and_json_without_ascii_escaping():
    rendered = render_output({"summary": "إعلان للحملة", "details": ["رسالة واضحة"]})

    assert "إعلان للحملة" in rendered
    assert "\\u" not in rendered
