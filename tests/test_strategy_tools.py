from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.ai.agents.omar.graph import OMAR_TOOLS
from app.ai.tools.strategy import (
    CampaignStrategyOutput,
    ContentCalendarOutput,
    MarketingStrategyOutput,
    generate_campaign_strategy,
    generate_content_calendar,
    generate_marketing_strategy,
)
from app.ai.tools.strategy.schemas import (
    CalendarPost,
    ChannelAllocation,
    StrategyPhase,
    TargetSegment,
)


def test_strategy_tools_registered_in_omar():
    """Verify all 3 strategy tools are in Omar's tools list."""
    tool_names = [t.name for t in OMAR_TOOLS]
    assert "generate_marketing_strategy" in tool_names
    assert "generate_campaign_strategy" in tool_names
    assert "generate_content_calendar" in tool_names


def test_marketing_strategy_requires_brand_context():
    """Without brand_context in state, returns an error message."""
    cmd = generate_marketing_strategy.invoke({
        "focus": "Eco-friendly positioning",
        "tool_call_id": "call_1",
        "state": {},
    })
    assert isinstance(cmd, Command)
    msg: ToolMessage = cmd.update["messages"][0]
    assert "ERROR: call get_brand_context first." in msg.content
    assert "strategy" not in cmd.update


def test_marketing_strategy_success():
    """With brand_context in state, calls structured_call and updates state['strategy']."""
    mock_strategy = MarketingStrategyOutput(
        positioning_statement="The premium organic coffee in Cairo",
        target_segments=[
            TargetSegment(
                name="Coffee Connoisseurs",
                demographics="25-45 urban professionals",
                pain_points=["Inconsistent quality", "Artificial additives"],
                buying_triggers=["Single-origin", "Fresh roasting"],
            )
        ],
        value_propositions=["100% organic beans", "Ethically sourced"],
        messaging_pillars=["Pure Origin", "Craft Roasting"],
        competitive_advantage="Direct trade with Ethiopian farms",
        ninety_day_roadmap=[
            StrategyPhase(
                phase=1,
                name="Testing & Validation",
                duration_days=30,
                focus_goals=["Validate taste profiles"],
                key_actions=["Sample boxes to micro-influencers"],
            )
        ],
        primary_kpis=["Customer Acquisition Cost", "Repeat Purchase Rate"],
    )

    with patch("app.ai.tools.strategy.generate_marketing_strategy.structured_call", return_value=mock_strategy):
        cmd = generate_marketing_strategy.invoke({
            "focus": "Organic authenticity",
            "tool_call_id": "call_mkt",
            "state": {
                "brand_context": {"name": "Roast Cairo", "industry": "Specialty Coffee"},
                "research_notes": ["Competitor A lacks organic certification"],
            },
        })

    assert isinstance(cmd, Command)
    assert "strategy" in cmd.update
    assert cmd.update["strategy"]["positioning_statement"] == "The premium organic coffee in Cairo"
    msg = cmd.update["messages"][0]
    assert "MARKETING_STRATEGY_SAVED" in msg.content
    assert msg.tool_call_id == "call_mkt"


def test_campaign_strategy_requires_strategy():
    """Without prior marketing strategy in state, returns an error."""
    cmd = generate_campaign_strategy.invoke({
        "objective": "Conversions",
        "budget": 5000.0,
        "tool_call_id": "call_2",
        "state": {"brand_context": "Brand X"},
    })
    assert isinstance(cmd, Command)
    msg = cmd.update["messages"][0]
    assert "ERROR: call generate_marketing_strategy first." in msg.content
    assert "campaign" not in cmd.update


def test_campaign_strategy_budget_validation():
    """Validates budget share percentage sums and adds warning if not 100%."""
    # Shares sum to 80% (not 100%)
    mock_campaign_unbalanced = CampaignStrategyOutput(
        big_idea="Awaken Your Senses",
        objective="Drive Online Orders",
        channels=[
            ChannelAllocation(channel="Instagram", budget_share_pct=50.0, role="Visual discovery"),
            ChannelAllocation(channel="Facebook", budget_share_pct=30.0, role="Community retargeting"),
        ],
    )

    with patch("app.ai.tools.strategy.generate_campaign_strategy.structured_call", return_value=mock_campaign_unbalanced):
        cmd = generate_campaign_strategy.invoke({
            "objective": "Conversions",
            "budget": 10000.0,
            "tool_call_id": "call_cmp_1",
            "state": {
                "brand_context": "Brand X",
                "strategy": {"positioning": "Organic Coffee"},
            },
        })

    msg = cmd.update["messages"][0]
    assert "WARNING: channel shares sum to 80.0; regenerate to fix." in msg.content
    assert "campaign" in cmd.update

    # Balanced shares sum to 100%
    mock_campaign_balanced = CampaignStrategyOutput(
        big_idea="Awaken Your Senses",
        objective="Drive Online Orders",
        channels=[
            ChannelAllocation(channel="Instagram", budget_share_pct=60.0),
            ChannelAllocation(channel="Facebook", budget_share_pct=40.0),
        ],
    )

    with patch("app.ai.tools.strategy.generate_campaign_strategy.structured_call", return_value=mock_campaign_balanced):
        cmd_balanced = generate_campaign_strategy.invoke({
            "objective": "Conversions",
            "budget": 10000.0,
            "tool_call_id": "call_cmp_2",
            "state": {
                "brand_context": "Brand X",
                "strategy": {"positioning": "Organic Coffee"},
            },
        })

    msg_balanced = cmd_balanced.update["messages"][0]
    assert "WARNING" not in msg_balanced.content
    assert "CAMPAIGN_STRATEGY_SAVED" in msg_balanced.content


def test_content_calendar_requires_campaign():
    """Without prior campaign strategy in state, returns an error."""
    cmd = generate_content_calendar.invoke({
        "duration_days": 14,
        "tool_call_id": "call_3",
        "state": {"strategy": {"some": "data"}},
    })
    assert isinstance(cmd, Command)
    msg = cmd.update["messages"][0]
    assert "ERROR: call generate_campaign_strategy first." in msg.content
    assert "content_calendar" not in cmd.update


def test_content_calendar_success():
    """With campaign in state, produces calendar posts and updates state."""
    mock_calendar = ContentCalendarOutput(
        duration_days=7,
        themes=["Origins Week"],
        posts=[
            CalendarPost(
                day=1,
                channel="Instagram",
                content_pillar="Craft",
                format="Reel",
                concept="Behind the roasting bar",
                caption_hook="Ever wonder how artisan beans are roasted?",
                call_to_action="Swipe up to get free delivery",
            )
        ],
    )

    with patch("app.ai.tools.strategy.generate_content_calendar.structured_call", return_value=mock_calendar):
        cmd = generate_content_calendar.invoke({
            "duration_days": 7,
            "tool_call_id": "call_cal",
            "state": {
                "brand_context": "Brand X",
                "strategy": {"positioning": "Organic"},
                "campaign": {"big_idea": "Awaken Your Senses"},
            },
        })

    assert isinstance(cmd, Command)
    assert "content_calendar" in cmd.update
    assert len(cmd.update["content_calendar"]["posts"]) == 1
    msg = cmd.update["messages"][0]
    assert "CONTENT_CALENDAR_SAVED" in msg.content
    assert msg.tool_call_id == "call_cal"
