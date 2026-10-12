from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.ai.agents.karim.graph import KARIM_TOOLS
from app.ai.settings import get_ai_settings
from app.ai.tools import fetch_ga4_performance, fetch_ga4_performance_tool
from app.ai.tools.fetch_ga4_performance import FetchGA4PerformanceInput


def test_input_validation_success():
    """Validates proper date formats and numeric property ID."""
    inp = FetchGA4PerformanceInput(
        property_id="123456789",
        start_date="2026-01-01",
        end_date="today",
    )
    assert inp.property_id == "123456789"
    assert inp.start_date == "2026-01-01"
    assert inp.end_date == "today"


def test_input_property_prefix_strip():
    """Strips 'properties/' prefix from property_id."""
    inp = FetchGA4PerformanceInput(
        property_id="properties/987654321",
        start_date="30daysAgo",
        end_date="yesterday",
    )
    assert inp.property_id == "987654321"


def test_input_invalid_property_id():
    """Non-numeric property ID raises ValueError."""
    with pytest.raises(ValidationError):
        FetchGA4PerformanceInput(
            property_id="not_a_number",
            start_date="2026-01-01",
            end_date="2026-01-31",
        )


def test_input_invalid_date_format():
    """Invalid date format raises ValueError."""
    with pytest.raises(ValidationError):
        FetchGA4PerformanceInput(
            property_id="123456789",
            start_date="last_month",
            end_date="today",
        )


def test_tool_registered_in_karim_tools():
    """Verifies fetch_ga4_performance is registered in Karim's toolset."""
    tool_names = [t.name for t in KARIM_TOOLS]
    assert "fetch_ga4_performance" in tool_names


@pytest.mark.asyncio
async def test_fetch_ga4_performance_missing_credentials(monkeypatch):
    """When credentials are not set, returns a readable error message without raising unhandled exception."""
    monkeypatch.delenv("GA4_CLIENT_EMAIL", raising=False)
    monkeypatch.delenv("GA4_PRIVATE_KEY", raising=False)
    get_ai_settings.cache_clear()

    result = await fetch_ga4_performance.ainvoke({
        "property_id": "123456789",
        "start_date": "30daysAgo",
        "end_date": "today",
    })

    assert "GA4 request failed:" in result


@pytest.mark.asyncio
async def test_fetch_ga4_performance_no_rows():
    """When response contains no rows, returns empty data message."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.rows = []
    mock_client.run_report.return_value = mock_response

    with patch("app.ai.tools.fetch_ga4_performance._build_client", return_value=mock_client):
        result = await fetch_ga4_performance.ainvoke({
            "property_id": "123456789",
            "start_date": "2026-01-01",
            "end_date": "2026-01-31",
        })

    assert result == "No GA4 data for property 123456789 between 2026-01-01 and 2026-01-31."


@pytest.mark.asyncio
async def test_fetch_ga4_performance_success():
    """Successful report returns formatted metrics JSON."""
    mock_client = AsyncMock()
    mock_response = MagicMock()

    # Create 7 mock metric values corresponding to _METRICS:
    # sessions, totalUsers, screenPageViews, bounceRate, engagementRate, averageSessionDuration, keyEvents
    class MockMetricValue:
        def __init__(self, val: str):
            self.value = val

    mock_row = MagicMock()
    mock_row.metric_values = [
        MockMetricValue("1520"),  # sessions
        MockMetricValue("1200"),  # users
        MockMetricValue("4500"),  # page_views
        MockMetricValue("0.354"), # bounce_rate (35.4%)
        MockMetricValue("0.646"), # engagement_rate (64.6%)
        MockMetricValue("123.456"), # avg_session_duration_sec (123.5)
        MockMetricValue("85"),    # conversions
    ]
    mock_response.rows = [mock_row]
    mock_client.run_report.return_value = mock_response

    with patch("app.ai.tools.fetch_ga4_performance._build_client", return_value=mock_client):
        raw_result = await fetch_ga4_performance.ainvoke({
            "property_id": "123456789",
            "start_date": "30daysAgo",
            "end_date": "today",
        })

    data = json.loads(raw_result)
    assert data["property_id"] == "123456789"
    assert data["date_range"] == {"start": "30daysAgo", "end": "today"}

    metrics = data["metrics"]
    assert metrics["sessions"] == 1520.0
    assert metrics["users"] == 1200.0
    assert metrics["page_views"] == 4500.0
    assert metrics["bounce_rate"] == 35.4
    assert metrics["engagement_rate"] == 64.6
    assert metrics["avg_session_duration_sec"] == 123.5
    assert metrics["conversions"] == 85.0
