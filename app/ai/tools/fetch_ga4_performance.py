"""GA4 Performance Fetcher (read-only tool for agent `karim`).

Auth uses a service account passed through environment variables
(GA4_CLIENT_EMAIL + GA4_PRIVATE_KEY). No JSON key file is used anywhere.
"""
from __future__ import annotations

import json
import re

from google.analytics.data_v1beta import BetaAnalyticsDataAsyncClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Metric,
    RunReportRequest,
)
from google.oauth2 import service_account
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from app.ai.settings import get_ai_settings, settings

_SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}|today|yesterday|\d+daysAgo)$")

# GA4 metric name -> friendly key in the output
_METRICS = {
    "sessions": "sessions",
    "totalUsers": "users",
    "screenPageViews": "page_views",
    "bounceRate": "bounce_rate",
    "engagementRate": "engagement_rate",
    "averageSessionDuration": "avg_session_duration_sec",
    "keyEvents": "conversions",
}


class FetchGA4PerformanceInput(BaseModel):
    property_id: str = Field(
        description="The brand's GA4 property ID (numbers only, e.g. '123456789')."
    )
    start_date: str = Field(
        description="Start date: 'YYYY-MM-DD', 'yesterday', or 'NdaysAgo' (e.g. '30daysAgo')."
    )
    end_date: str = Field(
        description="End date: 'YYYY-MM-DD', 'today', 'yesterday', or 'NdaysAgo'."
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def _check_date(cls, v: str) -> str:
        if not _DATE_RE.match(v):
            raise ValueError("Use YYYY-MM-DD, today, yesterday, or NdaysAgo.")
        return v

    @field_validator("property_id")
    @classmethod
    def _check_property(cls, v: str) -> str:
        v = v.replace("properties/", "").strip()
        if not v.isdigit():
            raise ValueError("property_id must be numeric.")
        return v


def _build_client() -> BetaAnalyticsDataAsyncClient:
    current_settings = get_ai_settings()
    client_email = current_settings.GA4_CLIENT_EMAIL or current_settings.ga4_client_email
    private_key = current_settings.GA4_PRIVATE_KEY or current_settings.ga4_private_key
    if not client_email or not private_key:
        raise ValueError("GA4_CLIENT_EMAIL and GA4_PRIVATE_KEY must be configured in environment variables.")

    info = {
        "type": "service_account",
        "client_email": client_email,
        # .env stores newlines as literal \n, so restore them
        "private_key": private_key.replace("\\n", "\n"),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    creds = service_account.Credentials.from_service_account_info(info, scopes=_SCOPES)
    return BetaAnalyticsDataAsyncClient(credentials=creds)


@tool(args_schema=FetchGA4PerformanceInput)
async def fetch_ga4_performance(property_id: str, start_date: str, end_date: str) -> str:
    """Fetches website traffic, bounce rate, engagement and conversion metrics
    from a brand's Google Analytics 4 property for a date range. Read-only."""
    try:
        client = _build_client()
        response = await client.run_report(
            RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                metrics=[Metric(name=name) for name in _METRICS],
            )
        )
    except Exception as exc:  # let the agent see a readable error
        return f"GA4 request failed: {exc}"

    if not response.rows:
        return f"No GA4 data for property {property_id} between {start_date} and {end_date}."

    values = response.rows[0].metric_values
    result = {
        friendly: float(values[i].value)
        for i, friendly in enumerate(_METRICS.values())
    }
    # GA4 returns rates as 0-1; convert to percentages for readability
    result["bounce_rate"] = round(result["bounce_rate"] * 100, 2)
    result["engagement_rate"] = round(result["engagement_rate"] * 100, 2)
    result["avg_session_duration_sec"] = round(result["avg_session_duration_sec"], 1)

    return json.dumps(
        {
            "property_id": property_id,
            "date_range": {"start": start_date, "end": end_date},
            "metrics": result,
        }
    )


# Alias matching convention of other tools
fetch_ga4_performance_tool = fetch_ga4_performance
