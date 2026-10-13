"""Safe, evidence-first composite research helpers for Nour.

This module intentionally does not import ``mcp_servers`` or read arbitrary
server paths. Web access is delegated to the bounded ``app.ai`` tools; uploaded
file analysis accepts content explicitly supplied by the caller.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.ai.security import redact_sensitive_data
from app.ai.tools.analyze_website import analyze_website_tool
from app.ai.tools.search import web_search_tool


class BrandReportInput(BaseModel):
    brand: str = Field(min_length=1, max_length=200)
    url: str = Field(default="", max_length=2000)
    country: str = Field(default="Egypt", max_length=100)
    country_code: str = Field(default="EG", min_length=2, max_length=2)


class FullAnalysisInput(BaseModel):
    brand: str = Field(min_length=1, max_length=200)
    report_data: str = Field(
        min_length=2,
        max_length=30000,
        description="JSON evidence returned by brand_report; no implicit or hidden file state is used.",
    )


class RouteFileInput(BaseModel):
    filename: str = Field(default="upload.json", max_length=240)
    file_content: str | None = Field(default=None, max_length=2_000_000)
    # Kept only as a migration hint for old prompts. It is never opened/read.
    file_path: str | None = Field(default=None, max_length=1000)
    brand: str = Field(default="", max_length=200)


def _parse_tool_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        except (TypeError, ValueError):
            return {"status": "partial", "text": value[:12000]}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {"status": "partial", "text": str(value)[:12000]}


@tool(args_schema=BrandReportInput)
async def brand_report(
    brand: str,
    url: str = "",
    country: str = "Egypt",
    country_code: str = "EG",
    config: RunnableConfig | None = None,
) -> str:
    """Gather one web-search batch and, when supplied, a bounded website audit.

    This is a convenience wrapper around the evidence-preserving ``web_search``
    and ``analyze_website`` tools. It does not scrape social networks, fetch
    reviews, access hidden MCP state, or infer unsupported claims.
    """
    safe_brand = str(redact_sensitive_data(brand)).strip()
    safe_country = str(redact_sensitive_data(country)).strip()
    child_config = config or {}
    search_text = await web_search_tool.ainvoke(
        {
            "query": f'"{safe_brand}" {safe_country} brand official products',
            "max_results": 5,
        },
        config=child_config,
    )
    report: dict[str, Any] = {
        "status": "ok",
        "brand": safe_brand,
        "country": safe_country,
        "country_code": country_code.upper(),
        "search": _parse_tool_json(search_text),
        "website": None,
        "scope": [
            "Search results are leads, not verified facts.",
            "Only the supplied website page is inspected; discovered links are not crawled.",
        ],
    }
    if url.strip():
        website_result = await analyze_website_tool.ainvoke(
            {"url": url},
            config=child_config,
        )
        report["website"] = _parse_tool_json(website_result)
        if (
            report["website"].get("status") != "ok"
            or report["search"].get("status") != "ok"
        ):
            report["status"] = "partial"
    elif report["search"].get("status") != "ok":
        report["status"] = "partial"
    return json.dumps(redact_sensitive_data(report), ensure_ascii=False, default=str)


@tool(args_schema=FullAnalysisInput)
def full_analysis(brand: str, report_data: str) -> str:
    """Summarize explicit evidence and gaps from a previous brand_report result.

    No web requests or hidden state are used here. Recommendations and
    performance predictions are deliberately excluded; the specialist agent
    should reason from the returned, source-linked observations.
    """
    try:
        parsed = json.loads(report_data)
    except (TypeError, ValueError):
        return json.dumps(
            {
                "status": "unavailable",
                "brand": brand,
                "unavailable_reason": "report_data must be JSON returned by brand_report.",
            },
            ensure_ascii=False,
        )
    if not isinstance(parsed, dict):
        return json.dumps(
            {"status": "unavailable", "brand": brand, "unavailable_reason": "Expected a JSON object."},
            ensure_ascii=False,
        )

    website = parsed.get("website") if isinstance(parsed.get("website"), dict) else {}
    search = parsed.get("search") if isinstance(parsed.get("search"), dict) else {}
    facts = website.get("observations") or website.get("offers_and_prices") or []
    sources: list[str] = []
    for row in search.get("results", []) if isinstance(search.get("results"), list) else []:
        if isinstance(row, dict) and row.get("url"):
            sources.append(str(row["url"]))
    for row in facts if isinstance(facts, list) else []:
        if isinstance(row, dict) and row.get("source_url"):
            sources.append(str(row["source_url"]))

    gaps = list(website.get("not_observed") or []) if isinstance(website, dict) else []
    if not website:
        gaps.append("No website audit was included in the supplied report.")
    if search.get("status") != "ok":
        gaps.append("Live search evidence is unavailable or incomplete.")
    output = {
        "status": "ok" if parsed.get("status") == "ok" else "partial",
        "brand": brand,
        "observed_facts": facts if isinstance(facts, list) else [],
        "source_urls": list(dict.fromkeys(sources)),
        "gaps": list(dict.fromkeys(str(gap) for gap in gaps)),
        "analysis_boundary": "Evidence inventory only; no unsupported marketing or performance claims.",
    }
    return json.dumps(redact_sensitive_data(output), ensure_ascii=False, default=str)


@tool(args_schema=RouteFileInput)
def route_file_to_tool(
    filename: str = "upload.json",
    file_content: str | None = None,
    file_path: str | None = None,  # noqa: ARG001 - paths are never read by this tool
    brand: str = "",
) -> str:
    """Classify explicitly supplied file content without reading server paths.

    A caller must provide file_content from an already-authorized upload. The
    legacy ``file_path`` parameter is ignored to prevent arbitrary file access.
    """
    if not file_content:
        return json.dumps(
            {
                "status": "unavailable",
                "filename": filename,
                "brand": brand,
                "unavailable_reason": (
                    "No file content was supplied. Server-local file paths are not read; "
                    "use the approved upload flow and pass its authorized content."
                ),
            },
            ensure_ascii=False,
        )

    text = file_content[:2_000_000]
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        keys = {str(key).casefold() for key in data}
        if {"impressions", "clicks", "spend"} & keys:
            route = "campaign_metrics"
        elif {"page_name", "ad_id", "ad_creative_bodies"} & keys:
            route = "competitor_ads"
        elif {"title", "description", "offers"} & keys:
            route = "website_or_brand_research"
        else:
            route = "general_json_evidence"
        preview: Any = data
    else:
        route = "text_evidence"
        preview = text[:12000]
    return json.dumps(
        redact_sensitive_data(
            {
                "status": "partial",
                "filename": filename,
                "brand": brand,
                "route": route,
                "content_preview": preview,
                "note": "Classification only; pass the supplied evidence to its matching tool for analysis.",
            }
        ),
        ensure_ascii=False,
        default=str,
    )


__all__ = [
    "BrandReportInput",
    "FullAnalysisInput",
    "RouteFileInput",
    "brand_report",
    "full_analysis",
    "route_file_to_tool",
]
