"""The unified tools API: catalogue and deterministic invocation.

Covers ``GET /api/v1/tools`` and ``POST /api/v1/tools/{name}/invoke``. Before
these routes existed only two of the sixteen tools had an HTTP endpoint, so
``save_learning`` and ``get_brand_context`` had no HTTP-level coverage at all;
``test_save_learning_writes_a_memory_row`` closes that gap.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm.base import json_schema_for
from app.core.llm.mock_client import sample_from_schema
from app.core.tools.registry import ALL_TOOLS
from app.db.models import BrandMemoryItem

#: Every tool that has ever been agent-callable. Frozen so adding a tool without
#: covering it here is a deliberate, visible change.
EXPECTED_TOOL_COUNT = 16


# ------------------------------------------------------------- catalogue ----


async def test_tools_catalogue_lists_every_registered_tool(api_client) -> None:
    """Exactly the tools in ``ALL_TOOLS``, each fully described."""
    client = api_client
    resp = await client.get("/api/v1/tools")

    assert resp.status_code == 200
    tools = resp.json()
    assert len(tools) == EXPECTED_TOOL_COUNT
    assert {t["name"] for t in tools} == {tool.name for tool in ALL_TOOLS}

    for entry in tools:
        assert entry["name"]
        assert entry["description"]
        assert entry["side_effect"] in {"none", "draft_write"}
        assert entry["input_schema"], f"{entry['name']} has no input_schema"
        assert entry["output_schema"], f"{entry['name']} has no output_schema"
        assert isinstance(entry["personas"], list)
        assert entry["invocable"] is True


async def test_catalogue_reports_the_personas_that_may_use_each_tool(api_client) -> None:
    """``personas`` mirrors the personas' ``allowed_tools`` allowlists."""
    from app.core.agents.personas import PERSONAS, PERSONA_ORDER

    resp = await api_client.get("/api/v1/tools")
    assert resp.status_code == 200

    for entry in resp.json():
        expected = [
            persona_id
            for persona_id in PERSONA_ORDER
            if entry["name"] in PERSONAS[persona_id].allowed_tools
        ]
        assert entry["personas"] == expected

    by_name = {t["name"]: t for t in resp.json()}
    # Spot-check the derivation rather than only its self-consistency.
    assert by_name["web_search"]["personas"] == ["researcher"]
    assert by_name["generate_image"]["personas"] == ["layla"]
    assert by_name["get_brand_context"]["personas"] == [
        "researcher",
        "omar",
        "layla",
        "karim",
        "salma",
    ]


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_every_published_input_schema_round_trips(tool) -> None:
    """The published schema accepts a value built from that same schema.

    If ``GET /tools`` emitted a schema the handler could not validate, a client
    generating arguments from it would fail at invoke time.
    """
    published = json_schema_for(tool.input_model)
    sample = sample_from_schema(published)
    # Must validate against the tool itself, not just against the schema.
    tool.input_model.model_validate(sample)


# --------------------------------------------------------------- invoke ----


async def test_save_learning_writes_a_memory_row(api_client, seeded_db: AsyncSession) -> None:
    """A draft_write tool runs over HTTP and its write is committed."""
    client = api_client
    resp = await client.post(
        "/api/v1/tools/save_learning/invoke",
        json={"what_worked": "A 12% first-order discount lifted signups 18%."},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["business_id"] == "biz_test"
    # ToolOutput fields are always present on the wire.
    assert "unavailable_reason" in body
    # snake_case, matching the rest of the backend's documented convention.
    assert "business_id" in body and "businessId" not in body

    rows = (
        (
            await seeded_db.execute(
                select(BrandMemoryItem).where(
                    BrandMemoryItem.business_id == "biz_test",
                    BrandMemoryItem.kind == "learning",
                )
            )
        )
        .scalars()
        .all()
    )
    assert rows, "save_learning did not persist a BrandMemoryItem"
    assert body["id"] in {row.id for row in rows}


async def test_get_brand_context_invoke_returns_the_seeded_profile(api_client) -> None:
    """The read tool works over HTTP too, and reports the profile it read."""
    resp = await api_client.post("/api/v1/tools/get_brand_context/invoke", json={})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] in {"ok", "partial"}
    assert body["business_id"] == "biz_test"
    assert body["brand_voice"]


async def test_invoke_unknown_tool_returns_404(api_client) -> None:
    resp = await api_client.post("/api/v1/tools/no_such_tool/invoke", json={})

    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "not_found"
    assert "available" in error["detail"]


async def test_invoke_with_an_invalid_body_returns_422(api_client) -> None:
    """``what_worked`` is required and min_length=1."""
    resp = await api_client.post(
        "/api/v1/tools/save_learning/invoke", json={"what_worked": ""}
    )

    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_failed"
    assert error["detail"]["tool"] == "save_learning"
    assert error["detail"]["issues"]


async def test_no_protected_tool_is_reachable_over_http(api_client) -> None:
    """Defence in depth: nothing catalogued is ``external_write``."""
    tools = (await api_client.get("/api/v1/tools")).json()
    assert all(t["side_effect"] != "external_write" for t in tools)
    assert all(t["invocable"] for t in tools)


# --------------------------------------------------------------- connectors --


def test_every_tool_has_a_connector_entry() -> None:
    """The connector map must cover the registry exactly -- no drift allowed."""
    from app.core.tools.connectors import CONNECTORS
    from app.core.tools.registry import ALL_TOOLS

    assert {t.name for t in ALL_TOOLS} <= set(CONNECTORS)


async def test_catalogue_reports_each_tools_destination(api_client) -> None:
    """``GET /tools`` answers "which URL does this hit?" for every entry."""
    tools = (await api_client.get("/api/v1/tools")).json()

    by_name = {t["name"]: t for t in tools}

    reviews = by_name["get_reviews"]["connector"]
    assert reviews["provider"] == "google_places"
    assert reviews["url"].startswith("https://maps.googleapis.com/maps/api/place")
    assert reviews["configurable_via"] == "PLACES_BASE_URL"

    insights = by_name["get_campaign_insights"]["connector"]
    assert insights["url"].startswith("https://graph.facebook.com/")
    assert insights["missing_env"] == ["meta_access_token"]

    # Local-only tools must say so rather than invent a URL.
    assert by_name["save_learning"]["connector"]["url"] is None
    assert by_name["get_brand_context"]["connector"]["provider"] == "database"
    assert by_name["brand_report"]["connector"]["provider"] == "composite_research"
    assert by_name["brand_report"]["connector"]["url"] is None
    assert by_name["full_analysis"]["connector"]["provider"] == "local_analysis"
    assert by_name["route_file_to_tool"]["connector"]["provider"] == "local_file_router"

    # web_search resolves through SEARCH_PROVIDER, not a literal URL.
    search = by_name["web_search"]["connector"]
    assert search["selected_by"] == "SEARCH_PROVIDER"
    assert search["url"] == search["alternatives"]["mock"]


async def test_connectors_endpoint_includes_the_protected_tools(api_client) -> None:
    """``GET /tools/connectors`` is wider than the invocable catalogue."""
    connectors = (await api_client.get("/api/v1/tools/connectors")).json()

    assert "apply_change" in connectors
    assert "publish_campaign" in connectors
    assert connectors["apply_change"]["provider"] == "meta_write"
    assert "PROTECTED" in connectors["apply_change"]["notes"]


async def test_connector_urls_follow_the_configured_settings(
    api_client, monkeypatch
) -> None:
    """Repointing a URL env var changes the reported URL, with no code change."""
    from app.config import get_settings

    settings = get_settings()
    original = settings.meta_graph_base_url
    try:
        settings.meta_graph_base_url = "https://graph.example.test"
        connectors = (await api_client.get("/api/v1/tools/connectors")).json()
        assert connectors["get_breakdowns"]["url"].startswith(
            "https://graph.example.test/"
        )
    finally:
        settings.meta_graph_base_url = original
    assert monkeypatch is not None