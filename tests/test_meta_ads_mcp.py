"""Offline protocol and safety tests for the read-only Meta Ads MCP client."""

from __future__ import annotations

import json

import httpx
import pytest

from app.ai.marketing.meta_ads_mcp import MetaAdsMcpClient, MetaAdsMcpError


async def test_initializes_discovers_and_calls_only_read_only_tool() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        method = body["method"]
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"mcp-session-id": "test-session"},
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}},
                },
            )
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            assert request.headers["mcp-session-id"] == "test-session"
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "ads_get_field_context",
                                "inputSchema": {"type": "object", "properties": {}},
                            }
                        ]
                    },
                },
            )
        if method == "tools/call":
            assert request.headers["mcp-protocol-version"] == "2025-06-18"
            assert body["params"]["name"] == "ads_get_field_context"
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"structuredContent": {"fields": ["optimization_goal"]}},
                },
            )
        raise AssertionError(f"Unexpected MCP method: {method}")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = MetaAdsMcpClient("secret-token", client=http)
    try:
        tools = await client.list_tools()
        result = await client.call_tool("ads_get_field_context", {"field": "optimization_goal"})
    finally:
        await http.aclose()

    assert tools[0]["name"] == "ads_get_field_context"
    assert result["structuredContent"]["fields"] == ["optimization_goal"]
    assert [item["method"] for item in calls] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]
    assert all("secret-token" not in json.dumps(item) for item in calls)


async def test_write_tool_is_rejected_before_network_call() -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail("network called")))
    client = MetaAdsMcpClient("secret-token", client=http)
    try:
        with pytest.raises(MetaAdsMcpError, match="read-only"):
            await client.call_tool("ads_create_campaign", {"name": "must not create"})
    finally:
        await http.aclose()


async def test_missing_mcp_scope_has_safe_actionable_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "permission denied"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = MetaAdsMcpClient("secret-token", client=http)
    try:
        with pytest.raises(MetaAdsMcpError, match="ads_mcp_management") as error:
            await client.initialize()
    finally:
        await http.aclose()

    assert error.value.status_code == 403
    assert "secret-token" not in str(error.value)


async def test_unauthorized_token_error_mentions_expiry_or_invalid_token() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid token"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = MetaAdsMcpClient("secret-token", client=http)
    try:
        with pytest.raises(MetaAdsMcpError, match="expired, revoked") as error:
            await client.initialize()
    finally:
        await http.aclose()

    assert error.value.status_code == 401
    assert "secret-token" not in str(error.value)


async def test_streamable_http_sse_response_is_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"protocolVersion": "2025-06-18", "capabilities": {}},
                },
            )
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                "event: message\n"
                f"data: {json.dumps({'jsonrpc': '2.0', 'id': body['id'], 'result': {'tools': []}})}\n\n"
            ),
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = MetaAdsMcpClient("secret-token", client=http)
    try:
        assert await client.list_tools() == []
    finally:
        await http.aclose()
