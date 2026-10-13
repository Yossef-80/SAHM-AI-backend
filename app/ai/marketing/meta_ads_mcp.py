"""Small, read-only MCP client for Meta Ads field metadata.

This client deliberately permits only ``ads_get_field_context``. Campaign,
Ad Set, Ad creation, updates and activation stay on the existing Graph API
approval path in ``meta_ads.py``.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Self

import httpx

META_ADS_MCP_URL = "https://mcp.facebook.com/ads"
_SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
_READ_ONLY_TOOLS = frozenset({"ads_get_field_context"})
_MAX_TOOL_PAGES = 5
_MAX_RESPONSE_BYTES = 1_000_000


class MetaAdsMcpError(RuntimeError):
    """Safe-to-display MCP failure; never includes an access token."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class MetaAdsMcpClient:
    """Minimal Streamable HTTP MCP client with explicit read-only enforcement."""

    def __init__(
        self,
        access_token: str,
        *,
        endpoint: str = META_ADS_MCP_URL,
        timeout: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        token = access_token.strip()
        if not token:
            raise MetaAdsMcpError("Meta Ads MCP requires a connected Meta access token.")
        self._access_token = token
        self.endpoint = endpoint
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._session_id: str | None = None
        self._protocol_version: str | None = None
        self._request_ids = itertools.count(1)
        self._initialized = False

    async def __aenter__(self) -> Self:
        try:
            await self.initialize()
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    @property
    def http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                trust_env=False,
                follow_redirects=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        last_error: MetaAdsMcpError | None = None
        for protocol_version in _SUPPORTED_PROTOCOL_VERSIONS:
            try:
                response, envelope = await self._send(
                    "initialize",
                    {
                        "protocolVersion": protocol_version,
                        "capabilities": {},
                        "clientInfo": {"name": "sahm-meta-plan-assist", "version": "1.0.0"},
                    },
                )
            except MetaAdsMcpError as exc:
                last_error = exc
                # Older hosted servers may not accept the newest MCP revision.
                # Retry only a version-negotiation failure, not auth/network errors.
                if exc.status_code not in {400, 406, 422}:
                    raise
                continue

            result = envelope.get("result")
            if not isinstance(result, dict):
                raise MetaAdsMcpError("Meta Ads MCP returned an invalid initialize response.")
            negotiated = result.get("protocolVersion")
            if not isinstance(negotiated, str) or not negotiated:
                negotiated = protocol_version
            self._session_id = response.headers.get("mcp-session-id")
            self._protocol_version = negotiated
            await self._send(
                "notifications/initialized",
                {},
                notification=True,
                protocol_version=negotiated,
            )
            self._initialized = True
            return

        raise last_error or MetaAdsMcpError(
            "Meta Ads MCP could not negotiate a supported protocol version."
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        await self.initialize()
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(_MAX_TOOL_PAGES):
            params = {"cursor": cursor} if cursor else {}
            _, envelope = await self._send("tools/list", params)
            result = envelope.get("result")
            if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
                raise MetaAdsMcpError("Meta Ads MCP returned an invalid tools/list response.")
            tools.extend(item for item in result["tools"] if isinstance(item, dict))
            next_cursor = result.get("nextCursor")
            cursor = next_cursor if isinstance(next_cursor, str) and next_cursor else None
            if cursor is None:
                return tools
        raise MetaAdsMcpError("Meta Ads MCP returned too many pages of tool definitions.")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a single allowlisted, read-only Meta tool."""
        if name not in _READ_ONLY_TOOLS:
            raise MetaAdsMcpError("This workflow is read-only; Meta write tools are disabled.")
        if not isinstance(arguments, dict):
            raise MetaAdsMcpError("Meta Ads MCP tool arguments must be an object.")
        try:
            encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise MetaAdsMcpError("Meta Ads MCP tool arguments were not valid JSON.") from exc
        if len(encoded.encode("utf-8")) > 20_000:
            raise MetaAdsMcpError("Meta Ads MCP tool arguments exceeded the safe size limit.")

        await self.initialize()
        _, envelope = await self._send(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise MetaAdsMcpError("Meta Ads MCP returned an invalid tool response.")
        if result.get("isError") is True:
            raise MetaAdsMcpError("Meta rejected the read-only Meta Ads MCP field-context request.")
        return result

    async def _send(
        self,
        method: str,
        params: dict[str, Any],
        *,
        notification: bool = False,
        protocol_version: str | None = None,
    ) -> tuple[httpx.Response, dict[str, Any]]:
        request_id = None if notification else next(self._request_ids)
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
        if request_id is not None:
            body["id"] = request_id

        headers = {
            "authorization": f"Bearer {self._access_token}",
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
        }
        version = protocol_version or self._protocol_version
        if version:
            headers["mcp-protocol-version"] = version
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        try:
            response = await self.http.post(self.endpoint, headers=headers, json=body)
        except httpx.HTTPError as exc:
            # Do not surface request exception text: it can contain the URL or
            # headers from a custom transport, and the UI only needs a safe hint.
            raise MetaAdsMcpError("Could not reach the Meta Ads MCP service.") from exc

        if response.status_code == 401:
            raise MetaAdsMcpError(
                "Meta Ads MCP rejected the saved access token (401). It may be expired, revoked, "
                "or not authorized for this app. Issue a fresh user token for the Sahm Meta app "
                "with ads_mcp_management and ads_read or ads_management, then replace the saved token.",
                status_code=401,
            )
        if response.status_code == 403:
            raise MetaAdsMcpError(
                "Meta Ads MCP denied the token (403). Grant ads_mcp_management together with "
                "ads_read or ads_management to this app, then reconnect Meta.",
                status_code=403,
            )
        if response.status_code >= 400:
            message = "Meta Ads MCP request failed."
            if response.status_code in {400, 406, 422}:
                body_text = response.text[:1000].lower()
                if "protocol" in body_text or "version" in body_text:
                    message = "Meta Ads MCP did not accept this protocol version."
            raise MetaAdsMcpError(message, status_code=response.status_code)
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise MetaAdsMcpError("Meta Ads MCP response exceeded the safe size limit.")
        if notification and not response.content:
            return response, {}

        envelope = self._decode_response(response, request_id=request_id)
        if not notification and envelope.get("error") is not None:
            error = envelope.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                if code in {-32600, -32601} and method == "initialize":
                    raise MetaAdsMcpError(
                        "Meta Ads MCP rejected protocol negotiation.", status_code=400
                    )
            raise MetaAdsMcpError("Meta Ads MCP returned a JSON-RPC error.")
        return response, envelope

    @staticmethod
    def _decode_response(response: httpx.Response, *, request_id: int | None) -> dict[str, Any]:
        content_type = response.headers.get("content-type", "").lower()
        candidates: list[Any] = []
        try:
            if "text/event-stream" in content_type:
                data_lines = [
                    line[5:].strip()
                    for line in response.text.splitlines()
                    if line.startswith("data:")
                ]
                for item in data_lines:
                    if item and item != "[DONE]":
                        candidates.append(json.loads(item))
            else:
                candidates.append(response.json())
        except (json.JSONDecodeError, ValueError) as exc:
            raise MetaAdsMcpError("Meta Ads MCP returned an unreadable response.") from exc

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if request_id is None or candidate.get("id") == request_id:
                return candidate
        if request_id is None:
            # Some Streamable HTTP implementations acknowledge a notification
            # with an empty JSON object rather than HTTP 202/204.
            return {}
        raise MetaAdsMcpError("Meta Ads MCP response did not match the request.")


__all__ = ["META_ADS_MCP_URL", "MetaAdsMcpClient", "MetaAdsMcpError"]
