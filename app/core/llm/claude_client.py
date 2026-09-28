"""Anthropic Claude client behind the LLMClient protocol.

Uses plain ``httpx`` rather than the Anthropic SDK: the agent layer must not
depend on a provider SDK, and one HTTP layer means one place where timeouts,
retries and usage extraction are implemented and tested.

The model string comes from config only (``LLM_MODEL_STRONG`` / ``LLM_MODEL_FAST``).
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.errors import LLMNotConfiguredError, LLMProviderError
from app.core.llm.base import (
    BaseLLMClient,
    LLMResult,
    Msg,
    ToolCall,
    ToolSpec,
    TierName,
    is_retryable_exception,
    is_retryable_status,
    schema_instruction,
)
from app.schemas.common import Usage

API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class ClaudeClient(BaseLLMClient):
    """Talks to the Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        base_delay: float = 0.5,
        base_url: str = API_URL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout, max_retries=max_retries, base_delay=base_delay)
        self.api_key = api_key
        self.base_url = base_url
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- request building ----------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise LLMNotConfiguredError(
                "ANTHROPIC_API_KEY is not set; cannot use the claude provider"
            )
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _build_body(
        self,
        *,
        system: str,
        messages: list[Msg],
        tools: list[ToolSpec] | None,
        response_schema: type | None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        effective_system = system
        if response_schema is not None:
            effective_system = f"{system}\n\n{schema_instruction(response_schema)}"

        api_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                # Anthropic takes system separately; fold it in.
                effective_system = f"{effective_system}\n\n{msg.content or ''}"
                continue
            if msg.role == "tool":
                api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content or "",
                            }
                        ],
                    }
                )
                continue
            if msg.role == "assistant" and msg.tool_calls:
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                api_messages.append({"role": "assistant", "content": content})
                continue
            api_messages.append({"role": msg.role, "content": msg.content or ""})

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": effective_system,
            "messages": api_messages or [{"role": "user", "content": "(no input)"}],
        }
        if tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
        return body

    # -- the one method the base class needs ---------------------------------

    async def _complete_once(
        self,
        *,
        system: str,
        messages: list[Msg],
        tools: list[ToolSpec] | None,
        response_schema: type | None,
        tier: TierName,
        max_tokens: int,
        temperature: float,
    ) -> LLMResult:
        headers = self._headers()
        body = self._build_body(
            system=system,
            messages=messages,
            tools=tools,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            response = await self.client.post(self.base_url, headers=headers, json=body)
        except Exception as exc:  # httpx timeout/connect
            if is_retryable_exception(exc):
                raise LLMProviderError(
                    f"claude request failed: {type(exc).__name__}", retryable=True
                ) from exc
            raise LLMProviderError(
                f"claude request failed: {exc}", retryable=False
            ) from exc

        if response.status_code >= 400:
            retryable = is_retryable_status(response.status_code)
            raise LLMProviderError(
                f"claude returned {response.status_code}",
                detail={"body": response.text[:500]},
                retryable=retryable,
                status_code=response.status_code,
            )

        payload = response.json()
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in payload.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}) or {},
                    )
                )
        usage_block = payload.get("usage", {}) or {}
        return LLMResult(
            text="\n".join(p for p in text_parts if p),
            tool_calls=tool_calls,
            usage=Usage(
                tokens_in=int(usage_block.get("input_tokens", 0)),
                tokens_out=int(usage_block.get("output_tokens", 0)),
                model=payload.get("model", self.model),
            ),
            model=payload.get("model", self.model),
        )
