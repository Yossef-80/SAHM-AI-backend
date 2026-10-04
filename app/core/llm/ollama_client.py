"""Ollama client behind the LLMClient protocol.

Connects to a local or remote Ollama instance (e.g. over ngrok).
Implements BaseLLMClient for automatic retries, tool execution, and structured validation.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx

from app.core.errors import LLMNotConfiguredError, LLMProviderError
from app.core.llm.base import (
    BaseLLMClient,
    LLMResult,
    Msg,
    TierName,
    ToolCall,
    ToolSpec,
    is_retryable_exception,
    is_retryable_status,
    schema_instruction,
)
from app.schemas.common import Usage


class OllamaClient(BaseLLMClient):
    """Talks to Ollama's /api/chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout, max_retries=max_retries, base_delay=base_delay)
        self.base_url = (base_url or "").rstrip("/")
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"ngrok-skip-browser-warning": "true", "User-Agent": "sahm-ai/1.0"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _endpoint(self) -> str:
        if not self.base_url:
            raise LLMNotConfiguredError("OLLAMA_BASE_URL (or NGROK_URL) is not set")
        return f"{self.base_url}/api/chat"

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
        if effective_system:
            api_messages.append({"role": "system", "content": effective_system})

        for msg in messages:
            if msg.role == "system":
                api_messages.append({"role": "system", "content": msg.content or ""})
            elif msg.role == "tool":
                api_messages.append(
                    {
                        "role": "tool",
                        "content": msg.content or "",
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                api_messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
            else:
                api_messages.append({"role": msg.role, "content": msg.content or ""})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 8192,
            },
        }

        if tools and not response_schema:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]

        if response_schema is not None:
            body["format"] = "json"

        return body

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
        endpoint = self._endpoint()
        body = self._build_body(
            system=system,
            messages=messages,
            tools=tools,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        try:
            response = await self.client.post(endpoint, json=body)
        except Exception as exc:
            if is_retryable_exception(exc):
                raise LLMProviderError(
                    f"ollama request failed: {type(exc).__name__}", retryable=True
                ) from exc
            raise LLMProviderError(f"ollama request failed: {exc}", retryable=False) from exc

        if response.status_code >= 400:
            from app.core.logging import get_logger
            err_text = response.text
            if "ERR_NGROK_3200" in err_text or "endpoint is offline" in err_text:
                err_summary = "Ngrok tunnel endpoint is offline (ERR_NGROK_3200). Ensure the Colab notebook and ngrok tunnel are running."
            elif "<!DOCTYPE html>" in err_text:
                err_summary = f"Remote endpoint returned HTML error page (status {response.status_code})"
            else:
                err_summary = err_text[:300]
            get_logger(__name__).error("ollama_http_error", extra={"status": response.status_code, "body": err_summary})
            retryable = is_retryable_status(response.status_code)
            raise LLMProviderError(
                f"ollama returned {response.status_code}: {err_summary}",
                detail={"body": err_summary},
                retryable=retryable,
                status_code=response.status_code,
            )

        payload = response.json()
        message = payload.get("message", {}) or {}
        raw_tool_calls = message.get("tool_calls") or []

        tool_calls: list[ToolCall] = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {}) or {}
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}

            call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:16]}"
            tool_calls.append(
                ToolCall(id=call_id, name=fn.get("name", ""), arguments=args)
            )

        tokens_in = int(payload.get("prompt_eval_count") or 0)
        tokens_out = int(payload.get("eval_count") or 0)

        return LLMResult(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage=Usage(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=payload.get("model", self.model),
            ),
            model=payload.get("model", self.model),
        )


__all__ = ["OllamaClient"]
