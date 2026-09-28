"""OpenAI client behind the LLMClient protocol.

Same design as ``claude_client``: plain ``httpx``, no provider SDK, model
string from config only (``LLM_MODEL_STRONG`` / ``LLM_MODEL_FAST``).
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

API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIClient(BaseLLMClient):
    """Talks to the OpenAI Chat Completions API."""

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
                "OPENAI_API_KEY is not set; cannot use the openai provider"
            )
        return {
            "authorization": f"Bearer {self.api_key}",
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

        api_messages: list[dict[str, Any]] = [{"role": "system", "content": effective_system}]
        for msg in messages:
            if msg.role == "system":
                api_messages.append({"role": "system", "content": msg.content or ""})
            elif msg.role == "tool":
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
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
                                    "arguments": _json_dumps(tc.arguments),
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
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if tools:
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
        except Exception as exc:
            if is_retryable_exception(exc):
                raise LLMProviderError(
                    f"openai request failed: {type(exc).__name__}", retryable=True
                ) from exc
            raise LLMProviderError(
                f"openai request failed: {exc}", retryable=False
            ) from exc

        if response.status_code >= 400:
            retryable = is_retryable_status(response.status_code)
            raise LLMProviderError(
                f"openai returned {response.status_code}",
                detail={"body": response.text[:500]},
                retryable=retryable,
                status_code=response.status_code,
            )

        payload = response.json()
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            raw_args = fn.get("arguments", "{}")
            try:
                args = _json_loads(raw_args)
            except Exception:
                args = {}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
            )
        usage_block = payload.get("usage", {}) or {}
        return LLMResult(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage=Usage(
                tokens_in=int(usage_block.get("prompt_tokens", 0)),
                tokens_out=int(usage_block.get("completion_tokens", 0)),
                model=payload.get("model", self.model),
            ),
            model=payload.get("model", self.model),
        )


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def _json_loads(text: str) -> dict[str, Any]:
    import json

    parsed = json.loads(text)
    return parsed if isinstance(parsed, dict) else {}
