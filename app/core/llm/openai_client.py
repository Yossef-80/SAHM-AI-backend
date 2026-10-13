"""OpenAI client behind the LLMClient protocol.

Same design as ``claude_client``: plain ``httpx``, no provider SDK, model
string from config only (``LLM_MODEL_STRONG`` / ``LLM_MODEL_FAST``).
"""

from __future__ import annotations

from typing import Any

import httpx

from app.config import GROQ_GPT_OSS_MODEL_PREFIX
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
from app.core.logging import get_logger
from app.schemas.common import Usage

#: Default OpenAI endpoint. Overridable via ``LLM_BASE_URL`` for any
#: OpenAI-compatible gateway (OpenRouter, Gemini, vLLM, ...). The override must
#: be the FULL endpoint including "/chat/completions" -- no path joining here.
API_URL = "https://api.openai.com/v1/chat/completions"

_logger = get_logger(__name__)


class OpenAIClient(BaseLLMClient):
    """Talks to the OpenAI Chat Completions API.

    ``base_url`` and ``extra_headers`` make this client usable against any
    OpenAI-compatible endpoint without touching the provider selection.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        base_delay: float = 0.5,
        base_url: str = API_URL,
        extra_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout, max_retries=max_retries, base_delay=base_delay)
        self.api_key = api_key
        self.base_url = base_url
        self.extra_headers: dict[str, str] = dict(extra_headers or {})
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
                "No API key is configured for the selected OpenAI-compatible provider"
            )
        # Extras first, ours last: a gateway may add headers (OpenRouter's
        # HTTP-Referer / X-Title) but can never override the auth header.
        return {
            **self.extra_headers,
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
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

        # Groq's GPT-OSS serving guidance is to put instructions in a user
        # message rather than a system message. It also documents
        # max_completion_tokens as the preferred output limit field.
        groq_gpt_oss = (
            "api.groq.com" in self.base_url
            and self.model.startswith(GROQ_GPT_OSS_MODEL_PREFIX)
        )
        # During structured synthesis, GPT-OSS may try to call an unavailable
        # tool again if the transcript still contains native tool-call records.
        # Turn completed tool outputs into ordinary user context for this final
        # no-tools request instead of replaying the provider's tool protocol.
        flatten_tool_history = groq_gpt_oss and response_schema is not None
        if flatten_tool_history:
            effective_system = (
                f"{effective_system}\n\nAll tools for this request have already run. "
                "Do not call or request tools again. Produce the required final output "
                "using only the completed tool results in the conversation. If those "
                "results are unavailable, report that limitation without guessing."
            )
        api_messages: list[dict[str, Any]] = (
            []
            if groq_gpt_oss
            else [{"role": "system", "content": effective_system}]
        )
        instruction_added = False
        for msg in messages:
            if msg.role == "system":
                if groq_gpt_oss:
                    effective_system = f"{effective_system}\n\n{msg.content or ''}"
                else:
                    api_messages.append({"role": "system", "content": msg.content or ""})
            elif msg.role == "tool":
                if flatten_tool_history:
                    api_messages.append(
                        {
                            "role": "user",
                            # The system prompt already includes the rendered,
                            # bounded tool results. Do not duplicate full tool
                            # payloads here; search snippets can otherwise push
                            # Groq's small TPM allowance over the limit.
                            "content": f"Completed tool result ({msg.name or 'tool'}) was included in the task context.",
                        }
                    )
                else:
                    api_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": msg.tool_call_id,
                            "content": msg.content or "",
                        }
                    )
            elif msg.role == "assistant" and msg.tool_calls:
                if not flatten_tool_history:
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
                text_content = msg.content or ""
                if groq_gpt_oss and msg.role == "user" and not instruction_added:
                    text_content = f"{effective_system}\n\n{text_content}"
                    instruction_added = True
                if msg.images:
                    content_parts: list[dict[str, Any]] = []
                    if text_content:
                        content_parts.append({"type": "text", "text": text_content})
                    content_parts.extend(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image.media_type};base64,{image.data}"
                            },
                        }
                        for image in msg.images
                    )
                    content: str | list[dict[str, Any]] = content_parts
                else:
                    content = text_content
                api_messages.append({"role": msg.role, "content": content})

        body: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens" if groq_gpt_oss else "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if tools and not flatten_tool_history:
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
            # GPT-OSS can otherwise attempt another call during the final
            # synthesis step after receiving tool results. Keep the schemas in
            # that request for valid tool-call history, but explicitly disable
            # further calls while it produces the structured final answer.
            body["tool_choice"] = "none" if response_schema is not None else "auto"
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
            retry_after_header = response.headers.get("retry-after")
            try:
                retry_after = float(retry_after_header) if retry_after_header else None
            except ValueError:
                retry_after = None
            raise LLMProviderError(
                f"openai returned {response.status_code}",
                detail={
                    "body": response.text[:1000],
                    "retry_after_header": retry_after_header or "",
                },
                retryable=retryable,
                status_code=response.status_code,
                retry_after=retry_after,
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
        finish_reason = str(choice.get("finish_reason") or "")
        if finish_reason == "length":
            _logger.warning(
                "llm_response_truncated",
                extra={
                    "model": payload.get("model", self.model),
                    "finish_reason": finish_reason,
                },
            )
        text = message.get("content") or ""
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip() and not text.strip():
            # Some OpenAI-compatible gateways refuse instead of answering. Put
            # the reason in `text` so extract_json raises a readable error
            # rather than reporting an empty response.
            text = refusal.strip()
        return LLMResult(
            text=text,
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
