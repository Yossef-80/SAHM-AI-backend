"""The exact wire contract the raw OpenAI-compatible client sends.

``app/core/llm/openai_client.py`` is the client every agent tool and the task
runtime talk through, and it used to send ``max_tokens`` plus ``temperature``
unconditionally. Against any reasoning model that is two separate ``400``s
("Unsupported parameter: 'max_tokens' ..." and "Unsupported value: 'temperature'
does not support 0.4 with this model"), so the whole product was limited to
GPT-4-era models without anyone noticing.

These tests assert the *body*, not the response: the response was always fine,
which is why this went unnoticed.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.core.llm.base import Msg, ToolSpec
from app.core.llm.openai_client import OpenAIClient

pytestmark = pytest.mark.anyio

TOOL = ToolSpec(
    name="generate_image",
    description="Generate an image.",
    input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}},
)


def _client(model: str, captured: list[dict], **kwargs: object) -> OpenAIClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    return OpenAIClient(
        api_key="sk-test",
        model=model,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kwargs,  # type: ignore[arg-type]
    )


async def test_gpt4_class_model_keeps_max_tokens_and_temperature() -> None:
    captured: list[dict] = []
    client = _client("gpt-4o-mini", captured)

    await client.complete(system="system", messages=[Msg(role="user", content="hi")], max_tokens=1234, temperature=0.3)

    body = captured[0]
    assert body["max_tokens"] == 1234
    assert body["temperature"] == 0.3
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body


async def test_reasoning_model_switches_the_limit_field_and_drops_temperature() -> None:
    captured: list[dict] = []
    client = _client("o3-mini", captured)

    await client.complete(system="system", messages=[Msg(role="user", content="hi")], max_tokens=1234, temperature=0.3)

    body = captured[0]
    assert body["max_completion_tokens"] == 1234
    assert "max_tokens" not in body
    # Not "0.3 sent anyway": the field is absent, because any non-default value
    # is a rejected request on this family.
    assert "temperature" not in body


async def test_reasoning_model_gets_effort_none_so_tools_are_accepted() -> None:
    """The 400 this prevents: tools on Chat Completions without effort ``none``."""
    captured: list[dict] = []
    client = _client("gpt-5.6-terra", captured)

    await client.complete(
        system="system",
        messages=[Msg(role="user", content="hi")],
        tools=[TOOL],
        max_tokens=800,
        temperature=0.2,
    )

    body = captured[0]
    assert body["tools"][0]["function"]["name"] == "generate_image"
    assert body["reasoning_effort"] == "none"
    assert body["max_completion_tokens"] == 800


async def test_a_configured_reasoning_effort_is_sent_as_the_user_set_it() -> None:
    captured: list[dict] = []
    client = _client("gpt-5.6-terra", captured, reasoning_effort="high")

    await client.complete(system="system", messages=[Msg(role="user", content="hi")], max_tokens=800, temperature=0.2)

    assert captured[0]["reasoning_effort"] == "high"


async def test_groq_gpt_oss_keeps_its_documented_field_name() -> None:
    """Groq's serving guidance overrides the family rule, not the other way round."""
    captured: list[dict] = []
    client = _client("openai/gpt-oss-20b", captured)
    client.base_url = "https://api.groq.com/openai/v1"

    await client.complete(system="system", messages=[Msg(role="user", content="hi")], max_tokens=512, temperature=0.1)

    body = captured[0]
    assert body["max_completion_tokens"] == 512
    assert "max_tokens" not in body
