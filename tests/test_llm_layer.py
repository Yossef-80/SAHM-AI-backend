"""Layer 1 tests: the LLM client abstraction.

All of these run offline. The real provider clients are exercised against a
local ``httpx.MockTransport`` so request construction and response parsing are
verified without any network call or API key.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel, Field

from app.core.errors import LLMNotConfiguredError, LLMOutputError, LLMProviderError
from app.core.llm.base import (
    Msg,
    ToolCall,
    ToolSpec,
    TierName,
    extract_json,
    json_schema_for,
)
from app.core.llm.claude_client import ClaudeClient
from app.core.llm.mock_client import MockClient, sample_for_model, sample_from_schema
from app.core.llm.openai_client import OpenAIClient
from app.core.llm.router import get_llm, reset_clients


# --------------------------------------------------------------- fixtures ----


class SmallOutput(BaseModel):
    title: str = Field(min_length=3)
    score: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(min_length=1)
    note: str | None = None


class NestedOutput(BaseModel):
    name: str
    kind: Literal["a", "b"] = "a"
    items: list[SmallOutput] = Field(min_length=1)


# ------------------------------------------------------------------- tests ----


def test_json_schema_for_is_closed_and_has_no_refs() -> None:
    schema = json_schema_for(NestedOutput)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)


def test_json_schema_keeps_a_field_named_title() -> None:
    """Regression: Pydantic's display titles share the key "title"."""
    schema = json_schema_for(SmallOutput)
    assert "title" in schema["properties"]
    assert schema["properties"]["title"]["type"] == "string"
    assert "title" not in schema  # the model's own display title is stripped


def test_sample_from_schema_satisfies_constraints() -> None:
    payload = sample_from_schema(json_schema_for(NestedOutput))
    parsed = NestedOutput.model_validate(payload)
    assert parsed.kind in ("a", "b")
    assert len(parsed.items) >= 1
    assert 0.0 <= parsed.items[0].score <= 1.0


def test_sample_for_model_round_trips_every_field() -> None:
    payload = sample_for_model(SmallOutput)
    parsed = SmallOutput.model_validate(payload)
    assert len(parsed.title) >= 3
    assert parsed.tags


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"a": 1} thanks') == {"a": 1}
    with pytest.raises(LLMOutputError):
        extract_json("no json here")
    with pytest.raises(LLMOutputError):
        extract_json("")


async def test_mock_client_returns_schema_valid_output() -> None:
    client = MockClient(model="mock-strong-1")
    result = await client.complete(
        system="sys",
        messages=[Msg(role="user", content="hi")],
        response_schema=NestedOutput,
    )
    assert isinstance(result.parsed, NestedOutput)
    assert result.usage.tokens_in > 0
    assert result.model == "mock-strong-1"
    assert result.latency_ms >= 0


async def test_mock_client_batches_tool_calls_then_answers() -> None:
    """Real models batch independent tool calls; the mock does too."""
    client = MockClient(model="mock-fast-1", max_tool_calls_per_turn=4)
    tools = [
        ToolSpec.from_model("web_search", "search the web", SmallOutput),
        ToolSpec.from_model("get_keywords", "get keywords", SmallOutput),
        ToolSpec.from_model("get_reviews", "reviews", SmallOutput),
        ToolSpec.from_model("get_competitor_ads", "ads", SmallOutput),
    ]
    messages: list[Msg] = [Msg(role="user", content="research this")]

    first = await client.complete(system="s", messages=messages, tools=tools)
    # First turn batches all four tools.
    assert [c.name for c in first.tool_calls] == [
        "web_search",
        "get_keywords",
        "get_reviews",
        "get_competitor_ads",
    ]

    messages.append(Msg(role="assistant", content=first.text, tool_calls=first.tool_calls))
    for call in first.tool_calls:
        messages.append(
            Msg(role="tool", content='{"status":"ok"}', tool_call_id=call.id, name=call.name)
        )

    # Every tool has been called, so the next turn yields the final answer.
    final = await client.complete(
        system="s", messages=messages, tools=tools, response_schema=SmallOutput
    )
    assert isinstance(final.parsed, SmallOutput)
    assert final.tool_calls == []


async def test_mock_client_respects_batch_size_limit() -> None:
    client = MockClient(model="mock-fast-1", max_tool_calls_per_turn=2)
    tools = [
        ToolSpec.from_model(f"tool_{i}", f"tool {i}", SmallOutput) for i in range(5)
    ]
    result = await client.complete(system="s", messages=[Msg(role="user", content="x")], tools=tools)
    assert len(result.tool_calls) == 2
    assert [c.name for c in result.tool_calls] == ["tool_0", "tool_1"]


async def test_invalid_structured_output_triggers_exactly_one_repair() -> None:
    """Script the model: schema-violating output first, valid output second."""
    client = MockClient(
        model="mock-strong-1",
        script=[
            # Valid JSON but violates the schema (score out of range, no tags).
            {"text": json.dumps({"title": "ab", "score": 9.9, "tags": []})},
            # Repair reply: valid.
            {"text": json.dumps({"title": "good", "score": 0.5, "tags": ["x"]})},
        ],
    )
    result = await client.complete(
        system="s", messages=[Msg(role="user", content="go")], response_schema=SmallOutput
    )
    assert isinstance(result.parsed, SmallOutput)
    assert result.parsed.title == "good"
    # Exactly two provider calls: original + one repair.
    assert len(client.calls) == 2


async def test_unrepairable_output_raises_after_one_retry() -> None:
    client = MockClient(
        model="mock-strong-1",
        script=[
            {"text": json.dumps({"title": "ab", "score": 9.9, "tags": []})},
            {"text": "still not valid json at all"},
        ],
    )
    with pytest.raises(LLMOutputError) as excinfo:
        await client.complete(
            system="s", messages=[Msg(role="user", content="go")], response_schema=SmallOutput
        )
    assert "repair" in str(excinfo.value).lower()
    assert len(client.calls) == 2


# ------------------------------------------------------- claude over httpx ---


def _claude_response() -> dict:
    return {
        "id": "msg_1",
        "model": "claude-test-1",
        "content": [
            {"type": "text", "text": "thinking..."},
            {"type": "tool_use", "id": "tu_1", "name": "web_search", "input": {"query": "x"}},
        ],
        "usage": {"input_tokens": 120, "output_tokens": 45},
    }


def _claude_client(handler: object) -> ClaudeClient:
    return ClaudeClient(
        api_key="test-key",
        model="claude-test-1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_claude_client_builds_correct_request_and_parses() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_claude_response())

    client = _claude_client(handler)
    tools = [ToolSpec.from_model("web_search", "search", SmallOutput)]
    result = await client.complete(
        system="be helpful",
        messages=[Msg(role="user", content="hello")],
        tools=tools,
        tier="strong",
    )
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    body = captured["body"]
    assert body["model"] == "claude-test-1"
    assert body["system"] == "be helpful"
    assert body["messages"][0]["role"] == "user"
    assert body["tools"][0]["name"] == "web_search"
    assert body["tools"][0]["input_schema"]["additionalProperties"] is False

    assert "thinking..." in result.text
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "x"}
    assert result.usage.tokens_in == 120
    assert result.usage.tokens_out == 45


async def test_claude_client_appends_schema_instruction() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": '{"title":"abc","score":0.4,"tags":["t"]}'}
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = _claude_client(handler)
    result = await client.complete(
        system="base",
        messages=[Msg(role="user", content="go")],
        response_schema=SmallOutput,
    )
    assert "JSON schema" in captured["body"]["system"]
    assert isinstance(result.parsed, SmallOutput)


async def test_claude_client_retries_on_429_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_claude_response())

    client = _claude_client(handler)
    client.base_delay = 0.0  # keep the test fast
    result = await client.complete(system="s", messages=[Msg(role="user", content="x")])
    assert calls["n"] == 3
    assert result.tool_calls[0].name == "web_search"


async def test_claude_client_does_not_retry_on_400() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = _claude_client(handler)
    with pytest.raises(LLMProviderError) as excinfo:
        await client.complete(system="s", messages=[Msg(role="user", content="x")])
    assert calls["n"] == 1
    assert excinfo.value.retryable is False
    assert excinfo.value.status_code == 400


async def test_claude_client_exhausts_retries_on_500() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": "unavailable"})

    client = _claude_client(handler)
    client.max_retries = 2
    client.base_delay = 0.0
    with pytest.raises(LLMProviderError):
        await client.complete(system="s", messages=[Msg(role="user", content="x")])
    assert calls["n"] == 3  # initial + 2 retries


async def test_claude_client_without_key_raises_not_configured() -> None:
    client = ClaudeClient(api_key=None, model="claude-test-1")
    with pytest.raises(LLMNotConfiguredError):
        await client.complete(system="s", messages=[Msg(role="user", content="x")])


# ------------------------------------------------------- openai over httpx ---


def _openai_response() -> dict:
    return {
        "model": "gpt-test-1",
        "choices": [
            {
                "message": {
                    "content": "here",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query":"y"}'},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 80, "completion_tokens": 30},
    }


def _openai_client(handler: object) -> OpenAIClient:
    return OpenAIClient(
        api_key="sk-test",
        model="gpt-test-1",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_openai_client_builds_correct_request_and_parses() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_openai_response())

    client = _openai_client(handler)
    result = await client.complete(
        system="be helpful", messages=[Msg(role="user", content="hello")], tools=[]
    )
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    body = captured["body"]
    assert body["model"] == "gpt-test-1"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"

    assert result.text == "here"
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "y"}
    assert result.usage.tokens_in == 80
    assert result.usage.tokens_out == 30


async def test_openai_client_tool_result_message_shape() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )

    client = _openai_client(handler)
    await client.complete(
        system="s",
        messages=[
            Msg(role="user", content="q"),
            Msg(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="call_1", name="web_search", arguments={})],
            ),
            Msg(
                role="tool",
                content='{"status":"ok"}',
                tool_call_id="call_1",
                name="web_search",
            ),
        ],
    )
    msgs = captured["body"]["messages"]
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"


# ----------------------------------------------------------------- router ----


def test_router_defaults_to_mock(test_settings) -> None:
    reset_clients()
    client = get_llm("strong", test_settings)
    assert isinstance(client, MockClient)
    assert client.model == test_settings.llm_model_strong


def test_router_selects_claude_from_config(test_settings) -> None:
    reset_clients()
    test_settings.llm_provider_strong = "claude"
    test_settings.llm_model_strong = "claude-sonnet-4-5"
    test_settings.anthropic_api_key = "key-123"
    client = get_llm("strong", test_settings)
    assert isinstance(client, ClaudeClient)
    assert client.model == "claude-sonnet-4-5"


def test_router_selects_openai_for_fast_tier(test_settings) -> None:
    reset_clients()
    test_settings.llm_provider_fast = "openai"
    test_settings.llm_model_fast = "gpt-5-mini"
    test_settings.openai_api_key = "sk-x"
    client = get_llm("fast", test_settings)
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-5-mini"


def test_router_caches_per_tier(test_settings) -> None:
    reset_clients()
    a = get_llm("fast", test_settings)
    b = get_llm("fast", test_settings)
    assert a is b


def test_no_model_strings_outside_config() -> None:
    """Guard: provider/model literals must live in config only."""
    root = pathlib.Path(__file__).resolve().parents[1] / "app"
    pattern = re.compile(r"claude-[a-z0-9\-]+|gpt-[a-z0-9\-]+|sonnet|haiku|opus")
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "config.py" or "prompts" in str(path):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.name}:{i}: {line.strip()[:80]}")
    assert not offenders, "model strings found outside config: " + "; ".join(offenders)
