"""Tier routing and per-role sampling: the guarantees this work added.

Before the change every agent graph built its model with

    get_llm(provider=get_ai_settings().llm_provider)

so the workspace's fast/strong choice, its model and its sampling controls never
reached an agent, and routing, research, performance analysis and ad copy all ran
on one model at one temperature (``AI_LLM_TEMPERATURE``, default 0.2). These
tests pin the behaviour that replaced it, at the three points where it can
silently regress: the role table, the graph seam, and the client factory.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from app.ai.providers.llm.agent_llm import (
    AGENT_ROLE,
    agent_llm_for_graph,
    process_handle,
    reset_process_handles,
)
from app.ai.roles import DEFAULTS, parse_temperature_overrides, resolve_role
from app.ai.settings import AISettings

# --------------------------------------------------------------------------- #
# 1. The role table                                                           #
# --------------------------------------------------------------------------- #


def test_every_agent_has_a_role_and_a_tier() -> None:
    for agent, role in AGENT_ROLE.items():
        assert role in DEFAULTS, f"{agent} points at unknown role {role}"
        assert resolve_role(role).tier in {"fast", "strong"}


def test_creative_work_is_hotter_than_extraction() -> None:
    """The defect this fixes: one temperature for every kind of work."""

    assert resolve_role("creative_copy").temperature > 0.7
    assert resolve_role("routing").temperature == 0.0
    assert resolve_role("extraction").temperature == 0.0
    assert resolve_role("critique").temperature < resolve_role("creative_copy").temperature


def test_writing_roles_use_the_strong_tier_and_mechanical_ones_the_fast_tier() -> None:
    assert resolve_role("creative_copy").tier == "strong"
    assert resolve_role("analysis").tier == "strong"
    assert resolve_role("routing").tier == "fast"
    assert resolve_role("vision").tier == "fast"


def test_unknown_role_falls_back_instead_of_raising() -> None:
    spec = resolve_role("not-a-role")
    assert spec.tier in {"fast", "strong"}
    assert spec.temperature is not None


def test_temperature_overrides_apply_and_typos_are_ignored() -> None:
    overrides = parse_temperature_overrides("creative_copy=0.95, routing=0.1, typo=2")
    assert resolve_role("creative_copy", overrides).temperature == 0.95
    assert resolve_role("routing", overrides).temperature == 0.1
    # An empty value is "no overrides" -- it must not break startup.
    assert parse_temperature_overrides("") == {}
    # Nonsense is dropped rather than raised: this is operator input.
    assert parse_temperature_overrides("{not json") == {}


def test_override_only_changes_temperature() -> None:
    base = resolve_role("creative_copy")
    overridden = resolve_role("creative_copy", {"creative_copy": 0.1})
    assert overridden.temperature == 0.1
    assert overridden.tier == base.tier
    assert overridden.max_tokens == base.max_tokens


# --------------------------------------------------------------------------- #
# 2. The graph seam                                                           #
# --------------------------------------------------------------------------- #


class _RecordingChat:
    def __init__(self) -> None:
        self.bound: list[Any] = []

    def bind_tools(self, tools: Any) -> "_RecordingChat":
        self.bound.append(tools)
        return self

    def invoke(self, *_a: Any, **_k: Any) -> Any:  # pragma: no cover - unused
        return None


class _RecordingHandle:
    def __init__(self) -> None:
        self.chat = _RecordingChat()
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> "_RecordingHandle":
        self.calls.append(kwargs)
        return self


SAMPLE_SETTINGS = AISettings(
    llm_provider="gemini",
    gemini_api_key="test-key",
    llm_provider_fast="gemini",
    llm_model_fast="fast-model",
    llm_provider_strong="gemini",
    llm_model_strong="strong-model",
)


def test_graph_gets_the_tier_model_and_the_role_temperature() -> None:
    """Layla's graph must ask for the strong model at its own temperature."""

    reset_process_handles()
    seen: list[dict[str, Any]] = []
    handle = _RecordingHandle()

    def builder(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return handle

    got = agent_llm_for_graph("layla", {}, builder=builder, settings=SAMPLE_SETTINGS)

    assert got is handle
    assert seen[0]["model"] == "strong-model"
    assert seen[0]["temperature"] == resolve_role("creative").temperature
    assert seen[0]["temperature"] > 0.2


def test_mechanical_roles_get_the_fast_model_and_zero_temperature() -> None:
    reset_process_handles()
    seen: list[dict[str, Any]] = []

    def builder(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return _RecordingHandle()

    process_handle("routing", builder=builder, settings=SAMPLE_SETTINGS)
    assert seen[0]["model"] == "fast-model"
    assert seen[0]["temperature"] == 0.0

    seen.clear()
    process_handle("analysis", builder=builder, settings=SAMPLE_SETTINGS)
    assert seen[0]["model"] == "strong-model"


def test_a_pre_resolved_handle_wins_over_the_role_default() -> None:
    """The orchestrator resolves once per turn with the workspace's settings."""

    handle = _RecordingHandle()
    config = {"configurable": {"agent_llm_handle": handle}}
    assert agent_llm_for_graph("layla", config) is handle


def test_a_handle_that_cannot_bind_tools_is_not_used_for_an_agent() -> None:
    """A routing-only chat cannot run an agent node; the graph builds its own."""

    reset_process_handles()
    routing_only = SimpleNamespace(chat=SimpleNamespace())  # no bind_tools
    built = _RecordingHandle()

    got = agent_llm_for_graph(
        "layla",
        {"configurable": {"agent_llm_handle": routing_only}},
        builder=lambda **kwargs: built,
        settings=SAMPLE_SETTINGS,
    )
    assert got is built


def test_a_stub_settings_object_is_not_forwarded_to_the_provider_factory() -> None:
    """Trimmed settings are common in tests and in partial call paths."""

    reset_process_handles()
    seen: list[dict[str, Any]] = []

    def builder(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return _RecordingHandle()

    process_handle("routing", builder=builder, settings=SimpleNamespace(llm_provider="mock"))
    assert "settings" not in seen[0]


# --------------------------------------------------------------------------- #
# 3. The provider factory and the client tier                                 #
# --------------------------------------------------------------------------- #


def test_gemini_keeps_an_explicit_temperature_on_every_model() -> None:
    """The old code dropped it for any 'flash-lite' model and used the default."""

    pytest.importorskip("langchain_google_genai")
    from app.ai.providers.llm.gemini import build_chat_model

    cfg = AISettings(llm_provider="gemini", gemini_api_key="test-key")
    for model in ("gemini-3.1-flash-lite", "gemini-3.6-flash"):
        chat = build_chat_model(cfg, model=model, temperature=0.85, max_tokens=512)
        assert float(chat.temperature) == 0.85, model
        assert int(chat.max_output_tokens) == 512, model


def test_gemini_falls_back_to_process_max_tokens_when_not_given() -> None:
    pytest.importorskip("langchain_google_genai")
    from app.ai.providers.llm.gemini import build_chat_model

    cfg = AISettings(llm_provider="gemini", gemini_api_key="test-key", llm_max_tokens=777)
    chat = build_chat_model(cfg, model="gemini-3.6-flash")
    assert int(chat.max_output_tokens) == 777


def test_task_clients_carry_workspace_sampling() -> None:
    """Settings -> LLM sampling used to be resolved, stored, then dropped."""

    from app.core.llm.openai_client import OpenAIClient

    client = OpenAIClient(
        api_key="test-key",
        model="gpt-test",
        default_temperature=0.9,
        default_max_tokens=123,
    )
    assert client.default_temperature == 0.9
    assert client.default_max_tokens == 123


def test_complete_uses_none_as_the_unspecified_sentinel() -> None:
    """0.0 is a value, not 'unspecified' -- the resolution happens inside."""

    from app.core.llm.base import BaseLLMClient

    signature = inspect.signature(BaseLLMClient.complete)
    assert signature.parameters["temperature"].default is None
    assert signature.parameters["max_tokens"].default is None
    assert BaseLLMClient.DEFAULT_TEMPERATURE == 0.4
    assert BaseLLMClient.DEFAULT_MAX_TOKENS == 2000
