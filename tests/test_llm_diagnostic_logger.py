"""Tests for LLM diagnostic logger and real-time cost estimation."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.ai.enums import LogFormat, LogKind
from app.ai.loggers.base import get_ai_settings
from app.ai.loggers.callbacks import AILoggingCallback
from app.ai.loggers.cost import estimate_token_cost, format_cost
from app.ai.settings import AISettings


def test_cost_estimator_pricing_and_formatting() -> None:
    # Free models
    assert estimate_token_cost("mock", 1000, 500) == 0.0
    assert estimate_token_cost("ollama/qwen3-8b", 5000, 2000) == 0.0
    assert format_cost(0.0) == "Free ($0.00)"

    # GPT-4o-mini ($0.15 in, $0.60 out per 1M)
    # 1,000,000 in ($0.15) + 1,000,000 out ($0.60) = $0.75
    cost = estimate_token_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert abs(cost - 0.75) < 1e-4
    assert format_cost(cost) == "$0.7500"

    # Small tokens format
    small_cost = estimate_token_cost("gpt-4o-mini", 100, 50)
    assert small_cost > 0.0
    assert "$" in format_cost(small_cost)


def test_llm_diagnostic_callback_logs_cost_and_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.ai.loggers.base as log_base

    settings = AISettings(_env_file=None, log_dir=str(tmp_path), log_format=LogFormat.JSON)
    monkeypatch.setattr(log_base, "get_ai_settings", lambda: settings)

    callback = AILoggingCallback(agent="omar")
    llm_run = uuid4()

    callback.on_chat_model_start(
        {"name": "test-model"},
        [[HumanMessage(content="Develop a comprehensive go-to-market plan for MENA.")]],
        run_id=llm_run,
        invocation_params={"model": "gpt-4o-mini"},
        metadata={"langgraph_node": "generate_strategy"},
    )

    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="Here is the target market positioning...",
                        usage_metadata={"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
                    )
                )
            ]
        ]
    )
    callback.on_llm_end(response, run_id=llm_run)

    llm_log = (tmp_path / "providers" / "llm.log").read_text(encoding="utf-8")
    omar_log = (tmp_path / "agents" / "omar.log").read_text(encoding="utf-8")

    assert "llm_started" in llm_log
    assert "llm_finished" in llm_log
    assert '"input_tokens": 500' in llm_log
    assert '"output_tokens": 200' in llm_log
    assert "estimated_cost_usd" in llm_log
    assert "gpt-4o-mini" in llm_log

    # Omar's agent log also received LLM lifecycle context
    assert "llm_call_initiated" in omar_log
    assert "llm_call_completed" in omar_log


def test_llm_error_diagnosis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.ai.loggers.base as log_base

    settings = AISettings(_env_file=None, log_dir=str(tmp_path), log_format=LogFormat.JSON)
    monkeypatch.setattr(log_base, "get_ai_settings", lambda: settings)

    callback = AILoggingCallback(agent="layla")
    llm_run = uuid4()

    callback.on_chat_model_start(
        {"name": "test-model"},
        [[HumanMessage(content="Create ad copy.")]],
        run_id=llm_run,
        invocation_params={"model": "mock-model"},
        metadata={"langgraph_node": "copy_generation"},
    )

    callback.on_llm_error(
        ConnectionRefusedError("Connection refused by host 127.0.0.1:11434"),
        run_id=llm_run,
    )

    llm_log = (tmp_path / "providers" / "llm.log").read_text(encoding="utf-8")
    layla_log = (tmp_path / "agents" / "layla.log").read_text(encoding="utf-8")
    diagnostic_log = (tmp_path / "agents" / "diagnostic.log").read_text(encoding="utf-8")

    assert "llm_failed" in llm_log
    assert "Network/Connection Failure" in llm_log
    assert "llm_call_failed" in layla_log
    assert "llm_failure_diagnosed" in diagnostic_log
