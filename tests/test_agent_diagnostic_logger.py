"""Tests for Agent diagnostic logging and specialist loggers."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.ai.enums import LogFormat
from app.ai.loggers.agents import (
    diagnostic_logger,
    karim_logger,
    layla_logger,
    log_agent_diagnosis,
    nour_logger,
    omar_logger,
    orchestrator_logger,
    salma_logger,
)
from app.ai.loggers.callbacks import AILoggingCallback
from app.ai.settings import AISettings


def test_specialist_agent_loggers_exist() -> None:
    assert omar_logger is not None
    assert layla_logger is not None
    assert karim_logger is not None
    assert nour_logger is not None
    assert salma_logger is not None
    assert orchestrator_logger is not None
    assert diagnostic_logger is not None


def test_agent_graph_lifecycle_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.ai.loggers.base as log_base

    settings = AISettings(_env_file=None, log_dir=str(tmp_path), log_format=LogFormat.JSON)
    monkeypatch.setattr(log_base, "get_ai_settings", lambda: settings)

    callback = AILoggingCallback(agent="karim")
    chain_run = uuid4()

    callback.on_chain_start(
        {"name": "fetch_metrics"},
        {"campaign_id": "cmp_123"},
        run_id=chain_run,
        metadata={"langgraph_node": "analytics_eval"},
    )
    callback.on_chain_end(
        {"status": "ok", "ctr": 2.4},
        run_id=chain_run,
    )

    karim_log = (tmp_path / "agents" / "karim.log").read_text(encoding="utf-8")
    assert "agent_step_started" in karim_log
    assert "agent_step_finished" in karim_log
    assert "analytics_eval" in karim_log


def test_agent_node_failure_diagnosis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.ai.loggers.base as log_base

    settings = AISettings(_env_file=None, log_dir=str(tmp_path), log_format=LogFormat.JSON)
    monkeypatch.setattr(log_base, "get_ai_settings", lambda: settings)

    callback = AILoggingCallback(agent="salma")
    chain_run = uuid4()

    callback.on_chain_start(
        {"name": "optimize_bids"},
        {},
        run_id=chain_run,
        metadata={"langgraph_node": "bid_calculator"},
    )
    callback.on_chain_error(
        ValueError("Invalid ROAS target: must be > 0"),
        run_id=chain_run,
    )

    salma_log = (tmp_path / "agents" / "salma.log").read_text(encoding="utf-8")
    diag_log = (tmp_path / "agents" / "diagnostic.log").read_text(encoding="utf-8")

    assert "agent_step_failed" in salma_log
    assert "agent_node_failed" in diag_log
    assert "Invalid ROAS target" in diag_log


def test_log_agent_diagnosis_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.ai.loggers.base as log_base

    settings = AISettings(_env_file=None, log_dir=str(tmp_path), log_format=LogFormat.JSON)
    monkeypatch.setattr(log_base, "get_ai_settings", lambda: settings)

    log_agent_diagnosis(
        "handoff_eval",
        agent="orchestrator",
        node="router",
        status="ok",
        detail="Handoff from Orchestrator to Omar for GTM plan",
    )

    diag_log = (tmp_path / "agents" / "diagnostic.log").read_text(encoding="utf-8")
    assert "handoff_eval" in diag_log
    assert "Handoff from Orchestrator to Omar" in diag_log
