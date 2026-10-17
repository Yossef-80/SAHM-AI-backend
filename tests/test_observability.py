"""Tests for LangSmith, Langfuse, and unified observability integrations."""

from __future__ import annotations

import os
import pytest

from app.ai.observability import (
    get_langfuse_callback,
    get_observability_callbacks,
    get_observability_status,
    is_langfuse_enabled,
    is_langsmith_enabled,
)
from app.ai.settings import AISettings


def test_observability_status_structure() -> None:
    status = get_observability_status()
    assert "local_logging" in status
    assert "langsmith" in status
    assert "langfuse" in status
    assert "cost_estimator" in status
    assert status["local_logging"]["status"] == "active"
    assert status["cost_estimator"]["status"] == "active"


def test_langsmith_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    assert is_langsmith_enabled() is True

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key_12345")
    assert is_langsmith_enabled() is True


def test_langfuse_detection_and_safe_init(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.ai.settings import get_ai_settings

    settings = get_ai_settings()
    monkeypatch.setattr(settings, "langfuse_public_key", None)
    monkeypatch.setattr(settings, "langfuse_secret_key", None)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert is_langfuse_enabled() is False
    assert get_langfuse_callback() is None

    # When keys are configured
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test-1234")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test-5678")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    assert is_langfuse_enabled() is True
    handler = get_langfuse_callback()
    assert handler is not None


def test_get_observability_callbacks_assembles_cleanly() -> None:
    callbacks = get_observability_callbacks(agent="omar")
    assert len(callbacks) >= 1
    # First is always AILoggingCallback
    from app.ai.loggers.callbacks import AILoggingCallback
    assert isinstance(callbacks[0], AILoggingCallback)
    assert callbacks[0].agent == "omar"
