"""A refused model request is retried once, on a different model; nothing else is.

Guard for the turn that died with a bare ``Layla graph failed (BadRequestError)``:
the provider had refused the request, the handler discarded the run, and the
operator had no way to tell a retired model from a bad tool schema. Two rules are
pinned here -- what counts as a refusal (never an error we might duplicate), and
what the logs then say.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.agents.orchestrator.brain import (  # noqa: E402
    _model_label,
    _provider_rejected_request,
)


class _OpenAIStyleBadRequest(Exception):
    """Shaped like the openai/groq SDKs raise."""

    status_code = 400


class _StatusCodeNamedError(Exception):
    """Shaped like Google's: a class name, no useful attributes."""


_StatusCodeNamedError.__name__ = "BadRequestError"


class _Wrapped:
    """Shaped like httpx-wrapped errors: status only on the response."""

    class response:  # noqa: N801 - attribute shape, not a class we use
        status_code = 422


@pytest.mark.parametrize(
    "exc",
    [
        _OpenAIStyleBadRequest("Error code: 400 - model_decommissioned"),
        _StatusCodeNamedError("invalid request"),
        _Wrapped(),
        type("E", (Exception,), {"code": 400})("google-style"),
        type("E", (Exception,), {"http_status": 404})("no such model"),
        type("E", (Exception,), {"status_code": 413})("payload too large"),
    ],
)
def test_a_refused_request_is_recognised(exc: BaseException) -> None:
    assert _provider_rejected_request(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("the request was sent and may have run"),
        ConnectionError("network dropped"),
        type("InternalServerError", (Exception,), {"status_code": 500})("provider hiccup"),
        type("RateLimitError", (Exception,), {"status_code": 429})("slow down"),
        type("AuthenticationError", (Exception,), {"status_code": 401})("bad key"),
        ValueError("something in our own code"),
    ],
)
def test_everything_else_is_left_alone(exc: BaseException) -> None:
    """A timeout may have run; a 5xx may have run; those must never be replayed."""
    assert _provider_rejected_request(exc) is False


def test_the_log_names_provider_and_model() -> None:
    class _Provider:
        value = "groq"

    class _Handle:
        provider = _Provider()
        model_name = "qwen/qwen3.8-27b"

    assert _model_label(_Handle()) == "groq/qwen/qwen3.8-27b"


def test_an_unknown_handle_does_not_break_the_log_line() -> None:
    assert _model_label(None) == "?/?"
    assert _model_label(object()) == "?/?"
