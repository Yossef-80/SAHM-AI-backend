"""Structured logging with request_id on every line.

A ``request_id`` is bound to a contextvar by middleware and appears in every
log record, so a single user request can be traced across LLM calls, tool
calls and DB writes.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_workspace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "workspace_id", default=None
)
_business_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "business_id", default=None
)

configured = False


def bind_request_id(request_id: str | None) -> None:
    _request_id.set(request_id)


def current_request_id() -> str | None:
    return _request_id.get()


def bind_workspace(workspace_id: str | None, business_id: str | None = None) -> None:
    _workspace_id.set(workspace_id)
    _business_id.set(business_id)


class ContextFilter(logging.Filter):
    """Injects request/workspace/business ids into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        record.workspace_id = _workspace_id.get()
        record.business_id = _business_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Never logs secrets (see redact())."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "workspace_id": getattr(record, "workspace_id", None),
            "business_id": getattr(record, "business_id", None),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(redact(extra))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


#: Keys whose values must never reach a log line.
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "token",
        "authorization",
        "password",
        "secret",
        "app_secret",
        "anthropic_api_key",
        "openai_api_key",
        "meta_access_token",
        "tavily_api_key",
        "serper_api_key",
        "brave_api_key",
        "dataforseo_password",
        "token_encryption_key",
        "credit_card",
    }
)


def redact(data: Any) -> Any:
    """Replace values of sensitive keys with a placeholder, recursively.

    Recurses through dicts AND lists: a tool-call log is a list of dicts, and a
    token hidden in one of them must not survive.
    """
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            if key.lower() in _SENSITIVE_KEYS:
                out[key] = "***REDACTED***"
            else:
                out[key] = redact(value)
        return out
    if isinstance(data, list):
        return [redact(item) for item in data]
    if isinstance(data, tuple):
        return tuple(redact(item) for item in data)
    return data


def configure_logging(level: str = "INFO", as_json: bool = True) -> None:
    global configured
    root = logging.getLogger()
    root.setLevel(level.upper())
    for existing in list(root.handlers):
        root.removeHandler(existing)
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())
    handler.setFormatter(JsonFormatter() if as_json else logging.Formatter("%(message)s"))
    root.addHandler(handler)
    configured = True


class _LoggerAdapter(logging.LoggerAdapter):
    """Adds an ``extra`` kwarg that the JSON formatter picks up."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        extra = kwargs.get("extra") or {}
        merged = dict(getattr(self, "extra", {}) or {})
        merged.update(extra)
        kwargs["extra"] = {"extra": merged}
        return msg, kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    if not configured:
        configure_logging()
    return _LoggerAdapter(logging.getLogger(name), {})
