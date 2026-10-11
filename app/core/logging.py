"""Structured logging with request_id on every line.

A ``request_id`` is bound to a contextvar by middleware and appears in every
log record, so a single user request can be traced across LLM calls, tool
calls and DB writes.
"""

from __future__ import annotations

import contextvars
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
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
    """Bind the request's tenant to the log context.

    Called as a side effect of ``resolve_workspace``
    (``app/api/deps.py``) -- dependencies run before the handler, which is why
    this cannot live in the request-id middleware. See D7.
    """
    _workspace_id.set(workspace_id)
    _business_id.set(business_id)


def current_workspace_id() -> str | None:
    """The workspace bound to the current request, or None."""
    return _workspace_id.get()


def current_business_id() -> str | None:
    """The business bound to the current request, or None."""
    return _business_id.get()


class ContextFilter(logging.Filter):
    """Injects request/workspace/business ids into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        record.workspace_id = _workspace_id.get()
        record.business_id = _business_id.get()
        return True


class TraceTextFormatter(logging.Formatter):
    """Clean, human-readable trace log formatter with timestamps and context."""

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}"
        req_id = getattr(record, "request_id", None) or "-"
        biz_id = getattr(record, "business_id", None) or "-"

        ctx_parts = [f"req={req_id}"]
        if biz_id != "-":
            ctx_parts.append(f"biz={biz_id}")
        ctx_str = f"[{' | '.join(ctx_parts)}]"

        msg = record.getMessage()
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            extra = redact(extra)

        header = f"{ts} | {record.levelname:<7} | {record.name:<24} | {ctx_str} | {msg}"
        lines = [header]

        if extra:
            for k, v in extra.items():
                if isinstance(v, (dict, list)):
                    try:
                        v_str = json.dumps(v, ensure_ascii=False, indent=2)
                        indented = "\n".join(f"      {line}" for line in v_str.splitlines())
                        lines.append(f"    * {k}:\n{indented}")
                    except Exception:
                        lines.append(f"    * {k}: {v}")
                elif isinstance(v, str) and "\n" in v:
                    indented = "\n".join(f"      {line}" for line in v.splitlines())
                    lines.append(f"    * {k}:\n{indented}")
                else:
                    lines.append(f"    * {k}: {v}")

        if record.exc_info:
            lines.append(self.formatException(record.exc_info))

        return "\n".join(lines)


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


#: Keys whose values must never reach a log line. Matched case-insensitively on
#: the whole key, so ``password`` also covers ``password_hash`` only by explicit
#: listing -- hence the compound names below.
#:
#: P1 added the session-token family: a bearer token, its sha256 and the stored
#: hash are all useless once leaked, so none of them may appear in a log line.
#:
#: **Maintenance rule (P5.5).** ``tests/test_deployment_ops.py`` walks every
#: column of every mapped table and fails if a secret-bearing column name is
#: missing from this set, so adding a ``*_enc`` column without adding its name
#: here is a red test, not a silent leak. Keep it that way: the redaction is the
#: only thing standing between a ``logger.info(extra=...)`` and a credential in
#: a log aggregator.
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "token",
        "authorization",
        "password",
        "password_hash",
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
        "app_secret_enc",
        "meta_app_secret_enc",
        "access_token_enc",
        "meta_access_token_enc",
        "refresh_token_enc",
        "google_places_api_key_enc",
        "dataforseo_password_enc",
        "image_api_key_enc",
        "search_api_key_enc",
        "google_places_api_key",
        "image_api_key",
        "meta_app_secret",
        # --- Supabase + gateway credentials (P5.5 audit) ------------------
        # ``SUPABASE_SECRET_KEY`` / ``SUPABASE_SERVICE_ROLE_KEY`` are the
        # service-role credentials: they bypass Row Level Security entirely.
        "supabase_secret_key",
        "supabase_service_role_key",
        "supabase_publishable_key",
        # ``llm_extra_headers`` is merged into every outbound LLM request, and
        # gateways (OpenRouter, LiteLLM, vLLM) routinely carry
        # ``Authorization: Bearer <key>`` in exactly that field. It looked
        # harmless because it is a config knob, but it is the one place a
        # gateway credential can reach a log line.
        "llm_extra_headers",
        # Generic spellings that appear in provider payloads and headers.
        "client_secret",
        "secret_key",
        "private_key",
        "x-api-key",
        "x_api_key",
        "authorization_header",
        "db_password",
        "pgpassword",
        # Cookies. The refresh token lives in an HttpOnly cookie (P1.2), and a
        # request dump that echoes headers would otherwise carry it.
        "cookie",
        "set-cookie",
        # --- session credentials (P1) -------------------------------------
        "access_token_bearer",
        "bearer_token",
        "session_token",
        "token_hash",
        "refresh_token",
        "refresh_token_hash",
        # --- rate limiting (P4.2) ------------------------------------------
        # Not a credential, but logging it would let anyone reading logs
        # fingerprint a customer and correlate their traffic across requests.
        "rate_limit_key",
        # --- LLM provider settings (P5.5 audit) ----------------------------
        "fast_api_key_enc",
        "fast_extra_headers_enc",
        "strong_api_key_enc",
        "strong_extra_headers_enc",
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


def configure_logging(
    level: str = "INFO",
    as_json: bool = True,
    log_file: str | Path | None = "logs/trace.log",
) -> None:
    """Configure both stdout and rotating file logging for full tracing."""
    global configured
    root = logging.getLogger()
    root.setLevel(level.upper())
    for existing in list(root.handlers):
        root.removeHandler(existing)

    # 1. Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(ContextFilter())
    console_handler.setFormatter(JsonFormatter() if as_json else TraceTextFormatter())
    root.addHandler(console_handler)

    # 2. File handler: always saves full structured trace history to logs/trace.log
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            file_path, maxBytes=30 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.addFilter(ContextFilter())
        file_handler.setFormatter(TraceTextFormatter())
        root.addHandler(file_handler)

    configured = True


def log_trace(category: str, message: str, **fields: Any) -> None:
    """Convenience helper to log categorized trace messages."""
    get_logger(f"trace.{category.lower()}").info(f"[{category.upper()}] {message}", extra=fields)


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
