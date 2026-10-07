"""Logger factory for the AI layer.

Every agent, tool and provider gets its OWN log file:

    logs/
    ├── agents/nour.log
    ├── tools/web_search.log
    ├── tools/analyze_website.log
    └── providers/llm.log

The per-component logger *declarations* live in ``app/ai/loggers/<kind>s/<name>.py``
so you can find (and tune) the logger of any component in one place.

Usage inside a component::

    from app.ai.loggers.tools.web_search import logger
    logger.info("search_done", query=q, count=5)

Files are created lazily on the first log call, so importing a logger never
touches the disk.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from app.ai.enums import LogFormat, LogKind
from app.ai.settings import AISettings, get_ai_settings

#: Keys whose values must never reach a log file.
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "token",
        "password",
        "secret",
        "tavily_api_key",
        "openai_api_key",
        "anthropic_api_key",
    }
)

#: Long values are cut so a single huge tool output can't bloat the log.
_MAX_VALUE_CHARS = 1500

_ROOT = "sahm.ai"
_lock = threading.Lock()


def redact(data: Any) -> Any:
    """Replace values of sensitive keys, recursively, and truncate long strings."""
    if isinstance(data, dict):
        return {
            k: ("***REDACTED***" if str(k).lower() in _SENSITIVE_KEYS else redact(v))
            for k, v in data.items()
        }
    if isinstance(data, (list, tuple)):
        return [redact(v) for v in data]
    if isinstance(data, str) and len(data) > _MAX_VALUE_CHARS:
        return data[:_MAX_VALUE_CHARS] + f"...(+{len(data) - _MAX_VALUE_CHARS} chars)"
    return data


class _TextFormatter(logging.Formatter):
    """``2026-10-04 13:20:01 | INFO | tool.web_search | search_done | query='x' count=5``"""

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        fields: dict[str, Any] = getattr(record, "fields", {}) or {}
        kv = " ".join(f"{k}={v!r}" for k, v in fields.items())
        line = f"{ts} | {record.levelname:<7} | {record.component} | {record.getMessage()}"
        if kv:
            line += f" | {kv}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class _JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "component": record.component,
            "event": record.getMessage(),
            **(getattr(record, "fields", {}) or {}),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _formatter(fmt: LogFormat) -> logging.Formatter:
    return _JsonFormatter() if fmt == LogFormat.JSON else _TextFormatter()


def build_logger(kind: LogKind, name: str, settings: AISettings | None = None) -> logging.Logger:
    """Return the stdlib logger for one component, attaching its file handler once."""
    logger = logging.getLogger(f"{_ROOT}.{kind.value}.{name}")
    if getattr(logger, "_sahm_ready", False):
        return logger

    with _lock:
        if getattr(logger, "_sahm_ready", False):
            return logger
        cfg = settings or get_ai_settings()
        folder = Path(cfg.log_dir) / f"{kind.value}s"
        folder.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            folder / f"{name}.log",
            maxBytes=cfg.log_max_bytes,
            backupCount=cfg.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(_formatter(cfg.log_format))
        logger.addHandler(file_handler)

        if cfg.log_to_console:
            console = logging.StreamHandler(sys.stderr)
            console.setFormatter(_TextFormatter())
            logger.addHandler(console)

        logger.setLevel(cfg.log_level.value)
        logger.propagate = False  # keep AI logs out of the API's root JSON stream
        logger._sahm_ready = True  # type: ignore[attr-defined]
    return logger


class ComponentLogger:
    """Thin structured wrapper: ``logger.info("event_name", key=value, ...)``.

    The underlying file handler is created on first use (lazy), so declaring a
    logger at import time is free.
    """

    def __init__(self, kind: LogKind, name: str) -> None:
        self.kind = kind
        self.name = str(name)
        self._logger: logging.Logger | None = None

    @property
    def component(self) -> str:
        return f"{self.kind.value}.{self.name}"

    @property
    def stdlib(self) -> logging.Logger:
        if self._logger is None:
            self._logger = build_logger(self.kind, self.name)
        return self._logger

    def _log(self, level: int, event: str, exc_info: bool = False, **fields: Any) -> None:
        self.stdlib.log(
            level,
            event,
            exc_info=exc_info,
            extra={"fields": redact(fields), "component": self.component},
        )

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, exc_info: bool = False, **fields: Any) -> None:
        self._log(logging.ERROR, event, exc_info=exc_info, **fields)


_cache: dict[tuple[LogKind, str], ComponentLogger] = {}


def get_component_logger(kind: LogKind, name: str) -> ComponentLogger:
    """Shared instance per (kind, name) -- used by the callback router."""
    key = (kind, str(name))
    if key not in _cache:
        _cache[key] = ComponentLogger(kind, str(name))
    return _cache[key]


def reset_loggers() -> None:
    """Close all file handlers (tests use this to switch ``AI_LOG_DIR``)."""
    manager = logging.Logger.manager
    for logger_name, obj in list(manager.loggerDict.items()):
        if logger_name.startswith(_ROOT) and isinstance(obj, logging.Logger):
            for h in list(obj.handlers):
                h.close()
                obj.removeHandler(h)
            obj._sahm_ready = False  # type: ignore[attr-defined]
    for cl in _cache.values():
        cl._logger = None


__all__ = [
    "ComponentLogger",
    "build_logger",
    "get_component_logger",
    "redact",
    "reset_loggers",
]
