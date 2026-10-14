"""Async engine/session management.

Supports both Postgres (asyncpg) and SQLite (aiosqlite) from the same
``DATABASE_URL``. Alembic migrations are the source of truth for schema in
Postgres; ``create_all`` is used for dev/test on SQLite.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.core.logging import get_logger, redact

from contextvars import ContextVar

_logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None

#: Per-request database metrics context variable
_request_db_metrics: ContextVar[dict[str, Any] | None] = ContextVar("_request_db_metrics", default=None)


def init_request_db_metrics() -> dict[str, Any]:
    """Initialize request-scoped DB metrics tracker."""
    metrics = {"query_count": 0, "total_duration_ms": 0}
    _request_db_metrics.set(metrics)
    return metrics


def get_request_db_metrics() -> dict[str, Any] | None:
    """Retrieve current request-scoped DB metrics tracker."""
    return _request_db_metrics.get()


def _clean_db_trace_param(val: Any) -> Any:
    """Recursively parse JSON string parameters to native dict/list for readable trace logging."""
    if isinstance(val, str):
        val_trimmed = val.strip()
        if (val_trimmed.startswith("{") and val_trimmed.endswith("}")) or (
            val_trimmed.startswith("[") and val_trimmed.endswith("]")
        ):
            try:
                parsed = json.loads(val_trimmed)
                return _clean_db_trace_param(parsed)
            except Exception:
                pass
    elif isinstance(val, dict):
        return {k: _clean_db_trace_param(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_clean_db_trace_param(v) for v in val]
    elif isinstance(val, tuple):
        return tuple(_clean_db_trace_param(v) for v in val)
    return val


#: Host fragments that identify a connection-pooler proxy (PgBouncer / Supavisor).
#: Matched against the parsed hostname only -- never against the whole URL, so a
#: password that happens to contain "pgbouncer" cannot flip the branch.
_POOLER_HOST_MARKERS: tuple[str, ...] = (
    "pooler.supabase.com",  # Supabase's Supavisor pooler (IPv4-native)
    ".pooler.",  # self-hosted / generic PgBouncer conventions
    "pgbouncer",
    "supavisor",
)


def is_pooler_url(url: str) -> bool:
    """True when ``url`` points at a transaction-mode pooler, not a real server.

    Deliberately host-based rather than config-based: the deployment that matters
    is the one in front of the *URL*, and a URL is the only thing this function
    is given. A self-hosted PgBouncer on a custom hostname is covered by the
    ``pgbouncer`` marker; add a marker here if you front the app with something
    else.
    """
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:  # pragma: no cover - malformed URL
        return False
    return any(marker in host for marker in _POOLER_HOST_MARKERS)


def _engine_kwargs(url: str, echo: bool) -> dict[str, Any]:
    """Build the ``create_async_engine`` kwargs for ``url``.

    Split out from :func:`_make_engine` so the pooler branch is assertable
    without opening a connection (see ``tests/test_deployment_ops.py``).

    **Why the pooler branch exists (P5.3):** a pooler in *transaction* mode hands
    the same backend connection to many clients, so a named prepared statement
    prepared by one session is still there when the next session reuses that
    connection. asyncpg's statement cache then fails with ``prepared statement
    "s1" already exists`` / ``prepared statement does not exist`` under load.
    ``statement_cache_size=0`` disables that cache; the statements are then
    re-prepared per transaction, which is slower but correct. See
    ``docs/PRODUCTION_ROADMAP.md`` §3.3.
    """
    kwargs: dict[str, Any] = {"future": True, "echo": echo}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        return kwargs

    # Postgres: sane pool defaults, pre-ping to survive restarts, warm pool reuse
    kwargs.update({
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_recycle": 1800,
        "pool_timeout": 30,
    })

    if is_pooler_url(url):
        kwargs["connect_args"] = {
            "statement_cache_size": 0,
            "command_timeout": 60.0,
            "timeout": 20.0,
        }
        kwargs["pool_size"] = 15
        # The pooler already multiplexes; a local overflow pool on top of it just
        # queues requests in the wrong place and multiplies the backend
        # connections we are trying not to exhaust.
        kwargs["max_overflow"] = 0

    # pool_pre_ping stays on in both branches: a pooler restart (or a rotated DB
    # password) kills every pooled connection, and pre-ping is what turns that
    # into one transparent retry instead of an outage.
    return kwargs


def _setup_db_tracing(engine: AsyncEngine) -> None:
    """Attach SQLAlchemy event listeners to trace all database queries and execution times."""
    sync_engine = engine.sync_engine
    if getattr(sync_engine, "_tracing_attached", False):
        return
    sync_engine._tracing_attached = True

    db_logger = get_logger("app.db.trace")

    @event.listens_for(sync_engine, "before_cursor_execute")
    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault("query_start_time", []).append(time.monotonic())
        stmt_clean = statement.strip()
        op = stmt_clean.split()[0].upper() if stmt_clean else "SQL"
        raw_params = dict(parameters) if isinstance(parameters, dict) else parameters
        cleaned_params = _clean_db_trace_param(raw_params)
        db_logger.debug(
            f"[DATABASE - {op}] Executing query",
            extra={
                "operation": op,
                "statement": stmt_clean,
                "parameters": redact(cleaned_params),
                "executemany": executemany,
            },
        )

    @event.listens_for(sync_engine, "after_cursor_execute")
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        start_times = conn.info.get("query_start_time", [])
        duration_ms = int((time.monotonic() - start_times.pop()) * 1000) if start_times else 0
        stmt_clean = statement.strip()
        op = stmt_clean.split()[0].upper() if stmt_clean else "SQL"
        rowcount = getattr(cursor, "rowcount", -1)

        # Track query count and duration in request state if active
        metrics = _request_db_metrics.get()
        if metrics is not None:
            metrics["query_count"] += 1
            metrics["total_duration_ms"] += duration_ms

        if duration_ms > 1000:
            db_logger.warning(
                f"[DATABASE - SLOW QUERY] Query exceeded 1000ms threshold: {duration_ms}ms (op: {op})",
                extra={
                    "operation": op,
                    "duration_ms": duration_ms,
                    "rowcount": rowcount,
                    "statement": stmt_clean[:300],
                },
            )
        else:
            db_logger.debug(
                f"[DATABASE - {op} RESULT] Completed in {duration_ms}ms (rows: {rowcount})",
                extra={
                    "operation": op,
                    "duration_ms": duration_ms,
                    "rowcount": rowcount,
                },
            )

    @event.listens_for(sync_engine, "handle_error")
    def handle_error(context):
        db_logger.error(
            f"[DATABASE - ERROR] SQL execution failed: {context.original_exception}",
            extra={
                "statement": getattr(context, "statement", None),
                "error": str(context.original_exception),
            },
        )


def _make_engine(url: str, echo: bool) -> AsyncEngine:
    engine = create_async_engine(
        url,
        json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
        **_engine_kwargs(url, echo),
    )
    _setup_db_tracing(engine)
    return engine


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = _make_engine(settings.database_url, settings.db_echo)
        _logger.info("db_engine_created", extra={"dialect": _engine.dialect.name})
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(), expire_on_commit=False, autoflush=False
        )
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope: commit on success, rollback on error."""
    maker = get_sessionmaker()
    async with maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped session."""
    maker = get_sessionmaker()
    async with maker() as session:
        yield session


async def create_all() -> None:
    """Create every table for disposable SQLite dev/test databases only."""
    from app.db.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _logger.info("db_schema_created", extra={"dialect": engine.dialect.name})


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


def set_engine_for_tests(engine: Any) -> None:
    """Point the app's singletons at an existing engine. Test-only.

    Lets HTTP-level tests share the same database as the fixture that seeded it.
    """
    global _engine, _sessionmaker
    _engine = engine
    _sessionmaker = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def reset_for_tests() -> None:
    """Drop and recreate. Test-only."""
    from app.db.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
