"""Async engine/session management.

Supports both Postgres (asyncpg) and SQLite (aiosqlite) from the same
``DATABASE_URL``. Alembic migrations are the source of truth for schema in
Postgres; ``create_all`` is used for dev/test on SQLite.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.core.logging import get_logger

_logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _make_engine(url: str, echo: bool) -> AsyncEngine:
    kwargs: dict[str, Any] = {"future": True, "echo": echo}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Postgres: sane pool defaults, pre-ping to survive restarts.
        kwargs.update({"pool_pre_ping": True, "pool_size": 10, "max_overflow": 20})
    return create_async_engine(url, **kwargs)


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
    """Create every table. Used for dev/test and for the mock-provider boot."""
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
