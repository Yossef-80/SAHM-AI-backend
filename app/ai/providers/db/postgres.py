"""PostgreSQL persistence. Selected when ``AI_DB_PROVIDER=postgres``.

Needs ``uv add langgraph-checkpoint-postgres "psycopg[binary]"`` and
``AI_POSTGRES_URL=postgresql://user:pass@host:5432/db``.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from app.ai.errors import ProviderNotConfiguredError, ProviderNotInstalledError
from app.ai.settings import AISettings


def _connect(settings: AISettings) -> Any:
    if not settings.postgres_url:
        raise ProviderNotConfiguredError("AI_DB_PROVIDER=postgres but AI_POSTGRES_URL is empty.")
    try:
        from psycopg import Connection
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ProviderNotInstalledError("postgres", "psycopg[binary]") from exc
    return Connection.connect(
        settings.postgres_url, autocommit=True, prepare_threshold=0, row_factory=dict_row
    )


def build_checkpointer(settings: AISettings) -> BaseCheckpointSaver:
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise ProviderNotInstalledError("postgres", "langgraph-checkpoint-postgres") from exc
    saver = PostgresSaver(_connect(settings))
    saver.setup()
    return saver


def build_store(settings: AISettings) -> BaseStore:
    try:
        from langgraph.store.postgres import PostgresStore
    except ImportError as exc:
        raise ProviderNotInstalledError("postgres", "langgraph-checkpoint-postgres") from exc
    store = PostgresStore(_connect(settings))
    store.setup()
    return store
