"""SQLite persistence. Selected when ``AI_DB_PROVIDER=sqlite``.

One file on disk (``AI_SQLITE_PATH``, default ``./var/ai_memory.sqlite``) holds
both the LangGraph checkpoints (conversation/thread state) and the long-term
store (brand profiles + learnings saved by agents).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from app.ai.errors import ProviderNotInstalledError
from app.ai.settings import AISettings


def _connect(settings: AISettings) -> sqlite3.Connection:
    path = Path(settings.sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
    return conn


def build_checkpointer(settings: AISettings) -> BaseCheckpointSaver:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover - installed by default
        raise ProviderNotInstalledError("sqlite", "langgraph-checkpoint-sqlite") from exc
    saver = SqliteSaver(_connect(settings))
    saver.setup()
    return saver


def build_store(settings: AISettings) -> BaseStore:
    try:
        from langgraph.store.sqlite import SqliteStore
    except ImportError as exc:  # pragma: no cover - installed by default
        raise ProviderNotInstalledError("sqlite", "langgraph-checkpoint-sqlite") from exc
    store = SqliteStore(_connect(settings))
    store.setup()
    return store
