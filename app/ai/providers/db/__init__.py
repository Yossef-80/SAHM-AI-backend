"""Database providers: LangGraph persistence selected by enum.

    from app.ai.providers.db import get_checkpointer, get_store
    graph = builder.compile(checkpointer=get_checkpointer(), store=get_store())

* checkpointer -> short-term memory: the state of each conversation thread.
* store        -> long-term memory: brand profiles + learnings shared by ALL agents.

Every module in this folder exposes:

    def build_checkpointer(settings) -> BaseCheckpointSaver
    def build_store(settings) -> BaseStore
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from app.ai.enums import DatabaseProvider
from app.ai.loggers.providers.db import logger
from app.ai.providers._loader import load_provider_module
from app.ai.settings import get_ai_settings

_PROVIDER_MODULES: dict[DatabaseProvider, str] = {
    DatabaseProvider.MEMORY: "app.ai.providers.db.memory",
    DatabaseProvider.SQLITE: "app.ai.providers.db.sqlite",
    DatabaseProvider.POSTGRES: "app.ai.providers.db.postgres",
}


@lru_cache(maxsize=4)
def _checkpointer(choice: DatabaseProvider) -> BaseCheckpointSaver:
    module = load_provider_module(_PROVIDER_MODULES, choice, "db")
    saver = module.build_checkpointer(get_ai_settings())
    logger.info("checkpointer_built", provider=choice.value, type=type(saver).__name__)
    return saver


@lru_cache(maxsize=4)
def _store(choice: DatabaseProvider) -> BaseStore:
    module = load_provider_module(_PROVIDER_MODULES, choice, "db")
    store = module.build_store(get_ai_settings())
    logger.info("store_built", provider=choice.value, type=type(store).__name__)
    return store


def get_checkpointer(provider: DatabaseProvider | None = None) -> BaseCheckpointSaver:
    """Process-wide checkpointer for the selected backend (``AI_DB_PROVIDER``)."""
    return _checkpointer(provider or get_ai_settings().db_provider)


def get_store(provider: DatabaseProvider | None = None) -> BaseStore:
    """Process-wide long-term store for the selected backend (``AI_DB_PROVIDER``)."""
    return _store(provider or get_ai_settings().db_provider)


def current_store() -> BaseStore:
    """The store of the running graph, or the process-wide one outside a graph.

    Tools call this so they work both inside a LangGraph run (where the graph's
    store is injected) and when invoked directly (scripts, tests).
    """
    try:
        from langgraph.config import get_store as graph_store

        store = graph_store()
        if store is not None:
            return store
    except RuntimeError:  # not inside a graph run
        pass
    return get_store()


def reset_db_providers() -> None:
    _checkpointer.cache_clear()
    _store.cache_clear()


__all__ = ["current_store", "get_checkpointer", "get_store", "reset_db_providers"]
