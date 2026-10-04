"""In-memory persistence. Selected when ``AI_DB_PROVIDER=memory``.

Everything is lost when the process exits -- ideal for tests and quick demos.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from app.ai.settings import AISettings


def build_checkpointer(settings: AISettings) -> BaseCheckpointSaver:  # noqa: ARG001
    return InMemorySaver()


def build_store(settings: AISettings) -> BaseStore:  # noqa: ARG001
    return InMemoryStore()
