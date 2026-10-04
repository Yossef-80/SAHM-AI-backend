"""Logger for the database providers (LangGraph checkpointer + store).

Writes to `logs/providers/db.log`. Import it with::

    from app.ai.loggers.providers.db import logger
"""

from __future__ import annotations

from app.ai.enums import LogKind
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.PROVIDER, "db")

__all__ = ["logger"]
