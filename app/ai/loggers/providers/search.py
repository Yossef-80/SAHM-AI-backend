"""Logger for the search providers (Tavily / mock).

Writes to `logs/providers/search.log`. Import it with::

    from app.ai.loggers.providers.search import logger
"""

from __future__ import annotations

from app.ai.enums import LogKind
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.PROVIDER, "search")

__all__ = ["logger"]
