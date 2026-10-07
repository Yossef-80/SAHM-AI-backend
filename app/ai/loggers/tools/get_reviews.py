"""Logger for the get_reviews tool.

Writes to `logs/tools/get_reviews.log`. Import it with::

    from app.ai.loggers.tools.get_reviews import logger
"""
from __future__ import annotations

from app.ai.enums import LogKind
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.TOOL, "get_reviews")

__all__ = ["logger"]
