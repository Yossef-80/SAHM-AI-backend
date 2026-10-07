"""Logger for the reviews providers.

Writes to `logs/providers/reviews.log`. Import it with::

    from app.ai.loggers.providers.reviews import logger
"""
from __future__ import annotations

from app.ai.enums import LogKind
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.PROVIDER, "reviews")

__all__ = ["logger"]
