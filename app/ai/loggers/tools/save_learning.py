"""Logger for the save_learning tool.

Writes to `logs/tools/save_learning.log`. Import it with::

    from app.ai.loggers.tools.save_learning import logger
"""

from __future__ import annotations

from app.ai.enums import LogKind, ToolName
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.TOOL, ToolName.SAVE_LEARNING)

__all__ = ["logger"]
