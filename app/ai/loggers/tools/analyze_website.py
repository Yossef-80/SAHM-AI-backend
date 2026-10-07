"""Logger for the analyze_website tool.

Writes to `logs/tools/analyze_website.log`. Import it with::

    from app.ai.loggers.tools.analyze_website import logger
"""

from __future__ import annotations

from app.ai.enums import LogKind, ToolName
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.TOOL, ToolName.ANALYZE_WEBSITE)

__all__ = ["logger"]
