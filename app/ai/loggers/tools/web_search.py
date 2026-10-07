"""Logger for the web_search tool.

Writes to `logs/tools/web_search.log`. Import it with::

    from app.ai.loggers.tools.web_search import logger
"""

from __future__ import annotations

from app.ai.enums import LogKind, ToolName
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.TOOL, ToolName.WEB_SEARCH)

__all__ = ["logger"]
