"""Logger for the get_brand_context tool.

Writes to `logs/tools/get_brand_context.log`. Import it with::

    from app.ai.loggers.tools.get_brand_context import logger
"""

from __future__ import annotations

from app.ai.enums import LogKind, ToolName
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.TOOL, ToolName.GET_BRAND_CONTEXT)

__all__ = ["logger"]
