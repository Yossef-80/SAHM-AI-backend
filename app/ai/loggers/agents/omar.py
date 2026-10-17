"""Logger for the Omar (strategist) agent.

Writes to `logs/agents/omar.log`. Import it with::

    from app.ai.loggers.agents.omar import logger
"""

from __future__ import annotations

from app.ai.enums import AgentName, LogKind
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.AGENT, AgentName.OMAR)

__all__ = ["logger"]
