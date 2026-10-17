"""Logger for the Salma (campaign optimizer) agent.

Writes to `logs/agents/salma.log`. Import it with::

    from app.ai.loggers.agents.salma import logger
"""

from __future__ import annotations

from app.ai.enums import AgentName, LogKind
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.AGENT, AgentName.SALMA)

__all__ = ["logger"]
