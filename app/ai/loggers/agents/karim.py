"""Logger for the Karim (analytics & insights) agent.

Writes to `logs/agents/karim.log`. Import it with::

    from app.ai.loggers.agents.karim import logger
"""

from __future__ import annotations

from app.ai.enums import AgentName, LogKind
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.AGENT, AgentName.KARIM)

__all__ = ["logger"]
