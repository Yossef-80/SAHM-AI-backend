"""Logger for the Nour (researcher) agent.

Writes to `logs/agents/nour.log`. Import it with::

    from app.ai.loggers.agents.nour import logger
"""

from __future__ import annotations

from app.ai.enums import LogKind, AgentName
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.AGENT, AgentName.NOUR)

__all__ = ["logger"]
