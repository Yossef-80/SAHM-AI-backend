"""Logger for the Layla (creative copy & design) agent.

Writes to `logs/agents/layla.log`. Import it with::

    from app.ai.loggers.agents.layla import logger
"""

from __future__ import annotations

from app.ai.enums import AgentName, LogKind
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.AGENT, AgentName.LAYLA)

__all__ = ["logger"]
