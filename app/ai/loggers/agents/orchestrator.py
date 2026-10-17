"""Logger for the Orchestrator (supervisor & routing) agent.

Writes to `logs/agents/orchestrator.log`. Import it with::

    from app.ai.loggers.agents.orchestrator import logger
"""

from __future__ import annotations

from app.ai.enums import LogKind
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.AGENT, "orchestrator")

__all__ = ["logger"]
