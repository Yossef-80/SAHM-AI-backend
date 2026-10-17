"""Per-agent loggers -- one file per agent, each writing to `logs/agents/<name>.log`."""

from app.ai.loggers.agents.diagnostic import logger as diagnostic_logger, log_agent_diagnosis
from app.ai.loggers.agents.karim import logger as karim_logger
from app.ai.loggers.agents.layla import logger as layla_logger
from app.ai.loggers.agents.nour import logger as nour_logger
from app.ai.loggers.agents.omar import logger as omar_logger
from app.ai.loggers.agents.orchestrator import logger as orchestrator_logger
from app.ai.loggers.agents.salma import logger as salma_logger

__all__ = [
    "diagnostic_logger",
    "karim_logger",
    "layla_logger",
    "log_agent_diagnosis",
    "nour_logger",
    "omar_logger",
    "orchestrator_logger",
    "salma_logger",
]
