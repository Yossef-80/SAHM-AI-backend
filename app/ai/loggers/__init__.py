"""Per-component and diagnostic loggers for the AI layer.

    loggers/
    ├── base.py        # factory: ComponentLogger -> logs/<kind>s/<name>.log
    ├── callbacks.py   # LangChain callback: auto-logs tool, LLM, and agent lifecycle
    ├── cost.py        # real-time token cost estimation across models
    ├── agents/        # one file per agent + diagnostic.log for issue troubleshooting
    ├── tools/         # one file per tool     (tools/web_search.py -> logs/tools/web_search.log)
    └── providers/     # one file per provider (providers/llm.py -> logs/providers/llm.log)
"""

from app.ai.loggers.agents.diagnostic import log_agent_diagnosis
from app.ai.loggers.base import ComponentLogger, get_component_logger, reset_loggers
from app.ai.loggers.callbacks import AILoggingCallback, AIToolAuditCallback
from app.ai.loggers.cost import estimate_token_cost, format_cost

__all__ = [
    "AILoggingCallback",
    "AIToolAuditCallback",
    "ComponentLogger",
    "estimate_token_cost",
    "format_cost",
    "get_component_logger",
    "log_agent_diagnosis",
    "reset_loggers",
]
