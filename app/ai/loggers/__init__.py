"""Per-component loggers for the AI layer.

    loggers/
    ├── base.py        # factory: ComponentLogger -> logs/<kind>s/<name>.log
    ├── callbacks.py   # LangChain callback: auto-logs every tool + LLM call
    ├── agents/        # one file per agent    (agents/nour.py   -> logs/agents/nour.log)
    ├── tools/         # one file per tool     (tools/web_search.py -> logs/tools/web_search.log)
    └── providers/     # one file per provider (providers/llm.py -> logs/providers/llm.log)
"""

from app.ai.loggers.base import ComponentLogger, get_component_logger, reset_loggers
from app.ai.loggers.callbacks import AILoggingCallback

__all__ = ["AILoggingCallback", "ComponentLogger", "get_component_logger", "reset_loggers"]
