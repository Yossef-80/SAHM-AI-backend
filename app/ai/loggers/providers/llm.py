"""Logger for the LLM providers (Ollama / OpenAI / Anthropic / mock) -- model calls, tokens, latency.

Writes to `logs/providers/llm.log`. Import it with::

    from app.ai.loggers.providers.llm import logger
"""

from __future__ import annotations

from app.ai.enums import LogKind
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.PROVIDER, "llm")

__all__ = ["logger"]
