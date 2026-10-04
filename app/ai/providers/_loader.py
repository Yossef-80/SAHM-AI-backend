"""Lazy enum -> module loader shared by every provider family.

Each provider family (llm / search / db) declares a map like::

    {LLMProvider.OLLAMA: "app.ai.providers.llm.ollama", ...}

and calls ``load_provider_module``. The module is imported ONLY when its enum
value is selected, so e.g. ``langchain-openai`` never needs to be installed
unless ``AI_LLM_PROVIDER=openai``.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from enum import Enum
from types import ModuleType

from app.ai.errors import ProviderNotConfiguredError


def load_provider_module(registry: Mapping[Enum, str], choice: Enum, family: str) -> ModuleType:
    """Import and return the module registered for ``choice``."""
    try:
        module_path = registry[choice]
    except KeyError as exc:
        supported = ", ".join(str(k.value) for k in registry)
        raise ProviderNotConfiguredError(
            f"No {family} provider module is registered for '{choice}'. Supported: {supported}"
        ) from exc
    return importlib.import_module(module_path)


__all__ = ["load_provider_module"]
