"""Which model an agent actually runs on, per kind of work.

Before this module every agent graph did this::

    llm_handle = get_llm(provider=get_ai_settings().llm_provider)

so the workspace's tier settings, its model choice and its sampling controls were
all bypassed, and routing, competitor research, performance analysis and ad copy
shared one model at one temperature.

Resolution order, most specific first:

1. **A handle the run already resolved** (``configurable["agent_llm_handle"]``).
   The orchestrator builds one per turn through
   :func:`app.ai.providers.llm.get_workspace_chat_llm`, so workspace provider,
   key, endpoint, model and sampling all apply, and the database is read once per
   turn instead of once per graph node.
2. **The tier default with the role's temperature** -- through the ``builder``
   the caller passes in (normally ``app.ai.providers.llm.get_llm``), so a graph
   keeps the module-level seam tests and embedding code already patch.

Both paths end at the same ``LLMHandle``, so a graph never has to know which one
it got.
"""

from __future__ import annotations

from typing import Any, Callable

from app.ai.enums import LLMProvider
from app.ai.roles import RoleSpec, resolve_role

#: Which role an agent graph runs as. A graph is one role for its whole turn:
#: Layla writes, Karim and Salma read metrics, Nour investigates, Omar plans.
AGENT_ROLE: dict[str, str] = {
    "layla": "creative",
    "nour": "research",
    "karim": "analysis",
    "salma": "analysis",
    "omar": "strategy",
}

#: Cache for the process-default path only. It is keyed on environment-derived
#: values that do not change while the process runs, so unlike a workspace handle
#: it can never go stale against a rotated key or an edited settings row. A caller
#: that passes its own ``builder`` opts out of the cache entirely -- a stubbed
#: builder in a test must be called on every turn, not remembered from the last.
_PROCESS_HANDLES: dict[tuple[str, float, int | None], Any] = {}


def _configurable(config: Any) -> dict[str, Any]:
    """The ``configurable`` mapping of a RunnableConfig, or an empty dict."""
    if isinstance(config, dict):
        return config.get("configurable") or {}
    return getattr(config, "configurable", None) or {}


def role_for_agent(agent_name: str) -> str:
    return AGENT_ROLE.get(agent_name, "analysis")


def _temperature_overrides(settings: Any) -> dict[str, float] | None:
    value = getattr(settings, "llm_temperature_overrides", None)
    return value if isinstance(value, dict) else None


def _tier_target(settings: Any, tier: str) -> tuple[LLMProvider | None, str | None]:
    """``(provider, model)`` for a tier, tolerant of stub settings objects."""
    provider: Any = None
    resolver = getattr(settings, "provider_for_tier", None)
    if callable(resolver):
        try:
            provider = resolver(tier)
        except Exception:  # pragma: no cover - defensive
            provider = None
    if provider is None:
        provider = getattr(settings, "llm_provider", None)
    if isinstance(provider, str):
        try:
            provider = LLMProvider(provider)
        except ValueError:
            provider = None
    model = getattr(settings, f"llm_model_{tier}", None)
    return provider, (str(model) if model else None)


def process_handle(
    role: str,
    *,
    builder: Callable[..., Any] | None = None,
    settings: Any = None,
) -> Any:
    """The tier-aware handle built from process settings alone.

    This is the fallback for call paths with no workspace in scope -- direct tool
    invocation, scripts, tests -- and it is what makes ``LLM_MODEL_STRONG`` mean
    something even without a settings row.
    """
    if settings is None or builder is None:
        from app.ai.providers.llm import get_llm
        from app.ai.settings import get_ai_settings

        settings = settings or get_ai_settings()
        builder = builder or get_llm

    spec = resolve_role(role, _temperature_overrides(settings))
    provider, model = _tier_target(settings, spec.tier)
    use_cache = builder is not None and builder.__name__ == "get_llm"

    key = (
        f"{getattr(provider, 'value', provider)}:{model or ''}:{spec.temperature}:{spec.max_tokens}",
        spec.temperature,
        spec.max_tokens,
    )
    if use_cache:
        cached = _PROCESS_HANDLES.get(key)
        if cached is not None:
            return cached

    build_kwargs: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    }
    # `role_spec` is the marker of a real AISettings. A trimmed settings object
    # (a stub, a partial namespace) must not be handed to the provider factory,
    # which would read `llm_model`/`gemini_api_key` off it and crash.
    if hasattr(settings, "role_spec"):
        build_kwargs["settings"] = settings
    handle = builder(**build_kwargs)
    if use_cache:
        _PROCESS_HANDLES[key] = handle
    return handle


def agent_llm(
    role: str,
    config: Any,
    *,
    builder: Callable[..., Any] | None = None,
    settings: Any = None,
) -> Any:
    """The chat handle for ``role`` in this run.

    Synchronous on purpose: it is called from inside a LangGraph node, which
    cannot await. Anything that needs the database is resolved by the caller and
    handed in through ``configurable``.
    """
    return agent_llm_with_role_spec(role, config, builder=builder, settings=settings)[0]


def agent_llm_with_role_spec(
    role: str,
    config: Any,
    *,
    builder: Callable[..., Any] | None = None,
    settings: Any = None,
) -> tuple[Any, RoleSpec]:
    """Same as :func:`agent_llm`, plus the resolved spec (used for logging)."""
    if settings is None:
        from app.ai.settings import get_ai_settings

        settings = get_ai_settings()
    spec = resolve_role(role, _temperature_overrides(settings))

    provided = _configurable(config).get("agent_llm_handle")
    if _can_bind_tools(provided):
        # Already resolved by the orchestrator with this role's tier and
        # temperature already applied to it.
        return provided, spec
    return process_handle(role, builder=builder, settings=settings), spec


def _can_bind_tools(handle: Any) -> bool:
    """True when ``handle`` is a chat model an agent graph can actually run on.

    Every agent node does ``handle.chat.bind_tools(...).invoke(...)``, so a handle
    without that method -- a routing-only chat, a partial stub -- is not a usable
    agent model. Checking the capability (rather than the mere presence of
    ``chat``) keeps the orchestrator's pre-resolved handle as the fast path while
    letting a graph build its own when what it was handed cannot do the job.
    """
    if handle is None:
        return False
    chat = getattr(handle, "chat", None)
    if chat is None:
        return False
    return callable(getattr(chat, "bind_tools", None)) or callable(
        getattr(handle, "bind_tools", None)
    )


def agent_llm_for_graph(
    agent_name: str,
    config: Any,
    *,
    builder: Callable[..., Any] | None = None,
    settings: Any = None,
) -> Any:
    """Convenience for graphs: agent name -> its role -> its handle.

    ``builder`` and ``settings`` default to the real provider factory and process
    settings; a graph passes its own module-level references so patching either
    one keeps working.
    """
    return agent_llm(
        role_for_agent(agent_name),
        config,
        builder=builder,
        settings=settings,
    )


def reset_process_handles() -> None:
    """Drop cached process handles. For tests and settings reloads."""
    _PROCESS_HANDLES.clear()


__all__ = [
    "AGENT_ROLE",
    "agent_llm",
    "agent_llm_for_graph",
    "agent_llm_with_role_spec",
    "process_handle",
    "reset_process_handles",
    "role_for_agent",
]
