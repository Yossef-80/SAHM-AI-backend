"""What each kind of LLM work needs: a tier, a temperature, and a token budget.

Why this table exists
---------------------
Every agent graph used to build its model the same way::

    get_llm(provider=get_ai_settings().llm_provider)

which meant one model and one temperature (``AI_LLM_TEMPERATURE``, default 0.2)
served routing, vision analysis, competitive research *and* ad copy. A 0.2
temperature is right for extraction and wrong for creative writing: it is the
reason "three variations" came back as the same sentence three times, and the
reason the workspace's strong-tier model never reached Layla at all.

A role names the *work*, not the agent, so the same table can serve a graph, a
service function and a one-off call::

    handle = build_role_handle("creative_copy")

Roles are resolved to a tier here and to a model in
``app.ai.providers.llm.agent_llm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TierName = Literal["fast", "strong"]


@dataclass(frozen=True)
class RoleSpec:
    """The sampling profile for one kind of LLM work."""

    tier: TierName
    #: 0.0 makes a task deterministic, which is what structure extraction wants.
    #: Creative writing needs room; see ``DEFAULTS`` below.
    temperature: float
    #: Optional ceiling override. None keeps the provider/workspace default.
    max_tokens: int | None = None
    #: Human-readable purpose, used in logs (and by the eval harness).
    purpose: str = ""


#: Role -> spec. Temperatures are chosen per *job*, and each one is overridable
#: from the environment via ``LLM_TEMPERATURE_OVERRIDES`` (see ``app/ai/settings``).
#:
#: Creative tiers are deliberately high. Ad copy is the one place in this product
#: where near-duplicate output is a defect rather than a virtue, and the low
#: default was measurably producing exactly that: variations that shared a
#: headline skeleton and an opening clause.
DEFAULTS: dict[str, RoleSpec] = {
    # ---------------------------------------------------------- creative ----
    "creative_copy": RoleSpec("strong", 0.85, purpose="ad copy and copy variations"),
    "creative_script": RoleSpec("strong", 0.7, purpose="commercial scripts and storyboards"),
    "creative_brief": RoleSpec("strong", 0.55, purpose="creative briefs and image/video prompts"),
    "creative": RoleSpec("strong", 0.7, purpose="Layla's conversational creative turns"),
    # ---------------------------------------------------------- analysis ----
    "analysis": RoleSpec("strong", 0.15, purpose="performance analysis and optimization plans"),
    "strategy": RoleSpec("strong", 0.3, purpose="campaign and marketing strategy"),
    "research": RoleSpec("strong", 0.3, purpose="market and competitor research"),
    "critique": RoleSpec("strong", 0.2, purpose="self-review of a drafted deliverable"),
    # --------------------------------------------------------- mechanical ---
    "routing": RoleSpec("fast", 0.0, purpose="turn classification and agent selection"),
    "summary": RoleSpec("fast", 0.1, purpose="conversation and memory summarization"),
    "vision": RoleSpec("fast", 0.0, purpose="reference-image analysis"),
    "extraction": RoleSpec("fast", 0.0, purpose="structured fact extraction"),
}

#: Fallback when a caller names a role that is not in the table. Analysis is the
#: safe middle: it is correct for anything that reads and reports, and it is
#: already the most conservative of the strong-tier profiles.
FALLBACK = RoleSpec("strong", 0.15, purpose="unclassified")


def resolve_role(role: str, overrides: dict[str, float] | None = None) -> RoleSpec:
    """The spec for ``role``, with any per-deployment temperature override applied.

    ``overrides`` comes from ``AISettings.llm_temperature_overrides``; an unknown
    key in it is ignored rather than fatal, because it is operator input.
    """
    spec = DEFAULTS.get(role, FALLBACK)
    if overrides:
        override = overrides.get(role)
        if override is not None:
            try:
                return RoleSpec(spec.tier, float(override), spec.max_tokens, spec.purpose)
            except (TypeError, ValueError):
                pass
    return spec


def parse_temperature_overrides(raw: Any) -> dict[str, float]:
    """Parse ``LLM_TEMPERATURE_OVERRIDES`` -- ``creative_copy=0.9,routing=0.1``.

    Also accepts the JSON object form, and drops anything unusable instead of
    raising: a typo in a diagnostic env var must not stop the app from booting.
    """
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, str):
        if not raw.strip():
            return {}
        text = raw.strip()
        if text.startswith("{"):
            import json

            try:
                parsed = json.loads(text)
            except ValueError:
                return {}
            items = parsed.items() if isinstance(parsed, dict) else []
        else:
            pairs = []
            for chunk in text.split(","):
                if "=" in chunk:
                    key, _, value = chunk.partition("=")
                    pairs.append((key.strip(), value.strip()))
                elif ":" in chunk:
                    key, _, value = chunk.partition(":")
                    pairs.append((key.strip(), value.strip()))
            items = pairs
    else:
        return {}

    out: dict[str, float] = {}
    for key, value in items:
        try:
            out[str(key).strip()] = float(value)
        except (TypeError, ValueError):
            continue
    return out


__all__ = [
    "DEFAULTS",
    "FALLBACK",
    "RoleSpec",
    "TierName",
    "parse_temperature_overrides",
    "resolve_role",
]
