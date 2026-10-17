"""The memory policy: every knob in one immutable object.

Why a policy object instead of reading settings at call sites
-------------------------------------------------------------
The two settings classes (``app.config.Settings`` for the core app and
``app.ai.settings.AISettings`` for the AI layer) both read the same environment
variables, and before this module each call site picked whichever copy it
happened to import. Retrieval weights, caps and the debounce interval were
therefore impossible to reason about as a set, and a change in one class could
silently disagree with the other.

:class:`MemoryPolicy` is built once from *either* object by attribute name, so:

* the store, the ranker and the assembler all read the same numbers;
* tests construct a policy directly with no environment at all;
* an unknown/missing attribute falls back to the documented default, which
  keeps older settings objects working.

Environment names are documented in ``.env.example`` under "MEMORY LAYERS" and
in ``docs/MEMORY_ARCHITECTURE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


def _pick(source: Any, name: str, default: Any) -> Any:
    value = getattr(source, name, None)
    return default if value is None else value


def _pick_float(source: Any, name: str, default: float) -> float:
    try:
        return float(_pick(source, name, default))
    except (TypeError, ValueError):
        return default


def _pick_int(source: Any, name: str, default: int) -> int:
    try:
        return int(_pick(source, name, default))
    except (TypeError, ValueError):
        return default


def _pick_bool(source: Any, name: str, default: bool) -> bool:
    value = _pick(source, name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MemoryPolicy:
    """Retrieval, budgeting and lifecycle rules for brand memory."""

    # ------------------------------------------------------------- master --
    enabled: bool = True

    # ------------------------------------------------- session (turns 1-2) --
    session_max_tokens: int = 10_000
    retain_recent_turns: int = 2
    ltm_debounce_seconds: int = 180
    brand_max_tokens: int = 10_000
    profile_max_tokens: int = 300

    # --------------------------------------------------- prompt assembly ---
    context_token_cap: int = 900
    context_top_k: int = 6
    recent_learnings: int = 3
    identity_share: float = 0.45
    plan_share: float = 0.15

    # ----------------------------------------------------------- ranking ---
    candidate_window: int = 300
    recency_half_life_days: float = 30.0
    weight_affinity: float = 0.20
    weight_lexical: float = 0.40
    weight_recency: float = 0.25
    weight_importance: float = 0.15
    pinned_boost: float = 0.35
    campaign_boost: float = 0.10
    phrase_boost: float = 0.10
    min_relevance: float = 0.12

    # --------------------------------------------------------- lifecycle ---
    dedupe_enabled: bool = True
    default_ttl_days: int = 180
    max_items_per_kind: int = 500
    item_max_tokens: int = 120
    sweep_on_write: bool = True

    # ----------------------------------------------------------------------
    @classmethod
    def from_settings(cls, source: Any = None) -> "MemoryPolicy":
        """Build a policy from ``app.config.Settings`` or ``AISettings``.

        Both classes expose the same attribute names; anything missing falls
        back to this dataclass's default.
        """
        if source is None:
            return cls()

        def f(name: str) -> float:
            return _pick_float(source, f"memory_{name}", getattr(cls, name))

        def i(name: str) -> int:
            return _pick_int(source, f"memory_{name}", getattr(cls, name))

        def b(name: str) -> bool:
            return _pick_bool(source, f"memory_{name}", getattr(cls, name))

        def alias(default: Any, *names: str) -> Any:
            """First attribute that exists on the settings object."""
            for candidate in names:
                value = getattr(source, candidate, None)
                if value is not None:
                    return value
            return default

        return cls(
            enabled=b("enabled"),
            session_max_tokens=int(
                alias(cls.session_max_tokens, "memory_session_max_tokens")
            ),
            retain_recent_turns=max(1, i("retain_recent_turns")),
            ltm_debounce_seconds=max(0, i("ltm_debounce_seconds")),
            brand_max_tokens=int(
                alias(
                    cls.brand_max_tokens,
                    "memory_brand_max_tokens",
                    "brand_memory_token_limit",
                )
            ),
            profile_max_tokens=int(
                alias(
                    cls.profile_max_tokens,
                    "memory_global_profile_max_tokens",
                )
            ),
            context_token_cap=i("context_token_cap"),
            context_top_k=max(1, i("context_top_k")),
            recent_learnings=max(0, i("recent_learnings")),
            identity_share=min(1.0, max(0.0, f("identity_share"))),
            plan_share=min(1.0, max(0.0, f("plan_share"))),
            candidate_window=max(1, i("candidate_window")),
            recency_half_life_days=max(0.1, f("recency_half_life_days")),
            weight_affinity=max(0.0, f("weight_affinity")),
            weight_lexical=max(0.0, f("weight_lexical")),
            weight_recency=max(0.0, f("weight_recency")),
            weight_importance=max(0.0, f("weight_importance")),
            pinned_boost=max(0.0, f("pinned_boost")),
            campaign_boost=max(0.0, f("campaign_boost")),
            phrase_boost=max(0.0, f("phrase_boost")),
            min_relevance=min(1.0, max(0.0, f("min_relevance"))),
            dedupe_enabled=b("dedupe_enabled"),
            default_ttl_days=max(1, i("default_ttl_days")),
            max_items_per_kind=max(1, i("max_items_per_kind")),
            item_max_tokens=max(1, i("item_max_tokens")),
            sweep_on_write=b("sweep_on_write"),
        )

    # -------------------------------------------------------------- derived --
    @property
    def weight_total(self) -> float:
        """Sum of the base weights, used to renormalise when a signal is absent."""
        return (
            self.weight_affinity
            + self.weight_lexical
            + self.weight_recency
            + self.weight_importance
        ) or 1.0

    def identity_budget(self) -> int:
        return max(1, int(self.context_token_cap * self.identity_share))

    def plan_budget(self) -> int:
        return max(1, int(self.context_token_cap * self.plan_share))

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def get_memory_policy(settings: Any | None = None) -> MemoryPolicy:
    """Return the effective policy for this process.

    Resolution order: an explicit ``settings`` object, then the core
    ``app.config.Settings``, then the AI settings. A settings object that has no
    ``memory_*`` attributes falls back to the defaults, so importing this module
    from a script never needs the full app configuration.
    """
    if settings is None:
        try:
            from app.config import get_settings as _core_settings

            settings = _core_settings()
        except Exception:
            try:
                from app.ai.settings import get_ai_settings

                settings = get_ai_settings()
            except Exception:
                settings = None
    return MemoryPolicy.from_settings(settings)


__all__ = ["MemoryPolicy", "get_memory_policy"]
