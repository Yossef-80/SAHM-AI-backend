"""The single registry of brand-memory kinds.

Before this module every kind was an implicit string: ``MEMORY_KINDS`` listed
seven, the writers used ``brand_identity`` / ``strategy_blueprint`` /
``market_research`` / ``market_gaps`` / ``website_analysis`` /
``chat_attachment`` as well, and the retrieval path only searched the original
seven. Six kinds were therefore written and never retrievable, and nothing
recorded what a kind *is* -- its lifetime, its weight, or which tasks it serves.

A kind is now a row in :data:`MEMORY_KIND_SPECS`, and every part of the memory
system reads it from here:

* ``layer``       -- which section of the prompt it belongs to (identity, plan,
  episodic, learning). The assembler budgets by layer.
* ``ttl_days``    -- how long an item stays eligible. ``None`` means it does not
  expire. Expiry is materialised into ``BrandMemoryItem.expires_at`` at write
  time so the read path is a plain indexed predicate, not Python arithmetic.
* ``importance``  -- the prior weight of the kind before any per-item signal.
* ``max_tokens``  -- the per-item cap applied on write. A memory item is
  injected into a prompt verbatim, so an unbounded row is a prompt-injection
  surface.
* ``tasks``       -- the task groups this kind is useful for. Retrieval gives an
  affinity bonus only inside those groups; a ``market_research`` note should not
  outrank a creative brief when Layla is writing copy.
* ``api_writable``-- whether ``POST /memory/items`` may create it. Service-owned
  kinds (``brand_identity``, ``strategy_blueprint``, research artifacts) are
  produced by their own pipeline and would only be corrupted by hand edits.

Adding a kind = adding one row here. Nothing else in the codebase enumerates
kinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MemoryLayer = Literal["identity", "plan", "episodic", "learning"]

#: Task groups used for affinity scoring. ``app/core/memory/kinds.py::task_group``
#: maps every concrete task name onto one of these.
TASK_GROUPS: tuple[str, ...] = (
    "research",
    "strategy",
    "creative",
    "analytics",
    "optimization",
)

#: Concrete task name -> group. Mirrors ``app/core/agents/tasks.py``; unknown
#: task names fall back to "general", which matches every kind at affinity 0.
TASK_GROUP_BY_NAME: dict[str, str] = {
    "competitor_research": "research",
    "market_analysis": "research",
    "market_gaps": "research",
    "marketing_strategy": "strategy",
    "campaign_strategy": "strategy",
    "content_calendar": "strategy",
    "creative_brief": "creative",
    "commercial_script": "creative",
    "image_prompt": "creative",
    "video_prompt": "creative",
    "ad_copy": "creative",
    "ad_variations": "creative",
    "performance_analysis": "analytics",
    "optimization_proposal": "optimization",
}


@dataclass(frozen=True)
class MemoryKindSpec:
    """Everything the memory system needs to know about one kind."""

    name: str
    label: str
    layer: MemoryLayer
    ttl_days: int | None
    importance: float
    max_tokens: int
    tasks: tuple[str, ...]
    api_writable: bool = True
    #: Retrieval slice. ``identity`` kinds are always considered before episodic
    #: ones regardless of score, because they are the brand's standing facts.
    priority: int = 0

    def serves(self, task_group: str) -> bool:
        return task_group in self.tasks


MEMORY_KIND_SPECS: dict[str, MemoryKindSpec] = {
    # ------------------------------------------------------------ identity --
    s.name: s
    for s in (
        MemoryKindSpec(
            name="brand_identity",
            label="Brand identity",
            layer="identity",
            ttl_days=None,
            importance=0.85,
            max_tokens=200,
            tasks=TASK_GROUPS,  # every task benefits from knowing the brand
            api_writable=False,
            priority=10,
        ),
        MemoryKindSpec(
            name="voice_example",
            label="Brand voice example",
            layer="identity",
            ttl_days=None,
            importance=0.70,
            max_tokens=120,
            tasks=("creative", "strategy"),
            priority=8,
        ),
        MemoryKindSpec(
            name="visual_style",
            label="Visual style",
            layer="identity",
            ttl_days=None,
            importance=0.70,
            max_tokens=120,
            tasks=("creative",),
            priority=8,
        ),
        MemoryKindSpec(
            name="offer",
            label="Offer",
            layer="identity",
            ttl_days=None,
            importance=0.70,
            max_tokens=120,
            tasks=("strategy", "creative"),
            priority=8,
        ),
        MemoryKindSpec(
            name="audience",
            label="Target audience",
            layer="identity",
            ttl_days=None,
            importance=0.70,
            max_tokens=120,
            tasks=("strategy", "research", "creative"),
            priority=8,
        ),
        # ------------------------------------------------------------- plan --
        MemoryKindSpec(
            name="strategy_blueprint",
            label="Active plan",
            layer="plan",
            ttl_days=90,
            importance=0.80,
            max_tokens=400,
            tasks=("strategy", "optimization"),
            api_writable=False,
            priority=9,
        ),
        # --------------------------------------------------------- episodic --
        MemoryKindSpec(
            name="competitor_note",
            label="Competitor note",
            layer="episodic",
            ttl_days=180,
            importance=0.55,
            max_tokens=150,
            tasks=("research", "strategy"),
            priority=5,
        ),
        MemoryKindSpec(
            name="market_research",
            label="Market research",
            layer="episodic",
            ttl_days=90,
            importance=0.50,
            max_tokens=150,
            tasks=("research", "strategy"),
            api_writable=False,
            priority=5,
        ),
        MemoryKindSpec(
            name="market_gaps",
            label="Market gaps",
            layer="episodic",
            ttl_days=90,
            importance=0.55,
            max_tokens=150,
            tasks=("research", "strategy"),
            api_writable=False,
            priority=5,
        ),
        MemoryKindSpec(
            name="website_analysis",
            label="Website analysis",
            layer="episodic",
            ttl_days=120,
            importance=0.45,
            max_tokens=150,
            tasks=("research",),
            api_writable=False,
            priority=4,
        ),
        MemoryKindSpec(
            name="reference_analysis",
            label="Reference analysis",
            layer="episodic",
            ttl_days=365,
            importance=0.50,
            max_tokens=150,
            tasks=("creative",),
            priority=4,
        ),
        MemoryKindSpec(
            name="chat_attachment",
            label="Chat attachment",
            layer="episodic",
            ttl_days=7,
            importance=0.10,
            max_tokens=40,
            tasks=(),
            api_writable=False,
            priority=0,
        ),
        # --------------------------------------------------------- learning --
        MemoryKindSpec(
            name="learning",
            label="Learning",
            layer="learning",
            ttl_days=None,
            importance=0.60,
            max_tokens=120,
            tasks=TASK_GROUPS,
            priority=6,
        ),
    )
}

#: The kind names, in declaration order. Kept as a tuple because callers
#: (API validation, tests, docs) treat it as an enumeration.
MEMORY_KINDS: tuple[str, ...] = tuple(MEMORY_KIND_SPECS)

#: Kinds a client may create through ``POST /memory/items``.
API_WRITABLE_KINDS: tuple[str, ...] = tuple(
    spec.name for spec in MEMORY_KIND_SPECS.values() if spec.api_writable
)

#: Kinds that make up the "standing facts" slice of the prompt.
IDENTITY_KINDS: tuple[str, ...] = tuple(
    spec.name for spec in MEMORY_KIND_SPECS.values() if spec.layer == "identity"
)

#: Kinds that make up the "active plan" slice.
PLAN_KINDS: tuple[str, ...] = tuple(
    spec.name for spec in MEMORY_KIND_SPECS.values() if spec.layer == "plan"
)

#: Kinds retrieved as evidence/notes.
EPISODIC_KINDS: tuple[str, ...] = tuple(
    spec.name for spec in MEMORY_KIND_SPECS.values() if spec.layer == "episodic"
)

LEARNING_KINDS: tuple[str, ...] = ("learning",)


def task_group(task_name: str | None) -> str:
    """Map a task name onto its affinity group (``"general"`` when unknown)."""
    if not task_name:
        return "general"
    return TASK_GROUP_BY_NAME.get(task_name, "general")


def spec_for(kind: str) -> MemoryKindSpec:
    """The spec for ``kind``. Unknown kinds raise -- silently accepting a typo
    is how six un-retrievable kinds accumulated in the first place."""
    try:
        return MEMORY_KIND_SPECS[kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown memory kind '{kind}'; expected one of {', '.join(MEMORY_KINDS)}"
        ) from exc


def is_known_kind(kind: str) -> bool:
    return kind in MEMORY_KIND_SPECS


def label_for(kind: str) -> str:
    spec = MEMORY_KIND_SPECS.get(kind)
    return spec.label if spec else kind.replace("_", " ").title()


def default_importance(kind: str) -> float:
    spec = MEMORY_KIND_SPECS.get(kind)
    return spec.importance if spec else 0.5


def max_tokens_for(kind: str, fallback: int) -> int:
    spec = MEMORY_KIND_SPECS.get(kind)
    return spec.max_tokens if spec else fallback


__all__ = [
    "API_WRITABLE_KINDS",
    "EPISODIC_KINDS",
    "IDENTITY_KINDS",
    "LEARNING_KINDS",
    "MEMORY_KINDS",
    "MEMORY_KIND_SPECS",
    "PLAN_KINDS",
    "TASK_GROUPS",
    "TASK_GROUP_BY_NAME",
    "MemoryKindSpec",
    "MemoryLayer",
    "default_importance",
    "is_known_kind",
    "label_for",
    "max_tokens_for",
    "spec_for",
    "task_group",
]
