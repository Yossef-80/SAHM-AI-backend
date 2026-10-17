"""Shared long-term memory helpers (brand profile + learnings)."""

from app.ai.memory.brand_memory import (
    DEFAULT_BUSINESS_ID,
    BrandProfile,
    Learning,
    get_brand_profile,
    learnings_namespace,
    profile_namespace,
    put_brand_profile,
)

__all__ = [
    "DEFAULT_BUSINESS_ID",
    "BrandProfile",
    "Learning",
    "get_brand_profile",
    "learnings_namespace",
    "profile_namespace",
    "put_brand_profile",
]

# The single per-workspace post-turn memory pass. Imported lazily by callers so
# that importing this package never starts asyncio work.
from app.ai.memory.consolidator import MemoryConsolidator, memory_consolidator  # noqa: E402

__all__ += ["MemoryConsolidator", "memory_consolidator"]
