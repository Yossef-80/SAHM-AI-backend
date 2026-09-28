"""Personas are data, not code.

One generic runtime is configured by persona objects. Adding a persona means
adding a dict entry in ``personas.py`` -- never a new class.

Overrides (name / tone / autonomy / language) are stored per workspace in
``persona_overrides`` and merged over the defaults at load time.
``allowed_tools`` and ``allowed_tasks`` are NOT user-editable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.schemas.common import Autonomy, Language, Tone
from app.schemas.personas import PERSONA_EDITABLE_FIELDS

_logger = get_logger(__name__)


class Persona(BaseModel):
    """A configured agent identity."""

    id: str
    name: str
    emoji: str
    role: str
    description: str
    tone: Tone = Tone.PROFESSIONAL
    autonomy: Autonomy = Autonomy.SUGGEST
    language: Language = Language.EN
    allowed_tools: list[str] = Field(default_factory=list)
    allowed_tasks: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    #: What the UI's "Edit persona" may change. Closed list, enforced on PATCH.
    editable_fields: list[str] = Field(default_factory=lambda: list(PERSONA_EDITABLE_FIELDS))

    # ------------------------------------------------------------- helpers --

    def can_use_tool(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools

    def can_run_task(self, task_name: str) -> bool:
        return task_name in self.allowed_tasks

    def with_overrides(self, overrides: dict[str, Any]) -> "Persona":
        """Return a copy with editable fields replaced.

        Only keys in ``editable_fields`` are honoured; anything else is ignored
        so a bad override row cannot widen a persona's permissions.
        """
        safe = {k: v for k, v in overrides.items() if k in self.editable_fields and v is not None}
        if not safe:
            return self
        return self.model_copy(update=safe)


# ------------------------------------------------------------------- loading --


async def load_persona(
    persona_id: str, db: Any, workspace_id: str = "default"
) -> Persona:
    """Defaults + workspace overrides, merged."""
    from app.core.agents.personas import PERSONAS

    base = PERSONAS.get(persona_id)
    if base is None:
        from app.core.errors import NotFoundError

        raise NotFoundError(
            f"persona '{persona_id}' not found", detail={"available": sorted(PERSONAS)}
        )

    overrides = await _load_overrides(db, workspace_id, persona_id)
    persona = base.with_overrides(overrides)
    if overrides:
        _logger.info(
            "persona_overrides_applied",
            extra={"persona_id": persona_id, "workspace_id": workspace_id},
        )
    return persona


async def _load_overrides(db: Any, workspace_id: str, persona_id: str) -> dict[str, Any]:
    from sqlalchemy import select

    from app.db.models import PersonaOverride

    stmt = select(PersonaOverride).where(
        PersonaOverride.workspace_id == workspace_id,
        PersonaOverride.persona_id == persona_id,
    )
    row = (await db.execute(stmt)).scalars().first()
    if row is None:
        return {}
    return {
        "name": row.name,
        "tone": row.tone,
        "autonomy": row.autonomy,
        "language": row.language,
    }


async def save_persona_override(
    db: Any,
    *,
    workspace_id: str,
    persona_id: str,
    fields: dict[str, Any],
) -> None:
    """Persist an editable-field override for a workspace."""
    from sqlalchemy import select

    from app.core.agents.personas import PERSONAS
    from app.core.errors import PolicyViolation
    from app.db.models import PersonaOverride

    base = PERSONAS.get(persona_id)
    if base is None:
        from app.core.errors import NotFoundError

        raise NotFoundError(f"persona '{persona_id}' not found")

    illegal = sorted(set(fields) - set(base.editable_fields))
    if illegal:
        raise PolicyViolation(
            f"fields {illegal} are not editable for persona '{persona_id}'",
            detail={"editable": base.editable_fields},
        )

    stmt = select(PersonaOverride).where(
        PersonaOverride.workspace_id == workspace_id,
        PersonaOverride.persona_id == persona_id,
    )
    row = (await db.execute(stmt)).scalars().first()
    if row is None:
        row = PersonaOverride(
            id=f"po_{persona_id}_{workspace_id}",
            workspace_id=workspace_id,
            persona_id=persona_id,
        )
        db.add(row)
    for key, value in fields.items():
        if value is not None:
            setattr(row, key, value)
    await db.flush()


__all__ = ["Persona", "load_persona", "save_persona_override"]
