"""Persona endpoints: GET /personas, PATCH /personas/{id}."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import AuthDep, SessionDep, SettingsDep, WorkspaceDep, require_role
from app.core.agents.persona import load_persona, save_persona_override
from app.core.agents.personas import PERSONAS, PERSONA_ORDER, all_personas
from app.schemas.personas import PersonaResponse, PersonaUpdateRequest

router = APIRouter(prefix="/personas", tags=["personas"])


def _to_response(persona, *, is_overridden: bool) -> PersonaResponse:
    return PersonaResponse(
        id=persona.id,
        name=persona.name,
        emoji=persona.emoji,
        role=persona.role,
        description=persona.description,
        tone=persona.tone,
        autonomy=persona.autonomy,
        language=persona.language,
        allowed_tools=list(persona.allowed_tools),
        allowed_tasks=list(persona.allowed_tasks),
        editable_fields=list(persona.editable_fields),
        system_prompt=persona.system_prompt,
        is_overridden=is_overridden,
    )


@router.get("", response_model=list[PersonaResponse])
async def list_personas(
    ws: WorkspaceDep, db: SessionDep
) -> list[PersonaResponse]:
    """All personas, with this workspace's overrides applied."""
    out: list[PersonaResponse] = []
    for persona_id in PERSONA_ORDER:
        base = PERSONAS.get(persona_id)
        if base is None:
            continue
        merged = await load_persona(persona_id, db, ws.workspace_id)
        overridden = (
            merged.name != base.name
            or merged.tone != base.tone
            or merged.autonomy != base.autonomy
            or merged.language != base.language
        )
        out.append(_to_response(merged, is_overridden=overridden))
    return out


@router.patch("/{persona_id}", response_model=PersonaResponse)
async def update_persona(
    persona_id: str,
    payload: PersonaUpdateRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> PersonaResponse:
    """Edit a persona. Only editable_fields are accepted.

    Reads stay open to every role; the write needs ``member`` or better. A
    persona override decides how an agent behaves -- its tone, its autonomy
    ceiling -- so a ``viewer`` editing one would be an escalation, not a typo.
    """
    await require_role(principal, settings, ws, minimum="member")
    if persona_id not in PERSONAS:
        from app.core.errors import NotFoundError

        raise NotFoundError(f"persona '{persona_id}' not found")
    changed = payload.changed_fields()
    if not changed:
        from app.core.errors import ValidationFailed

        raise ValidationFailed("no editable fields were provided")

    # ``changed_fields()`` reports the *names* that were set, so the values have
    # to be re-read here -- passing that set straight through raised
    # ``'set' object has no attribute 'items'`` inside the repository.
    fields = {
        name: value
        for name, value in payload.model_dump(exclude_unset=True).items()
        if value is not None
    }
    await save_persona_override(
        db, workspace_id=ws.workspace_id, persona_id=persona_id, fields=fields
    )
    await db.commit()
    merged = await load_persona(persona_id, db, ws.workspace_id)
    return _to_response(merged, is_overridden=True)


__all__ = ["all_personas", "list_personas", "update_persona"]
