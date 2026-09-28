"""API-level persona schemas.

Only ``editable_fields`` (name, tone, autonomy, language) may be changed via
PATCH. ``allowed_tools`` / ``allowed_tasks`` are never user-editable.
"""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import ApiModel, Autonomy, Language, Tone

#: The complete, closed list of fields the UI's "Edit persona" may change.
PERSONA_EDITABLE_FIELDS: tuple[str, ...] = ("name", "tone", "autonomy", "language")


class PersonaResponse(ApiModel):
    id: str
    name: str
    emoji: str
    role: str
    description: str
    tone: Tone
    autonomy: Autonomy
    language: Language
    allowed_tools: list[str]
    allowed_tasks: list[str]
    editable_fields: list[str] = Field(default_factory=lambda: list(PERSONA_EDITABLE_FIELDS))
    system_prompt: str = ""
    is_overridden: bool = False


class PersonaUpdateRequest(ApiModel):
    """Only the editable fields are accepted; anything else is a 422."""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    tone: Tone | None = None
    autonomy: Autonomy | None = None
    language: Language | None = None

    def changed_fields(self) -> set[str]:
        return {k for k, v in self.model_dump(exclude_unset=True).items() if v is not None}
