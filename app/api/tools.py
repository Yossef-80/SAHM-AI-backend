"""Unified tool endpoints: list the whole registry and invoke one tool by name.

Why this exists
---------------
Before these unified endpoints existed, only two of the sixteen agent tools had
a dedicated HTTP endpoint (``POST /creatives/analyze-reference`` and
``POST /creatives/generate``). The catalogue and invoke routes now expose the
same ``Tool`` objects the agent runtime uses, so tools can be listed and invoked
deterministically over HTTP without a second definition of anything.

Two invariants are preserved:

1. ``side_effect="external_write"`` tools are never reachable. The registry
   physically refuses to store one (``ToolRegistry.register``), and
   ``assert_no_external_write_reachable()`` is re-checked on every request as
   defence in depth.
2. Nothing a tool does leaves the app's own tables. ``invocable`` is therefore
   ``not tool.is_external_write()``.

Responses are snake_case: tool outputs are ``ToolOutput`` subclasses (plain
Pydantic, not ``ApiModel``), so they are serialised with ``model_dump()``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    AuthDep,
    LLMDep,
    RequestIdDep,
    SessionDep,
    SettingsDep,
    WorkspaceContext,
    WorkspaceDep,
    require_role,
)
from app.core.agents.personas import PERSONAS, PERSONA_ORDER
from app.core.errors import NotFoundError, ToolNotAllowedError, ValidationFailed
from app.core.llm.base import LLMClient, json_schema_for
from app.core.tools.base import Tool, ToolContext
from app.core.tools.connectors import connector_for
from app.core.tools.connectors import summary as connector_summary
from app.core.tools.settings import load_connector_settings
from app.core.tools.registry import (
    ALL_TOOLS,
    assert_no_external_write_reachable,
    register_all,
)

router = APIRouter(prefix="/tools", tags=["tools"])


class ToolInfo(BaseModel):
    """One catalogue entry. Field names are the wire names (snake_case)."""

    name: str
    description: str
    #: "none" | "draft_write" (an "external_write" tool can never appear here).
    side_effect: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    #: Persona ids (from ``PERSONA_ORDER``) whose ``allowed_tools`` include this.
    personas: list[str] = Field(default_factory=list)
    #: False for a protected (external_write) tool: it may be catalogued, never run.
    invocable: bool = True
    #: Where this tool sends data. See ``app/core/tools/connectors.py`` -- the
    #: single source of truth for every tool -> service URL mapping.
    connector: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ helpers --


def _ctx(
    db: AsyncSession, ws: WorkspaceContext, llm: LLMClient, request_id: str
) -> ToolContext:
    """Same shape as ``creatives._ctx``: nothing ambient, everything explicit."""
    return ToolContext(
        workspace_id=ws.workspace_id,
        business_id=ws.business_id,
        db=db,
        llm=llm,
        request_id=request_id,
    )


def _personas_for(name: str) -> list[str]:
    """Persona ids allowed to call ``name``, in display order."""
    return [
        persona_id
        for persona_id in PERSONA_ORDER
        if name in PERSONAS[persona_id].allowed_tools
    ]


def _resolve(name: str) -> Tool:
    """Look the tool up, enforcing the external-write invariant on the way.

    Defence in depth: ``ToolRegistry.register`` already refuses external-write
    tools, so this guard can never fire -- that is the point. It is re-checked
    per request so the invariant can never be quietly lost.
    """
    registry = register_all()
    assert_no_external_write_reachable(registry)
    if not registry.has(name):
        raise NotFoundError(
            f"tool '{name}' is not registered",
            detail={"name": name, "available": registry.names()},
        )
    return registry.get(name)


def _issues(exc: ValidationError) -> list[dict[str, Any]]:
    """A JSON-safe rendering of Pydantic's issues for the error envelope."""
    return [
        {
            "loc": [str(part) for part in err.get("loc", ())],
            "msg": str(err.get("msg", "")),
            "type": str(err.get("type", "")),
        }
        for err in exc.errors()
    ]


# ----------------------------------------------------------------- catalogue --


@router.get("", response_model=list[ToolInfo])
async def list_tools(settings: SettingsDep) -> list[ToolInfo]:
    """Every tool in ``ALL_TOOLS``, with schemas, persona permissions and URLs."""
    assert_no_external_write_reachable(register_all())
    return [
        ToolInfo(
            name=tool.name,
            description=tool.description,
            side_effect=str(tool.side_effect.value),
            input_schema=json_schema_for(tool.input_model),
            output_schema=json_schema_for(tool.output_model),
            personas=_personas_for(tool.name),
            invocable=not tool.is_external_write(),
            connector=connector_for(tool.name, settings),
        )
        for tool in ALL_TOOLS
    ]


@router.get("/connectors", response_model=dict[str, Any])
async def list_connectors(
    settings: SettingsDep, ws: WorkspaceDep, db: SessionDep
) -> dict[str, Any]:
    """Every tool -> external service URL, including the two protected tools.

    Separate from ``GET /tools`` because it also covers ``apply_change`` and
    ``publish_campaign``, which no agent may call but which still reach Meta.
    """
    assert_no_external_write_reachable(register_all())
    # P2.5: the catalogue reports what *this workspace* has configured, not just
    # what the process inherited, so it agrees with ``GET /settings/connectors``.
    resolved = await load_connector_settings(db, ws.business_id, settings)
    return connector_summary(settings, resolved)


# -------------------------------------------------------------------- invoke --


@router.post("/{name}/invoke", response_model=dict[str, Any])
async def invoke_tool(
    name: str,
    ws: WorkspaceDep,
    db: SessionDep,
    llm: LLMDep,
    request_id: RequestIdDep,
    principal: AuthDep,
    settings: SettingsDep,
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Validate ``payload`` against the tool's input model and run one call.

    The body is the raw arguments object -- exactly what a model emits as
    ``tool_calls[].arguments`` -- so a failing tool is reproducible by hand.

    Invoking a tool spends the workspace's token budget, writes drafts and
    reaches third-party APIs, so it is on the sensitive surface: a ``viewer``
    cannot do it. See ``app/api/deps.py::require_role`` for the anonymous-case
    semantics (``require_auth`` off -> unchanged behaviour for the current
    frontend).
    """
    await require_role(principal, settings, ws, minimum="member")
    tool = _resolve(name)
    if tool.is_external_write():  # pragma: no cover - _resolve guarantees it
        raise ToolNotAllowedError(
            f"tool '{name}' is protected and cannot be invoked over HTTP"
        )

    try:
        validated = tool.input_model.model_validate(payload)
    except ValidationError as exc:
        raise ValidationFailed(
            f"invalid arguments for tool '{name}'",
            detail={"tool": name, "issues": _issues(exc)},
        ) from exc

    output = await tool.handler(validated, _ctx(db, ws, llm, request_id))
    await db.commit()
    return output.model_dump()


__all__ = [
    "ToolInfo",
    "invoke_tool",
    "list_connectors",
    "list_tools",
    "router",
]