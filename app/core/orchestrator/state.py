"""Orchestrator state.

This is the single mutable object passed between LangGraph nodes. Everything the
graph needs is declared here; nothing is ambient.
"""

from __future__ import annotations

from typing import Any, TypedDict


class OrchestratorState(TypedDict, total=False):
    """State carried through the graph.

    Required by the brief:
        conversation_id, messages, context, brand_context_ref,
        pending_actions, agent_results, task_id
    Plus the fields the nodes themselves populate.
    """

    # ---- required by the brief ------------------------------------------
    conversation_id: str
    messages: list[dict[str, Any]]
    #: {campaign_id, creative_id, insight_id, business_id}
    context: dict[str, Any]
    brand_context_ref: str | None
    pending_actions: list[dict[str, Any]]
    agent_results: list[dict[str, Any]]
    task_id: str | None

    # ---- routing ---------------------------------------------------------
    request_id: str
    business_id: str
    workspace_id: str
    user_message: str
    #: research | strategy | creative | analyze | optimize | ask_clarification
    intent: str
    intent_confidence: float
    #: Info the classifier decided is required but absent.
    missing_info: list[str]
    #: The task the classifier picked for the chosen persona.
    task_name: str
    #: The persona the classifier picked.
    persona_id: str
    clarification: dict[str, Any] | None

    # ---- results ---------------------------------------------------------
    blocks: list[dict[str, Any]]
    agent_result: dict[str, Any] | None
    error: str | None


def initial_state(
    *,
    conversation_id: str,
    business_id: str,
    workspace_id: str,
    user_message: str,
    request_id: str,
    context: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> OrchestratorState:
    """A fresh state for one turn."""
    return OrchestratorState(
        conversation_id=conversation_id,
        business_id=business_id,
        workspace_id=workspace_id,
        user_message=user_message,
        request_id=request_id,
        context=context or {},
        messages=messages or [],
        brand_context_ref=None,
        pending_actions=[],
        agent_results=[],
        task_id=None,
        blocks=[],
        agent_result=None,
        error=None,
        clarification=None,
    )


__all__ = ["OrchestratorState", "initial_state"]
