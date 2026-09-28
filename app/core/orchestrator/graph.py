"""Layer 6: the LangGraph orchestrator.

    classify_intent -> [research | strategy | creative | analyze | optimize |
                        ask_clarification] -> compose_blocks -> END

Agents never call each other. A cross-agent flow (Karim finds fatigue -> Layla
makes variations) is a separate orchestrator step triggered by an ``action_card``
the user clicks, which arrives as a new turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.core.agents.persona import load_persona
from app.core.agents.runtime import RunOptions, run_agent
from app.core.llm.base import LLMClient
from app.core.logging import get_logger
from app.core.orchestrator.blocks import (
    compose_blocks,
    error_block,
    question_block,
)
from app.core.orchestrator.router import clarification_question, classify_intent
from app.core.orchestrator.state import OrchestratorState, initial_state
from app.core.tools.base import ToolContext
from app.schemas.common import IntentKind

_logger = get_logger(__name__)

#: persona -> the intent its node is keyed by.
_PERSONA_INTENT: dict[str, IntentKind] = {
    "researcher": IntentKind.RESEARCH,
    "omar": IntentKind.STRATEGY,
    "layla": IntentKind.CREATIVE,
    "karim": IntentKind.ANALYZE,
    "salma": IntentKind.OPTIMIZE,
}


@dataclass
class OrchestratorDeps:
    """Everything the graph's nodes need, injected once at build time."""

    db: Any
    llm: LLMClient
    business_id: str
    workspace_id: str = "default"
    request_id: str = ""
    options: RunOptions = field(default_factory=RunOptions)
    #: Set when the caller already resolved a persona (e.g. an action card).
    forced_persona_id: str | None = None
    forced_task: str | None = None


# --------------------------------------------------------------------- nodes --


async def _classify_node(state: OrchestratorState, deps: OrchestratorDeps) -> dict[str, Any]:
    message = state.get("user_message", "")
    context = state.get("context") or {}

    if deps.forced_persona_id and deps.forced_task:
        _logger.info("intent_forced", extra={"persona": deps.forced_persona_id})
        # Route with the persona's own natural intent so the graph still lands
        # on the right agent node rather than short-circuiting to clarification.
        forced_intent = _PERSONA_INTENT.get(deps.forced_persona_id, IntentKind.RESEARCH)
        return {
            "intent": forced_intent.value,
            "persona_id": deps.forced_persona_id,
            "task_name": deps.forced_task,
            "intent_confidence": 1.0,
            "missing_info": [],
        }

    classification = await classify_intent(
        message,
        context=context,
        llm=deps.llm,
        confidence_threshold=0.7,
    )
    _logger.info(
        "intent_classified",
        extra={
            "intent": classification.intent.value,
            "confidence": classification.confidence,
            "task": classification.task_name,
        },
    )
    return {
        "intent": classification.intent.value,
        "intent_confidence": classification.confidence,
        "missing_info": list(classification.missing_info),
        "task_name": classification.task_name or "",
        "persona_id": classification.persona_id or "",
    }


def _route(state: OrchestratorState) -> str:
    intent = state.get("intent") or IntentKind.RESEARCH.value
    missing = state.get("missing_info") or []
    # If required info is missing and cannot be found in brand memory, ask
    # rather than guess.
    if intent != IntentKind.ASK_CLARIFICATION.value and missing:
        return "ask_clarification"
    return intent


async def _agent_node(state: OrchestratorState, deps: OrchestratorDeps, persona_id: str) -> dict[str, Any]:
    task_name = state.get("task_name") or ""
    if not task_name:
        return {"error": "no task was selected", "agent_result": None}

    try:
        persona = await load_persona(persona_id, deps.db, deps.workspace_id)
    except Exception as exc:
        _logger.warning("persona_load_failed", extra={"error": str(exc)})
        return {"error": f"could not load persona '{persona_id}': {exc}", "agent_result": None}

    if not persona.can_run_task(task_name):
        return {
            "error": (
                f"{persona.name} cannot run '{task_name}'. "
                f"Allowed: {', '.join(persona.allowed_tasks)}."
            ),
            "agent_result": None,
        }

    ctx = ToolContext(
        workspace_id=deps.workspace_id,
        business_id=deps.business_id,
        db=deps.db,
        llm=deps.llm,
        request_id=deps.request_id,
        campaign_id=(state.get("context") or {}).get("campaign_id"),
        creative_id=(state.get("context") or {}).get("creative_id"),
        insight_id=(state.get("context") or {}).get("insight_id"),
        persona_id=persona.id,
        conversation_id=state.get("conversation_id"),
    )

    try:
        result = await run_agent(
            persona,
            task_name,
            dict(state.get("context") or {}),
            ctx,
            llm=deps.llm,
            options=deps.options,
            user_message=state.get("user_message", ""),
        )
    except Exception as exc:
        _logger.warning("agent_run_failed", extra={"error": str(exc)})
        return {"error": f"{persona.name} could not complete the task: {exc}", "agent_result": None}

    return {
        "agent_result": result.model_dump(mode="json"),
        "persona_id": persona.id,
        "task_name": task_name,
        "error": result.error,
    }


async def _ask_clarification_node(
    state: OrchestratorState, deps: OrchestratorDeps
) -> dict[str, Any]:
    missing = list(state.get("missing_info") or [])
    payload = clarification_question(missing)
    return {
        "clarification": payload,
        "blocks": [question_block(payload["question"], payload["options"], payload["missing_info"])],
    }


async def _compose_node(state: OrchestratorState, deps: OrchestratorDeps) -> dict[str, Any]:
    """Convert the agent result into validated Message.blocks."""
    agent_result = state.get("agent_result")
    if not agent_result:
        error = state.get("error") or "I could not complete that request."
        return {"blocks": [error_block(error, code="agent_failed")]}

    # Brand context for the context_used block.
    brand_context = None
    try:
        from app.core.memory.store import BrandMemoryStore
        from app.schemas.tools import MemoryScope

        store = BrandMemoryStore(deps.db)
        brand_context = await store.build_context(
            business_id=deps.business_id,
            scope=MemoryScope(query=state.get("user_message", ""), top_k=6),
            task_name=state.get("task_name"),
        )
    except Exception:  # pragma: no cover
        brand_context = None

    blocks = compose_blocks(
        agent_result,
        persona_id=str(agent_result.get("persona_id") or state.get("persona_id") or ""),
        task_name=str(state.get("task_name") or agent_result.get("task") or ""),
        task_id=state.get("task_id"),
        brand_context=brand_context,
    )

    return {"blocks": blocks}


# ---------------------------------------------------------------------- graph --


def build_graph(deps: OrchestratorDeps) -> Any:
    """Compile the orchestrator graph with ``deps`` closed over."""
    graph = StateGraph(OrchestratorState)

    async def classify(state: OrchestratorState) -> dict[str, Any]:
        return await _classify_node(state, deps)

    async def research(state: OrchestratorState) -> dict[str, Any]:
        return await _agent_node(state, deps, "researcher")

    async def strategy(state: OrchestratorState) -> dict[str, Any]:
        return await _agent_node(state, deps, "omar")

    async def creative(state: OrchestratorState) -> dict[str, Any]:
        return await _agent_node(state, deps, "layla")

    async def analyze(state: OrchestratorState) -> dict[str, Any]:
        return await _agent_node(state, deps, "karim")

    async def optimize(state: OrchestratorState) -> dict[str, Any]:
        return await _agent_node(state, deps, "salma")

    async def ask_clarification(state: OrchestratorState) -> dict[str, Any]:
        return await _ask_clarification_node(state, deps)

    async def compose_blocks_node(state: OrchestratorState) -> dict[str, Any]:
        return await _compose_node(state, deps)

    graph.add_node("classify_intent", classify)
    graph.add_node("research", research)
    graph.add_node("strategy", strategy)
    graph.add_node("creative", creative)
    graph.add_node("analyze", analyze)
    graph.add_node("optimize", optimize)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("compose_blocks", compose_blocks_node)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route,
        {
            "research": "research",
            "strategy": "strategy",
            "creative": "creative",
            "analyze": "analyze",
            "optimize": "optimize",
            "ask_clarification": "ask_clarification",
        },
    )
    for node in ("research", "strategy", "creative", "analyze", "optimize"):
        graph.add_edge(node, "compose_blocks")
    graph.add_edge("ask_clarification", END)
    graph.add_edge("compose_blocks", END)

    return graph.compile()


async def run_turn(
    deps: OrchestratorDeps,
    *,
    user_message: str,
    conversation_id: str,
    context: dict[str, Any] | None = None,
) -> OrchestratorState:
    """Run one conversational turn through the graph and return the final state."""
    app = build_graph(deps)
    state = initial_state(
        conversation_id=conversation_id,
        business_id=deps.business_id,
        workspace_id=deps.workspace_id,
        user_message=user_message,
        request_id=deps.request_id,
        context=context,
    )
    final = await app.ainvoke(state)
    return final  # type: ignore[no-any-return]


__all__ = [
    "OrchestratorDeps",
    "build_graph",
    "run_turn",
]
