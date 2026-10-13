from __future__ import annotations

from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.ai.agents.salma.schema import OptimizationPlan
from app.ai.agents.salma.state import SalmaState
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.ai.tools import get_brand_context_tool
from app.ai.tools.core_bridge import get_core_tools_for_agent, persist_ai_tools


@tool(args_schema=OptimizationPlan)
def submit_optimization_plan(**kwargs) -> str:
    """Submit Salma's evidence-backed proposal and approval status."""
    return "REPORT_SUBMITTED_SUCCESSFULLY"


SALMA_TOOLS = [
    *persist_ai_tools([get_brand_context_tool]),
    *get_core_tools_for_agent("salma"),
    submit_optimization_plan,
]

SALMA_PROMPT = """You are Salma, Sahm's campaign optimizer. Respond in the user's language.

Read actual campaign insights and, where useful, breakdowns before proposing a
change. Propose at most one specific, reversible change, and cite the exact
metric/source/date range that supports it. If evidence, campaign mapping, or a
required setting is missing, explain the gap and ask for it; do not create a
recommendation record without evidence.

A created recommendation is only a pending internal draft and always requires
human approval. Never execute it, write to Meta, publish, or create an ad. When
there is evidence, create one recommendation and include its returned ID/status
in the final plan. If the fix requires new creative, hand the brief to Layla
through the user/action-card flow; do not generate an image or call VLM yourself.
Finish by calling submit_optimization_plan.

OUTPUT CONTRACT: Deliver optimization analysis and budget tuning for running campaigns only. Do not draft initial brand strategies or create ad copy.
"""

def call_model(state: SalmaState, config: RunnableConfig) -> dict:
    messages = list(state["messages"])
    if not any(
        isinstance(message, SystemMessage) and message.content == SALMA_PROMPT
        for message in messages
    ):
        messages = [SystemMessage(content=SALMA_PROMPT)] + messages
    llm_handle = get_llm(provider=get_ai_settings().llm_provider)
    response = llm_handle.chat.bind_tools(SALMA_TOOLS).invoke(messages, config)
    return {"messages": [response]}


def should_continue(state: SalmaState) -> Literal["tools", END]:
    return "tools" if state["messages"][-1].tool_calls else END


def post_tools_routing(state: SalmaState) -> Literal["agent", END]:
    last_message = state["messages"][-1]
    if last_message.type == "tool" and last_message.name == "submit_optimization_plan":
        return END
    return "agent"


workflow = StateGraph(SalmaState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(SALMA_TOOLS))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_conditional_edges("tools", post_tools_routing, ["agent", END])
salma_graph = workflow.compile()
