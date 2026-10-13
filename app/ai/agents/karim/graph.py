from __future__ import annotations

from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.ai.agents.karim.schema import PerformanceReport
from app.ai.agents.karim.state import KarimState
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.ai.tools import get_brand_context_tool, save_learning_tool
from app.ai.tools.core_bridge import get_core_tools_for_agent, persist_ai_tools


@tool(args_schema=PerformanceReport)
def submit_performance_report(**kwargs) -> str:
    """Submit Karim's report using actual campaign metrics and evidence."""
    return "REPORT_SUBMITTED_SUCCESSFULLY"


KARIM_TOOLS = [
    *persist_ai_tools([get_brand_context_tool, save_learning_tool]),
    *get_core_tools_for_agent("karim"),
    submit_performance_report,
]

KARIM_PROMPT = """You are Karim, Sahm's campaign-performance analyst. Respond in the user's language.

Use get_campaign_insights and, when useful, get_breakdowns to retrieve actual
Meta Insights for the campaign and date range. If the campaign mapping, Meta
connection, or requested date range is unavailable, say so and ask for the
missing input. Do not substitute estimates.

Report only numbers returned by a tool or explicitly supplied by the user.
Attribute each metric to its source and date range. Do not infer campaign
performance from an image, prompt, or creative appearance. Do not fabricate a
benchmark, trend, cause, prediction, or result. If data is partial, label it.
Call save_learning_tool only when the user explicitly asks you to remember a
durable, non-sensitive finding. Finish by calling submit_performance_report.
"""


def call_model(state: KarimState, config: RunnableConfig) -> dict:
    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=KARIM_PROMPT)] + messages
    llm_handle = get_llm(provider=get_ai_settings().llm_provider)
    response = llm_handle.chat.bind_tools(KARIM_TOOLS).invoke(messages, config)
    return {"messages": [response]}


def should_continue(state: KarimState) -> Literal["tools", END]:
    return "tools" if state["messages"][-1].tool_calls else END


def post_tools_routing(state: KarimState) -> Literal["agent", END]:
    last_message = state["messages"][-1]
    if last_message.type == "tool" and last_message.name == "submit_performance_report":
        return END
    return "agent"


workflow = StateGraph(KarimState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(KARIM_TOOLS))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_conditional_edges("tools", post_tools_routing, ["agent", END])
karim_graph = workflow.compile()
