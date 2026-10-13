from __future__ import annotations

from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.ai.agents.nour.schema import MarketResearchReport
from app.ai.agents.nour.state import NourState
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.ai.tools import (
    analyze_website_tool,
    brand_report,
    full_analysis,
    get_brand_context_tool,
    get_reviews_tool,
    route_file_to_tool,
    save_learning_tool,
    web_search_tool,
)
from app.ai.tools.core_bridge import get_core_tools_for_agent, persist_ai_tools


@tool(args_schema=MarketResearchReport)
def submit_research_report(**kwargs) -> str:
    """Submit Nour's final source-backed research report."""
    return "REPORT_SUBMITTED_SUCCESSFULLY"


NOUR_TOOLS = [
    *persist_ai_tools(
        [
            get_brand_context_tool,
            web_search_tool,
            analyze_website_tool,
            get_reviews_tool,
            brand_report,
            full_analysis,
            route_file_to_tool,
            save_learning_tool,
        ]
    ),
    *get_core_tools_for_agent("nour"),
    submit_research_report,
]

NOUR_PROMPT = """You are Nour, an evidence-first market and competitor researcher. Respond in the user's language.

WORKFLOW AND EVIDENCE:
1. Read get_brand_context_tool first. If there is no profile, state that clearly.
2. Use the user's website URL with analyze_website_tool when supplied. It audits one page only.
3. Use no more than TWO live web-search queries total. brand_report performs one query internally; don't combine it with two more.
4. Use get_social_profile, get_keywords, or get_competitor_ads only when relevant and when a real source is configured. If a tool returns unavailable/partial/mock, label that status and never present placeholders as evidence.
5. Get reviews only if relevant. If the paid-provider guard blocks it, report that no review provider was contacted.
6. Keep observed facts, interpretations, gaps, and recommendations separate. Every factual claim needs a source URL or explicit user-supplied evidence.
7. Do not invent prices, budgets, market share, audience sizes, review counts, or performance. Use missing_inputs where data is absent.
8. route_file_to_tool accepts explicitly supplied content only; never ask it to read a server-local path.
9. Call save_learning_tool only when the user explicitly asks you to remember a durable, non-sensitive finding. Never store credentials.
10. Finish by calling submit_research_report. Do not append an unstructured answer after submission.
"""


def call_model(state: NourState, config: RunnableConfig) -> dict:
    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=NOUR_PROMPT)] + messages
    llm_handle = get_llm(provider=get_ai_settings().llm_provider)
    response = llm_handle.chat.bind_tools(NOUR_TOOLS).invoke(messages, config)
    return {"messages": [response]}


def should_continue(state: NourState) -> Literal["tools", END]:
    return "tools" if state["messages"][-1].tool_calls else END


def post_tools_routing(state: NourState) -> Literal["agent", END]:
    last_message = state["messages"][-1]
    if last_message.type == "tool" and last_message.name == "submit_research_report":
        return END
    return "agent"


workflow = StateGraph(NourState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(NOUR_TOOLS))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_conditional_edges("tools", post_tools_routing, ["agent", END])
nour_graph = workflow.compile()
