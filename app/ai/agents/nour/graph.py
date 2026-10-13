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
    fetch_competitor_ads,
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
            fetch_competitor_ads,
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

CRITICAL INSTRUCTIONS FOR RESEARCH & ANALYSIS:
1. TARGETED TOOL EXECUTION:
   - If the user specifies `@web_analyzer` or asks to analyze a website/URL: use `analyze_website_tool(url=...)`. If only a competitor name is given, search for their website or locate their URL, then run `analyze_website_tool`.
   - If the user specifies `@reviews` or asks for customer reviews/sentiment/ratings: use `get_reviews_tool(query=...)` or search specifically for customer feedback, Google reviews, and consumer sentiment.
   - If the user specifies competitor ads, Facebook ads, ad copy, or creative intelligence: use `fetch_competitor_ads(competitor_name=..., competitor_url=...)`.
   - If the user specifies `@gap_analysis` or asks for market gaps: identify unserved needs, weaknesses in competitor offerings, pricing discrepancies, and positioning gaps.
   - If the user mentions `@CompetitorName` (e.g. `@Tie House`, `@AbuAuf`): focus your analysis specifically on that competitor.
2. WORKFLOW AND EVIDENCE:
   - Read get_brand_context_tool first. If there is no profile, state that clearly.
   - Use the user's website URL with analyze_website_tool when supplied. It audits one page only.
   - Use focused queries with web_search_tool (no more than two live searches per entity).
   - Use get_social_profile, get_keywords, or get_competitor_ads only when relevant. If a tool returns unavailable/partial/mock, label that status and never present placeholders as evidence.
   - Keep observed facts, interpretations, gaps, and recommendations separate. Every factual claim needs a source URL or explicit user-supplied evidence.
   - Do not invent prices, budgets, market share, audience sizes, review counts, or performance. Use missing_inputs where data is absent.
   - route_file_to_tool accepts explicitly supplied content only; never ask it to read a server-local path.
   - Call save_learning_tool only when the user explicitly asks you to remember a durable, non-sensitive finding. Never store credentials.
3. LANGUAGE & FINAL OUTPUT:
   - If the user prompt or context is in Arabic, write all report fields (market_overview, competitor_summary positioning, strengths, weaknesses, opportunities, risks) in clear Arabic.
   - Finish by calling submit_research_report. Do not append an unstructured answer after submission.

OUTPUT CONTRACT: Deliver the research report only. Do not add strategy advice, campaign CTAs, or next-step offers. End after the sources section.
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
