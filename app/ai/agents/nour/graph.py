from typing import TypedDict, Literal, Any, cast
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.ai.agents.nour.state import NourState
from app.ai.providers.llm import get_llm
from app.ai.tools import web_search_tool, get_brand_context_tool, analyze_website_tool, brand_report, full_analysis, route_file_to_tool, get_reviews_tool, fetch_competitor_ads
from app.ai.settings import get_ai_settings
from app.ai.enums import AgentName
from app.ai.agents.nour.schema import MarketResearchReport

@tool(args_schema=MarketResearchReport)
def submit_research_report(**kwargs) -> str:
    """Call this tool ONLY when you have finished all your research and are ready to submit the final, highly structured competitive market research report."""
    return "REPORT_SUBMITTED_SUCCESSFULLY"

# Define the tools available to Nour
NOUR_TOOLS = [web_search_tool, get_brand_context_tool, analyze_website_tool, brand_report, full_analysis, route_file_to_tool, get_reviews_tool, fetch_competitor_ads, submit_research_report]

# System prompt for Nour
NOUR_PROMPT = """You are Nour, an expert digital marketing agent and competitive intelligence analyst.
Your primary role is to help analyze competitors, market trends, and brand strategies.

CRITICAL INSTRUCTIONS FOR RESEARCH & ANALYSIS:
1. TARGETED TOOL EXECUTION:
   - If the user specifies `@web_analyzer` or asks to analyze a website/URL: use `analyze_website_tool(url=...)`. If only a competitor name is given, search for their website or locate their URL, then run `analyze_website_tool`.
   - If the user specifies `@reviews` or asks for customer reviews/sentiment/ratings: use `get_reviews_tool(query=...)` or search specifically for customer feedback, Google reviews, and consumer sentiment.
   - If the user specifies competitor ads, Facebook ads, ad copy, or creative intelligence: use `fetch_competitor_ads(competitor_name=..., competitor_url=...)`.
   - If the user specifies `@gap_analysis` or asks for market gaps: identify unserved needs, weaknesses in competitor offerings, pricing discrepancies, and positioning gaps.
   - If the user mentions `@CompetitorName` (e.g. `@Tie House`, `@AbuAuf`): focus your analysis specifically on that competitor.
2. FACT-CHECKING: When analyzing competitors from web searches, carefully verify that the business is actually in the correct industry and market.
3. ATTRIBUTION: Do not mix up facts between competitors.
4. CONTEXT: Always check the brand context first using get_brand_context_tool to understand who you are researching for.
5. SEARCH: Formulate highly specific queries using web_search_tool. You may execute 1 to 2 targeted searches per competitor or entity (up to 4 searches total when comparing multiple competitors).
6. LANGUAGE: If the user prompt or context is in Arabic, write all report fields (market_overview, competitor_summary positioning, strengths, weaknesses, opportunities, risks) in clear Arabic.
7. FINAL OUTPUT: You MUST submit your final findings by calling the `submit_research_report` tool. Do not generate an unstructured conversational response at the end. Just call the tool with the structured data.

OUTPUT CONTRACT: Deliver the research report only. Do not add strategy advice, campaign CTAs, or next-step offers. End after the sources section.
"""

def call_model(state: NourState, config: RunnableConfig) -> dict:
    messages = state["messages"]
    
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=NOUR_PROMPT)] + list(messages)
        
    ai_settings = get_ai_settings()
    llm_handle = get_llm(provider=ai_settings.llm_provider)
    
    model_with_tools = llm_handle.chat.bind_tools(NOUR_TOOLS)
    response = model_with_tools.invoke(messages, config)
    
    return {"messages": [response]}


def should_continue(state: NourState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        # If the LLM called submit_research_report, we can stop the graph here.
        # But we must let the ToolNode execute it so it gets added to the state.
        return "tools"
    return END

def post_tools_routing(state: NourState) -> Literal["agent", END]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message is from a tool and it was the submit_research_report tool, end the graph.
    if last_message.type == "tool" and last_message.name == "submit_research_report":
        return END
    return "agent"

# Build the graph
workflow = StateGraph(NourState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(NOUR_TOOLS))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_conditional_edges("tools", post_tools_routing, ["agent", END])

nour_graph = workflow.compile()

