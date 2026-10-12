from typing import Literal
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.ai.agents.karim.state import KarimState
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.ai.agents.karim.schema import PerformanceReport
from app.ai.tools import get_brand_context_tool, fetch_ga4_performance

@tool(args_schema=PerformanceReport)
def submit_performance_report(**kwargs) -> str:
    """Call this tool ONLY when you have finished your work to submit the final output."""
    return "REPORT_SUBMITTED_SUCCESSFULLY"

# Define the tools available to Karim
KARIM_TOOLS = [get_brand_context_tool, fetch_ga4_performance, submit_performance_report]

KARIM_PROMPT = """You are Karim, the Analyst.
Your goal is to analyze campaign metrics and health.
You have access to fetch_ga4_performance to retrieve website traffic, bounce rates, conversions and user engagement for Google Analytics properties.
You must call submit_performance_report to submit your final results!

OUTPUT CONTRACT: Answer only from stored data. No recommendations, no campaign CTAs."""

def call_model(state: KarimState, config: RunnableConfig) -> dict:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=KARIM_PROMPT)] + list(messages)
        
    ai_settings = get_ai_settings()
    llm_handle = get_llm(provider=ai_settings.llm_provider)
    
    model_with_tools = llm_handle.chat.bind_tools(KARIM_TOOLS)
    response = model_with_tools.invoke(messages, config)
    return {"messages": [response]}

def should_continue(state: KarimState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

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
