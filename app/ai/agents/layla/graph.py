from typing import Literal
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.ai.agents.layla.state import LaylaState
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.ai.agents.layla.schema import CreativeBrief
from app.ai.tools import get_brand_context_tool

@tool(args_schema=CreativeBrief)
def submit_creative_brief(**kwargs) -> str:
    """Call this tool ONLY when you have finished your work to submit the final output."""
    return "REPORT_SUBMITTED_SUCCESSFULLY"

# Define the tools available to Layla
LAYLA_TOOLS = [get_brand_context_tool, submit_creative_brief]

LAYLA_PROMPT = """You are Layla, the Creative.
Your goal is to generate ad copy and visual briefs.
You must call submit_creative_brief to submit your final results!

OUTPUT CONTRACT: Deliver creative ad copy, visual briefs, and scripts explicitly requested now. Do not provide market analysis, budget allocations, or campaign launch CTAs."""

def call_model(state: LaylaState, config: RunnableConfig) -> dict:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=LAYLA_PROMPT)] + list(messages)
        
    ai_settings = get_ai_settings()
    llm_handle = get_llm(provider=ai_settings.llm_provider)
    
    model_with_tools = llm_handle.chat.bind_tools(LAYLA_TOOLS)
    response = model_with_tools.invoke(messages, config)
    return {"messages": [response]}

def should_continue(state: LaylaState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def post_tools_routing(state: LaylaState) -> Literal["agent", END]:
    last_message = state["messages"][-1]
    if last_message.type == "tool" and last_message.name == "submit_creative_brief":
        return END
    return "agent"

workflow = StateGraph(LaylaState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(LAYLA_TOOLS))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_conditional_edges("tools", post_tools_routing, ["agent", END])

layla_graph = workflow.compile()
