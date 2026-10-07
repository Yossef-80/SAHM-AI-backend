from typing import Literal
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.ai.agents.omar.state import OmarState
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.ai.agents.omar.schema import StrategyReport
from app.ai.tools import get_brand_context_tool, web_search_tool

@tool(args_schema=StrategyReport)
def submit_strategy_report(**kwargs) -> str:
    """Call this tool ONLY when you have finished your work to submit the final output."""
    return "REPORT_SUBMITTED_SUCCESSFULLY"

# Define the tools available to Omar
OMAR_TOOLS = [get_brand_context_tool, web_search_tool, submit_strategy_report]

OMAR_PROMPT = """You are Omar, the Strategist.
Your goal is to convert research into positioning and channel strategies.
You must call submit_strategy_report to submit your final results!"""

def call_model(state: OmarState, config: RunnableConfig) -> dict:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=OMAR_PROMPT)] + list(messages)
        
    ai_settings = get_ai_settings()
    llm_handle = get_llm(provider=ai_settings.llm_provider)
    
    model_with_tools = llm_handle.chat.bind_tools(OMAR_TOOLS)
    response = model_with_tools.invoke(messages, config)
    return {"messages": [response]}

def should_continue(state: OmarState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def post_tools_routing(state: OmarState) -> Literal["agent", END]:
    last_message = state["messages"][-1]
    if last_message.type == "tool" and last_message.name == "submit_strategy_report":
        return END
    return "agent"

workflow = StateGraph(OmarState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(OMAR_TOOLS))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_conditional_edges("tools", post_tools_routing, ["agent", END])

omar_graph = workflow.compile()
