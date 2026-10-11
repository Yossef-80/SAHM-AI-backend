from typing import Literal
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.ai.agents.salma.state import SalmaState
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.ai.agents.salma.schema import OptimizationPlan
from app.ai.tools import get_brand_context_tool

@tool(args_schema=OptimizationPlan)
def submit_optimization_plan(**kwargs) -> str:
    """Call this tool ONLY when you have finished your work to submit the final output."""
    return "REPORT_SUBMITTED_SUCCESSFULLY"

# Define the tools available to Salma
SALMA_TOOLS = [get_brand_context_tool, submit_optimization_plan]

SALMA_PROMPT = """You are Salma, the Optimizer.
Your goal is to optimize strategies and budgets.
You must call submit_optimization_plan to submit your final results!

OUTPUT CONTRACT: Deliver optimization analysis and budget tuning for running campaigns only. Do not draft initial brand strategies or create ad copy."""

def call_model(state: SalmaState, config: RunnableConfig) -> dict:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SALMA_PROMPT)] + list(messages)
        
    ai_settings = get_ai_settings()
    llm_handle = get_llm(provider=ai_settings.llm_provider)
    
    model_with_tools = llm_handle.chat.bind_tools(SALMA_TOOLS)
    response = model_with_tools.invoke(messages, config)
    return {"messages": [response]}

def should_continue(state: SalmaState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

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
