from typing import TypedDict, Literal, Any, cast
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.ai.agents.nour.state import NourState
from app.ai.providers.llm import get_llm
from app.ai.tools import web_search_tool, get_brand_context_tool
from app.ai.settings import get_ai_settings
from app.ai.enums import AgentName

# Define the tools available to Nour
NOUR_TOOLS = [web_search_tool,]

# System prompt for Nour
NOUR_PROMPT = """You are Nour, an expert researcher agent.
Your primary role is to help  analyze, search, and and competetoirs strenthes and current offers 

Always check the brand context first if it's a new conversation or if you need to understand the brand's voice and values.
Use the web search tool if you need to find recent trends, news, or external information.
help me to find the competeoris in calssic suits form egypt
"""

def call_model(state: NourState, config: RunnableConfig) -> dict:
    messages = state["messages"]
    
    # Prepend system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=NOUR_PROMPT)] + list(messages)
        
    # Get the LLM instance based on configuration
    ai_settings = get_ai_settings()
    llm_handle = get_llm(provider=ai_settings.llm_provider)
    
    # Bind tools to the LLM
    model_with_tools = llm_handle.chat.bind_tools(NOUR_TOOLS)
    
    # Invoke the model (the llm_handle includes the callbacks automatically)
    response = model_with_tools.invoke(messages, config)
    
    return {"messages": [response]}


def should_continue(state: NourState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, we transition to the "tools" node
    if last_message.tool_calls:
        return "tools"
    return END

# Build the graph
workflow = StateGraph(NourState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(NOUR_TOOLS))

# Add edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# Compile the graph
nour_graph = workflow.compile()
