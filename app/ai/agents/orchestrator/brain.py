from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings

from app.ai.agents.nour.graph import nour_graph
from app.ai.agents.omar.graph import omar_graph
from app.ai.agents.layla.graph import layla_graph
from app.ai.agents.karim.graph import karim_graph
from app.ai.agents.salma.graph import salma_graph

class RoutingDecision(BaseModel):
    agent: Literal["nour", "omar", "layla", "karim", "salma", "unknown"] = Field(description="The agent best suited to handle this prompt.")
    reason: str = Field(description="Why this agent was chosen.")

from app.schemas.blocks import MessageWithBlocks

def orchestrate(prompt: str) -> MessageWithBlocks:
    """The Brain that reads the prompt and routes it to the correct agent."""
    ai_settings = get_ai_settings()
    llm = get_llm(provider=ai_settings.llm_provider).chat.with_structured_output(RoutingDecision)
    
    sys_prompt = """You are the Orchestrator Hub. You route user prompts to the correct marketing agent.
Agents:
- nour: Researcher (uncover market truth, competitors, reviews)
- omar: Strategist (positioning, budgets, content calendars)
- layla: Creative (generate ad copy, visual briefs, scripts)
- karim: Analyst (monitor ads delivery, performance, metrics)
- salma: Optimizer (optimizations, budgets)
"""
    
    decision = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=prompt)])
    
    print(f"[ORCHESTRATOR] Routing to: {decision.agent.upper()} (Reason: {decision.reason})")
    
    graphs = {
        "nour": nour_graph,
        "omar": omar_graph,
        "layla": layla_graph,
        "karim": karim_graph,
        "salma": salma_graph
    }
    
    if decision.agent in graphs:
        config = {"configurable": {"thread_id": "test_thread"}}
        final_state = graphs[decision.agent].invoke({"messages": [HumanMessage(content=prompt)]}, config)
        
        # Format into UI blocks
        final_msg = final_state["messages"][-1].content
        block_llm = get_llm(provider=ai_settings.llm_provider).chat.with_structured_output(MessageWithBlocks)
        return block_llm.invoke([
            SystemMessage(content="Convert the following text into structured React UI Blocks (TextBlock for paragraphs, FindingsBlock for lists/bullet points)."),
            HumanMessage(content=final_msg)
        ])
    
    return None
