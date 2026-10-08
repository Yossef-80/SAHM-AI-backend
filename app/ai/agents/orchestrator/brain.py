from typing import Literal, Any
import asyncio
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings

from app.ai.agents.nour.graph import nour_graph
from app.ai.agents.omar.graph import omar_graph
from app.ai.agents.layla.graph import layla_graph
from app.ai.agents.karim.graph import karim_graph
from app.ai.agents.salma.graph import salma_graph

class RoutingDecision(BaseModel):
    agent: Literal["nour", "omar", "layla", "karim", "salma", "chat", "unknown"] = Field(description="The agent best suited to handle this prompt.")
    reason: str = Field(description="Why this agent was chosen.")

from app.schemas.blocks import MessageWithBlocks

async def orchestrate(prompt: str, history: list[dict[str, Any]], event_queue: asyncio.Queue[dict[str, Any]] | None = None) -> MessageWithBlocks | None:
    """The Brain that reads the prompt and routes it to the correct agent."""
    if event_queue:
        await event_queue.put({"type": "step_started", "step": "orchestrator", "detail": "Analyzing your prompt and deciding who to route it to..."})

    ai_settings = get_ai_settings()
    llm = get_llm(provider=ai_settings.llm_provider).chat.with_structured_output(RoutingDecision)
    
    sys_prompt = """You are the Orchestrator Hub. You route user prompts to the correct marketing agent.
Agents:
- nour: Researcher (uncover market truth, competitors, reviews) - USE FOR NEW RESEARCH TASKS.
- omar: Strategist (positioning, budgets, content calendars) - USE FOR NEW STRATEGY TASKS.
- layla: Creative (generate ad copy, visual briefs, scripts)
- karim: Analyst (monitor ads delivery, performance, metrics)
- salma: Optimizer (optimizations, budgets)
- chat: Conversational/Follow-up (use for fast clarifications, answering questions about previous messages, simple chat, or follow-ups that DO NOT require new heavy tool executions).

CRITICAL RULE: If the user is asking a follow-up question (e.g. "what does point 3 mean?", "clarify this"), route to 'chat'. Only route to other agents for new heavy tasks."""
    
    try:
        from langchain_core.messages import AIMessage
        chat_history = []
        for msg in history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))
                
        routing_sys = SystemMessage(content=sys_prompt)
        routing_msgs = [routing_sys] + chat_history + [HumanMessage(content=f"Based on the history above, analyze the user's NEWEST prompt: '{prompt}'\n\nWhich agent should handle this newest prompt?")]
        decision = await llm.ainvoke(routing_msgs)
    except Exception as e:
        print(f"[ORCHESTRATOR] Routing failed, defaulting to unknown. Error: {e}")
        decision = RoutingDecision(agent="unknown", reason="Failed to parse routing decision.")
    
    print(f"[ORCHESTRATOR] Routing to: {decision.agent.upper()} (Reason: {decision.reason})")
    
    graphs = {
        "nour": nour_graph,
        "omar": omar_graph,
        "layla": layla_graph,
        "karim": karim_graph,
        "salma": salma_graph
    }
    
    if decision.agent in graphs:
        if event_queue:
            await event_queue.put({"type": "step_started", "step": decision.agent, "detail": f"{decision.agent.title()} is working: {decision.reason}"})
        
        config = {"configurable": {"thread_id": "test_thread"}}
        final_state = await graphs[decision.agent].ainvoke({"messages": chat_history + [HumanMessage(content=f"Context is provided above. Please execute this NEW request: {prompt}")]}, config)
        
        if event_queue:
            await event_queue.put({"type": "step_started", "step": "formatting", "detail": "Structuring the final response for the UI..."})
            
        # Extract final text or tool arguments
        messages = final_state["messages"]
        last_msg = messages[-1]
        
        if last_msg.type == "tool":
            # Graph ended by calling a tool like 'submit_research_report'. The real data is in the args.
            ai_msg = messages[-2]
            if getattr(ai_msg, "tool_calls", None):
                import json
                final_msg = json.dumps(ai_msg.tool_calls[0]["args"], indent=2)
            else:
                final_msg = last_msg.content
        else:
            final_msg = last_msg.content
    else:
        # Fast Conversational Agent for follow-ups and clarifications
        if event_queue:
            detail = "Answering follow-up question..." if decision.agent == "chat" else "Answering general question..."
            await event_queue.put({"type": "step_started", "step": "chat", "detail": detail})
        general_llm = get_llm(provider=ai_settings.llm_provider).chat
        sys_msg = SystemMessage(content="You are a fast, conversational AI assistant. You will be provided with the conversation history for context. CRITICAL: Do NOT re-answer previous questions. ONLY answer the user's latest question. Do not use tools.")
        fallback_res = await general_llm.ainvoke([sys_msg] + chat_history + [HumanMessage(content=f"Context from previous turns is provided above. \n\nNow, answer this NEW question: {prompt}")])
        final_msg = fallback_res.content

    block_llm = get_llm(provider=ai_settings.llm_provider).chat.with_structured_output(MessageWithBlocks)
    try:
        return await block_llm.ainvoke([
            SystemMessage(content="""Convert the following text into structured React UI Blocks. 
CRITICAL RULE FOR THE 'type' FIELD: 
- For paragraphs, use exactly 'text' (NOT 'TextBlock').
- For lists/bullet points, use exactly 'findings' (NOT 'FindingsBlock').
Ensure the format and 'type' values are strictly adhered to."""),
            HumanMessage(content=final_msg)
        ])
    except Exception as e:
        print(f"[ORCHESTRATOR] UI Block structuring failed, falling back to plain text. Error: {e}")
        from app.schemas.blocks import TextBlock
        return MessageWithBlocks(blocks=[TextBlock(text=final_msg)])
