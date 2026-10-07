import os

BASE_DIR = r"D:\NTI-Final-Project\SAHM-AI-backend\app\ai\agents"

# --- AGENT TEMPLATES ---
STATE_TEMPLATE = """from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class {agent_name_capitalized}State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
"""

SCHEMA_TEMPLATE = """from pydantic import BaseModel, Field

class {submit_tool_name_camel}(BaseModel):
    \"\"\"Call this tool to submit your final {agent_name} output.\"\"\"
    summary: str = Field(description="Summary of your findings/strategy")
    details: list[str] = Field(description="Detailed bullet points")
"""

GRAPH_TEMPLATE = """from typing import Literal
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.ai.agents.{agent_name}.state import {agent_name_capitalized}State
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.ai.agents.{agent_name}.schema import {submit_tool_name_camel}
{imports}

@tool(args_schema={submit_tool_name_camel})
def {submit_tool_name}(**kwargs) -> str:
    \"\"\"Call this tool ONLY when you have finished your work to submit the final output.\"\"\"
    return "REPORT_SUBMITTED_SUCCESSFULLY"

# Define the tools available to {agent_name_capitalized}
{agent_name_upper}_TOOLS = [{tools_list}, {submit_tool_name}]

{agent_name_upper}_PROMPT = \"\"\"You are {agent_name_capitalized}, the {role}.
Your goal is to {goal}.
You must call {submit_tool_name} to submit your final results!\"\"\"

def call_model(state: {agent_name_capitalized}State, config: RunnableConfig) -> dict:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content={agent_name_upper}_PROMPT)] + list(messages)
        
    ai_settings = get_ai_settings()
    llm_handle = get_llm(provider=ai_settings.llm_provider)
    
    model_with_tools = llm_handle.chat.bind_tools({agent_name_upper}_TOOLS)
    response = model_with_tools.invoke(messages, config)
    return {{"messages": [response]}}

def should_continue(state: {agent_name_capitalized}State) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def post_tools_routing(state: {agent_name_capitalized}State) -> Literal["agent", END]:
    last_message = state["messages"][-1]
    if last_message.type == "tool" and last_message.name == "{submit_tool_name}":
        return END
    return "agent"

workflow = StateGraph({agent_name_capitalized}State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode({agent_name_upper}_TOOLS))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_conditional_edges("tools", post_tools_routing, ["agent", END])

{agent_name}_graph = workflow.compile()
"""

# --- AGENTS CONFIG ---
AGENTS = {
    "omar": {
        "role": "Strategist",
        "goal": "convert research into positioning and channel strategies",
        "submit_tool_name": "submit_strategy_report",
        "submit_tool_name_camel": "StrategyReport",
        "tools_list": "get_brand_context_tool, web_search_tool",
        "imports": "from app.ai.tools import get_brand_context_tool, web_search_tool"
    },
    "layla": {
        "role": "Creative",
        "goal": "generate ad copy and visual briefs",
        "submit_tool_name": "submit_creative_brief",
        "submit_tool_name_camel": "CreativeBrief",
        "tools_list": "get_brand_context_tool",
        "imports": "from app.ai.tools import get_brand_context_tool"
    },
    "karim": {
        "role": "Analyst",
        "goal": "analyze campaign metrics and health",
        "submit_tool_name": "submit_performance_report",
        "submit_tool_name_camel": "PerformanceReport",
        "tools_list": "get_brand_context_tool",
        "imports": "from app.ai.tools import get_brand_context_tool"
    },
    "salma": {
        "role": "Optimizer",
        "goal": "optimize strategies and budgets",
        "submit_tool_name": "submit_optimization_plan",
        "submit_tool_name_camel": "OptimizationPlan",
        "tools_list": "get_brand_context_tool",
        "imports": "from app.ai.tools import get_brand_context_tool"
    }
}

for agent_name, config in AGENTS.items():
    agent_dir = os.path.join(BASE_DIR, agent_name)
    os.makedirs(agent_dir, exist_ok=True)
    
    with open(os.path.join(agent_dir, "__init__.py"), "w") as f:
        f.write(f"from .{agent_name} import {agent_name}_graph\n") # will fix graph import later
        
    with open(os.path.join(agent_dir, "state.py"), "w") as f:
        f.write(STATE_TEMPLATE.format(agent_name_capitalized=agent_name.capitalize()))
        
    with open(os.path.join(agent_dir, "schema.py"), "w") as f:
        f.write(SCHEMA_TEMPLATE.format(
            agent_name=agent_name,
            submit_tool_name_camel=config["submit_tool_name_camel"]
        ))
        
    with open(os.path.join(agent_dir, "graph.py"), "w", encoding="utf-8") as f:
        f.write(GRAPH_TEMPLATE.format(
            agent_name=agent_name,
            agent_name_capitalized=agent_name.capitalize(),
            agent_name_upper=agent_name.upper(),
            role=config["role"],
            goal=config["goal"],
            submit_tool_name=config["submit_tool_name"],
            submit_tool_name_camel=config["submit_tool_name_camel"],
            tools_list=config["tools_list"],
            imports=config["imports"]
        ))
    
    with open(os.path.join(agent_dir, "__init__.py"), "w") as f:
        f.write(f"from .graph import {agent_name}_graph\n")

# --- BUILD THE BRAIN ORCHESTRATOR ---
ORCHESTRATOR_DIR = os.path.join(BASE_DIR, "orchestrator")
os.makedirs(ORCHESTRATOR_DIR, exist_ok=True)

ORCHESTRATOR_CODE = """from typing import Literal
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

def orchestrate(prompt: str) -> dict:
    \"\"\"The Brain that reads the prompt and routes it to the correct agent.\"\"\"
    ai_settings = get_ai_settings()
    llm = get_llm(provider=ai_settings.llm_provider).chat.with_structured_output(RoutingDecision)
    
    sys_prompt = \"\"\"You are the Orchestrator Hub. You route user prompts to the correct marketing agent.
Agents:
- nour: Researcher (uncover market truth, competitors, reviews)
- omar: Strategist (positioning, budgets, content calendars)
- layla: Creative (generate ad copy, visual briefs, scripts)
- karim: Analyst (monitor ads delivery, performance, metrics)
- salma: Optimizer (optimizations, budgets)
\"\"\"
    
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
        return {"agent": decision.agent, "state": final_state}
    
    return {"agent": "unknown", "error": "No suitable agent found."}
"""

with open(os.path.join(ORCHESTRATOR_DIR, "brain.py"), "w", encoding="utf-8") as f:
    f.write(ORCHESTRATOR_CODE)

print("Agents and Brain generated successfully!")
