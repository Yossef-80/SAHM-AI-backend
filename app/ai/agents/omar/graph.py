from typing import Annotated, Any, Literal

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command

from app.ai.agents.omar.schema import StrategyReport
from app.ai.agents.omar.state import OmarState
from app.ai.agents.omar.tools import (
    _state_update,
    generate_campaign_strategy,
    generate_content_calendar,
    generate_marketing_strategy,
)
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.ai.tools import (
    get_brand_context_tool,
    save_learning_tool,
    update_brand_identity_tool,
    web_search_tool,
)
from app.ai.tools.core_bridge import persist_ai_tools
from app.core.i18n import SUBAGENT_LANGUAGE_RULE


@tool(args_schema=StrategyReport)
def submit_strategy_report(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict[str, Any], InjectedState],
    **kwargs: Any,
) -> Command:
    """Submit Omar's final report and attach any structured plans from this run."""
    report = StrategyReport.model_validate(kwargs).model_dump(mode="json")
    for field in ("marketing_strategy", "campaign_strategy", "content_calendar"):
        value = state.get(field)
        if value:
            report[field] = value
    return _state_update(
        tool_call_id,
        "REPORT_SUBMITTED_SUCCESSFULLY",
        tool_name="submit_strategy_report",
        final_report=report,
    )

# The ZIP's duplicate brand/search tools are already present in the app. Only
# its useful structured planning capabilities are added here; save_learning is
# intentionally omitted because it would persist a side effect without approval.
OMAR_TOOLS = [
    *persist_ai_tools(
        [
            get_brand_context_tool,
            web_search_tool,
            generate_marketing_strategy,
            generate_campaign_strategy,
            generate_content_calendar,
            save_learning_tool,
            update_brand_identity_tool,
        ]
    ),
    submit_strategy_report,
]

OMAR_PROMPT = f"""You are Omar, Sahm's marketing strategist. {SUBAGENT_LANGUAGE_RULE}
Turn brand context and research into a concrete plan. Do not invent facts,
benchmarks, audience sizes, budgets, or KPI targets; put unavailable information
in missing_inputs and identify assumptions explicitly.

BRAND IDENTITY & POSITIONING UPDATES:
When the user asks to change, update, or set brand voice, tagline, positioning, target audience, or brand colors, call `update_brand_identity_tool` immediately to persist the change directly to the database.

Treat saved profile values as user-provided context, not independent verification. When a recommendation uses
an offer or claim from Settings, name that field in evidence and do not embellish it.
Never add guarantees, original-parts claims, superiority or price comparisons,
claims about the latest equipment, or promises of results unless the user or a source explicitly
supports the exact claim. Do not call a channel "most effective" without evidence.
Any channel split is a proposal, not an approved allocation; do not present one as
settled fact, and leave shares unset when no sound basis is available. Never call
a plan ready to launch while required inputs remain missing; present it as a draft
and list what must be confirmed first.

For a campaign plan that will be handed to Layla:
1. Call get_brand_context_tool first. Search only when external evidence is useful.
2. Call generate_marketing_strategy before generate_campaign_strategy.
3. Call generate_campaign_strategy with objective, audience, messages, channels,
   and creative direction only when supported by the request/context. Leave
   unknown settings unset and list them as missing_inputs.
4. Call generate_content_calendar when the user asks for a calendar or to schedule/distribute posts across days or weeks (e.g. 'خليهم في الخمس ايام الجايين', 'عايز اشوفهم في الcalendar', 'schedule posts for the next 5 days'). This is mandatory to save the posts into the system calendar.
5. Call save_learning_tool only when the user explicitly asks you to remember a durable, non-sensitive finding.
6. Finish by calling submit_strategy_report. Do not create or publish a Meta
   campaign, ad set, creative, or ad; those belong to the separate approval flow.

OUTPUT CONTRACT: Focus exclusively on marketing strategy, positioning, and content calendar roadmaps explicitly requested. Do not invent unconfirmed campaign execution CTAs or execute live campaign operations.
"""


def call_model(state: OmarState, config: RunnableConfig) -> dict[str, Any]:
    messages = list(state["messages"])
    if not any(
        isinstance(message, SystemMessage) and message.content == OMAR_PROMPT
        for message in messages
    ):
        messages = [SystemMessage(content=OMAR_PROMPT)] + messages

    ai_settings = get_ai_settings()
    llm_handle = get_llm(provider=ai_settings.llm_provider)
    response = llm_handle.chat.bind_tools(OMAR_TOOLS).invoke(messages, config)
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
