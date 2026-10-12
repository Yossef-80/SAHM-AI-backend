from typing import Literal, Any
import asyncio
import re
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from app.core.logging import get_logger
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.schemas.common import Evidence, canonicalize_url
from app.schemas.blocks import MessageWithBlocks, FindingsBlock, Finding, TextBlock
from app.ai.memory.compressor import compress_history_messages

from app.ai.agents.nour.graph import nour_graph
from app.ai.agents.omar.graph import omar_graph
from app.ai.agents.layla.graph import layla_graph
from app.ai.agents.karim.graph import karim_graph
from app.ai.agents.salma.graph import salma_graph

class RoutingDecision(BaseModel):
    agent: Literal["nour", "omar", "layla", "karim", "salma", "chat", "unknown"] = Field(description="The agent best suited to handle this prompt.")
    reason: str = Field(description="Why this agent was chosen.")

logger = get_logger(__name__)

ROUTER_SYSTEM_PROMPT = """You are the Orchestrator Hub. You route user prompts to the correct marketing agent based EXCLUSIVELY on the user's explicit intent in their newest message.

Agents and Scopes:
- nour  → LIVE RESEARCH & COMPETITOR TOOLS: The user explicitly asks to search, find, crawl, or analyze information NOW, or mentions a competitor or tool with @ (e.g. @AbuAuf, @web_analyzer, @reviews, @gap_analysis, @researcher, "حلل موقع", "افحص المنافس", "ريفيوهات", "تقييمات", "فجوات السوق", "search", "find competitors").
- karim → MEMORY RECALL only: Questions about what was ALREADY researched or stored in memory without asking to run new live tools ("مين المنافسين المسجلين عندنا", "إيه المتخزن في الذاكرة", "what do we know about competitors from previous research").
- omar  → STRATEGY & CONTENT CALENDAR: When THIS message explicitly asks for a high-level marketing strategy, strategic roadmap, market positioning, ad campaign creation / plan, OR a CONTENT CALENDAR / POST SCHEDULING across days or dates for the website calendar ("عايز اعمل حملة اعلانية", "حطلي خطة تسويق", "positioning strategy", "جدول محتوى", "خليهم في الخمس ايام الجايين", "في الكالندر", "عايز اشوفهم في الcalendar", "content calendar", "schedule posts").
- layla → CONTENT creation only: Creative assets explicitly requested NOW (writing ad copy, headlines, visual briefs, 15s commercial scripts, reels scripts).
- salma → LIVE CAMPAIGN OPS only: Real-time budget reallocation, ROAS adjustments, or performance optimization on currently RUNNING/active ad campaigns.
- chat  → CONVERSATIONAL / GENERAL: Questions about the user's stored brand profile ("ايه الي تعرفه عني", "what is my brand scope"), casual pleasantries, follow-ups, clarifications, or requests not asking for new agent work.

HARD ROUTING RULES:
1. Route strictly by the user's EXPLICIT intent in THIS NEWEST MESSAGE. Never route based on what earlier turns or the brand profile discuss (the presence of budgets or goals in the profile is NOT an intent).
2. If the user mentions `@` with a competitor name or tool (e.g. `@Tie House`, `@web_analyzer`, `@reviews`, `@gap_analysis`) requesting live analysis, reviews, web audit, or market gaps → ALWAYS route to 'nour'.
3. Competitor mentions WITHOUT an explicit request to search the web NOW or run live tools → karim (or chat if about stored profile), NOT nour.
4. Live campaign ops (salma) vs campaign strategy/creation (omar): 'salma' handles live/running campaign tweaks only; 'omar' handles strategy, roadmaps, and creating new campaigns.
5. Content calendar / post scheduling: When the user asks to schedule posts across days/dates, build a content calendar, or display them in the website calendar ("في الكالندر", "calendar", "schedule posts", "جدول نشر", "خليهم في الـX ايام") → ALWAYS route to 'omar'.
6. When unsure or if the request is ambiguous or a clarification/greeting → tie-break toward 'chat'.
"""

BLOCK_STRUCTURER_PROMPT = """Convert the following text into structured React UI Blocks. 
CRITICAL RULE FOR THE 'type' FIELD: 
- For paragraphs and headings, use exactly 'text' (NOT 'TextBlock').
- For lists, bullet points, competitor summaries, and research points, use exactly 'findings' (NOT 'FindingsBlock').
Ensure the format and 'type' values are strictly adhered to.

FINDINGS REQUIREMENTS:
- Every finding MUST have a concise 'title' AND substantive 'detail' (the full explanation or content). NEVER emit an empty detail or just title followed by colon.
- If sources, URLs, or citations are present in the text, wire them into 'source' (URL or provider name) and 'evidence' (list of {title, url, snippet}).
- Set 'confidence' to 'high', 'medium', or 'low' (or float 0.0-1.0) when appropriate.

CRITICAL CONSTRAINT:
Only create 'action_card' or 'question_with_options' blocks that EXPLICITLY exist in the source text. Never invent follow-up CTAs. 'recommendation' blocks must come from the source text, not be generated by you."""


def format_tool_payload_to_markdown(tool_name: str, args: dict[str, Any]) -> str:
    """Renders structured tool arguments into readable Markdown with headings and bullets.
    
    Guarantees no raw JSON dict braces or \\uXXXX unicode escapes reach the user or Message.content.
    """
    if not isinstance(args, dict):
        return str(args)

    lines: list[str] = []

    if tool_name == "submit_research_report" or "market_overview" in args or "competitor_summary" in args:
        if args.get("market_overview"):
            lines.append(f"## Market Overview\n{args['market_overview']}\n")

        competitors = args.get("competitor_summary") or []
        if isinstance(competitors, list) and competitors:
            lines.append("## Competitor Summary")
            for comp in competitors:
                if isinstance(comp, dict):
                    brand = comp.get("brand") or comp.get("name") or "Competitor"
                    lines.append(f"### {brand}")
                    if comp.get("positioning"):
                        lines.append(f"- **Positioning:** {comp['positioning']}")
                    strengths = comp.get("strengths") or []
                    if isinstance(strengths, list) and strengths:
                        lines.append(f"- **Strengths:** {'; '.join(str(s) for s in strengths)}")
                    weaknesses = comp.get("weaknesses_or_gaps") or comp.get("weaknesses") or []
                    if isinstance(weaknesses, list) and weaknesses:
                        lines.append(f"- **Weaknesses/Gaps:** {'; '.join(str(w) for w in weaknesses)}")
                    if comp.get("source"):
                        lines.append(f"- **Source:** {comp['source']}")
                elif isinstance(comp, str):
                    lines.append(f"- {comp}")
            lines.append("")

        section_map = [
            ("pricing_insights", "Pricing Insights"),
            ("offer_insights", "Offer Insights"),
            ("shipping_insights", "Shipping Insights"),
            ("search_insights", "Search Insights"),
            ("social_insights", "Social Insights"),
            ("opportunities_for_target_brand", "Opportunities for Target Brand"),
            ("risks", "Market Risks"),
        ]
        for key, header in section_map:
            items = args.get(key)
            if isinstance(items, list) and items:
                lines.append(f"## {header}")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")

        sources = args.get("sources") or []
        if isinstance(sources, list) and sources:
            lines.append("## Sources")
            for s in sources:
                lines.append(f"- {s}")
            lines.append("")

        evidence_items = args.get("evidence") or []
        if isinstance(evidence_items, list) and evidence_items:
            lines.append("## Evidence")
            for ev in evidence_items:
                if isinstance(ev, dict):
                    ev_title = ev.get("title") or "Source"
                    ev_url = ev.get("url")
                    if ev_url:
                        lines.append(f"- [{ev_title}]({canonicalize_url(ev_url)})")
                    else:
                        lines.append(f"- {ev_title}")
            lines.append("")
    elif "summary" in args and "details" in args:
        if args.get("summary"):
            lines.append(str(args["summary"]))
            lines.append("")
        details = args.get("details")
        if isinstance(details, list) and details:
            for item in details:
                lines.append(f"- {item}")
            lines.append("")
    else:
        for k, v in args.items():
            header = k.replace("_", " ").title()
            if isinstance(v, list):
                lines.append(f"## {header}")
                for item in v:
                    lines.append(f"- {item}")
                lines.append("")
            elif isinstance(v, dict):
                lines.append(f"## {header}")
                for sub_k, sub_v in v.items():
                    lines.append(f"- **{sub_k.replace('_', ' ').title()}:** {sub_v}")
                lines.append("")
            elif v is not None:
                lines.append(f"**{header}:** {v}\n")

    rendered = "\n".join(lines).strip()
    return rendered or "Research report completed."


async def _persist_content_calendar_posts(business_id: str, posts: list[dict[str, Any]]) -> None:
    """Save generated calendar posts directly to ContentItem table."""
    try:
        from datetime import date, timedelta
        import uuid
        from app.db.models import ContentItem
        from app.db.session import session_scope

        today = date.today()
        async with session_scope() as session:
            for p in posts:
                day_num = int(p.get("day", 1))
                post_date = (today + timedelta(days=max(0, day_num - 1))).strftime("%Y-%m-%d")
                raw_channel = str(p.get("channel", "instagram")).lower()
                valid_channels = {"instagram", "facebook", "tiktok", "email", "blog"}
                channel = raw_channel if raw_channel in valid_channels else "instagram"

                raw_format = str(p.get("format", "post")).lower()
                valid_formats = {"post", "reel", "story", "ad", "email", "article"}
                fmt = raw_format if raw_format in valid_formats else "post"

                pillar = str(p.get("content_pillar") or "Marketing")
                concept = str(p.get("concept") or "")
                title = concept[:250] if concept else f"{pillar} {fmt.title()}"
                hook = p.get("caption_hook")
                cta = p.get("call_to_action", "")
                caption = f"{hook}\n\n{concept}\n\n👉 {cta}".strip() if hook else concept
                time_val = str(p.get("best_time") or "18:00")[:5]
                if ":" not in time_val:
                    time_val = "18:00"

                item = ContentItem(
                    id=f"cnt_{uuid.uuid4().hex[:20]}",
                    business_id=business_id,
                    date=post_date,
                    time=time_val,
                    channel=channel,
                    format=fmt,
                    title=title,
                    hook=hook,
                    caption=caption,
                    hashtags=[f"#{channel}", f"#{pillar.replace(' ', '')}"],
                    status="scheduled",
                    pillar=pillar,
                )
                session.add(item)
    except Exception as exc:
        logger.error("failed_to_persist_content_calendar_posts", extra={"error": str(exc), "business_id": business_id})


async def orchestrate(
    prompt: str, 
    history: list[dict[str, Any]], 
    event_queue: asyncio.Queue[dict[str, Any]] | None = None,
    user_context: str | None = None,
    conversation_summary: str | None = None,
    active_plans: str | None = None,
    business_id: str | None = None,
) -> MessageWithBlocks | None:
    """The Brain that reads the prompt and routes it to the correct agent."""
    if event_queue:
        await event_queue.put({"type": "step_started", "step": "orchestrator", "detail": "Analyzing your prompt and deciding who to route it to..."})

    ai_settings = get_ai_settings()
    # Enforce deterministic routing with temperature=0.0
    llm = get_llm(provider=ai_settings.llm_provider, temperature=0.0).chat.with_structured_output(RoutingDecision)
    
    try:
        chat_history = []
        # Inject ONLY brand facts into the router call. PLANS are never injected into the router.
        if user_context and user_context.strip():
            chat_history.append(SystemMessage(content=f"USER BRAND & BUSINESS PROFILE (Brand Facts):\n{user_context.strip()}"))

        if conversation_summary and conversation_summary.strip():
            chat_history.append(SystemMessage(content=f"PREVIOUS CONVERSATION SUMMARY (Earlier topics in this chat):\n{conversation_summary.strip()}"))

        # Apply intelligent session history compression to prune token bloat
        compact_history = compress_history_messages(history)
        for msg in compact_history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))
                
        routing_sys = SystemMessage(content=ROUTER_SYSTEM_PROMPT)
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
        
        # Inject active plans ONLY into strategy agents (omar, salma) when their intent is selected.
        agent_messages = list(chat_history)
        if decision.agent in ("omar", "salma") and active_plans and active_plans.strip():
            agent_messages.append(SystemMessage(content=f"ACTIVE MARKETING PLANS & CAMPAIGN STRATEGIES:\n{active_plans.strip()}"))

        agent_messages.append(HumanMessage(content=f"Context is provided above. Please execute this NEW request: {prompt}"))

        config = {"configurable": {"thread_id": "test_thread", "business_id": business_id or "default"}}
        final_state = await graphs[decision.agent].ainvoke({"messages": agent_messages}, config)
        
        # If Omar generated a content calendar, persist the posts to the ContentItem DB table
        if decision.agent == "omar" and business_id:
            cal_data = final_state.get("content_calendar")
            if isinstance(cal_data, dict) and cal_data.get("posts"):
                await _persist_content_calendar_posts(business_id, cal_data["posts"])
        
        if event_queue:
            await event_queue.put({"type": "step_started", "step": "formatting", "detail": "Structuring the final response for the UI..."})
            
        # Extract final text or tool arguments
        messages = final_state["messages"]
        last_msg = messages[-1]
        
        collected_evidence: list[dict[str, Any]] = []
        for msg in messages:
            if getattr(msg, "type", "") == "tool":
                tool_content = str(getattr(msg, "content", ""))
                for match in re.finditer(r"\[([^\]]+)\]\((https?://[^\)]+)\)", tool_content):
                    c_url = canonicalize_url(match.group(2))
                    if c_url and not any(e.get("url") == c_url for e in collected_evidence):
                        collected_evidence.append({
                            "title": match.group(1).strip(),
                            "url": c_url,
                            "source": getattr(msg, "name", "tool"),
                        })
                for raw_url in re.findall(r"https?://[^\s)\]]+", tool_content):
                    c_url = canonicalize_url(raw_url)
                    if c_url and not any(e.get("url") == c_url for e in collected_evidence):
                        domain = c_url.split("/")[2] if "//" in c_url else "Web Source"
                        collected_evidence.append({
                            "title": domain,
                            "url": c_url,
                            "source": getattr(msg, "name", "tool"),
                        })

        if last_msg.type == "tool":
            # Graph ended by calling a tool like 'submit_research_report'. Render structured args into Markdown.
            ai_msg = messages[-2]
            tool_name = getattr(last_msg, "name", "")
            if getattr(ai_msg, "tool_calls", None):
                tool_args = ai_msg.tool_calls[0]["args"]
                tool_name = tool_name or ai_msg.tool_calls[0].get("name", "")
                final_msg = format_tool_payload_to_markdown(tool_name, tool_args)
                for s in tool_args.get("sources") or []:
                    if isinstance(s, str):
                        c_url = canonicalize_url(s)
                        if c_url and not any(e.get("url") == c_url for e in collected_evidence):
                            collected_evidence.append({"title": "Source", "url": c_url, "source": tool_name})
                for ev in tool_args.get("evidence") or []:
                    if isinstance(ev, dict) and ev.get("url"):
                        c_url = canonicalize_url(ev["url"])
                        if c_url and not any(e.get("url") == c_url for e in collected_evidence):
                            collected_evidence.append({
                                "title": ev.get("title") or "Source",
                                "url": c_url,
                                "source": ev.get("source") or tool_name,
                                "snippet": ev.get("snippet"),
                            })
            else:
                final_msg = str(last_msg.content)
        else:
            final_msg = str(last_msg.content)
    else:
        # Fast Conversational Agent for follow-ups, profile queries, and general chat
        collected_evidence = []
        if event_queue:
            detail = "Answering follow-up question..." if decision.agent == "chat" else "Answering general question..."
            await event_queue.put({"type": "step_started", "step": "chat", "detail": detail})
        general_llm = get_llm(provider=ai_settings.llm_provider).chat
        sys_msg = SystemMessage(content=(
            "You are a fast, conversational AI assistant. You will be provided with the conversation history and Brand Facts for context. "
            "CRITICAL: Do NOT re-answer previous questions. ONLY answer the user's latest question. Do not use tools.\n"
            "OUTPUT CONTRACT: Answer the question factually and stop. STRONGLY PROHIBITED: inventing or pitching marketing strategies, campaigns, budgets, or next actions the user did not ask for."
        ))
        fallback_res = await general_llm.ainvoke([sys_msg] + chat_history + [HumanMessage(content=f"Context from previous turns is provided above. \n\nNow, answer this NEW question: {prompt}")])
        final_msg = str(fallback_res.content)

    block_llm = get_llm(provider=ai_settings.llm_provider).chat.with_structured_output(MessageWithBlocks)
    try:
        structured = await block_llm.ainvoke([
            SystemMessage(content=BLOCK_STRUCTURER_PROMPT),
            HumanMessage(content=final_msg)
        ])
    except Exception as e:
        print(f"[ORCHESTRATOR] UI Block structuring failed, falling back to plain text. Error: {e}")
        from app.schemas.blocks import TextBlock
        structured = MessageWithBlocks(blocks=[TextBlock(text=final_msg)])

    # Post-process findings to attach evidence and validate receipt
    is_research_turn = decision.agent in ("nour", "research") or bool(collected_evidence)
    for block in structured.blocks:
        if isinstance(block, FindingsBlock):
            for finding in block.findings:
                if not finding.evidence and collected_evidence:
                    finding.evidence = [Evidence.model_validate(e) for e in collected_evidence]
                    if not finding.source and collected_evidence[0].get("url"):
                        finding.source = collected_evidence[0]["url"]
                if is_research_turn and not finding.evidence:
                    logger.warning(
                        "research_finding_missing_evidence",
                        extra={"title": finding.title, "agent": decision.agent},
                    )

    return structured
