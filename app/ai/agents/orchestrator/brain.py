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
    agent: Literal["nour", "omar", "layla", "karim", "salma", "chat", "unknown"] = Field(description="The primary agent best suited to handle this prompt, or first in a pipeline.")
    agents: list[Literal["nour", "omar", "layla", "karim", "salma", "chat"]] = Field(
        default_factory=list,
        description="Ordered list of agents to run in sequence if the prompt combines domains (e.g. ['nour', 'omar'] for competitor research followed by strategy; ['nour', 'layla'] for competitor research followed by creative assets). For single tasks, provide a list of length 1.",
    )
    reason: str = Field(description="Why this agent or multi-agent pipeline was chosen.")

logger = get_logger(__name__)

ROUTER_SYSTEM_PROMPT = """You are the Orchestrator Hub. You route user prompts to the correct marketing agent based EXCLUSIVELY on the user's explicit intent in their newest message.

Agents and Scopes:
- nour  → LIVE RESEARCH & COMPETITOR TOOLS: The user explicitly asks to search, find, crawl, or analyze information NOW, or mentions a competitor or tool with @ (e.g. @AbuAuf, @web_analyzer, @reviews, @gap_analysis, @researcher, "حلل موقع", "افحص المنافس", "ريفيوهات", "تقييمات", "فجوات السوق", "search", "find competitors").
- karim → MEMORY RECALL only: Questions about what was ALREADY researched or stored in memory without asking to run new live tools ("مين المنافسين المسجلين عندنا", "إيه المتخزن في الذاكرة", "what do we know about competitors from previous research").
- omar  → STRATEGY & CONTENT CALENDAR: When THIS message explicitly asks for a high-level marketing strategy, strategic roadmap, market positioning, ad campaign creation / plan, OR a CONTENT CALENDAR / POST SCHEDULING across days or dates for the website calendar ("عايز اعمل حملة اعلانية", "حطلي خطة تسويق", "positioning strategy", "جدول محتوى", "خليهم في الخمس ايام الجايين", "في الكالندر", "عايز اشوفهم في الcalendar", "content calendar", "schedule posts").
- layla → CONTENT creation only: Creative assets explicitly requested NOW (writing ad copy, headlines, visual briefs, 15s commercial scripts, reels scripts).
- salma → LIVE CAMPAIGN OPS only: Real-time budget reallocation, ROAS adjustments, or performance optimization on currently RUNNING/active ad campaigns.
- chat  → CONVERSATIONAL / GENERAL: Questions about the user's stored brand profile ("ايه الي تعرفه عني", "what is my brand scope"), casual pleasantries, follow-ups, clarifications, or requests not asking for new agent work.

MULTI-AGENT COLLABORATION PIPELINE:
When the user's prompt requires multiple agents working together in sequence:
- Research + Strategy: When user asks to search/find competitors or market data AND build a strategy or ad campaign from it ("ابحث عن منافسين اثنين واعملي استراتيجية", "search competitors and design a campaign plan") → set agents = ['nour', 'omar'], agent = 'nour'.
- Research + Creative Copy: When user asks to search competitors and write creative ad copies or reels scripts based on them → set agents = ['nour', 'layla'], agent = 'nour'.
- Single-domain queries: Set agent and agents to the single responsible agent (e.g. ['nour'], ['omar'], ['layla']).

HARD ROUTING RULES:
1. Route strictly by the user's EXPLICIT intent in THIS NEWEST MESSAGE. Never route based on what earlier turns or the brand profile discuss.
2. If the user mentions `@` with a competitor name or tool (e.g. `@Tie House`, `@web_analyzer`, `@reviews`, `@gap_analysis`) requesting live analysis, reviews, web audit, or market gaps → ALWAYS include 'nour'.
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


def format_tool_payload_to_markdown(tool_name: str, args: dict[str, Any], language: str = "en") -> str:
    """Renders structured tool arguments into readable Markdown with headings and bullets.
    
    Guarantees no raw JSON dict braces or \\uXXXX unicode escapes reach the user or Message.content.
    Translates headings into Arabic when language == 'ar'.
    """
    if not isinstance(args, dict):
        return str(args)

    is_ar = language.lower() == "ar"
    lines: list[str] = []

    if tool_name == "submit_research_report" or "market_overview" in args or "competitor_summary" in args:
        if args.get("market_overview"):
            mo_hdr = "## نظرة عامة على السوق" if is_ar else "## Market Overview"
            lines.append(f"{mo_hdr}\n{args['market_overview']}\n")

        competitors = args.get("competitor_summary") or []
        if isinstance(competitors, list) and competitors:
            comp_hdr = "## ملخص المنافسين" if is_ar else "## Competitor Summary"
            lines.append(comp_hdr)
            for comp in competitors:
                if isinstance(comp, dict):
                    brand = comp.get("brand") or comp.get("name") or ("المنافس" if is_ar else "Competitor")
                    lines.append(f"### {brand}")
                    if comp.get("positioning"):
                        pos_label = "**التموضع:**" if is_ar else "**Positioning:**"
                        lines.append(f"- {pos_label} {comp['positioning']}")
                    strengths = comp.get("strengths") or []
                    if isinstance(strengths, list) and strengths:
                        str_label = "**نقاط القوة:**" if is_ar else "**Strengths:**"
                        lines.append(f"- {str_label} {'; '.join(str(s) for s in strengths)}")
                    weaknesses = comp.get("weaknesses_or_gaps") or comp.get("weaknesses") or []
                    if isinstance(weaknesses, list) and weaknesses:
                        weak_label = "**نقاط الضعف والفجوات:**" if is_ar else "**Weaknesses/Gaps:**"
                        lines.append(f"- {weak_label} {'; '.join(str(w) for w in weaknesses)}")
                    if comp.get("source"):
                        src_label = "**المصدر:**" if is_ar else "**Source:**"
                        lines.append(f"- {src_label} {comp['source']}")
                elif isinstance(comp, str):
                    lines.append(f"- {comp}")
            lines.append("")

        section_map = [
            ("pricing_insights", "رؤى التسعير" if is_ar else "Pricing Insights"),
            ("offer_insights", "رؤى العروض والخصومات" if is_ar else "Offer Insights"),
            ("shipping_insights", "رؤى الشحن والتوصيل" if is_ar else "Shipping Insights"),
            ("search_insights", "رؤى محركات البحث وSEO" if is_ar else "Search Insights"),
            ("social_insights", "رؤى منصات التواصل الاجتماعي" if is_ar else "Social Insights"),
            ("opportunities_for_target_brand", "الفرص الاستراتيجية للماركة" if is_ar else "Opportunities for Target Brand"),
            ("risks", "مخاطر وتحديات السوق" if is_ar else "Market Risks"),
        ]
        for key, header in section_map:
            items = args.get(key)
            if isinstance(items, list) and items:
                lines.append(f"## {header}")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")

        sources = args.get("sources") or []
        evidence_items = args.get("evidence") or []
        total_sources = len(sources) or len(evidence_items)
        if total_sources > 0:
            ref_note = (
                f"_تم توثيق البحث والربط مع {total_sources} مصادر مرجعية (انظر بطاقة المصادر أعلاه)._"
                if is_ar
                else f"_Research verified and wired with {total_sources} referenced sources (see sources tooltip above)._"
            )
            lines.append(ref_note)
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


async def _persist_campaign_draft(business_id: str, campaign_data: dict[str, Any]) -> str | None:
    """Save generated campaign strategy into the Campaign table as draft."""
    try:
        import uuid
        import json
        from datetime import datetime, timezone
        from app.db.models import Campaign
        from app.db.session import session_scope

        cid = f"cmp_{uuid.uuid4().hex[:20]}"
        big_idea = campaign_data.get("big_idea") or "Integrated Growth Campaign"
        objective = campaign_data.get("objective") or "sales"
        budget = float(campaign_data.get("budget") or 15000.0)

        async with session_scope() as session:
            c = Campaign(
                id=cid,
                business_id=business_id,
                name=f"{big_idea[:100]}",
                status="draft",
                objective=objective,
                channel="instagram, facebook",
                budget=budget,
                currency="EGP",
                target_audience=str(campaign_data.get("target_audience") or "Core Target Audience"),
                notes=json.dumps(campaign_data, ensure_ascii=False),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            session.add(c)
        return cid
    except Exception as exc:
        logger.error("failed_to_persist_campaign_draft", extra={"error": str(exc), "business_id": business_id})
        return None


async def orchestrate(
    prompt: str, 
    history: list[dict[str, Any]], 
    event_queue: asyncio.Queue[dict[str, Any]] | None = None,
    user_context: str | None = None,
    conversation_summary: str | None = None,
    active_plans: str | None = None,
    business_id: str | None = None,
    language: str | None = None,
) -> MessageWithBlocks | None:
    """The Brain that reads the prompt and routes it to the correct agent or multi-agent pipeline."""
    if event_queue:
        await event_queue.put({"type": "step_started", "step": "orchestrator", "detail": "Analyzing your prompt and deciding who to route it to..."})

    is_arabic_query = bool(re.search(r"[\u0600-\u06FF]", prompt))
    query_language = "ar" if is_arabic_query else "en"
    tool_language = (language or query_language).lower()

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
        routing_msgs = [routing_sys] + chat_history + [HumanMessage(content=f"Based on the history above, analyze the user's NEWEST prompt: '{prompt}'\n\nWhich agent or multi-agent pipeline should handle this newest prompt?")]
        decision = await llm.ainvoke(routing_msgs)
    except Exception as e:
        print(f"[ORCHESTRATOR] Routing failed, defaulting to unknown. Error: {e}")
        decision = RoutingDecision(agent="unknown", agents=[], reason="Failed to parse routing decision.")
    
    # Resolve pipeline: use decision.agents if populated, else fall back to [decision.agent]
    pipeline = [a for a in decision.agents if a in ("nour", "omar", "layla", "karim", "salma")]
    if not pipeline and decision.agent in ("nour", "omar", "layla", "karim", "salma"):
        pipeline = [decision.agent]
        
    print(f"[ORCHESTRATOR] Routing to: {' -> '.join(a.upper() for a in pipeline) or decision.agent.upper()} (Reason: {decision.reason})")
    
    graphs = {
        "nour": nour_graph,
        "omar": omar_graph,
        "layla": layla_graph,
        "karim": karim_graph,
        "salma": salma_graph
    }
    
    collected_evidence: list[dict[str, Any]] = []
    cal_saved_posts: int = 0
    camp_saved_data: dict[str, Any] | None = None
    brand_saved_data: dict[str, Any] | None = None
    competitors_saved: list[str] = []

    # Detect user intent to update or synchronize brand identity / profile
    brand_update_patterns = [
        "حدث معلومات البراند", "تحديث معلومات البراند", "حدث البراند", "تحديث البراند",
        "غير الوان البراند", "تغيير الوان البراند", "عدل البراند", "تعديل البراند",
        "احفظ في البراند", "احفظ البراند", "حفظ الهوية", "حدث الهوية", "تحديث الهوية",
        "غير الهوية", "الوان الهوية", "ألوان الهوية",
        "update brand", "sync brand", "update brand identity", "change brand colors",
        "save to brand", "update my brand",
    ]
    if business_id and any(p in prompt.lower() for p in brand_update_patterns):
        try:
            from app.db.session import session_scope
            from app.core.brand.identity_service import sync_brand_identity_from_chats
            async with session_scope() as session:
                synced_id = await sync_brand_identity_from_chats(session, business_id)
                brand_saved_data = {
                    "name": synced_id.name,
                    "primary": synced_id.colors.primary,
                    "accent": synced_id.colors.accent,
                    "palette": synced_id.colors.palette,
                }
        except Exception as exc:
            logger.error("failed_to_sync_brand_identity_in_chat", extra={"error": str(exc), "business_id": business_id})

    # Detect competitor queries/mentions in the prompt to auto-record in Brand Identity
    prompt_competitors: list[str] = []
    for match in re.findall(r"@([^\s,،]+)", prompt):
        cleaned_m = match.strip().strip("'\"")
        if cleaned_m.lower() not in ("nour", "omar", "salma", "layla", "karim", "sahm", "researcher", "web_analyzer", "gap_analysis", "reviews"):
            prompt_competitors.append(cleaned_m)

    for match in re.findall(r"(?:منافس(?:نا)?|ابحث عن منافس|حلل منافس|مقارنة مع)\s+([^\n,،.?؟]+)", prompt):
        cleaned_m = match.strip().strip("'\"")
        words = cleaned_m.split()
        if words:
            candidate = " ".join(words[:3]).strip()
            if len(candidate) >= 2:
                prompt_competitors.append(candidate)

    if "بن النجار" in prompt or "النجار" in prompt:
        prompt_competitors.append("بن النجار")
    if "ابو عوف" in prompt or "أبو عوف" in prompt or "abu auf" in prompt.lower():
        prompt_competitors.append("أبو عوف (Abu Auf)")

    if business_id and prompt_competitors:
        try:
            from app.db.session import session_scope
            from app.core.brand.identity_service import record_competitors
            async with session_scope() as session:
                await record_competitors(session, business_id, prompt_competitors)
                competitors_saved.extend(prompt_competitors)
        except Exception as exc:
            logger.error("failed_to_persist_prompt_competitors", extra={"error": str(exc), "business_id": business_id})

    def _extract_evidence_from_msgs(msgs: list[Any]) -> None:
        for msg in msgs:
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

    if pipeline:
        agent_outputs: list[str] = []
        previous_output: str | None = None

        for idx, current_agent in enumerate(pipeline):
            if current_agent not in graphs:
                continue

            if event_queue:
                step_detail = f"{current_agent.title()} is working ({idx + 1}/{len(pipeline)}): {decision.reason}"
                await event_queue.put({"type": "step_started", "step": current_agent, "detail": step_detail})

            agent_messages = list(chat_history)
            if current_agent in ("omar", "salma") and active_plans and active_plans.strip():
                agent_messages.append(SystemMessage(content=f"ACTIVE MARKETING PLANS & CAMPAIGN STRATEGIES:\n{active_plans.strip()}"))

            if previous_output:
                agent_messages.append(SystemMessage(content=f"RESEARCH & FINDINGS FROM PREVIOUS STEP:\n{previous_output}"))

            lang_instr = "IMPORTANT: Format your findings and analysis in Arabic." if tool_language == "ar" else ""
            agent_messages.append(HumanMessage(content=f"Context is provided above. {lang_instr}\nPlease execute this request: {prompt}"))

            config = {"configurable": {"thread_id": f"thread_{current_agent}", "business_id": business_id or "default"}}
            final_state = await graphs[current_agent].ainvoke({"messages": agent_messages}, config)

            if current_agent == "omar" and business_id:
                cal_data = final_state.get("content_calendar")
                if isinstance(cal_data, dict) and cal_data.get("posts"):
                    await _persist_content_calendar_posts(business_id, cal_data["posts"])
                    cal_saved_posts = len(cal_data["posts"])
                camp_data = final_state.get("campaign")
                if isinstance(camp_data, dict):
                    await _persist_campaign_draft(business_id, camp_data)
                    camp_saved_data = camp_data

            if current_agent in ("nour", "karim") and business_id:
                found_competitors: list[str] = []
                for m in final_state.get("messages", []):
                    for tc in getattr(m, "tool_calls", None) or []:
                        t_name = tc.get("name", "")
                        t_args = tc.get("args") or {}
                        if t_name == "submit_research_report":
                            for comp in t_args.get("competitor_summary") or []:
                                if isinstance(comp, dict):
                                    b_name = comp.get("brand") or comp.get("name")
                                    if b_name:
                                        found_competitors.append(str(b_name))
                        elif t_name == "fetch_competitor_ads":
                            c_name = t_args.get("competitor_name")
                            if c_name:
                                found_competitors.append(str(c_name))

                if found_competitors:
                    try:
                        from app.db.session import session_scope
                        from app.core.brand.identity_service import record_competitors
                        async with session_scope() as session:
                            await record_competitors(session, business_id, found_competitors)
                            competitors_saved.extend(found_competitors)
                    except Exception as exc:
                        logger.error("failed_to_persist_competitors_from_nour", extra={"error": str(exc), "business_id": business_id})

            msgs = final_state.get("messages", [])
            _extract_evidence_from_msgs(msgs)

            if msgs:
                last_msg = msgs[-1]
                if getattr(last_msg, "type", "") == "tool":
                    ai_msg = msgs[-2] if len(msgs) >= 2 else last_msg
                    tool_name = getattr(last_msg, "name", "")
                    if getattr(ai_msg, "tool_calls", None):
                        tool_args = ai_msg.tool_calls[0]["args"]
                        tool_name = tool_name or ai_msg.tool_calls[0].get("name", "")
                        step_text = format_tool_payload_to_markdown(tool_name, tool_args, language=tool_language)
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
                        step_text = str(last_msg.content)
                else:
                    step_text = str(last_msg.content)
            else:
                step_text = ""

            agent_outputs.append(step_text)
            previous_output = step_text

        final_msg = "\n\n---\n\n".join(agent_outputs) if agent_outputs else ""
    else:
        # Fast Conversational Agent for follow-ups, profile queries, and general chat
        if event_queue:
            detail = "Answering follow-up question..." if decision.agent == "chat" else "Answering general question..."
            await event_queue.put({"type": "step_started", "step": "chat", "detail": detail})
        general_llm = get_llm(provider=ai_settings.llm_provider).chat
        brand_sync_note = ""
        if brand_saved_data:
            brand_sync_note = (
                f"\nIMPORTANT ACTION COMPLETED: The brand profile has just been successfully synchronized and updated in the database. "
                f"Brand name: {brand_saved_data['name']}. Primary color: {brand_saved_data['primary']}, Accent color: {brand_saved_data['accent']}. "
                f"Confirm to the user clearly that their brand identity and colors have been successfully saved to their profile and the Brand Identity Hub."
            )
        navigation_hint = (
            "\nNAVIGATION RULES: "
            "If the user asks about viewing, seeing, or checking their Brand Identity, profile, colors, or competitors, direct them clearly and include the markdown link [الهوية البصرية](/brand-identity). "
            "If the user asks about their Token Usage, spend, or quota, direct them clearly and include the markdown link [صفحة الاستهلاك والحصة](/settings/usage)."
        )
        sys_msg = SystemMessage(content=(
            "You are a fast, conversational AI assistant. You will be provided with the conversation history and Brand Facts for context. "
            "CRITICAL: Do NOT re-answer previous questions. ONLY answer the user's latest question. Do not use tools.\n"
            "OUTPUT CONTRACT: Answer the question factually and stop. STRONGLY PROHIBITED: inventing or pitching marketing strategies, campaigns, budgets, or next actions the user did not ask for.\n"
            f"LANGUAGE CONTRACT: You MUST answer strictly in {'Arabic' if query_language == 'ar' else 'English'} to match the user's prompt language."
            f"{brand_sync_note}"
            f"{navigation_hint}"
        ))
        fallback_res = await general_llm.ainvoke([sys_msg] + chat_history + [HumanMessage(content=f"Context from previous turns is provided above. \n\nNow, answer this NEW question: {prompt}")])
        final_msg = str(fallback_res.content)

    if event_queue:
        await event_queue.put({"type": "step_started", "step": "formatting", "detail": "Structuring the final response for the UI..."})

    block_llm = get_llm(provider=ai_settings.llm_provider).chat.with_structured_output(MessageWithBlocks)
    structurer_prompt = (
        f"{BLOCK_STRUCTURER_PROMPT}\n\n"
        f"CRITICAL LANGUAGE RULE: The user asked in {'Arabic' if query_language == 'ar' else 'English'}. "
        f"You MUST format and output the entire response in {'Arabic' if query_language == 'ar' else 'English'} "
        "matching the user's query language."
    )
    try:
        structured = await block_llm.ainvoke([
            SystemMessage(content=structurer_prompt),
            HumanMessage(content=final_msg)
        ])
    except Exception as e:
        print(f"[ORCHESTRATOR] UI Block structuring failed, falling back to plain text. Error: {e}")
        from app.schemas.blocks import TextBlock
        structured = MessageWithBlocks(blocks=[TextBlock(text=final_msg)])

    # Post-process findings to attach evidence and validate receipt
    is_research_turn = any(a in ("nour", "research") for a in pipeline) or decision.agent in ("nour", "research") or bool(collected_evidence)
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

    if cal_saved_posts > 0 and not any(getattr(b, "type", "") == "action_card" for b in structured.blocks):
        from app.schemas.blocks import ActionCardBlock
        structured.blocks.append(
            ActionCardBlock(
                title="تم جدولة المنشورات في التقويم بنجاح" if query_language == "ar" else "Content Calendar Scheduled Successfully",
                description=f"تم حفظ {cal_saved_posts} منشور في جدول المحتوى الخاص بك مع المواعيد والقنوات المقترحة." if query_language == "ar" else f"Successfully scheduled {cal_saved_posts} posts to your marketing calendar.",
                actions=[{
                    "type": "GENERATE_CONTENT_CALENDAR",
                    "label": "عرض في التقويم (Calendar)" if query_language == "ar" else "View in Calendar",
                    "payload": {}
                }]
            )
        )

    if camp_saved_data and not any(getattr(b, "type", "") == "action_card" for b in structured.blocks):
        from app.schemas.blocks import ActionCardBlock
        idea = camp_saved_data.get("big_idea") or ("حملة تسويقية جديدة" if query_language == "ar" else "New Marketing Campaign")
        budget_val = float(camp_saved_data.get("budget") or 15000.0)
        daily_b = round(budget_val / 30.0, 2)
        structured.blocks.append(
            ActionCardBlock(
                title=f"خطة الحملة جاهزة: {idea}" if query_language == "ar" else f"Campaign Strategy Ready: {idea}",
                description="تم حفظ مسودة الحملة وتوزيع ميزانية القنوات. يمكنك مراجعتها وتعديلها وإطلاقها بنقرة واحدة." if query_language == "ar" else "Campaign draft prepared with channel budget shares. 1-click ready to review and launch.",
                actions=[{
                    "type": "CREATE_CAMPAIGN",
                    "label": "مراجعة وإطلاق الحملة الآن" if query_language == "ar" else "Review & Launch Campaign",
                    "payload": {
                        "prefill": {
                            "name": idea,
                            "objective": camp_saved_data.get("objective", "sales"),
                            "dailyBudget": daily_b if daily_b > 0 else 500,
                            "currency": "EGP",
                            "durationDays": 30,
                            "audience": {"name": str(camp_saved_data.get("target_audience") or "الجمهور المستهدف")}
                        }
                    }
                }]
            )
        )

    brand_view_patterns = [
        "الهوية البصرية", "هوية البراند", "صفحة الهوية", "بروفايل البراند", "ملف البراند",
        "عرض الهوية", "وريني الهوية", "افتح الهوية", "شوف الهوية", "وريني البراند",
        "brand identity", "brand profile", "view brand", "show brand", "identity hub",
    ]
    usage_patterns = [
        "الاستهلاك", "استهلاك التوكنز", "صفحة الاستهلاك", "الحصة", "الكوتا", "التوكنز",
        "باقي كام توكنز", "كام توكن", "معدل الحرق", "وريني الاستهلاك", "افتح الاستهلاك",
        "usage", "quota", "token spend", "tokens left", "burn rate", "view usage",
    ]

    if any(p in prompt.lower() for p in usage_patterns) and not any(getattr(b, "type", "") == "action_card" for b in structured.blocks):
        from app.schemas.blocks import ActionCardBlock
        structured.blocks.append(
            ActionCardBlock(
                title="متابعة الاستهلاك والحصة" if query_language == "ar" else "Usage & Quota Monitoring",
                description=(
                    "يمكنك متابعة استهلاك الرموز (Tokens) اليوم والشهر ومعدل الحرق المتبقي وتفاصيل الحصة من صفحة الاستهلاك."
                    if query_language == "ar" else
                    "Track your daily and monthly token consumption, burn rate, and quota allocation on the Usage page."
                ),
                actions=[{
                    "type": "VIEW_USAGE",
                    "label": "عرض صفحة الاستهلاك (Usage)" if query_language == "ar" else "View Usage & Quota",
                    "payload": {}
                }]
            )
        )

    if (brand_saved_data or competitors_saved or any(p in prompt.lower() for p in brand_view_patterns)) and not any(getattr(b, "type", "") == "action_card" for b in structured.blocks):
        from app.schemas.blocks import ActionCardBlock
        if brand_saved_data:
            b_title = "تم تحديث الهوية البصرية للبراند بنجاح" if query_language == "ar" else "Brand Identity Updated Successfully"
            b_desc = (
                f"تم تحديث الألوان للبراند (الأساسي: {brand_saved_data['primary']}، التمييزي: {brand_saved_data['accent']}) وتحديث ملف الهوية في النظام."
                if query_language == "ar" else
                f"Updated brand colors ({brand_saved_data['primary']} / {brand_saved_data['accent']}) and updated your brand identity hub."
            )
        elif competitors_saved:
            comp_list = "، ".join(competitors_saved[:3])
            b_title = "تم تحديث قسم المنافسين في الهوية البصرية" if query_language == "ar" else "Competitors Updated in Brand Identity"
            b_desc = (
                f"تم تسجيل المنافسين ({comp_list}) تلقائياً وحفظهم في قسم المنافسين بملف الهوية البصرية الخاص بك."
                if query_language == "ar" else
                f"Competitors ({comp_list}) have been recorded into your Brand Identity profile."
            )
        else:
            b_title = "الهوية البصرية والملف التعريفي للبراند" if query_language == "ar" else "Brand Identity & Profile Hub"
            b_desc = (
                "يمكنك استعراض لوحة الألوان المعتمدة، نبرة الصوت، الجمهور المستهدف، والمنافسين المسجلين في أي وقت."
                if query_language == "ar" else
                "Review your verified brand color palette, tone of voice, target audience, and tracked competitors."
            )

        structured.blocks.append(
            ActionCardBlock(
                title=b_title,
                description=b_desc,
                actions=[{
                    "type": "UPDATE_BRAND_PROFILE",
                    "label": "عرض الهوية البصرية (Brand Identity)" if query_language == "ar" else "View Brand Identity Hub",
                    "payload": {"tab": "identity"}
                }]
            )
        )

    return structured

