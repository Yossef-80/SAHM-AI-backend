"""Conversation and long-task orchestration.

Routers call this; it holds the business logic. Keeps ``app/api/`` thin, as the
folder contract requires.

Responsibilities:
- persist conversations and messages
- run one assistant turn through the LangGraph orchestrator
- create/poll/cancel AgentTask records (queued|running|succeeded|failed|cancelled)
- stream a run's events
"""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.security import redact_sensitive_data
from app.core.errors import NotFoundError, ValidationFailed
from app.core.i18n import normalise_language, resolve_reply_language
from app.core.logging import get_logger
from app.core.orchestrator.blocks import compose_blocks, task_progress_block, validate_blocks
from app.db.models import (
    AgentTask,
    BrandMemoryItem,
    BrandProfile as BrandProfileRow,
    Business,
    Conversation,
    Message,
    Profile,
)
from app.schemas.assistant import (
    AssistantMessageRequest,
    AssistantMessageResponse,
    ConversationDetail,
    ConversationSummary,
    MessageContext,
)
from app.schemas.blocks import Message as MessageSchema
from app.schemas.common import StepStatus, TaskStatus

_logger = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


_IN_FLIGHT_NEW_CONVERSATIONS: dict[str, tuple[str, str, float]] = {}

_RETRY_MESSAGE = re.compile(
    r"\s*(?:please\s+)?(?:try\s+again|try\s+once\s+more|retry(?:\s+that)?|"
    r"run\s+it\s+again|do\s+it\s+again)\s*[.!?]*\s*",
    re.IGNORECASE,
)


def _resolve_retry_message(message: str, history: list[dict[str, Any]]) -> str:
    """Route a short retry command using the last substantive user request."""
    if not _RETRY_MESSAGE.fullmatch(message):
        return message
    previous = next(
        (
            str(item.get("content") or "").strip()
            for item in reversed(history)
            if item.get("role") == "user" and str(item.get("content") or "").strip()
        ),
        "",
    )
    if not previous:
        return message
    return f"Please retry the previous request: {previous}"


def _gate_campaign_action_cards(
    blocks: list[dict[str, Any]],
    *,
    user_message: str,
    conversation_context: dict[str, Any],
    user_context: str | None,
    business_budget: float | None = None,
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Server-side gate on CREATE_CAMPAIGN Action Cards.
    
    Strip any action_card of type CREATE_CAMPAIGN UNLESS ALL hold:
    (a) routed intent for THIS turn was strategy/campaign creation,
    (b) required prefill fields are actually known: budget, objective, audience (from conversation or brand facts),
    (c) the user has explicitly confirmed campaign creation in this conversation.
    
    If (b) fails but (a) holds: replaces the card with a question_with_options block.
    If (a) does not hold: strips the card completely.
    Logs every stripped card (agent/reason) for auditing.
    """
    gated_blocks: list[dict[str, Any]] = []
    
    # 1. Check (a): campaign creation intent in this turn
    campaign_keywords = [
        "create campaign", "launch campaign", "start campaign", "make campaign",
        "new campaign", "حملة اعلانية", "اعمل حملة", "انشئ حملة", "ابدأ حملة",
        "اطلق حملة", "عمل حملة", "حمله اعلانيه", "نموذج للحمله"
    ]
    has_campaign_intent = any(kw in user_message.lower() for kw in campaign_keywords)

    # 2. Check (c): explicit confirmation in this conversation
    explicit_confirmation_keywords = [
        "yes create", "confirm campaign", "launch the campaign", "go ahead and create",
        "ابدأ الحملة", "انشئ الحملة", "اطلق الحملة", "أكد انشاء الحملة", "موافق اعمل الحملة",
        "يلا نبدأ الحملة", "أكد", "confirm"
    ]
    is_explicit_confirmation_turn = any(kw in user_message.lower() for kw in explicit_confirmation_keywords)
    user_confirmed = bool(conversation_context.get("campaign_creation_confirmed", False) or is_explicit_confirmation_turn)
    if is_explicit_confirmation_turn:
        conversation_context["campaign_creation_confirmed"] = True

    # 3. Check (b): are budget, objective, audience known from conversation or brand facts?
    all_text = " ".join([
        user_message,
        user_context or "",
        " ".join(str(h.get("content") or "") for h in history),
    ]).lower()

    has_budget = bool(
        business_budget
        or any(w in all_text for w in ["جنيه", "egp", "le", "$", "usd", "budget", "ميزانية", "ميزانيه", "الف"])
    )
    has_objective = any(w in all_text for w in [
        "objective", "goal", "sales", "awareness", "leads", "conversions", "هدف", "مبيعات", "وعي", "رسائل", "زيارات"
    ])
    has_audience = any(w in all_text for w in [
        "audience", "target", "men", "women", "مستهدف", "جمهور", "شباب", "فئة", "عرسان", "موظفين"
    ])
    all_fields_known = has_budget and has_objective and has_audience

    for block in blocks:
        b_type = block.get("type")
        if b_type == "action_card":
            action_type = str(block.get("action_type") or "").lower()
            task = str(block.get("task") or "").lower()
            label = str(block.get("label") or "").lower()
            
            is_create_campaign = (
                action_type in ("create_campaign", "create-campaign")
                or task in ("create_campaign", "campaign_strategy")
                or "create campaign" in label
                or "انشاء حملة" in label
                or "إطلاق حملة" in label
            )

            if is_create_campaign:
                if not has_campaign_intent and not user_confirmed:
                    _logger.warning(
                        "create_campaign_card_stripped",
                        extra={"reason": "missing_campaign_intent", "user_message": user_message[:80]}
                    )
                    continue

                # Ensure prefill has sane defaults so user can immediately launch or review
                actions = block.get("actions") or []
                for act in actions:
                    if act.get("type") in ("CREATE_CAMPAIGN", "create_campaign"):
                        act_payload = act.get("payload") or {}
                        prefill = act_payload.get("prefill") or {}
                        if not prefill.get("dailyBudget"):
                            prefill["dailyBudget"] = business_budget or 500
                        if not prefill.get("currency"):
                            prefill["currency"] = "EGP"
                        if not prefill.get("objective"):
                            prefill["objective"] = "sales"
                        act_payload["prefill"] = prefill
                        act["payload"] = act_payload


        gated_blocks.append(block)

    return gated_blocks


# ------------------------------------------------------------- conversations --


async def get_or_create_conversation(
    db: AsyncSession, conversation_id: str | None, business_id: str
) -> Conversation:
    if conversation_id:
        existing = await db.get(Conversation, conversation_id)
        if existing is not None:
            if existing.business_id != business_id:
                raise NotFoundError(f"conversation '{conversation_id}' not found")
            return existing
        raise NotFoundError(f"conversation '{conversation_id}' not found")
    conversation = Conversation(
        id=_new_id("conv"),
        business_id=business_id,
        title="",
        context={},
    )
    db.add(conversation)
    await db.flush()
    return conversation


async def _resolve_answer_selection(
    db: AsyncSession,
    *,
    conversation_id: str,
    answer: AssistantAnswer,
) -> dict[str, Any]:
    msg = await db.get(Message, answer.message_id)
    if msg is None or msg.conversation_id != conversation_id:
        raise ValidationFailed("Invalid answer message.")
    blocks = msg.blocks or []
    question_block = next(
        (b for b in blocks if isinstance(b, dict) and b.get("type") == "question_with_options"),
        None,
    )
    if not question_block:
        raise ValidationFailed("No options block found in answered message.")

    answered_ids = set(question_block.get("answeredOptionIds") or [])
    for opt_id in answer.option_ids:
        if opt_id in answered_ids:
            raise ValidationFailed("Option has already been answered.")

    options = question_block.get("options") or []
    chosen = next((opt for opt in options if opt.get("id") in answer.option_ids), None)
    if not chosen:
        raise ValidationFailed("Selected option not found in question.")

    return chosen.get("payload") or {}


async def list_conversations(db: AsyncSession, business_id: str) -> list[ConversationSummary]:
    last_msg_subquery = (
        select(Message.content)
        .where(Message.conversation_id == Conversation.id)
        .order_by(Message.created_at.desc())
        .limit(1)
        .correlate(Conversation)
        .scalar_subquery()
    )

    msg_count_subquery = (
        select(func.count(Message.id))
        .where(Message.conversation_id == Conversation.id)
        .correlate(Conversation)
        .scalar_subquery()
    )

    stmt = (
        select(
            Conversation.id,
            Conversation.title,
            Conversation.created_at,
            Conversation.updated_at,
            msg_count_subquery.label("message_count"),
            last_msg_subquery.label("last_message_content"),
        )
        .where(Conversation.business_id == business_id)
        .order_by(Conversation.updated_at.desc())
    )
    rows = (await db.execute(stmt)).all()
    out: list[ConversationSummary] = []
    for row in rows:
        conv_id, title, created_at, updated_at, count, last_content = row
        preview = (last_content or "")[:120] if last_content else ""
        out.append(
            ConversationSummary(
                id=conv_id,
                title=title,
                created_at=created_at,
                updated_at=updated_at,
                message_count=count or 0,
                last_message_preview=preview,
            )
        )
    return out


def _message_schema(row: Message) -> MessageSchema:
    """Serialize a persisted message without changing its database identity."""
    return MessageSchema(
        id=row.id,
        role=row.role,
        content=row.content or "",
        blocks=[block for block in (row.blocks or [])],
        created_at=row.created_at,
        task_id=row.task_id,
        persona_id=row.persona_id,
    )


async def get_conversation(
    db: AsyncSession, conversation_id: str, *, business_id: str | None = None
) -> ConversationDetail:
    """One conversation.

    ``business_id`` scopes the read to one workspace. A conversation owned by
    another workspace raises ``NotFoundError`` -- 404, never 403, because a 403
    would confirm the id exists to a stranger.
    """
    stmt = select(Conversation).where(Conversation.id == conversation_id)
    if business_id is not None:
        stmt = stmt.where(Conversation.business_id == business_id)
    conversation = await db.scalar(stmt)
    if conversation is None:
        raise NotFoundError(f"conversation '{conversation_id}' not found")
    def _msg_sort_key(m: Message) -> tuple[float, int]:
        t = m.created_at
        if t is None:
            return (0.0, 0 if m.role == "user" else 1)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (t.timestamp(), 0 if m.role == "user" else 1)

    sorted_messages = sorted(
        conversation.messages,
        key=_msg_sort_key,
    )
    messages = [_message_schema(row) for row in sorted_messages]
    ctx = conversation.context or {}
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=messages,
        context=MessageContext(**{k: v for k, v in ctx.items()
                                  if k in MessageContext.model_fields}),
    )


async def _persist_message(
    db: AsyncSession,
    *,
    conversation_id: str,
    role: str,
    content: str,
    blocks: list[dict[str, Any]],
    task_id: str | None = None,
    persona_id: str | None = None,
    created_at: datetime | None = None,
) -> Message:
    message = Message(
        id=_new_id("msg"),
        conversation_id=conversation_id,
        role=role,
        content=content,
        blocks=blocks,
        task_id=task_id,
        persona_id=persona_id,
        created_at=created_at or _now(),
    )
    db.add(message)
    await db.flush()
    return message


# ---------------------------------------------------------------- the turn ----


async def handle_message(
    db: AsyncSession,
    request: AssistantMessageRequest,
    *,
    llm: Any,
    fast_llm: Any | None = None,
    request_id: str,
    business: Business,
    event_queue: asyncio.Queue[dict[str, Any]] | None = None,
    defer_post_memory: bool = False,
) -> AssistantMessageResponse:
    """Run one assistant turn synchronously.

    ``business`` is the already-resolved workspace row from
    ``WorkspaceDep``; this layer never picks a tenant for itself.
    """
    # Server-side debounce to prevent duplicate conversation creation from rapid client double-submits
    if not request.conversation_id:
        b_key = business.id
        now_ts = _now().timestamp()
        cached = _IN_FLIGHT_NEW_CONVERSATIONS.get(b_key)
        conversation = None
        if cached and cached[0] == request.message and (now_ts - cached[2] < 2.5):
            try:
                conversation = await get_or_create_conversation(db, cached[1], business.id)
                _logger.info(
                    "duplicate_new_conversation_deduplicated",
                    extra={"business_id": business.id, "reused_conversation_id": cached[1]},
                )
            except NotFoundError:
                conversation = None
        if conversation is None:
            conversation = await get_or_create_conversation(db, None, business.id)
            _IN_FLIGHT_NEW_CONVERSATIONS[b_key] = (request.message, conversation.id, now_ts)
    else:
        conversation = await get_or_create_conversation(
            db, request.conversation_id, business.id
        )

    if event_queue is not None:
        await event_queue.put(
            {"type": "conversation_started", "conversationId": conversation.id}
        )

    # Merge sticky context. Reference identity and workflow metadata are persisted;
    # image bytes and the full VLM payload are deliberately loaded only for this turn.
    context = dict(redact_sensitive_data(conversation.context or {}))
    if request.context:
        supplied_context = redact_sensitive_data(request.context.model_dump())
        context.update({k: v for k, v in supplied_context.items() if v is not None})
    if request.persona_id:
        context["persona_id"] = request.persona_id
    # --- reply language, decided once per turn -------------------------------
    # The UI sends its current language on every message, so switching the
    # language mid-conversation switches the very next answer. Only *explicit*
    # choices are persisted: remembering a language we merely inferred from one
    # message would trap a user who pasted a single line of the other script.
    explicit_language = normalise_language(request.language)
    if explicit_language:
        context["language"] = explicit_language
    reply_language = resolve_reply_language(
        requested=request.language,
        conversation=context.get("language"),
        project=getattr(business, "language", None),
        prompt=request.message,
    )
    # Attachments live for exactly one turn. Pulled out *before* the context is
    # persisted, otherwise the same photo would be re-sent to the model on every
    # later turn of the conversation and quietly multiply the token bill.
    turn_attachment_ids = [
        str(item) for item in (context.pop("attachment_ids", None) or []) if str(item).strip()
    ]
    # Assign a *copy*. A plain JSON column is not mutation-tracked, so handing the
    # row the working dict makes that same object the ORM's committed-state
    # baseline; every later in-place mutation silently mutates the baseline too,
    # so the re-assignment after orchestrate() compares equal and emits no UPDATE.
    # That is why creative_prompt_stage (and persona_id) never reached the DB and
    # each turn restarted at `collecting_details`.
    conversation.context = dict(context)
    safe_message = str(redact_sensitive_data(request.message))

    # Query conversation messages chronologically. The AI layer owns workflow
    # state transitions; this bridge only validates answer IDs against persisted UI blocks.
    msg_rows = (
        await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.asc(), desc(Message.role))
        )
    ).scalars().all()

    validated_prompt_answer: dict[str, Any] | None = None
    ad_flow_selection: dict[str, Any] | None = None
    if request.answer:
        selected_payload = await _resolve_answer_selection(
            db, conversation_id=conversation.id, answer=request.answer
        )
        if isinstance(selected_payload, dict) and selected_payload.get("kind") == "meta_ad_preview":
            ad_flow_selection = selected_payload
        if context.get("workflow") == "creative_prompt":
            validated_prompt_answer = {
                "message_id": str(request.answer.message_id),
                "option_ids": list(request.answer.option_ids),
            }

    # Manage session memory with configurable token threshold:
    # Below threshold: passes raw past questions and answers like a normal chat agent.
    # When threshold is reached: summarizes past turns into conversation.summary in DB,
    # and only passes the summary + latest turns for this and subsequent turns.
    from app.ai.memory.compressor import manage_session_memory_with_token_cap
    history, conv_summary = await manage_session_memory_with_token_cap(
        db,
        conversation=conversation,
        messages=list(msg_rows),
    )

    # A bare "try again" is an instruction about the preceding user request.
    # Routing only that short phrase sent it to normal chat, where the model
    # could claim it was starting research without actually invoking a task.
    routed_message = _resolve_retry_message(safe_message, history)
    if context.get("competitor_name") and str(context["competitor_name"]).lower() not in routed_message.lower():
        routed_message = f"[Target Competitor: {context['competitor_name']}] {routed_message}"

    user_now = _now()
    user_row = await _persist_message(
        db,
        conversation_id=conversation.id,
        role="user",
        content=safe_message,
        blocks=[{"type": "text", "text": safe_message}],
        created_at=user_now,
    )
    if not conversation.title:
        conversation.title = safe_message[:80]

    # Load Long-Term Memory (Profile.user_context_summary - FACTS ONLY)
    profile_row = await db.scalar(
        select(Profile).where(Profile.business_id == business.id).limit(1)
    )
    user_context = str(redact_sensitive_data(profile_row.user_context_summary)) if profile_row and profile_row.user_context_summary else None

    # The Settings form writes the canonical profile to Business/BrandProfile,
    # while the optional AI long-term store may lag or be empty. Read the saved
    # workspace profile here and pass it ephemerally to app.ai on every turn.
    brand_profile_row = await db.scalar(
        select(BrandProfileRow).where(BrandProfileRow.business_id == business.id).limit(1)
    )
    business_profile_context = {
        "name": str(business.name or "")[:200],
        "industry": str(business.industry or "")[:120],
        "website_url": str(business.website_url or "")[:500],
        "country": str(business.country or "")[:2],
        "language": str(business.language or "")[:8],
        "currency": str(business.currency or "")[:8],
        "monthly_budget": business.monthly_budget,
        "brand_voice": str(brand_profile_row.brand_voice or "")[:2000] if brand_profile_row else "",
        "visual_style": str(brand_profile_row.visual_style or "")[:2000] if brand_profile_row else "",
        "target_audience": str(brand_profile_row.target_audience or "")[:2000] if brand_profile_row else "",
        "key_offers": [str(item)[:200] for item in (brand_profile_row.key_offers or [])[:20]] if brand_profile_row else [],
        "goals": [str(item)[:200] for item in (brand_profile_row.goals or [])[:20]] if brand_profile_row else [],
        "competitors": [str(item)[:200] for item in (brand_profile_row.competitors or [])[:20]] if brand_profile_row else [],
    }

    # Rehydrate a workspace-scoped reference analysis for Layla. The database
    # read lives in the conversation bridge; interpretation stays in app/ai.
    ai_context = dict(context)
    ai_context["latest_user_message"] = safe_message
    # The resolved language for *this* turn (not the remembered one) plus the
    # workspace's configured project language, so the AI layer never has to
    # re-derive either from the text.
    ai_context["language"] = reply_language
    ai_context["project_language"] = str(getattr(business, "language", "") or "")[:8]
    ai_context["business_profile_context"] = redact_sensitive_data(business_profile_context)
    if (
        context.get("workflow") == "creative_prompt"
        and not context.get("creative_prompt_user_facts")
    ):
        evidence_history: list[dict[str, str]] = []
        evidence_chars = 0
        for row in reversed(msg_rows):
            if row.role not in {"user", "assistant"} or not row.content:
                continue
            content = str(redact_sensitive_data(row.content))[:2000]
            if evidence_history and evidence_chars + len(content) > 12000:
                break
            evidence_history.append({"role": row.role, "content": content})
            evidence_chars += len(content)
            if len(evidence_history) >= 24:
                break
        ai_context["creative_prompt_evidence_history"] = list(reversed(evidence_history))
    if validated_prompt_answer:
        ai_context["creative_prompt_answer"] = validated_prompt_answer
    reference_id = str(context.get("reference_analysis_id") or "").strip()
    if reference_id:
        reference = await db.get(BrandMemoryItem, reference_id)
        if (
            reference is None
            or reference.business_id != business.id
            or reference.kind != "reference_analysis"
        ):
            raise NotFoundError("creative reference not found")
        reference_metadata = reference.metadata_ or {}
        style = reference_metadata.get("style")
        if not isinstance(style, dict):
            raise NotFoundError("creative reference analysis is incomplete")
        ai_context["reference_analysis_context"] = {
            "source_name": str(reference_metadata.get("source_name") or "Saved reference")[:160],
            "summary": str(reference_metadata.get("summary") or "")[:2000],
            "style": style,
        }

    # A Research-page result the user chose to discuss. Research runs already
    # store their full report in brand memory (`app/api/research.py`, kind_map),
    # but until now nothing could read it back: the UI sent a memory id as
    # `competitor_id` and no code path consumed that field, so the model only
    # ever saw the competitor's *name* and re-ran the research. Loading the row
    # here turns the chip into real context.
    research_id = str(context.get("research_item_id") or "").strip()
    if research_id:
        research_item = await db.get(BrandMemoryItem, research_id)
        if research_item is None or research_item.business_id != business.id:
            raise NotFoundError("research item not found")
        research_meta = research_item.metadata_ or {}
        research_details = research_meta.get("details")
        ai_context["research_item_context"] = {
            "id": research_item.id,
            "kind": research_item.kind,
            "title": str(research_meta.get("title") or "Research result")[:200],
            "query": str(research_meta.get("query") or "")[:300],
            "tool": str(research_meta.get("tool") or "")[:80],
            "report": research_item.text[:6000],
            "details": research_details if isinstance(research_details, dict) else {},
        }

    # Images attached to *this* message. Loaded here (bytes never live in the
    # conversation context) so the AI layer can show them to a vision-capable
    # chat model without the creative-reference workflow being involved.
    if turn_attachment_ids:
        from app.core.chat_attachments import load_chat_attachments

        attachments = await load_chat_attachments(
            db, business_id=business.id, ids=turn_attachment_ids
        )
        if attachments:
            ai_context["chat_attachments"] = attachments

    # Load Active Plans separately from BrandMemoryItem (PLANS ONLY)
    plan_item = await db.scalar(
        select(BrandMemoryItem.text)
        .where(
            BrandMemoryItem.business_id == business.id,
            BrandMemoryItem.kind == "strategy_blueprint",
        )
        .limit(1)
    )
    active_plans = plan_item if plan_item else None

    def _ad_tool_context_factory():
        from app.core.tools.base import ToolContext
        return ToolContext(
            workspace_id=business.workspace_id or "",
            business_id=business.id,
            db=db,
            llm=llm,
            fast_llm=fast_llm or llm,
            request_id=request_id or "ai-chat",
            conversation_id=conversation.id,
            persona_id="researcher",
            campaign_id=(str(context.get("campaign_id") or "") or None) if context else None,
            creative_id=(str(context.get("creative_id") or "") or None) if context else None,
            insight_id=(str(context.get("insight_id") or "") or None) if context else None,
            user_supplied=dict(context or {}),
        )

    # LANGGRAPH UI BLOCKS OVERRIDE
    from app.ai.agents.orchestrator.brain import orchestrate
    from app.core.ai_usage import usage_action

    with usage_action(
        "chat.turn", persona_id=str(request.persona_id or context.get("persona_id") or "") or None
    ):
        final_gui = await orchestrate(
            routed_message,
            history,
            event_queue,
            user_context=user_context,
            conversation_summary=conv_summary,
            active_plans=active_plans,
            business_id=business.id,
            workspace_id=business.workspace_id or "",
            conversation_id=conversation.id,
            request_id=request_id,
            db=db,
            core_llm=llm,
            fast_llm=fast_llm,
            context=ai_context,
            forced_agent=request.persona_id or context.get("persona_id"),
            language=reply_language,
            ad_flow_selection=ad_flow_selection,
            ad_tool_context_factory=_ad_tool_context_factory,
        )
    next_prompt_stage = getattr(final_gui, "creative_prompt_stage", None) if final_gui else None
    if final_gui and getattr(final_gui, "creative_prompt_workflow", False):
        context["workflow"] = "creative_prompt"
        prompt_type = getattr(final_gui, "creative_prompt_type", None)
        if prompt_type in {"image", "video"}:
            context["prompt_type"] = prompt_type
        context["persona_id"] = "layla"
    else:
        if context.get("workflow") == "creative_prompt":
            context.pop("workflow", None)
            context.pop("creative_prompt_stage", None)
            context.pop("creative_prompt_reference_shown_key", None)
            context.pop("creative_prompt_user_facts", None)
            context.pop("creative_prompt_confirmed_facts", None)
            context.pop("creative_prompt_unresolved_fields", None)
            conversation.context = dict(context)
    if context.get("workflow") == "creative_prompt":
        if next_prompt_stage:
            context["creative_prompt_stage"] = next_prompt_stage
        reference_shown_key = (
            getattr(final_gui, "creative_prompt_reference_shown_key", None)
            if final_gui
            else None
        )
        if reference_shown_key:
            context["creative_prompt_reference_shown_key"] = str(reference_shown_key)[:120]
        user_facts = getattr(final_gui, "creative_prompt_user_facts", None) if final_gui else None
        if isinstance(user_facts, list):
            safe_facts = redact_sensitive_data(user_facts)
            context["creative_prompt_user_facts"] = [
                str(item)[:4000] for item in safe_facts[-16:] if str(item).strip()
            ]
        semantic_facts = (
            getattr(final_gui, "creative_prompt_confirmed_facts", None)
            if final_gui
            else None
        )
        if isinstance(semantic_facts, dict):
            context["creative_prompt_confirmed_facts"] = redact_sensitive_data(
                semantic_facts
            )
        unresolved_fields = (
            getattr(final_gui, "creative_prompt_unresolved_fields", None)
            if final_gui
            else None
        )
        if isinstance(unresolved_fields, list):
            context["creative_prompt_unresolved_fields"] = [
                str(field)[:40] for field in unresolved_fields[:16]
            ]
        # `context` is the *same* dict object already assigned to the row above
        # (line ~229). A plain JSON column is not mutation-tracked, so re-assigning
        # the identical object records no change and the unit of work skips the
        # UPDATE -- every key added here (creative_prompt_stage above all) was
        # silently lost, and the next turn restarted at `collecting_details`,
        # which re-asked answered questions and re-presented an approved plan.
        conversation.context = dict(context)
    blocks_raw = final_gui.model_dump(mode="json", by_alias=True)["blocks"] if final_gui else []
    blocks_raw = redact_sensitive_data(blocks_raw)
    final = {"blocks": blocks_raw}
    # END OVERRIDE
    raw_blocks = validate_blocks(list(final.get("blocks") or []))
    
    # Server-side gate on CREATE_CAMPAIGN action cards
    blocks = _gate_campaign_action_cards(
        raw_blocks,
        user_message=request.message,
        conversation_context=context,
        user_context=user_context,
        business_budget=business.monthly_budget,
        history=history,
    )
    persona_id = getattr(final_gui, "persona_id", None) if final_gui else None
    if not persona_id:
        agent_result = final.get("agent_result") or {}
        persona_id = str(agent_result.get("persona_id") or "") or None

    from app.schemas.blocks import blocks_to_plaintext

    content = blocks_to_plaintext(blocks)
    if not content or content == "[]":
        content = str(final.get("error") or "")

    from datetime import timedelta
    asst_now = _now()
    if asst_now <= user_now:
        asst_now = user_now + timedelta(milliseconds=100)

    assistant_row = await _persist_message(
        db,
        conversation_id=conversation.id,
        role="assistant",
        content=content,
        blocks=blocks,
        task_id=final.get("task_id"),
        persona_id=persona_id,
        created_at=asst_now,
    )
    # Return the same durable IDs the conversation endpoint will return.
    user_message = _message_schema(user_row)
    assistant_message = _message_schema(assistant_row)
    if request.answer is not None:
        ans_msg = await db.get(Message, request.answer.message_id)
        if ans_msg and ans_msg.blocks:
            up_blocks = [dict(b) if isinstance(b, dict) else b for b in ans_msg.blocks]
            for ub in up_blocks:
                if isinstance(ub, dict) and ub.get("type") == "question_with_options":
                    ub["answeredOptionIds"] = list(request.answer.option_ids)
            ans_msg.blocks = up_blocks
    await db.commit()

    if not defer_post_memory:
        # The single per-workspace memory pass (session compaction, then the
        # debounced LTM rewrite, then the rate-limited identity sync).
        try:
            from app.ai.memory.consolidator import memory_consolidator

            memory_consolidator.schedule(
                business_id=business.id,
                conversation_id=conversation.id,
            )
        except Exception as exc:
            _logger.warning(
                "memory_consolidation_schedule_failed", extra={"error": str(exc)}
            )

    try:
        from app.ai.observability import flush_observability

        flush_observability()
    except Exception:
        pass

    return AssistantMessageResponse(
        message=assistant_message,
        user_message=user_message,
        task_id=final.get("task_id"),
        blocks=blocks,
        conversation_id=conversation.id,
    )


# -------------------------------------------------------------- agent tasks ----


async def create_task(
    db: AsyncSession,
    *,
    task_name: str,
    persona_id: str,
    conversation_id: str | None,
    business_id: str,
    inputs: dict[str, Any],
    request_id: str,
) -> AgentTask:
    task = AgentTask(
        id=_new_id("task"),
        business_id=business_id,
        conversation_id=conversation_id,
        task_name=task_name,
        persona_id=persona_id,
        status=TaskStatus.QUEUED.value,
        steps=[
            {"name": "queued", "status": StepStatus.PENDING.value, "detail": "waiting to start"}
        ],
        inputs=inputs,
        request_id=request_id,
    )
    db.add(task)
    await db.flush()
    if conversation_id:
        progress = task_progress_block(
            task_id=task.id,
            task_name=task.task_name,
            persona_id=task.persona_id,
            status=task.status,
            steps=task.steps,
            progress=task.progress,
        )
        await _persist_message(
            db,
            conversation_id=conversation_id,
            role="assistant",
            content=f"{persona_id.title()} started {task_name.replace('_', ' ')}.",
            blocks=validate_blocks([progress]),
            task_id=task.id,
            persona_id=persona_id,
        )
        conversation = await db.get(Conversation, conversation_id)
        if conversation is not None:
            conversation.updated_at = _now()
    return task


async def get_task(
    db: AsyncSession, task_id: str, *, business_id: str | None = None
) -> AgentTask:
    """One task, optionally scoped to one workspace (404 across the boundary)."""
    stmt = select(AgentTask).where(AgentTask.id == task_id)
    if business_id is not None:
        stmt = stmt.where(AgentTask.business_id == business_id)
    task = await db.scalar(stmt)
    if task is None:
        raise NotFoundError(f"task '{task_id}' not found")
    return task


async def cancel_task(
    db: AsyncSession, task_id: str, *, business_id: str | None = None
) -> AgentTask:
    task = await get_task(db, task_id, business_id=business_id)
    if task.status in (TaskStatus.SUCCEEDED.value, TaskStatus.FAILED.value,
                       TaskStatus.CANCELLED.value):
        # Already terminal: cancelling is a no-op, not an error.
        return task
    task.status = TaskStatus.CANCELLED.value
    task.finished_at = _now()
    task.error = task.error or "cancelled by user"
    await db.commit()
    return task


async def run_task_in_background(
    task_id: str,
    *,
    db_factory: Any,
    llm: Any,
    persona_id: str,
    task_name: str,
    inputs: dict[str, Any],
    user_message: str,
    request_id: str,
) -> None:
    """Execute a queued AgentTask. Called from a background asyncio task."""
    async with db_factory() as db:
        task = await db.get(AgentTask, task_id)
        if task is None or task.status == TaskStatus.CANCELLED.value:
            return
        task.status = TaskStatus.RUNNING.value
        task.started_at = _now()
        task.steps = [{"name": "running", "status": StepStatus.RUNNING.value}]
        await db.flush()
        await db.commit()

        try:
            # Legacy queued Layla writing tasks must not fall back to the core
            # task runner/LLM. New requests use /creatives/generate-text in
            # app.ai; fail any pre-deployment queued task closed with guidance.
            direct_layla_tasks = {
                "creative_brief",
                "commercial_script",
                "image_prompt",
                "video_prompt",
                "ad_copy",
                "ad_variations",
            }
            if persona_id.strip().lower() == "layla" and task_name in direct_layla_tasks:
                from app.core.errors import TaskNotAllowedError

                raise TaskNotAllowedError(
                    "Legacy Layla text tasks are disabled; use /creatives/generate-text."
                )

            from app.core.agents.persona import load_persona
            from app.core.agents.runtime import run_agent
            from app.core.tools.base import ToolContext

            # The task row carries a business, not a workspace, so the
            # workspace is read back off it rather than assumed. A task with no
            # business cannot be attributed to a tenant and runs with an empty
            # workspace id.
            business = await db.get(Business, task.business_id) if task.business_id else None
            workspace_id = business.workspace_id or "" if business is not None else ""
            persona = await load_persona(persona_id, db, workspace_id)
            # The language this run writes in. Same policy as the chat: an
            # explicit per-task locale (the creative dialog sends one) wins, then
            # the workspace's project language, then the message itself. Passed to
            # the runtime as an override so no persona row is mutated and the
            # normal assistant's configured language is untouched.
            from app.core.i18n import resolve_reply_language

            task_language = resolve_reply_language(
                requested=inputs.get("language"),
                project=getattr(business, "language", None) if business else None,
                prompt=user_message or "",
            )
            ctx = ToolContext(
                workspace_id=workspace_id,
                business_id=task.business_id or "",
                db=db,
                llm=llm,
                request_id=request_id,
                task_id=task.id,
                persona_id=persona.id,
                conversation_id=task.conversation_id,
                campaign_id=(str(inputs.get("campaign_id") or inputs.get("campaignId") or "") or None),
                creative_id=(str(inputs.get("creative_id") or inputs.get("creativeId") or "") or None),
                insight_id=(str(inputs.get("insight_id") or inputs.get("insightId") or "") or None),
            )
            result = await run_agent(
                persona,
                task_name,
                inputs,
                ctx,
                llm=llm,
                user_message=user_message,
                language=task_language,
            )
            task.status = (
                TaskStatus.SUCCEEDED.value
                if not result.error and not result.incomplete
                else TaskStatus.FAILED.value
            )
            task.result = result.model_dump(mode="json")
            task.incomplete = result.incomplete
            task.error = result.error or result.incomplete_reason
            task.tokens_in = result.usage.tokens_in
            task.tokens_out = result.usage.tokens_out
            task.model = result.usage.model
            task.blocks = compose_blocks(
                result.model_dump(mode="json"),
                persona_id=persona.id,
                task_name=task_name,
                task_id=task.id,
            )
            tool_steps = [
                {
                    "name": call.tool,
                    "status": (
                        StepStatus.FAILED.value
                        if call.status in {"error", "blocked"}
                        else StepStatus.SUCCEEDED.value
                    ),
                    "detail": call.output_summary[:500],
                }
                for call in result.tool_calls_made
            ]
            task.progress = 1.0
            task.finished_at = _now()
            if task.error:
                task.steps = tool_steps + [
                    {
                        "name": "complete",
                        "status": StepStatus.FAILED.value,
                        "detail": task.error[:500],
                    }
                ]
            else:
                task.steps = tool_steps + [
                    {
                        "name": "complete",
                        "status": StepStatus.SUCCEEDED.value,
                        "detail": "Result saved.",
                    }
                ]
            await db.commit()
        except Exception as exc:
            _logger.warning("background_task_failed", extra={"task_id": task_id, "error": str(exc)})
            task.status = TaskStatus.FAILED.value
            task.error = str(exc)
            task.progress = 1.0
            task.steps = [
                {"name": "complete", "status": StepStatus.FAILED.value, "detail": task.error[:500]}
            ]
            task.blocks = validate_blocks(
                [{"type": "error", "message": "The task failed before a result could be saved.", "retryable": True}]
            )
            task.finished_at = _now()
            await db.commit()


async def stream_task(task: AgentTask) -> Any:
    """Yield SSE-formatted events for an already-authorised running task.

    Takes the row, not an id: the caller loads it with the workspace filter
    *before* the response starts, because an ownership check raised from inside
    a streaming body happens after the headers are sent and can no longer be
    turned into a 404.
    """
    yield _sse("task_progress", {"task_id": task.id, "status": task.status,
                                 "progress": task.progress})
    yield _sse("done", {"task_id": task.id, "status": task.status})


def _sse(event: str, data: dict[str, Any]) -> str:
    import json

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


__all__ = [
    "cancel_task",
    "create_task",
    "get_conversation",
    "get_task",
    "handle_message",
    "list_conversations",
    "run_task_in_background",
    "stream_task",
]
