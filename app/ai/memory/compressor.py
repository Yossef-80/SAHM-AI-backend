from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Profile, Conversation, Message, BrandMemoryItem
from app.db.session import session_scope
from app.ai.providers.llm import get_llm
from app.ai.security import redact_sensitive_data
from app.core.logging import get_logger
from app.schemas.blocks import blocks_to_plaintext

logger = get_logger(__name__)


def _message_to_text(msg: Any) -> str:
    """Canonical text representation of a message row or history dict."""
    blocks = getattr(msg, "blocks", None) if hasattr(msg, "blocks") else (msg.get("blocks") if isinstance(msg, dict) else None)
    if blocks:
        txt = blocks_to_plaintext(blocks)
        if txt.strip():
            return txt
    content = getattr(msg, "content", None) if hasattr(msg, "content") else (msg.get("content") if isinstance(msg, dict) else None)
    return str(content or "")



def count_tokens(text: str) -> int:
    """Measure real tokens using tiktoken (cl100k_base).
    
    If tiktoken is unavailable, uses a language-aware approximation:
    - Arabic characters (\u0600-\u06FF): ~1.8 characters per token.
    - Other/Latin characters: ~3.5 characters per token.
    Never uses a flat chars/3.5 across Arabic text.
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        pass
    
    # Language-aware fallback
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    other_chars = len(text) - arabic_chars
    est_tokens = int((arabic_chars / 1.8) + (other_chars / 3.5))
    return max(1, est_tokens)


def _truncate_to_boundary(text: str, max_tokens: int) -> str:
    """Truncate text at a bullet or sentence boundary without breaking mid-sentence."""
    if not text:
        return ""
    lines = text.split("\n")
    kept_lines: list[str] = []
    
    for line in lines:
        candidate = "\n".join(kept_lines + [line]) if kept_lines else line
        if count_tokens(candidate) > max_tokens:
            if not kept_lines:
                # If even the first line is over, split by sentences (. , ؟ , !)
                sentences = re.split(r"(?<=[.!?؟])\s+", line)
                kept_sents: list[str] = []
                for s in sentences:
                    cand_s = " ".join(kept_sents + [s]) if kept_sents else s
                    if count_tokens(cand_s) <= max_tokens:
                        kept_sents.append(s)
                    else:
                        break
                if kept_sents:
                    return " ".join(kept_sents)
                # Word-level fallback if a single sentence exceeds cap
                words = line.split()
                kept_words: list[str] = []
                for w in words:
                    cand_w = " ".join(kept_words + [w]) if kept_words else w
                    if count_tokens(cand_w) <= max_tokens:
                        kept_words.append(w)
                    else:
                        break
                return " ".join(kept_words)
            break
        kept_lines.append(line)
        
    return "\n".join(kept_lines)


async def enforce_token_cap(
    text: str,
    max_tokens: int,
    *,
    context_label: str = "text",
    retry_prompt_func: Any | None = None,
) -> str:
    """Enforce hard token cap on summarized or distilled text.
    
    1. Measures token count with real tokenizer.
    2. If over cap, logs a WARNING.
    3. Retries once with a stricter instruction containing the measured size (if retry_prompt_func supplied).
    4. If still over cap, hard-truncates at sentence/bullet boundary.
    """
    if not text:
        return ""
    measured = count_tokens(text)
    if measured <= max_tokens:
        return text

    logger.warning(
        "token_cap_overflow_detected",
        extra={
            "context": context_label,
            "measured_tokens": measured,
            "max_tokens": max_tokens,
            "overage": measured - max_tokens,
        },
    )

    # 1. Retry once if retry callback is provided
    if retry_prompt_func is not None:
        try:
            retried_text = await retry_prompt_func(measured, max_tokens, text)
            if retried_text:
                retried_tokens = count_tokens(retried_text)
                if retried_tokens <= max_tokens:
                    return retried_text
                measured = retried_tokens
                text = retried_text
        except Exception as exc:
            logger.warning(
                "token_cap_retry_failed",
                extra={"context": context_label, "error": str(exc)},
            )

    # 2. Hard truncate at bullet or sentence boundary
    truncated = _truncate_to_boundary(text, max_tokens)
    final_tokens = count_tokens(truncated)
    logger.warning(
        "token_cap_hard_truncated",
        extra={
            "context": context_label,
            "initial_measured": measured,
            "final_tokens": final_tokens,
            "max_tokens": max_tokens,
        },
    )
    return truncated


def resplit_summary_text(summary_text: str) -> tuple[str, str]:
    """Split an existing combined summary into BRAND FACTS vs ACTIVE PLANS without LLM.
    
    Parses bullet points and headers to partition descriptive facts away from
    marketing goals, campaign pitches, and budgets.
    """
    if not summary_text:
        return "", ""

    plan_keywords = [
        "goal", "goals", "budget", "budgets", "active marketing", "marketing goal",
        "ad angle", "ad angles", "campaign", "campaigns", "channel strateg",
        "خطة", "أهداف", "حملة", "حملات", "ميزانية", "زوايا إعلانية", "استراتيجية الحملة"
    ]

    facts_lines: list[str] = []
    plans_lines: list[str] = []

    lines = summary_text.split("\n")
    current_target = facts_lines

    for line in lines:
        lower = line.lower()
        # Header or bullet classification
        if any(kw in lower for kw in plan_keywords) and any(marker in line for marker in ["**", "##", "###", ":", "-"]):
            current_target = plans_lines
        elif any(kw in lower for kw in ["brand name", "industry", "sector", "audience", "competitor", "اسم البراند", "المجال", "الجمهور", "المنافسين"]):
            current_target = facts_lines

        current_target.append(line)

    facts = "\n".join(facts_lines).strip()
    plans = "\n".join(plans_lines).strip()
    return facts, plans


async def resplit_existing_profile(
    business_id: str,
    db: AsyncSession,
    llm: Any | None = None,
) -> tuple[str, str]:
    """Migration helper: re-splits an existing over-long user_context_summary into FACTS and PLANS.
    
    Stores FACTS in Profile.user_context_summary (capped <= 300 tokens).
    Stores PLANS in BrandMemoryItem of kind="strategy_blueprint".
    """
    from app.ai.settings import get_ai_settings
    ai_settings = get_ai_settings()
    max_profile_tokens = ai_settings.memory_global_profile_max_tokens

    stmt = select(Profile).where(Profile.business_id == business_id).limit(1)
    profile = await db.scalar(stmt)
    if not profile or not profile.user_context_summary:
        return "", ""

    raw_summary = profile.user_context_summary.strip()
    
    # Check if LLM re-split is possible
    if llm:
        sys_msg = SystemMessage(content=(
            "You are a database repair and memory separation assistant for SAHM Marketing OS.\n"
            "Separate the provided long-term memory into TWO DISTINCT ARTIFACTS:\n\n"
            "=== BRAND FACTS ===\n"
            "- Extract ONLY permanent descriptive facts: Brand Name, Industry/Sector, Geography, Target Audience, Pricing Policy, Known Competitors.\n"
            f"- STRICT CONSTRAINT: MAXIMUM {max_profile_tokens} TOKENS. Descriptive only.\n\n"
            "=== ACTIVE PLANS ===\n"
            "- Extract marketing goals, budgets, ad angles, channel strategies, and campaign learnings.\n"
        ))
        usr_msg = HumanMessage(content=f"Original Profile Memory:\n{raw_summary}\n\nSplit into === BRAND FACTS === and === ACTIVE PLANS ===")
        try:
            resp = await llm.ainvoke([sys_msg, usr_msg])
            content = resp.content if isinstance(resp.content, str) else "\n".join(str(p) for p in resp.content)
            facts_part = ""
            plans_part = ""
            if "=== BRAND FACTS ===" in content:
                parts = content.split("=== ACTIVE PLANS ===")
                facts_part = parts[0].replace("=== BRAND FACTS ===", "").strip()
                if len(parts) > 1:
                    plans_part = parts[1].strip()
            else:
                facts_part, plans_part = resplit_summary_text(raw_summary)
        except Exception:
            facts_part, plans_part = resplit_summary_text(raw_summary)
    else:
        facts_part, plans_part = resplit_summary_text(raw_summary)

    # Enforce hard token cap on FACTS
    facts_part = await enforce_token_cap(
        facts_part, max_profile_tokens, context_label=f"resplit_profile_facts_{business_id}"
    )

    profile.user_context_summary = facts_part

    # Persist plans in BrandMemoryItem(kind="strategy_blueprint")
    if plans_part:
        item_stmt = (
            select(BrandMemoryItem)
            .where(
                BrandMemoryItem.business_id == business_id,
                BrandMemoryItem.kind == "strategy_blueprint",
            )
            .limit(1)
        )
        plan_item = await db.scalar(item_stmt)
        now = datetime.now(timezone.utc)
        if plan_item:
            plan_item.text = plans_part
            plan_item.metadata_ = {
                **(plan_item.metadata_ or {}),
                "active_plans": plans_part,
                "updated_at": now.isoformat(),
            }
        else:
            new_item = BrandMemoryItem(
                id=f"mem_plan_{uuid.uuid4().hex[:12]}",
                business_id=business_id,
                kind="strategy_blueprint",
                text=plans_part,
                metadata_={
                    "active_plans": plans_part,
                    "source": "ltm_resplit_migration",
                    "created_at": now.isoformat(),
                },
                created_at=now,
            )
            db.add(new_item)

    await db.commit()
    logger.info(
        "profile_resplit_completed",
        extra={
            "business_id": business_id,
            "facts_tokens": count_tokens(facts_part),
            "plans_tokens": count_tokens(plans_part),
        },
    )
    return facts_part, plans_part


def compress_history_messages(history: list[dict[str, Any]], max_chars: int = 900) -> list[dict[str, Any]]:
    """
    Intelligent Session Memory:
    Compresses long assistant responses (>max_chars) in the working history
    so subsequent turns don't blow up token budgets.
    """
    compact_history = []
    for item in history:
        role = item.get("role")
        content = _message_to_text(item)
        
        # If assistant generated a massive response, distill it
        if role == "assistant" and len(content) > max_chars:
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            # Extract high-signal bullet points or headers
            bullets = [line for line in lines if line.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.", "#"))]
            if bullets:
                compressed = "Summary of previous AI response:\n" + "\n".join(bullets[:8])
            else:
                compressed = content[:max_chars] + "... [summary truncated for context efficiency]"
            
            compact_history.append({
                **item,
                "content": compressed,
            })
        else:
            compact_history.append({
                **item,
                "content": content,
            })
            
    return compact_history


def _deterministic_summary(text: str, max_chars: int) -> str:
    """Bound the older transcript without inventing or losing all context."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    bullets = [line for line in lines if line.lstrip().startswith(("-", "*", "•", "1.", "2.", "3."))]
    selected = bullets[:20] if bullets else lines[-12:]
    summary = "Earlier conversation (locally compacted; no model summary available):\n" + "\n".join(selected)
    return summary[:max(1, max_chars)]


class IntelligentMemoryCompressor:
    def __init__(self, fast_llm=None):
        # Do not construct a provider during module import/startup. The model
        # is resolved only if a real compression is requested.
        self._fast_llm = fast_llm
        # Watermark cache to prevent redundant consolidation calls: {business_id: max_message_created_at}
        self._watermarks: dict[str, datetime] = {}

    @property
    def fast_llm(self):
        if self._fast_llm is None:
            self._fast_llm = get_llm().chat
        return self._fast_llm

    async def update_global_profile(self, business_id: str, force: bool = False) -> str | None:
        """
        Updates long-term memory for the user profile, separating BRAND FACTS from ACTIVE PLANS.
        Enforces a hard token cap on FACTS (<= 300 tokens).
        Stores PLANS separately in BrandMemoryItem(kind="strategy_blueprint").
        Skips consolidation if no new messages arrived since last consolidation (dirty check).
        """
        async with session_scope() as db:
            # 1. Dirty check: has any new message arrived since the last consolidation?
            watermark_stmt = (
                select(func.max(Message.created_at))
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(Conversation.business_id == business_id)
            )
            latest_msg_time = await db.scalar(watermark_stmt)
            if not force and latest_msg_time is not None:
                last_time = self._watermarks.get(business_id)
                if last_time is not None and latest_msg_time <= last_time:
                    logger.info(
                        "ltm_consolidation_skipped_no_new_messages",
                        extra={"business_id": business_id, "watermark": str(latest_msg_time)},
                    )
                    stmt = select(Profile.user_context_summary).where(Profile.business_id == business_id).limit(1)
                    return await db.scalar(stmt)

            # 2. Fetch Profile for this business
            stmt = select(Profile).where(Profile.business_id == business_id).limit(1)
            profile = await db.scalar(stmt)
            if not profile:
                profile = Profile(
                    id=uuid.uuid4(),
                    supabase_user_id=uuid.uuid4(),
                    business_id=business_id,
                    role="owner",
                    user_context_summary=""
                )
                db.add(profile)
                await db.flush()

            old_summary = str(redact_sensitive_data(profile.user_context_summary or "No previous summary recorded."))

            # Check if existing summary needs migration/resplit (has plans in old summary)
            if old_summary and any(kw in old_summary.lower() for kw in ["active marketing", "goals", "budget", "winning ad angles"]):
                logger.info("ltm_auto_resplit_triggering", extra={"business_id": business_id})
                old_summary, _ = await resplit_existing_profile(business_id, db, llm=self.fast_llm)

            # 3. Fetch recent messages for this business across conversations
            conv_stmt = (
                select(Conversation.id)
                .where(Conversation.business_id == business_id)
                .order_by(desc(Conversation.updated_at))
                .limit(3)
            )
            conv_ids = (await db.execute(conv_stmt)).scalars().all()
            if not conv_ids:
                return profile.user_context_summary

            msg_stmt = (
                select(Message)
                .where(Message.conversation_id.in_(conv_ids))
                .order_by(desc(Message.created_at))
                .limit(20)
            )
            recent_rows = (await db.execute(msg_stmt)).scalars().all()
            if not recent_rows:
                return profile.user_context_summary

            recent_chat_lines = [
                f"{r.role.upper()}: {str(redact_sensitive_data(_message_to_text(r)[:400]))}"
                for r in reversed(recent_rows)
                if _message_to_text(r).strip()
            ]
            recent_text = "\n".join(recent_chat_lines)

            # 4. Prompt fast model for dual-layer distillation (FACTS vs PLANS)
            from app.ai.settings import get_ai_settings
            ai_settings = get_ai_settings()
            max_profile_tokens = ai_settings.memory_global_profile_max_tokens

            sys_msg = SystemMessage(content=(
                "You are an expert AI memory distillation assistant for the SAHM Marketing OS.\n"
                "Your job is to update the user's permanent memory by producing TWO SEPARATE ARTIFACTS:\n\n"
                "ARTIFACT 1: === BRAND FACTS ===\n"
                "- Extract ONLY permanent descriptive facts: Brand Name, Industry/Sector, Geography/Country, Target Audience, Pricing Policy, Known Competitors.\n"
                "- STRICTLY FORBIDDEN: Do NOT include active marketing goals, ad campaigns, ad angles, budgets, or channel roadmaps here.\n"
                f"- STRICT CONSTRAINT: MAXIMUM {max_profile_tokens} TOKENS. Descriptive only.\n\n"
                "ARTIFACT 2: === ACTIVE PLANS ===\n"
                "- Extract active marketing goals, budget allocations, ad angles, channel strategies, and campaign learnings.\n"
                "- Format as concise bullet points."
            ))

            usr_msg = HumanMessage(content=(
                f"=== CURRENT BRAND FACTS ===\n{old_summary or 'No previous facts recorded.'}\n\n"
                f"=== RECENT CONVERSATION LOGS ===\n{recent_text}\n\n"
                "Produce the updated artifacts using the headers '=== BRAND FACTS ===' and '=== ACTIVE PLANS ===':"
            ))

            try:
                response = await self.fast_llm.ainvoke([sys_msg, usr_msg])
                raw_response = (
                    "\n".join([str(p.get("text", p) if isinstance(p, dict) else p) for p in response.content]).strip()
                    if isinstance(response.content, list)
                    else str(response.content).strip()
                )

                # Parse two artifacts
                new_facts = ""
                new_plans = ""
                if "=== BRAND FACTS ===" in raw_response:
                    parts = raw_response.split("=== ACTIVE PLANS ===")
                    new_facts = parts[0].replace("=== BRAND FACTS ===", "").strip()
                    if len(parts) > 1:
                        new_plans = parts[1].strip()
                else:
                    new_facts, new_plans = resplit_summary_text(raw_response)

                # 5. Enforce hard token cap on FACTS (Task 5)
                async def _retry_facts(cur_tok: int, cap: int, text: str) -> str:
                    retry_sys = SystemMessage(content=(
                        f"Your previous BRAND FACTS output was {cur_tok} tokens, which EXCEEDS the hard limit of {cap} tokens.\n"
                        f"Compress it strictly under {cap} tokens while keeping Brand Name, Industry, Audience, and Competitors."
                    ))
                    r_res = await self.fast_llm.ainvoke([retry_sys, HumanMessage(content=text)])
                    return str(r_res.content).strip()

                new_facts = await enforce_token_cap(
                    new_facts,
                    max_profile_tokens,
                    context_label=f"ltm_profile_{business_id}",
                    retry_prompt_func=_retry_facts,
                )

                # 6. Check if facts changed (no-op check)
                if old_summary and new_facts.strip() == old_summary.strip():
                    logger.info("ltm_profile_unchanged_noop", extra={"business_id": business_id})
                    if latest_msg_time is not None:
                        self._watermarks[business_id] = latest_msg_time
                    return profile.user_context_summary

                # Scrub model output too: a distiller must never persist credentials.
                new_facts = str(redact_sensitive_data(new_facts))[: max_profile_tokens * 4]
                new_plans = str(redact_sensitive_data(new_plans))

                # 7. Save BRAND FACTS to Profile.user_context_summary
                profile.user_context_summary = new_facts

                # 8. Save ACTIVE PLANS to BrandMemoryItem(kind="strategy_blueprint")
                if new_plans:
                    plan_stmt = (
                        select(BrandMemoryItem)
                        .where(
                            BrandMemoryItem.business_id == business_id,
                            BrandMemoryItem.kind == "strategy_blueprint",
                        )
                        .limit(1)
                    )
                    existing_plan = await db.scalar(plan_stmt)
                    now = datetime.now(timezone.utc)
                    if existing_plan:
                        existing_plan.text = new_plans
                        existing_plan.metadata_ = {
                            **(existing_plan.metadata_ or {}),
                            "active_plans": new_plans,
                            "updated_at": now.isoformat(),
                        }
                    else:
                        new_plan_item = BrandMemoryItem(
                            id=f"mem_plan_{uuid.uuid4().hex[:12]}",
                            business_id=business_id,
                            kind="strategy_blueprint",
                            text=new_plans,
                            metadata_={
                                "active_plans": new_plans,
                                "source": "ltm_compressor",
                                "created_at": now.isoformat(),
                            },
                            created_at=now,
                        )
                        db.add(new_plan_item)

                await db.commit()

                # Update watermark
                if latest_msg_time is not None:
                    self._watermarks[business_id] = latest_msg_time

                logger.info(
                    "ltm_profile_updated",
                    extra={
                        "business_id": business_id,
                        "facts_tokens": count_tokens(new_facts),
                        "plans_tokens": count_tokens(new_plans),
                    },
                )
                return new_facts
            except Exception as e:
                logger.error("ltm_update_failed", extra={"business_id": business_id, "error": str(e), "error_type": type(e).__name__})
                return profile.user_context_summary if profile else None


class MemoryDebouncer:
    """
    Manages configurable inactivity debounce timers per business.
    Every new message resets the countdown timer. When the user is inactive for the duration,
    the background compressor fires automatically.
    """
    def __init__(self, delay_seconds: int | None = None):
        if delay_seconds is None:
            try:
                from app.ai.settings import get_ai_settings
                delay_seconds = get_ai_settings().memory_ltm_debounce_seconds
            except Exception:
                delay_seconds = 180
        self.delay_seconds = delay_seconds
        self._timers: dict[str, asyncio.Task[None]] = {}
        self.compressor = IntelligentMemoryCompressor()

    def schedule_ltm_update(self, business_id: str):
        """Schedule or reset the inactivity timer."""
        # Cancel any existing active timer for this business
        existing_task = self._timers.get(business_id)
        if existing_task and not existing_task.done():
            existing_task.cancel()

        # Start a new countdown task
        task = asyncio.create_task(self._countdown_and_execute(business_id))
        self._timers[business_id] = task

    async def _countdown_and_execute(self, business_id: str):
        try:
            await asyncio.sleep(self.delay_seconds)
            logger.info("ltm_debounce_triggered", extra={"business_id": business_id})
            await self.compressor.update_global_profile(business_id)
        except asyncio.CancelledError:
            # Timer was reset by a new incoming message; normal behavior
            pass
        except Exception as e:
            logger.error("ltm_debounce_error", extra={"business_id": business_id, "error": str(e)})
        finally:
            if self._timers.get(business_id) is asyncio.current_task():
                self._timers.pop(business_id, None)


# Process-wide singleton debouncer
memory_debouncer = MemoryDebouncer()


async def manage_session_memory_with_token_cap(
    db: AsyncSession,
    conversation: Conversation,
    messages: list[Message],
    max_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Manages chat session memory using a configurable threshold token cap:
    - Before reaching max_tokens: behaves like a normal chat agent, passing raw past questions and answers.
    - When max_tokens is reached: summarizes the older conversation turns using the fast LLM,
      persists the summary to `conversation.summary` in the database, and returns only the 
      concise summary + the latest turns for the next question.
    - Updates the summary next time the token threshold is exceeded in the same chat.
    """
    if max_tokens is None:
        try:
            from app.ai.settings import get_ai_settings
            max_tokens = get_ai_settings().memory_session_max_tokens
        except Exception:
            max_tokens = 10000

    try:
        from app.ai.settings import get_ai_settings
        retain_count = get_ai_settings().memory_retain_recent_turns
    except Exception:
        retain_count = 2

    valid_msgs = [m for m in messages if m.role in ("user", "assistant") and (getattr(m, "content", None) or _message_to_text(m)).strip()]
    safe_content = {id(m): str(redact_sensitive_data(getattr(m, "content", None) or _message_to_text(m))) for m in valid_msgs}
    
    # Calculate measured tokens using real tokenizer
    total_tokens = sum(count_tokens(safe_content[id(m)]) for m in valid_msgs)

    # 1. Normal mode: below max_tokens
    if total_tokens < max_tokens:
        history = [
            {
                "role": m.role,
                "content": safe_content[id(m)],
                "persona_id": m.persona_id,
                "blocks": redact_sensitive_data(m.blocks or []),
            }
            for m in valid_msgs
        ]
        return history, str(redact_sensitive_data(conversation.summary or "")) or None

    # Retain the configured number of recent messages for immediate conversational context
    if len(valid_msgs) > retain_count:
        older_msgs = valid_msgs[:-retain_count]
        recent_msgs = valid_msgs[-retain_count:]
    else:
        older_msgs = valid_msgs[:-1]
        recent_msgs = valid_msgs[-1:]

    # 2. Watermark / growth gating check
    conv_ctx = dict(getattr(conversation, "context", None) or {})
    last_summary_tokens = conv_ctx.get("summary_token_watermark")

    if conversation.summary and last_summary_tokens is not None and last_summary_tokens > 0:
        growth = (total_tokens - last_summary_tokens) / last_summary_tokens
        if growth < 0.25:
            logger.info(
                "session_memory_growth_gated",
                extra={
                    "conversation_id": conversation.id,
                    "growth": round(growth, 3),
                    "total_tokens": total_tokens,
                    "last_summary_tokens": last_summary_tokens,
                },
            )
            compact_history = [
                {
                    "role": m.role,
                    "content": safe_content[id(m)],
                    "persona_id": m.persona_id,
                    "blocks": redact_sensitive_data(m.blocks or []),
                }
                for m in recent_msgs
            ]
            return compact_history, conversation.summary

    # 3. Threshold exceeded & growth >= 25% (or no prior summary): summarize older turns
    logger.info(
        "session_memory_token_cap_exceeded", 
        extra={"conversation_id": conversation.id, "tokens": total_tokens, "max_tokens": max_tokens}
    )

    older_text = "\n".join([f"{m.role.upper()}: {safe_content[id(m)]}" for m in older_msgs])
    current_summary = str(redact_sensitive_data(conversation.summary or "No previous summary."))

    sys_prompt = SystemMessage(content=(
        "You are an expert conversation memory assistant.\n"
        "Summarize the past conversation history between the user and the AI assistant into a compact, "
        "high-density summary.\n"
        "CRITICAL RULES:\n"
        "- Retain all essential user questions, stated goals, brand facts, numbers, competitor names, and key AI recommendations.\n"
        "- Drop chit-chat, greetings, and temporary clarification loops.\n"
        "- Merge this new conversation chunk seamlessly into the existing summary."
    ))
    usr_prompt = HumanMessage(content=(
        f"=== EXISTING CHAT SUMMARY ===\n{current_summary}\n\n"
        f"=== CONVERSATION TURNS TO SUMMARIZE ===\n{older_text}\n\n"
        "Provide the updated, unified chat summary:"
    ))

    try:
        fast_llm = get_llm().chat
        response = await fast_llm.ainvoke([sys_prompt, usr_prompt])
        if isinstance(response.content, list):
            new_summary = "\n".join([
                str(p.get("text", p) if isinstance(p, dict) else p)
                for p in response.content
            ]).strip()
        else:
            new_summary = str(response.content).strip()

        # Scrub model output and enforce token cap
        new_summary = str(redact_sensitive_data(new_summary))
        new_summary = await enforce_token_cap(
            new_summary,
            max_tokens // 2,
            context_label=f"session_summary_{conversation.id}",
        )
        new_summary = new_summary[: max_tokens * 4]
        # Update the database
        conversation.summary = new_summary
        conv_ctx = dict(getattr(conversation, "context", None) or {})
        conv_ctx["summary_token_watermark"] = total_tokens
        conversation.context = conv_ctx
        await db.flush()
        logger.info(
            "conversation_summary_updated",
            extra={
                "conversation_id": conversation.id,
                "tokens": count_tokens(new_summary),
                "watermark_tokens": total_tokens,
            },
        )
    except Exception as e:
        logger.error("conversation_summary_failed", extra={"conversation_id": conversation.id, "error_type": type(e).__name__})
        new_summary = _deterministic_summary(older_text, max_tokens * 4)
        new_summary = str(redact_sensitive_data(new_summary))
        conversation.summary = new_summary
        await db.flush()

    # Return only recent turns + the summary
    compact_history = [
        {
            "role": m.role,
            "content": safe_content[id(m)],
            "persona_id": m.persona_id,
            "blocks": redact_sensitive_data(m.blocks or []),
        }
        for m in recent_msgs
    ]
    return compact_history, new_summary
