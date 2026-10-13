from __future__ import annotations

import asyncio
from typing import Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Profile, Conversation, Message
from app.db.session import session_scope
from app.ai.providers.llm import get_llm
from app.ai.security import redact_sensitive_data
from app.core.logging import get_logger

logger = get_logger(__name__)


def compress_history_messages(history: list[dict[str, Any]], max_chars: int = 900) -> list[dict[str, Any]]:
    """
    Intelligent Session Memory:
    Compresses long assistant responses (>max_chars) in the working history
    so subsequent turns don't blow up token budgets.
    """
    compact_history = []
    for item in history:
        role = item.get("role")
        content = item.get("content") or ""
        
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
                "content": compressed
            })
        else:
            compact_history.append(item)
            
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

    @property
    def fast_llm(self):
        if self._fast_llm is None:
            self._fast_llm = get_llm().chat
        return self._fast_llm

    async def update_global_profile(self, business_id: str) -> str | None:
        """
        Updates the global 500-token long-term memory for the user profile.
        Triggered after 3 minutes of inactivity.
        """
        async with session_scope() as db:
            # 1. Fetch Profile for this business
            stmt = select(Profile).where(Profile.business_id == business_id).limit(1)
            profile = await db.scalar(stmt)
            if not profile:
                return None

            old_summary = str(redact_sensitive_data(profile.user_context_summary or "No previous summary recorded."))

            # 2. Fetch recent messages for this business across conversations
            conv_stmt = (
                select(Conversation.id)
                .where(Conversation.business_id == business_id)
                .order_by(desc(Conversation.updated_at))
                .limit(3)
            )
            conv_ids = (await db.execute(conv_stmt)).scalars().all()
            if not conv_ids:
                return None

            msg_stmt = (
                select(Message)
                .where(Message.conversation_id.in_(conv_ids))
                .order_by(desc(Message.created_at))
                .limit(20)
            )
            recent_rows = (await db.execute(msg_stmt)).scalars().all()
            if not recent_rows:
                return None

            recent_chat_lines = [
                f"{r.role.upper()}: {str(redact_sensitive_data(r.content[:400]))}"
                for r in reversed(recent_rows)
                if r.content
            ]
            recent_text = "\n".join(recent_chat_lines)

            # 3. Prompt fast model for configurable token distillation
            from app.ai.settings import get_ai_settings
            ai_settings = get_ai_settings()
            max_profile_tokens = ai_settings.memory_global_profile_max_tokens
            max_words = int(max_profile_tokens * 0.75)

            sys_msg = SystemMessage(content=(
                "You are an expert AI memory distillation assistant for the SAHM Marketing OS.\n"
                "Your job is to update the user's permanent profile background.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "- Extract ONLY permanent facts: Brand Name, Industry/Niche, Target Audience, Brand Voice/Tone, "
                "Known Competitors, Active Marketing Goals, and Budget Preferences.\n"
                "- DISCARD casual pleasantries, transient questions, and repetitive chit-chat.\n"
                "- MERGE the new conversation facts into the existing summary seamlessly.\n"
                f"- STRICT CONSTRAINT: MAXIMUM {max_profile_tokens} TOKENS (under {max_words} words). Format as high-density bullet points."
            ))

            usr_msg = HumanMessage(content=(
                f"=== CURRENT USER PROFILE SUMMARY ===\n{old_summary}\n\n"
                f"=== RECENT CONVERSATION LOGS ===\n{recent_text}\n\n"
                "Provide the updated, unified Long-Term Memory summary:"
            ))

            try:
                response = await self.fast_llm.ainvoke([sys_msg, usr_msg])
                if isinstance(response.content, list):
                    new_summary = "\n".join([
                        str(p.get("text", p) if isinstance(p, dict) else p)
                        for p in response.content
                    ]).strip()
                else:
                    new_summary = str(response.content).strip()

                # Scrub model output too: a distiller must never persist credentials.
                new_summary = str(redact_sensitive_data(new_summary))
                new_summary = new_summary[: max_profile_tokens * 4]

                # 4. Save to DB
                profile.user_context_summary = new_summary
                await db.commit()
                logger.info("ltm_profile_updated", extra={"business_id": business_id, "summary_len": len(new_summary)})
                return new_summary
            except Exception as e:
                logger.error("ltm_update_failed", extra={"business_id": business_id, "error_type": type(e).__name__})
                return None


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
            max_tokens = 1500

    try:
        from app.ai.settings import get_ai_settings
        retain_count = get_ai_settings().memory_retain_recent_turns
    except Exception:
        retain_count = 2

    valid_msgs = [m for m in messages if m.role in ("user", "assistant") and m.content]
    safe_content = {id(m): str(redact_sensitive_data(m.content or "")) for m in valid_msgs}
    
    # Calculate approximate tokens (~3.5 chars per token across English/Arabic)
    total_chars = sum(len(safe_content[id(m)]) for m in valid_msgs)
    approx_tokens = int(total_chars / 3.5)

    # 1. Normal mode: below max_tokens
    if approx_tokens < max_tokens:
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

    # 2. Threshold exceeded: summarize older turns and keep only the latest turns
    logger.info(
        "session_memory_token_cap_exceeded", 
        extra={"conversation_id": conversation.id, "tokens": approx_tokens, "max_tokens": max_tokens}
    )

    # Retain the configured number of recent messages for immediate conversational context
    if len(valid_msgs) > retain_count:
        older_msgs = valid_msgs[:-retain_count]
        recent_msgs = valid_msgs[-retain_count:]
    else:
        older_msgs = valid_msgs[:-1]
        recent_msgs = valid_msgs[-1:]

    older_text = "\n".join([f"{m.role.upper()}: {safe_content[id(m)]}" for m in older_msgs])
    current_summary = str(redact_sensitive_data(conversation.summary or "No previous summary."))

    sys_prompt = SystemMessage(content=(
        "You are an expert conversation memory assistant.\n"
        "Summarize the past conversation history between the user and the AI assistant into a compact, "
        "high-density summary (under 250 words).\n"
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

        # Scrub model output before saving or reusing it.
        new_summary = str(redact_sensitive_data(new_summary))
        new_summary = new_summary[: max_tokens * 4]
        # Update the database
        conversation.summary = new_summary
        await db.flush()
        logger.info("conversation_summary_updated", extra={"conversation_id": conversation.id, "len": len(new_summary)})
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

