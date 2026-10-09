from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import Profile, Conversation, Message
from app.ai.providers.llm import get_llm
from app.ai.enums import LLMProvider

class IntelligentMemoryCompressor:
    def __init__(self, db: AsyncSession):
        self.db = db
        # We use the system's default fast LLM for background compression
        self.fast_llm = get_llm().chat

    async def compress_latest_turn(self, conversation_id: str):
        """
        Compresses the latest AI message if it's too long, and cleans up junk.
        (This is a conceptual implementation of the Intelligent Session Memory).
        """
        # In a real scenario, we'd fetch the latest uncompressed messages
        # from the database, evaluate their token length, and rewrite them.
        pass

    async def update_global_profile(self, profile_id: str, recent_messages: List[str]):
        """
        Updates the global 500-token long-term memory for the user profile.
        Triggered after 3 minutes of inactivity.
        """
        # 1. Fetch Profile
        result = await self.db.execute(select(Profile).where(Profile.id == profile_id))
        profile = result.scalar_one_or_none()
        
        if not profile:
            return

        old_summary = profile.user_context_summary or ""
        
        # 2. Run summarization
        sys_msg = SystemMessage(content=(
            "You are a background memory processor. Update the user profile summary "
            "with the newly provided recent conversation details. Keep only permanent facts: "
            "brand details, competitors, tone preferences, and major goals. Discard casual chat. "
            "MAXIMUM 500 TOKENS. DO NOT EXCEED THIS LIMIT."
        ))
        
        recent_text = "\n".join(recent_messages)
        usr_msg = HumanMessage(content=f"Old Summary:\n{old_summary}\n\nRecent Chat to Extract From:\n{recent_text}")
        
        response = await self.fast_llm.ainvoke([sys_msg, usr_msg])
        new_summary = response.content
        
        # 3. Save to DB
        profile.user_context_summary = new_summary
        await self.db.commit()
        
        return new_summary
