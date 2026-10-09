import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.runnables.config import RunnableConfig
from app.ai.tools.search import web_search_tool
from app.ai.memory.compressor import IntelligentMemoryCompressor
from app.db.session import session_scope, create_all
from app.db.models import Profile, Business
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import uuid

async def test_search():
    print("=== Testing Summarized Search ===")
    config = RunnableConfig()
    try:
        # We search for something highly specific
        query = "best mechanical keyboards under 100 dollars"
        result = web_search_tool.invoke({"query": query}, config)
        print("Search Result:\n", result)
    except Exception as e:
        print(f"Error during search: {e}")


async def test_memory():
    print("\n=== Testing Intelligent Memory Compressor ===")
    
    # Ensure tables are created for SQLite
    await create_all()
    
    async with session_scope() as db:
        # Create a dummy business and profile to test the DB update
        biz_id = uuid.uuid4().hex[:12]
        user_uuid = uuid.uuid4()
        profile_id = uuid.uuid4()
        
        # Raw SQL to bypass complex object setup for a quick test
        await db.execute(text(f"INSERT INTO businesses (id, name) VALUES ('{biz_id}', 'Test Business')"))
        await db.execute(text(f"INSERT INTO profiles (id, supabase_user_id, business_id, role, user_context_summary) VALUES ('{profile_id}', '{user_uuid}', '{biz_id}', 'owner', 'Old summary: None')"))
        await db.commit()
        
        compressor = IntelligentMemoryCompressor(db)
        
        recent_chat = [
            "User: We are a high-end coffee shop in Cairo.",
            "AI: Sounds great, what is your tone of voice?",
            "User: We like to be playful and modern. Our main competitor is Brown Nose.",
            "AI: Understood. Let's make a campaign.",
            "User: Sounds good."
        ]
        
        print("Running Compressor on recent chat...")
        new_summary = await compressor.update_global_profile(str(profile_id), recent_chat)
        print("\nNew Global Profile Summary:")
        print(new_summary)
        
        # Cleanup
        await db.execute(text(f"DELETE FROM profiles WHERE id = '{profile_id}'"))
        await db.execute(text(f"DELETE FROM businesses WHERE id = '{biz_id}'"))
        await db.commit()

async def main():
    # Only test search if keys are available
    if os.getenv("TAVILY_API_KEY") and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        await test_search()
    else:
        print("Skipping search test: TAVILY_API_KEY and GEMINI_API_KEY required.")
        
    await test_memory()

if __name__ == "__main__":
    asyncio.run(main())
