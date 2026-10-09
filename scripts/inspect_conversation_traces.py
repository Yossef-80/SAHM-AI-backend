import asyncio
from app.db.session import get_sessionmaker
from sqlalchemy import text

async def main():
    maker = get_sessionmaker()
    async with maker() as session:
        # Check conversations
        res_conv = await session.execute(text("SELECT id, title, created_at FROM conversations ORDER BY created_at ASC"))
        convs = res_conv.fetchall()
        print(f"Total conversations: {len(convs)}")
        for c in convs:
            print(f"  ID: {c[0]} | Title: {c[1]} | Created: {c[2]}")
            
        # Check tool calls
        res_tools = await session.execute(text("SELECT count(*) FROM agent_tool_calls"))
        print(f"Total agent_tool_calls: {res_tools.scalar()}")
        
        # Check usage records
        res_usage = await session.execute(text("SELECT count(*) FROM usage_records"))
        print(f"Total usage_records: {res_usage.scalar()}")

        # Check agent_tasks
        res_tasks = await session.execute(text("SELECT count(*) FROM agent_tasks"))
        print(f"Total agent_tasks: {res_tasks.scalar()}")

asyncio.run(main())
