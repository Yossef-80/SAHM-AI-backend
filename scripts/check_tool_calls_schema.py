import asyncio
from app.db.session import get_sessionmaker
from sqlalchemy import text

async def main():
    maker = get_sessionmaker()
    async with maker() as session:
        res = await session.execute(text("SELECT tool_name, duration_ms, input, output_summary, created_at FROM agent_tool_calls LIMIT 5"))
        print("Sample agent_tool_calls:")
        for r in res.fetchall():
            print(" ", r)
            
        res_u = await session.execute(text("SELECT task_id, persona_id, tier, model, tokens_in, tokens_out, created_at FROM usage_records LIMIT 5"))
        print("\nSample usage_records:")
        for r in res_u.fetchall():
            print(" ", r)

asyncio.run(main())
