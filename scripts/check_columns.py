import asyncio
from app.db.session import get_sessionmaker
from sqlalchemy import text

async def main():
    maker = get_sessionmaker()
    async with maker() as session:
        for tbl in ['agent_tool_calls', 'usage_records', 'messages', 'agent_tasks']:
            res = await session.execute(text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{tbl}'"))
            cols = [r[0] for r in res.fetchall()]
            print(f"Table {tbl}: {cols}")

asyncio.run(main())
