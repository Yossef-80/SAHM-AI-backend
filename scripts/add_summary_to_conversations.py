import asyncio
from app.db.session import get_sessionmaker
from sqlalchemy import text

async def main():
    maker = get_sessionmaker()
    async with maker() as session:
        print("Adding summary column to conversations...")
        await session.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS summary TEXT;"))
        await session.commit()
        print("Column added!")
        res = await session.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'conversations'"))
        for col in res.fetchall():
            print(f"  - {col[0]} ({col[1]})")

asyncio.run(main())
