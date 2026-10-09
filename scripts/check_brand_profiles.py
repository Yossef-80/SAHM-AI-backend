import asyncio
from app.db.session import get_sessionmaker
from sqlalchemy import text

async def check():
    maker = get_sessionmaker()
    async with maker() as session:
        res = await session.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'brand_profiles'"))
        for col in res.fetchall():
            print(f"  - {col[0]} ({col[1]})")

asyncio.run(check())
