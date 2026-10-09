import asyncio
from app.db.session import get_sessionmaker
from sqlalchemy import text

async def check():
    maker = get_sessionmaker()
    async with maker() as session:
        res = await session.execute(text("SELECT business_id, user_context_summary FROM profiles WHERE business_id = 'biz_d335699704e740219870'"))
        row = res.fetchone()
        if row:
            print("Business:", row[0])
            print("LTM Profile Context:\n", row[1])
        else:
            print("Profile not found")

asyncio.run(check())
