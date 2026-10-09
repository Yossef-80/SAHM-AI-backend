import asyncio
from app.db.session import get_sessionmaker
from sqlalchemy import text

async def check():
    maker = get_sessionmaker()
    async with maker() as session:
        res = await session.execute(text("SELECT id, user_id, business_id, role FROM memberships WHERE business_id = 'biz_d335699704e740219870'"))
        print("Memberships:", res.fetchall())
        res2 = await session.execute(text("SELECT id, email FROM users WHERE id = 'usr_ddd3592bfe5946398f4b345f'"))
        print("Users:", res2.fetchall())

asyncio.run(check())
