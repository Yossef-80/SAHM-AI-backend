import asyncio
from app.db.session import get_sessionmaker
from sqlalchemy import select, func, text

async def check():
    maker = get_sessionmaker()
    async with maker() as session:
        result = await session.execute(text('SELECT count(*) FROM conversations'))
        print(f'Conversations: {result.scalar()}')
        result = await session.execute(text('SELECT count(*) FROM messages'))
        print(f'Messages: {result.scalar()}')

asyncio.run(check())
