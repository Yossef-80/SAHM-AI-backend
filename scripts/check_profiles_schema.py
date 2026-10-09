import asyncio
from app.db.session import get_sessionmaker
from sqlalchemy import text

async def main():
    maker = get_sessionmaker()
    async with maker() as session:
        res = await session.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'profiles'"))
        columns = res.fetchall()
        print('Columns in profiles:')
        for col in columns:
            print(f'  - {col[0]} ({col[1]})')

asyncio.run(main())
