import asyncio
from app.db.session import get_sessionmaker
from sqlalchemy import text

async def main():
    maker = get_sessionmaker()
    async with maker() as session:
        print("Adding user_context_summary column to profiles...")
        await session.execute(text("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS user_context_summary TEXT;"))
        await session.commit()
        print("Column added successfully!")
        
        # Verify
        res = await session.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'profiles' AND column_name = 'user_context_summary'"))
        col = res.fetchone()
        print(f"Verification: {col}")

asyncio.run(main())
