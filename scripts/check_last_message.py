import asyncio
from app.db.session import get_sessionmaker
from sqlalchemy import text
import json

async def check():
    maker = get_sessionmaker()
    async with maker() as session:
        res = await session.execute(text("SELECT role, content, blocks, created_at FROM messages WHERE conversation_id = 'conv_13eed086a5ff4037baa7' ORDER BY created_at ASC"))
        rows = res.fetchall()
        for idx, (role, content, blocks, created_at) in enumerate(rows):
            print(f"=== Turn {idx+1}: {role.upper()} ({created_at}) ===")
            print("Content Preview:\n", (content[:300] + "...") if content and len(content) > 300 else content)
            if blocks:
                print("Blocks count:", len(blocks))
                for b in blocks[:4]:
                    print("  Block:", b.get("type"), "-", str(b.get("title") or b.get("text"))[:80])

asyncio.run(check())
