import asyncio
import ast
import json
from sqlalchemy import select
from app.db.session import get_sessionmaker
from app.db.models import BrandMemoryItem, Profile

def clean_raw_string(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if (s.startswith("{'type'") or s.startswith("[{'type'") or 
        s.startswith('{"type"') or s.startswith('[{"type"')):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                texts = [item.get("text", "") for item in parsed if isinstance(item, dict)]
                if texts:
                    s = "\n".join(texts)
            elif isinstance(parsed, dict) and "text" in parsed:
                s = str(parsed["text"])
        except Exception:
            try:
                data = json.loads(s)
                if isinstance(data, list):
                    texts = [item.get("text", "") for item in data if isinstance(item, dict)]
                    if texts:
                        s = "\n".join(texts)
                elif isinstance(data, dict) and "text" in data:
                    s = str(data["text"])
            except Exception:
                pass
    s = s.replace("\ufffd", "")
    return s.strip()

async def main():
    maker = get_sessionmaker()
    async with maker() as session:
        # 1. Clean brand_memory_items
        items = (await session.execute(select(BrandMemoryItem))).scalars().all()
        updated_items = 0
        for item in items:
            cleaned = clean_raw_string(item.text)
            if cleaned != item.text:
                item.text = cleaned
                updated_items += 1
        
        # 2. Clean profile LTM
        profiles = (await session.execute(select(Profile))).scalars().all()
        updated_profiles = 0
        for p in profiles:
            if p.user_context_summary:
                cleaned = clean_raw_string(p.user_context_summary)
                if cleaned != p.user_context_summary:
                    p.user_context_summary = cleaned
                    updated_profiles += 1
        
        await session.commit()
        print(f"Cleaned {updated_items} BrandMemoryItem rows and {updated_profiles} Profile LTM rows.")

if __name__ == "__main__":
    asyncio.run(main())
