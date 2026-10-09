import asyncio
from app.ai.memory.compressor import IntelligentMemoryCompressor, compress_history_messages
from app.db.session import session_scope
from sqlalchemy import text

async def test_history_compression():
    print("=== 1. Testing Session History Compression ===")
    sample_history = [
        {"role": "user", "content": "Tell me your top 5 strategy recommendations for men suits in Egypt."},
        {
            "role": "assistant",
            "content": (
                "Here is an in-depth 2000-word analysis:\n"
                "1. Focus on lightweight Egyptian cotton blends to tackle seasonal heat.\n"
                "2. Offer tailored fitting services via home appointments in New Cairo and Sheikh Zayed.\n"
                "3. Partner with wedding organizers and corporate event planners.\n"
                "4. Leverage short-form video on TikTok showcasing suit durability and craftsmanship.\n"
                "5. Introduce an installment payment model via ValU or Sympl.\n"
                + ("A" * 1500)
            )
        },
        {"role": "user", "content": "What about point 2?"}
    ]
    compressed = compress_history_messages(sample_history, max_chars=400)
    print("Original assistant length:", len(sample_history[1]["content"]))
    print("Compressed assistant length:", len(compressed[1]["content"]))
    print("Compressed preview:\n", compressed[1]["content"][:300])
    assert len(compressed[1]["content"]) < len(sample_history[1]["content"])
    print("PASSED!\n")

async def test_ltm_db_update():
    print("=== 2. Testing LTM Global Profile Update in Supabase ===")
    async with session_scope() as db:
        # Check an existing business id in the database
        res = await db.execute(text("SELECT business_id FROM profiles LIMIT 1"))
        row = res.fetchone()
        if not row:
            print("No profiles found to test with.")
            return
        biz_id = row[0]
        print(f"Testing with business_id: {biz_id}")

        compressor = IntelligentMemoryCompressor()
        new_summary = await compressor.update_global_profile(biz_id)
        print("Generated LTM Summary:\n", new_summary)
        
        # Verify in DB
        res_verify = await db.execute(text(f"SELECT user_context_summary FROM profiles WHERE business_id = '{biz_id}'"))
        saved = res_verify.scalar()
        print("Verified from DB:\n", saved)
        assert saved is not None
        print("PASSED!\n")

async def main():
    await test_history_compression()
    await test_ltm_db_update()
    print("ALL LTM INTEGRATION TESTS PASSED!")

if __name__ == "__main__":
    asyncio.run(main())
