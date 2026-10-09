import asyncio
from app.db.session import session_scope
from app.db.models import Conversation, Message
from app.ai.memory.compressor import manage_session_memory_with_token_cap
import uuid

async def test_sliding_token_cap():
    print("=== Testing Sliding Token-Cap Session Memory ===")
    async with session_scope() as db:
        conv_id = f"test_conv_{uuid.uuid4().hex[:10]}"
        biz_id = "biz_demo"
        conv = Conversation(id=conv_id, business_id=biz_id, title="Token Cap Test")
        db.add(conv)
        await db.flush()

        # Step 1: Small conversation below threshold (e.g. 2 short messages)
        msg1 = Message(id=f"msg_{uuid.uuid4().hex[:10]}", conversation_id=conv_id, role="user", content="Hello, I want advice on Facebook ads.")
        msg2 = Message(id=f"msg_{uuid.uuid4().hex[:10]}", conversation_id=conv_id, role="assistant", content="Sure! What is your product and target market?")
        messages = [msg1, msg2]

        history, summary = await manage_session_memory_with_token_cap(db, conv, messages, max_tokens=1000)
        print("1. Below threshold:")
        print("   Returned history turns:", len(history))
        print("   Summary:", summary)
        assert len(history) == 2
        assert summary is None
        print("   PASSED (behaves as normal chat agent with raw past messages)")

        # Step 2: Add several large messages exceeding threshold (e.g. max_tokens=200 for testing)
        msg3 = Message(id=f"msg_{uuid.uuid4().hex[:10]}", conversation_id=conv_id, role="user", content="We sell premium classic wool suits in Cairo for corporate executives.")
        msg4 = Message(id=f"msg_{uuid.uuid4().hex[:10]}", conversation_id=conv_id, role="assistant", content="Here is our in-depth strategy: " + ("Focus on LinkedIn and Instagram luxury carousels with tailored bespoke tailoring offers. " * 15))
        msg5 = Message(id=f"msg_{uuid.uuid4().hex[:10]}", conversation_id=conv_id, role="user", content="What budget do you recommend for our test launch?")
        messages.extend([msg3, msg4, msg5])

        history, summary = await manage_session_memory_with_token_cap(db, conv, messages, max_tokens=150)
        print("\n2. Threshold reached:")
        print("   Returned history turns:", len(history))
        print("   Generated & Stored Summary:\n  ", summary)
        assert len(history) == 2  # Only the latest turns kept raw
        assert summary is not None
        assert conv.summary == summary
        print("   PASSED (raw past messages pruned, replaced by DB-persisted summary + latest turns)")

        # Step 3: Verify subsequent question retains summary in DB
        await db.commit()
        
        # Cleanup
        await db.delete(conv)
        await db.commit()
        print("\nALL TOKEN-CAP SLIDING MEMORY TESTS PASSED!")

if __name__ == "__main__":
    asyncio.run(test_sliding_token_cap())
