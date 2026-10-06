import asyncio
from langchain_core.messages import HumanMessage
from app.ai.agents.nour import nour_graph
from app.ai.settings import get_ai_settings
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

async def run_test():
    print("Starting Modular Nour Test...\n")
    
    # 1. This is where you add your query!
    user_query = """What are the latest marketing trends for Classis Suits and what is the compatators offers and strenghts now
    and specificly analysi these competetors `tie house`: [https://tie-house.com/], `U-turn`: [uturn-store.com]"""
    print(f"User Query: {user_query}\n")
    
    # 2. We pass the business_id via RunnableConfig (so the tools can access it implicitly)
    config = {
        "configurable": {
            "business_id": "test_business_123"
        }
    }
    
    # 3. Create the initial state with the user's message
    initial_state = {
        "messages": [HumanMessage(content=user_query)]
    }
    
    print("Nour is thinking... (and potentially calling tools like web_search)\n")
    
    # 4. Run the graph!
    print("Nour is researching and writing the report...")
    final_state = await nour_graph.ainvoke(initial_state, config=config)
    
    # 5. Extract messages
    messages = final_state.get("messages", [])
    
    # 6. Save the step-by-step trace to a file
    import os
    os.makedirs("logs", exist_ok=True)
    trace_file = "logs/nour_trace.txt"
    
    import json
    with open(trace_file, "w", encoding="utf-8") as f:
        f.write("=== NOUR RESEARCH TRACE ===\n")
        f.write(f"User Query: {user_query}\n\n")
        
        for idx, msg in enumerate(messages):
            role = "System" if msg.type == "system" else "User" if msg.type == "human" else "Nour" if msg.type == "ai" else "Tool"
            f.write(f"[{idx}] {role.upper()}:\n")
            
            # Print tool calls if any
            if msg.type == "ai" and getattr(msg, "tool_calls", []):
                for call in msg.tool_calls:
                    f.write(f"  -> CALLED TOOL: {call['name']}\n")
                    f.write(f"  -> ARGS: {json.dumps(call['args'], ensure_ascii=False)}\n")
            
            # Print content
            if msg.content:
                f.write(f"{msg.content}\n")
                
            f.write("-" * 80 + "\n")
            
    print(f"\n✅ Step-by-step trace saved to: {trace_file}")
    
    # 7. Print the final structured response to the terminal
    print("\n--- FINAL STRUCTURED REPORT ---")
    for msg in reversed(messages):
        if msg.type == "ai" and getattr(msg, "tool_calls", []):
            for call in msg.tool_calls:
                if call["name"] == "submit_research_report":
                    print(json.dumps(call["args"], indent=2, ensure_ascii=False))
                    return
                    
    print("Warning: Nour finished but did not submit a structured report.")

if __name__ == "__main__":
    asyncio.run(run_test())
