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
    user_query = "What are the latest marketing trends for Classis Suits and what is the compatators offers and strenghts now"
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
    # We use .astream to see the events as they happen, or .invoke to just get the final result.
    # Here we use .invoke for simplicity.
    final_state = await nour_graph.ainvoke(initial_state, config=config)
    
    # 5. Extract and print the final response
    messages = final_state.get("messages", [])
    
    print("--- CONVERSATION HISTORY ---")
    for msg in messages:
        role = "System" if msg.type == "system" else "User" if msg.type == "human" else "Nour" if msg.type == "ai" else "Tool"
        
        # We only print the first 100 characters of tool responses to avoid clutter
        content_preview = str(msg.content)
        
        if msg.type == "ai" and msg.tool_calls:
            print(f"Nour called tools: {[t['name'] for t in msg.tool_calls]}")
        else:
            print(f"[{role}]: {content_preview}\n")

if __name__ == "__main__":
    asyncio.run(run_test())
