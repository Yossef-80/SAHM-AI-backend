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
    user_query = """Task: We need to analyze our brand and our main competitor using the new Brand Research MCP tools.

Target Brands:

Our Brand: [Your Brand Name] (Website: [https://xmegacorner.com], Country: [Egypt], Code: [EG])
Competitor: [Competitor Brand Name] (Website: [https://nacitaautocare.com/], Country: [Egypt], Code: [EG])

Execution Rules & Constraints (STRICTLY FOLLOW):

Limit Tool Calls: Do NOT make separate search tool calls for basic concepts like "strengths" or "weaknesses". Group your analysis.
Be Concise: I do not need long, conversational paragraphs. Give me dense, bulleted data.
Workflow:
Call the brand_report tool for Our Brand.
Call the brand_report tool for the Competitor.
Call full_analysis for both brands to get sentiment and product insights.

Final Output Required: Once the tools finish collecting and analyzing the data, provide a single, concise comparative summary. Include:

Engagement Comparison: Who has a better social engagement rate?
Sentiment Comparison: Who has a higher positive review percentage and average stars?
Strengths vs Weaknesses (Combined): Based on the product_insights top praises and complaints, give me a single bulleted list of where we beat them and where they beat us."""
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
