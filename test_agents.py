import sys
import os
import pprint

# Add backend to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "")))

from app.ai.agents.orchestrator.brain import orchestrate

def test_prompt(prompt: str):
    print("\n" + "="*80)
    print(f"TESTING PROMPT:\n{prompt}")
    print("="*80)
    
    result = orchestrate(prompt)
    
    if result.get("agent") != "unknown":
        print(f"\n[EXECUTED AGENT]: {result['agent'].upper()}")
        last_message = result['state']['messages'][-1]
        print(f"\n[FINAL RESPONSE / TOOL CALLS]:\n")
        pprint.pprint(last_message.model_dump())
    else:
        print("\n[ERROR]: Could not route to an agent.")

if __name__ == "__main__":
    prompts = [
        "Can you write 3 catchy Facebook ad variations for our new winter jacket line, focusing on warmth and style?",
        "Please analyze our Facebook ads performance for the last 30 days. Is the CTR dropping?",
        "I need a complete content strategy for Q4, including budget splits across Meta and TikTok.",
        "Check KFC Egypt's latest google maps reviews and tell me what people are complaining about."
    ]
    
    for p in prompts:
        test_prompt(p)
