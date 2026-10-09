import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.ai.providers.llm import get_llm
from app.ai.providers.search import get_search_provider
from app.ai.settings import get_ai_settings

settings = get_ai_settings()
provider = get_search_provider(settings.search_provider)
llm = get_llm().chat

typo_agent_sys = (
    "You are an expert Brand & Entity Verification Agent for the Egyptian and Global commercial markets.\n"
    "Your job is ground-truth verification of queries entered by users for competitor & business research.\n\n"
    "Tasks:\n"
    "1. Check if the query is a recognized active brand or business.\n"
    "2. If the query has typos or phonetic misspellings (e.g. 'Tiee Hose' -> 'Tie House', 'Concrette' -> 'Concrete', 'Town Tem' -> 'Town Team', 'Adiddas' -> 'Adidas', 'Zarra' -> 'Zara'), identify the intended real brand!\n"
    "3. If the query is completely nonsensical, gibberish, fake, or not an actual commercial brand/business in the market (e.g. 'asdfghjkl9999fake', 'randomfakeentityxyz'), mark it as not found.\n\n"
    "Respond STRICTLY in JSON format with keys:\n"
    "{\n"
    '  "status": "verified" | "typo_corrected" | "not_found",\n'
    '  "target_brand": "Clean or corrected brand name",\n'
    '  "original_query": "The input query",\n'
    '  "explanation": "Why verified, what typo was fixed, or why not found",\n'
    '  "confidence": 0.0 to 1.0\n'
    "}"
)

test_cases = [
    "Tie House",
    "Tiee Hose",
    "Concrette",
    "asdfghjkl9999fake",
    "Town Tem Egypt",
    "FakeBrandThatDoesNotExist12345"
]

for q in test_cases:
    # Ask LLM first with general knowledge + brand taxonomy
    res = llm.invoke([
        SystemMessage(content=typo_agent_sys),
        HumanMessage(content=f"User entered competitor/brand query: '{q}'\nCheck whether this is a real brand, a typo of a real brand, or not a real brand.")
    ])
    
    content = res.content
    if isinstance(content, list):
        content = "\n".join([p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content])
    print(f"\n--- Testing: {q} ---")
    print(content)
