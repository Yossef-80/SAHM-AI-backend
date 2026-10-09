import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.ai.providers.llm import get_llm
from app.ai.providers.search import get_search_provider
from app.ai.settings import get_ai_settings

settings = get_ai_settings()
provider = get_search_provider(settings.search_provider)
llm = get_llm().chat

verify_prompt = (
    "You are a Ground-Truth Entity Verifier for a business intelligence system.\n"
    "Determine if the user query refers to a real, verifiable brand, company, product, or competitor in the market.\n"
    "Evaluate against the search results.\n\n"
    "Respond STRICTLY in JSON (inside ```json ``` codeblock or raw JSON) with keys:\n"
    "- status: 'verified' | 'typo_corrected' | 'not_found'\n"
    "- corrected_name: string (if typo_corrected or clean entity name, otherwise empty)\n"
    "- explanation: string (short explanation of why it was verified, corrected, or why it does not exist)\n"
    "- confidence: float between 0.0 and 1.0\n"
)

test_queries = ["Tie House", "Tiee Hose", "asdfghjkl9999fake", "Concrete Suits Egypt"]

for q in test_queries:
    hits = provider.search(f"{q} egypt business brand company", max_results=3)
    hits_text = "\n".join([f"Title: {h.title}\nURL: {h.url}\nSnippet: {h.snippet}" for h in hits])
    
    usr = f"User Query: '{q}'\n\nSearch Results:\n{hits_text}"
    res = llm.invoke([SystemMessage(content=verify_prompt), HumanMessage(content=usr)])
    print(f"\n*** Query: {q} ***")
    content = res.content
    if isinstance(content, list):
        content = "\n".join([p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content])
    print(content)
