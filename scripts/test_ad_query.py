import asyncio
from langchain_core.messages import SystemMessage, HumanMessage
from app.ai.providers.llm import get_llm
from app.ai.providers.search import get_search_provider
from app.ai.settings import get_ai_settings

settings = get_ai_settings()
provider = get_search_provider(settings.search_provider)
llm = get_llm().chat

brand = "Tie House"
ad_query = f'"{brand}" ads ad creatives "Meta Ad Library" facebook ads instagram campaigns video hooks commercial Egypt'
hits = provider.search(ad_query, max_results=4)

print(f"=== Hits for ad query: '{ad_query}' ===")
for h in hits:
    print(f"- {h.title} | {h.url}")
    print(f"  Snippet: {h.snippet[:120]}...\n")
