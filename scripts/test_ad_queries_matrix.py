from app.ai.settings import get_ai_settings
from app.ai.providers.search import get_search_provider

settings = get_ai_settings()
provider = get_search_provider(settings.search_provider)

queries = [
    'Tie House facebook ads instagram campaign commercial Egypt',
    'Tie House campaign commercial ad creative video hook اعلان',
    'Tie House Mohamed Zidan RACK Agency commercial ad campaign Egypt'
]

for q in queries:
    hits = provider.search(q, max_results=3)
    print(f"\n=== QUERY: {q} ===")
    for h in hits:
        print(f"- {h.title} | {h.url}")
        print(f"  {h.snippet[:140]}...")
