from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

from app.ai.providers.search import get_search_provider
from app.ai.enums import SearchProvider
from app.ai.settings import get_ai_settings


@tool
def web_search_tool(query: str, config: RunnableConfig) -> str:
    """
    Perform a web search using the configured search provider.
    
    CRITICAL SEARCH GUIDELINES:
    1. Be highly specific with your queries. Avoid generic searches.
    2. If looking for competitors, include the industry and region (e.g., "men's classic suits competitors Egypt").
    3. Do not search for generic business directories unless necessary.
    4. If you need deep information, make multiple targeted searches rather than one broad one.
    
    Args:
        query: The highly specific search query string.
    """
    # The search provider is initialized based on the AI settings
    ai_settings = get_ai_settings()
    provider_enum = ai_settings.search_provider
    search_provider = get_search_provider(provider_enum)
    
    try:
        # Enforce strict maximum results limit to save tokens
        results = search_provider.search(query, max_results=2)
        if not results:
            return "No results found for the query."
        
        # Summarize tool outputs to significantly reduce token usage
        summarized = []
        for i, res in enumerate(results):
            summarized.append(f"Result {i+1}: {res.title} - {res.snippet[:200]}")
            
        return "\n\n".join(summarized)
    except Exception as e:
        return f"Error performing web search: {str(e)}"
