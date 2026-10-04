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
    This tool should be used when you need to find up-to-date information,
    news, market trends, or external data that is not in your memory.
    
    Args:
        query: The search query string.
    """
    # The search provider is initialized based on the AI settings
    ai_settings = get_ai_settings()
    provider_enum = ai_settings.search_provider
    search_provider = get_search_provider(provider_enum)
    
    try:
        results = search_provider.search(query)
        if not results:
            return "No results found for the query."
        return results
    except Exception as e:
        return f"Error performing web search: {str(e)}"
