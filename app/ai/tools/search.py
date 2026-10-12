import re
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage

from app.ai.providers.search import get_search_provider
from app.ai.providers.llm import get_llm
from app.ai.enums import SearchProvider, LLMProvider
from app.ai.settings import get_ai_settings


from app.schemas.common import canonicalize_url


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
    ai_settings = get_ai_settings()
    provider_enum = ai_settings.search_provider
    search_provider = get_search_provider(provider_enum)
    
    try:
        results = search_provider.search(query, max_results=3)
        if not results:
            return "No results found for the query."
        
        raw_text = "\n\n".join([
            f"Title: {res.title}\nURL: {canonicalize_url(res.url)}\nContent: {res.snippet}"
            for res in results
        ])
        
        # Use the configured fast LLM to summarize the findings intelligently
        fast_llm = get_llm().chat
        
        is_ar_query = bool(re.search(r"[\u0600-\u06FF]", query))
        lang_note = " IMPORTANT: Extract and summarize the facts strictly in Arabic." if is_ar_query else ""
        sys_msg = SystemMessage(content=(
            "You are an extraction assistant. Extract strictly the numbers, facts, and key details "
            f"from the following search results that directly answer the query.{lang_note} "
            "Keep it under 150 words. Do not add conversational filler."
        ))
        usr_msg = HumanMessage(content=f"Query: {query}\n\nSearch Results:\n{raw_text}")
        
        summary_response = fast_llm.invoke([sys_msg, usr_msg])
        sources = [
            f"- [{res.title}]({canonicalize_url(res.url)})" if res.url else f"- {res.title}"
            for res in results
        ]
        sources_text = "\n".join(sources)
        return f"Search Summary for '{query}':\n{summary_response.content}\n\nSources:\n{sources_text}"
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"
