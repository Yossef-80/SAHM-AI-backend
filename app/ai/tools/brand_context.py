from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

from app.ai.memory.brand_memory import get_brand_profile, business_id_from, learnings_namespace
from app.ai.providers.db import get_store

@tool
def get_brand_context_tool(config: RunnableConfig) -> str:
    """
    Retrieve the long-term context and profile of the brand/business.
    Use this tool to understand the brand's voice, target audience, 
    key values, and previous learnings before generating content.
    
    This tool requires no arguments. It automatically knows which brand 
    to look up based on the current session context.
    """
    # Extract business_id from the configurable parameters passed by the graph
    business_id = business_id_from(config)
        
    store = get_store()
    
    try:
        profile = get_brand_profile(store, business_id)
        if not profile:
            return f"No brand profile found for business ID {business_id}."
            
        # Format the profile into a readable string for the LLM
        context = f"Brand Profile for Business {business_id}:\n"
        context += f"Name: {profile.name}\n"
        context += f"Industry: {profile.industry}\n"
        context += f"Tone/Voice: {profile.tone}\n"
        context += f"Audience: {profile.audience}\n"
        context += f"Values/Description: {profile.description}\n"
        
        # We can also pull learnings
        items = store.search(learnings_namespace(business_id))
        learnings = [item.value for item in items]
        if learnings:
            context += "\nKey Learnings from past interactions:\n"
            for l in learnings:
                context += f"- {l.get('insight', '')}\n"
                
        return context
    except Exception as e:
        return f"Error retrieving brand context: {str(e)}"
