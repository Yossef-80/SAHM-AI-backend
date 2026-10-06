from typing import Optional
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
import requests
import trafilatura
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

from app.ai.settings import get_ai_settings

def normalize_url(url: str) -> str:
    if not url:
        return ""
    url = str(url).strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url.split("#")[0]

@tool
def analyze_website_tool(url: str, config: RunnableConfig) -> str:
    """
    Fetch and analyze the raw text content of a specific website URL.
    Use this tool when you need to read a competitor's website, an article,
    or verify specific details from a URL you found during a web search.
    
    Args:
        url: The full URL to fetch and analyze (e.g., https://example.com/pricing).
    """
    settings = get_ai_settings()
    target_url = normalize_url(url)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        r = requests.get(
            target_url,
            headers=headers,
            timeout=settings.website_timeout_seconds,
            allow_redirects=True,
        )
        if r.status_code >= 400:
            return f"Error: Received HTTP {r.status_code} when fetching {target_url}."
            
        # Try trafilatura first for clean article extraction
        extracted = trafilatura.extract(
            r.text,
            include_links=True,
            include_tables=True,
        )
        
        # Fallback to BeautifulSoup if trafilatura fails to find main content
        if not extracted:
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()
            extracted = soup.get_text(" ", strip=True)
            
        if not extracted:
            return f"Error: Could not extract any readable text from {target_url}."
            
        # Truncate to avoid context window explosion
        max_chars = settings.website_text_chars
        content = (extracted or "")[:max_chars]
        
        return f"Content from {r.url}:\n\n{content}"
        
    except requests.exceptions.RequestException as e:
        return f"Error connecting to {target_url}: {str(e)}"
    except Exception as e:
        return f"Unexpected error parsing {target_url}: {str(e)}"
