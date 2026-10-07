"""Nour — Researcher Agent Demo (Clean, Modular & Easy to Debug).

Demonstrates the complete end-to-end flow:
1. Environment & Backend Configuration (.env)
2. Live Web Search via Tavily (retrieves real market evidence)
3. Remote Ollama reasoning over ngrok (same proven method as ngorck_request.py)
4. Structured research report output (Pydantic validation)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add backend root to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import requests
    from pydantic import BaseModel, Field
    from app.config import get_settings
except ModuleNotFoundError as err:
    print(f"\n[ERROR] Missing dependency: {err}")
    print("\nPlease run the demo using the backend virtual environment:")
    print("   uv run python scripts/demo_nour.py")
    print("OR:")
    print(r"   .\.venv\Scripts\python.exe scripts/demo_nour.py")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Structured Output Schema (Pydantic)
# -----------------------------------------------------------------------------
class Competitor(BaseModel):
    name: str = Field(description="Brand or company name")
    positioning: str = Field(default="", description="Market positioning or niche")
    strengths: list[str] = Field(default_factory=list, description="Key competitive strengths")
    weaknesses: list[str] = Field(default_factory=list, description="Observed weaknesses or gaps")


class CompetitorResearchReport(BaseModel):
    summary: str = Field(description="High-level market overview")
    competitors: list[Competitor] = Field(default_factory=list, description="Identified competitors")
    common_patterns: list[str] = Field(default_factory=list, description="Shared market patterns")
    differentiation_opportunities: list[str] = Field(default_factory=list, description="Opportunities to win")


# -----------------------------------------------------------------------------
# Modular Step 1: Tavily Live Search
# -----------------------------------------------------------------------------
def search_tavily(query: str, api_key: str, max_results: int = 4) -> list[dict[str, str]]:
    """Fetch live web evidence using the Tavily Search API."""
    print(f"\n>>> [STEP 1] Searching web via Tavily for: '{query}'")
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    
    results = []
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")[:200],
        })
    
    print(f"    Done in {time.time()-t0:.1f}s. Found {len(results)} results:")
    for idx, item in enumerate(results, 1):
        print(f"    [{idx}] {item['title']}")
        print(f"        URL: {item['url']}")
        print(f"        Snippet: {item['snippet'][:100]}...")
    
    return results


# -----------------------------------------------------------------------------
# Modular Step 2: Remote Ollama Inference over Ngrok
# -----------------------------------------------------------------------------
def ask_nour_ollama(
    user_query: str,
    evidence_list: list[dict[str, str]],
    base_url: str,
    model: str,
    timeout: int = 180,
) -> str:
    """Send retrieved evidence to Ollama over ngrok (same logic as ngorck_request.py)."""
    print(f"\n>>> [STEP 2] Sending evidence to Nour via Ollama/ngrok ({model})")

    # Format the evidence clearly and concisely
    formatted_evidence = "\n\n".join(
        f"- Brand: {e['title']}\n  Source: {e['url']}\n  Notes: {e['snippet']}"
        for e in evidence_list
    )

    prompt = (
        "You are Nour, an expert AI Market Researcher.\n"
        "Analyze the following retrieved market evidence for the query.\n\n"
        f"USER QUERY: {user_query}\n\n"
        f"MARKET EVIDENCE:\n{formatted_evidence}\n\n"
        "Return a JSON object with this exact structure:\n"
        "{\n"
        '  "summary": "overview of the competitive market",\n'
        '  "competitors": [\n'
        '    {"name": "...", "positioning": "...", "strengths": ["..."], "weaknesses": ["..."]}\n'
        "  ],\n"
        '  "common_patterns": ["..."],\n'
        '  "differentiation_opportunities": ["..."]\n'
        "}\n\n"
        "Return ONLY the JSON object. Do not add markdown or conversational filler."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1000,
        },
    }

    headers = {
        "ngrok-skip-browser-warning": "true",
    }

    endpoint = f"{base_url.rstrip('/')}/api/chat"
    print(f"    Target endpoint: {endpoint}")
    print(f"    Waiting for model reasoning (timeout={timeout}s)...", flush=True)

    t0 = time.time()
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        print(f"\n[ERROR] Request timed out after {timeout} seconds.")
        print("Tip: Google Colab models take longer when loading for the first time.")
        sys.exit(1)
    except requests.exceptions.SSLError as ssl_err:
        print(f"\n[ERROR] SSL Connection error: {ssl_err}")
        print("Tip: Verify your ngrok URL is active in Google Colab.")
        sys.exit(1)

    if response.status_code != 200:
        print(f"\n[ERROR] Ollama returned HTTP {response.status_code}: {response.text}")
        sys.exit(1)

    data = response.json()
    elapsed = time.time() - t0
    content = data.get("message", {}).get("content", "")
    print(f"    Response received in {elapsed:.1f}s!")
    return content


# -----------------------------------------------------------------------------
# Modular Step 3: Validate and Display Structured Report
# -----------------------------------------------------------------------------
def validate_and_display(raw_text: str) -> None:
    """Parse JSON and validate against the Pydantic model."""
    print("\n>>> [STEP 3] Validating Structured Research Report")

    # Extract JSON even if the model enclosed it in markdown code fences
    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        parsed_dict = json.loads(cleaned)
        report = CompetitorResearchReport.model_validate(parsed_dict)
    except Exception as e:
        print(f"    [WARNING] Model did not return pure JSON ({e}). Displaying raw content:")
        print("-" * 60)
        print(raw_text)
        print("-" * 60)
        return

    print("\n" + "=" * 70)
    print(" NOUR'S COMPETITIVE RESEARCH REPORT (VALIDATED)")
    print("=" * 70)
    print(f"\n📌 SUMMARY:\n{report.summary}")

    print("\n🏢 IDENTIFIED COMPETITORS:")
    for c in report.competitors:
        print(f"\n  • {c.name}")
        if c.positioning:
            print(f"    - Positioning: {c.positioning}")
        if c.strengths:
            print(f"    - Strengths:   {', '.join(c.strengths)}")
        if c.weaknesses:
            print(f"    - Weaknesses:  {', '.join(c.weaknesses)}")

    if report.common_patterns:
        print("\n📈 COMMON MARKET PATTERNS:")
        for p in report.common_patterns:
            print(f"  • {p}")

    if report.differentiation_opportunities:
        print("\n💡 DIFFERENTIATION OPPORTUNITIES:")
        for opp in report.differentiation_opportunities:
            print(f"  • {opp}")

    print("\n" + "=" * 70)
    print(" DEMO COMPLETED SUCCESSFULLY — NOUR IS OPERATIONAL!")
    print("=" * 70)


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print(" SAHM AI MARKETING OS — NOUR (RESEARCHER AGENT) DEMO")
    print("=" * 70)

    settings = get_settings()

    query = sys.argv[1] if len(sys.argv) > 1 else "specialty coffee brands in Egypt"

    print(f"\nConfiguration:")
    print(f"  - Tavily Key Present: {'Yes' if settings.tavily_api_key else 'No'}")
    print(f"  - Ollama Base URL:    {settings.ollama_base_url or 'Missing'}")
    print(f"  - Ollama Model:       {settings.ollama_model or 'Missing'}")
    print(f"  - Timeout:            {settings.ollama_timeout_seconds}s")

    if not settings.tavily_api_key:
        print("\n[ERROR] TAVILY_API_KEY is not configured in .env.")
        sys.exit(1)

    if not settings.ollama_base_url or not settings.ollama_model:
        print("\n[ERROR] OLLAMA_BASE_URL or OLLAMA_MODEL is not configured in .env.")
        sys.exit(1)

    # 1. Search
    evidence = search_tavily(query, settings.tavily_api_key)

    # 2. Reasoning
    raw_response = ask_nour_ollama(
        user_query=query,
        evidence_list=evidence,
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout_seconds,
    )

    # 3. Validation & Display
    validate_and_display(raw_response)


if __name__ == "__main__":
    main()
