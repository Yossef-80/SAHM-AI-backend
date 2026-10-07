# Sahm AI Tools Documentation

This document serves as the structured reference for how tools are called, what inputs they take, and what outputs they produce across all 5 agents.

## 1. Research & Analysis Tools (Used by Nour & Analyst)

### `web_search_tool`
- **Purpose:** Searches the web using DuckDuckGo or Tavily.
- **Input:** 
  - `query` (str): The search query to execute.
- **Output:** A JSON string containing a list of `SearchResult` objects (title, URL, snippet).

### `analyze_website_tool`
- **Purpose:** Scrapes and extracts readable text from a given URL.
- **Input:** 
  - `url` (str): The target URL.
- **Output:** Cleaned plain text content of the website (truncated to prevent context overflow).

### `brand_report` (MCP Tool)
- **Purpose:** Collects metadata, social media posts, ads, and maps reviews into a single JSON profile saved on disk.
- **Input:**
  - `brand` (str): Name of the brand (e.g., "KFC Egypt").
  - `url` (str): Brand's website.
  - `country` (str): Name of the country.
  - `country_code` (str): 2-letter ISO code (e.g., "EG").
- **Output:** JSON summary of the collected assets (e.g., "collected 50 reviews, 20 posts").

### `full_analysis` (MCP Tool)
- **Purpose:** Runs AI-driven sentiment analysis, engagement metrics, and product insights extraction on the data collected by `brand_report`.
- **Input:**
  - `brand` (str): The name of the brand previously collected.
- **Output:** JSON mapping of happiest products, top complaints, star distribution, and engagement percentage.

### `get_reviews_tool`
- **Purpose:** Fetches raw, recent Google Maps reviews for a specific business location.
- **Input:**
  - `business_name` (str): e.g., "McDonald's"
  - `location` (str): e.g., "Cairo, Egypt"
  - `provider` (str): "auto" or "apify"
  - `max_reviews` (int): 5-200.
- **Output:** A JSON array of reviews (stars, text, timestamp).

### `route_file_to_tool` (MCP Tool)
- **Purpose:** Reads a local data file and determines the best tool to analyze it.
- **Input:**
  - `file_path` (str): Path to the file.
- **Output:** JSON recommendation for the next tool to call (e.g., "Call analyze_sentiment").

---

## 2. Core Context Tools (Used by All Agents)

### `get_brand_context_tool`
- **Purpose:** Fetches the overarching business profile, target audience, and brand voice guidelines.
- **Input:**
  - `business_id` (str): Automatically extracted from the context.
- **Output:** A structured profile containing `brand_voice`, `target_audience`, `products`, etc.

---

## 3. Submission Tools (Agent Graph Endpoints)

*To prevent infinite loops, each agent ends its graph by calling a dedicated submission tool that outputs structured JSON conforming to Pydantic schemas.*

### `submit_research_report` (Nour)
- **Input:** `market_overview`, `competitor_summary`, `pricing_insights`, `opportunities_for_target_brand`, etc.
- **Output:** "REPORT_SUBMITTED_SUCCESSFULLY"

### `submit_strategy_report` (Omar)
- **Input:** `summary`, `details` (representing channels, content calendar frameworks, budget splits).
- **Output:** "REPORT_SUBMITTED_SUCCESSFULLY"

### `submit_creative_brief` (Layla)
- **Input:** `summary`, `details` (representing ad copy, visual direction, scripts).
- **Output:** "REPORT_SUBMITTED_SUCCESSFULLY"

### `submit_performance_report` (Karim)
- **Input:** `summary`, `details` (representing CTR, ROAS, fatigue warnings).
- **Output:** "REPORT_SUBMITTED_SUCCESSFULLY"

### `submit_optimization_plan` (Salma)
- **Input:** `summary`, `details` (representing bid adjustments, audience tweaks).
- **Output:** "REPORT_SUBMITTED_SUCCESSFULLY"
