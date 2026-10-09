# 🛠️ SAHM Marketing OS — Tools & Agents Improvement Guide

> **Audience:** Developers, AI engineers, and contributors tasked with improving **Tools** and **Agents** in SAHM AI.  
> **Purpose:** A complete, hands-on engineering playbook. You can hand this document directly to an AI (Claude, ChatGPT, Gemini) to collaborate on optimizing any tool or agent.

---

## 1. Quick Orientation (Tools & Agents in SAHM)

In SAHM AI:
- **Tools** are the sensory organs and action levers of the system. They search the live web, scrape landing pages, analyze ad creatives, extract customer reviews, and distill market opportunities.
- **Agents** are the specialized personas (Nour, Omar, Karim, Layla, Salma, Orchestrator) that interact with the user, synthesize tool data, make marketing decisions, and draft campaigns.

### The Two Ways Tools Are Used:
1. **Direct Execution (The Research Tab):** The user clicks "Run" on a tool card (e.g., Competitor research on *"Tie House"*). The tool runs directly through `app/api/research.py`, verifies the entity, searches live web data, creates a comprehensive report for the UI, and distills a ~100-token memory item into the brand database.
2. **Autonomous Invocation (The Assistant Chat):** A user asks Nour or Salma: *"Who is advertising against our luxury coffee brand in Cairo?"* The agent selects the appropriate tool, calls it, receives the clean output, and weaves it into its conversational response.

---

## 2. Developer Boundaries: What Files Can You Edit?

To protect database schemas, multi-tenant workspace isolation, and authentication, strictly adhere to these boundaries:

```
+---------------------------------------------------------------------------------------+
|  🟢 SAFE TO EDIT (Your Daily Workspace)                                               |
|  -----------------------------------------------------------------------------------  |
|  • SAHM-AI-backend/app/ai/tools/*.py          (Tool implementations & scrapers)       |
|  • SAHM-AI-backend/app/ai/agents/*/*.py       (Agent prompts, personas & tool wiring) |
|  • SAHM-AI-backend/app/api/research.py        (Research prompts, validation, storage) |
|  • SAHM-AI-frontend/src/components/research/  (Tool inputs, cards, Markdown renderer) |
+---------------------------------------------------------------------------------------+
|  🟡 EDIT WITH CAUTION (Coordinate with backend lead)                                  |
|  -----------------------------------------------------------------------------------  |
|  • SAHM-AI-backend/app/api/chat.py            (Chat routing & streaming logic)        |
|  • SAHM-AI-backend/app/ai/providers/*.py      (Search & LLM model adapters)           |
+---------------------------------------------------------------------------------------+
|  🔴 DO NOT EDIT (System Core & Security)                                              |
|  -----------------------------------------------------------------------------------  |
|  • app/db/models.py                           (Requires DB migrations)                |
|  • app/db/session.py                          (Postgres pooler & session manager)     |
|  • app/api/deps.py                            (Multi-tenant X-Workspace-Id resolver)  |
|  • app/config.py                              (Global environment schema)             |
+---------------------------------------------------------------------------------------+
```

---

## 3. What Files to Give to an AI for Context

When asking an AI (Claude, GPT-4, etc.) to help you improve or build a feature, **do not dump the entire repository**. Give it only the relevant files:

### Scenario A: Improving a Research Tool (e.g., Competitor Ads, Reviews, Keywords)
Give the AI:
1. This file: `TEAM_TOOLS_AND_AGENTS_GUIDE.md` (for rules and memory contracts)
2. `SAHM-AI-backend/app/api/research.py` (the execution endpoint & prompt catalog)
3. The specific tool file:
   - For website scraping: `app/ai/tools/analyze_website.py`
   - For search & scraping: `app/ai/tools/search.py`
   - For reviews: `app/ai/tools/get_reviews.py`
   - For brand context: `app/ai/tools/brand_context.py`

### Scenario B: Improving an Agent (e.g., Nour, Omar, Salma)
Give the AI:
1. This file: `TEAM_TOOLS_AND_AGENTS_GUIDE.md`
2. `SAHM-AI-backend/app/ai/agents/<agent_name>/agent.py`
3. `SAHM-AI-backend/app/ai/agents/<agent_name>/prompts.py`
4. `SAHM-AI-backend/app/api/chat.py`

---

## 4. The 5-Stage Life Cycle of a SAHM Tool

Every tool must adhere to this standardized architecture:

```
[User Input / Query]
       │
       ▼
1. Validation & Ground-Truth Verification
   (Checks if entity exists; fixes typos like "Tiee Hose" -> "Tie House"; rejects gibberish)
       │
       ▼
2. Live Data Gathering (SERP / Scrape / API)
   (Collects live web snippets, page content, or external ad library links with real URLs)
       │
       ▼
3. Structured Synthesis (LLM)
   (Generates clean Markdown: 300-380 words max, dense bullet points, clickable source links)
       │
       ▼
4. Dual-Layer Storage (Preventing Token Bloat)
   ├── A. BrandMemoryItem.text      ──> Distilled by fast_llm to 80-120 words (~100 tokens)
   ├── B. BrandMemoryItem.metadata_ ──> Full 380-word report stored under "full_report"
   └── C. Profile.user_context_summary ──> Long-term memory updated (<300 tokens total)
       │
       ▼
5. Output Presentation (UI & Agent Response)
   ├── UI: Full report rendered via MarkdownReport (h2, h3, h4, lists, source link cards)
   └── Agent: Compact fact retrieved on future chats without token overflow
```

---

## 5. Detailed Catalog of All 9 Research Tools

### Tool 1: Website Analyzer (`analyze_website`)
- **Card Title in UI:** Website analyzer
- **Input Needed:** `url` (e.g., `https://yourbrand.com` or competitor URL).
- **Files to Edit:**
  - `SAHM-AI-backend/app/ai/tools/analyze_website.py`
  - `SAHM-AI-backend/app/api/research.py` (prompt in `TOOL_PROMPTS["analyze_website"]`)
- **Execution Mechanism:**
  - Fetches the webpage HTML using `httpx`.
  - Cleans HTML via BeautifulSoup, extracting title, meta description, H1/H2 tags, pricing cues, and body text.
  - LLM evaluates brand positioning, value propositions, trust signals, and CRO friction points.
- **Output Structure:**
  - `## Overview & Brand Positioning`
  - `## Core Offers & Value Propositions`
  - `## Trust Signals & Social Proof`
  - `## Conversion Friction & Gaps`
  - `## Actionable Recommendations for Growth`
  - `## 🌐 Sources & References`
- **User Experience Impact:** Users paste their own URL or a competitor's URL and immediately discover why visitors bounce and how to fix their landing page copy.
- **Marketing Result Impact:** Direct lift in Conversion Rate (CVR) and clearer Unique Value Proposition (UVP).
- **Improvement Ideas to Discuss with AI:**
  - Add extraction of OpenGraph tags and schema.org microdata (e.g., Product pricing, ratings).
  - Add page speed / mobile viewport responsiveness heuristics.

---

### Tool 2: Competitor Research (`analyze_competitor`)
- **Card Title in UI:** Competitor research
- **Input Needed:** `competitor` name or category (e.g., `"Tie House"` or `"specialty coffee cairo"`). Optional; falls back to category context.
- **Files to Edit:**
  - `SAHM-AI-backend/app/api/research.py` (`TOOL_PROMPTS["analyze_competitor"]`)
  - `SAHM-AI-backend/app/ai/providers/search.py`
- **Execution Mechanism:**
  - Passes through `verify_and_resolve_entity` to ensure the brand is real and correct misspellings.
  - Queries search engine: `"{brand} egypt brand store products positioning reviews"`.
  - Analyzes market share, target demographic, promotional bundles, and vulnerabilities.
- **Output Structure:**
  - `## Competitor Overview & Market Share in Egypt`
  - `## Target Audience & Customer Demographics`
  - `## Advertising Angles, Hooks & Campaigns`
  - `## Pricing Strategy & Promotional Bundles`
  - `## Competitor Vulnerabilities & How We Outcompete Them`
  - `## 🌐 Sources & References` (Clickable links)
- **User Experience Impact:** Replaces hours of manual competitor browsing with an actionable counter-positioning brief.
- **Marketing Result Impact:** Enables competitive ad campaigns that exploit competitor price weaknesses and service gaps.

---

### Tool 3: Social Profile Analyzer (`analyze_social_profile`)
- **Card Title in UI:** Social profile
- **Input Needed:** `handle` (e.g., `@yourbrand` or `@competitorbrand`).
- **Files to Edit:**
  - `SAHM-AI-backend/app/api/research.py` (`TOOL_PROMPTS["analyze_social_profile"]`)
  - `SAHM-AI-backend/app/core/tools/connectors.py`
- **Execution Mechanism:**
  - Searches for recent public social posts, bio text, Reels engagement patterns, and comment sentiment.
- **Output Structure:**
  - `## Core Content Pillars & Themes`
  - `## Engagement Dynamics & Top-Performing Formats (Reels vs Carousels)`
  - `## Brand Tone of Voice & Visual Aesthetic`
  - `## Audience Reception & Comment Sentiment`
  - `## 'What to Post More Of' Strategic Action Items`
  - `## 🌐 Sources & References`
- **User Experience Impact:** Gives founders and content creators a clear roadmap of high-engagement content topics.
- **Marketing Result Impact:** Higher organic reach, reduced cost per follower, and improved social engagement rates.

---

### Tool 4: Reviews & Sentiment (`analyze_reviews`)
- **Card Title in UI:** Reviews
- **Input Needed:** `url` or brand name (e.g., Google Maps link, Trustpilot URL, or store link). Optional.
- **Files to Edit:**
  - `SAHM-AI-backend/app/ai/tools/get_reviews.py`
  - `SAHM-AI-backend/app/api/research.py` (`TOOL_PROMPTS["analyze_reviews"]`)
- **Execution Mechanism:**
  - Searches Google reviews, Facebook page reviews, and consumer forum discussions.
  - Synthesizes recurring emotional themes and complaints.
- **Output Structure:**
  - `## Overall Sentiment Score & Review Trends`
  - `## Top Customer Praises (What customers love)`
  - `## Top Customer Complaints & Friction Points (What hurts sales)`
  - `## Representative Customer Quotes`
  - `## Strategic Fixes & Positioning Opportunities`
  - `## 🌐 Sources & References`
- **User Experience Impact:** Highlights the exact vocabulary and emotional pain points customers express in their own words.
- **Marketing Result Impact:** Copywriters use actual customer quotes in ad hooks, drastically improving Click-Through Rate (CTR).

---

### Tool 5: Market Analysis (`analyze_market`)
- **Card Title in UI:** Market analysis
- **Input Needed:** Category or question (e.g., `"ready-to-drink coffee egypt"`). Optional.
- **Files to Edit:**
  - `SAHM-AI-backend/app/api/research.py` (`TOOL_PROMPTS["analyze_market"]`)
- **Execution Mechanism:**
  - Searches recent market reports, Egyptian economic news, and consumer spending data.
- **Output Structure:**
  - `## Market Size & 2026 Growth Trajectory in Egypt`
  - `## Key Consumer Segments & Spending Behavior`
  - `## Emerging Category Trends & Consumer Shifts`
  - `## Economic Realities (Currency, imports, price elasticity)`
  - `## High-Growth Whitespaces & Category Opportunities`
  - `## 🌐 Sources & References`
- **User Experience Impact:** Provides macro-level clarity on market sizing, consumer price sensitivity, and category trends.
- **Marketing Result Impact:** Prevents businesses from launching products in shrinking categories; identifies emerging high-growth niches.

---

### Tool 6: Keyword Clusters (`analyze_keywords`)
- **Card Title in UI:** Keywords
- **Input Needed:** `seed` keyword (e.g., `"cold brew cairo"`).
- **Files to Edit:**
  - `SAHM-AI-backend/app/api/research.py` (`TOOL_PROMPTS["analyze_keywords"]`)
- **Execution Mechanism:**
  - Analyzes search intent, clustering keywords into Transactional, Informational, and Long-Tail groups.
- **Output Structure:**
  - `## High-Intent Transactional Keyword Clusters (High buying intent)`
  - `## Informational & Education Keyword Clusters (Research stage)`
  - `## Estimated Search Demand & Competition Intensity`
  - `## Long-Tail Search Queries & Question Clusters`
  - `## Recommended Content Silos & Target Pages`
  - `## 🌐 Sources & References`
- **User Experience Impact:** Organizes search terms by intent so the user knows what to target in Google Ads vs. SEO blog posts.
- **Marketing Result Impact:** Lowers Google Search Ad Cost Per Click (CPC) and builds organic search traffic.

---

### Tool 7: Competitor Creatives & Ads (`analyze_competitor_creatives`)
- **Card Title in UI:** Competitor ads
- **Input Needed:** `competitor` name (e.g., `"Town Team"` or `"Tie House"`). Optional.
- **Files to Edit:**
  - `SAHM-AI-backend/app/api/research.py` (`TOOL_PROMPTS["analyze_competitor_creatives"]`)
- **Execution Mechanism:**
  - Generates live deep-link to Meta Ad Library Egypt for the brand.
  - Analyzes competitor visual patterns, 3-second opening video hooks, promotional offers, and copywriting psychology.
  - **Crucial Rule:** Does *not* write generic corporate histories; focuses 100% on ad creatives and psychology.
- **Output Structure:**
  - `## 🎬 Long-Running Creative Concepts & Core Ad Angles`
  - `## 🪝 3-Second Hook Analysis (Visual & Headline Hooks)`
  - `## 📱 Creative Formats & Production Styles Breakdown (UGC vs Studio)`
  - `## 🏷️ Promotional Offers, Hooks & Call-to-Actions (CTAs)`
  - `## 💡 How to Out-Create Them (3 Actionable Ad Concepts for Our Brand)`
  - `## 🌐 Sources & References` (Includes Meta Ad Library link)
- **User Experience Impact:** Users immediately see the exact ad styles and opening hooks their competitors are spending budget on.
- **Marketing Result Impact:** Direct increase in Thumb-Stop Ratio (3-second video views) and ROAS.

---

### Tool 8: Market Gaps (`detect_market_gaps`)
- **Card Title in UI:** Market gaps
- **Input Needed:** Optional (synthesizes all research stored so far in Brand Memory).
- **Files to Edit:**
  - `SAHM-AI-backend/app/api/research.py` (`TOOL_PROMPTS["detect_market_gaps"]`)
- **Execution Mechanism:**
  - Queries recent `BrandMemoryItem` rows (competitor analyses, review sentiment, website audits) and identifies unserved customer frustrations and pricing gaps.
- **Output Structure:**
  - `## Executive Opportunity Summary`
  - `## Unserved Customer Needs & Frustrations in the Category`
  - `## Pricing Gaps (Missing luxury or quality mid-tier options)`
  - `## Product, Quality & Service Whitespaces`
  - `## Ranked Opportunities by Revenue Potential & Feasibility`
  - `## 🌐 Sources & References`
- **User Experience Impact:** Identifies high-margin product or service opportunities that competitors ignore.
- **Marketing Result Impact:** Unlocks new revenue streams and defensible product-market fit.

---

### Tool 9: Marketing Strategy (`generate_marketing_strategy`)
- **Card Title in UI:** Marketing strategy
- **Input Needed:** Notes / focus areas (optional).
- **Files to Edit:**
  - `SAHM-AI-backend/app/api/research.py` (`TOOL_PROMPTS["generate_marketing_strategy"]`)
- **Execution Mechanism:**
  - Acts as a Fractional CMO, pulling brand context and market research to construct a 90-day execution blueprint.
- **Output Structure:**
  - `## Core Brand Positioning & The Unfair Advantage Hook`
  - `## 90-Day Execution Roadmap (Month 1: Foundation, Month 2: Scale, Month 3: Optimize)`
  - `## Channel Allocation & Budget Strategy (Meta, TikTok, Google, WhatsApp)`
  - `## Key Metrics & Targets (CAC, ROAS, Target Order Volume)`
  - `## Immediate Top 3 Action Items for This Week`
  - `## 🌐 Sources & References`
- **User Experience Impact:** Transforms fragmented marketing tactics into a cohesive, prioritized 90-day plan.
- **Marketing Result Impact:** Eliminates wasted ad spend and aligns budget with profitable channels.

---

## 6. The Memory Distillation Contract (Preventing Token Bloat)

Whenever you add or modify a tool in `app/api/research.py`, **you must preserve this distillation pattern**:

```python
# 1. Distill the 380-word report to an 80-120 word memory note
distill_sys = SystemMessage(content=(
    "You are an AI Memory Distiller for SAHM OS.\n"
    "Distill this research report into an ultra-compact brand memory note for AI marketing agents.\n"
    "STRICT CONSTRAINT: Exactly 80 to 120 words maximum.\n"
    "Focus only on key factual intelligence: competitor strengths/weaknesses, audience desires, price points, and winning creative angles.\n"
    "Format as 3-4 dense bullet points. Zero conversational filler."
))
distill_usr = HumanMessage(content=f"Research Title: {title}\nReport:\n{summary_text}")
distill_res = await asyncio.to_thread(fast_llm.invoke, [distill_sys, distill_usr])
compact_memory_text = extract_clean_text(distill_res.content)

# 2. Store compact text in DB column, full report in metadata
mem_item = BrandMemoryItem(
    id=memory_id,
    business_id=ws.business_id,
    kind=kind,
    text=compact_memory_text,             # <-- ONLY ~100 tokens loaded by AI agents!
    metadata_={
        "tool": tool_type,
        "query": query,
        "title": title,
        "full_report": summary_text,       # <-- Full report preserved for UI display!
        "details": {...},
    },
)
db.add(mem_item)
```

---

## 7. How to Talk to AI: Ready-to-Use Prompt Templates

Copy and paste these prompts into your AI coding assistant (ChatGPT, Claude, Gemini) when working on tools or agents:

### Prompt Template 1: Improving an Existing Tool
```text
I am working on the SAHM Marketing OS project. Please read TEAM_TOOLS_AND_AGENTS_GUIDE.md for our architectural rules.
I want to improve the tool: [insert tool name, e.g., analyze_competitor_creatives].
Relevant file: SAHM-AI-backend/app/api/research.py.

Goals:
1. Ensure the prompt produces strictly under 380 words of high-density bullet points in clean Markdown.
2. Ground findings in live evidence and include clickable markdown source citations.
3. Preserve the dual-layer memory pattern (compact ~100 tokens in BrandMemoryItem.text, full report in metadata).
4. Eliminate any Python dictionary leaks or raw JSON formatting in the output.

Please review the existing implementation and provide the updated code with rationale.
```

### Prompt Template 2: Adding a Brand New Tool
```text
I am adding a new tool to SAHM Marketing OS: [tool name, e.g., analyze_whatsapp_funnel].
Please follow the guidelines in TEAM_TOOLS_AND_AGENTS_GUIDE.md:
1. Define the Pydantic request schema and tool type.
2. Define the prompt in TOOL_PROMPTS with DENSE_REPORT_PREFIX and structured Markdown headings.
3. Define the live data gathering step (web search / scraping).
4. Implement entity verification & typo handling if applicable.
5. Save the distilled result to BrandMemoryItem with the dual-layer storage pattern.
6. Provide the frontend ToolConfig entry for SAHM-AI-frontend/src/components/research/research-view.tsx.
```

### Prompt Template 3: Tuning an Agent Persona (e.g., Nour, Omar, Salma)
```text
I want to enhance the agent [e.g., Omar the copywriter] in SAHM Marketing OS.
Files: SAHM-AI-backend/app/ai/agents/omar/prompts.py and agent.py.

Requirements:
1. Infuse culturally native Egyptian marketing psychology (e.g., EGP currency considerations, urgency hooks, colloquial Arabic & English).
2. Ensure the agent uses retrieved BrandMemoryItems effectively without repeating old conversations.
3. Ensure the agent provides structured deliverables (Hook, Body, CTA, Visual Concept) rather than generic marketing advice.
```

---

## 8. Verification & Testing Checklist Before Pushing

Before committing any tool or agent changes, run this 4-step checklist:

1. **Verify No Raw Hash `#` Leaks in UI:**
   Ensure headings render as styled HTML (`<h2>`, `<h3>`), not plain text `# Title`.
2. **Verify Memory Item Token Count:**
   Query `BrandMemoryItem.text` in the database and verify it is under 120 words (~100 tokens).
3. **Verify Full Report in Metadata:**
   Check that `BrandMemoryItem.metadata_["full_report"]` contains the complete report and that `GET /api/v1/research/items` returns it to the UI.
4. **Run TypeScript Check:**
   In `SAHM-AI-frontend`, run:
   ```bash
   npx tsc --noEmit
   ```
   Must complete with **0 errors**.
5. **Git Synchronization:**
   Commit to `tools-development`, merge/push to `ahmed` and `main` branches.
