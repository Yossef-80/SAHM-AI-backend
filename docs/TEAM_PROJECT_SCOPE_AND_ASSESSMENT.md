# 🚀 SAHM Marketing OS — Project Scope, Architecture & State Assessment

> **Target Audience:** Engineering team members joining the project who need a complete, ground-up understanding of SAHM AI, its business goals, current implementation status, and multi-agent system.  
> **How to Use:** You can read this document directly or upload/paste it into an AI assistant (Claude, ChatGPT, Gemini) to collaborate on features, architectural improvements, and strategic roadmaps.

---

## 1. Executive Summary & Vision

### What is SAHM AI?
**SAHM AI (سهم)** is an autonomous **Multi-Agent Marketing Operating System** designed primarily for small-to-medium businesses (SMBs), e-commerce brands, and agency operators in Egypt and the MENA region.

### The Problem It Solves
Traditional marketing platforms (Meta Ads Manager, Google Ads, TikTok Ads) are fragmented, complex, and require expensive specialized teams:
1. **High Headcount Cost:** Businesses need media buyers, copywriters, creative designers, SEO analysts, and CRO consultants.
2. **Context Fragmentation:** Ad tools do not understand the brand's unique history, customer reviews, competitor moves, or market dynamics.
3. **Regional Gap (Egypt/MENA):** Most AI marketing tools fail at Arabic/Egyptian dialect nuance, local economic realities (e.g., EGP currency fluctuations, local payment habits, WhatsApp-first conversion funnels).

### The Solution: An Autonomous Marketing Department in a Box
SAHM AI replaces disjointed marketing workflows with a **specialized squad of AI marketing specialists** coordinated by an Orchestrator. These agents research competitors, monitor live web data, write culturally resonant Arabic/English copy, draft visual ad concepts, optimize budget allocations, and remember everything across customer touchpoints.

```
+-----------------------------------------------------------------------------------+
|                                SAHM CLIENT (Next.js)                              |
|   +-----------------------+   +----------------------+   +--------------------+   |
|   |   🧠 Research Hub     |   |   💬 Assistant Chat  |   |   🎨 Creatives &   |   |
|   | (Live Market Intel)   |   | (Multi-Agent Squad)  |   |    Campaign Hub    |   |
|   +-----------+-----------+   +----------+-----------+   +---------+----------+   |
+---------------|--------------------------|-------------------------|--------------+
                |                          |                         |
                v                          v                         v
+-----------------------------------------------------------------------------------+
|                           SAHM BACKEND API (FastAPI)                              |
|                          app/api/research.py & app/api/chat.py                     |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                        CENTRAL BRAIN & MEMORY ENGINE                              |
|   1. Short-Term Memory: Active chat session window                               |
|   2. Brand Memory Items: Compact distilled facts (80-120 words / ~100 tokens)     |
|   3. Profile Long-Term Memory (LTM): user_context_summary (<300 tokens)           |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                     SPECIALIZED MULTI-AGENT SQUAD (LangChain)                    |
|                                                                                   |
|    👑 Orchestrator   --> Routes intents & coordinates agent team                  |
|    🎯 Nour           --> Growth & Marketing Director (Strategy, KPIs, Reviews)    |
|    ✍️  Omar           --> Senior Copywriter (Ad scripts, Hooks, Arabic/Eng copy)   |
|    🎨 Karim          --> Creative Director (Visual concepts, Storyboards, UGC)    |
|    📊 Layla          --> Media Buyer & Quant (Campaign pacing, ROAS, Allocations) |
|    🔍 Salma          --> Market & Competitor Intelligence (Scraping, SERP, Intel) |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Pillars & Value Streams

1. **Autonomous Intelligence (Research Tab):**
   - Live web search and URL scraping.
   - Ground-truth entity validation (corrects typos like `"Tiee Hose"` → `"Tie House"`, rejects non-existent businesses).
   - Generates structured market intelligence reports with clickable web source citations.
2. **Context-Aware Team Collaboration (Assistant Chat):**
   - Natural conversation with Nour, Omar, Karim, Layla, or Salma.
   - The Orchestrator automatically selects the best specialist or delegates sub-tasks.
   - Every agent accesses the shared brand knowledge base before making recommendations.
3. **Token-Optimized Brand Memory:**
   - Dual-layer storage prevents context window overflow.
   - Detailed reports are preserved for human inspection in metadata, while distilled ~100-token summaries are injected into agent prompts.
4. **Actionable Deliverables:**
   - Ad scripts with 3-second visual hooks.
   - 90-day marketing roadmaps.
   - Competitor counter-positioning matrices.
   - Keyword cluster maps.

---

## 3. Architecture Assessment (Current State)

### What Is Built and Fully Working ✅

| Component | File Path | Current Status | Notes / Capabilities |
| :--- | :--- | :--- | :--- |
| **Research Hub API** | `app/api/research.py` | **100% Operational** | 9 live research tools with real web search, scraping, entity typo verification, and dual-layer memory persistence. |
| **Ground-Truth Engine** | `app/api/research.py` (`verify_and_resolve_entity`) | **100% Operational** | Prevents hallucinations on fake queries; auto-corrects misspellings of Egyptian/regional brands. |
| **Memory Distiller** | `app/api/research.py` | **100% Operational** | Distills research reports to 80–120 words (~100 tokens) in `BrandMemoryItem.text`; saves full report in metadata. |
| **Markdown Parser** | `SAHM-AI-frontend/.../research-view.tsx` | **100% Operational** | Renders `#` through `######` cleanly into styled HTML; zero raw hash leaks; clickable link cards. |
| **Multi-Agent Squad** | `app/ai/agents/` | **Operational** | Nour, Omar, Karim, Layla, Salma, and Orchestrator have defined system prompts, personas, and tool bindings. |
| **Live Search Tooling** | `app/ai/tools/search.py`, `app/ai/tools/analyze_website.py` | **Operational** | DuckDuckGo / Tavily web search integration, live HTML extraction, and metadata parsing. |
| **Brand Memory Store** | `app/db/models.py` (`BrandMemoryItem`, `Profile`) | **Operational** | Multi-tenant isolation by `business_id`; supports embedding vector search or SQL fallback. |
| **Frontend UI** | `SAHM-AI-frontend/` | **Operational** | Next.js 14 App Router, Tailwind CSS, shadcn/ui components, responsive layout. |

### Current Architectural Gaps & Improvement Opportunities ⚠️

1. **Tool Invocation Parity Between Research Tab & Chat Agents:**
   - *Current Reality:* The Research tab (`app/api/research.py`) executes tools directly via dedicated REST endpoints with entity verification and web citations.
   - *Improvement Opportunity:* Chat agents (e.g., Nour, Salma) should call these exact same tool pipelines with streaming status updates so users can say: *"Salma, research Tie House ads"* and receive the exact same high-fidelity report in the chat thread.
2. **Real Social API Connectors:**
   - *Current Reality:* `analyze_social_profile` gathers evidence via search engine indexation (`site:instagram.com`).
   - *Improvement Opportunity:* Connect official Instagram Graph API / TikTok API when credentials are provided by the user.
3. **Competitor Ad Library Automation:**
   - *Current Reality:* `analyze_competitor_creatives` searches indexed ad copy and links directly to the Meta Ad Library URL for Egypt.
   - *Improvement Opportunity:* Integrate Meta Ad Library API directly to extract live active creative images, copy text, and run dates into the response JSON.
4. **Agent-to-Agent Handoffs:**
   - *Current Reality:* Orchestrator routes the user prompt to one primary agent.
   - *Improvement Opportunity:* Multi-turn collaborative chains (e.g., Salma researches competitors → passes data to Karim to design creatives → passes to Omar to write the copy → Nour delivers the unified package).

---

## 4. Multi-Agent Roster & Persona Matrix

Each agent in `app/ai/agents/` has a distinct domain of expertise, tone, and tool privileges:

### 👑 The Orchestrator (`app/ai/agents/orchestrator/`)
- **Role:** Chief of Staff / Dispatcher.
- **Goal:** Understands user intent, determines which specialist should answer, and routes the query. If a request spans multiple disciplines, it decomposes the task.
- **Tone:** Professional, decisive, structured.

### 🎯 Nour — Head of Growth & Marketing Strategy (`app/ai/agents/nour/`)
- **Role:** Strategic Director & Account Lead.
- **Specialty:** High-level 90-day roadmaps, brand positioning, customer review synthesis, UVPs, unit economics, and campaign KPIs.
- **Language/Tone:** Confident, executive, bilingual (Egyptian business Arabic & English).
- **Key Tools:** `brand_context`, `analyze_market`, `analyze_reviews`, `detect_market_gaps`, `generate_marketing_strategy`.

### ✍️ Omar — Senior Direct-Response Copywriter (`app/ai/agents/omar/`)
- **Role:** Conversion Copywriter & Storyteller.
- **Specialty:** Meta/TikTok ad copy, 3-second video hooks, carousel caption copy, WhatsApp closing scripts, email newsletters.
- **Language/Tone:** Punchy, persuasive, culturally native (understands Egyptian consumer psychology, humor, pain points, and urgency).
- **Key Tools:** `brand_context`, `analyze_competitor_creatives`, `search`.

### 🎨 Karim — Creative Director & Visual Strategist (`app/ai/agents/karim/`)
- **Role:** Visual Concept & Creative Asset Architect.
- **Specialty:** Creative briefs, storyboard design, UGC format recommendations, visual pattern interrupts, image prompt generation.
- **Language/Tone:** Visually descriptive, trend-savvy, focused on production feasibility and high CTRs.
- **Key Tools:** `generate_image`, `analyze_reference_creative`, `analyze_competitor_creatives`.

### 📊 Layla — Performance Media Buyer & Analytics Lead (`app/ai/agents/layla/`)
- **Role:** Quantitative Advertising & Media Optimization.
- **Specialty:** Budget distribution across Meta/TikTok/Google, ROAS modeling, audience targeting structures (CBO vs ABO, Lookalikes), bid strategies.
- **Language/Tone:** Data-driven, analytical, numbers-first, risk-aware.
- **Key Tools:** `get_campaign_insights`, `get_breakdowns`, `create_recommendation`.

### 🔍 Salma — Competitive Intelligence & Market Researcher (`app/ai/agents/salma/`)
- **Role:** Lead Market Scout & Fact Gatherer.
- **Specialty:** Competitor teardowns, website CRO auditing, keyword demand clustering, live web search.
- **Language/Tone:** Objective, evidence-based, thorough, always citing sources.
- **Key Tools:** `analyze_website`, `analyze_competitor`, `analyze_keywords`, `search`.

---

## 5. Central Brain & Memory Architecture

```
                  +----------------------------------------------+
                  |               USER REQUEST                   |
                  +----------------------+-----------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |         CONTEXT INJECTION (P0-P2)            |
                  |                                              |
                  |  1. Brand Profile (Name, Industry, Currency) |
                  |  2. Brain LTM: Profile.user_context_summary  |
                  |     (Strictly < 300 tokens)                  |
                  |  3. Relevant BrandMemoryItems                |
                  |     (Each item strictly 80-120 words)        |
                  |  4. Active Chat History (Recent turns)       |
                  +----------------------+-----------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |          AGENT EXECUTION / LLM               |
                  +----------------------+-----------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |          COMPACT DISTILLATION                |
                  |  fast_llm distills learnings to 80-120 words |
                  +----------------------+-----------------------+
                                         |
                        +----------------+---------------+
                        |                                |
                        v                                v
         +-----------------------------+  +-----------------------------+
         |     BrandMemoryItem.text    |  |     Profile LTM Summary     |
         |    (~100 tokens per item)   |  |   (< 300 tokens combined)   |
         +-----------------------------+  +-----------------------------+
```

### The "No-Token-Bloat" Law
Every team member must strictly follow this principle:
- **Never save full 1,000-word essays or raw scraped HTML into `BrandMemoryItem.text` or `Profile.user_context_summary`.**
- AI agents retrieve these memory items on *every single prompt*. Storing 10 long reports would consume 15,000 tokens per user message, slowing down response times and exhausting token limits.
- **The Golden Rule:** Always distill memory text down to **80–120 words** of bulleted facts, and store the complete report in `metadata_["full_report"]` for UI display only.

---

## 6. Project Directory Map (What Lives Where)

```
d:/NTI-Final-Project/
│
├── SAHM-AI-backend/                      # FastAPI Python Backend
│   ├── app/
│   │   ├── ai/
│   │   │   ├── agents/                   # Multi-agent definitions (Nour, Omar, Karim, Layla, Salma, Orchestrator)
│   │   │   ├── tools/                    # Tool implementations (search, analyze_website, get_reviews, etc.)
│   │   │   ├── providers/                # LLM & Search provider adapters (OpenAI, Anthropic, Gemini, Tavily, DDG)
│   │   │   └── memory/                   # Vector store & memory retrieval logic
│   │   ├── api/
│   │   │   ├── research.py               # Research Hub REST endpoints, ground-truth engine, dual memory storage
│   │   │   ├── chat.py                   # Chat & conversation streaming endpoints
│   │   │   └── deps.py                   # Multi-tenant workspace resolver (X-Workspace-Id) & auth guards
│   │   ├── core/
│   │   │   └── tools/                    # Core tool schemas & execution framework
│   │   ├── db/
│   │   │   ├── models.py                 # SQLAlchemy 2 models (Business, BrandMemoryItem, Profile, etc.)
│   │   │   └── session.py                # Database connection pools (Postgres / SQLite)
│   │   └── main.py                       # FastAPI entrypoint, middleware, routers
│   ├── .env                              # Backend environment secrets & API keys
│   └── pyproject.toml / requirements.txt
│
└── SAHM-AI-frontend/                     # Next.js 14 Frontend
    ├── src/
    │   ├── app/                          # Next.js App Router pages (/research, /assistant, /creatives, etc.)
    │   ├── components/
    │   │   ├── research/
    │   │   │   └── research-view.tsx     # Research Hub UI: tool cards, live execution, MarkdownReport renderer
    │   │   ├── chat/                     # Assistant conversation threads, agent switcher, message bubbles
    │   │   └── ui/                       # Reusable shadcn/ui primitives (cards, buttons, badges, dialogs)
    │   ├── store/                        # Client state (auth tokens, active workspace ID)
    │   └── lib/                          # Client API fetchers & utilities
    └── package.json
```

---

## 7. Strategic Success Metrics for Your Work
When you improve any tool or agent, judge your success by these four metrics:
1. **Fact Grounding & Reliability:** Does the tool cite real URLs? Does it handle typos and reject fake brand names?
2. **Speed & Latency:** Does the tool complete its analysis in under 8–15 seconds?
3. **Actionability for Marketing:** Does the output contain specific copy angles, target numbers, and execution steps, rather than generic fluff?
4. **Token Efficiency:** Does the tool save compact memories (<120 words) so that future agent conversations remain fast and affordable?
