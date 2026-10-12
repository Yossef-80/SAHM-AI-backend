# 🚀 SAHM Marketing OS — Project Scope, Architecture & State Assessment

> **Target Audience:** Engineering team members, technical leads, product managers, and presentation speakers who need a complete, ground-up understanding of SAHM AI, its commercial value proposition, actual implementation status, and multi-agent system.  
> **How to Use:** Read this document to understand the full system scope, copy sections into presentation slide decks, or feed it to AI assistants (Claude, ChatGPT, Gemini) to collaborate on new features and architectural optimizations.

---

## 1. Executive Summary & Product Vision

### What is SAHM AI?
**SAHM AI (سهم)** is an autonomous **Multi-Agent Marketing Operating System** engineered specifically for small-to-medium businesses (SMBs), e-commerce brands, and digital agency operators in Egypt and the MENA region. 

SAHM transforms fragmented, high-friction marketing workflows into an integrated, autonomous **"Marketing Department in a Box"**. Through a unified, bilingual interface (Arabic & English), business owners and marketing teams collaborate with specialized AI agents who execute market research, competitive audits, direct-response copywriting, creative design briefs, multi-channel campaign planning, automated content calendar scheduling, and performance analytics.

```
+---------------------------------------------------------------------------------------------------+
|                                      SAHM CLIENT (Next.js 14)                                     |
|  +--------------------+  +--------------------+  +--------------------+  +---------------------+  |
|  |  🧠 Research Hub   |  | 💬 Assistant Chat  |  | 📅 Content Calendar|  | 🎨 Creatives Studio |  |
|  | (Live Market Intel)|  | (Multi-Agent Squad)|  | (Auto-Scheduled)   |  | & Campaign Manager  |  |
|  +---------+----------+  +---------+----------+  +---------+----------+  +----------+----------+  |
+------------|-----------------------|-----------------------|------------------------|-------------+
             |                       |                       |                        |
             v                       v                       v                        v
+---------------------------------------------------------------------------------------------------+
|                                  SAHM BACKEND API (FastAPI)                                       |
|      /api/v1/research  •  /api/v1/assistant  •  /api/v1/content  •  /api/v1/campaigns           |
|      /api/v1/creatives •  /api/v1/recommendations • /api/v1/insights • /api/v1/settings          |
+----------------------------------------------------+----------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------+
|                                CENTRAL BRAIN & MEMORY ENGINE                                      |
|  Tier 1: Working Turn History (Current active session messages)                                   |
|  Tier 2: Dynamic Session Summary (conversation.summary compressed at token cap)                   |
|  Tier 3: Brand Memory Items (Vector-embedded; compact 80-120 word text + full report metadata)    |
|  Tier 4: Global Brand Brain (Profile.user_context_summary < 300 tokens, 3-min debounced update)    |
+----------------------------------------------------+----------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------+
|                       SPECIALIZED MULTI-AGENT SQUAD (LangChain & LangGraph)                       |
|                                                                                                   |
|   👑 The Orchestrator   --> Intent classification, deterministic routing & UI Block structuring    |
|   🔍 Nour (نور)         --> Lead Researcher: Web crawling, competitor teardowns, reviews, ads     |
|   🎯 Omar (عمر)         --> Chief Strategist: Positioning, 90-day roadmaps, content calendars     |
|   ✍️ Layla (ليلى)       --> Creative Director: Ad copy, 3-sec visual hooks, video scripts, briefs  |
|   📊 Karim (كريم)       --> Performance Analyst: Campaign metrics, GA4 traffic, ROAS, conversions  |
|   ⚙️ Salma (سلمى)       --> Optimizer: Spend re-allocations, A/B tuning, 2-phase human approvals   |
|   💬 Chat Assistant     --> Fast conversational engine for profile queries, pleasantries & context |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. The Core Problems SAHM Solves

### Problem 1: Crushing Marketing Headcount Costs
Running an effective modern growth engine requires at least 4 to 6 specialized roles:
- A Market Researcher & Competitor Analyst (\$1,000–\$2,500/mo)
- A Direct-Response Copywriter (\$800–\$2,000/mo)
- A Creative Director / Visual Concept Designer (\$1,200–\$3,000/mo)
- A Media Buyer / Quant Optimizer (\$1,000–\$2,500/mo)
- A Content & Social Media Coordinator (\$600–\$1,500/mo)

For an SMB or early-stage brand in Egypt or the GCC, spending \$4,000–\$10,000/month on human payroll before running a single dollar of paid ads is cost-prohibitive. SAHM provides an entire autonomous department for a fraction of the cost.

### Problem 2: Context Fragmentation & Knowledge Loss
Traditional marketing tools operate in silos:
- Google Analytics knows traffic but doesn't know ad copy.
- Meta Ads Manager knows ROAS but doesn't understand customer reviews or competitor weaknesses.
- ChatGPT creates isolated text copy with zero memory of the brand's profit margins, past campaign wins, or previous competitor analyses.

SAHM connects every touchpoint into a **unified Brand Memory Engine**. When Omar drafts a campaign or Layla writes an ad script, they automatically inherit the competitors uncovered by Nour, the friction points extracted from Google Maps reviews, and the budget guardrails established by the business owner.

### Problem 3: The Egypt & MENA Regional Execution Gap
Western AI marketing tools (Jasper, Copy.ai, AdCreative.ai) fail in regional Arabic markets:
- They produce literal Modern Standard Arabic (MSA / الفصحى) that sounds stiff, robotic, and unnatural in direct-response ads.
- They do not understand Egyptian consumer psychology, local slang, humorous pattern interrupts, or cultural seasonalities (Ramadan campaigns, Eid offers, White Friday).
- They ignore local economic realities: Egyptian Pound (EGP) currency devaluations, cash-on-delivery (COD) friction, WhatsApp-first ordering funnels, and consumer price sensitivity.

SAHM is architected from day one with bilingual Arabic/English prompts, Egyptian market entity databases, local currency support (EGP/SAR/AED/USD), and regional marketing heuristics.

---

## 3. What SAHM Offers (The Full Product Capabilities)

SAHM delivers 8 cohesive feature pillars:

### 1. Autonomous Research Hub (`/research`)
- **9 Live Research Tools:** Website Analyzer, Competitor Research, Social Profile Audit, Customer Reviews & Sentiment, Market Dynamics, SEO Keyword Demand, Competitor Creative Audit, Market Gap Detection, and 90-Day Strategy Generation.
- **Ground-Truth Entity Verification:** AI entity resolver detects whether a queried competitor is real, auto-corrects misspellings (e.g., *"Tiee Hose"* → *"Tie House"*), and rejects hallucinated brand names before running searches.
- **Real-Time Web Intelligence:** Live search integration (Tavily & DuckDuckGo) with clickable, canonical web citations in every report.
- **Deep Web Scraping:** BeautifulSoup & Trafilatura landing page scraper for extracting H1s, UVPs, trust badges, pricing anchors, and conversion friction.
- **Google Maps & Review Extraction:** Playwright-powered headless browser extractor for Google Maps place ratings, customer review counts, and verbatim feedback.
- **Meta Ad Library Scraping:** Apify Facebook Ads scraper actor integration for monitoring active competitor ad creatives, run dates, copy hooks, and image assets.

### 2. Multi-Agent Assistant Hub (`/assistant`)
- **Natural Language Orchestration:** Users talk naturally to the team. The Orchestrator determines which specialist handles the request without manual agent selection.
- **Agent Switcher & Multi-Turn Threading:** Dedicated conversation threads with persistent memory, full message histories, and title auto-generation.
- **Real-Time Streaming (SSE):** Token-by-token streaming with live agent state indicators (e.g., *"Nour is searching web evidence..."*, *"Structuring UI blocks..."*).
- **Generative React UI Blocks:** Agent responses do not output wall-of-text JSON. They stream structured React UI Blocks:
  - `text`: Executive summaries and explanations.
  - `findings`: Deep bulleted evidence with source URLs, clickable citations, and confidence badges.
  - `action_card`: One-click actions to trigger follow-up tasks.
  - `question_with_options`: Interactive multiple-choice prompts for user feedback.

### 3. Automated Content Calendar & Scheduling (`/calendar`)
- **Direct Multi-Day Post Generation:** Generates comprehensive daily content schedules (concepts, hooks, captions, hashtags, channels, formats, and best posting times).
- **Database Persistence:** Generated calendar items are automatically saved to the backend `ContentItem` table (`app/db/models.py`) with `scheduled` status.
- **Omnichannel Support:** Formats for Instagram (Reel, Story, Post), Facebook (Post, Ad), TikTok (Video), Email, and Blog.
- **Pillar Categorization:** Tagged by strategic pillars: Educational, Engagement, Product Spotlight, Customer Proof, and Promotional.

### 4. Paid Ad Campaigns Hub (`/campaigns`)
- **Campaign Architecture:** Create and track campaigns with budgets, target channels (Meta, TikTok, Google), date ranges, and core objectives.
- **Strategy Alignment:** Links creative assets, performance insights, and AI recommendations directly to individual campaigns.

### 5. Creatives Studio (`/creatives`)
- **Direct-Response Copywriting:** High-converting Meta/TikTok ad copy, 3-second pattern interrupt hooks, headline variations, and WhatsApp closing scripts.
- **Commercial & Video Scripts:** 15-second and 30-second video storyboards with visual framing cues and voiceover lines.
- **Reference Asset Analysis:** Upload or link reference creative images to extract a structured `StyleSpec` (color palette, typography, visual weight, lighting).
- **AI Image Generation:** Generates image ad concepts matching the brand's visual identity using external image providers or fallback mock rendering.

### 6. Analytics, GA4 & Insights Hub (`/insights`)
- **Google Analytics (GA4) Integration:** Automated connection via Google Analytics Data API (service account auth) fetching sessions, page views, bounce rate, engagement rate, and conversions.
- **Meta Ad Insights:** Real-time campaign performance reads (ROAS, CPA, CTR, Impressions, Spend).
- **Anomaly Detection:** Identifies creative fatigue, budget waste, and audience saturation.

### 7. Human-in-the-Loop Safe Approvals (`/settings/audit`, `/recommendations`)
- **Two-Phase Approval Gate:** Agents can NEVER make direct external ad account changes autonomously.
- **Phase 1 (Propose & Diff):** Agent generates an AI recommendation with estimated financial impact and a concrete `before` vs `after` diff.
- **Phase 2 (Confirm & Execute):** A human owner/member must explicitly click "Approve" with confirmation. Only then does the backend execute the change on Meta Graph API and write an immutable record to `audit_logs`.

### 8. Multi-Tenant Brand Memory & Governance (`/settings`)
- **Zero-Token-Bloat Architecture:** 4-tier memory hierarchy ensures that agents retrieve rich historical context without blowing LLM context windows.
- **Enterprise Security:** Fernet-encrypted API credentials at rest, SSRF request filtering, strict CORS rules, and per-workspace token budget guards.

---

## 4. Current Architectural State Assessment

### What Is Built and 100% Operational ✅

| Layer / Component | Implementation Location | Operational Status | Key Highlights |
| :--- | :--- | :--- | :--- |
| **API Entrypoint & Routers** | `app/main.py`, `app/api/*` | **100% Operational** | FastAPI with 16 modular routers, lifecycle hooks, unified error envelope, request timing headers (`X-DB-Queries`, `X-DB-Time`). |
| **LangGraph Multi-Agent Engine** | `app/ai/agents/` | **100% Operational** | Fully migrated to LangGraph StateGraphs (`nour_graph`, `omar_graph`, `layla_graph`, `karim_graph`, `salma_graph`). |
| **Deterministic Orchestrator Brain** | `app/ai/agents/orchestrator/brain.py` | **100% Operational** | `temperature=0.0` routing with `RoutingDecision` schema, plan/fact context separation, and dedicated `chat` fallback agent. |
| **UI Block Structurer** | `app/ai/agents/orchestrator/brain.py` | **100% Operational** | Translates raw agent tool arguments and Markdown into typed `MessageWithBlocks` (`text`, `findings`, `action_card`). |
| **Research Hub API** | `app/api/research.py` | **100% Operational** | 9 live tools with ground-truth entity resolution, web scraping, and dual-layer memory storage. |
| **Content Calendar System** | `app/api/content.py`, `app/ai/tools/strategy/` | **100% Operational** | REST endpoints (`GET /content`, `POST /content/generate`, `PATCH`, `DELETE`) + auto-persistence in `ContentItem` table. |
| **GA4 Performance Tool** | `app/ai/tools/fetch_ga4_performance.py` | **100% Operational** | Google Analytics Data API client for sessions, bounce rate, and conversions using service account credentials. |
| **Apify Facebook Ads Scraper** | `app/ai/tools/fetch_competitor_ads.py` | **100% Operational** | Actor-based scraper extracting active ad copy, media URLs, and keywords from Facebook Page URLs. |
| **Playwright Reviews Scraper** | `app/ai/tools/get_reviews_playwright.py` | **100% Operational** | Headless Chromium extractor for Google Maps review counts, star ratings, and review cards. |
| **Memory Compaction Engine** | `app/ai/memory/compressor.py` | **100% Operational** | Tiktoken cl100k_base with Arabic token ratios (~1.8 chars/token), session threshold summary, and 3-min debounced LTM. |
| **Database & Migration Layer** | `app/db/models.py`, `app/db/migrations/` | **100% Operational** | PostgreSQL + pgvector (with SQLite JSON fallback), Alembic migrations, multi-tenant isolation on `business_id`. |
| **Dual-Tier LLM Provider Router** | `app/core/llm/router.py` | **100% Operational** | Supports OpenAI, Gemini, Claude, Groq, Ollama, DeepSeek, and Mock with 60s cached workspace settings. |
| **Protected Approvals System** | `app/core/approvals/`, `app/core/tools/protected.py` | **100% Operational** | Strict isolation: external-write tools cannot be registered in agent catalogs; human confirmation mandatory. |
| **Next.js 14 Frontend** | `SAHM-AI-frontend/src/app/*` | **100% Operational** | Next.js App Router, Tailwind CSS, shadcn/ui primitives, SSE chat streaming, research views, settings panels. |

---

## 5. Architectural Map: The Two Backend Systems Explained

The codebase currently contains two parallel organizational folders under `SAHM-AI-backend/app/`:

```
SAHM-AI-backend/app/
│
├── core/                     # The Foundation & System Services
│   ├── approvals/            # 2-Phase human approval workflow & Meta change execution
│   ├── auth.py & security.py # TokenCipher (Fernet encryption), password hashing, JWT/Supabase auth
│   ├── llm/                  # Low-level client router (OpenAI, Claude, Ollama, Mock)
│   ├── memory/               # Embedding store (pgvector cosine search + Python fallback)
│   ├── orchestrator/         # REST message handling service (service.py), block validation
│   ├── ratelimit.py          # Request throttling & IP rate limiting
│   └── tools/                # Static tool registry (base.py, registry.py, protected.py)
│
└── ai/                       # The Live LangChain & LangGraph Cognitive Engine
    ├── agents/               # LangGraph compiled state machines (Nour, Omar, Layla, Karim, Salma, Orchestrator)
    ├── loggers/              # LangGraph event callbacks and agent run tracers
    ├── memory/               # Token-aware memory compressor & debouncer (compressor.py)
    ├── providers/            # Live LLM & Search provider instances (get_llm, get_search_provider)
    ├── settings.py           # Engine timeouts, model provider strings, and API secrets
    └── tools/                # Live agent tools (search, analyze_website, strategy, GA4, Apify)
```

### How They Collaborate:
1. A user prompt enters `POST /api/v1/assistant/message/stream` handled by `app/core/orchestrator/service.py`.
2. `service.py` handles database transactions, loads the conversation history, checks token budgets, and resolves workspace context.
3. It hands execution over to `app/ai/agents/orchestrator/brain.py` (`orchestrate(...)`), which routes to the appropriate LangGraph agent.
4. The agent executes tools from `app/ai/tools/`, generates output, and returns it to `brain.py`.
5. `brain.py` structures the result into React UI blocks (`MessageWithBlocks`) and streams it back to the client while `service.py` commits the message to PostgreSQL.

---

## 6. The Multi-Agent Roster

```
                                  👑 THE ORCHESTRATOR
                         Intent Classifier & Chief of Staff
                                          │
            ┌──────────────┬──────────────┼──────────────┬──────────────┐
            ▼              ▼              ▼              ▼              ▼
         🔍 NOUR        🎯 OMAR        ✍️ LAYLA       📊 KARIM       ⚙️ SALMA
        Researcher     Strategist      Creative       Analyst       Optimizer
```

### 1. 🔍 Nour (نور) — Lead Market & Competitive Intelligence Scout
- **Role:** Head of Market Research.
- **Mission:** Discover ground-truth facts about competitors, category dynamics, and consumer friction. Nour never speculates; every claim is backed by web citations.
- **LangGraph Node:** `app/ai/agents/nour/graph.py`
- **Output Schema:** `MarketResearchReport` (market overview, competitor positioning, pricing insights, opportunities, risks, sources, evidence).
- **Core Tools:** `web_search_tool`, `get_brand_context_tool`, `analyze_website_tool`, `get_reviews_tool`, `fetch_competitor_ads`.

### 2. 🎯 Omar (عمر) — Chief Marketing Strategist
- **Role:** Growth & Strategic Director.
- **Mission:** Synthesize research into brand positioning, channel budget splits, 90-day execution roadmaps, and content calendars. Bold, direct, and focused on unit economics.
- **LangGraph Node:** `app/ai/agents/omar/graph.py`
- **Output Schema:** `StrategyReport` (positioning, target segments, channel budget shares summing to 100%, content calendar).
- **Core Tools:** `get_brand_context_tool`, `generate_marketing_strategy`, `generate_campaign_strategy`, `generate_content_calendar`.

### 3. ✍️ Layla (ليلى) — Creative Director & Direct-Response Copywriter
- **Role:** Creative Director.
- **Mission:** Craft compelling ad copy, 3-second pattern interrupts, headline angles, WhatsApp closing scripts, and image/video briefs tailored to Egyptian consumer psychology.
- **LangGraph Node:** `app/ai/agents/layla/graph.py`
- **Output Schema:** `CreativeBrief` (ad copy, visual directions, audio hooks, target emotions).
- **Core Tools:** `get_brand_context_tool`, `analyze_reference_creative`, `generate_image`.

### 4. 📊 Karim (كريم) — Performance & Analytics Lead
- **Role:** Quantitative Media Analyst.
- **Mission:** Analyze campaign performance, website traffic, conversion funnels, and ROAS. Blunt, minimal, and never interpolates missing metrics.
- **LangGraph Node:** `app/ai/agents/karim/graph.py`
- **Output Schema:** `PerformanceReport` (sessions, bounce rates, conversion trends, metric summaries).
- **Core Tools:** `get_brand_context_tool`, `fetch_ga4_performance`, `get_campaign_insights`.

### 5. ⚙️ Salma (سلمى) — Campaign Optimizer & Operations Lead
- **Role:** Performance Optimizer.
- **Mission:** Detect campaign inefficiencies and propose concrete, approvable adjustments (budget reallocations, bid tweaks). Salma proposes changes with evidence and defers execution to human approval.
- **LangGraph Node:** `app/ai/agents/salma/graph.py`
- **Output Schema:** `OptimizationPlan` (proposed change, evidence, impact estimation).
- **Core Tools:** `get_brand_context_tool`, `create_recommendation`, `get_campaign_insights`.

### 6. 💬 Conversational Chat Assistant
- **Role:** Instant Knowledge & Profile Assistant.
- **Mission:** Handles casual greetings, follow-up questions, and queries about the brand profile without invoking expensive agent graphs or search loops.

---

## 7. Strategic Success Metrics for Improving the System

When modifying or expanding any tool, agent, or endpoint in SAHM, measure your progress against these 5 engineering KPIs:

1. **Fact Grounding & Citation Integrity:** Does every research claim provide a valid, canonical URL? Does the entity resolver reject fake brand names?
2. **Speed & Latency:** Does a full research tool run finish in under 8–15 seconds? Does conversational chat stream its first token in under 1.5 seconds?
3. **Token Efficiency:** Are all memory items strictly distilled to 80–120 words (~100 tokens)? Does the session compressor maintain active history under 1,500 tokens?
4. **Actionability:** Does the output contain concrete Egyptian pound figures, ad copy hooks, and timeline dates rather than vague generic advice?
5. **Safety & Zero Unintended Writes:** Are all ad account modifications gated by the two-phase approval workflow and logged in `audit_logs`?
