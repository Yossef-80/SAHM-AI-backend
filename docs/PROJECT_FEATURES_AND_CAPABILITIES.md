# 🌟 SAHM Marketing OS — Features & Capabilities Manual

> **Purpose:** The definitive feature-by-feature reference for SAHM AI. This document explains **what every feature does**, **how it works under the hood**, **what we offer to users**, and **how each feature connects to the backend and database**.  
> **Audience:** Product managers, presenters, engineers, and users who want a 360-degree understanding of every capability in the platform.

---

## 1. Feature Matrix Overview

| Feature Module | Route | Key AI Specialists | Primary Capabilities |
| :--- | :--- | :--- | :--- |
| **Autonomous Research Hub** | `/research` | Nour (نور) | 9 real-time tools, ground-truth entity validation, live web search, URL scraping, review extraction, competitor ad audit. |
| **Multi-Agent Assistant Chat** | `/assistant`, `/assistant/[id]` | Orchestrator, Nour, Omar, Layla, Karim, Salma, Chat | Real-time SSE streaming, automatic intent routing, Generative UI Blocks (`text`, `findings`, `action_card`). |
| **Brand Identity & Strategy Hub** | `/brand-identity` | Brand Intelligence Agent, Central Brain | Living brand profile: color palette tokens, essence (voice/tagline/archetype), target audience, SWOT analysis, tracked competitors, 1-click chat sync. |
| **Content Calendar & Scheduler** | `/calendar` | Omar (عمر) | Automated multi-day post generation (concepts, hooks, captions, hashtags), channel/format filters, auto-saved to DB. |
| **Paid Ad Campaigns Hub** | `/campaigns`, `/campaigns/[id]` | Omar, Salma | Multi-channel campaign structures (Meta, TikTok, Google), budget tracking, linked creatives and recommendations. |
| **Creatives Studio** | `/creatives` | Layla (ليلى) | Direct-response ad copy, 3-second video hooks, commercial storyboards, reference image style extraction, image prompt generation. |
| **Analytics & GA4 Insights Hub** | `/insights` | Karim (كريم) | Google Analytics 4 (GA4) traffic and conversion metrics, Meta ad performance, anomaly alerts. |
| **Recommendations & Approvals**| `/recommendations` | Salma (سلمى) | Automated performance audits, 2-phase human-in-the-loop approvals with diff inspection, Meta write execution. |
| **Brand Memory & Knowledge Base**| `/settings/memory` | Central Brain | 13 lifecycle-managed memory kinds, lexical + recency ranking (no embeddings), global brand profile LTM anchor (<300 tokens), zero token bloat. |
| **Integrations & Connectors** | `/settings/connectors`, `/integrations` | System Core | Encrypted Meta OAuth, Google Analytics service account, Google Places, Apify Facebook ads scraper, Tavily/DuckDuckGo. |
| **Governance & Usage Guardrails**| `/settings/audit`, `/settings/usage`, `/settings/llm`| System Core | Per-workspace daily token spend ceilings, immutable audit logging, multi-tenant RBAC, dual-tier LLM provider routing. |

---

## 2. Feature 1: Autonomous Research Hub (`/research`)

### What It Does
The Research Hub is the real-time market intelligence engine of SAHM AI. Instead of spending hours manually searching Google, scrolling social media, or reading customer reviews, the user selects a tool card, types a query or URL, and receives a dense, structured intelligence report with clickable source links.

### How It Works Under the Hood
1. **Frontend Trigger:** The user interacts with `research-view.tsx` and clicks "Run" on any tool card.
2. **API Call:** Sends a request to `POST /api/v1/research/run` with the tool name and query.
3. **Ground-Truth Entity Verification:**
   - For brand and competitor queries, `verify_and_resolve_entity` queries a fast LLM.
   - Evaluates whether the brand exists in the market.
   - **Typo Correction:** Automatically detects phonetic misspellings (e.g., *"Tiee Hose"* → *"Tie House"*, *"Concrette"* → *"Concrete"*).
   - **Nonsense Rejection:** Rejects gibberish queries (e.g., *"asdfghjkl9999fake"*) and outputs an informative explanation without wasting API credits on searches.
4. **Live Data Collection:**
   - `web_search_tool` queries Tavily or DuckDuckGo with market-specific operators.
   - `analyze_website_tool` fetches landing page HTML using `httpx` and extracts content via `Trafilatura` and `BeautifulSoup`.
5. **Report Generation:** LLM structures findings according to the tool's prompt template into clean Markdown (strictly 300–380 words).
6. **Dual-Layer Memory Persistence:**
   - Fast LLM compresses the report to 80–120 words (~100 tokens) and saves it to `BrandMemoryItem.text`.
   - The full 380-word report is saved in `BrandMemoryItem.metadata_["full_report"]` for UI display.
7. **Frontend Rendering:** `research-view.tsx` renders the full report with styled headers, bullets, and clickable source links.

### The 9 Available Research Tools:
1. **Website Analyzer:** Audits landing page UVPs, trust badges, conversion friction points, and recommendations.
2. **Competitor Research:** Analyzes competitor market share, target audience, pricing bundles, and vulnerabilities in Egypt.
3. **Social Profile Audit:** Analyzes content pillars, video vs carousel performance, brand tone, and audience comments.
4. **Reviews & Sentiment:** Extracts customer sentiment, top praises, recurring complaints, and verbatim quotes.
5. **Market Dynamics:** Reports 2026 category size, growth trends, consumer spending shifts, and currency inflation impact.
6. **Search Keywords:** Clusters high-intent transactional queries, educational searches, and long-tail silos.
7. **Competitor Ad Creatives:** Analyzes long-running ad concepts, 3-second visual hooks, UGC vs studio styles, and promotional CTAs.
8. **Market Gaps & Opportunities:** Synthesizes accumulated brand memory to identify unserved needs and pricing whitespaces.
9. **Strategic Roadmap:** Generates an executive 90-day marketing blueprint with monthly milestones and channel budget splits.

---

## 3. Feature 2: Multi-Agent Assistant Hub (`/assistant`)

### What It Does
An interactive conversational interface where business owners collaborate with their entire AI marketing squad in one unified thread. Users don't need to manually pick an agent—the system understands the request and summons the right specialist.

### How It Works Under the Hood
1. **Streaming Endpoint:** Requests enter `POST /api/v1/assistant/message/stream`.
2. **Context Assembly:** `service.py` retrieves:
   - `Profile.user_context_summary` (Brand Facts, <300 tokens).
   - Recent message turns from the `messages` table.
   - Dynamic session summary from `conversation.summary` if the thread exceeds 1,500 tokens.
3. **Orchestrator Routing:**
   - In `app/ai/agents/orchestrator/brain.py`, a fast LLM runs with `temperature=0.0`.
   - Evaluates the user's explicit intent in the newest message.
   - Routes to the appropriate specialist:
     - Live research / crawls → **Nour**
     - Strategy & content calendars → **Omar**
     - Copywriting & visual creative briefs → **Layla**
     - GA4 analytics & metrics → **Karim**
     - Running campaign optimization → **Salma**
     - General chat / profile recall → **Conversational Agent**
4. **Agent Execution:**
   - The chosen agent runs its LangGraph StateGraph, invoking tools as needed.
   - Omar automatically saves generated calendar posts to the `ContentItem` table.
5. **UI Block Structuring:**
   - `BLOCK_STRUCTURER_PROMPT` converts raw markdown and tool arguments into typed React UI Blocks.
   - Emits Server-Sent Events (`SSE`) carrying progress updates and the final blocks payload.
6. **Client Rendering:**
   - The frontend renders dynamic blocks: expandable finding cards, source citations, action buttons, and markdown text.
7. **Answer language:**
   - The thread answers in the language chosen in the UI (`ar` / `en`), and follows
     a change immediately — including mid-conversation. The rule, the resolution
     order and the tests are in [`LANGUAGE_POLICY.md`](LANGUAGE_POLICY.md).

---

## 4. Feature 3: Unified Brand Identity & Strategy Hub (`/brand-identity`)

### What It Does
The Brand Identity & Strategy Hub provides a living, centralized source of truth for the brand's visual identity, strategic essence, audience definition, competitive landscape, and market opportunities. Instead of treating brand assets as static settings, SAHM's autonomous **Brand Intelligence Agent** dynamically synthesizes, updates, and synchronizes the brand identity across all conversations, research reports, and memory facts.

### Key Capabilities & What It Offers:
1. **Interactive Brand Visual Tokens & Palette:**
   - Visual swatch cards for **Primary Color**, **Secondary Color**, and **Accent Color**.
   - Curated 5-color palette tailored to the brand's industry and explicit user selections.
   - 1-click **Copy Hex** utility with instant clipboard feedback.
2. **Brand Essence & Strategic Positioning:**
   - **Tagline / Slogan:** Punchy, high-recall slogan crafted for the brand's category.
   - **Mission & Vision Statements:** Long-term brand purpose and operational goals.
   - **Brand Voice & Personality:** Verified tone guidelines injected into every creative agent (e.g. Layla, Omar).
   - **Tone Keywords:** Core emotional anchors (e.g. *"Authentic"*, *"Artisanal"*, *"Direct"*, *"Reliable"*).
   - **Brand Archetype:** Strategic positioning archetype (e.g. *"The Craftsman & Explorer"*).
3. **Verified Target Audience Profile:**
   - **Executive Summary:** Core demographic and behavioral persona.
   - **Demographics & Geography:** Age groups, metropolitan locations (Cairo, Alexandria, Giza), income brackets.
   - **Customer Pain Points:** Exact frustrations customers experience with commercial alternatives.
   - **Customer Goals & Motivations:** Desired outcomes driving purchasing decisions.
4. **Living SWOT Analysis:**
   - **Strengths:** Internal advantages, proprietary formulas, fresh fulfillment guarantees.
   - **Weaknesses:** Early-stage retail distribution limits, supply chain challenges.
   - **Opportunities:** Untapped subscription models, Arabic-first video storytelling, local partnerships.
   - **Threats:** Currency fluctuation risks, price wars from mass commercial brands.
5. **Key Product Offers & Tracked Competitors:**
   - Direct inventory of flagship product lines, bundles, and promotions.
   - Auto-curated and deduplicated list of competitors mined from user research and chat history.
   - Identification of market gaps and whitespace opportunities to seize.
6. **1-Click Sync from Chats & Research:**
   - The user clicks **"Sync from Chats & Research"** to invoke `BrandIntelligenceAgent`.
   - Analyzes recent conversational turns in full, applies token-aware rolling memory compression on older history, extracts new facts via LLM, and persists the updated profile to both the database and long-term memory.
7. **Full In-Place Editing Modal:**
   - Marketers can click **"Edit Profile"** to manually refine colors, tagline, tone of voice, audience traits, and competitor lists.
   - Changes are saved immediately via `PATCH /api/v1/business/brand-identity` and reflected in all future agent executions.
8. **Home Page Brand Identity Banner:**
   - Prominently featured on the main dashboard (`/`) via `BrandIdentityBanner`, displaying real-time brand name, niche badge, slogan, color preview swatches, and a quick launch button to the Hub.
9. **Orchestrator Proactive Action Card:**
   - When users discuss brand colors, identity, or competitor positioning in chat, the Orchestrator emits an interactive `UPDATE_BRAND_PROFILE` Action Card providing 1-click navigation to the Hub.

### Backend Architecture & Endpoints:
- `GET /api/v1/business/brand-identity` — Retrieves the unified brand identity profile.
- `POST /api/v1/business/brand-identity/sync` — Synthesizes and syncs identity from active chats and memory facts.
- `PATCH /api/v1/business/brand-identity` — Granularly updates identity fields with immediate memory and database synchronization.
- **Service Layer:** `app/core/brand/identity_service.py` (`BrandIntelligenceAgent`).

---

## 5. Feature 4: Automated Content Calendar & Scheduler (`/calendar`)

### What It Does
A dynamic content scheduling dashboard that plans and organizes a brand's social media and marketing content across multiple platforms and dates.

### What It Offers:
- **Instant Calendar Generation:** Generate 5-day, 14-day, or 30-day content calendars in seconds.
- **Complete Post Specifications:** Every scheduled item includes:
  - **Date & Time:** Target publishing day (e.g., `2026-10-15`) and optimal posting hour (e.g., `18:00`).
  - **Channel:** Instagram, Facebook, TikTok, Email, or Blog.
  - **Format:** Reel, Story, Static Post, Video, Carousel, or Article.
  - **Content Pillar:** Educational, Engagement, Product Spotlight, Customer Proof, or Promotional.
  - **Concept & Title:** Core message and visual description.
  - **Hook & Caption:** Scroll-stopping opening line + engaging body copy + Call to Action.
  - **Hashtags:** Auto-generated, niche-relevant hashtags.
  - **Status:** `scheduled`, `draft`, or `published`.
- **Calendar Views:** Interactive monthly and weekly grid views with channel color-coding and status badges.
- **Backend Endpoints:**
  - `GET /api/v1/content?month=YYYY-MM`: Returns all posts for a given month.
  - `POST /api/v1/content/generate`: Generates a full month's posts based on brand context and posts-per-week frequency.
  - `PATCH /api/v1/content/{id}`: Edit post dates, captions, or status.
  - `DELETE /api/v1/content/{id}`: Delete scheduled items.

---

## 5. Feature 4: Paid Ad Campaigns Hub (`/campaigns`)

### What It Does
Enables business owners and media buyers to organize, launch, and monitor marketing campaigns across Meta, TikTok, and Google Ads.

### Key Capabilities:
- **Campaign Architecture:** Define campaign names, channels, objectives (Conversions, Awareness, Traffic, Leads), budgets, currency (EGP, USD, SAR), and flight dates.
- **Linked Assets:** Every campaign links directly to its generated creative assets (`creatives` table), performance data (`insights` table), and AI recommendations (`recommendations` table).
- **Meta Integration:** Links to live `meta_campaign_id` for automated synchronization with Meta Ads Manager.
- **Endpoints:**
  - `POST /api/v1/campaigns`: Create new campaign.
  - `GET /api/v1/campaigns`: List all workspace campaigns with creative and recommendation counts.
  - `GET /api/v1/campaigns/{id}`: Detailed view with attached assets and insights.

---

## 6. Feature 5: Creatives Studio (`/creatives`)

### What It Does
The visual and copy production hub where Layla (Creative Director) produces ad assets that drive clicks and conversions.

### Key Capabilities:
- **Direct-Response Copywriting:** Crafts persuasive ad copy utilizing proven conversion frameworks (PAS: Problem-Agitate-Solution, AIDA: Attention-Interest-Desire-Action, Before-After-Bridge).
- **3-Second Pattern Interrupts:** Generates high-impact visual opening hooks designed to stop feed scrolling on Instagram Reels and TikTok.
- **Commercial & Video Scripts:** Writes 15-second and 30-second video storyboards with audio cues, on-screen text, and visual framing directions.
- **Reference Asset Style Extraction (`POST /creatives/analyze-reference`):**
  - Users provide a reference image URL or upload.
  - The system analyzes lighting, color palette, visual hierarchy, typography, and mood into a structured `StyleSpec`.
  - Saves the `StyleSpec` in brand memory for consistent future asset creation.
- **AI Image Generation (`POST /creatives/generate`):**
  - Synthesizes prompts with brand visual guidelines and style specifications.
  - Generates image ad concepts across standard aspect ratios (`1:1` Feed, `9:16` Stories/Reels, `16:9` Landscape).

---

## 7. Feature 6: Analytics & GA4 Insights Hub (`/insights`)

### What It Does
A unified performance monitoring dashboard that tracks traffic, engagement, and conversion efficiency.

### Key Capabilities:
- **Google Analytics 4 (GA4) Integration:**
  - Uses the official Google Analytics Data API (`app/ai/tools/fetch_ga4_performance.py`).
  - Authenticates via environment service account (`GA4_CLIENT_EMAIL` + `GA4_PRIVATE_KEY`).
  - Retrieves real metrics: Sessions, Total Users, Screen Page Views, Bounce Rate, Engagement Rate, Average Session Duration, and Key Events (Conversions).
- **Meta Ads Analytics:**
  - Pulls live campaign insights (Spend, Impressions, Clicks, CTR, CPC, ROAS).
- **Anomaly Detection:**
  - Generates `Insight` records with severity ratings (`info`, `warning`, `critical`).
  - Alerts users when bounce rates spike or ad sets suffer creative fatigue.

---

## 8. Feature 7: Autonomous Recommendations & Approvals (`/recommendations`)

### What It Does
A safe, governed optimization loop where AI agents discover performance bottlenecks and propose specific, approvable adjustments to active campaigns.

### The 2-Phase Safe Approval Workflow:
1. **Phase 1: Discovery & Proposal:**
   - Salma identifies an optimization opportunity (e.g. *"Reallocate 200 EGP daily budget from fatigued Ad Set A to high-ROAS Ad Set B"*).
   - Calls `create_recommendation` tool, saving a record with `status="pending"` and `requires_approval=True`.
   - Generates a concrete `diff` payload:
     ```json
     {
       "field": "daily_budget",
       "before": 500.0,
       "after": 300.0,
       "target": "adset_238510294819"
     }
     ```
2. **Phase 2: Human Review & Execution (`POST /recommendations/{id}/approve`):**
   - The user opens the review dialog in the UI and inspects the rationale, estimated impact, and diff.
   - When the user confirms (`confirm=true`), the backend invokes the protected tool (`apply_change`).
   - Executes the change against the Meta Graph API.
   - Writes an immutable record to the `audit_logs` table recording who approved it, timestamp, and request ID.

---

## 9. Feature 8: Multi-Tier Brand Memory & Knowledge Base (`/settings/memory`)

### What It Does
Ensures that every agent understands the brand's unique history, competitors, and products without causing context window bloat or runaway API bills.

### Key Capabilities:
- **Tier 1: Active History:** Maintains recent conversational context.
- **Tier 2: Dynamic Session Summary:** Condenses conversation threads once they reach 1,500 tokens.
- **Tier 3: Brand Memory Items:** Stores verified facts in the `BrandMemoryItem` table; retrieval is BM25 + recency + kind affinity (no embeddings), full reports stay in metadata.
- **Tier 4: Global Brand Brain (LTM):** Permanently maintains `Profile.user_context_summary` (<300 tokens) updated in the background after 3 minutes of inactivity.
- **Separation of Facts vs Plans:** General brand facts are available to the router, while active marketing plans are isolated to strategy agents.
- **CRUD Management:** Users can inspect, edit, or delete any memory item in `/settings/memory`.

---

## 10. Feature 9: Security, Governance & Settings (`/settings`)

### What It Does
Enterprise-grade management of credentials, model routing, usage limits, and audit trails.

### Key Capabilities:
- **Encrypted Connector Credentials (`/settings/connectors`):** All third-party secrets (Meta tokens, Google API keys, Apify keys) are encrypted at rest using Fernet symmetric encryption (`TokenCipher`). Secrets are never logged or returned over HTTP.
- **Workspace LLM Routing (`/settings/llm`):** Configure separate LLM providers and models for fast tasks (summarization, entity resolution) vs strong tasks (deep strategy, creative copywriting). Supports OpenAI, Gemini, Claude, Groq, Ollama, and OpenRouter.
- **Daily Token Budget Guard (`/settings/usage`):** Workspaces set daily token spend limits. Every LLM call records tokens in `usage_records`. Requests are halted if the budget is exceeded.
- **Immutable Audit Trail (`/settings/audit`):** Every configuration change, connector update, recommendation approval, and campaign publication is permanently logged with actor email, timestamp, and IP address.
- **Multi-Tenant Isolation:** All database rows are partitioned by `business_id` and `workspace_id`. Cross-tenant reads return 404 Not Found to prevent tenant discovery.
