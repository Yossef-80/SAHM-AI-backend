# Backend Project Plan — Sahm AI Marketing Operating System

> **Status:** APPROVED FOR IMPLEMENTATION  
> **Source of Truth:** Frontend codebase (`SAHM-AI-frontend`), Zod schema definitions (`src/types/index.ts`), API contract (`docs/API_CONTRACT.md`), and Action Registry (`docs/ACTIONS.md`).  
> **Target System:** FastAPI Python Backend (`SAHM-AI-backend`) serving Next.js 14 Frontend (`SAHM-AI-frontend`).  
> **Document Purpose:** Engineering roadmap, architectural specification, data model, API contract, and 5-developer parallel execution plan.

---

## Executive Summary & Quick Reference

This planning document establishes the complete backend architecture for **Sahm**, an AI-powered Marketing Operating System. The frontend has been implemented in Next.js 14 with a fully typed contract layer. 

The 13 key questions for kickoff are summarized below:
1. **What are we building?** A multi-agent marketing operating system that automates market research, campaign strategy, creative generation, content calendar management, performance analysis, and Meta Ads publication while enforcing strict human-in-the-loop approvals for spend and live changes.
2. **What does the frontend expect?** A RESTful JSON API accompanied by a Server-Sent Events (SSE) streaming endpoint for chat, returning data that strictly matches the Zod schemas in `SAHM-AI-frontend/src/types/index.ts`.
3. **What backend functionality is missing?** The database schema, persistence layer, agent orchestration, LLM/search tool execution, Meta OAuth and Graph API sync, background task processing, and SSE streaming pipeline.
4. **What is absolutely required first?** Database models, session handling, configuration loading, unified error handling, and the core Business/Settings endpoints (The Foundation).
5. **What can be developed in parallel?** After Foundation and Shared Contracts are established, Campaigns & Recommendations (Dev 2), Creatives & Asset Generation (Dev 3), Agent Orchestrator & SSE Chat (Dev 4), and Content Calendar & Meta Integration (Dev 5) can execute 100% concurrently.
6. **What does each developer own?**
   - **Dev 1:** Infrastructure, Database, Config, Business, Audiences, Settings, Auth/Session.
   - **Dev 2:** Campaigns, Strategy Generation, Recommendations (with diffs), Performance, Learnings.
   - **Dev 3:** Creative Generation, Variations, Reference Image/Video Analysis, Asset Storage.
   - **Dev 4:** Multi-Agent Orchestrator, SSE Chat Streaming, Async Task Runner, Tool Registry.
   - **Dev 5:** Content Calendar, Meta Ads OAuth & Publishing Integration, Insights Engine.
7. **What contracts must everyone agree on?** Unified error envelope (`ApiError`), ISO-8601 UTC dates, string ID formats, SSE `data: <json>\n\n` snapshot protocol, and human-in-the-loop `ChangeDiff` structures.
8. **What should we implement first tomorrow morning?** Stage 0 contract sign-off and Developer 1 running database migrations and core FastAPI lifespan setup.

---

# PHASE 1 — INSPECT THE PROJECT

### 1. Complete Project Structure
The repository is split into two primary workspaces:
- `SAHM-AI-frontend/`: Next.js 14 App Router application with full TypeScript strict typing, Tailwind CSS, shadcn/ui primitives, TanStack Query, Zustand, and `next-intl`.
- `SAHM-AI-backend/`: Python project managed via `uv` with FastAPI, SQLAlchemy 2.0, Alembic, Pydantic v2, LangGraph, and aiosqlite/asyncpg.

### 2. Frontend Framework & Architecture
- **Framework:** Next.js 14.2.35 (React 18, App Router).
- **Styling:** Tailwind CSS with CSS custom properties (`globals.css`), Radix UI / shadcn/ui components.
- **Data Fetching:** TanStack React Query (`@tanstack/react-query` v5). Query keys are centralized in `src/lib/query-keys.ts`.
- **State Management:**
  - `useAssistantStore` (`src/store/assistant-store.ts`): Manages active chat stream, user optimistic messages, context chips, and pending prompts.
  - `useUiStore` (`src/store/ui-store.ts`): Manages global dialogs (campaign wizard prefill, create creative dialog state, generated text modal).
- **Internationalization:** `next-intl` v4.14 supporting English (`en`) and Arabic (`ar`) with dynamic document direction (`ltr`/`rtl`).
- **Data Layer:** `MarketingApi` interface (`src/lib/api/types.ts`) with a runtime toggle:
  - `MockApi` (`src/lib/api/mock-api.ts`): In-memory deterministic mock state with canned fixtures.
  - `HttpApi` (`src/lib/api/http-api.ts`): Fetch-based HTTP client validating every payload via Zod schemas.

### 3. Frontend Routes & Pages
- `/` (`src/app/page.tsx`): Dashboard homepage containing:
  - `AskAiHero`: Input forwarding natural-language prompts to `/assistant`.
  - `Capabilities`: Quick-start action tiles (Competitor research, create campaign, optimize).
  - `NeedsAttention`: High/critical severity insight items with inline action buttons.
  - `ActiveCampaigns`: Summary table of running campaigns.
- `/assistant` & `/assistant/[id]` (`src/app/assistant/page.tsx`, `[id]/page.tsx`):
  - Multi-turn AI chat with conversational history list, SSE streaming renderer, interactive block cards (findings, metrics, diffs, progress, questions), and context chip selector.
- `/research` (`src/app/research/page.tsx`):
  - Hub for 9 AI research tools (Website, Competitor, Social Profile, Reviews, Market, Keywords, Competitor Creatives, Market Gaps, Marketing Strategy) triggering background agent tasks.
- `/campaigns` (`src/app/campaigns/page.tsx`):
  - Filterable list and data table of campaigns (status, search, metrics, health scores).
- `/campaigns/new` (`src/app/campaigns/new/page.tsx`):
  - 6-step campaign creation wizard (Goal, Audience, Strategy Generation, Creative Picker, Review, Launch).
- `/campaigns/[id]` (`src/app/campaigns/[id]/page.tsx`):
  - Detailed campaign dashboard with 7 tabs: Overview, Strategy, Audience, Creatives, Performance (14-day chart), Recommendations (with diff approvals), and Learnings.
- `/calendar` (`src/app/calendar/page.tsx`):
  - Monthly content calendar grid with month navigation, channel filters, AI calendar generation dialog, and side sheet content editor.
- `/creatives` (`src/app/creatives/page.tsx`):
  - Asset library for ad images, videos, and copy. Includes "Create Creative", "Analyze Reference Ad" (multimodal upload), and "Write with AI" text tools.
- `/insights` (`src/app/insights/page.tsx`):
  - Filterable feed of AI-generated insights, anomalies, and recommendations categorized by severity (`info`, `warning`, `critical`).
- `/agents` (`src/app/agents/page.tsx`):
  - Management screen for 5 agent personas (Researcher, Strategist, Creative, Analyst, Optimizer) with tone, autonomy, prompt instructions, and tool access toggles.
- `/integrations` (`src/app/integrations/page.tsx`):
  - Integration hub featuring the Meta Ads Connection Card with explicit connection state-machine (`not_connected`, `connecting`, `connected`, `needs_reauth`, `error`).
- `/settings` (`src/app/settings/page.tsx`):
  - Business profile, brand voice rules, locale switcher, currency, timezone, and notification toggles.

### 4. Forms and User Inputs
1. **Campaign Wizard Form:** Inputs for campaign name, objective, daily budget, currency, start/end dates, audience demographics (age range, gender, locations, interests), brief, and creative multi-selection.
2. **Create Creative Form:** Brief textarea, campaign dropdown, aspect ratio selector (`1:1`, `4:5`, `9:16`, `16:9`), variation count (1–6).
3. **Reference Analyzer Form:** Local image/video file upload (converted via `FileReader` to Base64 `data:` URI) or external video URL, plus kind selector (`image`/`video`/`copy`).
4. **Generate Calendar Form:** Month picker, posts per week (2–7), channel checkboxes, campaign scope, brief textarea.
5. **Content Item Side Sheet:** Inline editing for title, date, time, channel, format, status, caption, hook, and hashtags.
6. **Agent Persona Sheet:** Name, emoji, tone selector, language, custom instructions textarea (max 2000 chars), autonomy level, and tool toggles.
7. **Settings Form:** Currency selector (`EGP`, `USD`, `SAR`, `AED`, `EUR`), timezone selector, notification switches.
8. **Chat Composer & Interactive Blocks:** Text prompt with context chips; multi-select or single-select interactive question submissions.

### 5. API Client Implementation & Mock Data
- `HttpApi` enforces strict schema checks using `schema.safeParse(json)`. If a backend response violates the Zod contract, it throws `ApiError("SCHEMA_MISMATCH")` with detailed Zod issues.
- `MockApi` simulates network delay (300–800ms) and provides deterministic seeded data for "Nile Roasters" (`biz_1`), 3 campaigns (`cmp_1`, `cmp_2`, `cmp_3`), 6 creatives, 5 agent personas, and a deterministic monthly calendar generator.

### 6. Authentication-Related UI
- **Observation:** `SAHM-AI-frontend` contains **no login, registration, or password UI**.
- **Evidence:** `HttpApi` sets `credentials: "include"` on every request. The frontend operates as an already-authenticated workspace application.
- **Backend Implication:** The backend must support session cookies or a pre-configured workspace context.

---

# PHASE 2 — RECONSTRUCT THE PRODUCT

### A. What is the Product?
**Sahm** is an autonomous and human-in-the-loop AI marketing operating system tailored for direct-to-consumer (D2C) brands and agencies (with primary support for Middle Eastern / North African markets and bilingual English/Arabic operations).

### B. Problems Solved
1. **Fragmented Marketing Operations:** Eliminates jumping between separate tools for market research, copywriting, creative design briefs, campaign setup, and analytics.
2. **High Agency / Contractor Costs:** Automates routine agency tasks (competitor scans, SEO keyword clustering, creative briefs, ad variations).
3. **Runaway Ad Spend & Halting Mistakes:** Implements strict human-in-the-loop safeguards requiring manual diff review before budgets, statuses, or ads are published to live platforms.
4. **Ad Fatigue & Creative Stagnation:** Provides instant reference ad deconstruction and automated variation generation.

### C. Major User Workflows
1. **Market & Competitor Discovery:** User inputs a competitor URL, social handle, or keyword seed in `/research`. The system spins up an asynchronous agent task, performs search and extraction, and writes a findings report into an interactive chat conversation.
2. **End-to-End Campaign Creation:** User navigates `/campaigns/new`, inputs high-level goals, requests AI strategy generation, selects generated creatives, reviews budget diffs, and publishes directly to Meta Ads.
3. **Reference-to-Creative Reverse Engineering:** User uploads a winning TikTok or Instagram ad file. The AI deconstructs its hook, structure, tone, and visual style, producing an "inspired brief" that directly feeds creative generation.
4. **Monthly Content Planning:** User triggers calendar generation for a given month and channels. The system populates weekly scheduled posts, reels, and stories that can be reviewed and edited.
5. **Continuous Optimization Loop:** The background analyst identifies declining CTR or ROAS, creates an `Insight` with a proposed `Recommendation`, generates a `ChangeDiff`, and alerts the user on the home screen. The user reviews the before/after change and clicks "Confirm & Apply", triggering live campaign mutation.

### D. System Information Boundaries
- **Information to Store:** Business metadata, brand profiles (voice, colors, forbidden words), audience personas, campaigns, creative assets (images, videos, copy), insights, recommendations, performance time-series, conversations, chat messages, content calendar items, agent configurations, and OAuth tokens.
- **Information to Calculate/Process:** Aggregated campaign metrics (CTR, CPC, CPA, ROAS), performance health scores (0–100), vector embeddings of brand knowledge, comparative deltas, and reference ad video/image deconstruction.
- **Backend vs. Frontend Logic:**
  - *Backend Logic:* LLM prompt chaining, web search tool execution, image/video AI generation, Meta Graph API sync, token encryption, change execution, and recommendation generation.
  - *Frontend-Only Logic:* Date formatting, currency rendering, RTL/LTR layout flipping, optimistic chat UI, and client-side form validation.

### E. Ambiguities & Clarifications
- `UNKNOWN — requires team/business clarification`: User identity and multi-tenant authentication model. The frontend currently assumes a single active business without a login gate.
- `UNKNOWN — requires team/business clarification`: Video creative generation. Does the system produce actual rendered MP4 video files or video prompt briefs / storyboards?

---

# PHASE 3 — FEATURE INVENTORY

| Feature Name | Frontend Source Path | User Action | Required Backend Behavior | Required Data & APIs | External Services | AI/ML Required | Priority | Reason for Priority |
|---|---|---|---|---|---|---|---|---|
| **Business & Brand Context** | `src/app/settings/page.tsx` | View/load brand settings and voice | Return business metadata, brand voice guidelines, color palettes, and do-not-use rules | `GET /business`<br>`GET /brand-profile` | None | None | **P0** | **BLOCKING:** Required by prompt generator, agent personas, and settings view. |
| **Campaign Listing & Filtering** | `src/app/campaigns/page.tsx` | Filter by status (`active`, `draft`, etc.) or search | Return filtered list of campaigns with summary metrics and health scores | `GET /campaigns?status=&search=` | None | None | **P0** | **BLOCKING:** Core dashboard functionality; homepage depends on it. |
| **Campaign Detail & Tabs** | `src/app/campaigns/[id]/page.tsx` | View campaign tabs (Overview, Strategy, Performance, etc.) | Return full campaign record, time-series metrics, audience, recommendations | `GET /campaigns/:id`<br>`GET /campaigns/:id/performance`<br>`GET /audiences/:id` | None | None | **P0** | **BLOCKING:** Primary inspection screen for all advertising activity. |
| **AI Strategy Generation** | `src/components/campaigns/wizard/campaign-wizard.tsx` | Click "Generate Strategy" | Generate multi-phase marketing strategy, channel allocations, and KPIs | `POST /campaigns/strategy` | None | Fast/Strong LLM | **P1** | **HIGH:** Key value proposition of the campaign wizard. |
| **Campaign Creation & Launch** | `src/components/campaigns/wizard/campaign-wizard.tsx` | Save draft or click "Launch Now" | Create campaign in DB; if `launchNow=true`, publish campaign to Meta Ads | `POST /campaigns` | Meta Marketing API | None | **P0** | **BLOCKING:** Enables campaign creation and deployment. |
| **Live Campaign Updates** | `src/components/campaigns/detail/campaign-detail.tsx` | Pause, Resume, or Edit Daily Budget | Mutate campaign status or budget; sync change to Meta if published | `PATCH /campaigns/:id` | Meta Marketing API | None | **P0** | **BLOCKING:** Critical operational control for spend management. |
| **Recommendations & Diffs** | `src/components/campaigns/detail/tabs.tsx` | Review and click "Confirm & Apply" | Fetch recommendation diff; apply patch to campaign/creatives; mark applied | `GET /recommendations/:id`<br>`POST /recommendations/:id/apply`<br>`POST /recommendations/:id/dismiss` | Meta Marketing API | None | **P0** | **BLOCKING:** Core human-in-the-loop workflow mechanism. |
| **Conversational Chat (SSE)** | `src/components/assistant/assistant-view.tsx` | Send message or context-linked query | Stream AI response via Server-Sent Events returning structured block snapshots | `POST /conversations/messages/stream`<br>`GET /conversations/:id` | None | Fast/Strong LLM | **P0** | **BLOCKING:** Primary interface for AI assistant and research reporting. |
| **Async Agent Tasks Engine** | `src/components/research/research-view.tsx` | Run research tool (e.g. Website Analysis) | Create queued task, return conversation ref, execute research asynchronously | `POST /tasks`<br>`GET /tasks/:id` | Search APIs (Tavily/Serper) | LLM + Web Scraping | **P0** | **BLOCKING:** All 9 research tools depend on task polling and execution. |
| **Creative Library & Generation** | `src/app/creatives/page.tsx` | Generate image/copy ad creatives | Run creative generation pipeline; produce image URLs or copy headlines | `GET /creatives`<br>`POST /creatives/generate` | Image Provider (OpenAI/Stability) | Image Gen / Copy LLM | **P1** | **HIGH:** Essential for creating advertising assets. |
| **Creative Variations** | `src/components/creatives/creative-card.tsx` | Click "Generate Variations" | Produce 1–6 variations of an existing creative maintaining brand voice | `POST /creatives/:id/variations` | Image/LLM Provider | LLM / Diffusion | **P1** | **HIGH:** Drives creative fatigue mitigation. |
| **Reference Ad Reverse-Engineering** | `src/components/creatives/analyze-reference-dialog.tsx` | Upload video/image ad or enter URL | Deconstruct visual style, tone, hook, structure, and create inspired brief | `POST /creatives/analyze-reference` | Vision / Multimodal LLM | Multimodal Vision | **P1** | **HIGH:** Key competitive intelligence feature. |
| **Add Creative to Campaign** | `src/components/creatives/creative-card.tsx` | Select "Add to Campaign" | Associate creative ID with campaign; require confirmation diff | `POST /campaigns/:id/creatives` | Meta Marketing API | None | **P1** | **HIGH:** Connects creative assets to delivery campaigns. |
| **Content Calendar Management** | `src/app/calendar/page.tsx` | View month grid, edit scheduled post | Query, update, or delete scheduled posts for a given month and channels | `GET /content?month=`<br>`PATCH /content/:id`<br>`DELETE /content/:id` | None | None | **P1** | **HIGH:** Organic and paid calendar planning. |
| **AI Calendar Generation** | `src/components/calendar/generate-calendar-dialog.tsx` | Click "Generate Calendar" | Overwrite month's items with structured multi-channel calendar plan | `POST /content/generate` | None | Strong LLM | **P1** | **HIGH:** Major workflow automation tool. |
| **Copywriting & Text Tools** | `src/components/creatives/generate-text-dialog.tsx` | Run "Write with AI" (scripts, briefs) | Generate structured text sections with plain text export | `POST /generate/text` | None | Fast/Strong LLM | **P2** | **MEDIUM:** Utility tool for copywriting. |
| **Agent Persona Management** | `src/app/agents/page.tsx` | Toggle agent tool, adjust tone/autonomy | Save custom agent instructions, allowed tools, and autonomy settings | `GET /agents`<br>`PATCH /agents/:id` | None | None | **P2** | **MEDIUM:** Customizes system prompt behaviors per role. |
| **Meta OAuth & Connection Sync** | `src/components/integrations/meta-connection-card.tsx` | Connect / Reauth / Disconnect Meta | Handle OAuth 2.0 flow, verify ad account permissions, store access tokens | `GET /integrations/meta`<br>`POST /integrations/meta/connect`<br>`POST /integrations/meta/disconnect` | Meta Graph API | None | **P1** | **HIGH:** Required for live ad publishing and real metrics sync. |
| **Settings & Notifications** | `src/app/settings/page.tsx` | Update currency, timezone, notifications | Persist user preferences and notification flags | `GET /settings`<br>`PATCH /settings` | None | None | **P2** | **MEDIUM:** Application settings customization. |

---

# PHASE 4 — BACKEND ARCHITECTURE

```
                                  +---------------------------------------+
                                  |         Next.js 14 Frontend           |
                                  |  (TanStack Query / HttpApi / Events)  |
                                  +-------------------+-------------------+
                                                      |
                                           REST JSON / SSE Stream
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |        FastAPI Gateway Router         |
                                  |    (CORS / RequestID / ErrorHandler)  |
                                  +-------------------+-------------------+
                                                      |
                    +---------------------------------+---------------------------------+
                    |                                 |                                 |
                    v                                 v                                 v
+-----------------------------------+ +-------------------------------+ +--------------------------------+
|         Business & Core           | |      Agent Orchestrator       | |       External Integrations    |
| - Campaigns & Audiences           | | - LangGraph Multi-Agent       | | - Meta Marketing API           |
| - Creatives & Media Assets        | | - Tool Registry (Search/Web)  | | - LLM Router (Claude / OpenAI) |
| - Calendar & Settings             | | - SSE Streaming Generator     | | - Image Gen (DALL-E/Stability) |
| - Recommendations & Diffs         | | - Background Task Runner      | | - Search (Tavily/Serper)       |
+-----------------+-----------------+ +---------------+---------------+ +----------------+---------------+
                  |                                   |                                |
                  +-----------------------------------+--------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       SQLAlchemy 2.0 Async ORM        |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      PostgreSQL 16 + pgvector         |
                                  |   (SQLite + aiosqlite in Dev/Test)    |
                                  +---------------------------------------+
```

### Architectural Decisions

#### 1. Backend Framework [REQUIRED]
- **FastAPI (Python 3.11+)**: Selected because of native asynchronous support, first-class Pydantic v2 validation matching frontend Zod schemas, high performance, and deep integration with Python AI/LLM ecosystems.

#### 2. API Architecture [REQUIRED]
- Standard **REST JSON** for all CRUD operations, mutations, and task polling.
- **Server-Sent Events (SSE)** for `/conversations/messages/stream`. The frontend expects each event to be formatted as `data: <full SendMessageOutput snapshot JSON>\n\n`.

#### 3. Database & ORM [REQUIRED]
- **Production:** PostgreSQL 16 with the `pgvector` extension for vector similarity search (brand voice and knowledge memory).
- **Development & Testing:** SQLite with `aiosqlite` using an in-process Python cosine similarity fallback (via custom SQLAlchemy `Embedding` type decorator already architected in backend models).
- **ORM:** SQLAlchemy 2.0 async engine with Alembic for database migrations.

#### 4. Authentication & Security [REQUIRED]
- **Session / Workspace Context:** Since the frontend does not present a login view and uses `credentials: "include"`, the backend must implement cookie-based session identification, defaulting to a primary workspace (`default_workspace`) and business (`biz_1`) if no session cookie exists.
- **SSRF Protection:** Enforce DNS resolution and IP address range checks before scraping external URLs in research tools (preventing access to internal cloud metadata IP `169.254.169.254` or private subnets).
- **Token Encryption:** Meta OAuth access tokens and refresh tokens must be encrypted at rest using Fernet (AES-128-CBC with HMAC).

#### 5. File Storage [REQUIRED]
- **Local / S3-Compatible Object Store:** Uploaded reference files and generated image/video creatives must be stored in object storage (MinIO locally, AWS S3 / Cloudflare R2 in production) with public URLs returned in `assetUrl`.

#### 6. Background Jobs & Concurrency [REQUIRED]
- **In-Process Async Tasks:** Agent research tasks and long-running AI generation jobs run via `asyncio.create_task` or a lightweight task worker updating the `AgentTask` database entity.
- The frontend actively polls `GET /tasks/:id` every 1000ms until status is `succeeded` or `failed`.

#### 7. AI & External Integrations [REQUIRED]
- **LLM Abstraction Layer:** Centralized LLM Router supporting `mock` (canned responses for zero-cost testing), `anthropic` (Claude 3.5 Sonnet for reasoning), and `openai` (GPT-4o).
- **Web Search:** Abstracted search provider supporting `mock`, `tavily`, and `serper`.
- **Meta Marketing API:** Direct integration with Meta Graph API v20.0 for OAuth authorization, campaign synchronization, ad set creation, and performance telemetry fetching.

---

# PHASE 5 — DATA MODEL

The data model is derived directly from the Zod schemas in `SAHM-AI-frontend/src/types/index.ts`.

```
+--------------------+        1:1        +--------------------+
|      Business      | <---------------> |    BrandProfile    |
+--------------------+                   +--------------------+
       |          |
       | 1:N      | 1:N
       v          v
+-----------+  +--------------------+    1:N     +--------------------+
|  Audience |  |      Campaign      | ---------> |      Creative      |
+-----------+  +--------------------+            +--------------------+
                      |       |
                 1:N  |       | 1:N
                      v       v
         +----------------+ +--------------------+
         | Recommendation | |      Insight       |
         +----------------+ +--------------------+

+--------------------+        1:N        +--------------------+
|    Conversation    | ----------------> |      Message       |
+--------------------+                   +--------------------+

+--------------------+                   +--------------------+
|    ContentItem     |                   |     AgentTask      |
+--------------------+                   +--------------------+
```

### Entity Specifications

#### 1. Business
- **Purpose:** Primary business tenant entity.
- **Fields:**
  - `id` (String, PK, e.g. `biz_1`) `[OBSERVED]`
  - `name` (String) `[OBSERVED]`
  - `industry` (String) `[OBSERVED]`
  - `website` (String, Optional) `[OBSERVED]`
  - `country` (String, 2-letter ISO) `[OBSERVED]`
  - `description` (Text) `[OBSERVED]`
  - `createdAt` (DateTime, UTC) `[OBSERVED]`
- **Relationships:** Has one `BrandProfile`; has many `Campaigns`, `Audiences`, `ContentItems`.
- **Frontend Usage:** Settings view (`SettingsView`), brand context chips.

#### 2. BrandProfile
- **Purpose:** Persistent brand identity, voice guidelines, and constraints.
- **Fields:**
  - `id` (String, PK) `[OBSERVED]`
  - `businessId` (String, FK -> `Business.id`, Unique) `[OBSERVED]`
  - `tone` (JSON Array of Strings, e.g. `["warm", "confident"]`) `[OBSERVED]`
  - `values` (JSON Array of Strings) `[OBSERVED]`
  - `primaryColor` (String, Hex) `[OBSERVED]`
  - `secondaryColor` (String, Hex, Optional) `[OBSERVED]`
  - `logoUrl` (String, URL, Optional) `[OBSERVED]`
  - `tagline` (String, Optional) `[OBSERVED]`
  - `doNotUse` (JSON Array of Strings, forbidden words) `[OBSERVED]`
  - `languages` (JSON Array of Strings: `["en", "ar"]`) `[OBSERVED]`
- **Frontend Usage:** Brand settings, prompt injection for ad copy and creative generation.

#### 3. Audience
- **Purpose:** Demographic and behavioral targeting parameters for campaigns.
- **Fields:**
  - `id` (String, PK, e.g. `aud_1`) `[OBSERVED]`
  - `businessId` (String, FK -> `Business.id`, Indexed) `[PROPOSED]`
  - `name` (String) `[OBSERVED]`
  - `ageMin` (Integer, 13–65) `[OBSERVED]`
  - `ageMax` (Integer, 13–65) `[OBSERVED]`
  - `genders` (JSON Array of Strings: `["male" | "female" | "all"]`) `[OBSERVED]`
  - `locations` (JSON Array of Strings) `[OBSERVED]`
  - `interests` (JSON Array of Strings) `[OBSERVED]`
  - `behaviors` (JSON Array of Strings) `[OBSERVED]`
  - `estimatedReach` (Integer, Optional) `[OBSERVED]`
- **Frontend Usage:** Campaign wizard Step 2, Campaign Detail Audience Tab.

#### 4. Campaign
- **Purpose:** Central paid advertising campaign container.
- **Fields:**
  - `id` (String, PK, e.g. `cmp_1`) `[OBSERVED]`
  - `businessId` (String, FK -> `Business.id`, Indexed) `[OBSERVED]`
  - `name` (String, min 2 chars) `[OBSERVED]`
  - `objective` (Enum: `awareness`, `traffic`, `engagement`, `leads`, `sales`, `app_installs`) `[OBSERVED]`
  - `status` (Enum: `draft`, `scheduled`, `active`, `paused`, `completed`, `failed`) `[OBSERVED]`
  - `dailyBudget` (Float, > 0) `[OBSERVED]`
  - `currency` (String, 3-letter, default `USD`) `[OBSERVED]`
  - `startDate` (DateTime, ISO-8601) `[OBSERVED]`
  - `endDate` (DateTime, ISO-8601, Nullable) `[OBSERVED]`
  - `audienceId` (String, FK -> `Audience.id`, Nullable) `[OBSERVED]`
  - `strategy` (JSON, contains summary, keyMessages, channels, placements, biddingStrategy, phases, kpis) `[OBSERVED]`
  - `creativeIds` (JSON Array of String IDs) `[OBSERVED]`
  - `metaCampaignId` (String, Nullable, Meta Graph API ID) `[OBSERVED]`
  - `metrics` (JSON, impressions, reach, clicks, ctr, cpc, cpm, conversions, cpa, roas, spend) `[OBSERVED]`
  - `healthScore` (Integer, 0–100, Nullable) `[OBSERVED]`
  - `createdAt` (DateTime) `[OBSERVED]`
  - `updatedAt` (DateTime) `[OBSERVED]`
- **Frontend Usage:** `/campaigns`, `/campaigns/[id]`, Home `ActiveCampaigns`.

#### 5. CampaignMetricPoint
- **Purpose:** Daily performance history point for time-series charts.
- **Fields:**
  - `id` (String, PK) `[PROPOSED]`
  - `campaignId` (String, FK -> `Campaign.id`, Indexed) `[OBSERVED]`
  - `date` (String, `YYYY-MM-DD`, Indexed) `[OBSERVED]`
  - `impressions` (Integer) `[OBSERVED]`
  - `clicks` (Integer) `[OBSERVED]`
  - `conversions` (Integer) `[OBSERVED]`
  - `spend` (Float) `[OBSERVED]`
- **Frontend Usage:** Campaign Detail Performance Tab (`14-day chart`).

#### 6. Creative
- **Purpose:** Ad creative unit (image, video, or copy).
- **Fields:**
  - `id` (String, PK, e.g. `cr_1`) `[OBSERVED]`
  - `businessId` (String, FK -> `Business.id`, Indexed) `[PROPOSED]`
  - `campaignId` (String, FK -> `Campaign.id`, Nullable, Indexed) `[OBSERVED]`
  - `kind` (Enum: `image`, `video`, `copy`) `[OBSERVED]`
  - `title` (String) `[OBSERVED]`
  - `status` (Enum: `draft`, `generating`, `ready`, `approved`, `rejected`) `[OBSERVED]`
  - `assetUrl` (String, URL, Optional) `[OBSERVED]`
  - `thumbnailUrl` (String, URL, Optional) `[OBSERVED]`
  - `headline` (String, Optional) `[OBSERVED]`
  - `primaryText` (Text, Optional) `[OBSERVED]`
  - `cta` (String, Optional) `[OBSERVED]`
  - `durationSec` (Float, Optional) `[OBSERVED]`
  - `aspectRatio` (Enum: `1:1`, `4:5`, `9:16`, `16:9`, Optional) `[OBSERVED]`
  - `tags` (JSON Array of Strings) `[OBSERVED]`
  - `score` (Float, 0–100, Optional) `[OBSERVED]`
  - `createdAt` (DateTime) `[OBSERVED]`
  - `updatedAt` (DateTime) `[OBSERVED]`
- **Frontend Usage:** `/creatives`, Campaign Wizard Step 4, Campaign Creatives Tab.

#### 7. ReferenceAnalysis
- **Purpose:** Deconstructed analysis of an uploaded competitor or reference ad.
- **Fields:**
  - `id` (String, PK) `[OBSERVED]`
  - `sourceUrl` (String, URL, Optional) `[OBSERVED]`
  - `sourceName` (String) `[OBSERVED]`
  - `kind` (Enum: `image`, `video`, `copy`) `[OBSERVED]`
  - `hook` (Text) `[OBSERVED]`
  - `structure` (JSON Array of Strings) `[OBSERVED]`
  - `visualStyle` (JSON Array of Strings) `[OBSERVED]`
  - `tone` (JSON Array of Strings) `[OBSERVED]`
  - `cta` (String, Optional) `[OBSERVED]`
  - `strengths` (JSON Array of Strings) `[OBSERVED]`
  - `weaknesses` (JSON Array of Strings) `[OBSERVED]`
  - `estimatedPerformance` (Enum: `low`, `medium`, `high`) `[OBSERVED]`
  - `inspiredBrief` (Text) `[OBSERVED]`
  - `createdAt` (DateTime) `[OBSERVED]`
- **Frontend Usage:** `AnalyzeReferenceDialog`, creative generation inspiration prefill.

#### 8. Insight
- **Purpose:** Diagnostic observation and recommended course of action.
- **Fields:**
  - `id` (String, PK, e.g. `ins_1`) `[OBSERVED]`
  - `campaignId` (String, FK -> `Campaign.id`, Nullable, Indexed) `[OBSERVED]`
  - `creativeId` (String, FK -> `Creative.id`, Nullable, Indexed) `[OBSERVED]`
  - `severity` (Enum: `info`, `warning`, `critical`, Indexed) `[OBSERVED]`
  - `title` (String) `[OBSERVED]`
  - `observation` (Text) `[OBSERVED]`
  - `explanation` (Text) `[OBSERVED]`
  - `recommendedAction` (Text) `[OBSERVED]`
  - `action` (JSON, `ActionRef` object `{ type, label, payload }`, Nullable) `[OBSERVED]`
  - `metric` (JSON, `{ label, value, unit, delta }`, Optional) `[OBSERVED]`
  - `createdAt` (DateTime) `[OBSERVED]`
- **Frontend Usage:** Home `NeedsAttention`, `/insights`.

#### 9. Recommendation
- **Purpose:** Actionable campaign optimization proposal containing human-in-the-loop diffs.
- **Fields:**
  - `id` (String, PK, e.g. `rec_1`) `[OBSERVED]`
  - `campaignId` (String, FK -> `Campaign.id`, Indexed) `[OBSERVED]`
  - `title` (String) `[OBSERVED]`
  - `rationale` (Text) `[OBSERVED]`
  - `expectedImpact` (String) `[OBSERVED]`
  - `confidence` (Float, 0–1) `[OBSERVED]`
  - `changes` (JSON Array of `ChangeDiff` objects: `[{ field, label, before, after }]`) `[OBSERVED]`
  - `status` (Enum: `pending`, `applied`, `dismissed`, Indexed) `[OBSERVED]`
  - `createdAt` (DateTime) `[OBSERVED]`
- **Frontend Usage:** Campaign Recommendations Tab, Chat `recommendation` block cards.

#### 10. Learning
- **Purpose:** Empirical knowledge discovered from campaign execution.
- **Fields:**
  - `id` (String, PK) `[OBSERVED]`
  - `campaignId` (String, FK -> `Campaign.id`, Indexed) `[OBSERVED]`
  - `category` (Enum: `audience`, `creative`, `timing`, `budget`, `messaging`) `[OBSERVED]`
  - `statement` (Text) `[OBSERVED]`
  - `evidence` (Text) `[OBSERVED]`
  - `confidence` (Float, 0–1) `[OBSERVED]`
  - `createdAt` (DateTime) `[OBSERVED]`
- **Frontend Usage:** Campaign Detail Learnings Tab.

#### 11. Conversation & Message
- **Purpose:** Chat thread and individual block-based chat messages.
- **Conversation Fields:**
  - `id` (String, PK) `[OBSERVED]`
  - `title` (String) `[OBSERVED]`
  - `lastMessagePreview` (String) `[OBSERVED]`
  - `messageCount` (Integer) `[OBSERVED]`
  - `createdAt` (DateTime) `[OBSERVED]`
  - `updatedAt` (DateTime) `[OBSERVED]`
- **Message Fields:**
  - `id` (String, PK) `[OBSERVED]`
  - `conversationId` (String, FK -> `Conversation.id`, Indexed) `[OBSERVED]`
  - `role` (Enum: `user`, `assistant`, `system`) `[OBSERVED]`
  - `blocks` (JSON Array of discriminated union blocks) `[OBSERVED]`
  - `context` (JSON Array of `ContextChip` objects) `[OBSERVED]`
  - `createdAt` (DateTime) `[OBSERVED]`
- **Frontend Usage:** `/assistant`, `/assistant/[id]`, research task landing destination.

#### 12. AgentTask
- **Purpose:** Asynchronous agent workflow tracking.
- **Fields:**
  - `id` (String, PK, e.g. `task_1`) `[OBSERVED]`
  - `type` (Enum: `ActionType`) `[OBSERVED]`
  - `status` (Enum: `queued`, `running`, `succeeded`, `failed`) `[OBSERVED]`
  - `label` (String) `[OBSERVED]`
  - `progress` (Float, 0–100) `[OBSERVED]`
  - `steps` (JSON Array of `{ label: string, status: string }`) `[OBSERVED]`
  - `resultRef` (JSON, `{ entity: string, id: string }`, Nullable) `[OBSERVED]`
  - `error` (Text, Nullable) `[OBSERVED]`
  - `createdAt` (DateTime) `[OBSERVED]`
  - `updatedAt` (DateTime) `[OBSERVED]`
- **Frontend Usage:** Polled by `useTask` hook, displayed in `TaskProgressBlockView`.

#### 13. ContentItem
- **Purpose:** Marketing content calendar post/reel/story record.
- **Fields:**
  - `id` (String, PK) `[OBSERVED]`
  - `businessId` (String, FK -> `Business.id`, Indexed) `[PROPOSED]`
  - `date` (String, `YYYY-MM-DD`, Indexed) `[OBSERVED]`
  - `time` (String, `HH:mm`, Optional) `[OBSERVED]`
  - `channel` (Enum: `instagram`, `facebook`, `tiktok`, `email`, `blog`, Indexed) `[OBSERVED]`
  - `format` (Enum: `post`, `reel`, `story`, `ad`, `email`, `article`) `[OBSERVED]`
  - `title` (String) `[OBSERVED]`
  - `hook` (Text, Optional) `[OBSERVED]`
  - `caption` (Text, Optional) `[OBSERVED]`
  - `hashtags` (JSON Array of Strings) `[OBSERVED]`
  - `status` (Enum: `idea`, `draft`, `scheduled`, `published`) `[OBSERVED]`
  - `campaignId` (String, FK -> `Campaign.id`, Nullable, Indexed) `[OBSERVED]`
  - `creativeId` (String, FK -> `Creative.id`, Nullable) `[OBSERVED]`
  - `pillar` (String, Optional) `[OBSERVED]`
  - `createdAt` (DateTime) `[OBSERVED]`
  - `updatedAt` (DateTime) `[OBSERVED]`
- **Frontend Usage:** `/calendar`, `ContentItemSheet`, `GenerateCalendarDialog`.

#### 14. GeneratedText
- **Purpose:** Copywriting generation result (briefs, scripts, prompts).
- **Fields:**
  - `id` (String, PK) `[OBSERVED]`
  - `tool` (Enum: `generate_creative_brief`, `generate_commercial_script`, `generate_image_prompt`, `generate_video_prompt`, `generate_ad_copy`) `[OBSERVED]`
  - `title` (String) `[OBSERVED]`
  - `sections` (JSON Array of `{ heading: string, body: string }`) `[OBSERVED]`
  - `plainText` (Text) `[OBSERVED]`
  - `createdAt` (DateTime) `[OBSERVED]`
- **Frontend Usage:** `GeneratedTextDialog` ("Write with AI").

#### 15. AgentPersona
- **Purpose:** Configurable persona parameters for each of the 5 AI roles.
- **Fields:**
  - `id` (String, PK, e.g. `agent_researcher`) `[OBSERVED]`
  - `role` (Enum: `researcher`, `strategist`, `creative`, `analyst`, `optimizer`) `[OBSERVED]`
  - `name` (String) `[OBSERVED]`
  - `emoji` (String) `[OBSERVED]`
  - `description` (Text) `[OBSERVED]`
  - `persona` (JSON, `{ tone, language, instructions, signOff }`) `[OBSERVED]`
  - `allowedTools` (JSON Array of String tool names) `[OBSERVED]`
  - `availableTools` (JSON Array of String tool names) `[OBSERVED]`
  - `autonomy` (Enum: `ask_first`, `suggest`, `auto_draft`) `[OBSERVED]`
  - `enabled` (Boolean) `[OBSERVED]`
  - `updatedAt` (DateTime) `[OBSERVED]`
- **Frontend Usage:** `/agents`, `AgentEditorSheet`.

#### 16. MetaConnection
- **Purpose:** Meta OAuth status and advertising account metadata.
- **Fields:**
  - `id` (String, PK) `[PROPOSED]`
  - `businessId` (String, FK -> `Business.id`, Unique) `[PROPOSED]`
  - `status` (Enum: `not_connected`, `connecting`, `connected`, `needs_reauth`, `error`) `[OBSERVED]`
  - `accountId` (String, Nullable) `[OBSERVED]`
  - `accountName` (String, Nullable) `[OBSERVED]`
  - `pageName` (String, Nullable) `[OBSERVED]`
  - `permissions` (JSON Array of Strings) `[OBSERVED]`
  - `connectedAt` (DateTime, Nullable) `[OBSERVED]`
  - `expiresAt` (DateTime, Nullable) `[OBSERVED]`
  - `errorMessage` (Text, Nullable) `[OBSERVED]`
  - `accessTokenEncrypted` (Text, Nullable) `[PROPOSED]`
- **Frontend Usage:** `MetaConnectionCard`, Campaign Wizard Step 6 guard.

#### 17. Settings
- **Purpose:** Workspace locale, currency, and notification configurations.
- **Fields:**
  - `id` (String, PK) `[PROPOSED]`
  - `businessId` (String, FK -> `Business.id`, Unique) `[PROPOSED]`
  - `locale` (Enum: `en`, `ar`) `[OBSERVED]`
  - `timezone` (String, e.g. `Africa/Cairo`) `[OBSERVED]`
  - `currency` (String, 3-letter, e.g. `EGP`, `USD`) `[OBSERVED]`
  - `notifications` (JSON, `{ email: bool, criticalInsights: bool, weeklyDigest: bool }`) `[OBSERVED]`
- **Frontend Usage:** `/settings`, `SettingsView`.

---

# PHASE 6 — API CONTRACT

All endpoints are prefixed with `/api/v1` (or `${NEXT_PUBLIC_API_BASE}`). Responses return JSON validating against the schemas below.

### 1. Business & Brand Profile
- **`GET /business`**
  - **Purpose:** Retrieve active business details.
  - **Auth:** Session cookie (`credentials: "include"`).
  - **Response:** `200 OK` -> `Business`
  - **Frontend Callers:** `src/hooks/use-api.ts` -> `useBusiness()`.
- **`GET /brand-profile`**
  - **Purpose:** Retrieve active brand profile, voice, and rules.
  - **Response:** `200 OK` -> `BrandProfile`
  - **Frontend Callers:** `src/hooks/use-api.ts` -> `useBrandProfile()`.

### 2. Campaigns
- **`GET /campaigns`**
  - **Query Params:** `status?: CampaignStatus`, `search?: string`
  - **Response:** `200 OK` -> `Campaign[]`
  - **Frontend Callers:** `useCampaigns()` in `/campaigns`, Home `ActiveCampaigns`.
- **`GET /campaigns/:id`**
  - **Response:** `200 OK` -> `Campaign` | `404 Not Found` -> `ApiError`
  - **Frontend Callers:** `useCampaign(id)` in `/campaigns/[id]`.
- **`POST /campaigns`**
  - **Purpose:** Create draft campaign or launch immediately to Meta.
  - **Request Body:** `CampaignDraft`
  - **Response:** `201 Created` -> `Campaign`
  - **Frontend Callers:** `useCreateCampaign()` in Campaign Wizard.
- **`PATCH /campaigns/:id`**
  - **Purpose:** Update status (pause/resume), daily budget, name, or endDate.
  - **Request Body:** `{ name?: string, status?: CampaignStatus, dailyBudget?: number, endDate?: string | null }`
  - **Response:** `200 OK` -> `Campaign`
  - **Frontend Callers:** Campaign Detail Header Actions via `UPDATE_CAMPAIGN` action.
- **`POST /campaigns/strategy`**
  - **Purpose:** Generate marketing strategy from objective and audience brief.
  - **Request Body:** `GenerateStrategyInput` `{ objective, dailyBudget, audience, brief? }`
  - **Response:** `200 OK` -> `CampaignStrategy`
  - **Frontend Callers:** `useGenerateStrategy()` in Wizard Step 3.
- **`GET /campaigns/:id/performance`**
  - **Response:** `200 OK` -> `MetricPoint[]` (ordered oldest to newest)
  - **Frontend Callers:** `useCampaignPerformance(id)` in Performance Tab.
- **`GET /campaigns/:id/recommendations`**
  - **Response:** `200 OK` -> `Recommendation[]`
  - **Frontend Callers:** `useCampaignRecommendations(id)` in Recommendations Tab.
- **`GET /campaigns/:id/learnings`**
  - **Response:** `200 OK` -> `Learning[]`
  - **Frontend Callers:** `useCampaignLearnings(id)` in Learnings Tab.
- **`GET /audiences/:id`**
  - **Response:** `200 OK` -> `Audience`
  - **Frontend Callers:** `useAudience(id)` in Audience Tab.
- **`POST /campaigns/:campaignId/creatives`**
  - **Request Body:** `{ creativeId: string }`
  - **Response:** `200 OK` -> `Campaign` (updated with new `creativeIds`)
  - **Frontend Callers:** `ADD_TO_CAMPAIGN` action handler.

### 3. Creatives
- **`GET /creatives`**
  - **Query Params:** `campaignId?: string`
  - **Response:** `200 OK` -> `Creative[]`
  - **Frontend Callers:** `useCreatives()` in `/creatives` and Wizard Step 4.
- **`GET /creatives/:id`**
  - **Response:** `200 OK` -> `Creative`
  - **Frontend Callers:** `api.getCreative(id)`.
- **`POST /creatives/generate`**
  - **Request Body:** `GenerateCreativeInput` `{ kind, brief, campaignId?, referenceAnalysisId?, aspectRatio?, count }`
  - **Response:** `201 Created` -> `Creative[]` (length equal to `count`)
  - **Frontend Callers:** `CREATE_CREATIVE` action handler in `CreateCreativeDialog`.
- **`POST /creatives/:creativeId/variations`**
  - **Request Body:** `GenerateVariationsInput` `{ creativeId: string, count: number }`
  - **Response:** `201 Created` -> `Creative[]`
  - **Frontend Callers:** `GENERATE_VARIATIONS` action in `CreativeCard`.
- **`POST /creatives/analyze-reference`**
  - **Request Body:** `AnalyzeReferenceInput` `{ source: string, fileName?: string, kind: CreativeKind }`
  - **Response:** `200 OK` -> `ReferenceAnalysis`
  - **Frontend Callers:** `useAnalyzeReference()` in `AnalyzeReferenceDialog`.

### 4. Insights & Recommendations
- **`GET /insights`**
  - **Query Params:** `campaignId?: string`, `severity?: Severity`
  - **Response:** `200 OK` -> `Insight[]`
  - **Frontend Callers:** `useInsights()` in `/insights` and Home `NeedsAttention`.
- **`GET /recommendations/:id`**
  - **Response:** `200 OK` -> `Recommendation`
  - **Frontend Callers:** `APPLY_RECOMMENDATION` action `getDiff`.
- **`POST /recommendations/:id/apply`**
  - **Purpose:** Applies the exact `ChangeDiff[]` approved by the user.
  - **Response:** `200 OK` -> `Recommendation` (`status: "applied"`)
  - **Frontend Callers:** `APPLY_RECOMMENDATION` action handler.
- **`POST /recommendations/:id/dismiss`**
  - **Response:** `200 OK` -> `Recommendation` (`status: "dismissed"`)
  - **Frontend Callers:** `useDismissRecommendation()` in Recommendations Tab.

### 5. Assistant & Chat Streaming
- **`GET /conversations`**
  - **Response:** `200 OK` -> `Conversation[]` (ordered by `updatedAt` desc)
  - **Frontend Callers:** `useConversations()` in Assistant sidebar and Research view.
- **`GET /conversations/:id`**
  - **Response:** `200 OK` -> `{ conversation: Conversation, messages: Message[] }`
  - **Frontend Callers:** `useConversation(id)` in `/assistant/[id]`.
- **`POST /conversations/messages`**
  - **Request Body:** `SendMessageInput` `{ conversationId: string | null, text: string, context: ContextChip[], answer? }`
  - **Response:** `200 OK` -> `SendMessageOutput` `{ conversation, userMessage, assistantMessage }`
  - **Frontend Callers:** Non-streaming fallback in `api.sendMessage()`.
- **`POST /conversations/messages/stream` (SSE)**
  - **Purpose:** Real-time multi-agent message generator.
  - **Headers:** `Accept: text/event-stream`
  - **Payload Format:** Stream of `data: <SendMessageOutput JSON>\n\n`.
  - **Streaming Behavior:** In-progress text blocks include `streaming: true`. The final snapshot has `streaming: false` on all blocks.
  - **Frontend Callers:** `useChat()` hook in `src/hooks/use-chat.ts`.

### 6. Agent Tasks
- **`POST /tasks`**
  - **Request Body:** `StartTaskInput` `{ type: ActionType, payload: Record<string, unknown> }`
  - **Response:** `202 Accepted` -> `AgentTask`
  - **Behavior:** Backend spawns an async agent task, creates a linked conversation, sets `resultRef = { entity: "conversation", id }`, and returns immediately.
  - **Frontend Callers:** `startTask` in `src/lib/actions/registry.ts` for all 9 research tools.
- **`GET /tasks/:id`**
  - **Response:** `200 OK` -> `AgentTask`
  - **Frontend Callers:** `useTask(id)` polling every 1000ms.

### 7. Content Calendar
- **`GET /content`**
  - **Query Params:** `month: string` (`YYYY-MM`), `campaignId?: string`
  - **Response:** `200 OK` -> `ContentItem[]`
  - **Frontend Callers:** `useContentItems(input)` in `/calendar`.
- **`POST /content/generate`**
  - **Request Body:** `GenerateContentCalendarInput` `{ month, campaignId?, postsPerWeek, channels, brief? }`
  - **Response:** `200 OK` -> `ContentItem[]` (replaces previous items for that month/campaign)
  - **Frontend Callers:** `GENERATE_CONTENT_CALENDAR` action.
- **`PATCH /content/:id`**
  - **Request Body:** `UpdateContentItemInput.patch`
  - **Response:** `200 OK` -> `ContentItem`
  - **Frontend Callers:** `useUpdateContentItem()` in `ContentItemSheet`.
- **`DELETE /content/:id`**
  - **Response:** `200 OK` -> `{ id: string }`
  - **Frontend Callers:** `useDeleteContentItem()` in `ContentItemSheet`.

### 8. Text Generation Tools
- **`POST /generate/text`**
  - **Request Body:** `GenerateTextInput` `{ tool, brief, campaignId?, creativeId?, language }`
  - **Response:** `200 OK` -> `GeneratedText` `{ id, tool, title, sections, plainText, createdAt }`
  - **Frontend Callers:** `GENERATE_TEXT` action in `GenerateTextDialog`.

### 9. Agents & Personas
- **`GET /agents`**
  - **Response:** `200 OK` -> `Agent[]` (5 personas)
  - **Frontend Callers:** `useAgents()` in `/agents`.
- **`PATCH /agents/:id`**
  - **Request Body:** `UpdateAgentInput.patch` `{ name?, emoji?, persona?, allowedTools?, autonomy?, enabled? }`
  - **Response:** `200 OK` -> `Agent`
  - **Frontend Callers:** `UPDATE_AGENT` action in `AgentEditorSheet`.

### 10. Integrations (Meta)
- **`GET /integrations/meta`**
  - **Response:** `200 OK` -> `MetaConnection`
  - **Frontend Callers:** `useMetaConnection()` polling while status is `connecting`.
- **`POST /integrations/meta/connect`**
  - **Response:** `200 OK` -> `MetaConnection`
  - **Frontend Callers:** `CONNECT_META` action (mode: `connect`).
- **`POST /integrations/meta/reauth`**
  - **Response:** `200 OK` -> `MetaConnection`
  - **Frontend Callers:** `CONNECT_META` action (mode: `reauth`).
- **`POST /integrations/meta/disconnect`**
  - **Response:** `200 OK` -> `MetaConnection` (`status: "not_connected"`)
  - **Frontend Callers:** `CONNECT_META` action (mode: `disconnect`).

### 11. Settings
- **`GET /settings`**
  - **Response:** `200 OK` -> `Settings`
  - **Frontend Callers:** `useSettings()` in `/settings`.
- **`PATCH /settings`**
  - **Request Body:** `Partial<Settings>`
  - **Response:** `200 OK` -> `Settings`
  - **Frontend Callers:** `useUpdateSettings()` in `SettingsView`.

---

# PHASE 7 — URGENT BACKEND WORK

## Critical Path

The following diagram defines the strict technical execution order required to make the frontend functional:

```
[1. Database Engine & Schema Migrations]
                     │
                     ▼
[2. Core Shared Models & Pydantic Envelopes]
                     │
                     ▼
[3. Session & Workspace Dependency Injection]
                     │
                     ▼
[4. Core Metadata APIs: Business, Brand, Settings, Audiences]
                     │
                     ├───────────────────────────────┐
                     ▼                               ▼
[5. Campaign CRUD & Mutating APIs]       [6. Task Engine & SSE Streaming]
                     │                               │
                     ▼                               ▼
[7. Creative Generation & Variations]    [8. Agent Orchestrator & Tool Registry]
                     │                               │
                     ├───────────────────────────────┘
                     ▼
[9. Recommendations & Diffs Execution]
                     │
                     ▼
[10. Meta Marketing API & Content Calendar]
```

### Critical Path Steps:
1. **Database Engine & Alembic Migrations:** Resolve `greenlet` dependency for async SQLite, initialize tables matching Phase 5.
2. **Core Shared Schemas:** Implement Pydantic v2 schemas in `app/schemas/` mirroring `src/types/index.ts` with strict alias and validation rules.
3. **Session & Workspace Provider:** Implement FastAPI dependency `get_current_business()` to guarantee a valid `business_id` on all authenticated routes.
4. **Foundational APIs:** Stand up `/business`, `/brand-profile`, `/settings`, and `/audiences/:id` so the frontend application shell loads without mock fallbacks.
5. **Campaign Engine:** Build `/campaigns` CRUD, budget updates, and `/campaigns/strategy` generator.
6. **Task Runner & SSE Protocol:** Stand up the Server-Sent Events generator and background task state manager.
7. **Creatives & Assets:** Implement creative generation and reference analysis pipelines.
8. **Recommendation Diff Engine:** Enforce before/after diff computation and transaction-safe application.
9. **Meta Ads & Calendar:** Wire Meta OAuth state machine and month-scoped content calendar generator.

---

# PHASE 8 — SPLIT THE WORK FOR 5 PEOPLE

```
+----------------------------------------------------------------------------------------------------+
|                                    STAGE 0: SHARED CONTRACTS                                       |
|  - API error envelope: { code, message, details }                                                 |
|  - Pydantic models matching Zod                                                                    |
|  - Database schema & Alembic baseline                                                              |
+----------------------------------------------------------------------------------------------------+
       |                              |                              |                               |
       v                              v                              v                               v
+--------------+              +---------------+              +---------------+              +---------------+
|   PERSON 1   |              |   PERSON 2    |              |   PERSON 3    |              |   PERSON 4    |
| Foundation   |              | Campaigns &   |              | Creatives &   |              | Multi-Agent & |
| DB, Config,  |              | Strategy      |              | Asset Gen     |              | SSE Chat      |
| Business,    |              | - Campaign    |              | - Creative    |              | - SSE stream  |
| Audiences,   |              |   CRUD        |              |   generation  |              | - LangGraph   |
| Settings     |              | - Strategy    |              | - Variations  |              |   agents      |
+--------------+              | - Recs &      |              | - Reference   |              | - Tool        |
       |                      |   Diffs       |              |   analysis    |              |   registry    |
       |                      +---------------+              +---------------+              +---------------+
       |                              |                              |                               |
       +------------------------------+------------------------------+-------------------------------+
                                      |
                                      v
                              +---------------+
                              |   PERSON 5    |
                              | Calendar &    |
                              | Meta Ads      |
                              | - Calendar    |
                              |   month grid  |
                              | - Meta OAuth  |
                              |   & sync      |
                              | - Insights    |
                              +---------------+
```

---

### Person 1 — Foundation, Core Data Model, Business & Settings
**Area:** Core Infrastructure, Database, Configuration, and Foundation APIs.

- **Responsibilities:**
  - Fix backend virtual environment dependencies (add `greenlet` to enable `sqlalchemy[asyncio]`).
  - Configure SQLAlchemy 2.0 async engine and write Alembic migrations for all Phase 5 tables.
  - Implement request-id tracing middleware, CORS configuration, and unified `ApiError` exception handlers.
  - Implement workspace/business context resolver (`get_current_business`).
  - Implement Business, Brand Profile, Audience, and Settings endpoints.
- **Files / Modules Owned:**
  - `SAHM-AI-backend/app/main.py`
  - `SAHM-AI-backend/app/config.py`
  - `SAHM-AI-backend/app/db/models.py`
  - `SAHM-AI-backend/app/db/session.py`
  - `SAHM-AI-backend/app/api/business.py`
  - `SAHM-AI-backend/app/api/settings.py`
  - `SAHM-AI-backend/app/api/audiences.py` (new)
  - `SAHM-AI-backend/app/schemas/business.py`, `settings.py`, `common.py`
- **APIs Owned:**
  - `GET /business`
  - `GET /brand-profile`
  - `GET /audiences/:id`
  - `GET /settings`
  - `PATCH /settings`
- **Database Entities Owned:**
  - `Business`, `BrandProfile`, `Audience`, `Settings`
- **Dependencies:**
  - None. Acts as the foundation for Developers 2–5.
- **Can Start Immediately:** **YES**

---

### Person 2 — Campaigns, Strategy Generation & Recommendation Diffs
**Area:** Paid Advertising Lifecycle and Human-in-the-Loop Optimization.

- **Responsibilities:**
  - Build campaign listing with status filtering and text search.
  - Implement campaign creation wizard backend, handling draft saves and status initialization.
  - Implement `PATCH /campaigns/:id` for budget and status changes.
  - Build AI Campaign Strategy Generator (`POST /campaigns/strategy`) using brand context.
  - Implement 14-day performance metric generator and learnings endpoints.
  - Build the Recommendation Diff engine: generate `ChangeDiff` objects and execute approved patches.
- **Files / Modules Owned:**
  - `SAHM-AI-backend/app/api/campaigns.py`
  - `SAHM-AI-backend/app/api/recommendations.py`
  - `SAHM-AI-backend/app/core/campaigns/strategy.py`
  - `SAHM-AI-backend/app/core/approvals/`
  - `SAHM-AI-backend/app/schemas/campaigns.py`, `recommendations.py`
- **APIs Owned:**
  - `GET /campaigns`
  - `GET /campaigns/:id`
  - `POST /campaigns`
  - `PATCH /campaigns/:id`
  - `POST /campaigns/strategy`
  - `GET /campaigns/:id/performance`
  - `GET /campaigns/:id/recommendations`
  - `GET /campaigns/:id/learnings`
  - `POST /campaigns/:campaignId/creatives`
  - `GET /recommendations/:id`
  - `POST /recommendations/:id/apply`
  - `POST /recommendations/:id/dismiss`
- **Database Entities Owned:**
  - `Campaign`, `CampaignMetricPoint`, `Recommendation`, `Learning`
- **Dependencies:**
  - Depends on Person 1's DB models and Pydantic schemas. Can work against mock schemas immediately.
- **Can Start Immediately:** **YES** (with Pydantic schemas mock-bound).

---

### Person 3 — Creatives, Media Pipelines & Reference Ad Deconstruction
**Area:** Creative Generation, Asset Storage, and Multimodal Ad Analysis.

- **Responsibilities:**
  - Implement creative asset listing and querying by campaign.
  - Build image and copy generation pipelines (`POST /creatives/generate`).
  - Implement creative variation generator (`POST /creatives/:id/variations`).
  - Build multimodal reference ad analyzer (`POST /creatives/analyze-reference`) accepting Base64 data URLs or remote URLs.
  - Extract hooks, visual styles, tones, and structure to generate the `inspiredBrief`.
  - Build text generation endpoint (`POST /generate/text`) supporting the 5 copywriting tools.
- **Files / Modules Owned:**
  - `SAHM-AI-backend/app/api/creatives.py`
  - `SAHM-AI-backend/app/api/generate.py`
  - `SAHM-AI-backend/app/core/creatives/`
  - `SAHM-AI-backend/app/core/vision/`
  - `SAHM-AI-backend/app/schemas/creatives.py`
- **APIs Owned:**
  - `GET /creatives`
  - `GET /creatives/:id`
  - `POST /creatives/generate`
  - `POST /creatives/:creativeId/variations`
  - `POST /creatives/analyze-reference`
  - `POST /generate/text`
- **Database Entities Owned:**
  - `Creative`, `ReferenceAnalysis`, `GeneratedText`
- **Dependencies:**
  - Depends on Person 1's models and storage configuration.
- **Can Start Immediately:** **YES** (can build prompt pipelines and vision analyzers in isolation).

---

### Person 4 — Multi-Agent Orchestrator, SSE Chat Streaming & Task Engine
**Area:** LangGraph Agents, Tool Execution, Realtime Streaming, and Asynchronous Jobs.

- **Responsibilities:**
  - Implement Server-Sent Events (SSE) streaming endpoint (`POST /conversations/messages/stream`) outputting full `SendMessageOutput` snapshots.
  - Implement asynchronous task runner (`POST /tasks`, `GET /tasks/:id`) with real-time progress steps.
  - Wire the 9 research tools (Website, Competitor, Social Profile, Keywords, Market Gaps, etc.) into the tool registry.
  - Implement SSRF-safe URL scraping and search provider connectors (Tavily/Serper).
  - Implement interactive block generation (`findings`, `metric`, `action_card`, `task_progress`, `question_with_options`).
  - Support user answers to interactive questions (`answer: { messageId, optionIds }`).
- **Files / Modules Owned:**
  - `SAHM-AI-backend/app/api/assistant.py`
  - `SAHM-AI-backend/app/api/tasks.py`
  - `SAHM-AI-backend/app/core/orchestrator/`
  - `SAHM-AI-backend/app/core/tools/`
  - `SAHM-AI-backend/app/core/llm/`
  - `SAHM-AI-backend/app/schemas/assistant.py`, `tasks.py`
- **APIs Owned:**
  - `GET /conversations`
  - `GET /conversations/:id`
  - `POST /conversations/messages`
  - `POST /conversations/messages/stream` (SSE)
  - `POST /tasks`
  - `GET /tasks/:id`
  - `GET /agents`
  - `PATCH /agents/:id`
- **Database Entities Owned:**
  - `Conversation`, `Message`, `AgentTask`, `AgentPersona`
- **Dependencies:**
  - Depends on Person 1's models. Can test SSE generator and tools independently.
- **Can Start Immediately:** **YES**

---

### Person 5 — Content Calendar, Meta Integration & Insights Engine
**Area:** Social Content Calendar, Meta Ads OAuth/Graph API, and Proactive Insights.

- **Responsibilities:**
  - Build Content Calendar CRUD endpoints scoped by `YYYY-MM`.
  - Implement AI calendar generation algorithm distributing posts across requested channels.
  - Implement Meta connection state-machine (`GET /integrations/meta`, connect, reauth, disconnect).
  - Build Meta OAuth 2.0 authorization URL generator and callback token exchange handler.
  - Build Meta campaign publisher (creating Campaign, AdSet, and Ad objects on Meta Graph API).
  - Implement `GET /insights` endpoint generating diagnostic observations with attached `ActionRef` buttons.
- **Files / Modules Owned:**
  - `SAHM-AI-backend/app/api/content.py`
  - `SAHM-AI-backend/app/api/integrations.py`
  - `SAHM-AI-backend/app/api/insights.py`
  - `SAHM-AI-backend/app/core/integrations/meta.py`
  - `SAHM-AI-backend/app/core/calendar/`
  - `SAHM-AI-backend/app/schemas/content.py`, `integrations.py`, `insights.py`
- **APIs Owned:**
  - `GET /content`
  - `POST /content/generate`
  - `PATCH /content/:id`
  - `DELETE /content/:id`
  - `GET /integrations/meta`
  - `POST /integrations/meta/connect`
  - `POST /integrations/meta/reauth`
  - `POST /integrations/meta/disconnect`
  - `GET /insights`
- **Database Entities Owned:**
  - `ContentItem`, `MetaConnection`, `Insight`
- **Dependencies:**
  - Depends on Person 1's database models and Person 2's campaign entities.
- **Can Start Immediately:** **YES** (Calendar engine and Meta OAuth mock state machine start immediately).

---

# PHASE 9 — SHARED CONTRACTS

Before writing functional code, all 5 developers must commit to the following specifications:

### 1. Unified Response Envelope & Error Format
All non-2xx responses must return JSON conforming to `ApiErrorSchema`:
```json
{
  "code": "RESOURCE_NOT_FOUND",
  "message": "Campaign cmp_104 was not found.",
  "details": {
    "campaignId": "cmp_104"
  }
}
```
Standard Error Codes:
- `BAD_REQUEST`: Invalid input format or query parameters.
- `NOT_FOUND`: Target entity ID does not exist.
- `VALIDATION_ERROR`: Request body failed schema constraints.
- `SCHEMA_MISMATCH`: Emitted if serialization drifts from Zod contracts.
- `INTEGRATION_ERROR`: Third-party provider (Meta, OpenAI) failed.
- `ACTION_NOT_PERMITTED`: Attempted live campaign mutation without required parameters.

### 2. Schema and Field Naming Conventions
- **Wire Format:** CamelCase field names for all JSON responses matching TypeScript interfaces (`dailyBudget`, `createdAt`, `assetUrl`, `healthScore`).
- **Database Format:** Snake_case column names in PostgreSQL (`daily_budget`, `created_at`, `asset_url`).
- **Pydantic Serialization:** All Pydantic models must use `populate_by_name = True` and configure camelCase aliases using `pydantic.alias_generators.to_camel`.

### 3. Date and Time Format
- All dates must be serialized as ISO-8601 strings with timezone offset:  
  `YYYY-MM-DDTHH:mm:ss.sssZ` (e.g. `2026-10-01T11:30:00.000Z`).
- Calendar and performance dates use `YYYY-MM-DD`.

### 4. SSE Streaming Protocol
The endpoint `/conversations/messages/stream` must emit valid Server-Sent Events where every `data:` payload is a complete, valid JSON string of `SendMessageOutput`:
```
event: message
data: {"conversation":{...},"userMessage":{...},"assistantMessage":{"id":"msg_1","role":"assistant","blocks":[{"type":"text","text":"Analyzing campaign...","streaming":true}],"context":[],"createdAt":"2026-10-01T11:30:00.000Z"}}

event: message
data: {"conversation":{...},"userMessage":{...},"assistantMessage":{"id":"msg_1","role":"assistant","blocks":[{"type":"text","text":"Analysis complete.","streaming":false}],"context":[],"createdAt":"2026-10-01T11:30:05.000Z"}}
```

### 5. Human-in-the-Loop Diff Format
Recommendations and campaign mutations must emit changes in exact `ChangeDiff` structures:
```json
{
  "field": "dailyBudget",
  "label": "Daily Budget",
  "before": 500,
  "after": 750
}
```

---

# PHASE 10 — GIT & BRANCH STRATEGY

```
                          main (production releases)
                            │
                            ▼
                         develop (integration branch)
                            │
    ┌──────────────┬────────┼──────────────┬──────────────┐
    │              │        │              │              │
    ▼              ▼        ▼              ▼              ▼
feature/       feature/  feature/       feature/       feature/
dev1-found     dev2-camp dev3-creat     dev4-chat      dev5-cal-meta
```

### Branch Assignments:
- `feature/dev1-foundation`: Person 1 (Core models, config, business/settings).
- `feature/dev2-campaigns`: Person 2 (Campaigns, wizard, strategy, recommendations).
- `feature/dev3-creatives`: Person 3 (Creatives, variations, reference analyzer, text generation).
- `feature/dev4-agent-chat`: Person 4 (SSE chat streaming, async tasks, tool runner).
- `feature/dev5-calendar-meta`: Person 5 (Content calendar, Meta integration, insights).

### Merge Conflict Prevention & Rules:
1. **Router Separation:** Every developer owns their respective router file in `app/api/`. No developer modifies another person's router.
2. **Schema Separation:** Schemas are segregated into `app/schemas/<domain>.py`.
3. **Database Migration Coordination:** Developer 1 creates the baseline migration `0001_initial_schema.py`. Developers 2–5 only add column extensions after Developer 1 merges to `develop`.
4. **Shared File Protocol (`app/main.py`):** Developer 1 registers routers in `main.py`. Other developers register stubs during Stage 1.

---

# PHASE 11 — IMPLEMENTATION ORDER

### Stage 0 — Team Agreement (Day 1 Morning)
- All 5 developers review and sign off on `BACKEND_PROJECT_PLAN.md`.
- Agree on error response envelopes, date serialization, and SSE snapshot format.

### Stage 1 — Foundation (Day 1 Afternoon)
- **Dev 1:** Resolves `greenlet` dependency; establishes SQLAlchemy async engine, `Base` model, and Alembic migrations; deploys `/business`, `/brand-profile`, and `/settings`.
- **Dev 2–5:** Implement domain Pydantic schemas in `app/schemas/` matching Zod contracts and write unit test fixtures.

### Stage 2 — Parallel Development (Day 2 – Day 3)
- **Dev 1:** Finalizes Audiences API, Session resolver, and SSRF security guards.
- **Dev 2:** Implements Campaigns CRUD, strategy generator, and recommendation diff logic.
- **Dev 3:** Implements Creative generator, variations, and reference ad analyzer.
- **Dev 4:** Implements SSE streaming chat endpoint and async task runner.
- **Dev 5:** Implements Content calendar CRUD and Meta OAuth state machine.

### Stage 3 — Integration (Day 4)
- Merge feature branches into `develop`.
- Run comprehensive backend test suite (`uv run --extra dev pytest`).
- Verify database relationships and foreign key cascades across modules.

### Stage 4 — Frontend Live Integration (Day 5 Morning)
- Update `SAHM-AI-frontend/.env.local`:
  ```bash
  NEXT_PUBLIC_USE_MOCKS=false
  NEXT_PUBLIC_API_BASE=http://localhost:8000/api/v1
  ```
- Start frontend with real backend. Verify each page loads live data without schema mismatch errors.

### Stage 5 — Verification & End-to-End Testing (Day 5 Afternoon)
- Complete the 5 key end-to-end workflows:
  1. Create campaign via wizard -> verify DB record.
  2. Launch campaign -> verify Meta mock publish.
  3. Upload reference ad -> verify analysis and inspired brief.
  4. Stream assistant chat -> verify live SSE chunks.
  5. Apply recommendation -> verify before/after diff application.

---

# PHASE 12 — RISKS AND UNKNOWNS

| Risk / Ambiguity | Potential Impact | Severity | Mitigation Strategy |
|---|---|---|---|
| **Multi-Tenancy & Auth Missing in UI** | Unauthenticated public endpoints if deployed directly to internet. | High | Implement secure session cookie defaulting to primary business tenant; add login gateway before production deployment. |
| **Meta Graph API Rate Limits & Token Expiry** | Campaign publishing and metric sync failures during production. | High | Store encrypted refresh tokens; implement token refresh interceptors and exponential backoff retry. |
| **SSE Streaming Dropouts in Proxies** | Chat responses cut off prematurely behind Nginx/Cloudflare reverse proxies. | Medium | Set `X-Accel-Buffering: no` headers and configure 15-second heartbeat SSE comment pings (`: ping\n\n`). |
| **Zod Schema Mismatch Failures** | Frontend `HttpApi` throws `SCHEMA_MISMATCH` and shows error boundary if backend response drifts. | High | Enforce automated contract testing using Pydantic schemas generated directly from the frontend Zod specifications. |
| **Large Media Uploads in Reference Analysis** | Memory spikes when processing base64-encoded video reference ads. | Medium | Enforce a 50MB file size limit and validate MIME types via magic number headers before passing to multimodal models. |

---

# PHASE 13 — FINAL TEAM CHECKLIST

## Team Starting Checklist

### Before Coding
- [ ] Backend architecture and directory structure agreed by all 5 developers.
- [ ] Database schema models and naming conventions approved.
- [ ] API error format (`ApiErrorSchema`) confirmed across all domains.
- [ ] Git branch strategy and pull request guidelines adopted.

### Developer 1 (Foundation & Core)
- [ ] Add `greenlet` to `SAHM-AI-backend/pyproject.toml` and verify `uv sync`.
- [ ] Create SQLAlchemy async engine and baseline Alembic migration.
- [ ] Implement `GET /business` and `GET /brand-profile`.
- [ ] Implement `GET /audiences/:id`.
- [ ] Implement `GET /settings` and `PATCH /settings`.

### Developer 2 (Campaigns & Recommendations)
- [ ] Implement `GET /campaigns` (with status and search filters).
- [ ] Implement `GET /campaigns/:id` and `POST /campaigns` (draft/launch).
- [ ] Implement `PATCH /campaigns/:id` for live status and budget updates.
- [ ] Implement `POST /campaigns/strategy` generator.
- [ ] Implement `GET /recommendations/:id`, `/apply`, and `/dismiss`.

### Developer 3 (Creatives & Generation)
- [ ] Implement `GET /creatives` and `GET /creatives/:id`.
- [ ] Implement `POST /creatives/generate` (image and copy pipelines).
- [ ] Implement `POST /creatives/:creativeId/variations`.
- [ ] Implement `POST /creatives/analyze-reference` (vision reverse-engineering).
- [ ] Implement `POST /generate/text` for copywriting tools.

### Developer 4 (Multi-Agent & Chat Streaming)
- [ ] Implement SSE streaming route `POST /conversations/messages/stream`.
- [ ] Implement conversation history `GET /conversations` and `GET /conversations/:id`.
- [ ] Implement background task engine (`POST /tasks`, `GET /tasks/:id`).
- [ ] Wire the 9 research tool definitions into the orchestrator.
- [ ] Implement interactive question submission handling.

### Developer 5 (Content Calendar & Meta Integration)
- [ ] Implement `GET /content` (scoped by month and campaign).
- [ ] Implement `POST /content/generate`, `PATCH /content/:id`, `DELETE /content/:id`.
- [ ] Implement Meta connection state-machine (`GET /integrations/meta`, connect, reauth, disconnect).
- [ ] Implement `GET /insights` with linked `ActionRef` buttons.
- [ ] Wire live Meta ad publishing client.

### Integration & Verification
- [ ] All 5 feature branches merged into `develop` without conflicts.
- [ ] Full backend test suite passing with 100% contract compliance.
- [ ] Switch `NEXT_PUBLIC_USE_MOCKS=false` and verify frontend dashboard runs cleanly against live backend.
- [ ] End-to-end execution of Campaign Wizard, Chat SSE, and Recommendation Diff workflows verified.
