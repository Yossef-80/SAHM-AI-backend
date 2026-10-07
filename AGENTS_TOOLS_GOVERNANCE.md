# Agent Tools Architecture & Governance Guide

> **Document:** `AGENTS_TOOLS_GOVERNANCE.md`  
> **Status:** APPROVED FOR IMPLEMENTATION  
> **Target Framework:** Python 3.11+ · Pydantic v2 · SQLAlchemy 2.0 Async · FastMCP Ready  
> **Purpose:** Single source of truth for all agent tools, governance laws for decoupled development, and a 5-developer parallel work distribution.

---

## 1. Executive Summary

In a multi-agent system, **agents should not possess hardcoded logic or duplicate tools**. Instead:
1. **Tools are pure, decoupled capabilities** (typed input $\rightarrow$ execution handler $\rightarrow$ typed output).
2. **Agents are distinct cognitive personas** that receive a filtered subset of tools through an allowlist.
3. **Common tools** (like customer brand context, web search, and evidence logging) are shared across multiple agents without code duplication.

This document collects all **21 unique tools** across the 5 agents, establishes the **8 Architectural Governance Laws** to keep them modular, and splits implementation into **5 parallel developer work packages**.

---

## 2. Master Tool Catalog & Cross-Agent Matrix

Every tool across the 5 personas is cataloged below, showing its side-effect type and which agents have access:

```
Tool Side-Effect Classes:
  [READ_ONLY]      -> No state modified. Safe to run repeatedly.
  [INTERNAL_WRITE] -> Updates database/memory. Agent-executable under autonomy policy.
  [EXTERNAL_WRITE] -> Mutates real ad spend / Meta Ads. PROTECTED: Never agent-callable;
                      only executable after human approval.
```

### The Matrix

| # | Tool Identifier | Module Location | Side Effect | Consumed By Agents | Primary Purpose |
|---|---|---|---|---|---|
| **1** | `get_brand_context` | `app/core/tools/memory.py` | `READ_ONLY` | **Nour, Omar, Layla, Karim, Salma** *(All)* | Injects brand voice, audience identity, tone, and forbidden words. |
| **2** | `save_learning` | `app/core/tools/memory.py` | `INTERNAL_WRITE` | **Nour, Omar, Karim** | Stores empirical campaign/market findings in vector memory. |
| **3** | `web_search` | `app/core/tools/research.py` | `READ_ONLY` | **Nour, Omar, Layla** | Performs live web searches via Tavily/Serper with mock fallback. |
| **4** | `analyze_website` | `app/core/tools/research.py` | `READ_ONLY` | **Nour** | Scrapes & extracts text from target URLs with SSRF protection. |
| **5** | `get_social_profile` | `app/core/tools/research.py` | `READ_ONLY` | **Nour** | Extracts bio, follower metrics, and recent post themes for a handle. |
| **6** | `get_reviews` | `app/core/tools/research.py` | `READ_ONLY` | **Nour** | Mines customer pain points and sentiments from review sources. |
| **7** | `get_keywords` | `app/core/tools/research.py` | `READ_ONLY` | **Nour** | Fetches search volumes, CPC, and intent for keyword clusters. |
| **8** | `get_competitor_ads` | `app/core/tools/research.py` | `READ_ONLY` | **Nour** | Inspects active competitor ad creatives, copy, and CTAs. |
| **9** | `detect_market_gaps` | `app/core/tools/research.py` | `READ_ONLY` | **Nour** | Synthesizes unmet customer needs from reviews and competitors. |
| **10** | `generate_marketing_strategy` | `app/core/tools/strategy.py` | `READ_ONLY` | **Omar** | Synthesizes positioning, value props, and 90-day phase roadmap. |
| **11** | `generate_campaign_strategy` | `app/core/tools/strategy.py` | `READ_ONLY` | **Omar** | Allocates channel budgets (FB, IG, Messenger) to meet objectives. |
| **12** | `generate_content_calendar` | `app/core/tools/strategy.py` | `READ_ONLY` | **Omar** | Plans multi-channel post schedules matching requested cadence. |
| **13** | `analyze_reference_creative` | `app/core/tools/media.py` | `READ_ONLY` | **Layla** | Deconstructs ad images/videos into hook, visual style, and tone. |
| **14** | `generate_image` | `app/core/tools/media.py` | `READ_ONLY` | **Layla** | Generates visual assets via DALL-E 3 / Stability in brand palette. |
| **15** | `generate_ad_copy` | `app/core/tools/media.py` | `READ_ONLY` | **Layla** | Writes headlines, primary texts, and CTAs filtered by brand rules. |
| **16** | `generate_ad_variations` | `app/core/tools/media.py` | `READ_ONLY` | **Layla** | Produces angle/hook variations of an existing creative. |
| **17** | `get_campaign_insights` | `app/core/tools/meta_read.py` | `READ_ONLY` | **Karim, Salma** | Queries Meta delivery metrics (impressions, clicks, spend, ROAS). |
| **18** | `get_breakdowns` | `app/core/tools/meta_read.py` | `READ_ONLY` | **Karim, Salma** | Slices performance by age, gender, region, and placement. |
| **19** | `analyze_performance` | `app/core/tools/meta_read.py` | `READ_ONLY` | **Karim** | Detects creative fatigue and metric anomalies over 14-day windows. |
| **20** | `create_recommendation` | `app/core/tools/protected.py` | `INTERNAL_WRITE` | **Salma** | Creates a pending recommendation with a concrete `ChangeDiff`. |
| **21** | `apply_change` / `publish_campaign` | `app/core/tools/protected.py` | `EXTERNAL_WRITE` | *None (Approval Service)* | Mutates live Meta ad campaigns. **Protected from direct agent invocation.** |

---

## 3. The 8 Governance Laws of Decoupled Modular Tools

To ensure our team builds tools that can be combined seamlessly without merge conflicts or unexpected runtime crashes, every developer must obey these 8 architectural laws.

```
                           THE 8 TOOL GOVERNANCE LAWS
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ 1. Strict Typed Input/Output         │ 5. SSRF & Inbound Security           │
│ 2. ToolContext Dependency Injection  │ 6. Secret & Data Masking             │
│ 3. Never Crash (Factual Honesty)     │ 7. Token Budget & Bounded Output     │
│ 4. Total Agent Agnosticism           │ 8. Protected Mutation Barrier        │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

### Law 1: Strict Typed Input/Output Contracts
- **Rule:** Every tool must define a dedicated Pydantic v2 `input_model` and inherit from `ToolOutput` for its `output_model`.
- **Prohibited:** Never accept raw dictionaries (`dict[str, Any]`) or return raw untyped JSON.
- **Why:** Pydantic models automatically generate OpenAI/Anthropic tool schemas and guarantee schema validation before execution.

```python
# GOOD:
class WebSearchInput(BaseModel):
    query: str = Field(min_length=2, description="The search query")
    max_results: int = Field(default=5, ge=1, le=10)

class WebSearchOutput(ToolOutput):
    results: list[SearchResult] = Field(default_factory=list)

# BAD:
async def web_search_handler(params: dict, ctx):  # VIOLATION: untyped dictionary
    return {"results": [...]}
```

---

### Law 2: Context Injection, Never Ambient Globals
- **Rule:** All external capabilities arrive via `ToolContext`. A tool handler must NEVER import a database session, an LLM singleton, or read environment variables directly.
- **Allowed Context Fields:**
  - `ctx.db`: Asynchronous database session (`AsyncSession`).
  - `ctx.llm`: LLM client for sub-generation or embeddings.
  - `ctx.business_id`: Active business tenant.
  - `ctx.workspace_id`: Tenant boundary.
  - `ctx.request_id`: Tracing correlation ID.
- **Why:** Eliminates hidden side effects and allows 100% isolated unit testing with mocks.

```python
# GOOD:
async def handler(input_data: MyInput, ctx: ToolContext) -> MyOutput:
    result = await ctx.db.execute(...)
    return MyOutput(...)

# BAD:
from app.db.session import async_session  # VIOLATION: ambient global session
async def handler(input_data: MyInput, ctx: ToolContext) -> MyOutput:
    async with async_session() as db:
        ...
```

---

### Law 3: Zero-Crash & Graceful Degradation (Factual Honesty)
- **Rule:** A tool handler must NEVER raise an unhandled exception or crash the agent. If an external service fails, network drops, or a website blocks scraping, the tool MUST return:
  `ToolOutput.unavailable(source="target_url", reason="HTTP 403 Forbidden")`.
- **The "Never Hallucinate" Clause:** If data cannot be retrieved, return `status=UNAVAILABLE` and leave data fields empty. Never insert mock numbers or guess metrics.
- **Why:** The agent runtime inspects `unavailable_sources` and truthfully informs the user instead of hallucinating fake statistics.

---

### Law 4: Total Agent Agnosticism
- **Rule:** A tool must have ZERO knowledge of which agent called it. Never inspect or branch on `ctx.persona_id` inside a tool handler.
- **Prohibited:**
  ```python
  if ctx.persona_id == "layla":  # VIOLATION: Tool is coupled to an agent persona!
      ...
  ```
- **Why:** If a tool is agnostic, Nour, Omar, and Layla can all call `get_brand_context` or `web_search` identically. Authorization is enforced by the Tool Registry allowlist, not by the tool itself.

---

### Law 5: SSRF & Inbound Network Security
- **Rule:** Any tool that fetches an external URL (`analyze_website`, `analyze_reference_creative`) MUST validate the target using `resolve_and_check(url)` before sending an HTTP request.
- **SSRF Blocklist:**
  - IPv4 private networks: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`.
  - Loopback & Link-local: `127.0.0.1`, `169.254.169.254` (Cloud Metadata Service).
  - Non-HTTP schemes: `ftp://`, `file://`, `gopher://`.
- **Why:** Prevents malicious user prompts from using our agents as an internal network port scanner.

---

### Law 6: Secret & Data Masking
- **Rule:** Tools that interact with third-party providers (Meta Graph API, OpenAI, Tavily) must NEVER emit raw access tokens, encrypted secrets, or customer passwords in tool outputs or logging summaries.
- **Audit Logging:** Every tool execution is recorded via `log_tool_call(ctx, ...)`. Handlers must provide a safe `output_summary` string (max 200 chars) devoid of PII or secrets.

---

### Law 7: Token Budget & Truncation Governance
- **Rule:** A tool output must never flood the LLM context window. Raw HTML or multi-megabyte API responses must be parsed, stripped of boilerplate, and bounded before returning.
- **Caps:**
  - `web_search`: Max 5 snippets, each $\le 300$ characters.
  - `analyze_website`: Main content markdown $\le 3,000$ characters.
  - `get_brand_context`: Total assembled tokens $\le 900$ tokens.
- **Why:** Prevents a single tool call from exceeding model context limits or burning daily token budgets.

---

### Law 8: The Protected Mutation Barrier (Human-in-the-Loop)
- **Rule:** No tool with `side_effect == ToolSideEffect.EXTERNAL_WRITE` may be registered in the agent `ToolRegistry`.
- **Enforcement:** If an engineer attempts to register an external write tool, `ToolRegistry.register()` will immediately raise a `ValueError`.
- **The Protocol:**
  1. Agents wanting live changes must invoke `create_recommendation`, emitting a `ChangeDiff`.
  2. The frontend shows a before/after confirmation dialog to the user.
  3. Only the backend `approvals/service.py` executes the protected tool after receiving the user's signature.

---

## 4. Division of Tools for 5 Developers

To accelerate delivery, we divide the 21 tools into **5 independent modules**. Each developer owns one complete domain and can write, test, and verify their tools in parallel using mock dependencies.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        5-DEVELOPER TOOL DISTRIBUTION MATRIX                            │
├─────────────┬──────────────────────────────────────────┬───────────────────────────────┤
│ Developer   │ Module & Domain                          │ Tools Owned                   │
├─────────────┼──────────────────────────────────────────┼───────────────────────────────┤
│ Developer 1 │ app/core/tools/memory.py (Identity/Core) │ get_brand_context             │
│             │                                          │ save_learning                 │
│             │                                          │ get_business_profile          │
├─────────────┼──────────────────────────────────────────┼───────────────────────────────┤
│ Developer 2 │ app/core/tools/research.py (Web & Comps) │ web_search, analyze_website   │
│             │                                          │ get_social_profile            │
│             │                                          │ get_reviews, get_keywords     │
│             │                                          │ get_competitor_ads, gaps      │
├─────────────┼──────────────────────────────────────────┼───────────────────────────────┤
│ Developer 3 │ app/core/tools/strategy.py (Strategy)    │ generate_marketing_strategy   │
│             │                                          │ generate_campaign_strategy    │
│             │                                          │ generate_content_calendar     │
├─────────────┼──────────────────────────────────────────┼───────────────────────────────┤
│ Developer 4 │ app/core/tools/media.py (Creatives)      │ analyze_reference_creative    │
│             │                                          │ generate_image                │
│             │                                          │ generate_ad_copy              │
│             │                                          │ generate_ad_variations        │
├─────────────┼──────────────────────────────────────────┼───────────────────────────────┤
│ Developer 5 │ app/core/tools/meta_read.py & protected  │ get_campaign_insights         │
│             │ (Telemetry & Optimization)               │ get_breakdowns                │
│             │                                          │ analyze_performance           │
│             │                                          │ create_recommendation         │
│             │                                          │ apply_change (Protected)      │
└─────────────┴──────────────────────────────────────────┴───────────────────────────────┘
```

---

### Package 1 (Developer 1): Shared Memory & Customer Identity Suite
**File:** `SAHM-AI-backend/app/core/tools/memory.py`  
**Consumers:** Nour, Omar, Layla, Karim, Salma *(Every Agent)*

- **Tools to Build:**
  1. `GetBrandContextTool` (`get_brand_context`):
     - **Inputs:** `query: str | None` (used for semantic search against past winning ads).
     - **Behavior:** Reads `BrandProfile` + active `Audience` + vector search on `BrandMemoryItem`. Assembles compact context strictly under 900 tokens.
  2. `SaveLearningTool` (`save_learning`):
     - **Inputs:** `category: str`, `statement: str`, `evidence: str`, `confidence: float`.
     - **Behavior:** Persists learning into database and computes vector embedding.
- **Why this comes first:** Layla, Omar, and Nour all depend on `get_brand_context`. Developer 1 provides the foundation for customer identity injection.

---

### Package 2 (Developer 2): Web Intelligence & Competitor Scraping Suite
**File:** `SAHM-AI-backend/app/core/tools/research.py`  
**Consumers:** Nour, Omar, Layla

- **Tools to Build:**
  1. `WebSearchTool` (`web_search`): Tavily/Serper wrapper with mock fallback.
  2. `AnalyzeWebsiteTool` (`analyze_website`): BeautifulSoup HTML scraper with `resolve_and_check` SSRF validation.
  3. `GetSocialProfileTool` (`get_social_profile`): Instagram/TikTok public data extractor.
  4. `GetReviewsTool` (`get_reviews`): Aggregates sentiment and pain points.
  5. `GetKeywordsTool` (`get_keywords`): Keyword volume, difficulty, and CPC data.
  6. `GetCompetitorAdsTool` (`get_competitor_ads`): Ad hook and creative scanner.
  7. `DetectMarketGapsTool` (`detect_market_gaps`): Compares competitor weaknesses to audience desires.
- **Key Requirement:** Must implement Law 5 (SSRF check) and Law 3 (Graceful failure).

---

### Package 3 (Developer 3): Strategy & Calendar Synthesis Suite
**File:** `SAHM-AI-backend/app/core/tools/strategy.py`  
**Consumers:** Omar, Campaign Wizard

- **Tools to Build:**
  1. `GenerateMarketingStrategyTool` (`generate_marketing_strategy`):
     - Produces positioning, value propositions, and 90-day phase roadmap.
  2. `GenerateCampaignStrategyTool` (`generate_campaign_strategy`):
     - Calculates budget distribution across Facebook, Instagram, Messenger, and Audience Network. Enforces mathematical constraint: $\sum \text{channel\_shares} = 1.0$.
  3. `GenerateContentCalendarTool` (`generate_content_calendar`):
     - Produces weekly scheduled items matching `postsPerWeek` (2–7) and target channels.
- **Key Requirement:** Pure reasoning tools; must utilize `ctx.llm` and adhere to token caps.

---

### Package 4 (Developer 4): Creative, Copywriting & Multimodal Media Suite
**File:** `SAHM-AI-backend/app/core/tools/media.py`  
**Consumers:** Layla, Creatives Page

- **Tools to Build:**
  1. `AnalyzeReferenceCreativeTool` (`analyze_reference_creative`):
     - Accepts Base64 data URL or remote URL. Uses vision LLM to deconstruct hook, visual style, structure, tone, and generate `inspiredBrief`.
  2. `GenerateImageTool` (`generate_image`):
     - Injects brand color palette into prompt and dispatches to DALL-E / Stability / Mock provider.
  3. `GenerateAdCopyTool` (`generate_ad_copy`):
     - Writes headlines, body copy, and CTAs in English or Arabic.
     - **Identity Governance:** Verifies zero occurrences of words in `brand.do_not_use`.
  4. `GenerateAdVariationsTool` (`generate_ad_variations`):
     - Shifts angle/hook while keeping the core product value intact.

---

### Package 5 (Developer 5): Telemetry, Analytics & Protected Mutation Suite
**Files:** `SAHM-AI-backend/app/core/tools/meta_read.py` & `app/core/tools/protected.py`  
**Consumers:** Karim, Salma, Approvals Service

- **Tools to Build:**
  1. `GetCampaignInsightsTool` (`get_campaign_insights`):
     - Fetches delivery metrics (impressions, clicks, CTR, spend, ROAS).
  2. `GetBreakdownsTool` (`get_breakdowns`):
     - Demographic and regional slice metrics.
  3. `AnalyzePerformanceTool` (`analyze_performance`):
     - Rolling 14-day anomaly detector; flags creative fatigue when frequency $> 3.5$ and CTR drops $> 30\%$.
  4. `CreateRecommendationTool` (`create_recommendation`):
     - Packages a proposal with a structured `ChangeDiff` (`field`, `label`, `before`, `after`).
  5. `ApplyChangeTool` & `PublishCampaignTool` (`protected.py`):
     - External write tools called ONLY by `approvals/service.py` after human confirmation.

---

## 5. Tool Implementation Pattern (Code Template)

Every developer must follow this exact standard when building their tools:

```python
from pydantic import BaseModel, Field
from app.core.tools.base import Tool, ToolContext, ToolOutput, log_tool_call
from app.schemas.common import ToolSideEffect, ToolStatus

# 1. Input Model
class AnalyzeWebsiteInput(BaseModel):
    url: str = Field(description="Public HTTP/HTTPS URL of the competitor or landing page")

# 2. Output Model (inherits ToolOutput)
class AnalyzeWebsiteOutput(ToolOutput):
    url: str = ""
    title: str = ""
    headings: list[str] = Field(default_factory=list)
    body_summary: str = ""

# 3. Async Handler
async def analyze_website_handler(params: AnalyzeWebsiteInput, ctx: ToolContext) -> AnalyzeWebsiteOutput:
    start_time = time.monotonic()
    
    # Law 5: SSRF Protection
    try:
        safe_url = resolve_and_check(params.url)
    except BlockedTargetError as exc:
        return AnalyzeWebsiteOutput.unavailable(source=params.url, reason=str(exc))
    
    # Execution with Law 3 Graceful Error Handling
    try:
        # Perform HTTP fetch and extraction
        extracted_data = await scrape(safe_url.url)
    except Exception as exc:
        duration = int((time.monotonic() - start_time) * 1000)
        await log_tool_call(ctx, tool="analyze_website", tool_input=params, 
                            output_summary=f"Failed: {exc}", status="error", duration_ms=duration)
        return AnalyzeWebsiteOutput.unavailable(source=params.url, reason="Failed to fetch website")

    output = AnalyzeWebsiteOutput(
        status=ToolStatus.OK,
        source=params.url,
        title=extracted_data.title,
        headings=extracted_data.headings[:5],
        body_summary=extracted_data.text[:1500]  # Law 7: Bounded output
    )
    
    # Law 6: Audit logging
    duration = int((time.monotonic() - start_time) * 1000)
    await log_tool_call(ctx, tool="analyze_website", tool_input=params,
                        output_summary=f"Extracted {len(output.body_summary)} chars",
                        status="ok", duration_ms=duration)
    return output

# 4. Tool Export
analyze_website_tool = Tool(
    name="analyze_website",
    description="Scrapes a public landing page or competitor site and returns core headlines and copy.",
    input_model=AnalyzeWebsiteInput,
    output_model=AnalyzeWebsiteOutput,
    side_effect=ToolSideEffect.NONE,  # Read-only
    handler=analyze_website_handler
)
```

---

## 6. Verification & Quality Checklist for Every Tool

Before submitting a Pull Request for any tool, the developer must verify:

- [ ] **Law 1:** Inputs and outputs are strictly typed Pydantic models.
- [ ] **Law 2:** No ambient imports or global sessions; uses `ctx.db` and `ctx.llm`.
- [ ] **Law 3:** Tested with broken URLs / failing APIs; returns `status="unavailable"` instead of throwing an unhandled exception.
- [ ] **Law 4:** No mention of `persona_id` or specific agent names inside the handler.
- [ ] **Law 5:** All external URLs checked with `resolve_and_check` (SSRF safe).
- [ ] **Law 6:** No API keys, passwords, or encrypted tokens appear in logs or output summaries.
- [ ] **Law 7:** Outputs are trimmed and bounded (no unbounded text dumps).
- [ ] **Law 8:** If the tool modifies external ad platforms, it lives in `protected.py` with `side_effect=EXTERNAL_WRITE` and is NOT in `ALL_TOOLS`.
- [ ] **Unit Test:** Automated test exists using `respx` or mock DB asserting both success and unavailable paths.
