# AGENTS.md — read this before you touch a single file

> **Who this is for:** AI coding agents and new engineers working in this repo.
> **Goal:** You never scan the tree to find the file that owns a behaviour. Use this map.
> **What it is not:** A tutorial. For the "why", read `SAHM-AI-backend/docs/SYSTEM_ARCHITECTURE_AND_ENGINEERING_PLAYBOOK.md`.

---

## 0. Rules for working in this repo

1. **Do not open a file that is not listed for your task.** Use §3 to find your task, read only the files in its row.
2. **Never read a whole large file.** `brain.py` is 2,873 lines, `service.py` 905, `meta_ads.py` 1,689. Use the grep recipes in §8, read the function, not the module.
3. **Respect §5.** The invariants there are enforced by tests — breaking one fails the suite.
4. **Run the narrow test first, the full suite before you finish** (§7).
5. **If your task is not in §3,** look in §4 (module map), then §2 (concern → folder). If it is still not there, the code does not exist yet — say so instead of inventing a module.
6. **If you change a path, contract, or status in this file, update it in the same change.** A stale map is worse than no map.

---

## 1. Fast facts

| Field | Value |
|---|---|
| **Product** | Sahm — multi-agent marketing OS, Arabic + English, for SMBs in MENA |
| **Stack** | Python ≥3.11, FastAPI, Pydantic v2, SQLAlchemy 2 (async), Alembic, LangGraph/LangChain |
| **Entry point** | `SAHM-AI-backend/app/main.py` → `create_app()` (the module-level `app` is what uvicorn imports) |
| **Frontend** | Next.js 14 (App Router), TypeScript, Tailwind, Zustand, TanStack Query — in `SAHM-AI-frontend/` |
| **One request** | `api/assistant.py` → `core/orchestrator/service.py` → `ai/agents/orchestrator/brain.py` → `ai/agents/<agent>/graph.py` → `core/tools/` + `ai/tools/` → `core/orchestrator/blocks.py` |
| **DB** | 26 tables in `app/db/models.py`; Postgres in prod, SQLite locally |
| **Run backend** | `uv run uvicorn app.main:app --reload` (from `SAHM-AI-backend/`) |
| **Run offline** | `set AI_LLM_PROVIDER=mock` — no keys, no network |
| **Run frontend** | `npm run dev` (from `SAHM-AI-frontend/`) |
| **Test** | `PYTHONPATH=. python3 -m pytest tests -q` (from `SAHM-AI-backend/`) |

### The five largest files — do NOT read end-to-end

| File | Lines | What is in it |
|---|---|---|
| `app/ai/agents/orchestrator/brain.py` | 2,873 | routing, system-message assembly, every specialist branch, creative-prompt state machine |
| `app/ai/marketing/meta_ads.py` | 1,689 | Meta Ads plan validation → paused hierarchy creation, ad-level insights |
| `app/ai/agents/nour/meta_ads.py` | 1,213 | competitor-ad library + vision workflow |
| `app/core/orchestrator/service.py` | 905 | conversation service, SSE, task queue |
| `app/core/agents/runtime.py` | 1,028 | Layer-5 generic agent loop |

---

## 2. Where things live, by concern

```
SAHM-AI-backend/
  app/api/            HTTP only. Validate → call a service → serialise. No business logic.
  app/core/           Business logic. No FastAPI imports except in api/.
    orchestrator/     conversation service, blocks.py, SSE, task queue
    agents/           personas, task specs, prompt templates, generic runtime
    memory/           brand memory store + embeddings
    tools/            governed workspace tools + registry + protected writes
    approvals/        the only path allowed to perform an external write
    brand/            brand identity synthesis and sync
    llm/              Layer-5 client protocol (base, router, per-provider clients)
  app/ai/             The LangGraph side. May import app.core freely.
    agents/           orchestrator brain + 5 specialist graphs (omar, layla, karim, nour, salma)
    tools/            LangChain tools bound to those graphs
    providers/        LLM / search / db adapters, selected by enum, imported lazily
    memory/           session compressor + brand memory fallback
    marketing/        Meta Ads logic, creative prompt, reference vision
    loggers/          per-agent / per-provider file logs, wired through callbacks
  app/db/             models + session + Alembic migrations
  app/schemas/        every request, response and UI block contract
  tests/              contract tests, one file per behaviour area

SAHM-AI-frontend/
  src/app/            Next.js App Router pages (one folder per page)
  src/components/     UI components (organised by domain, same names as app/)
  src/lib/            API client, server actions, Supabase client, utils
  src/hooks/          use-chat.ts, use-api.ts, use-action.ts
  src/store/          Zustand stores (assistant, auth, ui)
  src/types/          Shared TS types (index.ts)
  src/i18n/           next-intl config + locale messages (en.json, ar.json)
  messages/           ar.json, en.json — all UI strings live here
```

**Import direction:** `app.ai` imports `app.core` freely — that is the normal direction.
`app.core` may import leaf `app.ai` modules (`app.ai.security`, `app.ai.settings`, `app.ai.providers.*`).
What is **not** accepted: eagerly importing agent graphs from core. The orchestrator is imported
lazily inside `handle_message` (`core/orchestrator/service.py`, grep `orchestrate`).

---

## 3. Task router — "I want to change X"

Read files in the order listed. *Verify* is the narrowest test that will catch a mistake.

| # | Task | Read (in order) | Verify |
|---|---|---|---|
| 1 | Change what happens during a chat turn (context, persistence, SSE) | `api/assistant.py` → `core/orchestrator/service.py` (grep `handle_message`) | `tests/test_contract.py`, `tests/test_conversation_context_persistence.py` |
| 2 | Change which agent answers, or add a routing branch | `ai/agents/orchestrator/brain.py` (grep `ROUTING_PROMPT`, `def orchestrate`, `RoutingDecision`) | `tests/run_routing_eval.py`, `tests/test_agent_graphs.py` |
| 3 | Change one agent's behaviour / prompt | `ai/agents/<agent>/graph.py` (`<AGENT>_PROMPT`) + `schema.py` | `tests/test_agent_graphs.py` + flow tests |
| 4 | Change how a report is rendered (headings, labels, order) | `ai/agents/orchestrator/presentation.py` | `tests/test_report_presentation.py` |
| 5 | Change the UI block contract | `schemas/blocks.py` → `core/orchestrator/blocks.py` | `tests/test_blocks_flattener_and_evidence.py`, `tests/test_contract.py` |
| 6 | Add or change a workspace tool | `core/tools/base.py` → `schemas/tools.py` → `core/tools/<family>.py` → `core/tools/registry.py` → persona allowlist in `core/agents/personas.py` | `tests/test_tools.py`, `tests/test_registry_policy.py` |
| 7 | Add or change an AI-native tool bound to a graph | `ai/tools/<name>.py` → its binding in `ai/agents/<agent>/graph.py` | `tests/test_ai_wire_contracts.py` |
| 8 | Change memory retrieval / context assembly | `core/memory/store.py` (`BrandMemoryStore.build_context`, `BrandMemoryStore.search`) | `tests/test_memory_performance.py` |
| 9 | Change memory write / session compression | `core/memory/store.py` (`write_item`, `save_learning`) → `ai/memory/compressor.py` | `tests/test_memory_performance.py` |
| 10 | Change the bilingual / reply-language behaviour | `ai/agents/orchestrator/brain.py` (grep `_language_directive`, `RESPONSE_LANGUAGE`) | `tests/test_agent_graphs.py` (grep `language`) |
| 11 | Change the post-turn background work (summary, profile) | `ai/memory/compressor.py` → `core/brand/identity_service.py` | `tests/test_brand_identity.py` |
| 12 | Change the database schema | `db/models.py` → new revision in `db/migrations/versions/` | migration round-trip + full suite |
| 13 | Add or change an endpoint | `api/<area>.py` → `schemas/<area>.py` → `core/<area>/service.py` | matching `tests/test_*_api.py` |
| 14 | Change auth, tenancy or workspace resolution | `api/deps.py` → `core/auth.py` → `core/supabase_auth.py` | `tests/test_auth.py`, `tests/test_workspace_isolation.py` |
| 15 | Change Meta campaign reading | `core/tools/meta_read.py` (`ai/marketing/insights.py` is UNUSED — `meta_ads.py` is live) | `tests/test_meta_campaign_discovery.py`, `tests/test_ga4_tool.py` |
| 16 | Change Meta ad creation | `ai/marketing/meta_ads.py` → `core/approvals/service.py` → `core/tools/protected.py` | `tests/test_meta_ad_plans.py`, `tests/test_approvals.py` |
| 17 | Change the research hub | `api/research.py` → `core/tools/research.py` → `core/research_reports.py` | `tests/test_api_tools.py`, `tests/test_research_report_shaping.py` |
| 18 | Change content calendar generation | `api/content.py` → `ai/tools/strategy/generate_content_calendar.py` | `tests/test_content_calendar*.py` |
| 19 | Change creatives (copy, image, reference style) | `api/creatives.py` → `core/tools/media.py` → `ai/marketing/reference_vision.py` | `tests/test_creative_*`, `tests/test_adimage_bytes.py` |
| 20 | Change the competitor-ad flow | `ai/agents/nour/meta_ads.py` → `ai/tools/ad_previews.py` → `docs/AD_CREATIVE_FLOW.md` | `tests/test_meta_competitor_ads_flow.py` |
| 21 | Change brand identity / profile sync | `core/brand/identity_service.py` | `tests/test_brand_identity.py`, `tests/test_profile_field_families.py` |
| 22 | Change settings, connectors or provider selection | `api/settings.py` → `core/tools/settings.py` → `core/tools/connectors.py` → `core/llm/router.py` | `tests/test_api_tools.py`, `tests/test_llm_layer.py` |
| 23 | Add a setting | `app/config.py` AND `app/ai/settings.py` (two separate objects by design) → `.env.example` | `tests/test_config_surface.py` |
| 24 | Change the generic agent runtime (Layer 5) | `core/agents/runtime.py` (`RunOptions`, `_execute`, `run_agent`) → `core/agents/tasks.py` | `tests/test_runtime.py` |
| 25 | Change rate limits / security headers / CORS | `core/ratelimit.py`, `core/security_headers.py`, `app/config.py` | `tests/test_security.py`, `tests/test_deployment_ops.py` |
| 26 | Change persona defaults or autonomy gates | `core/agents/personas.py` → `core/agents/persona.py` → `core/agents/runtime.py` (`_apply_autonomy_policy`) | `tests/test_registry_policy.py`, `tests/test_runtime.py` |
| 27 | Change frontend chat UI or SSE streaming | `src/hooks/use-chat.ts` → `src/components/assistant/` → `src/store/assistant-store.ts` | Smoke scripts in `SAHM-AI-frontend/scripts/` |
| 28 | Change frontend API client | `src/lib/api/` → `src/lib/actions/` (server actions) | Smoke scripts |
| 29 | Add or change UI translations | `messages/ar.json` and `messages/en.json` (both) → `src/i18n/config.ts` | Visual check on both locales |

---

## 4. Module map

**Status legend:** `LIVE` = in the request path · `SUPPORT` = imported by live code · `LEGACY` = works but not the source of truth · `UNUSED` = not wired anywhere · `GEN` = generated

### HTTP layer — `app/api/`

| File | One line | Status |
|---|---|---|
| `app/main.py` | app factory, middleware order, router registration, lifespan | LIVE |
| `api/assistant.py` | `/assistant/message`, `/message/stream`, conversations, attachments, ad previews | LIVE |
| `api/deps.py` | `WorkspaceDep`, `SessionDep`, `LLMDep`; the tenant boundary | LIVE |
| `api/auth.py`, `api/auth_complete_profile.py`, `core/auth.py`, `core/supabase_auth.py` | login / refresh / logout / me, JWT + opaque sessions | LIVE |
| `api/campaigns.py` | campaign CRUD, wizard strategy, Meta plan create/assist | LIVE |
| `api/content.py` | calendar read / generate / patch / delete | LIVE |
| `api/creatives.py` | generate copy/image, reference analysis, upload, list | LIVE |
| `api/insights.py` | routes `/campaigns/{id}/insights` and `/campaigns/{id}/ads/insights` (no own prefix) | LIVE |
| `api/recommendations.py` | list + approve / reject / activate (the approval gate) | LIVE |
| `api/memory.py` | `/memory/items` CRUD, archive, `/memory/prune` | LIVE |
| `api/research.py` | `/research/run`, `/research/items` | LIVE |
| `api/tasks.py` | background agent tasks + SSE progress | LIVE |
| `api/tools.py` | tool catalog, connectors, `/tools/{name}/invoke` | LIVE |
| `api/settings.py` | connectors + per-workspace LLM settings | LIVE |
| `api/supabase_exchange.py` | token exchange with Supabase | LIVE |
| `api/subscriptions.py` | plan catalog, current quota status, free tier switching | LIVE |
| `api/audiences.py`, `api/personas.py`, `api/business.py`, `api/integrations.py`, `api/media_assets.py`, `api/usage.py`, `api/audit.py` | CRUD / config / governance surfaces | LIVE |

### Conversation & orchestration

| File | One line | Status |
|---|---|---|
| `core/orchestrator/service.py` | `handle_message`, sticky context, language persistence, SSE, task queue | LIVE |
| `core/orchestrator/blocks.py` | builds `Message.blocks` from a result; validates them | LIVE |
| `ai/agents/orchestrator/brain.py` | `orchestrate()`: router, system messages, agent branches, creative-prompt machine | LIVE |
| `ai/agents/orchestrator/handoff.py` | extracts a submitted report from a graph's final state | LIVE |
| `ai/agents/orchestrator/presentation.py` | structured report → Markdown (per-language labels) | LIVE |

### Agents — `app/ai/agents/`

The five specialists. Each follows the same shape: `graph.py`, `schema.py`, `state.py`.

| Agent | Role | Extra files | Status |
|---|---|---|---|
| **Omar** | Strategy & marketing plans | `omar/tools.py`, `omar/service.py` (wizard) | LIVE |
| **Layla** | Creative copy & image generation | `layla/service.py` (direct text deliverables) | LIVE |
| **Karim** | Campaign analytics & performance | — | LIVE |
| **Nour** | Competitor research & ad library | `nour/meta_ads.py` (competitor-ad + vision) | LIVE |
| **Salma** | Campaign optimisation | — | LIVE |
| `nour/gui_graph.py` | Alternate single-agent formatter, imported only by a test | — | UNUSED |

All graphs share the same structure: `call_model` → `should_continue` → `post_tools_routing`. Each has a `submit_*` tool as the terminal output.

### Tools — `app/core/tools/` and `app/ai/tools/`

| File | One line | Status |
|---|---|---|
| `core/tools/base.py` | `Tool`, `ToolContext`, `ToolRegistry` (refuses `external_write`) | LIVE |
| `core/tools/registry.py` | assembles `ALL_TOOLS`; the only way agents get tools | LIVE |
| `core/tools/research.py` | 9 research tools | LIVE |
| `core/tools/memory.py` | `get_brand_context`, `save_learning` | LIVE |
| `core/tools/meta_read.py` | read-only Meta campaign / insight tools | LIVE |
| `core/tools/media.py` | reference analysis + image generation | LIVE |
| `core/tools/records.py` | `create_recommendation` | LIVE |
| `core/tools/protected.py` | external writes — importable only from approvals | LIVE |
| `core/tools/settings.py` | tool settings CRUD | LIVE |
| `core/tools/connectors.py` | connector configuration | LIVE |
| `ai/tools/*.py` | LangChain tools bound to graphs (`web_search`, `analyze_website`, `fetch_ga4_performance`, …) | LIVE |
| `ai/tools/core_bridge.py` | adapters from the typed registry into the graphs | SUPPORT |
| `ai/tools/strategy/` | `generate_content_calendar`, `generate_campaign_strategy`, `generate_marketing_strategy` | LIVE |
| `ai/observability.py` | unified LangSmith, Langfuse, and diagnostic callback router | LIVE |
| `ai/marketing/meta_ads.py` | Meta plan validation + paused creation + ad insights | LIVE |
| `ai/marketing/meta_plan_assist.py`, `meta_ads_mcp.py` | read-only MCP assistance for the wizard | LIVE |
| `ai/marketing/reference_vision.py`, `creative_prompt.py` | vision + prompt composition for creatives | LIVE |
| `ai/marketing/assets.py`, `context.py`, `generation.py`, `insights.py`, `references.py`, `schemas.py` | earlier generation layer, superseded by `ai/tools/strategy/*` and `core/tools/*` | UNUSED |

### Memory — `app/core/memory/` and `app/ai/memory/`

| File | One line | Status |
|---|---|---|
| `core/memory/store.py` | `BrandMemoryStore` — the only writer and reader of `brand_memory_items`; `build_context`, `search`, `write_item` | LIVE |
| `core/memory/embeddings.py` | embedding provider abstraction (pgvector in prod, Python cosine in SQLite) | LIVE |
| `ai/memory/compressor.py` | session summary at the token cap; `IntelligentMemoryCompressor`, `MemoryDebouncer` | LIVE |
| `ai/memory/brand_memory.py` | LangGraph `BaseStore` memory — fallback only when no DB is in scope | LEGACY |

### Providers, runtime, infrastructure

| File | One line | Status |
|---|---|---|
| `ai/providers/llm/*` | one module per provider (openai, anthropic, gemini, groq, ollama, openrouter, mock), selected by enum, imported lazily | LIVE |
| `core/llm/router.py` + `*_client.py` | the Layer-5 client protocol used by `runtime.py` | LIVE |
| `ai/providers/search/*`, `db/*` | search and checkpoint stores | LIVE |
| `core/agents/personas.py`, `core/agents/persona.py` | persona catalog (Omar, Layla, Karim, Nour, Salma) + override merge | LIVE |
| `core/agents/tasks.py` | task specs, prompt templates, `build_prompt` | LIVE |
| `core/agents/runtime.py` | generic tool-calling loop with hard caps (`run_agent`, `run_agent_stream`) | LIVE |
| `core/approvals/service.py` | the only executor of external writes | LIVE |
| `core/subscriptions.py` | free tier plans catalog, workspace subscription status, and budget switcher | LIVE |
| `core/brand/identity_service.py` | brand identity synthesis and sync | LIVE |
| `core/ai_usage.py` | AI usage metering | LIVE |
| `core/security.py`, `core/auth.py`, `core/logging.py`, `core/usage.py`, `core/errors.py` | cross-cutting services | LIVE |
| `db/models.py`, `db/session.py` | schema + engine | LIVE |
| `db/migrations/versions/*` | Alembic history | GEN |
| `ai/loggers/*` | per-agent (omar, layla, karim, nour, salma, orchestrator, diagnostic), provider, and tool file logs + real-time cost estimation | LIVE |

### Frontend — `SAHM-AI-frontend/src/`

| Path | One line | Notes |
|---|---|---|
| `app/<page>/page.tsx` | Next.js App Router page component | One folder per feature |
| `components/<domain>/` | Domain-grouped UI components | Mirrors `app/` folders |
| `lib/api/` | Typed API client (fetch wrappers) | One file per backend resource |
| `lib/actions/` | Next.js server actions | Used for mutations from Server Components |
| `hooks/use-chat.ts` | SSE streaming + message state | Core of the assistant experience |
| `store/assistant-store.ts` | Zustand — conversation + block state | |
| `store/auth-store.ts` | Zustand — session, user, workspace | |
| `messages/ar.json`, `messages/en.json` | All UI strings | Edit both when adding a key |
| `src/middleware.ts` | Next.js middleware — auth guard, locale routing | |

---

## 5. Invariants — break one and the suite fails

1. **Tenancy comes from the dependency, never the body.** Every query filters by `business_id` / `workspace_id` from `WorkspaceDep`. A foreign id returns 404, not 403. → `tests/test_workspace_isolation.py`

2. **No agent may cause an external write.** `ToolRegistry.register()` raises on `side_effect="external_write"`; `get_tools_for()` filters; `_run_one_tool` re-checks. External writes live in `core/tools/protected.py` and run only from `core/approvals/service.py`. → `tests/test_registry_policy.py`, `test_approvals.py`

3. **The reply language is decided in `brain.py` (`_language_directive`), not re-derived.** An explicit UI choice beats everything. → `tests/test_agent_graphs.py` (grep `language`)

4. **Agents return typed Pydantic reports, not prose.** The client only ever sees UI blocks. Adding a block kind means `schemas/blocks.py` + the discriminator. → `tests/test_contract.py`

5. **Memory writes go through `BrandMemoryStore`.** Hand-building a `BrandMemoryItem` loses the dedupe key, TTL, and token cap. → `tests/test_memory_performance.py`

6. **The orchestrator stays a lazy import inside `service.py`.** `app.core` may import leaf `app.ai` modules (security, settings, providers); it must not pull the agent graphs into the import path of `app.main`.

7. **Every graph must prepend its own role prompt only if not already present.** Do not test `messages[0]` — the orchestrator prepends system messages. → `tests/test_agent_graphs.py`

8. **Schema changes are two files:** the model and an Alembic revision. Run it forward and back once. → `tests/test_deployment_ops.py`

9. **Structured logging takes `extra={...}`.** `logger.info("x", {"a": 1})` raises.

10. **A tool that cannot reach its source returns `status="unavailable"` with a reason.** It never returns invented data and never raises into the graph.

11. **`app/config.py` and `app/ai/settings.py` are two separate settings objects by design.** Always update both when adding a setting.

---

## 6. Recipes

### Add an endpoint (backend)
1. `app/schemas/<area>.py` — request + response models (aliases are camelCase).
2. `app/api/<area>.py` — route, `WorkspaceDep`/`SessionDep`, no logic.
3. `app/core/<area>/service.py` — the logic, transaction-bound.
4. Register the router in `app/main.py` if it is a new module.
5. Test: `tests/test_<area>_api.py`.

### Add a workspace tool
1. Input/output models → `app/schemas/tools.py`.
2. `async def _handle(request, ctx: ToolContext)` in the right family module under `core/tools/`.
3. `Tool(name=..., side_effect=READ|DRAFT_WRITE, handler=_handle)`.
4. Append to `ALL_TOOLS` in `app/core/tools/registry.py`.
5. Add the name to the personas that may use it in `app/core/agents/personas.py`.
6. Test success and `status="unavailable"` paths with a fake `ToolContext`.

### Add an AI-native tool to one agent
Follow `docs/TEAM_TOOLS_AND_AGENTS_GUIDE.md`. Bind it in that agent's `graph.py`. Do **not** touch the registry — those tools are not workspace-governed.

### Change a prompt
- **Chat agents:** `<AGENT>_PROMPT` in `app/ai/agents/<agent>/graph.py`.
- **Layer-5 tasks:** `app/core/agents/prompts/<task>.md`.
- Never hardcode a language — the `[RESPONSE_LANGUAGE]` block owns it.

### Add or change a setting
`app/config.py` AND `app/ai/settings.py` (both) → `.env.example` with a comment → `tests/test_config_surface.py`.

### Add a migration
```
cd SAHM-AI-backend
alembic revision --autogenerate -m "..."
# review it (SQLite and Postgres differ)
alembic upgrade head
alembic downgrade -1
```

### Add a frontend page
1. Create `src/app/<page>/page.tsx`.
2. Add API calls to `src/lib/api/<resource>.ts`.
3. Add UI strings to **both** `messages/ar.json` and `messages/en.json`.
4. Add navigation link to the sidebar component.

---

## 7. Verification

```bash
# Run from SAHM-AI-backend/

PYTHONPATH=. python3 -m pytest tests -q                          # everything
PYTHONPATH=. python3 -m pytest tests -q -x                       # stop at first failure
PYTHONPATH=. python3 -m pytest tests/test_memory_performance.py -v
PYTHONPATH=. python3 -m pytest tests -q -k "language or memory"
```

> **Environment note:** tests that need optional provider SDKs (langchain-ollama, -openai, -groq, -google-genai, fastmcp) fail (not skip) in a bare environment. That is expected and unrelated to your change.

### Test file → subsystem

| Tests | Subsystem |
|---|---|
| `test_contract.py`, `test_blocks_flattener_and_evidence.py`, `test_error_response_surface.py` | API + block contract |
| `test_workspace_isolation.py`, `test_auth.py`, `test_security.py` | tenancy, auth, SSRF, headers |
| `test_agent_graphs.py`, `test_omar_layla_handoff*.py`, `test_karim_campaign_context.py` | graphs and handoffs |
| `test_creative_*.py`, `test_ad_creative_flow_contract.py`, `test_layla_*.py` | creative flows |
| `test_memory_performance.py` | memory store and ranking |
| `test_runtime.py`, `test_registry_policy.py`, `test_tools.py` | runtime, tool policy, tool behaviour |
| `test_meta_*.py`, `test_ga4_tool.py`, `test_targeting_search.py` | Meta and analytics |
| `test_content_calendar*.py`, `test_strategy_tools.py`, `test_campaign_*.py` | calendar, strategy, campaigns |
| `test_task_routes.py`, `test_usage_rate_limit.py`, `test_deployment_ops.py` | tasks, governance, ops |
| `test_brand_identity.py`, `test_profile_field_families.py` | brand identity |
| `test_ai_wire_contracts.py`, `test_ai_layer_offline.py`, `test_ai_provider_adapters_offline.py` | AI layer |
| `test_analyze_website.py`, `test_competitor_ads_tool.py`, `test_reference_vision.py` | specific tools |
| `conftest.py` | isolated SQLite per test, seeded workspace, API client, all providers mocked |

---

## 8. Finding a symbol without reading a file

```bash
# Run from SAHM-AI-backend/

# Orchestrator brain
grep -n "def orchestrate\|RoutingDecision\|ROUTING_PROMPT" app/ai/agents/orchestrator/brain.py
grep -n "_language_directive\|creative_prompt_stage\|RESPONSE_LANGUAGE" app/ai/agents/orchestrator/brain.py

# Conversation service
grep -n "def handle_message\|def _persist_message\|def get_or_create_conversation" app/core/orchestrator/service.py

# Memory store
grep -n "def build_context\|def write_item\|def save_learning\|def search" app/core/memory/store.py

# Agent runtime
grep -n "async def run_agent\|class RunOptions\|_apply_autonomy_policy" app/core/agents/runtime.py

# Tool registry
grep -n "ALL_TOOLS\|def register_all\|def get_tools_for" app/core/tools/registry.py

# Broadcast grep
grep -rn "_creative_prompt_blocks\|REPORT_LANGUAGE" app --include="*.py"
grep -rn "TODO\|FIXME\|HACK" app --include="*.py"        # known rough edges
```

> `brain.py` is organised top-to-bottom: constants → prompt text → small helpers → `orchestrate()` → branches. Grep a distinctive string first, read ±80 lines.

---

## 9. Documentation

| Need | File |
|---|---|
| Product overview + features | `SAHM-AI-backend/docs/PROJECT_FEATURES_AND_CAPABILITIES.md` |
| System architecture & engineering playbook | `SAHM-AI-backend/docs/SYSTEM_ARCHITECTURE_AND_ENGINEERING_PLAYBOOK.md` |
| Running the backend | `SAHM-AI-backend/docs/RUNNING_BACKEND.md` |
| Ad creative flow | `SAHM-AI-backend/docs/AD_CREATIVE_FLOW.md` |
| Brand memory & orchestrator deep-dive | `TEAM_BRAND_MEMORY_AND_ORCHESTRATOR_ARCHITECTURE.md` (root) |
| Tools and agents guide | `TEAM_TOOLS_AND_AGENTS_GUIDE.md` (root) |
| Project scope and assessment | `TEAM_PROJECT_SCOPE_AND_ASSESSMENT.md` (root) |
| Social integrations plan | `SOCIAL_INTEGRATIONS_PLAN.md` (root) |
| Frontend API contract | `SAHM-AI-frontend/docs/API_CONTRACT.md` |
| Frontend actions reference | `SAHM-AI-frontend/docs/ACTIONS.md` |
| Frontend README | `SAHM-AI-frontend/README.md` |

---

## 10. The five specialists at a glance

| Agent | Persona ID | Submit tool | Primary task |
|---|---|---|---|
| **Omar** | `omar` | `submit_strategy_report` | Marketing strategy, campaign planning |
| **Layla** | `layla` | `submit_creative_plan` | Copy, images, creative assets |
| **Karim** | `karim` | `submit_performance_report` | Analytics, campaign performance |
| **Nour** | `nour` | (competitor report) | Competitor ads research, ad library |
| **Salma** | `salma` | `submit_optimization_plan` | Campaign optimization; never generates creatives herself — returns `needs_creative_variations=True` and orchestrator routes to Layla |
