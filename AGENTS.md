# AGENTS.md — read this before you read the code

> **Who this is for:** an AI coding agent (or a new engineer) working in this
> repository. The goal is that you **never have to scan the tree** to find the
> file that owns a behaviour.
>
> **What it is:** a map. Where things live, what is wired to what, what is
> already working, what is dead, and how to verify a change without running the
> whole product.
>
> **What it is not:** a tutorial. For the "why", read [`README.md`](README.md)
> (10 minutes) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 0. Rules for working in this repo

1. **Do not open a file that is not listed for your task.** Use §3 to find your
   task, read only the files in its row, then stop.
2. **Never read a whole large file.** `brain.py` is 3,100 lines,
   `service.py` 1,000, `store.py` 950. Use the grep recipes in §8, read the
   function, not the module.
3. **Respect §5.** The invariants there are enforced by tests, and breaking one
   fails the suite rather than producing a subtle bug you can hide.
4. **Run the narrow test first, the full suite before you finish** (§7).
5. **If your task is not in §3**, look in §4 (module map → one line per file),
   then §2 (concern → folder). If it is still not there, the code does not
   exist yet — say so instead of inventing a module.
6. **If you change a path, a contract or a status in this file, update it in the
   same change.** A stale map is worse than no map.

---

## 1. Fast facts

| | |
| :--- | :--- |
| **Product** | Sahm — multi-agent marketing OS, Arabic + English, for SMBs in MENA |
| **Stack** | Python ≥3.11, FastAPI, Pydantic v2, SQLAlchemy 2 (async), Alembic, LangGraph/LangChain |
| **Entry point** | `app/main.py::create_app()` (the module-level `app` is what uvicorn imports) |
| **One request** | `api/assistant.py` → `core/orchestrator/service.py` → `ai/agents/orchestrator/brain.py` → `ai/agents/<agent>/graph.py` → `core/tools/` + `ai/tools/` → `core/orchestrator/blocks.py` |
| **DB** | 23 tables in `app/db/models.py`; Postgres in prod, SQLite locally |
| **Run** | `uv run uvicorn app.main:app --reload` |
| **Run offline** | set `AI_LLM_PROVIDER=mock` — no keys, no network |
| **Test** | `PYTHONPATH=. python3 -m pytest tests -q` → 880 tests, all passing |
| **Counts** | 5 specialist agents · 17 workspace tools · 9 research tools · 13 memory kinds · 19 routers |

### The five largest files (do not read end-to-end)

| File | Lines | What is in it |
| :--- | ---: | :--- |
| `app/ai/agents/orchestrator/brain.py` | 3,119 | routing, system-message assembly, every specialist branch, creative-prompt state machine |
| `app/ai/marketing/meta_ads.py` | 1,860 | Meta Ads plan validation → paused hierarchy creation, ad-level insights |
| `app/ai/agents/nour/meta_ads.py` | 1,345 | competitor-ad library + vision workflow |
| `app/core/agents/runtime.py` | 1,145 | Layer-5 generic agent loop |
| `app/core/orchestrator/service.py` | 1,032 | conversation service, SSE, task queue |

---

## 2. Where things live, by concern

```
app/api/            HTTP only. Validate → call a service → serialise. No business logic.
app/core/           Business logic. No FastAPI imports except in api/.
  orchestrator/     conversation service, blocks, SSE, task queue
  agents/           personas, task specs, prompt templates, generic runtime
  memory/           brand memory (kinds, tokens, lexical, ranking, lifecycle, store)
  tools/            the governed workspace tools + registry + protected writes
  approvals/        the only path allowed to perform an external write
  brand/            brand identity synthesis and sync
app/ai/             The LangGraph side. May import app.core freely
  agents/           the orchestrator brain + 5 specialist graphs
  tools/            LangChain tools bound to those graphs
  providers/        LLM / search / db adapters, selected by enum, imported lazily
  memory/           session compressor + the post-turn consolidator
app/db/             models + session + Alembic history
app/schemas/        every request, response and UI block contract
mcp_servers/        standalone FastMCP brand-research server (not a package)
scripts/            operator + dev scripts (backup, seed, schema checks)
tests/              contract tests, one file per behaviour area
```

**Import direction.** `app.ai` imports `app.core` freely — that is the normal
direction. `app.core` also imports a handful of *leaf* `app.ai` modules
(`app.ai.security`, `app.ai.settings`, `app.ai.providers.*`), which is accepted.
What is not accepted is eagerly importing the **agent graphs** from core: the
orchestrator is imported lazily inside `handle_message`
(`app/core/orchestrator/service.py`, grep `orchestrate`), so that
`import app.main` does not build the whole LangGraph stack. Keep new heavy AI
imports lazy in `service.py`.

---

## 3. Task router — "I want to change X"

Read the files in the order listed. `verify` is the narrowest test file that will
catch a mistake.

| # | Task | Read (in order) | Verify |
| :-- | :--- | :--- | :--- |
| 1 | Change what happens during a chat turn (context, persistence, SSE) | `app/api/assistant.py` → `app/core/orchestrator/service.py` (grep `handle_message`) | `tests/test_contract.py`, `tests/test_conversation_context_persistence.py` |
| 2 | Change **which agent** answers, or add a workflow | `app/ai/agents/orchestrator/brain.py` (grep `ROUTING_PROMPT`, `def orchestrate`, `RoutingDecision`) | `tests/run_routing_eval.py`, `tests/test_agent_graphs.py` |
| 3 | Change one agent's behaviour / prompt | `app/ai/agents/<agent>/graph.py` (`<AGENT>_PROMPT`) + `schema.py` | `tests/test_agent_graphs.py`, and the flow tests |
| 4 | Change how a report is rendered (headings, labels, order) | `app/ai/agents/orchestrator/presentation.py` | `tests/test_report_presentation.py` |
| 5 | Change the UI block contract | `app/schemas/blocks.py` → `app/core/orchestrator/blocks.py` | `tests/test_blocks_flattener_and_evidence.py`, `tests/test_contract.py` |
| 6 | Add or change a **workspace tool** | `app/core/tools/base.py` → `app/schemas/tools.py` → `app/core/tools/<family>.py` → `registry.py` → persona allowlist in `app/core/agents/personas.py` | `tests/test_tools.py`, `tests/test_registry_policy.py` |
| 7 | Add or change an **AI-native tool** bound to a graph | `app/ai/tools/<name>.py` → its binding in `app/ai/agents/<agent>/graph.py` | `tests/test_ai_wire_contracts.py` |
| 8 | Change memory retrieval / ranking / prompt assembly | `app/core/memory/store.py` (`build_context`) → `ranking.py` → `lexical.py` → `policy.py` | `tests/test_memory_layers.py`, `tests/test_memory_performance.py` |
| 9 | Change memory retention (TTL, pruning, dedupe) | `app/core/memory/kinds.py` → `lifecycle.py` → `store.py` (`write_item`, `prune`) | `tests/test_memory_layers.py` |
| 10 | Add a **memory kind** | `app/core/memory/kinds.py` only (one row) — nothing else enumerates kinds | `tests/test_memory_layers.py` |
| 11 | Change the reply language / bilingual behaviour | `app/core/i18n.py` → call sites listed in `docs/LANGUAGE_POLICY.md` §5 | `tests/test_reply_language.py` |
| 12 | Change the post-turn background work (summary, profile, identity) | `app/ai/memory/consolidator.py` → `app/ai/memory/compressor.py` → `app/core/brand/identity_service.py` | `tests/test_brand_identity.py` |
| 13 | Change the database schema | `app/db/models.py` → **new** revision in `app/db/migrations/versions/` | migration round-trip + full suite |
| 14 | Add or change an endpoint | `app/api/<area>.py` → `app/schemas/<area>.py` → `app/core/<area>/service.py` | the matching `tests/test_*_api.py` |
| 15 | Change auth, tenancy or workspace resolution | `app/api/deps.py` → `app/core/auth.py` → `app/core/supabase_auth.py` | `tests/test_auth.py`, `tests/test_workspace_isolation.py` |
| 16 | Change Meta campaign reading | `app/core/tools/meta_read.py` → `app/ai/marketing/insights.py` is **unused**, `meta_ads.py` is live | `tests/test_meta_campaign_discovery.py`, `tests/test_ga4_tool.py` |
| 17 | Change Meta ad **creation** | `app/ai/marketing/meta_ads.py` → `app/core/approvals/service.py` → `app/core/tools/protected.py` | `tests/test_meta_ad_plans.py`, `tests/test_approvals.py` |
| 18 | Change the research hub | `app/api/research.py` → `app/core/tools/research.py` → `app/core/research_reports.py` | `tests/test_api_tools.py`, `tests/test_research_report_shaping.py` |
| 19 | Change content calendar generation | `app/api/content.py` → `app/ai/tools/strategy/generate_content_calendar.py` | `tests/test_content_calendar*.py` |
| 20 | Change creatives (copy, image, reference style) | `app/api/creatives.py` → `app/core/tools/media.py` → `app/ai/marketing/reference_vision.py` | `tests/test_creative_*`, `tests/test_adimage_bytes.py` |
| 21 | Change LLM tracing, logs or **cost estimation** | `app/ai/observability.py` → `app/ai/loggers/callbacks.py` → `app/ai/loggers/cost.py` | `tests/test_observability.py`, `tests/test_llm_diagnostic_logger.py` |
| 22 | Change plans, token budgets or plan switching | `app/core/subscriptions.py` → `app/schemas/subscriptions.py` → `app/api/subscriptions.py` | `tests/test_subscriptions.py` |
| 23 | Diagnose *why* an agent took a wrong turn | `app/ai/loggers/agents/diagnostic.py` (`log_agent_diagnosis`) → `logs/agents/diagnostic.log` | `tests/test_agent_diagnostic_logger.py` |
| 21 | Change the competitor-ad flow | `app/ai/agents/nour/meta_ads.py` → `app/ai/tools/ad_previews.py` → `docs/AD_CREATIVE_FLOW.md` | `tests/test_meta_competitor_ads_flow.py` |
| 22 | Change brand identity / profile sync | `app/core/brand/identity_service.py` | `tests/test_brand_identity.py`, `tests/test_profile_field_families.py` |
| 23 | Change settings, connectors or provider selection | `app/api/settings.py` → `app/core/tools/settings.py` → `app/core/tools/connectors.py` → `app/core/llm/router.py` | `tests/test_api_tools.py`, `tests/test_llm_layer.py` |
| 24 | Add a setting | `app/config.py` **and** `app/ai/settings.py` (separate objects) → `.env.example` | `tests/test_config_surface.py` |
| 25 | Change the generic agent runtime (Layer 5) | `app/core/agents/runtime.py` (`RunOptions`, `_execute`) → `tasks.py` | `tests/test_runtime.py` |
| 26 | Change rate limits / security headers / CORS | `app/core/ratelimit.py`, `app/core/security_headers.py`, `app/config.py` | `tests/test_security.py`, `tests/test_deployment_ops.py` |
| 27 | Change persona defaults or autonomy gates | `app/core/agents/personas.py` → `persona.py` → `app/core/agents/runtime.py` (`_apply_autonomy_policy`) | `tests/test_registry_policy.py`, `tests/test_runtime.py` |

---

## 4. Module map

**Status legend** — `LIVE` in the request path · `SUPPORT` imported by live code ·
`LEGACY` works but is not the source of truth · `UNUSED` not wired anywhere ·
`GEN` generated.

### HTTP layer — `app/api/`

| File | One line | Status |
| :--- | :--- | :--- |
| `main.py` (`app/`) | app factory, middleware order, router registration, lifespan | LIVE |
| `assistant.py` | `/assistant/message`, `/message/stream`, conversations, attachments, ad previews | LIVE |
| `deps.py` | `WorkspaceDep`, `SessionDep`, `LLMDep`; the tenant boundary | LIVE |
| `auth.py`, `supabase_auth.py` (core), `auth_complete_profile.py` | login/refresh/logout/me, JWT + opaque sessions | LIVE |
| `campaigns.py` | campaign CRUD, wizard strategy, Meta plan create/assist | LIVE |
| `content.py` | calendar read/generate/patch/delete | LIVE |
| `creatives.py` | generate copy/image, reference analysis, upload, list | LIVE |
| `insights.py` | **no router prefix** — routes are `/campaigns/{id}/insights` and `/campaigns/{id}/ads/insights` | LIVE |
| `recommendations.py` | list + approve/reject/activate (the approval gate) | LIVE |
| `memory.py` | `/memory/items` CRUD, archive, `/memory/prune` | LIVE |
| `research_compare.py` | `POST /research/compare` — two competitors, fixed 8-dimension gap analysis, saved to memory | LIVE |
| `subscriptions.py` | `GET /subscriptions/plans\|current\|observability`, `POST /subscriptions/switch` | LIVE |
| `research.py` | `/research/run`, `/research/items` | LIVE |
| `tasks.py` | background agent tasks + SSE progress | LIVE |
| `tools.py` | tool catalog, connectors, `/tools/{name}/invoke` | LIVE |
| `settings.py` | connectors + per-workspace LLM settings | LIVE |
| `audiences.py`, `personas.py`, `business.py`, `integrations.py`, `media_assets.py`, `usage.py`, `audit.py` | CRUD / config / governance surfaces | LIVE |

### Observability, cost & plans

| File | One line | Status |
| :--- | :--- | :--- |
| `ai/observability.py` | unified LangSmith + Langfuse + local-diagnostic callback router; each backend is optional and unset = silently skipped | LIVE |
| `ai/loggers/base.py` | `ComponentLogger` factory — one file per component under `logs/<kind>/` | LIVE |
| `ai/loggers/callbacks.py` | LangChain callback: LLM start/finish/error, token counts, latency, USD cost, failure classification | LIVE |
| `ai/loggers/cost.py` | token pricing table (OpenAI, Anthropic, Gemini, Groq, DeepSeek, Ollama/mock free) + `estimate_token_cost` | LIVE |
| `ai/loggers/agents/{omar,layla,karim,nour,salma,orchestrator}.py` | per-specialist file loggers | LIVE |
| `ai/loggers/agents/diagnostic.py` | `log_agent_diagnosis()` — branches, fallbacks, handoffs → `logs/agents/diagnostic.log` | LIVE |
| `core/subscriptions.py` | plan catalog (free / pro / enterprise) + workspace budget switcher | LIVE |
| `schemas/subscriptions.py` | Pydantic contracts for plans and subscription status | LIVE |

Log files: `logs/providers/llm.log` (every model call), `logs/agents/<agent>.log`,
`logs/agents/diagnostic.log`. Full walkthrough —
[`docs/OBSERVABILITY_LOGGING_AND_SUBSCRIPTIONS_GUIDE.md`](docs/OBSERVABILITY_LOGGING_AND_SUBSCRIPTIONS_GUIDE.md).

### Conversation & orchestration

| File | One line | Status |
| :--- | :--- | :--- |
| `core/orchestrator/service.py` | `handle_message`, sticky context, language persistence, SSE, task queue | LIVE |
| `core/orchestrator/blocks.py` | builds `Message.blocks` from a result; validates them | LIVE |
| `ai/agents/orchestrator/brain.py` | `orchestrate()`: router, system messages, agent branches, creative-prompt machine | LIVE |
| `ai/agents/orchestrator/handoff.py` | extracts a submitted report from a graph's final state | LIVE |
| `ai/agents/orchestrator/presentation.py` | structured report → Markdown (per-language labels) | LIVE |

### Agents — `app/ai/agents/`

| File | One line | Status |
| :--- | :--- | :--- |
| `<agent>/graph.py` | LangGraph: prompt, tools, ReAct loop, `submit_*` | LIVE |
| `<agent>/schema.py` | the typed report that graph must return | LIVE |
| `<agent>/state.py` | graph state shape | LIVE |
| `omar/tools.py`, `omar/service.py` | run-local planning tools; wizard strategy | LIVE |
| `layla/service.py` | direct text deliverables (`/creatives/generate-text`) | LIVE |
| `nour/meta_ads.py` | competitor-ad library + vision + prompt | LIVE |
| `ai/agents/nour/gui_graph.py` | an alternate single-agent formatter, imported only by a test | UNUSED |

### Tools — `app/core/tools/` and `app/ai/tools/`

| File | One line | Status |
| :--- | :--- | :--- |
| `core/tools/base.py` | `Tool`, `ToolContext`, `ToolRegistry` (refuses `external_write`) | LIVE |
| `core/tools/registry.py` | assembles `ALL_TOOLS`; the only way agents get tools | LIVE |
| `core/tools/research.py` | 9 research tools | LIVE |
| `core/tools/memory.py` | `get_brand_context`, `save_learning` | LIVE |
| `core/tools/meta_read.py` | read-only Meta campaign/insight tools | LIVE |
| `core/tools/media.py` | reference analysis + image generation | LIVE |
| `core/tools/records.py` | `create_recommendation` | LIVE |
| `core/tools/protected.py` | external writes — importable **only** from approvals | LIVE |
| `ai/tools/*.py` | LangChain tools bound to graphs (`web_search`, `analyze_website`, `fetch_ga4_performance`, …) | LIVE |
| `ai/tools/core_bridge.py` | adapters from the typed registry into the graphs | SUPPORT |
| `ai/marketing/meta_ads.py` | Meta plan validation + paused creation + ad insights | LIVE |
| `ai/marketing/meta_plan_assist.py`, `meta_ads_mcp.py` | read-only MCP assistance for the wizard | LIVE |
| `ai/marketing/reference_vision.py`, `creative_prompt.py` | vision + prompt composition for creatives | LIVE |
| `ai/marketing/assets.py`, `context.py`, `generation.py`, `insights.py`, `references.py`, `schemas.py` | earlier generation layer, superseded by `ai/tools/strategy/*` and `core/tools/*` | UNUSED |

### Memory

| File | One line | Status |
| :--- | :--- | :--- |
| `core/memory/kinds.py` | the 13 kinds and their layer/TTL/importance/token cap | LIVE |
| `core/memory/store.py` | the only writer and reader of `brand_memory_items`; `build_context` | LIVE |
| `core/memory/ranking.py`, `lexical.py` | BM25 + recency + affinity scoring | LIVE |
| `core/memory/lifecycle.py` | dedupe digest, expiry, prune order | LIVE |
| `core/memory/policy.py` | every knob, one frozen object | LIVE |
| `core/memory/tokens.py` | one token counter (Arabic-aware) | LIVE |
| `ai/memory/compressor.py` | session summary at the token cap | LIVE |
| `ai/memory/consolidator.py` | one debounced post-turn pass per workspace | LIVE |
| `ai/memory/brand_memory.py` | LangGraph `BaseStore` memory — fallback only when no DB is in scope | LEGACY |

### Providers, runtime, infrastructure

| File | One line | Status |
| :--- | :--- | :--- |
| `ai/providers/llm/*` | one module per provider, selected by enum, imported lazily | LIVE |
| `core/llm/router.py` + `*_client.py` | the Layer-5 client protocol (used by `runtime.py`) | LIVE |
| `ai/providers/search/*`, `db/*` | search and checkpoint stores | LIVE |
| `core/agents/personas.py`, `core/agents/persona.py` | persona catalog + override merge | LIVE |
| `core/agents/tasks.py` | task specs, prompt templates, `build_prompt` | LIVE |
| `core/agents/runtime.py` | generic tool-calling loop with hard caps | LIVE |
| `core/approvals/service.py` | the only executor of external writes | LIVE |
| `core/i18n.py` | reply-language resolver | LIVE |
| `core/security.py`, `core/auth.py`, `core/logging.py`, `core/usage.py`, `core/errors.py` | cross-cutting services | LIVE |
| `db/models.py`, `db/session.py` | schema + engine | LIVE |
| `db/migrations/versions/*` | Alembic history | GEN |
| `mcp_servers/brand_mcp.py` | brand_report / full_analysis MCP server; `app/core/tools/research.py` puts this directory on `sys.path` and imports `brand_report` from it | LIVE |
| `scripts/db_admin.py` | Postgres backup/restore; imported by `tests/test_deployment_ops.py` | LIVE |
| `scripts/seed_demo_data.py`, `scripts/check_*.py` | demo seeding and schema/integrity checks | SUPPORT |
| `scripts/test_*.py` | ad-hoc probes, **not** part of the suite (`testpaths = ["tests"]`); safe to run by hand, never collected | SUPPORT |
| `scripts/validate_memory.py` | 15-check behavioural harness for brand memory (cost, retrieval, latency, dedupe, lifecycle, multi-worker lock); reproduces the numbers in `docs/MEMORY_VALIDATION_REPORT.md`. Needs no LLM or DB server; writes `memory_validation_results.json` (gitignored) | SUPPORT |
| `ai/loggers/*` | per-agent / per-provider file logs, wired through callbacks | SUPPORT |

---

## 5. Invariants — break one and the suite fails

1. **Tenancy comes from the dependency, never the body.** Every query filters by
   `business_id` / `workspace_id` from `WorkspaceDep`. A foreign id returns **404**,
   not 403. → `tests/test_workspace_isolation.py`
2. **No agent may cause an external write.** `ToolRegistry.register()` raises on
   `side_effect="external_write"`; `get_tools_for()` filters; `_run_one_tool`
   re-checks. External writes live in `core/tools/protected.py` and run only from
   `core/approvals/service.py`. → `tests/test_registry_policy.py`, `test_approvals.py`
3. **The reply language is decided once, in `core/i18n.py`.** Never re-derive it
   from `prompt`. An explicit UI choice beats everything, including the script the
   user typed. → `tests/test_reply_language.py`
4. **Agents return typed Pydantic reports, not prose.** The client only ever sees
   UI blocks. Adding a block kind means `schemas/blocks.py` + the discriminator.
   → `tests/test_contract.py`
5. **Memory writes go through `BrandMemoryStore`.** Hand-building
   `BrandMemoryItem` loses the dedupe key, the TTL and the token cap.
   → `tests/test_memory_layers.py`
6. **The orchestrator stays a lazy import inside `service.py`.** `app.core` may
   import leaf `app.ai` modules (`security`, `settings`, providers); it must not
   pull the agent graphs into the import path of `app.main`.
7. **Every graph must prepend its own role prompt** by checking whether that
   prompt is already present — not by testing `messages[0]`, because the
   orchestrator prepends system messages. → `tests/test_reply_language.py::test_karim_keeps_his_role_prompt_*`
8. **Schema changes are two files:** the model **and** an Alembic revision. Run
   it forward and back once. → `tests/test_deployment_ops.py`
9. **Structured logging takes `extra={...}`.** `logger.info("x", {"a": 1})` raises.
10. **A tool that cannot reach its source returns `status="unavailable"` with a
    reason.** It never returns invented data and never raises into the graph.

---

## 6. Recipes

### Add an endpoint
1. `app/schemas/<area>.py` — request + response models (aliases are camelCase).
2. `app/api/<area>.py` — route, `WorkspaceDep`/`SessionDep`, no logic.
3. `app/core/<area>/service.py` — the logic, transaction-bound.
4. Register the router in `app/main.py` if it is a new module.
5. Test: `tests/test_<area>_api.py`.

### Add a workspace tool
1. Input/output models → `app/schemas/tools.py`.
2. `async def _handle(request, ctx: ToolContext)` in the right family module.
3. `Tool(name=..., side_effect=READ|DRAFT_WRITE, handler=_handle)`.
4. Append to the family list in `app/core/tools/registry.py`.
5. Add the name to the personas that may use it (`app/core/agents/personas.py`).
6. Test with a fake `ToolContext`: success **and** unavailable path.

### Add an AI-native tool to one agent
Follow `docs/DEVELOPER_GUIDE.md` §7. Bind it in that agent's `graph.py`. Do not
touch the registry — those tools are not workspace-governed.

### Add a memory kind
One row in `app/core/memory/kinds.py`. Nothing else enumerates kinds. If the kind
must be creatable from the API, set `api_writable=True`.

### Change a prompt
- Chat agents: `<AGENT>_PROMPT` in `app/ai/agents/<agent>/graph.py`.
- Layer-5 tasks: `app/core/agents/prompts/<task>.md` (templates own the wording;
  `tasks.py` owns the slots).
- Never hardcode a language in a prompt — the `[RESPONSE_LANGUAGE]` block owns it.

### Add or change a setting
`app/config.py` **and** `app/ai/settings.py` (two objects by design) →
`.env.example` with a comment → `tests/test_config_surface.py`.

### Add a migration
`alembic revision --autogenerate -m "..."` in `app/db/migrations/`; review it
(SQLite and Postgres differ), run `upgrade head` and `downgrade -1`.

---

## 7. Verification

```bash
PYTHONPATH=. python3 -m pytest tests -q                 # everything (880)
PYTHONPATH=. python3 -m pytest tests -q -x              # stop at first failure
PYTHONPATH=. python3 -m pytest tests/test_memory_layers.py -v
PYTHONPATH=. python3 -m pytest tests -q -k "language or memory"
```

`tests/test_memory_performance.py` is a latency guard; `tests/run_routing_eval.py`
is a routing evaluation script, not a unit test.

**Environment note:** install with extras — `pip install -e ".[dev]"` — or the two
MCP tests fail on a missing `fastmcp` rather than skipping. A bare runtime install
is enough to run the app, not the suite. `tests/test_docs_map.py` verifies this
file: if you move a file it references, that test fails until the map is updated.

### Test file → subsystem

| Tests | Pin |
| :--- | :--- |
| `test_contract.py`, `test_blocks_flattener_and_evidence.py`, `test_error_response_surface.py` | API + block contract |
| `test_workspace_isolation.py`, `test_auth.py`, `test_security.py` | tenancy, auth, SSRF, headers |
| `test_agent_graphs.py`, `test_omar_layla_handoff*.py`, `test_karim_campaign_context.py` | graphs and handoffs |
| `test_creative_*.py`, `test_ad_creative_flow_contract.py`, `test_layla_*.py` | creative flows |
| `test_memory_layers.py`, `test_memory_performance.py` | memory store and ranking |
| `test_reply_language.py` | the bilingual contract |
| `test_runtime.py`, `test_registry_policy.py`, `test_tools.py` | runtime, tool policy, tool behaviour |
| `test_meta_*.py`, `test_ga4_tool.py`, `test_targeting_search.py` | Meta and analytics |
| `test_content_calendar*.py`, `test_strategy_tools.py`, `test_campaign_*.py` | calendar, strategy, campaigns |
| `test_task_routes.py`, `test_usage_rate_limit.py`, `test_deployment_ops.py` | tasks, governance, ops |
| `test_docs_map.py` | keeps this map honest: every path and link in it resolves |
| `conftest.py` | isolated SQLite per test, seeded workspace, API client |

---

## 8. Finding a symbol without reading a file

```bash
grep -n "def <name>" app/ai/agents/orchestrator/brain.py
grep -n "def handle_message\|def _persist_message" app/core/orchestrator/service.py
grep -n "def build_context\|def write_item\|def prune" app/core/memory/store.py
grep -n "async def run_agent\|class RunOptions" app/core/agents/runtime.py
grep -rn "_creative_prompt_blocks\|REPORT_LANGUAGE" app --include=*.py
grep -rn "TODO\|FIXME\|HACK" app --include=*.py        # known rough edges
```

`brain.py` is organised top-to-bottom as: constants → prompt text → small
helpers → `orchestrate()` → branches. If you are looking for a behaviour, grep a
distinctive string first (`[RESPONSE_LANGUAGE]`, `creative_prompt_stage`,
`ROUTING_PROMPT`) and read ±80 lines.

---

## 9. Documentation

| Need | File |
| :--- | :--- |
| Product in one page, for anyone | [`README.md`](README.md) |
| Design, schema, API, runbook | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Routing, graphs, post-turn workers | [`docs/ORCHESTRATOR.md`](docs/ORCHESTRATOR.md) |
| Memory: kinds, ranking, retention | [`docs/MEMORY_ARCHITECTURE.md`](docs/MEMORY_ARCHITECTURE.md) |
| Bilingual answer contract | [`docs/LANGUAGE_POLICY.md`](docs/LANGUAGE_POLICY.md) |
| Adding tools/agents, safe zones | [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) |
| Tables and conventions | [`docs/DATABASE_GUIDE.md`](docs/DATABASE_GUIDE.md) |
| Run it / operate it | [`docs/RUNNING_BACKEND.md`](docs/RUNNING_BACKEND.md) |
| PR rules | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
