# Sahm — AI Marketing Operating System (Backend)

> **سهم**: an autonomous marketing department for small businesses, in Arabic and English.
> FastAPI · LangGraph multi-agent · PostgreSQL · deterministic brand memory.

Sahm turns one chat box into a working marketing team. A business owner asks for a
competitor analysis, a campaign strategy, an ad script, a content calendar or a
performance review, and a squad of specialist agents does the work — with live
market evidence, the brand's own stored context, and a human approving anything
that would change a live ad account.

This repository is the **backend**: the API, the agent orchestration, the memory
system and the database. The React frontend is a separate project that consumes
this API's JSON "UI blocks".

---

## Table of contents

1. [What Sahm offers](#1-what-sahm-offers)
2. [How it works (in one page)](#2-how-it-works-in-one-page)
3. [The request path, file by file](#3-the-request-path-file-by-file)
4. [Design decisions a reviewer should know](#4-design-decisions-a-reviewer-should-know)
5. [Repository map](#5-repository-map)
6. [Quick start](#6-quick-start)
7. [Configuration](#7-configuration)
8. [API surface](#8-api-surface)
9. [Data model](#9-data-model)
10. [Language behaviour](#10-language-behaviour)
11. [Testing](#11-testing)
12. [Documentation](#12-documentation)
13. [Known limitations](#13-known-limitations)

---

## 1. What Sahm offers

### In plain language

A small brand in Cairo cannot afford five specialists. It also cannot use six
disconnected AI tools, because none of them remembers the brand, the competitor
research, or the last campaign's numbers. Sahm is one product where the research,
the strategy, the creative, the calendar and the analytics **share one memory**,
so each step starts from what the previous step learned.

| Who you are | What you get |
| :--- | :--- |
| A brand owner | Ask a question in Arabic or English; get research, a plan, ad copy, a content calendar, and a review of what happened — in one conversation. |
| A marketer | A workspace that remembers the brand voice, offers, audience, competitors and past results, and reuses them in every deliverable. |
| An engineer | A typed, workspace-isolated API with SSE progress events, deterministic memory retrieval, approval-gated writes, and an audit trail. |

### The five specialists

| Agent | Role | Typical request |
| :--- | :--- | :--- |
| **Nour** نور | Market & competitor researcher | "Analyse this competitor's site and reviews", "Where are the market gaps?" |
| **Omar** عمر | Marketing strategist | "Build me a 30-day plan", "Campaign strategy for Ramadan", "Content calendar" |
| **Layla** ليلى | Creative director | "Write three ad hooks", "A 15-second script", "An image prompt for this product" |
| **Karim** كريم | Performance analyst | "Why did CTR drop?", "Read the campaign insights" |
| **Salma** سلمى | Optimiser | "What should I change in this campaign?" (proposals only, never silent changes) |

Behind them sits an **orchestrator** that routes each turn, and a fast chat
fallback for ordinary questions that need no tools.

### Platform capabilities

- **Research hub** — 9 research tools: web search, page analysis, social profiles,
  reviews, keywords, competitor ad library, brand report, full analysis, and file
  routing.
- **Assistant chat** — streaming answers with progress events; the answer is a
  list of typed UI blocks, not a blob of markdown.
- **Brand identity & profile** — one living record of voice, palette, audience,
  offers, competitors, with a one-click sync from chats.
- **Content calendar** — multi-day posts generated and persisted.
- **Campaigns & creatives** — campaign structures, ad copy, image/video prompts,
  reference-image style extraction.
- **Insights & recommendations** — findings, proposed changes, two-phase approval
  with before/after diffs.
- **Memory** — 13 lifecycle-managed kinds, retrieved deterministically in under
  900 tokens per prompt.
- **Governance** — per-workspace token budgets, immutable audit log, RBAC, and an
  agent runtime that physically cannot call an external write tool.

---

## 2. How it works (in one page)

```
Browser (React)
    │  POST /api/v1/assistant/message/stream   { message, language, context }
    ▼
FastAPI router ................ app/api/assistant.py          thin, validating
    ▼
Conversation service .......... app/core/orchestrator/service.py
    │   • loads the conversation, its summary and its sticky context
    │   • resolves the reply language  (app/core/i18n.py)
    │   • builds brand memory context  (app/core/memory/store.py)
    ▼
Orchestrator brain ............ app/ai/agents/orchestrator/brain.py
    │   • one LLM routing decision: which specialist, single or two-step
    │   • injects [RESPONSE_LANGUAGE], brand profile, campaigns, memory
    ▼
Specialist graph (LangGraph) .. app/ai/agents/{nour,omar,layla,karim,salma}/graph.py
    │   • ReAct loop: model → tool → model → … → submit_<report> tool
    │   • tools come from a registry the persona must be allowlisted for
    ▼
Tool layer .................... app/core/tools/  +  app/ai/tools/
    │   • read tools: search, scraping, Meta read, GA4, memory
    │   • write tools: none with external side effects (see §4)
    ▼
Presentation .................. app/ai/agents/orchestrator/presentation.py
    │   • structured output → Markdown → typed UI blocks
    ▼
SSE stream + persisted message (content + blocks + usage)
```

Two orchestrators exist in the tree and this trips people up:

- **`app/core/orchestrator/`** is the *service* layer — conversations, tasks,
  persistence, SSE plumbing. Despite the name it does not route agents.
- **`app/ai/agents/orchestrator/`** is the *brain* — it holds `orchestrate()`,
  the routing prompt, the specialist graph wiring and the presentation layer.

The AI layer is organised as five layers, each with one job:

| Layer | Location | Responsibility |
| :--- | :--- | :--- |
| 1. Providers | `app/ai/providers/`, `app/core/llm/` | One interface over Ollama / OpenAI / Anthropic / Gemini / Groq / OpenRouter / mock. |
| 2. Personas | `app/core/agents/persona.py`, `personas.py` | Who an agent is, which tasks it may run, which tools it may use. |
| 3. Tasks | `app/core/agents/tasks.py`, `prompts/*.md` | Versioned prompt templates + Pydantic output schemas. |
| 4. Graphs | `app/ai/agents/*/graph.py` | LangGraph state machines for the chat agents. |
| 5. Runtime | `app/core/agents/runtime.py` | One generic tool-calling loop, with hard caps and an autonomy gate. |

---

## 3. The request path, file by file

| Step | File | What it does |
| :--- | :--- | :--- |
| 1 | `app/api/assistant.py` | Validates the body, resolves the workspace, opens the SSE response. |
| 2 | `app/core/orchestrator/service.py` | Loads conversation + summary, merges sticky context, **persists the UI language choice**, resolves this turn's reply language, builds memory context, calls `orchestrate()`, persists the message. |
| 3 | `app/ai/agents/orchestrator/brain.py` | Prepends system messages (brand profile, `[RESPONSE_LANGUAGE]`, campaign list, saved settings), asks the router LLM for a `RoutingDecision`, invokes the chosen graph. |
| 4 | `app/ai/agents/<agent>/graph.py` | Runs the agent's ReAct loop until it calls its `submit_*` tool. |
| 5 | `app/ai/agents/orchestrator/presentation.py` | Renders the structured report as Markdown in the chosen language. |
| 6 | `app/core/orchestrator/service.py` | Saves `content` + `blocks`, records usage, schedules memory consolidation. |

Everything in step 3 is *data*, not instructions: the brand profile, memory rows
and campaign names are wrapped in labelled system messages that say so, because
a brand can be named anything at all.

---

## 4. Design decisions a reviewer should know

**Deterministic memory, no vector database.** Brand memory is 13 kinds stored in
one table and retrieved with BM25 + recency + kind affinity. It is bilingual,
explainable (every hit reports why it ranked), and needs no embedding provider,
no reindex job, and no second store. See `docs/MEMORY_ARCHITECTURE.md`.

**The UI language is a contract, not a hint.** If the client says Arabic, every
user-facing string is Arabic — even when the user pastes English — and switching
the language mid-conversation switches the next answer. Implemented once in
`app/core/i18n.py` and threaded through the orchestrator, the specialist prompts
and the formatters. See §10 and `docs/LANGUAGE_POLICY.md`.

**The agent runtime cannot write externally.** Tools with
`side_effect="external_write"` are refused by the registry, filtered out of every
persona's list, and re-checked at call time. Anything that would change a live ad
account goes through a recommendation + human approval, and the approval writes
an immutable audit row. This is enforced in code, not in a prompt.

**Output is a typed protocol.** Agents never return free-form markdown to the
client. They return a Pydantic report, which the presentation layer turns into
UI blocks (`text`, `findings`, `metric`, `action_card`, `question_with_options`,
`image_preview`, …). The frontend renders a contract, so a model change cannot
silently break the UI.

**Tenancy is a parameter, not a convention.** Every query is scoped by
`business_id` / `workspace_id` resolved from the auth dependency, never from the
request body. Workspace isolation has its own test file.

**Fail closed, degrade loudly.** Missing provider keys produce a typed
`ProviderNotConfigured` error, not a crash; missing evidence produces
`missing_inputs` in the report instead of an invented number; a failed agent turn
returns a safe message and saves nothing partial.

---

## 5. Repository map

```
app/
├── main.py                  app factory, middleware, routers, lifespan
├── config.py                core settings (env → typed object)
├── api/                     HTTP layer only: validate, delegate, serialise
├── core/
│   ├── orchestrator/        conversation service, blocking/SSE turns, tasks
│   ├── agents/              personas, task specs, prompts, generic runtime
│   ├── memory/              brand memory: kinds, tokens, lexical, ranking, store
│   ├── tools/               workspace tools (research, memory, Meta read, media)
│   ├── brand/               brand identity service
│   ├── auth/, approvals/, campaigns/, audiences/, research_reports/
│   └── i18n.py              one reply-language resolver for the whole API
├── ai/
│   ├── agents/              orchestrator brain + 5 specialist LangGraphs
│   ├── providers/           LLM / VLM / image adapters, one interface
│   ├── memory/              session compressor + consolidator
│   └── tools/               AI-native tools (search, ads, GA4, image, save_learning)
├── db/                      SQLAlchemy models + session + Alembic migrations
└── schemas/                 every request/response/block contract

tests/                       880 tests: contracts, isolation, workflows, adapters
docs/                        architecture, memory, language, product, presentations
mcp_servers/brand_mcp.py     standalone brand-research MCP server (brand_report,
                             full_analysis); imported by app/core/tools/research.py
scripts/                     operator and dev scripts (db_admin backup/restore,
                             seed_demo_data, migration + schema checks)
AGENTS.md                    the map: task → files → tests (read this before the code)
```

---

## 6. Quick start

**Prerequisites:** Python ≥ 3.11, [`uv`](https://docs.astral.sh/uv/) (or plain
`pip`). No database server is required for a local run — SQLite is the default.

```bash
# 1. install
uv sync                              # or: python -m venv .venv && pip install -e .

# 2. configure
cp .env.example .env

# 3. create the schema (SQLite by default)
uv run alembic upgrade head

# 4. run
uv run uvicorn app.main:app --reload --port 8000
```

Check it:

```bash
curl -s localhost:8000/health          # liveness: process only
curl -s localhost:8000/health/ready    # readiness: dependencies
```

**No API key? Run it on the mock model.** `.env.example` ships with
`AI_LLM_PROVIDER=ollama`; set it to `mock` (and optionally
`LLM_PROVIDER_FAST` / `LLM_PROVIDER_STRONG`, which otherwise inherit it) and the
whole product runs offline: the mock model returns deterministic structured
output, so every route works and the test suite is green without a single key.

```bash
sed -i 's/^AI_LLM_PROVIDER=.*/AI_LLM_PROVIDER=mock/' .env
```

---

## 7. Configuration

Everything is environment-driven and documented inline in
[`.env.example`](.env.example). The groups that matter most:

| Group | Variables | Notes |
| :--- | :--- | :--- |
| Core | `ENVIRONMENT`, `DATABASE_URL`, `API_PREFIX`, `CORS_ORIGINS`, `REQUIRE_AUTH` | `sqlite+aiosqlite:///./sahm.db` by default. |
| Models | `AI_LLM_PROVIDER`, `AI_LLM_MODEL`, provider keys | `ollama`, `openai`, `anthropic`, `gemini`, `groq`, `openrouter`, `mock`. `LLM_PROVIDER_FAST`/`_STRONG` default to `AI_LLM_PROVIDER`. |
| Vision | `AI_ADS_VLM_PROVIDER`, `AI_ADS_VLM_MODEL`, `AI_ADS_VLM_BASE_URL` | Isolated from chat. |
| Images | `IMAGE_PROVIDER`, `IMAGE_BASE_URL`, `IMAGE_MODEL` | `mock` labels previews as placeholders. |
| Memory | `MEMORY_ENABLED`, `MEMORY_CONTEXT_TOKEN_CAP`, `MEMORY_CONTEXT_TOP_K`, `MEMORY_IDENTITY_SHARE`, `MEMORY_TTL_*`, fusion weights | Documented in `docs/MEMORY_ARCHITECTURE.md` §8. |
| Safety | `REQUIRE_AUTH`, `RATE_LIMIT_*`, `TOKEN_ENCRYPTION_KEY` | The dev key is published; production must set its own. |

Secrets never appear in logs: `app/core/logging.py` redacts token-like values
before they reach a handler, and the tool layer logs names and durations, not
payloads.

---

## 8. API surface

Base prefix: `/api/v1`. All routes are workspace-scoped by the auth dependency.

| Area | Endpoints |
| :--- | :--- |
| Assistant | `POST /assistant/message`, `POST /assistant/message/stream`, `GET /assistant/conversations`, `GET /assistant/conversations/{id}`, `POST /assistant/attachments`, `GET /assistant/ad-previews/{id}` |
| Tasks | `POST /tasks`, `GET /tasks/{id}`, `GET /tasks/{id}/stream`, `POST /tasks/{id}/cancel` |
| Research | `GET /research/items`, `POST /research/run` |
| Campaigns | `POST /campaigns`, `GET /campaigns`, `GET|PATCH /campaigns/{id}`, `POST /campaigns/meta-ad-plan`, `POST /campaigns/strategy`, `POST /campaigns/generate-strategy` |
| Content | `GET /content`, `POST /content/generate`, `PATCH|DELETE /content/{id}` |
| Creatives | `POST /creatives/generate`, `POST /creatives/generate-text`, `POST /creatives/analyze-reference`, `POST /creatives/upload`, `GET /creatives` |
| Analytics | `GET /campaigns/{campaign_id}/insights`, `GET /campaigns/{campaign_id}/ads/insights` |
| Recommendations | `GET /recommendations`, `POST /recommendations/{id}/approve`, `/activate`, `/reject` |
| Memory | `GET|POST /memory/items`, `PATCH|DELETE /memory/items/{id}`, `POST /memory/items/{id}/archive`, `POST /memory/prune` |
| Settings | `GET /settings/connectors`, `PATCH /settings/connectors/{connector}`, `GET|PATCH /settings/llm` |
| Governance | `GET /usage`, `GET /audit`, `GET /personas`, `PATCH /personas/{id}`, `GET /tools`, `POST /tools/{name}/invoke` |
| Business | `POST /business`, `GET|PATCH /business/profile`, `GET|PATCH /business/brand-identity`, `POST /business/brand-identity/sync` |
| Auth | `POST /auth`, `/auth/login`, `/auth/refresh`, `/auth/logout`, `GET /auth/me` |

`/assistant/message/stream` and `/tasks/{id}/stream` speak **Server-Sent Events**.
The stream emits `progress` frames while the turn runs (each carrying a
`step_started` / `tool_result` / `failed` detail), then one `complete` frame with
the final message and its blocks, or an `error` frame. The UI can therefore show
an agent working rather than a spinner.

---

## 9. Data model

23 tables in `app/db/models.py`, grouped by purpose:

| Cluster | Tables |
| :--- | :--- |
| Identity & tenancy | `businesses`, `brand_profiles`, `users`, `memberships`, `sessions`, `profiles` |
| Conversations & memory | `conversations`, `messages`, `brand_memory_items` |
| Marketing operations | `campaigns`, `content_items`, `creatives`, `audiences` (`saved_audiences`), `insights`, `recommendations` |
| Agents & audit | `agent_tasks`, `agent_tool_calls`, `audit_log`, `persona_overrides` |
| Configuration & usage | `integrations`, `tool_settings`, `llm_provider_settings`, `usage_records` |

Migrations live in `app/db/migrations/versions/` and run with
`alembic upgrade head`. `docs/DATABASE_GUIDE.md` explains the columns and the
conventions for adding a table.

---

## 10. Language behaviour

Sahm is bilingual by design (Arabic and English in the same workspace, often in
the same conversation). The rule is deliberately simple:

> **The language the user picked in the UI wins.** Arabic UI → Arabic answers.
> English UI → English answers. Switching the UI language switches the next
> answer, including mid-conversation.

Resolution order, applied once per turn in `app/core/i18n.py`:

1. the language sent with this request (the UI's current choice),
2. the last explicit choice stored on the conversation,
3. an **Arabic** project language (`businesses.language`),
4. the script of the message itself,
5. otherwise English.

Notes that matter in practice:

- A pasted English link from an Arabic-UI user no longer flips the conversation
  to English, and an Arabic-UI user still gets Arabic when they type English.
- A *detected* language is never persisted. Only an explicit choice is
  remembered, so a single Arabic sentence cannot trap an English user.
- The same resolver drives the chat, the specialist agents, the block labels, the
  creative-prompt interview and queued background tasks.

Full contract and test matrix: [`docs/LANGUAGE_POLICY.md`](docs/LANGUAGE_POLICY.md).

---

## 11. Testing

```bash
uv run pytest -q                     # whole suite
uv run pytest tests/test_memory_layers.py -v
uv run pytest -q -k "language or memory"
```

`tests/` holds 880 tests. They are contract tests, not smoke tests: the routing
decision, the UI block shapes, the approval gate, the workspace boundary, the
memory ranking and the reply language are each pinned by name.

A full install passes all 880. Two of them wrap the MCP server and need
`fastmcp`, which is a `[dev]` extra — so use `pip install -e ".[dev]"` (or
`uv sync --extra dev`) rather than a bare runtime install if you want the whole
suite green.

No test calls a live provider, a live Meta endpoint or the open web: every
external boundary has a fake, and `tests/conftest.py` builds an isolated SQLite
database per test.

---

## 12. Documentation

| Start here if you want… | Read |
| :--- | :--- |
| **To change code without reading the whole tree** | [`AGENTS.md`](AGENTS.md) — task router, module map, invariants, test matrix |
| The business story, in non-technical language | [`docs/NON_TECHNICAL_PRESENTATION_GUIDE.md`](docs/NON_TECHNICAL_PRESENTATION_GUIDE.md) |
| Every feature, end to end | [`docs/PROJECT_FEATURES_AND_CAPABILITIES.md`](docs/PROJECT_FEATURES_AND_CAPABILITIES.md) |
| System design, schema, API, runbooks | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| How memory actually works | [`docs/MEMORY_ARCHITECTURE.md`](docs/MEMORY_ARCHITECTURE.md) |
| How the language rule works | [`docs/LANGUAGE_POLICY.md`](docs/LANGUAGE_POLICY.md) |
| How to add a tool or an agent | [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) |
| How to run and operate it | [`docs/RUNNING_BACKEND.md`](docs/RUNNING_BACKEND.md) |
| Slide decks and demo scripts | [`docs/PRESENTATION_AND_DEMO_GUIDE.md`](docs/PRESENTATION_AND_DEMO_GUIDE.md) |

The full index, with reading paths per audience, is [`docs/README.md`](docs/README.md).
If you are an AI agent working in this repository, start with [`AGENTS.md`](AGENTS.md)
and use it as your routing table.

---

## 13. Known limitations

Honest list, because a reviewer will find them anyway:

- **Memory writes are serialised per process.** A single-worker deployment is
  correct; with several workers, two consolidations for the same workspace can
  race. The fix is a distributed lock or a single consolidation worker.
- **The router is an LLM call.** Routing is one cheap completion with a typed
  schema; a misroute is possible and is recovered by the chat fallback rather
  than by a hard failure.
- **Long conversations are summarised, not re-read.** Compaction is lossy by
  design; the summary keeps decisions and open questions, not transcripts.
- **Meta is read-only from the agent side.** Publishing and budget changes only
  happen through the approval flow.
- **Arabic quality depends on the model.** The prompts constrain register and
  script mixing, but a small local model will still produce weaker Arabic than a
  frontier model.
