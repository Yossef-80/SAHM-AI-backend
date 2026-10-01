# Project Memory

> Persistent memory for AI coding agents working on this project.
> Updated after each completed task.

## 1. Project Goal

Sahm is an AI marketing operating system: a multi-agent marketing platform providing autonomous and human-in-the-loop marketing agents. Features include market research, campaign strategy and planning, ad copy and creative brief generation, content calendar management, performance insights, and channel integrations.

## 2. Current State

- Status: `IN PROGRESS`
- Current milestone: Technical backend plan complete (`BACKEND_PROJECT_PLAN.md`); ready for backend parallel implementation across 5 developers.
- What currently works:
  - Frontend Next.js 14 App Router project with mock fixtures (`MockApi`), bilingual support (`next-intl` EN/AR with RTL), TanStack Query hooks, Zustand state, and action dispatcher for human-in-the-loop approvals.
  - Comprehensive architectural blueprint, data model (17 entities), API contracts, and 5-developer parallel execution plan documented in `BACKEND_PROJECT_PLAN.md`.
  - Backend FastAPI scaffolding with router structures and initial models.
- What is currently incomplete:
  - Real backend implementation to replace mocks (`NEXT_PUBLIC_USE_MOCKS=false`).
  - Missing `greenlet` in backend dependencies preventing `sqlalchemy.ext.asyncio` test execution under Python 3.13.
  - Node.js runtime not accessible on system PATH in current shell session for frontend script execution.

## 3. Project Structure

- `BACKEND_PROJECT_PLAN.md` — Comprehensive backend engineering roadmap, data model, API contract, and 5-developer parallel work split.
- `PROJECT_MEMORY.md` — Persistent handoff memory across AI coding agents.
- `SAHM-AI-backend/app/main.py` — FastAPI application entry point, lifespan, error handlers, and router wiring.
- `SAHM-AI-backend/app/config.py` — Central environment configuration settings and LLM/provider routing.
- `SAHM-AI-backend/app/api/` — API routers for assistant, campaigns, creatives, personas, tasks, etc.
- `SAHM-AI-backend/app/core/` — Multi-agent orchestrator, LLM abstractions, tool registry, memory, and approvals.
- `SAHM-AI-backend/app/db/` — SQLAlchemy models (`models.py`) and async session management (`session.py`).
- `SAHM-AI-backend/tests/` — Pytest test suite covering approvals, API contract, LLM layer, and tool execution.
- `SAHM-AI-backend/pyproject.toml` — Backend project dependencies, build metadata, and test configuration.
- `SAHM-AI-frontend/src/app/` — Next.js App Router pages (assistant, research, campaigns, calendar, creatives, insights, agents, settings).
- `SAHM-AI-frontend/src/lib/api/` — `MarketingApi` interface, `MockApi`, and `HttpApi` client implementations.
- `SAHM-AI-frontend/src/lib/actions/` — Action Registry and human-in-the-loop execution dispatcher.
- `SAHM-AI-frontend/src/store/` — Zustand client state stores (assistant stream, dialogs).
- `SAHM-AI-frontend/docs/` — API contract specifications (`API_CONTRACT.md`) and action mappings (`ACTIONS.md`).
- `SAHM-AI-frontend/package.json` — Frontend dependencies and scripts.

## 4. Completed Work

### Task: Author Backend Technical Planning Document
- Date/time: 2026-10-01
- Changed:
  - `BACKEND_PROJECT_PLAN.md` — Created complete backend plan covering product reconstruction, feature inventory (P0–P3), architecture, data model, API contracts, critical path, 5-developer parallel split, git strategy, and implementation checklist.
- Result:
  - Complete, unambiguous technical specification ready for a 5-developer backend team to execute.

### Task: Initialize Project Memory
- Date/time: 2026-10-01
- Changed:
  - `PROJECT_MEMORY.md` — Created initial persistent project memory file adhering to specified format and maintenance rules.
- Result:
  - Repository inspected, backend and frontend structures mapped, environment verified, and initial blockers documented.

## 5. Important Decisions

- Dual-mode architecture: Backend boots with `LLM_PROVIDER=mock` and frontend with `NEXT_PUBLIC_USE_MOCKS=true` allowing zero-external-dependency execution and testing.
- Single configuration authority: All model names and providers live strictly in `SAHM-AI-backend/app/config.py` driven by env variables; no hardcoded provider strings elsewhere.
- Human-in-the-loop safety: Destructive or spending marketing actions (campaign launch, budget change, ad publication) require explicit user approval via action registry.
- SQLite default with pgvector capability: Uses SQLite (`sqlite+aiosqlite`) with in-process cosine similarity for dev/tests, with production target being Postgres + pgvector.
- SSE Streaming Snapshot Protocol: Each `data:` line in `/conversations/messages/stream` contains a full `SendMessageOutput` snapshot JSON, with `streaming: true` on in-progress text blocks.

## 6. Requirements / Constraints

- Persistent memory: Always maintain and update `PROJECT_MEMORY.md` after every completed task.
- Documentation integrity: Preserve all existing comments, docstrings, and contracts unless explicitly asked to modify them.
- No exposed secrets: Never store API keys, access tokens, credentials, or `.env` secret values in `PROJECT_MEMORY.md`.
- File paths: Always reference real repository file paths in memory and logs.

## 7. Environment

- OS: Windows (x86_64)
- Python: 3.13.14 (in `SAHM-AI-backend/.venv`)
- Environment: Virtual environment managed via `uv`
- Package manager: `uv` (v0.12.21) for backend; `npm` for frontend
- Important dependencies:
  - Backend: FastAPI (>=0.115), SQLAlchemy (>=2.0.30), Pydantic (>=2.7), LangGraph (>=0.2.60), aiosqlite, asyncpg, httpx
  - Frontend: Next.js (14.2.35), React (18), TanStack Query (5.103.2), Zustand (5.0.15), Zod (4.6.5), next-intl (4.14.7), Tailwind CSS
- Runtime requirements:
  - Python 3.11+
  - SQLite (`sahm.db`) or PostgreSQL database

## 8. Known Issues / Blockers

- `ImportError: The SQLAlchemy asyncio module requires that the Python 'greenlet' library is installed` — `SAHM-AI-backend/pyproject.toml` — Running `uv run --extra dev pytest` fails during `conftest.py` loading because `greenlet` (or `SQLAlchemy[asyncio]`) is not explicitly listed in dependencies.
- `node` / `npm` command not found on shell PATH — `SAHM-AI-frontend` — Node.js executable is not available in the current environment PATH, preventing local frontend build/test commands from shell.

## 9. Important Commands

```bash
# Backend: run tests with dev dependencies
uv run --extra dev pytest

# Backend: start FastAPI server locally
uv run uvicorn app.main:app --reload --port 8000

# Backend: format / lint check (using ruff if configured)
uv run ruff check .

# Frontend: start development server (when node is available)
npm run dev

# Frontend: typecheck and lint
./node_modules/.bin/tsc --noEmit && npm run lint
```

## 10. Current Continuation Point

### Last Completed Task
Created [`BACKEND_PROJECT_PLAN.md`](file:///d:/NTI-Final-Project/BACKEND_PROJECT_PLAN.md) at the project root after deep inspection of the frontend codebase, establishing product feature inventory, data model, API contracts, critical path, and a 5-developer parallel execution plan.

### Current Blocker
Backend pytest suite fails to load due to missing `greenlet` dependency required by `sqlalchemy.ext.asyncio` under Python 3.13.

### Next Recommended Step
Add `greenlet` (or update dependency to `SQLAlchemy[asyncio]>=2.0.30`) in `SAHM-AI-backend/pyproject.toml`, run `uv sync --extra dev`, and run `uv run --extra dev pytest` to verify tests pass before Developer 1 begins foundational implementation.

### Expected Result
Pytest passes cleanly, and the 5 backend developers can begin their respective modules as outlined in `BACKEND_PROJECT_PLAN.md`.

## 11. Agent Notes

- Frontend expects camelCase responses matching TypeScript types, even though database columns are snake_case. Ensure all Pydantic schemas configure camelCase aliases (`to_camel`).
- The Server-Sent Events (SSE) chat stream sends full snapshot JSON payloads (`SendMessageOutput`), not text deltas.
- Polling for `AgentTask` occurs every 1000ms on `GET /tasks/:id`; tasks must transition from `queued` -> `running` -> `succeeded` or `failed`.
