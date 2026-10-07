# Project Memory

> Persistent memory for AI coding agents working on this project.
> Updated after each completed task.

## 1. Project Goal

Sahm is an AI marketing operating system: a multi-agent marketing platform providing autonomous and human-in-the-loop marketing agents. Features include market research, campaign strategy and planning, ad copy and creative brief generation, content calendar management, performance insights, and channel integrations.

## 2. Current State

- Status: `IN PROGRESS`
- Current milestone: Agent Memory Protocol Established (`AGENT_MEMORY_INSTRUCTIONS.md`). Specifications and governance guides complete; ready for 5 backend developers to build tools and agents concurrently.
- What currently works:
  - Frontend Next.js 14 App Router project with mock fixtures (`MockApi`), bilingual support (`next-intl` EN/AR with RTL), TanStack Query hooks, Zustand state, and action dispatcher for human-in-the-loop approvals.
  - Complete backend engineering blueprint, data model (17 entities), and API contracts in `BACKEND_PROJECT_PLAN.md`.
  - Multi-agent LangGraph specifications, customer identity injection protocols, and visual Mermaid workflows in `AGENTS_ARCHITECTURE_PLAN.md`.
  - Master catalog of 21 unique tools, cross-agent matrix, 8 governance laws, and 5-package tool distribution in `AGENTS_TOOLS_GOVERNANCE.md`.
  - Standardized agent onboarding prompt and memory maintenance instructions in `AGENT_MEMORY_INSTRUCTIONS.md`.
  - Backend FastAPI scaffolding with router structures and initial models.
- What is currently incomplete:
  - Real tool and agent implementations across the 5 personas.
  - Missing `greenlet` in backend dependencies preventing `sqlalchemy.ext.asyncio` test execution under Python 3.13.
  - Node.js runtime not accessible on system PATH in current shell session for frontend script execution.

## 3. Project Structure

- `AGENT_MEMORY_INSTRUCTIONS.md` — Operating prompt and protocol guide for any AI coding agent joining the project to maintain `PROJECT_MEMORY.md`.
- `AGENTS_TOOLS_GOVERNANCE.md` — Master catalog of 21 unique tools, cross-agent consumption matrix, 8 governance laws for modular decoupled tools, and 5-developer tool packages.
- `AGENTS_ARCHITECTURE_PLAN.md` — Detailed architecture plan for the 5 agents, customer identity injection, inter-agent workflows, schemas, and 5 parallel developer task assignments.
- `BACKEND_PROJECT_PLAN.md` — Comprehensive backend engineering roadmap, data model, API contract, and 5-developer parallel work split.
- `PROJECT_MEMORY.md` — Persistent handoff memory across AI coding agents.
- `SAHM-AI-backend/app/main.py` — FastAPI application entry point, lifespan, error handlers, and router wiring.
- `SAHM-AI-backend/app/config.py` — Central environment configuration settings and LLM/provider routing.
- `SAHM-AI-backend/app/api/` — API routers for assistant, campaigns, creatives, personas, tasks, etc.
- `SAHM-AI-backend/app/core/agents/` — Persona catalog (`personas.py`), runtime execution loop (`runtime.py`), and task specs (`tasks.py`).
- `SAHM-AI-backend/app/core/orchestrator/` — LangGraph state machine (`graph.py`), intent classifier (`router.py`), and block composer (`blocks.py`).
- `SAHM-AI-backend/app/core/memory/` — Vector embedding provider and BrandMemoryStore (`store.py`).
- `SAHM-AI-backend/app/core/tools/` — Tool base class (`base.py`), registry (`registry.py`), research (`research.py`), media (`media.py`), memory (`memory.py`), and protected actions (`protected.py`).
- `SAHM-AI-backend/app/db/` — SQLAlchemy models (`models.py`) and async session management (`session.py`).
- `SAHM-AI-backend/tests/` — Pytest test suite.
- `SAHM-AI-backend/pyproject.toml` — Backend project dependencies and test configuration.
- `SAHM-AI-frontend/` — Next.js App Router frontend codebase.

## 4. Completed Work

### Task: Author Agent Memory Instructions & Protocol Guide
- Date/time: 2026-10-01
- Changed:
  - `AGENT_MEMORY_INSTRUCTIONS.md` — Created standard operating prompt and onboarding guide for any AI agent working on the project. Details the `CURRENT STATE → TASK → RELEVANT FILES → DEPENDENCIES → EXPECTED RESULT` lifecycle, focused update template, rules, anti-patterns, and concrete good vs bad examples.
- Result:
  - Any future AI agent can be given this file to maintain persistent, high-quality memory updates in `PROJECT_MEMORY.md`.

### Task: Author Agent Tools Architecture & Governance Guide
- Date/time: 2026-10-01
- Changed:
  - `AGENTS_TOOLS_GOVERNANCE.md` — Cataloged all 21 unique tools, created the cross-agent matrix distinguishing shared tools (`get_brand_context`, `save_learning`, `web_search`) from domain-specific tools, formulated the 8 Governance Laws for decoupled modular code, and divided tools into 5 parallel developer packages.
- Result:
  - Clear architectural rules and tool work packages ready for 5 developers to implement in parallel without blocking each other.

### Task: Author 5-Agent Architecture & Parallel Plan
- Date/time: 2026-10-01
- Changed:
  - `AGENTS_ARCHITECTURE_PLAN.md` — Created full architectural specification for the 5 agents (Nour, Omar, Layla, Karim, Salma). Documented shared `OrchestratorState` blackboard, Customer Identity (`BrandContext`) injection protocol, 3 Mermaid diagrams (Topology, Content Generation with Identity, Closed-Loop Optimization), and 5 discrete parallel developer assignments.
- Result:
  - Complete, visual, and actionable multi-agent blueprint ready for 5 developers to implement in parallel.

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
- Customer Identity Injection: Agents never make isolated guesses; consumer-facing agents (Layla/Omar) call `get_brand_context` to inject `BrandProfile` + `Audience` + retrieved vector memories (capped at 900 tokens) with strict `do_not_use` word filtering.
- Blackboard Pattern: Agents communicate via LangGraph `OrchestratorState` and Action Cards rather than unconstrained chat loops.
- Tool Decoupling: Tools are pure functions receiving all dependencies via `ToolContext` (no globals); tools are agent-agnostic and do not branch on caller identity. External-write tools are protected and never registered for direct agent execution.
- Memory Governance: Every AI agent session must read and update `PROJECT_MEMORY.md` directly using the protocol in `AGENT_MEMORY_INSTRUCTIONS.md`.

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
Created [`AGENT_MEMORY_INSTRUCTIONS.md`](file:///d:/NTI-Final-Project/AGENT_MEMORY_INSTRUCTIONS.md) establishing the universal prompt guide and operational lifecycle for any AI coding agent joining this project to maintain `PROJECT_MEMORY.md`.

### Current Blocker
Backend pytest suite fails to load due to missing `greenlet` dependency required by `sqlalchemy.ext.asyncio` under Python 3.13.

### Next Recommended Step
Add `greenlet` to `SAHM-AI-backend/pyproject.toml`, run `uv sync --extra dev`, and run `uv run --extra dev pytest` to verify backend tests pass.

### Expected Result
Pytest executes and passes cleanly, unblocking Developer 1 through Developer 5 to begin tool and agent development.

## 11. Agent Notes

- To onboard a new AI session, pass the prompt inside `AGENT_MEMORY_INSTRUCTIONS.md`.
- Incoming agents must inspect `PROJECT_MEMORY.md` first and continue from Section 10 ("Current Continuation Point").
- Every completed task must update `PROJECT_MEMORY.md` immediately without waiting for a user handoff request.