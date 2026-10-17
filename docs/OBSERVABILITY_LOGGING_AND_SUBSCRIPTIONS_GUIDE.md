# Observability, Diagnostic Logging & Subscriptions Guide

This document details the diagnostic logging, multi-platform observability (LangSmith & Langfuse), real-time cost estimation, and the subscription management system implemented in Sahm AI. It also serves as an **Agent Integration Playbook** instructing AI coding agents and engineers on how to leverage and extend these capabilities in future tasks.

---

## 1. Executive Summary: What Was Done

| Area | Component | Description |
|---|---|---|
| **LLM Diagnostic Logger** | `app/ai/loggers/callbacks.py`<br>`app/ai/loggers/cost.py`<br>`logs/providers/llm.log` | Records every LLM start/finish/error with prompt previews, token counts (`in`, `out`, `total`), latency (`ms`), USD cost estimation, and automated failure diagnosis. |
| **Agent Diagnostic Logger** | `app/ai/loggers/agents/*.py`<br>`logs/agents/diagnostic.log` | Tracks LangGraph node transitions (`agent_step_started`, `agent_step_finished`, `agent_step_failed`), specialist handoffs (Omar, Layla, Karim, Nour, Salma, Orchestrator), and structured error diagnosis. |
| **Observability Router** | `app/ai/observability.py` | Unified callback assembler that connects local diagnostic file logs, LangSmith traces, and Langfuse monitoring with graceful offline fallback. |
| **Cost Estimator** | `app/ai/loggers/cost.py` | Real-time token pricing table across OpenAI, Anthropic, Gemini, Groq, DeepSeek, and `$0.00` for Ollama and mock. |
| **Subscription System** | `app/core/subscriptions.py`<br>`app/schemas/subscriptions.py`<br>`app/api/subscriptions.py` | 100% free, professional tier management (`free`, `pro`, `enterprise`) that controls daily token budgets without requiring paid gateways or Stripe. |
| **Configuration Surface** | `app/config.py`<br>`app/ai/settings.py`<br>`.env.example` | Dual-settings synchronization (Invariant §5.11) with sensitive key redaction for LangSmith and Langfuse credentials. |

---

## 2. File & Architecture Map

```
SAHM-AI-backend/
├── app/
│   ├── ai/
│   │   ├── loggers/
│   │   │   ├── base.py                   # ComponentLogger factory (logs/<kind>s/<name>.log)
│   │   │   ├── callbacks.py              # LangChain AILoggingCallback with LLM & Agent hooks
│   │   │   ├── cost.py                   # Real-time token cost computation & pricing table
│   │   │   └── agents/                   # Dedicated specialist loggers
│   │   │       ├── omar.py               # logs/agents/omar.log
│   │   │       ├── layla.py              # logs/agents/layla.log
│   │   │       ├── karim.py              # logs/agents/karim.log
│   │   │       ├── nour.py               # logs/agents/nour.log
│   │   │       ├── salma.py              # logs/agents/salma.log
│   │   │       ├── orchestrator.py       # logs/agents/orchestrator.log
│   │   │       └── diagnostic.py         # logs/agents/diagnostic.log & log_agent_diagnosis()
│   │   ├── observability.py              # LangSmith & Langfuse unified callback router
│   │   └── agents/orchestrator/brain.py  # Orchestrator wired to get_observability_callbacks()
│   ├── core/
│   │   ├── subscriptions.py              # Plans catalog (Free, Pro, Enterprise) & budget switcher
│   │   └── logging.py                    # _SENSITIVE_KEYS redaction for observability tokens
│   ├── schemas/
│   │   └── subscriptions.py              # Pydantic v2 ApiModel schemas for plans & subscriptions
│   └── api/
│       └── subscriptions.py              # /api/v1/subscriptions HTTP endpoints
└── tests/
    ├── test_llm_diagnostic_logger.py     # LLM token, cost, and diagnostic classification tests
    ├── test_agent_diagnostic_logger.py   # Agent lifecycle, step, and error diagnosis tests
    ├── test_observability.py             # LangSmith & Langfuse safe detection tests
    └── test_subscriptions.py             # Plans, budget update, and API route tests
```

---

## 3. Agent Integration Playbook: How an Agent Should Work With This

If you are an AI coding agent or an engineer working on a feature in this repository, follow these standard patterns to integrate observability, logging, and subscriptions into your work:

### Rule 1: Attaching Observability to Graphs, Chains, or Tool Invocations
Never instantiate raw, unconfigured callbacks. Always assemble callbacks using `get_observability_callbacks()`:

```python
from app.ai.observability import get_observability_callbacks

# 1. Inside an agent graph invocation:
callbacks = get_observability_callbacks(agent="omar", audit=audit_callback)
result = await graph.ainvoke(graph_input, config={"callbacks": callbacks})

# 2. Inside a standalone chain or router:
callbacks = get_observability_callbacks(agent="orchestrator")
decision = await router.ainvoke(messages, config={"callbacks": callbacks})
```

**Why:** This automatically guarantees that:
1. Local detailed diagnostic logs are written to `logs/providers/llm.log` and `logs/agents/<agent>.log`.
2. Token costs are computed and recorded locally.
3. If the user configured LangSmith or Langfuse in `.env`, traces and spans are streamed to their dashboard.
4. If third-party services are offline or unconfigured, the call degrades cleanly without breaking.

---

### Rule 2: Recording Agent Step Diagnostics & Troubleshooting Events
When implementing complex decision trees, fallback branches, or handoffs between agents, log diagnostic records to `logs/agents/diagnostic.log`:

```python
from app.ai.loggers.agents.diagnostic import log_agent_diagnosis

# When an agent takes a non-obvious branch:
log_agent_diagnosis(
    event="agent_branch_selected",
    agent="layla",
    node="copy_generation",
    status="ok",
    detail="Branching to Arabic colloquial Saudi dialect for local Ramadan campaign.",
)

# When an agent recovers from a failure or tool error:
log_agent_diagnosis(
    event="fallback_triggered",
    agent="nour",
    node="competitor_ad_scraper",
    status="fallback",
    detail="Competitor Ad Library timed out; loaded cached competitor findings from BrandMemory.",
)

# When an unhandled error occurs:
try:
    ...
except Exception as exc:
    log_agent_diagnosis(
        event="step_execution_failed",
        agent="salma",
        node="bid_optimizer",
        status="error",
        error=exc,
        remediation="Check Meta Graph API connector token permissions.",
    )
```

---

### Rule 3: Estimating Token Usage & Costs
To display or log estimated USD cost for any LLM interaction:

```python
from app.ai.loggers.cost import estimate_token_cost, format_cost

# 1. Compute numerical cost:
cost_usd = estimate_token_cost(model="gpt-4o-mini", input_tokens=1500, output_tokens=350)
# Returns float: 0.000435

# 2. Format for UI or logs:
cost_label = format_cost(cost_usd)
# Returns str: "$0.000435" (or "Free ($0.00)" for Ollama / mock)
```

---

### Rule 4: Querying & Managing Workspace Subscriptions
To check a workspace's current tier, token budget, and consumption:

```python
from app.core.subscriptions import get_workspace_subscription, switch_workspace_plan
from app.schemas.subscriptions import PlanId

# 1. Read current status:
status = await get_workspace_subscription(db, workspace_id="ws_123")
print(status.current_plan.name)        # "Pro Marketing OS"
print(status.daily_token_budget)       # 1000000
print(status.tokens_used_today)        # 42150
print(status.percent_consumed)         # 4.22%

# 2. Switch tier (100% free and instantaneous):
response = await switch_workspace_plan(db, workspace_id="ws_123", new_plan_id=PlanId.ENTERPRISE)
# Updates business.daily_token_budget = 10,000,000 in DB
```

---

## 4. Free Tiers Setup Guide (LangSmith & Langfuse)

Both services provide generous, **100% free tiers** that do not require entering credit card details for personal projects and SMB operations:

### Option A: LangSmith (Free Developer Plan)
* **What you get:** 5,000 free traces / month forever, project tracing, latency breakdown, prompt Playground.
* **Setup steps:**
  1. Visit [smith.langchain.com](https://smith.langchain.com) and sign up with GitHub or Google.
  2. Navigate to **Settings** $\to$ **API Keys** $\to$ **Create API Key**.
  3. Add to `SAHM-AI-backend/.env`:
     ```env
     LANGCHAIN_TRACING_V2=true
     LANGCHAIN_API_KEY=lsv2_pt_...
     LANGCHAIN_PROJECT=sahm-ai
     ```

### Option B: Langfuse (Free Cloud Plan or Free Self-Hosted)
* **What you get:**
  * **Cloud:** 50,000 free events / month forever, token cost dashboards, trace waterfall view.
  * **Self-Hosted:** 100% free unlimited forever on your own server or Docker.
* **Cloud Setup steps:**
  1. Visit [cloud.langfuse.com](https://cloud.langfuse.com) and create a free account.
  2. Create a new project named `Sahm Marketing OS`.
  3. Go to **Settings** $\to$ **API Keys** $\to$ **Create new API credentials**.
  4. Add to `SAHM-AI-backend/.env`:
     ```env
     LANGFUSE_PUBLIC_KEY=pk-lf-...
     LANGFUSE_SECRET_KEY=sk-lf-...
     LANGFUSE_HOST=https://cloud.langfuse.com
     ```
* **Self-Hosted Docker Setup (Optional, Unlimited Free):**
  ```bash
  docker run --name langfuse -d -p 3000:3000 \
    -e DATABASE_URL="postgresql://postgres:postgres@localhost:5432/langfuse" \
    -e NEXTAUTH_SECRET="mysecret" \
    -e NEXTAUTH_URL="http://localhost:3000" \
    langfuse/langfuse:latest
  ```
  Then set `LANGFUSE_HOST=http://localhost:3000` in `.env`.

---

## 5. Subscription HTTP API Reference

All routes are mounted under `/api/v1/subscriptions`:

| Method | Endpoint | Description | Sample Response / Headers |
|---|---|---|---|
| `GET` | `/subscriptions/plans` | Catalog of all free tiers (`free`, `pro`, `enterprise`) | `[{"id": "free", "name": "Community Free", "dailyTokenBudget": 100000, "isFree": true, ...}]` |
| `GET` | `/subscriptions/current` | Active plan, daily budget, usage today, and percentage consumed | `{"currentPlan": {...}, "dailyTokenBudget": 1000000, "tokensUsedToday": 15000, "percentConsumed": 1.5}` |
| `POST` | `/subscriptions/switch` | Switch tier immediately | Body: `{"planId": "pro"}` $\to$ Updates budget to `1,000,000` |
| `GET` | `/subscriptions/observability` | Live status of LangSmith, Langfuse, and local loggers | `{"langsmith": {"enabled": true}, "langfuse": {"enabled": true}, "cost_estimator": {"status": "active"}}` |

---

## 6. Invariants Preserved (§5 Compliance)

1. **Dual Settings Mirroring (Invariant §5.11):** Both `app/config.py` and `app/ai/settings.py` maintain identical definitions for all observability settings.
2. **Tenancy Boundary (Invariant §5.1):** All subscription queries resolve workspace and business context via `WorkspaceDep`, never from unverified request payloads.
3. **External Write Safety (Invariant §5.2):** No agent directly executes unapproved external mutations; subscription switching is scoped to database token budget quotas.
4. **Credential Redaction (Invariant §5.9):** `_SENSITIVE_KEYS` in `app/core/logging.py` masks `langchain_api_key`, `langsmith_api_key`, `langfuse_public_key`, and `langfuse_secret_key` so secrets can never leak into trace logs.
