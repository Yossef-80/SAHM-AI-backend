# Database guide

> **Audience:** Backend engineers adding a table, a query or a migration.
> **Purpose:** The 23 tables, what owns them, the conventions they follow, and how
> to change the schema safely.

Sahm runs on PostgreSQL in deployment and on SQLite for local development and
tests: SQLAlchemy 2 (async) + Alembic, with one model file (`app/db/models.py`)
and one migration history (`app/db/migrations/versions/`). Any schema change is
two things — an edit to the model **and** a migration — never one.

---

## 1. Database Table Breakdown

The database is built with multi-tenancy in mind, meaning almost every table relates back to a `business_id` or `workspace_id`. Here are the core tables:

### A. Core Architecture & Multi-Tenancy
* **`User` & `Membership`**: Handles authentication and RBAC. A user belongs to a workspace with a specific role (`owner`, `member`, `viewer`).
* **`Business`**: The core entity representing a brand. Contains budget, currency, industry, etc.
* **`BrandProfile`**: Tied 1:1 with `Business`. Contains stable brand fields like `brand_voice`, `visual_style`, `target_audience`, `goals`, and `competitors`.
* **`Integration`**: Stores encrypted tokens (e.g., Meta/Facebook ads API keys) for a business.

### B. Chat & AI Memory
* **`Conversation` & `Message`**: Standard chat history. Messages contain both raw `content` (text) and `blocks` (the strict JSON UI blocks consumed by React).
* **`BrandMemoryItem`**: The long-term brand memory -- **not** a vector table. There is no `embedding` column and no RAG: rows are retrieved with a deterministic lexical + recency + kind-affinity ranker (`app/core/memory/`). It stores the 13 kinds in `app/core/memory/kinds.py` ("learnings", "voice examples", "strategy_blueprint", research artifacts, chat attachments ...) together with the lifecycle columns that make retention work: `expires_at`, `importance`, `pinned`, `dedupe_digest`, `archived_at`. See `docs/MEMORY_ARCHITECTURE.md`.

### C. Marketing Entities (The Heavy Lifters)
* **`Campaign`**: A marketing campaign with dates, budgets, and objectives.
* **`Creative`**: An asset (image/video/copy) tied to a campaign. Contains a `url` and the AI `prompt` used to generate it.
* **`Insight`**: Data discoveries (e.g., "Competitor X is running 50% off sales").
* **`Recommendation`**: AI-suggested actions (e.g., "Increase budget by $500"). Contains an approval flow (`requires_approval`, `approved_by`, `status`).

### D. Content, Audiences & Configuration
* **`ContentItem`** (`content_items`): one calendar post — date, time, channel, format, hook, caption, hashtags, status, pillar.
* **`SavedAudience`** (`saved_audiences`): a reusable audience definition per workspace.
* **`ToolSettings`** / **`LlmProviderSettings`**: per-workspace connector credentials (encrypted) and fast/strong model routing.
* **`UsageRecord`**: one row per model call — tokens in/out, model, task.

### E. System Logging & Tasks
* **`AgentTask`**: Long-running background processes.
* **`AgentToolCall`**: A log of every single tool an agent uses (e.g., when Nour searches the web, it gets logged here with `input`, `output_summary`, and `duration_ms`).
* **`AuditLog`**: Tracks when a human approves or rejects a Recommendation.
* **`PersonaOverride`**: Custom modifications a user makes to an agent's tone or autonomy.

---

## 2. How to Build Tools that Save to the Database

When you design a new tool for an agent (like Nour or Omar), you should **never** have the agent just write raw text if the goal is to save structured data (like a Campaign or an Insight). 

Instead, use **Pydantic** to define the tool's input schema, and use **SQLAlchemy** inside the tool to save the data.

### Example: Building a "Create Insight" Tool

**Step 1: Define the Pydantic Schema (The Tool Input)**
This tells the AI exactly what fields to generate.
```python
from pydantic import BaseModel, Field

class CreateInsightInput(BaseModel):
    title: str = Field(description="A short, punchy title for the insight")
    detail: str = Field(description="Detailed explanation of the data found")
    severity: str = Field(description="One of: 'info', 'warning', 'critical'")
    metrics: dict = Field(description="JSON dictionary of relevant numbers (e.g., {'roi': 1.5})")
```

**Step 2: Create the Tool Function**
Use LangChain's `@tool` decorator. The tool receives the Pydantic model and injects it into the SQLAlchemy database.
```python
from langchain_core.tools import tool
from app.db.models import Insight
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

@tool(args_schema=CreateInsightInput)
async def create_insight(title: str, detail: str, severity: str, metrics: dict, db: AsyncSession, campaign_id: str):
    \"\"\"Saves a new market insight to the database.\"\"\"
    
    # 1. Create the SQLAlchemy Model
    new_insight = Insight(
        id=f"ins_{uuid.uuid4().hex[:12]}",
        campaign_id=campaign_id,
        kind="market_trend",
        title=title,
        detail=detail,
        severity=severity,
        metrics=metrics
    )
    
    # 2. Save to DB
    db.add(new_insight)
    await db.commit()
    
    # 3. Return success to the Agent
    return f"Successfully saved insight: {title}"
```

### 3. Formatting Output for the Frontend (UI Blocks)

If the agent is just chatting with the user (not saving to a database table), the frontend expects strict **React UI Blocks**.

You don't need to force the agent to write JSON manually! Thanks to the new architecture in `app/ai/agents/orchestrator/brain.py`, the AI agents just write plain markdown text, and the **Orchestrator's UI Bridge** automatically uses Pydantic (`MessageWithBlocks`) to instantly convert their text into the required frontend JSON blocks before saving it to the `Message` table.
