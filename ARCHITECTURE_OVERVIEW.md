# SAHM Marketing OS - Architecture Overview

## The Great Migration
This document serves as the absolute truth for the current backend architecture of the SAHM Marketing OS. 

Originally, the backend relied on a highly complex, custom-built AI graph routing engine located in `app/core/orchestrator/`. This custom engine became deeply entangled, suffered from context window issues, failed on local LLMs, and was notoriously difficult to debug.

We executed a massive refactor to bypass this legacy engine and adopt a standard, robust **LangChain & LangGraph** architecture, while strictly preserving the React frontend's JSON UI requirements.

---

## 1. The Core Architecture (Hybrid Approach)

The architecture is now a hybrid of the original API layer and a brand new LangGraph AI engine.

### A. The API Layer (`app/api/` & `app/core/`)
The FastAPI endpoints and Database interactions (Supabase + AsyncPG) remain untouched. 
When a user submits a prompt via the frontend (whether from a chat or a Research Tool tab), it hits `POST /api/v1/assistant/message/stream`. 
The request is processed by `app/core/orchestrator/service.py`.

### B. The AI Brain (`app/ai/agents/orchestrator/brain.py`)
Instead of using the legacy core orchestrator, `service.py` intercepts the chat history and the user's prompt and routes it directly to our new LangGraph Orchestrator (`brain.py`).

1. **Routing:** The Orchestrator LLM reads the context and routes the prompt to the appropriate agent.
2. **Task Agents (Nour, Omar, etc.):** If the user asks for new research or a heavy task, it is routed to a LangGraph compiled state graph (e.g., `nour_graph`). These agents run in loops, utilizing tools (like Web Search or Website Scrapers) until their task is complete.
3. **The Chat Agent (`chat`):** If the user asks a simple follow-up question or for clarification, the Orchestrator instantly routes it to a fast, tool-free conversational agent. This prevents the heavy Task Agents from getting confused and re-running expensive searches.

### C. The UI Schema Bridge
The React frontend strictly requires responses to be formatted in a custom JSON schema of "UI Blocks" (`text`, `findings`, `action_card`). 
To achieve this without breaking the frontend, the Orchestrator contains a "Bridge":
- Once the LangGraph agents finish generating their raw text or JSON reports, the Orchestrator catches it.
- It passes this raw data to a specialized LLM structurer.
- The structurer maps the raw data perfectly into the required React UI Blocks and streams it to the user.

---

## 2. Solved Issues & Bug Fixes

During the migration, we solved several critical bugs that were crashing the backend:

1. **Pydantic Enum Crash:** Fixed a fatal serialization bug where Pydantic Enums in the UI blocks were causing the FastAPI stream to crash. We implemented `mode="json", by_alias=True` to safely dump the blocks.
2. **LangGraph Postgres Crash:** The `langgraph-checkpoint-postgres` library was crashing on Windows due to missing binaries and an unconfigured `AI_POSTGRES_URL`. We safely moved the graph checkpointing to `MemorySaver` in `.env`.
3. **Groq/Gemini Rate Limits (HTTP 429):** The old engine had zero fallbacks if an API rate limit was hit, causing the UI to say "The assistant could not complete this request." We wrapped the UI Structurer in a bulletproof `try/except` block that safely falls back to standard text blocks.
4. **Tool Call Extraction Bug:** Task agents like Nour finish their work by calling a tool (`submit_research_report`). The Orchestrator was mistakenly grabbing the literal string "REPORT_SUBMITTED_SUCCESSFULLY" instead of the massive report generated inside the tool arguments. This was fixed by implementing deep argument extraction in `brain.py`.
5. **Memory & Context Confusion:** Small local models (like Ollama 8B) were getting confused by the chat history and attempting to answer old historical questions alongside new ones. We fixed this by introducing the dedicated `chat` agent and explicitly wrapping prompts with context boundaries: *"Context is provided above. CRITICAL: Do NOT re-answer previous questions. Now, answer this NEW question:"*
6. **Dead Code Purge:** Safely removed thousands of lines of unused code, including `app/core/orchestrator/graph.py`, `app/core/orchestrator/state.py`, and `app/core/orchestrator/router.py`, cleaning the repository.

---

## 3. Configuration & Timeouts

The `app/ai/settings.py` file dictates all timeouts to ensure slow local models (like Ollama) are not prematurely terminated.

- `OLLAMA_TIMEOUT_SECONDS`: 1000s (16+ minutes)
- `AI_LLM_TIMEOUT_SECONDS`: 180s (3 minutes)
- `AI_SEARCH_TIMEOUT_SECONDS`: 20s
- `WEBSITE_FETCH_TIMEOUT_SECONDS`: 15s

To switch models, simply update `AI_LLM_PROVIDER` in `.env` to `ollama`, `groq`, or `gemini`.
