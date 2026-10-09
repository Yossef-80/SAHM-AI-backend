## File: ARCHITECTURE_OVERVIEW.md
- **Purpose:** Configuration/Documentation file. (60 lines)
- **Content Summary:**
  > # SAHM Marketing OS - Architecture Overview
  > 
  > ## The Great Migration
  > This document serves as the absolute truth for the current backend architecture of the SAHM Marketing OS. 
  > 
  > Originally, the backend relied on a highly complex, custom-built AI graph routing engine located in `app/core/orchestrator/`. This custom engine became deeply entangled, suffered from context window issues, failed on local LLMs, and was notoriously difficult to debug.
  > 
  > We executed a massive refactor to bypass this legacy engine and adopt a standard, robust **LangChain & LangGraph** architecture, while strictly preserving the React frontend's JSON UI requirements.
  > 
  > ---
  > ...


## File: app\config.py
- **Purpose:** Application configuration.
- **Dependencies/Imports:** pydantic, functools, pydantic_settings, typing, json, __future__
- **Key Classes:** Settings
- **Key Functions:** get_settings, _split_csv, _parse_extra_headers, provider_for_tier, model_for_tier, is_mock_llm, has_real_llm_key
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 5 instances (e.g. Line 83: # SQLAlchemy uses; a Supabase connection string needs the database password)


## File: GIT_WORKFLOW_GUIDE.md
- **Purpose:** Configuration/Documentation file. (90 lines)
- **Content Summary:**
  > # Git Workflow & Best Practices for Tool Developers
  > 
  > Welcome! All new tool development happens under the `tools-development` branch (or feature branches branching off from it). 
  > 
  > Follow this guide to easily access the branch, write your tools, and submit your code without breaking the main backend.
  > 
  > ---
  > 
  > ## 1. How to Access the Branch
  > 
  > ...


## File: app\ai\enums.py
- **Purpose:** Every selectable option in the AI layer, as an Enum.
- **Dependencies/Imports:** enum, __future__
- **Key Classes:** LLMProvider, SearchProvider, DatabaseProvider, StructuredOutputMethod, SearchDepth, LogKind, LogLevel, LogFormat, AgentName, ToolName, LearningCategory
- **Key Functions:** 
- **Status:** Fully implemented


## File: app\main.py
- **Purpose:** Sahm backend: FastAPI app, router registration, middleware, error handling.
- **Dependencies/Imports:** typing, app.core.llm.router, fastapi.middleware.cors, app.core.security, contextlib, app.core.errors, uuid, app.core.security_headers, fastapi.responses, starlette.exceptions, app.core.tools.registry, app.core.ratelimit, fastapi.exceptions, app.api, sqlalchemy, time, app.config, app.schemas.common, app.core.logging, fastapi, app.db.session, __future__
- **Key Classes:** 
- **Key Functions:** _http_error_code, _jsonable_errors, _check_database, lifespan, create_app, add_request_id, envelope, sahm_error_handler, validation_error_handler, http_exception_handler, unhandled_error_handler, health, readiness, root
- **Status:** Fully implemented


## File: TOOLS_ASSIGNMENT.md
- **Purpose:** Configuration/Documentation file. (60 lines)
- **Content Summary:**
  > # Tools Development Assignment Matrix
  > 
  > This document outlines the high-value AI tools we need to build for the SAHM OS. Developers should claim a tool, follow the `DEVELOPER_TOOL_GUIDE.md`, and submit their PRs.
  > 
  > ---
  > 
  > ## 🟢 FREE TOOLS (No Paid APIs Required)
  > 
  > ### 1. Facebook/Instagram Organic Poster
  > * **Description:** A tool that takes a finished creative and caption from the AI and publishes it directly to a brand's Facebook or Instagram page.
  > ...


## File: .env
- **Purpose:** Configuration/Documentation file. (122 lines)
- **Env Keys:** SAHM_ENVIRONMENT, LOG_LEVEL, LOG_JSON, API_PREFIX, APP_NAME, CORS_ORIGINS, SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, SUPABASE_SECRET_KEY, SUPABASE_JWKS_URL, DATABASE_URL, AI_LLM_PROVIDER, GROQ_MODEL, GROQ_API_KEY, OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SECONDS, GEMINI_MODEL, GEMINI_API_KEY, OPENAI_MODEL, OPENAI_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_API_KEY, OPENROUTER_MODEL, OPENROUTER_API_KEY, AI_SEARCH_PROVIDER, TAVILY_API_KEY, DAILY_TOKEN_BUDGET, WEBSITE_ALLOWED_SCHEMES, WEBSITE_FETCH_TIMEOUT_SECONDS, WEBSITE_MAX_BYTES, AI_SQLITE_PATH, AI_DB_PROVIDER, META_GRAPH_BASE_URL, META_GRAPH_VERSION, META_API_TIMEOUT_SECONDS, META_APP_ID, META_APP_SECRET, META_AD_ACCOUNT_ID, TOKEN_ENCRYPTION_KEY, UPLOAD_DIR, UPLOAD_MAX_BYTES, UPLOAD_ALLOWED_TYPES, GOOGLE_PLACES_API_KEY, DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD, APIFY_TOKEN, AI_LOG_DIR, AI_LOG_LEVEL, AI_LOG_FORMAT, AI_LOG_TO_CONSOLE, AI_LLM_MAX_TOKENS


## File: app\db\models.py
- **Purpose:** SQLAlchemy 2 models.
- **Dependencies/Imports:** uuid, app.schemas.common, sqlalchemy.orm, datetime, sqlalchemy.dialects.postgresql, sqlalchemy.types, math, typing, pgvector.sqlalchemy, json, sqlalchemy, app.config, __future__
- **Key Classes:** Base, Embedding, TimestampMixin, Business, BrandProfile, BrandMemoryItem, Conversation, Message, Campaign, Creative, Insight, Recommendation, AuditLog, PersonaOverride, AgentTask, AgentToolCall, Integration, User, Membership, Session, Profile, ToolSettings, LlmProviderSettings, UsageRecord, comparator_factory
- **Key Functions:** cosine, __init__, load_dialect_impl, process_bind_param, process_result_value, cosine_distance, l2_distance, max_inner_product
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 5 instances (e.g. Line 61: pass)


## File: DEVELOPER_TOOL_GUIDE.md
- **Purpose:** Configuration/Documentation file. (88 lines)
- **Content Summary:**
  > # Developer Guide: Building AI Tools for SAHM OS
  > 
  > Welcome to the SAHM AI Tool Development team! We recently upgraded our entire AI engine to a modern **LangChain & LangGraph** architecture. 
  > 
  > This guide will explain exactly how you should build tools, how they interact with our database, how the UI displays them, and how to submit your work so it can be merged and tested instantly.
  > 
  > ---
  > 
  > ## 1. The Architecture (What you need to know)
  > * **The Agents:** Our AI agents (Nour, Omar, Layla, etc.) are built using LangGraph. They are located in `app/ai/agents/`.
  > ...


## File: DATABASE_GUIDE.md
- **Purpose:** Configuration/Documentation file. (91 lines)
- **Content Summary:**
  > # SAHM Marketing OS - Database & Tooling Guide
  > 
  > This guide explains the backend database schema (powered by SQLAlchemy + Postgres/SQLite) and how you should design AI tools to seamlessly save their outputs into these tables using Pydantic schemas.
  > 
  > ---
  > 
  > ## 1. Database Table Breakdown
  > 
  > The database is built with multi-tenancy in mind, meaning almost every table relates back to a `business_id` or `workspace_id`. Here are the core tables:
  > 
  > ...
