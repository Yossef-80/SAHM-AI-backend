## File: app/ai/providers/llm/openai.py
- **Purpose:** OpenAI provider (also any OpenAI-compatible server via ``OPENAI_BASE_URL``).
- **Dependencies/Imports:** langchain_openai, __future__, langchain_core.language_models, app.ai.enums, app.ai.settings, app.ai.errors
- **Key Classes:** 
- **Key Functions:** build_chat_model
- **Status:** Fully implemented


## File: GIT_WORKFLOW_GUIDE.md
- **Purpose:** Configuration/Documentation file. (90 lines)
- **Content Summary:**
  > # Git Workflow & Best Practices for Tool Developers
  > 
  > Welcome! All new tool development happens under the `tools-development` branch (or feature branches branching off from it). 
  > 
  > Follow this guide to easily access the branch, write your tools, and submit your code without breaking the main backend.
  > ...


## File: app/core/logging.py
- **Purpose:** Structured logging with request_id on every line.
- **Dependencies/Imports:** __future__, typing, logging, json, time, sys, contextvars
- **Key Classes:** ContextFilter, JsonFormatter, _LoggerAdapter
- **Key Functions:** bind_request_id, current_request_id, bind_workspace, current_workspace_id, current_business_id, redact, configure_logging, get_logger, filter, format, process
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 9 instances (e.g. Line 90: #: the whole key, so ``password)


## File: app/core/orchestrator/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** 
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/tools/meta_read.py
- **Purpose:** meta_read tools (9-10). READ-ONLY against the Meta Marketing API.
- **Dependencies/Imports:** app.schemas.insights, __future__, app.schemas.common, typing, httpx, sqlalchemy, json, app.core.security, app.core.tools.settings, app.core.tools.base, app.core.errors, datetime, app.core.logging, app.schemas.tools, app.db.models
- **Key Classes:** 
- **Key Functions:** _resolve_meta_campaign_id, _get_meta_token, _meta_config, _default_range, _handle_get_campaign_insights, _handle_get_breakdowns, _is_num, json_dumps, join_metrics, _now
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 102: # where a caller passed no wor)


## File: app/ai/memory/compressor.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langchain_core.messages, typing, sqlalchemy, app.db.models, app.ai.providers.llm, app.ai.enums, sqlalchemy.ext.asyncio
- **Key Classes:** IntelligentMemoryCompressor
- **Key Functions:** __init__, compress_latest_turn, update_global_profile
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 23: pass)


## File: app/schemas/campaigns.py
- **Purpose:** Schemas for campaign endpoints.
- **Dependencies/Imports:** pydantic, app.schemas.common, __future__, datetime
- **Key Classes:** CampaignCreateRequest, CampaignUpdateRequest, Campaign, CampaignListResponse, CampaignDetailResponse
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/db/models.py
- **Purpose:** SQLAlchemy 2 models.
- **Dependencies/Imports:** uuid, __future__, app.schemas.common, typing, sqlalchemy, pgvector.sqlalchemy, json, math, app.config, sqlalchemy.orm, datetime, sqlalchemy.types, sqlalchemy.dialects.postgresql
- **Key Classes:** Base, Embedding, TimestampMixin, Business, BrandProfile, BrandMemoryItem, Conversation, Message, Campaign, Creative, Insight, Recommendation, AuditLog, PersonaOverride, AgentTask, AgentToolCall, Integration, User, Membership, Session, Profile, ToolSettings, LlmProviderSettings, UsageRecord, comparator_factory
- **Key Functions:** cosine, __init__, load_dialect_impl, process_bind_param, process_result_value, cosine_distance, l2_distance, max_inner_product
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 5 instances (e.g. Line 61: pass)


## File: app/ai/agents/nour/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** app.ai.agents.nour.graph
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/auth.py
- **Purpose:** Identity: password hashing, opaque sessions, and the authenticated principal.
- **Dependencies/Imports:** base64, uuid, hmac, secrets, hashlib, __future__, dataclasses, sqlalchemy, app.db.models, datetime, app.core.logging, sqlalchemy.ext.asyncio
- **Key Classes:** MembershipGrant, Principal
- **Key Functions:** _now, new_user_id, new_membership_id, new_session_id, normalise_email, _derive_key, hash_password, verify_password, needs_rehash, hash_token, new_session_token, create_session, resolve_session, revoke_session, revoke_all_for_user, load_principal, _as_utc, membership_for_workspace, membership_for_business, primary_workspace, role_for, has_role
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 21 instances (e.g. Line 1: """Identity: password hashing, o)


## File: app/ai/tools/brand_context.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langchain_core.runnables.config, typing, langchain_core.tools, app.ai.memory.brand_memory, app.ai.providers.db
- **Key Classes:** 
- **Key Functions:** get_brand_context_tool
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 18: # Extract business_id from the )


## File: app/schemas/integrations.py
- **Purpose:** Schemas for integration status/connect endpoints.
- **Dependencies/Imports:** pydantic, app.schemas.common, __future__, datetime
- **Key Classes:** MetaIntegrationStatus, MetaConnectRequest, MetaConnectResponse, IntegrationInfo
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/ai/providers/llm/groq.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** pydantic, langchain_core.language_models, app.ai.enums, app.ai.settings, langchain_groq
- **Key Classes:** 
- **Key Functions:** build_chat_model
- **Status:** Fully implemented


## File: ARCHITECTURE_OVERVIEW.md
- **Purpose:** Configuration/Documentation file. (60 lines)
- **Content Summary:**
  > # SAHM Marketing OS - Architecture Overview
  > 
  > ## The Great Migration
  > This document serves as the absolute truth for the current backend architecture of the SAHM Marketing OS. 
  > 
  > ...


## File: app/api/tools.py
- **Purpose:** Unified tool endpoints: list the whole registry and invoke one tool by name.
- **Dependencies/Imports:** app.core.tools.connectors, __future__, app.core.tools.registry, typing, pydantic, app.core.tools.settings, app.core.agents.personas, app.core.tools.base, app.core.llm.base, app.core.errors, fastapi, app.api.deps, sqlalchemy.ext.asyncio
- **Key Classes:** ToolInfo
- **Key Functions:** _ctx, _personas_for, _resolve, _issues, list_tools, list_connectors, invoke_tool
- **Status:** Fully implemented


## File: app/api/assistant.py
- **Purpose:** Assistant endpoints: POST /assistant/message, GET conversations.
- **Dependencies/Imports:** traceback, __future__, asyncio, app.core.orchestrator.service, json, app.schemas.assistant, fastapi.responses, fastapi, app.api.deps
- **Key Classes:** 
- **Key Functions:** _sse, post_message, post_message_stream, get_conversations, get_conversation_detail, events
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 93: pass)


## File: app/core/agents/personas.py
- **Purpose:** The persona catalog. Data, not code.
- **Dependencies/Imports:** app.schemas.common, __future__, app.core.agents.persona
- **Key Classes:** 
- **Key Functions:** all_personas
- **Status:** Fully implemented


## File: app/schemas/common.py
- **Purpose:** Shared primitives and enums mirrored from the frontend contract.
- **Dependencies/Imports:** __future__, enum, typing, pydantic, datetime, pydantic.alias_generators
- **Key Classes:** ApiModel, Tone, Autonomy, Language, Tier, ToolSideEffect, ToolStatus, TaskStatus, StepStatus, RecommendationStatus, MessageRole, BlockKind, IntentKind, Evidence, Option, Usage, TaskStep, ApiError, ErrorResponse
- **Key Functions:** __add__
- **Status:** Fully implemented


## File: app/core/llm/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** 
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/ai/agents/nour/gui_graph.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langgraph.graph, langchain_core.messages, langchain_openai, app.schemas.blocks, typing, langchain_core.prompts, os
- **Key Classes:** NourState
- **Key Functions:** call_llm
- **Status:** Fully implemented


## File: app/ai/tools/get_reviews.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** app.ai.loggers.tools.get_reviews, typing, pydantic, json, time, langchain_core.tools, app.ai.providers.reviews, app.ai.errors
- **Key Classes:** GetReviewsInput
- **Key Functions:** _result, run_get_reviews, get_reviews_tool
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 110: 1. Always pass `location` (cit)


## File: app/schemas/agent.py
- **Purpose:** Schemas for the agent runtime's own results and events.
- **Dependencies/Imports:** pydantic, app.schemas.common, __future__
- **Key Classes:** ToolCallRecord, AgentResult, AgentEvent, RunAgentRequest
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/tools/records.py
- **Purpose:** records tool (13): create_recommendation.
- **Dependencies/Imports:** uuid, __future__, app.schemas.common, app.core.tools.base, app.core.logging, app.schemas.tools, app.schemas.recommendations, app.db.models
- **Key Classes:** 
- **Key Functions:** _handle_create_recommendation
- **Status:** Fully implemented


## File: app/api/business.py
- **Purpose:** Business endpoints: POST /business, GET|PATCH /business/profile.
- **Dependencies/Imports:** app.core.memory.store, __future__, app.schemas.business, sqlalchemy, app.core.errors, fastapi, app.api.deps, app.db.models
- **Key Classes:** 
- **Key Functions:** create_business, get_business_profile, update_business_profile, _profile_response
- **Status:** Fully implemented


## File: app/api/creatives.py
- **Purpose:** Creative endpoints: analyze-reference and generate.
- **Dependencies/Imports:** app.schemas.creatives, __future__, app.core.tools.media, sqlalchemy, app.core.tools.base, fastapi, app.schemas.tools, app.api.deps, app.db.models
- **Key Classes:** 
- **Key Functions:** _ctx, analyze_reference, generate_creative, list_creatives
- **Status:** Fully implemented


## File: app/schemas/llm_settings.py
- **Purpose:** Workspace-scoped LLM settings API schemas.
- **Dependencies/Imports:** pydantic, app.schemas.common, __future__, typing
- **Key Classes:** LlmTierUpdate, LlmSettingsUpdate, LlmTierOut, LlmSettingsOut
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/api/recommendations.py
- **Purpose:** Recommendation endpoints, including the two-phase approval flow.
- **Dependencies/Imports:** __future__, app.core.approvals, sqlalchemy, app.db.models, app.schemas.recommendations, fastapi, app.api.deps, sqlalchemy.ext.asyncio
- **Key Classes:** 
- **Key Functions:** _owned_recommendations, _load_owned, list_recommendations, get_recommendation, approve_recommendation, reject_recommendation
- **Status:** Fully implemented


## File: app/core/tools/research.py
- **Purpose:** Research tools (1-6).
- **Dependencies/Imports:** __future__, app.core.errors, app.core.logging, sys, app.schemas.tools, httpx, app.core.security, app.core.tools.settings, app.core.llm.base, urllib.parse, brand_mcp, hashlib, pydantic, json, app.config, app.schemas.common, bs4, typing, app.core.tools.base, os
- **Key Classes:** _WebsiteExtraction, BrandReportInput, BrandReportOutput, FullAnalysisInput, FullAnalysisOutput, RouteFileInput, RouteFileOutput
- **Key Functions:** _settings, _configured_search_key, _connectors, _unavailable, _search_tavily, _search_serper, _search_brave, _search_mock, _handle_web_search, _fetch_website, _parse_html, _handle_analyze_website, _handle_get_social_profile, _reviews_mock, _handle_get_reviews, _keywords_mock, _handle_get_keywords, _handle_get_competitor_ads, _handle_brand_report, _handle_full_analysis, _handle_route_file, _meta
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 6 instances (e.g. Line 275: # Validate every hop; redirect)


## File: app/core/llm/claude_client.py
- **Purpose:** Anthropic Claude client behind the LLMClient protocol.
- **Dependencies/Imports:** __future__, app.schemas.common, typing, httpx, app.core.llm.base, app.core.errors
- **Key Classes:** ClaudeClient
- **Key Functions:** __init__, client, aclose, _headers, _build_body, _complete_once
- **Status:** Fully implemented


## File: app/ai/providers/llm/anthropic.py
- **Purpose:** Anthropic (Claude) provider.
- **Dependencies/Imports:** langchain_anthropic, __future__, langchain_core.language_models, app.ai.enums, app.ai.settings, app.ai.errors
- **Key Classes:** 
- **Key Functions:** build_chat_model
- **Status:** Fully implemented


## File: app/api/usage.py
- **Purpose:** ``GET /api/v1/usage`` -- the meter the UI shows (P4.1).
- **Dependencies/Imports:** __future__, sqlalchemy, app.core.errors, app.schemas.usage, datetime, fastapi, app.core.usage, app.api.deps, app.db.models
- **Key Classes:** 
- **Key Functions:** _percent, _breakdown, get_usage
- **Status:** Fully implemented


## File: app/schemas/usage.py
- **Purpose:** Schemas for ``GET /api/v1/usage`` (P4.1).
- **Dependencies/Imports:** pydantic, app.schemas.common, __future__, datetime
- **Key Classes:** UsageBreakdown, UsageResponse
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/ai/agents/layla/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** graph
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/api/campaigns.py
- **Purpose:** Campaign endpoints.
- **Dependencies/Imports:** uuid, __future__, sqlalchemy, app.schemas.campaigns, fastapi, app.api.deps, app.db.models
- **Key Classes:** 
- **Key Functions:** _to_schema, create_campaign, list_campaigns, get_campaign
- **Status:** Fully implemented


## File: app/ai/agents/nour/state.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langchain_core.messages, langgraph.graph.message, typing
- **Key Classes:** NourState
- **Key Functions:** 
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 9: Additional state like business_i)


## File: .env
- **Purpose:** Configuration/Documentation file. (122 lines)
- **Env Keys:** SAHM_ENVIRONMENT, LOG_LEVEL, LOG_JSON, API_PREFIX, APP_NAME, CORS_ORIGINS, SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, SUPABASE_SECRET_KEY, SUPABASE_JWKS_URL, DATABASE_URL, AI_LLM_PROVIDER, GROQ_MODEL, GROQ_API_KEY, OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SECONDS, GEMINI_MODEL, GEMINI_API_KEY, OPENAI_MODEL, OPENAI_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_API_KEY, OPENROUTER_MODEL, OPENROUTER_API_KEY, AI_SEARCH_PROVIDER, TAVILY_API_KEY, DAILY_TOKEN_BUDGET, WEBSITE_ALLOWED_SCHEMES, WEBSITE_FETCH_TIMEOUT_SECONDS, WEBSITE_MAX_BYTES, AI_SQLITE_PATH, AI_DB_PROVIDER, META_GRAPH_BASE_URL, META_GRAPH_VERSION, META_API_TIMEOUT_SECONDS, META_APP_ID, META_APP_SECRET, META_AD_ACCOUNT_ID, TOKEN_ENCRYPTION_KEY, UPLOAD_DIR, UPLOAD_MAX_BYTES, UPLOAD_ALLOWED_TYPES, GOOGLE_PLACES_API_KEY, DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD, APIFY_TOKEN, AI_LOG_DIR, AI_LOG_LEVEL, AI_LOG_FORMAT, AI_LOG_TO_CONSOLE, AI_LLM_MAX_TOKENS


## File: app/ai/tools/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** app.ai.tools.search, app.ai.tools.brand_context, app.ai.tools.analyze_website, app.ai.tools.get_reviews, app.ai.tools.brand_research
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/schemas/creatives.py
- **Purpose:** Schemas for creative endpoints.
- **Dependencies/Imports:** app.schemas.common, __future__, pydantic, app.core.tools.base, datetime
- **Key Classes:** AnalyzeReferenceRequest, StyleSpec, ReferenceAnalysis, GenerateCreativeRequest, GeneratedAsset, GenerateCreativeResponse
- **Key Functions:** model_post_init
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 2 instances (e.g. Line 6: passed through.)


## File: app/schemas/tasks.py
- **Purpose:** Schemas for long-running agent tasks (AgentTask records).
- **Dependencies/Imports:** pydantic, app.schemas.common, __future__, datetime
- **Key Classes:** AgentTask, CancelTaskResponse
- **Key Functions:** 
- **Status:** Fully implemented


## File: TOOLS_ASSIGNMENT.md
- **Purpose:** Configuration/Documentation file. (60 lines)
- **Content Summary:**
  > # Tools Development Assignment Matrix
  > 
  > This document outlines the high-value AI tools we need to build for the SAHM OS. Developers should claim a tool, follow the `DEVELOPER_TOOL_GUIDE.md`, and submit their PRs.
  > 
  > ---
  > ...


## File: app/ai/agents/karim/schema.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** pydantic
- **Key Classes:** PerformanceReport
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/tools/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** 
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: full_report.md
- **Purpose:** Configuration/Documentation file. (118 lines)
- **Content Summary:**
  > ## File: ARCHITECTURE_OVERVIEW.md
  > - **Purpose:** Configuration/Documentation file. (60 lines)
  > - **Content Summary:**
  >   > # SAHM Marketing OS - Architecture Overview
  >   > 
  > ...


## File: app/api/tasks.py
- **Purpose:** Task endpoints: GET /tasks/{id}, POST /tasks/{id}/cancel.
- **Dependencies/Imports:** app.core.llm.router, __future__, app.core.orchestrator.service, asyncio, app.schemas.tasks, app.core.agents.persona, app.db.session, app.core.errors, fastapi.responses, fastapi, app.api.deps, app.db.models
- **Key Classes:** 
- **Key Functions:** _to_schema, read_task, cancel, stream, start_task
- **Status:** Fully implemented


## File: app/core/tools/memory.py
- **Purpose:** Memory tools: get_brand_context (read) and save_learning (draft_write).
- **Dependencies/Imports:** app.core.memory.store, __future__, app.schemas.common, typing, app.core.tools.base, app.schemas.tools
- **Key Classes:** 
- **Key Functions:** _handle_get_brand_context, _handle_save_learning
- **Status:** Fully implemented


## File: app/schemas/settings.py
- **Purpose:** Schemas for ``/api/v1/settings/connectors`` (P2.4).
- **Dependencies/Imports:** pydantic, app.schemas.common, __future__, typing
- **Key Classes:** ConnectorFieldOut, ConnectorOut, ConnectorSettingsResponse, ConnectorUpdateRequest, ConnectorDeleteResponse
- **Key Functions:** sent
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 106: password: SecretStr | None = N)


## File: app/ai/agents/salma/state.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langchain_core.messages, langgraph.graph.message, typing
- **Key Classes:** SalmaState
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/tools/base.py
- **Purpose:** Layer 2: the Tool model, ToolContext, and the registry.
- **Dependencies/Imports:** uuid, __future__, app.schemas.common, typing, pydantic, dataclasses, app.db.models, time, app.core.llm.base, app.core.errors, app.config, app.core.logging, fastmcp, sqlalchemy.ext.asyncio
- **Key Classes:** ToolContext, ToolOutput, Tool, ToolRegistry
- **Key Functions:** get_registry, reset_registry, log_tool_call, to_mcp_server, require_settings, unavailable, partial, to_spec, is_external_write, __init__, register, register_all, get, all, names, has, for_names, specs_for, _make, _handler
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 36: Handlers never read globals or )


## File: app/ai/agents/layla/state.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langchain_core.messages, langgraph.graph.message, typing
- **Key Classes:** LaylaState
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/ai/providers/search/base.py
- **Purpose:** Contract shared by every search provider.
- **Dependencies/Imports:** pydantic, abc, __future__
- **Key Classes:** SearchResult, BaseSearchProvider
- **Key Functions:** search
- **Status:** Fully implemented


## File: app/ai/memory/brand_memory.py
- **Purpose:** Long-term brand memory layout on top of a LangGraph ``BaseStore``.
- **Dependencies/Imports:** langgraph.store.base, __future__, pydantic, app.ai.enums, datetime
- **Key Classes:** BrandProfile, Learning
- **Key Functions:** profile_namespace, learnings_namespace, put_brand_profile, get_brand_profile, business_id_from, agent_from
- **Status:** Fully implemented


## File: app/ai/providers/db/postgres.py
- **Purpose:** PostgreSQL persistence. Selected when ``AI_DB_PROVIDER=postgres``.
- **Dependencies/Imports:** langgraph.store.base, __future__, psycopg, typing, langgraph.checkpoint.postgres, langgraph.checkpoint.base, langgraph.store.postgres, app.ai.settings, psycopg.rows, app.ai.errors
- **Key Classes:** 
- **Key Functions:** _connect, build_checkpointer, build_store
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 4: ``AI_POSTGRES_URL=postgresql://u)


## File: app/api/integrations.py
- **Purpose:** Integration endpoints: Meta status and connect.
- **Dependencies/Imports:** uuid, __future__, app.schemas.integrations, httpx, sqlalchemy, app.db.models, app.core.security, datetime, app.core.logging, fastapi, app.api.deps, sqlalchemy.ext.asyncio
- **Key Classes:** 
- **Key Functions:** _meta_integration, meta_status, meta_connect, meta_disconnect
- **Status:** Fully implemented


## File: app/ai/providers/reviews.py
- **Purpose:** Google Places and Apify providers for customer reviews.
- **Dependencies/Imports:** __future__, typing, httpx, pydantic, unidecode, urllib.parse, app.ai.settings, re, app.ai.errors, app.ai.loggers.providers.reviews, difflib
- **Key Classes:** PlaceReview, PlaceResult, GooglePlacesProvider, ApifyReviewsProvider
- **Key Functions:** _review_from_google, _name_score, _review_from_apify, get_google_places, get_apify, __init__, search, __init__, _run, search, fetch
- **Status:** Fully implemented


## File: app/ai/providers/search/tavily.py
- **Purpose:** Tavily search provider. Selected when ``AI_SEARCH_PROVIDER=tavily``.
- **Dependencies/Imports:** app.ai.providers.search.base, __future__, httpx, app.ai.settings, app.ai.loggers.providers.search, app.ai.errors
- **Key Classes:** TavilySearchProvider
- **Key Functions:** build_search_provider, __init__, search
- **Status:** Fully implemented


## File: app/schemas/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** 
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/errors.py
- **Purpose:** Typed error hierarchy.
- **Dependencies/Imports:** __future__
- **Key Classes:** SahmError, LLMError, LLMOutputError, LLMProviderError, LLMNotConfiguredError, ToolError, ToolUnavailable, ToolValidationError, ToolNotFoundError, PolicyViolation, ToolNotAllowedError, TaskNotAllowedError, ExternalWriteBlockedError, ApprovalRequired, ApprovalNotFoundError, ApprovalStateError, TokenBudgetExceeded, NotFoundError, ValidationFailed, UnauthorizedError, ForbiddenError, WorkspaceRequiredError, EmailTakenError, BlockedTargetError, EncryptionKeyMissingError, RateLimitExceeded, UploadRejectedError
- **Key Functions:** __init__, to_dict, __init__, __init__
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 206: for both "unknown address" and)


## File: app/api/auth_complete_profile.py
- **Purpose:** Complete profile endpoint for new OAuth users.
- **Dependencies/Imports:** uuid, __future__, app.schemas.auth, sqlalchemy, app.db.models, app.core.supabase_auth, app.core.errors, app.api.auth, fastapi, app.api.deps, sqlalchemy.ext.asyncio
- **Key Classes:** 
- **Key Functions:** complete_profile
- **Status:** Fully implemented


## File: app/schemas/audit.py
- **Purpose:** Schemas for ``GET /api/v1/audit`` (P4.4).
- **Dependencies/Imports:** pydantic, app.schemas.common, __future__, datetime
- **Key Classes:** AuditEntry, AuditList
- **Key Functions:** 
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 10: put there. They are passed thro)


## File: app/schemas/business.py
- **Purpose:** Schemas for POST /business and GET /business/profile.
- **Dependencies/Imports:** app.schemas.common, __future__, typing, pydantic, datetime
- **Key Classes:** BusinessCreateRequest, BrandProfile, BusinessProfileResponse, BusinessProfileUpdateRequest
- **Key Functions:** provided
- **Status:** Fully implemented


## File: app/schemas/blocks.py
- **Purpose:** The ``Message.blocks`` discriminated union.
- **Dependencies/Imports:** app.schemas.common, __future__, typing, pydantic, datetime
- **Key Classes:** BaseBlock, TextBlock, Finding, FindingsBlock, RecommendationBlock, MetricBlock, ActionCard, TaskProgressBlock, ContextItem, ContextUsedBlock, QuestionWithOptionsBlock, ErrorBlock, AgentEventData, Message, MessageWithBlocks
- **Key Functions:** _block_discriminator, _accept_plain_list
- **Status:** Fully implemented


## File: app/api/auth.py
- **Purpose:** Authentication endpoints: register, login, refresh, logout, me.
- **Dependencies/Imports:** uuid, __future__, app.schemas.auth, sqlalchemy, app.db.models, app.core.auth, app.core.errors, app.config, datetime, app.core.logging, fastapi, sqlalchemy.exc, app.api.deps, sqlalchemy.ext.asyncio
- **Key Classes:** _FailureCounter, TooManyAttempts
- **Key Functions:** _client_ip, _user_out, _workspaces_for, _active_workspace, _token_response, _user_out_from_principal, register, _email_taken, login, _invalid_credentials, refresh, logout, me, __init__, is_locked, record_failure, reset
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 16 instances (e.g. Line 3: Five routes, all under ``/api/v1)


## File: app/api/audit.py
- **Purpose:** ``GET /api/v1/audit`` -- the audit log, finally readable (P4.4).
- **Dependencies/Imports:** __future__, app.schemas.audit, sqlalchemy, app.core.approvals.service, app.core.logging, fastapi, app.api.deps, app.db.models
- **Key Classes:** 
- **Key Functions:** _entry, get_audit
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 72: Scoped to the caller's business)


## File: app/ai/enums.py
- **Purpose:** Every selectable option in the AI layer, as an Enum.
- **Dependencies/Imports:** __future__, enum
- **Key Classes:** LLMProvider, SearchProvider, DatabaseProvider, StructuredOutputMethod, SearchDepth, LogKind, LogLevel, LogFormat, AgentName, ToolName, LearningCategory
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/llm/ollama_client.py
- **Purpose:** Ollama client behind the LLMClient protocol.
- **Dependencies/Imports:** uuid, __future__, app.schemas.common, typing, httpx, json, app.core.llm.base, app.core.errors, app.core.logging
- **Key Classes:** OllamaClient
- **Key Functions:** __init__, client, aclose, _endpoint, _build_body, _complete_once
- **Status:** Fully implemented


## File: app/ai/tools/search.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langchain_core.messages, langchain_core.runnables.config, typing, langchain_core.tools, app.ai.providers.llm, app.ai.enums, app.ai.providers.search, app.ai.settings
- **Key Classes:** 
- **Key Functions:** web_search_tool
- **Status:** Fully implemented


## File: app/core/tools/settings.py
- **Purpose:** P2.4/P2.5: per-workspace connector configuration.
- **Dependencies/Imports:** __future__, typing, dataclasses, sqlalchemy, app.db.models, app.core.security, app.config, app.core.logging, sqlalchemy.ext.asyncio
- **Key Classes:** FieldSpec, ConnectorSpec, ResolvedField, ResolvedConnector, ConnectorSettings
- **Key Functions:** connector_spec, _settings_value, resolve_connector, load_connector_settings, resolve_tool_settings, field, required_fields, key, has_value, field, value_of, provider, is_placeholder, configured, source, connector, value, source, meta_access_token, meta_app_id, meta_app_secret, meta_ad_account_id, meta_graph_base_url, google_places_api_key, places_base_url, keyword_provider, dataforseo_login, dataforseo_password, image_provider, image_api_key, image_model, image_base_url, search_provider, search_api_key
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 5 instances (e.g. Line 228: key="password",)


## File: app/core/llm/router.py
- **Purpose:** Provider selection. The ONLY place a provider is chosen.
- **Dependencies/Imports:** hashlib, __future__, sqlalchemy, app.db.models, json, app.core.security, app.core.llm.base, app.core.errors, app.config, app.core.llm.claude_client, app.core.llm.openai_client, app.core.logging, app.core.llm.mock_client, sqlalchemy.ext.asyncio
- **Key Classes:** 
- **Key Functions:** build_client, resolve_workspace_tier, get_workspace_llm, get_llm, reset_clients, close_clients
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 202: pass)


## File: app/ai/memory/__init__.py
- **Purpose:** Shared long-term memory helpers (brand profile + learnings).
- **Dependencies/Imports:** app.ai.memory.brand_memory
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/ai/providers/__init__.py
- **Purpose:** Providers: the ONLY place that knows about concrete backends.
- **Dependencies/Imports:** 
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: DATABASE_GUIDE.md
- **Purpose:** Configuration/Documentation file. (91 lines)
- **Content Summary:**
  > # SAHM Marketing OS - Database & Tooling Guide
  > 
  > This guide explains the backend database schema (powered by SQLAlchemy + Postgres/SQLite) and how you should design AI tools to seamlessly save their outputs into these tables using Pydantic schemas.
  > 
  > ---
  > ...


## File: app/ai/agents/salma/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** graph
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/approvals/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** 
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/approvals/service.py
- **Purpose:** Layer 7: human-in-the-loop approvals.
- **Dependencies/Imports:** uuid, __future__, app.core.tools.protected, typing, sqlalchemy, app.db.models, app.core.tools, app.core.tools.base, app.core.errors, datetime, app.core.logging, app.core.llm.mock_client, app.schemas.recommendations, sqlalchemy.ext.asyncio
- **Key Classes:** 
- **Key Functions:** _now, _new_id, _write_audit, _current_value_for, build_diff, _to_schema, get_recommendation, approve_recommendation, _tenant_of_campaign, _execute, publish_campaign, reject_recommendation, list_audit
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 2 instances (e.g. Line 320: # The protected handler does n)


## File: app/api/personas.py
- **Purpose:** Persona endpoints: GET /personas, PATCH /personas/{id}.
- **Dependencies/Imports:** app.schemas.personas, __future__, app.core.agents.persona, app.core.agents.personas, app.core.errors, fastapi, app.api.deps
- **Key Classes:** 
- **Key Functions:** _to_response, list_personas, update_persona
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 81: # to be re-read here -- passing)


## File: app/schemas/assistant.py
- **Purpose:** Schemas for the assistant endpoints (POST /assistant/message, ...).
- **Dependencies/Imports:** app.schemas.blocks, app.schemas.common, __future__, pydantic, datetime
- **Key Classes:** MessageContext, AssistantMessageRequest, AssistantMessageResponse, ConversationSummary, ConversationDetail
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/tools/connectors.py
- **Purpose:** One map of every tool to the external service it talks to.
- **Dependencies/Imports:** dataclasses, app.config, __future__, typing
- **Key Classes:** Connector
- **Key Functions:** _setting, resolve_url, connector_for, summary, resolve, _selected_provider, _missing
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 4 instances (e.g. Line 31: :mod:`app.core.tools.settings`.)


## File: app/core/memory/store.py
- **Purpose:** Brand memory: read/write plus token-bounded context assembly.
- **Dependencies/Imports:** uuid, __future__, typing, sqlalchemy, app.db.models, json, math, app.core.tools.base, app.core.errors, app.core.memory.embeddings, datetime, app.core.logging, pgvector.sqlalchemy, app.schemas.tools, sqlalchemy.ext.asyncio
- **Key Classes:** BrandMemoryStore
- **Key Functions:** _estimate_tokens, _new_id, _row_to_item, _label_for, get_brand_context, format_context_for_prompt, __init__, embeddings, write_item, save_learning, _search_pgvector, _search_python, search, recent_learnings, get_business, get_profile, count_items, list_items, get_item, update_item, delete_item, reindex_items, build_context
- **Status:** Fully implemented


## File: app/ai/agents/orchestrator/brain.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** app.schemas.blocks, langchain_core.messages, asyncio, typing, pydantic, app.ai.agents.layla.graph, json, app.ai.agents.nour.graph, app.ai.providers.llm, app.ai.settings, app.ai.agents.karim.graph, app.ai.agents.omar.graph, app.ai.agents.salma.graph, langchain_core.runnables
- **Key Classes:** RoutingDecision
- **Key Functions:** orchestrate
- **Status:** Fully implemented


## File: app/core/supabase_auth.py
- **Purpose:** Supabase Auth integration: JWT verification and profile loading.
- **Dependencies/Imports:** uuid, __future__, jose, typing, httpx, dataclasses, sqlalchemy, app.db.models, json, time, secrets, jose.exceptions, app.config, datetime, app.core.logging, functools, sqlalchemy.ext.asyncio
- **Key Classes:** MembershipGrant, Principal, SupabaseAuthError
- **Key Functions:** _fetch_jwks, _get_signing_key, verify_supabase_token, load_profile, load_principal_from_profile, get_or_create_profile, create_legacy_token, membership_for_business, primary_workspace, role_for, has_role, __init__
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 3: Replaces the custom auth module )


## File: app/core/memory/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** 
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/ai/agents/omar/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** graph
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/ai/agents/karim/graph.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langgraph.graph, langchain_core.messages, langgraph.prebuilt, typing, app.ai.agents.karim.state, app.ai.agents.karim.schema, langchain_core.tools, app.ai.providers.llm, app.ai.settings, app.ai.tools, langchain_core.runnables
- **Key Classes:** 
- **Key Functions:** submit_performance_report, call_model, should_continue, post_tools_routing
- **Status:** Fully implemented


## File: app/ai/agents/salma/graph.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langgraph.graph, langchain_core.messages, langgraph.prebuilt, typing, langchain_core.tools, app.ai.providers.llm, app.ai.settings, app.ai.agents.salma.state, app.ai.tools, langchain_core.runnables, app.ai.agents.salma.schema
- **Key Classes:** 
- **Key Functions:** submit_optimization_plan, call_model, should_continue, post_tools_routing
- **Status:** Fully implemented


## File: app/api/insights.py
- **Purpose:** Insights endpoints: GET /campaigns/{id}/insights.
- **Dependencies/Imports:** app.schemas.insights, app.core.llm.mock_client, __future__, sqlalchemy, app.core.tools.base, app.core.tools.meta_read, datetime, fastapi, app.schemas.tools, app.api.deps, app.db.models
- **Key Classes:** 
- **Key Functions:** get_campaign_insights
- **Status:** Fully implemented


## File: app/ai/providers/db/sqlite.py
- **Purpose:** SQLite persistence. Selected when ``AI_DB_PROVIDER=sqlite``.
- **Dependencies/Imports:** sqlite3, langgraph.store.sqlite, langgraph.store.base, pathlib, __future__, langgraph.checkpoint.sqlite, langgraph.checkpoint.base, app.ai.settings, app.ai.errors
- **Key Classes:** 
- **Key Functions:** _connect, build_checkpointer, build_store
- **Status:** Fully implemented


## File: app/schemas/tools.py
- **Purpose:** Input and output models for every tool.
- **Dependencies/Imports:** app.schemas.creatives, app.schemas.common, __future__, pydantic, app.core.tools.base, datetime
- **Key Classes:** SearchResult, WebSearchInput, WebSearchOutput, WebsiteProfileInput, Offer, WebsiteProfile, SocialProfileInput, SocialPost, SocialProfile, GetReviewsInput, Review, GetReviewsOutput, KeywordMetric, GetKeywordsInput, GetKeywordsOutput, AdRecord, GetCompetitorAdsInput, GetCompetitorAdsOutput, MemoryScope, GetBrandContextInput, BrandContextItem, BrandContext, SaveLearningInput, Learning, GetCampaignInsightsInput, GetBreakdownsInput, AnalyzeReferenceInput, GenerateImageInput, GeneratedAssetRecord, GenerateImageOutput, CreateRecommendationInput, CreateRecommendationOutput, ContextUsedView
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/llm/openai_client.py
- **Purpose:** OpenAI client behind the LLMClient protocol.
- **Dependencies/Imports:** __future__, app.schemas.common, typing, httpx, json, app.core.llm.base, app.core.errors, app.core.logging
- **Key Classes:** OpenAIClient
- **Key Functions:** _json_dumps, _json_loads, __init__, client, aclose, _headers, _build_body, _complete_once
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 29: #: OpenAI-compatible gateway (O)


## File: app/ai/agents/karim/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** graph
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/ai/providers/llm/__init__.py
- **Purpose:** LLM providers: pick a chat model by enum, import its module lazily.
- **Dependencies/Imports:** __future__, dataclasses, pydantic, app.ai.loggers.providers.llm, langchain_core.language_models, app.ai.enums, app.ai.settings, app.ai.providers._loader, langchain_core.runnables
- **Key Classes:** LLMHandle
- **Key Functions:** get_llm, structured
- **Status:** Fully implemented


## File: app/ai/providers/search/mock.py
- **Purpose:** Mock search provider. Selected when ``AI_SEARCH_PROVIDER=mock``.
- **Dependencies/Imports:** hashlib, __future__, app.ai.providers.search.base, app.ai.settings
- **Key Classes:** MockSearchProvider
- **Key Functions:** build_search_provider, search
- **Status:** Fully implemented


## File: app/ai/providers/llm/gemini.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** pydantic, langchain_google_genai, langchain_core.language_models, app.ai.enums, app.ai.settings
- **Key Classes:** 
- **Key Functions:** build_chat_model
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 12: # Note: ensure settings has gem)


## File: app/schemas/personas.py
- **Purpose:** API-level persona schemas.
- **Dependencies/Imports:** pydantic, app.schemas.common, __future__
- **Key Classes:** PersonaResponse, PersonaUpdateRequest
- **Key Functions:** changed_fields
- **Status:** Fully implemented


## File: app/core/agents/persona.py
- **Purpose:** Personas are data, not code.
- **Dependencies/Imports:** app.schemas.personas, app.schemas.common, __future__, typing, pydantic, sqlalchemy, app.core.agents.personas, app.core.errors, app.core.logging, app.db.models
- **Key Classes:** Persona
- **Key Functions:** load_persona, _load_overrides, save_persona_override, can_use_tool, can_run_task, with_overrides
- **Status:** Fully implemented


## File: app/core/orchestrator/service.py
- **Purpose:** Conversation and long-task orchestration.
- **Dependencies/Imports:** __future__, sqlalchemy, app.core.orchestrator.blocks, app.core.errors, datetime, app.core.logging, app.core.agents.runtime, app.schemas.blocks, app.core.agents.persona, app.ai.agents.orchestrator.brain, uuid, asyncio, json, app.db.models, app.schemas.common, typing, app.schemas.assistant, app.core.tools.base, re, sqlalchemy.ext.asyncio
- **Key Classes:** 
- **Key Functions:** _now, _new_id, _resolve_retry_message, get_or_create_conversation, list_conversations, get_conversation, _persist_message, handle_message, create_task, get_task, cancel_task, run_task_in_background, stream_task, _sse
- **Status:** Fully implemented


## File: app/ai/agents/omar/schema.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** pydantic
- **Key Classes:** StrategyReport
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/ratelimit.py
- **Purpose:** P4.2: dependency-free in-process rate limiting.
- **Dependencies/Imports:** __future__, asyncio, app.schemas.common, dataclasses, time, starlette.types, app.core.auth, starlette.responses, app.core.errors, app.config, app.core.logging, starlette.datastructures, starlette.requests, app.db.session
- **Key Classes:** Limit, _Bucket, Verdict, RateLimiter, RateLimitMiddleware
- **Key Functions:** limits_from_settings, classify, _client_key, build_limiter, get_limiter, reset_limiter, refill_per_minute, headers, consume, _evict, reset, __init__, __call__, _resolve_identity, _send_429, send_wrapper
- **Status:** Fully implemented


## File: app/core/llm/base.py
- **Purpose:** Layer 1: the single LLM abstraction.
- **Dependencies/Imports:** __future__, app.schemas.common, typing, asyncio, pydantic, httpx, json, time, app.core.errors, abc, app.core.logging, re
- **Key Classes:** ToolCall, Msg, ToolSpec, LLMResult, LLMClient, BaseLLMClient
- **Key Functions:** json_schema_for, schema_instruction, extract_json, format_validation_error, _sleep, _retry_after_seconds, is_retryable_status, is_retryable_exception, from_model, complete, _resolve, __init__, _complete_once, complete, _call_with_retries, _validate, _log_usage
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 163: pass)


## File: app/schemas/memory.py
- **Purpose:** Schemas for ``/api/v1/memory/*`` (P2.2).
- **Dependencies/Imports:** app.core.memory.store, __future__, app.schemas.common, typing, pydantic, json, datetime
- **Key Classes:** MemoryItemCreate, MemoryItemUpdate, MemoryItemOut, MemoryItemList, MemoryReindexRequest, MemoryReindexResponse
- **Key Functions:** _check_metadata, _bound_metadata, _bound_metadata
- **Status:** Fully implemented


## File: app/core/agents/tasks.py
- **Purpose:** Layer 3: TaskSpecs. This is where most of the product lives.
- **Dependencies/Imports:** app.core.memory.store, pathlib, __future__, app.schemas.common, typing, pydantic, json, app.core.llm.base, app.core.errors, app.core.logging, re
- **Key Classes:** TaskOutputBase, CompetitorProfile, CompetitorResearchOutput, AudienceSegment, MarketAnalysisOutput, Gap, MarketGapsOutput, ChannelPlan, MarketingStrategyOutput, CampaignStrategyOutput, ContentItem, ContentCalendarOutput, CreativeBriefOutput, Scene, CommercialScriptOutput, ImagePromptOutput, Shot, VideoPromptOutput, AdCopyOutput, AdVariation, AdVariationsOutput, PerformanceFinding, PerformanceAnalysisOutput, OptimizationProposalOutput, PromptTemplate, TaskSpec
- **Key Functions:** load_prompt, available_prompt_tasks, _build, get_task, all_tasks, validate_task_catalog, _persona_style_block, _brand_context_block, _render_value, _inputs_block, _tool_results_block, build_prompt, __init__, render, render_prompt, _sub
- **Status:** Fully implemented


## File: app/ai/providers/db/__init__.py
- **Purpose:** Database providers: LangGraph persistence selected by enum.
- **Dependencies/Imports:** langgraph.store.base, __future__, app.ai.providers._loader, langgraph.checkpoint.base, app.ai.enums, app.ai.settings, langgraph.config, functools, app.ai.loggers.providers.db
- **Key Classes:** 
- **Key Functions:** _checkpointer, _store, get_checkpointer, get_store, current_store, reset_db_providers
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 73: pass)


## File: app/core/security.py
- **Purpose:** Security helpers: SSRF protection, token encryption, upload validation.
- **Dependencies/Imports:** cryptography.fernet, __future__, typing, dataclasses, app.core.errors, os, app.config, urllib.parse, app.core.logging, ipaddress, socket
- **Key Classes:** SafeUrl, TokenCipher
- **Key Functions:** _is_private_ip, resolve_and_check, check_redirect, verify_runtime_security, sniff_mime, validate_upload, __init__, is_dev_mode, _dev_fernet, encrypt, decrypt
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 2 instances (e.g. Line 90: pass)


## File: app/ai/agents/salma/schema.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** pydantic
- **Key Classes:** OptimizationPlan
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/agents/runtime.py
- **Purpose:** Layer 5: the generic agent runtime.
- **Dependencies/Imports:** app.core.memory.store, __future__, collections.abc, app.core.errors, app.core.logging, app.schemas.tools, app.core.llm.router, app.schemas.agent, app.core.tools.registry, time, app.core.agents.persona, app.core.llm.base, app.core.usage, app.core.agents.tasks, uuid, asyncio, dataclasses, pydantic, json, app.config, app.schemas.common, typing, app.core.tools.base, re
- **Key Classes:** RunOptions, RunDependencies
- **Key Functions:** _llm_failure_message, describe_autonomy, blocks_draft_write, _apply_autonomy_policy, _tool_result_summary, _collect_unavailable, _has_research_data, _competitor_research_evidence, _tool_results_block, _compact_tool_payload, _bound_tool_transcript, _execute, _status_value, _run_one_tool, run_agent, run_agent_stream, from_settings, _result
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 4 instances (e.g. Line 77: pass)


## File: app/ai/providers/llm/ollama.py
- **Purpose:** Ollama provider (local, or remote over ngrok / Google Colab).
- **Dependencies/Imports:** __future__, typing, langchain_core.language_models, app.ai.enums, app.ai.settings, app.ai.errors, langchain_ollama
- **Key Classes:** 
- **Key Functions:** build_chat_model
- **Status:** Fully implemented


## File: app/ai/providers/llm/openrouter.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langchain_openai, pydantic, langchain_core.language_models, app.ai.enums, app.ai.settings
- **Key Classes:** 
- **Key Functions:** build_chat_model
- **Status:** Fully implemented


## File: DEVELOPER_TOOL_GUIDE.md
- **Purpose:** Configuration/Documentation file. (88 lines)
- **Content Summary:**
  > # Developer Guide: Building AI Tools for SAHM OS
  > 
  > Welcome to the SAHM AI Tool Development team! We recently upgraded our entire AI engine to a modern **LangChain & LangGraph** architecture. 
  > 
  > This guide will explain exactly how you should build tools, how they interact with our database, how the UI displays them, and how to submit your work so it can be merged and tested instantly.
  > ...


## File: app/ai/tools/brand_research.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** brand_mcp, json, os, langchain_core.tools, sys
- **Key Classes:** 
- **Key Functions:** brand_report, full_analysis, route_file_to_tool
- **Status:** Fully implemented


## File: app/ai/agents/omar/state.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langchain_core.messages, langgraph.graph.message, typing
- **Key Classes:** OmarState
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/ai/agents/karim/state.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langchain_core.messages, langgraph.graph.message, typing
- **Key Classes:** KarimState
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/ai/providers/db/memory.py
- **Purpose:** In-memory persistence. Selected when ``AI_DB_PROVIDER=memory``.
- **Dependencies/Imports:** langgraph.checkpoint.memory, langgraph.store.base, __future__, langgraph.store.memory, langgraph.checkpoint.base, app.ai.settings
- **Key Classes:** 
- **Key Functions:** build_checkpointer, build_store
- **Status:** Fully implemented


## File: app/core/memory/embeddings.py
- **Purpose:** Embedding providers behind one interface.
- **Dependencies/Imports:** hashlib, __future__, typing, httpx, math, app.core.llm.base, app.core.errors, app.config, app.core.logging, re
- **Key Classes:** EmbeddingProvider, MockEmbeddingProvider, OpenAIEmbeddingProvider
- **Key Functions:** get_embedding_provider, reset_embedding_provider, embed, embed_batch, __init__, _vector, embed, embed_batch, __init__, client, aclose, embed_batch, embed
- **Status:** Fully implemented


## File: app/ai/agents/layla/schema.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** pydantic
- **Key Classes:** CreativeBrief
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/usage.py
- **Purpose:** Cost guard: per-workspace daily token budget and per-run caps.
- **Dependencies/Imports:** uuid, __future__, app.schemas.common, typing, dataclasses, sqlalchemy, app.core.errors, app.config, datetime, app.core.logging, app.db.models
- **Key Classes:** BudgetStatus, UsageWindow, RunUsageTracker
- **Key Functions:** workspace_usage_today, workspace_usage_since, period_start, period_end, workspace_usage_period, usage_breakdown, effective_daily_budget, check_budget, record_usage, exceeded, total_tokens, __init__, add, tokens, exceeded
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 205: own, so the caller (the agent )


## File: app/ai/agents/nour/graph.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langgraph.graph, langchain_core.messages, langgraph.prebuilt, app.ai.agents.nour.state, typing, app.ai.agents.nour.schema, json, langchain_core.tools, app.ai.providers.llm, app.ai.settings, app.ai.enums, app.ai.tools, langchain_core.runnables
- **Key Classes:** 
- **Key Functions:** submit_research_report, call_model, should_continue, post_tools_routing
- **Status:** Fully implemented


## File: app/api/supabase_exchange.py
- **Purpose:** Supabase token exchange endpoint.
- **Dependencies/Imports:** __future__, app.core.supabase_auth, app.config, app.core.logging, app.api.auth, fastapi, app.api.deps, sqlalchemy.ext.asyncio
- **Key Classes:** 
- **Key Functions:** supabase_exchange
- **Status:** Fully implemented


## File: app/schemas/insights.py
- **Purpose:** Schemas for insights endpoints and the meta_read tools.
- **Dependencies/Imports:** app.schemas.common, __future__, pydantic, app.core.tools.base, datetime
- **Key Classes:** DateRange, InsightsQuery, MetricSeries, InsightsPayload, BreakdownPayload, InsightRecord, CampaignInsightsResponse
- **Key Functions:** model_post_init
- **Status:** Fully implemented


## File: app/main.py
- **Purpose:** Sahm backend: FastAPI app, router registration, middleware, error handling.
- **Dependencies/Imports:** fastapi.exceptions, fastapi.middleware.cors, __future__, app.api, sqlalchemy, starlette.exceptions, app.core.errors, app.core.logging, app.core.security_headers, app.core.ratelimit, app.core.llm.router, app.core.tools.registry, time, app.core.security, uuid, app.config, fastapi, app.schemas.common, typing, contextlib, fastapi.responses, app.db.session
- **Key Classes:** 
- **Key Functions:** _http_error_code, _jsonable_errors, _check_database, lifespan, create_app, add_request_id, envelope, sahm_error_handler, validation_error_handler, http_exception_handler, unhandled_error_handler, health, readiness, root
- **Status:** Fully implemented


## File: app/core/orchestrator/blocks.py
- **Purpose:** Compose the frontend's ``Message.blocks`` from agent results.
- **Dependencies/Imports:** app.schemas.blocks, __future__, app.schemas.common, typing, pydantic, app.core.logging, urllib.parse
- **Key Classes:** 
- **Key Functions:** validate_block, safe_block, validate_blocks, text_block, error_block, findings_block, metric_block, action_card, recommendation_block, task_progress_block, context_used_block, question_block, _evidence_from_output, compose_blocks, _research_blocks, _strategy_block, _creative_blocks, _creative_block, _analysis_blocks, _optimization_blocks
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 1 instances (e.g. Line 301: pass)


## File: app/core/agents/prompts/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** 
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/ai/providers/_loader.py
- **Purpose:** Lazy enum -> module loader shared by every provider family.
- **Dependencies/Imports:** __future__, enum, collections.abc, importlib, types, app.ai.errors
- **Key Classes:** 
- **Key Functions:** load_provider_module
- **Status:** Fully implemented


## File: app/ai/agents/omar/graph.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langgraph.graph, langchain_core.messages, langgraph.prebuilt, typing, app.ai.agents.omar.schema, langchain_core.tools, app.ai.providers.llm, app.ai.settings, app.ai.tools, langchain_core.runnables, app.ai.agents.omar.state
- **Key Classes:** 
- **Key Functions:** submit_strategy_report, call_model, should_continue, post_tools_routing
- **Status:** Fully implemented


## File: app/api/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** 
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/agents/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** 
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/llm/mock_client.py
- **Purpose:** Deterministic fake LLM.
- **Dependencies/Imports:** __future__, app.schemas.common, typing, pydantic, json, app.core.llm.base
- **Key Classes:** MockClient
- **Key Functions:** _sample_string, _sample_number, _sample_int, sample_from_schema, sample_for_model, _inline, __init__, _called_tools, _complete_once
- **Status:** Fully implemented


## File: app/ai/providers/search/__init__.py
- **Purpose:** Search providers: pick one by enum, import its module lazily.
- **Dependencies/Imports:** functools, __future__, app.ai.providers.search.base, app.ai.enums, app.ai.settings, app.ai.loggers.providers.search, app.ai.providers._loader
- **Key Classes:** 
- **Key Functions:** _build, get_search_provider, reset_search_providers
- **Status:** Fully implemented


## File: app/ai/providers/llm/mock.py
- **Purpose:** Mock provider -- deterministic, offline chat model for tests and dry runs.
- **Dependencies/Imports:** langchain_core.messages, __future__, langchain_core.outputs, typing, pydantic, langchain_core.callbacks, langchain_core.language_models, types, app.ai.enums, app.ai.settings, langchain_core.runnables
- **Key Classes:** MockChatModel
- **Key Functions:** _last_human_text, _fake_value, fake_instance, build_chat_model, _llm_type, _generate, bind_tools, with_structured_output
- **Status:** Fully implemented


## File: app/config.py
- **Purpose:** Application configuration.
- **Dependencies/Imports:** __future__, typing, pydantic, json, pydantic_settings, functools
- **Key Classes:** Settings
- **Key Functions:** get_settings, _split_csv, _parse_extra_headers, provider_for_tier, model_for_tier, is_mock_llm, has_real_llm_key
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 5 instances (e.g. Line 83: # SQLAlchemy uses; a Supabase c)


## File: app/api/memory.py
- **Purpose:** Brand-memory CRUD: ``/api/v1/memory/*`` (P2.2).
- **Dependencies/Imports:** app.core.memory.store, __future__, sqlalchemy, app.core.errors, fastapi, app.schemas.memory, app.api.deps, app.db.models
- **Key Classes:** 
- **Key Functions:** _out, _validate_kind, list_memory_items, create_memory_item, update_memory_item, delete_memory_item, reindex_memory
- **Status:** Fully implemented


## File: app/ai/agents/nour/schema.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** pydantic
- **Key Classes:** CompetitorSummary, MarketResearchReport
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/tools/registry.py
- **Purpose:** Tool registry assembly.
- **Dependencies/Imports:** app.core.tools.research, __future__, app.core.tools.media, app.core.tools.memory, app.core.tools.base, app.core.tools.records, app.core.tools.meta_read, app.core.logging
- **Key Classes:** 
- **Key Functions:** get_registry, register_all, reset_registry, assert_no_external_write_reachable, get_tools_for, get_tool_names_for
- **Status:** Fully implemented


## File: app/schemas/auth.py
- **Purpose:** Auth request/response schemas.
- **Dependencies/Imports:** app.schemas.common, __future__, pydantic, app.core.auth, datetime
- **Key Classes:** RegisterRequest, LoginRequest, RefreshRequest, WorkspaceMembership, UserOut, AuthTokenResponse, LogoutResponse, MeResponse, CompleteProfileRequest, CompleteProfileResponse
- **Key Functions:** role_rank, _normalise, _policy, _normalise
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 11 instances (e.g. Line 16: #: Minimum password length. Del)


## File: app/core/tools/protected.py
- **Purpose:** Protected tools: the ONLY code that writes to Meta.
- **Dependencies/Imports:** uuid, __future__, app.schemas.common, typing, httpx, pydantic, sqlalchemy, app.core.security, app.core.tools.settings, app.core.tools.base, app.core.errors, datetime, app.core.logging, app.db.models
- **Key Classes:** ApplyChangeInput, ApplyChangeOutput, PublishCampaignInput, PublishCampaignOutput
- **Key Functions:** _meta_token, _graph_post_async, _graph_config, _handle_apply_change, _handle_publish_campaign
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 2 instances (e.g. Line 130: * the caller must have passed )


## File: app/api/deps.py
- **Purpose:** Shared API dependencies.
- **Dependencies/Imports:** uuid, app.core.llm.router, __future__, collections.abc, typing, dataclasses, sqlalchemy, app.db.models, app.core.supabase_auth, app.db.session, app.core.auth, app.core.llm.base, app.core.errors, app.config, app.core.logging, fastapi, sqlalchemy.ext.asyncio
- **Key Classes:** WorkspaceContext
- **Key Functions:** request_id_of, bearer_token, get_principal, get_current_principal, require_role, _single_tenant_business, _bind, _resolve_for_principal, resolve_workspace, get_llm_client, get_fast_llm_client, assert_owned, assert_campaign_owned
- **Status:** Fully implemented


## File: app/core/security_headers.py
- **Purpose:** P4.8/P5.9: standard security headers on every response.
- **Dependencies/Imports:** __future__, starlette.types, app.config, app.core.logging, starlette.datastructures
- **Key Classes:** SecurityHeadersMiddleware
- **Key Functions:** build_headers, validate_cors, request_origin, __init__, __call__, send_wrapper
- **Status:** Fully implemented


## File: app/ai/providers/search/none.py
- **Purpose:** Disabled search. Selected when ``AI_SEARCH_PROVIDER=none``.
- **Dependencies/Imports:** app.ai.errors, __future__, app.ai.providers.search.base, app.ai.settings
- **Key Classes:** DisabledSearchProvider
- **Key Functions:** build_search_provider, search
- **Status:** Fully implemented


## File: app/ai/agents/layla/graph.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langgraph.graph, langchain_core.messages, langgraph.prebuilt, typing, app.ai.agents.layla.schema, app.ai.agents.layla.state, langchain_core.tools, app.ai.providers.llm, app.ai.settings, app.ai.tools, langchain_core.runnables
- **Key Classes:** 
- **Key Functions:** submit_creative_brief, call_model, should_continue, post_tools_routing
- **Status:** Fully implemented


## File: app/api/settings.py
- **Purpose:** Workspace settings: ``/api/v1/settings/connectors`` (P2.4).
- **Dependencies/Imports:** app.schemas.llm_settings, uuid, __future__, sqlalchemy, app.db.models, json, app.core.security, app.core.tools.settings, app.core.errors, app.core.logging, app.schemas.settings, fastapi, app.api.deps, app.core.llm.router
- **Key Classes:** 
- **Key Functions:** _require_spec, _render, get_connectors, update_connector, delete_connector, _llm_tier_out, get_llm_settings, update_llm_settings
- **Status:** Fully implemented


## File: app/schemas/recommendations.py
- **Purpose:** Schemas for recommendations and the human-in-the-loop approval flow.
- **Dependencies/Imports:** pydantic, app.schemas.common, __future__, datetime
- **Key Classes:** ChangeDiff, Recommendation, ApproveRecommendationRequest, ApproveRecommendationResponse, RejectRecommendationRequest, RecommendationCreatedByAgent
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/ai/tools/analyze_website.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** langchain_core.runnables.config, bs4, typing, requests, trafilatura, langchain_core.tools, urllib.parse, app.ai.settings, re
- **Key Classes:** 
- **Key Functions:** normalize_url, analyze_website_tool
- **Status:** Fully implemented


## File: app/core/__init__.py
- **Purpose:** No module docstring.
- **Dependencies/Imports:** 
- **Key Classes:** 
- **Key Functions:** 
- **Status:** Fully implemented


## File: app/core/tools/media.py
- **Purpose:** media tools (11-12): reference analysis and image generation.
- **Dependencies/Imports:** app.core.memory.store, pathlib, __future__, sqlalchemy, app.core.errors, datetime, app.core.logging, app.schemas.tools, base64, httpx, app.core.security, app.core.tools.settings, app.core.llm.base, uuid, hashlib, pydantic, app.db.models, app.schemas.creatives, app.schemas.common, typing, app.core.tools.base
- **Key Classes:** _StyleExtraction
- **Key Functions:** _load_reference_image, _guess_mime, _vision_extract, _handle_analyze_reference, _mock_asset, _generate_openai_image, _handle_generate_image, _compose_image_prompt, _resolve_reference
- **Status:** Contains stubs/TODOs
- **Stubs/TODOs found:** 4 instances (e.g. Line 9: A prose description of the refer)
