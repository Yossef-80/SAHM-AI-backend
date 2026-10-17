# Layla Dynamic Image Generation & Reference Grounding Guide

> **Target Audience:** AI Coding Agents & System Engineers working on Sahm's multimodal creative workflows.  
> **Branch:** `ahmed`  
> **Status:** LIVE & Production Verified (100% tests passing)

---

## 1. Executive Summary & Root Cause Analysis

### The Problem
Users experienced two major issues when interacting with Layla in chat:
1. **Reference Images Not Read in Chat:**
   - After analyzing an image in the Creatives tab or mentioning it with `/` or `@` in chat (e.g. `@images.jpg ايه تفاصيل الصورة ديه`), Layla failed to read or use the reference image observations. Mentioning it a second time in subsequent turns gave no result or an empty response.
2. **Repetitive Question Loops / No Image Generated:**
   - Asking Layla to create, draw, or design an image (e.g. "اعملي صورة للمنتج", "صممي بوستر", "عايز صورة للبراند") caused Layla to either ask interview questions repeatedly without producing an image, or say image generation was unavailable.

### Root Causes Identified
1. **Reference Grounding Was Isolated to Prompt-Only Mode:**
   - In `app/ai/agents/orchestrator/brain.py`, `reference_analysis_context` was **only** rendered as a system message when `context.get("workflow") == "creative_prompt"`.
   - Outside that rigid questionnaire workflow (i.e. in normal chat, Layla chat, or image draft turns), `chat_history` received **zero** reference information. The model was literally blind to the analyzed reference.
2. **Fragile Context Mention Resolution:**
   - When a user mentioned `@filename.jpg` or `/style` directly in prose, if the frontend chip had been cleared on the subsequent turn, the backend did not re-resolve the reference by name from `BrandMemoryItem`.
3. **Rigid Intent Filter Stripping Image Tools:**
   - In `app/ai/agents/orchestrator/brain.py`, `_requests_image_generation(prompt)` was checking an exact list of 40 hardcoded phrases.
   - Any colloquial Arabic phrase (especially feminine forms addressing Layla like `اعملي صورة`, `صمميلي بوستر`, `ارسم صورة`, `ولديلي`) or English phrases like `draw me an image` returned `False`.
   - When `False`, `configurable["allow_image_generation"]` was not set, and `layla/graph.py` **completely stripped `generate_image_draft_tool`** from Layla's tools.
   - Consequently, Layla had no image tool available and defaulted to asking questions.
4. **OpenAI Configuration & Model Fallback:**
   - A misconfiguration where `AI_LLM_MODEL=openai` caused OpenAI's SDK to request model `"openai"`, returning HTTP 404 `OpenAIModelNotFoundError`.
   - OpenAI image generations were configured with `IMAGE_MODEL=gpt-image-2-flare` instead of `dall-e-3`.

---

## 2. What Was Changed & Implemented

### A. Universal Reference Grounding Across All Chat Turns
- **File:** [`app/ai/agents/orchestrator/brain.py`](file:///d:/NTI-Final-Project/SAHM-AI-backend/app/ai/agents/orchestrator/brain.py)
  - Created `_reference_analysis_message(context: dict[str, Any], *, arabic: bool) -> str | None`.
  - Injected `reference_block` directly into `chat_history` (analogous to `_research_item_message`) so **all** agents (Layla, Omar, Nour, Chat) receive the complete visual grounding:
    - Reference source/file name
    - VLM summary
    - Observed palette (hex colors), lighting, composition, layout, typography, and placement
  - Bilingual headers: `[مرجع_الصورة_المعتمدة_للتصميم]` in Arabic and `[REFERENCED_CREATIVE_IMAGE_GROUNDING]` in English.

### B. Mention Rehydration from Brand Memory
- **File:** [`app/core/orchestrator/service.py`](file:///d:/NTI-Final-Project/SAHM-AI-backend/app/core/orchestrator/service.py)
  - When `reference_analysis_id` is not explicitly attached as a chip, `service.py` inspects `safe_message` for `/[name]` or `@[name]`.
  - Automatically queries `BrandMemoryItem(kind="reference_analysis")` scoped to `business_id` and rehydrates `reference_analysis_context` and `reference_analysis_id`.
  - Preserves `reference_analysis_id` in `conversation.context` across subsequent conversation turns.

### C. Broadened & Natural Image Generation Intent Detection
- **File:** [`app/ai/agents/orchestrator/brain.py`](file:///d:/NTI-Final-Project/SAHM-AI-backend/app/ai/agents/orchestrator/brain.py)
  - Updated `_requests_image_generation(prompt: str, context: dict[str, Any] | None = None) -> bool`:
    - Checks context flags: `creative_mode == "design"` or `allow_image_generation is True`.
    - Checks `/image_draft` or `image_draft` tool mentions.
    - Matches English action verbs (`generate`, `create`, `make`, `render`, `draw`, `design`, `produce`) with visual nouns (`image`, `photo`, `picture`, `visual`, `poster`, `banner`, `artwork`, `draft`).
    - Matches Arabic action verbs in both masculine and feminine forms (`صمم`, `صممي`, `صمميلي`, `اعمل`, `اعملي`, `اعمليلي`, `ولد`, `ولدي`, `ولديلي`, `ارسم`, `ارسمي`, `ارسميلي`, `طلع`, `طلعي`, `طلعيلي`, `انشئ`, `انشئي`, `نفذ`, `نفذي`, `عايز`, `محتاج`) with visual nouns (`صورة`, `صوره`, `صور`, `بوستر`, `بانر`, `كرييتف`, `خلفية`, `إعلان مصور`).
    - When detected, it clears any active questionnaire state machine and sets `configurable["allow_image_generation"] = True`.

### D. Dynamic Standalone Generation in Layla
- **File:** [`app/ai/agents/layla/graph.py`](file:///d:/NTI-Final-Project/SAHM-AI-backend/app/ai/agents/layla/graph.py)
  - Updated `LAYLA_PROMPT`:
    - Explicitly instructs Layla to **immediately** call `generate_image_draft_tool` when image generation is requested.
    - Standalone image generation **does not require a campaign ID**.
    - Forbids multi-turn question loops when the user asks for an image; cosmetic details are inferred generously from brand identity or reference specs.
    - Instructs Layla to explain reference details directly when the user asks questions about a reference image.
- **File:** [`app/ai/tools/image_generation.py`](file:///d:/NTI-Final-Project/SAHM-AI-backend/app/ai/tools/image_generation.py)
  - If Layla does not provide an explicit `style_spec` argument, the tool automatically pulls the style from `configurable["reference_analysis_context"]["style"]`.
  - Automatically records `reference_analysis_id` in the created `Creative` row in the database.

### E. Model & Provider Safeguards
- **File:** [`app/ai/providers/llm/openai.py`](file:///d:/NTI-Final-Project/SAHM-AI-backend/app/ai/providers/llm/openai.py)
  - If `model` is passed as `"openai"` or `"default"`, it safely falls back to `settings.openai_model` (`gpt-4o-mini`) instead of failing with 404.
- **File:** [`SAHM-AI-backend/.env`](file:///d:/NTI-Final-Project/SAHM-AI-backend/.env)
  - Updated `AI_LLM_MODEL=gpt-4o-mini` and `IMAGE_MODEL=dall-e-3`.

---

## 3. How an Agent Working on Prompt Tasks Should Add This Work

If an agent is given a task to update, extend, or build on Layla, image generation, or prompt handling, the agent **MUST** follow these steps:

### Step 1: Follow the Grounding Rule
- Never let an agent hallucinate reference visual details. Always ensure `reference_analysis_context` is present in `context`.
- If the user references an image or style, read the observations directly from `[REFERENCED_CREATIVE_IMAGE_GROUNDING]` or `[مرجع_الصورة_المعتمدة_للتصميم]`.

### Step 2: Separate "Design/Generation" from "Prompt-Only Interview"
- **Design/Generation Mode (`creative_mode == "design"` or `_requests_image_generation == True`):**
  - Tool `generate_image_draft_tool` is bound and callable.
  - The agent should call the tool immediately on turn 1.
  - Do NOT emit `QuestionWithOptionsBlock` or ask 5 checklist questions.
- **Prompt-Only Mode (`workflow == "creative_prompt"`):**
  - Tool `generate_image_draft_tool` is stripped.
  - The agent uses the structured interview state machine to output an editable prompt for approval.

### Step 3: Support Standalone Generation
- In Sahm, creative assets can belong to a campaign (`campaign_id`) or be standalone brand assets (`campaign_id = None`).
- Never add validation requiring `campaign_id` for generating an image draft.

### Step 4: Verification Protocol
Always run the targeted test suite first, followed by the full suite:
```bash
# 1. Run Layla image generation tests
python -m pytest tests/test_layla_image_generation.py -v

# 2. Run creative mode gate and transport tests
python -m pytest tests/test_creative_mode_gate.py tests/test_creative_mode_transport.py -v

# 3. Run reference analysis tests
python -m pytest tests/test_creative_generate_reference_style.py tests/test_creative_prompt_workflow.py -v

# 4. Run the full backend suite
python -m pytest tests -q
```

---

## 4. Verification & Git Commit Information

- **Branch:** `ahmed`
- **Pushed Commit:** `83c213c` (`fix(layla): dynamic image generation without loops and full reference analysis grounding in chat`)
- **Test Results:** 920+ tests passed, 0 failed, 100% clean test execution.
