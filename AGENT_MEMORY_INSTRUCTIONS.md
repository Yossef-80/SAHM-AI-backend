# Agent Memory Instructions & Protocol Guide

> **File:** `AGENT_MEMORY_INSTRUCTIONS.md`  
> **Purpose:** Standard operating instructions for any AI coding agent joining this project.  
> **Usage:** Paste this prompt into any new AI agent session or configure it in the agent system prompt to ensure consistent handoff and persistent memory maintenance in `PROJECT_MEMORY.md`.

---

## The System Prompt to Provide to Any New AI Agent

```markdown
You are an AI coding agent working on this repository.

From now on, maintain the persistent project memory file:
`PROJECT_MEMORY.md` (located at the root of the project).

This file is the persistent handoff memory between different AI coding agents. Another agent may take over the project later, so the file must always contain accurate, concise information to continue the work without relying on conversation context.

---

### BEFORE STARTING A TASK:

1. **Read `PROJECT_MEMORY.md` first.**
2. Understand the current state, completed work, known issues, and next steps.
3. Inspect the relevant source files before making assumptions — the repository files are the source of truth.
4. Continue from the existing continuation point; do not redo completed work unnecessarily.
5. Formulate an internal plan:
   `CURRENT STATE → TASK → RELEVANT FILES → DEPENDENCIES → EXPECTED RESULT`

---

### AFTER COMPLETING EVERY TASK:

1. Update `PROJECT_MEMORY.md` with what you actually did.
2. **Modify the existing `PROJECT_MEMORY.md` file — NEVER create a new memory file.**
3. Keep the update concise, technical, and token-efficient.
4. Record only information that another AI agent genuinely needs to continue the project.
5. Always reference real repository file paths (e.g. `app/core/tools/research.py`).
6. Record important implementation decisions, architecture changes, and trade-offs.
7. Record errors or blockers if they exist (never pretend a task succeeded if it failed).
8. Update the current continuation point with the exact next recommended step.
9. Remove or correct obsolete information (update, don't endlessly append history).
10. **NEVER include API keys, passwords, tokens, secrets, or `.env` values.**

---

### CORE SECTIONS TO MAINTAIN IN `PROJECT_MEMORY.md`:

```markdown
# Project Memory

> Persistent memory for AI coding agents working on this project.
> Updated after each completed task.

## 1. Project Goal
<Short description of the product and what we are building>

## 2. Current State
- Status: `IN PROGRESS`
- Current milestone: <active milestone>
- What currently works: <verified working capabilities>
- What is currently incomplete: <pending items>

## 3. Project Structure
Only list files/directories relevant to current work:
- `path/to/file` — <short purpose>

## 4. Completed Work
### Task: <short task name>
- Date/time: <date>
- Changed:
  - `path/to/file` — <what actually changed>
- Result:
  - <what now works>
- Important implementation details:
  - <only details needed by a future agent>

## 5. Important Decisions
- <architectural decision>
- <framework or design pattern chosen>
- <approach rejected and why>

## 6. Requirements / Constraints
- <preservation constraint>
- <security / human-in-the-loop requirement>

## 7. Environment
- OS: <operating system>
- Python / Node: <versions>
- Package managers: <uv / npm>
- Important dependencies: <frameworks>

## 8. Known Issues / Blockers
- `<issue description>` — `path/to/file` — <current status>
(or `None currently known.`)

## 9. Important Commands
```bash
# test command
# run dev server
```

## 10. Current Continuation Point
### Last Completed Task
<one concise paragraph of what was just verified>

### Current Blocker
<blocker or None>

### Next Recommended Step
<exact next command, file to edit, or feature to build>

### Expected Result
<what should happen when the next step is executed>

## 11. Agent Notes
- <critical context for incoming AI agents>
```

---

### OPERATIONAL RULES FOR AI AGENTS:

- **Do NOT write a conversation transcript.**
- **Do NOT explain basic concepts.**
- **Do NOT copy large blocks of code.**
- **Do NOT repeat information already present unless it changed.**
- **Do NOT invent work that was not completed.**
- **Do NOT mark a task complete if it was only partially completed.**
- **Keep the memory file short and token-efficient so a new agent can read it quickly.**
- **The project files are the source of truth. `PROJECT_MEMORY.md` is only a concise persistent map of the project state.**
```

---

## Fast Agent Onboarding Checklist

When an agent takes over this repository, it should follow this 3-step sequence:

```
Step 1: Read Memory
  └─► view_file("d:/NTI-Final-Project/PROJECT_MEMORY.md")

Step 2: Inspect Continuation Point & Relevant Files
  └─► Identify "Next Recommended Step"
  └─► Verify source code against the memory map

Step 3: Execute Task & Update Memory
  └─► Perform implementation / debugging
  └─► Verify tests / execution
  └─► Update PROJECT_MEMORY.md with what was actually accomplished
```

---

## Concrete Example: Good vs. Bad Memory Updates

### ❌ Bad Update (Verbose, transcript-like, vague paths):
> "I had a chat with the user about building agents. We discussed using LangGraph and then I edited some code in the agent file and everything looks good. I tried running tests and maybe there's a database error."

### ✅ Good Update (Precise, technical, real paths, clear continuation point):
> **Completed Work**  
> `Task: Add SSRF Protection to Web Scraping Tool`  
> - Changed:  
>   - `app/core/tools/research.py` — added `resolve_and_check(url)` to validate outbound domains against private IPv4/IPv6 blocks and cloud metadata IP (`169.254.169.254`).  
> - Result:  
>   - `AnalyzeWebsiteTool` now gracefully catches `BlockedTargetError` and returns `ToolOutput.unavailable` instead of throwing unhandled exceptions.  
>  
> **Current Continuation Point**  
> - *Last Completed Task:* Implemented SSRF DNS resolution guard in `research.py` and passed unit tests in `tests/test_tools.py`.  
> - *Current Blocker:* None.  
> - *Next Recommended Step:* Implement `GetSocialProfileTool` in `app/core/tools/research.py` to extract follower metrics and post themes.  
> - *Expected Result:* Tool returns structured `SocialProfileOutput` matching Zod schema.
