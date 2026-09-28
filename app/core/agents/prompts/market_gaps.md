---
version: 1
task: market_gaps
owner: researcher
---

You are running the **market gaps** task.

## Brand context

{{brand_context}}

## Inputs from the user

{{inputs}}

## Tool results

{{tool_results}}

## What to produce

- `gaps`: 3-6 unmet needs or underserved angles. For each gap:
  - `title`: short and specific.
  - `detail`: what the gap is and who feels it.
  - `who_it_serves`: the segment.
  - `evidence_source`: which tool result supports this gap existing.
  - `opportunity_size`: **only** fill this in when a tool result or the user gave
    you a number. Otherwise leave it as "not available". Never estimate it.
- `recommended_focus`: the 1-3 gaps this business should pursue first, and why
  that follows from its own brand context (offers, audience, budget).

## Hard rules

- **Never invent a market size, revenue estimate, or percentage.** If it is not in
  a tool result or the user's input, write "not available".
- A gap with no supporting evidence is not a gap. If the evidence is thin, say so
  in `notes` and keep the gap list short.
- If a research tool returned `status="unavailable"`, list it in
  `unavailable_sources` and tell the user in `notes` what to paste to close it.
- Populate `evidence` for each gap.
