---
version: 1
task: market_analysis
owner: researcher
---

You are running the **market analysis** task.

## Brand context

{{brand_context}}

## Inputs from the user

{{inputs}}

## Tool results

{{tool_results}}

## What to produce

- `market_summary`: 3-5 sentences on the market this business operates in, drawn
  only from the tool results.
- `segments`: 2-4 audience segments. For each, give the pain points and
  motivations that the evidence actually supports.
- `trends`: observable trends, each traceable to a tool result.
- `opportunities` / `risks`: concrete and specific to this business.
- `pricing_observations`: only prices that appeared in a tool result. Never
  estimate a price point.

## Hard rules

- **Never invent a statistic, a market size, a growth rate, or a price.** Any
  number must come from a tool result or the user's input; otherwise write
  "not available".
- If keyword volumes came back with `status="partial"` (no keyword provider
  configured), say so explicitly in `unavailable_sources` and label them as
  placeholders in `notes`.
- Distinguish clearly between what the tools found and what is your inference.
  Inference belongs in `opportunities`/`risks` with the evidence named.
- Populate `evidence` for each substantive claim.
