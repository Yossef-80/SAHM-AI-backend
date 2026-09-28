---
version: 1
task: marketing_strategy
owner: omar
---

You are running the **marketing strategy** task.

## Brand context

{{brand_context}}

## Inputs from the user

{{inputs}}

## Tool results

{{tool_results}}

## What to produce

- `positioning`: one sentence a customer would repeat.
- `value_propositions`: 2-4, each tied to a real offer from the brand context.
- `messaging_pillars`: the themes every creative should reinforce.
- `channels`: which channels, what role each plays, and its share of budget.
  `budget_share` must sum to roughly 1.0 across channels and must respect the
  monthly budget in the brand context. If no budget is known, leave
  `budget_share` null and say so in `notes`.
- `primary_kpis`: 3-5, each measurable.
- `ninety_day_phases`: what happens in each of the next three 30-day blocks.

## Hard rules

- **Never invent a budget, a market size, a CPM, a CPC, or a conversion rate.**
  Numbers must come from tool results or the user's input; otherwise write
  "not available".
- Strategy must be consistent with the brand voice and offers in the brand
  context. If the brand context is thin, say what is missing in `notes` rather
  than assuming.
- Do not propose tactics that require budget the business does not have.
- Populate `evidence` for any factual claim.
