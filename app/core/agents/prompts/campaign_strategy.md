---
version: 1
task: campaign_strategy
owner: omar
---

You are running the **campaign strategy** task.

## Brand context

{{brand_context}}

## Inputs from the user

{{inputs}}

## Tool results

{{tool_results}}

## What to produce

- `objective`: the single outcome this campaign is buying.
- `hypothesis`: "If we X for Y, then Z, because ..." -- falsifiable.
- `audiences`: who we are targeting and why they are reachable.
- `budget_split`: allocation by channel. Shares must sum to roughly 1.0 and must
  respect the campaign budget. If the budget is unknown, leave `budget_share` null
  and explain in `notes`.
- `creatives_needed`: the specific creative assets required, with enough detail
  that the creative persona can brief them without guessing.
- `timeline_weeks`: an integer, only if the user gave a duration.
- `success_metrics`: the numbers that will decide whether this worked, with the
  target values the user or a tool result provided.

## Hard rules

- **Never invent a budget, a target ROAS, a CPM, or a conversion rate.** If the
  user did not give a number, write "not available" and put the question in
  `notes`.
- Every creative in `creatives_needed` must follow from the strategy above, not
  from generic best practice.
- Populate `evidence` for any factual claim about the market or past performance.
