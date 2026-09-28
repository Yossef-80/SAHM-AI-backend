---
version: 1
task: ad_variations
owner: layla
---

You are running the **ad variations** task.

## Brand context

{{brand_context}}

## Inputs from the user

{{inputs}}

## Tool results

{{tool_results}}

## What to produce

- `variations`: 3-5 distinct ad variations. Each needs:
  - `angle`: the single idea being tested (one phrase).
  - `headline`, `primary_text`, `cta`: publishable copy, no placeholders.
  - `style_notes`: how the visual should differ for this angle.
- `based_on`: which existing creative or performance finding these variations
  respond to, taken from the inputs or tool results.

## Hard rules

- Each variation must test a genuinely different angle (e.g. price, social proof,
  urgency, problem/solution, identity) -- not a rephrasing.
- **Never invent a price, discount, statistic, testimonial, or review count.**
  Only what the brand context, tool results, or user input establishes.
- If a reference analysis is present in the tool results, its structured style
  spec governs `style_notes`; do not describe the reference in prose.
- If performance data is present, variations should respond to what it showed.
  If it is absent, say so in `unavailable_sources`.
