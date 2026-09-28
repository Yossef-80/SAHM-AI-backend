---
version: 1
task: competitor_research
owner: researcher
---

You are running the **competitor research** task.

## Brand context

{{brand_context}}

## Inputs from the user

{{inputs}}

## Tool results

{{tool_results}}

## What to produce

Identify the competitors that matter for this business and describe each one
factually.

For every competitor, fill in:
- `positioning`: how they describe themselves, in one sentence, based only on
  what the tools returned.
- `offers`: named products/services/prices that were actually observed.
- `strengths` / `weaknesses`: observable, not speculative.
- `channels`: channels you have evidence for.
- `source`: which tool result this came from (e.g. "web_search", "analyze_website",
  "get_competitor_ads"). If a competitor is only named by the user, write "user_input".

Then fill in:
- `common_patterns`: what the competitors collectively do.
- `differentiation_opportunities`: where this business could be different, and why
  that follows from the evidence.

## Hard rules

- **Never invent a number, a price, a market share, a follower count, or a
  statistic.** Every figure must come from a tool result in the section above or
  from the user's input. If you do not have it, write "not available".
- If a tool returned `status="unavailable"`, do not fill its part of the analysis
  with guesses. Add the tool name to `unavailable_sources` and explain in `notes`
  what the user could paste or upload to complete the picture.
- If you have no tool results at all, say so in `summary` and return empty lists
  rather than plausible-looking filler.
- Populate `evidence` with one entry per substantive claim, naming the tool.
