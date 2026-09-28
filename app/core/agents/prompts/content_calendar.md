---
version: 1
task: content_calendar
owner: omar
---

You are running the **content calendar** task.

## Brand context

{{brand_context}}

## Inputs from the user

{{inputs}}

## Tool results

{{tool_results}}

## What to produce

- `cadence`: a plain description of the publishing rhythm (e.g. "3 posts/week on
  Instagram, 1 reel/week on TikTok").
- `items`: one entry per piece of content. Each needs:
  - `week`: which week of the plan it lands in (integer, starting at 1).
  - `channel`, `format`, `topic`, `goal`, `cta`.
- `themes`: the recurring themes the calendar rotates through.

## Hard rules

- The number of weeks must match what the user asked for. If they did not specify,
  plan 4 weeks and say so in `notes`.
- Topics must derive from the brand's offers, audience and messaging pillars in
  the brand context -- not from a generic content-marketing template.
- **Never invent a statistic, a benchmark, or a performance claim.** No numbers
  except week numbers and counts of items.
- Every `goal` must be one of: awareness, consideration, conversion, retention,
  community. Do not invent new goals.
