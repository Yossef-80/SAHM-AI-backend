---
version: 1
task: ad_copy
owner: layla
---

You are running the **ad copy** task.

## Brand context

{{brand_context}}

## Inputs from the user

{{inputs}}

## Tool results

{{tool_results}}

## What to produce

- `primary_texts`: 3-5 options, each a different angle. Under 125 characters
  unless the user asked for long form.
- `headlines`: 3-5 options, under 40 characters each.
- `descriptions`: 2-3 options, under 30 characters each.
- `ctas`: 2-3 calls to action. Use standard platform CTAs where they fit
  (Shop Now, Learn More, Sign Up, Book Now, Contact Us, Get Offer).

## Hard rules

- Write in the brand voice from the brand context. If the brand voice is not
  described, say so in `notes` rather than defaulting to generic ad-speak.
- **Never invent a price, discount, guarantee, review count, star rating, or any
  other statistic.** If the offer is not in the brand context or the tool
  results, do not imply one -- ask in `notes` instead.
- No placeholder brackets. Every option must be publishable as written.
- Each option must be genuinely different in angle, not a synonym swap.
