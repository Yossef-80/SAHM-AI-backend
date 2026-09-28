---
version: 1
task: commercial_script
owner: layla
---

You are running the **commercial script** task.

## Brand context

{{brand_context}}

## Inputs from the user

{{inputs}}

## Tool results

{{tool_results}}

## What to produce

- `hook`: the first 3 seconds. One line, in the brand's voice.
- `scenes`: an ordered shot list. Each scene needs `order` (starting at 1),
  `visual`, `voiceover`, `on_screen_text`, and `duration_seconds`.
- `voiceover`: the complete voiceover script.
- `cta`: the closing call to action.
- `duration_seconds`: the total runtime. If the user gave a duration, match it
  exactly; otherwise target 30 seconds and say so in `notes`.

## Hard rules

- Scene durations must add up to `duration_seconds`. Do not leave gaps.
- **Never invent a price, a discount, a statistic, a customer count, or an award.**
  Only claims present in the brand context, tool results, or user input.
- The voiceover must be speakable: short sentences, no invented brand names, no
  placeholder brackets.
- Keep on-screen text under 8 words per scene.
