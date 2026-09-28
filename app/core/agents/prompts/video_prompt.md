---
version: 1
task: video_prompt
owner: layla
---

You are running the **video prompt** task.

## Brand context

{{brand_context}}

## Inputs from the user

{{inputs}}

## Tool results

{{tool_results}}

## What to produce

- `prompt`: the master video-generation prompt.
- `shots`: an ordered shot list with `order`, `description`, `camera`, and
  `duration_seconds`.
- `voiceover`: the spoken script.
- `music_direction`: mood and tempo, no specific copyrighted tracks.
- `duration_seconds`: total runtime, matching the user's request if given.
- `aspect_ratio`: 9:16 for reels/shorts/tiktok, 1:1 or 4:5 for feed, 16:9 for
  YouTube. Use the placement from the inputs if given.

## Hard rules

- Shot durations must sum to `duration_seconds`.
- **Never invent a price, statistic, testimonial, or product claim.**
- The voiceover must be speakable and free of placeholder brackets.
- Do not name real songs, artists, or trademarks you were not given.
