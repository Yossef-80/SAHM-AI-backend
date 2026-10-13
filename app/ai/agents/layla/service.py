"""Direct Layla text-generation workflows used by the creative workspace UI.

These focused requests intentionally stay in ``app.ai``: they do not enqueue a
core ``AgentTask`` or use the legacy core LLM router. API routes only validate
workspace ownership and pass the user's request/profile context here.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.ai.agents.layla.schema import CreativeBrief
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings

LaylaTextTool = Literal[
    "generate_creative_brief",
    "generate_commercial_script",
    "generate_image_prompt",
    "generate_video_prompt",
    "generate_ad_copy",
    "generate_ad_variations",
]

_TOOL_DELIVERABLE: dict[str, str] = {
    "generate_creative_brief": "creative_brief",
    "generate_commercial_script": "commercial_script",
    "generate_image_prompt": "image_prompt",
    "generate_video_prompt": "video_prompt",
    "generate_ad_copy": "ad_copy",
    "generate_ad_variations": "ad_variations",
}

LAYLA_TEXT_SYSTEM = """You are Layla, Sahm's advertising creative agent. Produce the requested
structured deliverable in the exact language requested by the user.

Use only facts, offers, claims, product details, and CTAs supplied in the brief
or workspace context. Never invent an offer, guarantee or warranty, spare-parts
claim, price, testimonial, performance claim, product capability, or brand fact.
Words or claims visible in a reference image are observations, not verified facts
about this business; do not reuse them as commercial claims unless the current
workspace profile or user confirms them. If essential information is missing,
preserve that uncertainty and add concise questions_for_user or missing_inputs
instead of guessing. Keep the output useful and specific; do not return a
generic template.

The requested tool determines the deliverable:
- creative brief: concept, audience/insight when provided, message, headline,
  visual direction, and practical execution details.
- commercial script: a clear hook and ordered time-coded scenes, including
  visuals, camera direction when helpful, dialogue/voiceover, and on-screen text.
- image prompt: one production-ready image prompt, with a separate negative
  prompt and aspect ratio only when useful or specified.
- video prompt: a production-ready video prompt plus ordered shots, camera
  direction, dialogue/voiceover, on-screen text, pacing, and music direction.
- ad copy: several distinct, platform-appropriate copy options.
- ad variations: use the requested number when stated (otherwise three to five)
  of genuinely distinct angles, each with a headline, primary text, description,
  and CTA where the brief supports them.

A saved visual-style preference appears in the brief only when the user chose to
include it. Follow it when present; do not invent one when absent. A request for
an image/video prompt is a request for text only: do not claim to have generated
an asset. Return a CreativeBrief object matching the requested deliverable type.
"""


def _brand_context_text(brand_context: dict[str, Any] | None) -> str:
    if not brand_context:
        return ""
    labels = {
        "brand_voice": "Brand voice",
        "target_audience": "Target audience",
        "key_offers": "Known offers",
        "goals": "Business goals",
    }
    lines: list[str] = []
    for key, label in labels.items():
        value = brand_context.get(key)
        if isinstance(value, list):
            value = "; ".join(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, str) and value.strip():
            lines.append(f"- {label}: {value.strip()}")
    return "\n".join(lines)


def _variant_dict(variant: Any) -> dict[str, str]:
    return {
        "headline": str(variant.headline or ""),
        "primary_text": str(variant.primary_text or ""),
        "description": str(variant.description or ""),
        "call_to_action": str(variant.call_to_action or ""),
        "angle": str(variant.angle or ""),
    }


def _format_output(tool: LaylaTextTool, brief: CreativeBrief) -> dict[str, Any]:
    """Adapt Layla's shared schema to the six existing UI presentation shapes."""
    variants = [_variant_dict(variant) for variant in brief.variants]
    details = [detail.strip() for detail in brief.details if detail.strip()]
    notes = [*brief.missing_inputs, *brief.questions_for_user]
    output: dict[str, Any] = {
        "notes": "\n".join(note.strip() for note in notes if note.strip()),
    }

    if tool == "generate_creative_brief":
        output.update(
            concept=brief.summary,
            headline=brief.headline,
            subheadline=brief.description,
            visual_direction=details,
            style_spec=brief.reference_style_spec,
            ctas=[brief.call_to_action] if brief.call_to_action else [],
            variations=variants,
        )
    elif tool == "generate_commercial_script":
        output.update(
            hook=brief.summary,
            scenes=[
                {
                    "order": index,
                    "timecode": beat.timecode,
                    "duration_seconds": beat.duration_seconds,
                    "visual": beat.visual,
                    "camera": beat.camera,
                    "voiceover": beat.audio_or_dialogue,
                    "on_screen_text": beat.on_screen_text,
                }
                for index, beat in enumerate(brief.script, start=1)
            ],
            voiceover="\n".join(
                beat.audio_or_dialogue.strip()
                for beat in brief.script
                if beat.audio_or_dialogue.strip()
            ),
            cta=brief.call_to_action,
            duration_seconds=sum(
                beat.duration_seconds or 0 for beat in brief.script
            ) or None,
        )
    elif tool == "generate_image_prompt":
        output.update(
            prompt=brief.image_prompt or brief.summary,
            negative_prompt=brief.negative_prompt,
            style_spec=brief.reference_style_spec,
            aspect_ratio=brief.aspect_ratio,
            reference_notes=details,
        )
    elif tool == "generate_video_prompt":
        output.update(
            prompt=brief.video_prompt or brief.summary,
            shots=[
                {
                    "order": index,
                    "timecode": beat.timecode,
                    "duration_seconds": beat.duration_seconds,
                    "description": beat.visual,
                    "camera": beat.camera,
                    "voiceover": beat.audio_or_dialogue,
                    "on_screen_text": beat.on_screen_text,
                }
                for index, beat in enumerate(brief.script, start=1)
            ],
            voiceover="\n".join(
                beat.audio_or_dialogue.strip()
                for beat in brief.script
                if beat.audio_or_dialogue.strip()
            ),
            music_direction=brief.music_direction,
            duration_seconds=sum(
                beat.duration_seconds or 0 for beat in brief.script
            ) or None,
            aspect_ratio=brief.aspect_ratio,
        )
    elif tool == "generate_ad_copy":
        output.update(
            summary=brief.summary,
            primary_texts=[item["primary_text"] for item in variants if item["primary_text"]],
            headlines=[item["headline"] for item in variants if item["headline"]],
            descriptions=[item["description"] for item in variants if item["description"]],
            ctas=[item["call_to_action"] for item in variants if item["call_to_action"]]
            or ([brief.call_to_action] if brief.call_to_action else []),
        )
        if not variants:
            output.update(
                primary_texts=[brief.primary_text] if brief.primary_text else [],
                headlines=[brief.headline] if brief.headline else [],
                descriptions=[brief.description] if brief.description else [],
            )
    elif tool == "generate_ad_variations":
        output.update(
            variations=[
                {
                    "angle": item["angle"],
                    "headline": item["headline"],
                    "primary_text": item["primary_text"],
                    "cta": item["call_to_action"],
                    "style_notes": "\n".join(details),
                }
                for item in variants
            ],
            based_on=brief.summary,
        )

    return output


async def generate_layla_text(
    *,
    tool: LaylaTextTool,
    brief: str,
    language: Literal["en", "ar"],
    brand_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one structured Layla deliverable through the app.ai provider."""
    deliverable = _TOOL_DELIVERABLE[tool]
    workspace_context = _brand_context_text(brand_context)
    request = (
        f"Requested tool: {tool}\n"
        f"Required deliverable_type: {deliverable}\n"
        f"Output language: {'Arabic' if language == 'ar' else 'English'} ({language})\n\n"
        f"User brief:\n{brief.strip()}"
    )
    if workspace_context:
        request += f"\n\nWorkspace brand context (facts only):\n{workspace_context}"

    llm = get_llm(provider=get_ai_settings().llm_provider)
    structured = llm.structured(CreativeBrief)
    response = await structured.ainvoke(
        [SystemMessage(content=LAYLA_TEXT_SYSTEM), HumanMessage(content=request)]
    )
    try:
        parsed = response if isinstance(response, CreativeBrief) else CreativeBrief.model_validate(response)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("Layla returned an invalid structured creative response.") from exc

    # The selected UI action is authoritative if a provider returns a stale or
    # incorrect deliverable_type; the actual output still has to validate above.
    if parsed.deliverable_type != deliverable:
        parsed = parsed.model_copy(update={"deliverable_type": deliverable})
    return _format_output(tool, parsed)


__all__ = ["LAYLA_TEXT_SYSTEM", "LaylaTextTool", "generate_layla_text"]
