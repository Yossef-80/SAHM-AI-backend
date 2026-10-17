"""Prompt construction for reference-faithful ad creative generation."""

from __future__ import annotations

from typing import Any


def _as_text(value: Any, *, limit: int = 800) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _format_elements(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    lines: list[str] = []
    for index, item in enumerate(value[:24], 1):
        if not isinstance(item, dict):
            continue
        fields = [
            ("element", "Element"),
            ("relative_position", "Position"),
            ("relative_size", "Relative size"),
            ("visual_attributes", "Appearance"),
            ("exact_text", "Visible text"),
            ("layer_order", "Layer order"),
        ]
        details = [
            f"{label}: {str(item[key])[:250]}"
            for key, label in fields
            if item.get(key) not in (None, "", [])
        ]
        if details:
            lines.append(f"{index}. " + "; ".join(details))
    return lines


def compose_reference_faithful_prompt(prompt: str, style: dict[str, Any]) -> str:
    """Ask for a close visual recreation while retaining the original as input.

    The image itself is supplied separately to the image-edit endpoint. This text
    steers layout/text fidelity but never substitutes for, or overwrites, that
    reference file. The generated output is persisted as a different asset.
    """
    instruction = [
        "REFERENCE-MATCH MODE — create a new final advertising image using the attached reference image as the visual source.",
        "Match the reference as closely as the image model can: keep the same canvas orientation, layout, number of major elements, relative positions and sizes, spacing, alignment, crop, camera/viewpoint, background, lighting, color relationships, typography appearance, and visual hierarchy.",
        "Preserve every legible word from the reference verbatim, in the same language, capitalization, line breaks, and approximate position. Do not add new wording, badges, logos, objects, or decorative elements. If reference text is unreadable, do not invent replacement text.",
        "Only change items explicitly requested in the brief below. Treat all other visible design choices as locked. Do not turn this into a loose style inspiration or a different concept.",
        "Return a distinct generated final image; do not return, link to, or overwrite the reference file.",
        "USER BRIEF (the only requested changes):",
        _as_text(prompt, limit=3000),
    ]

    if style:
        p_lower = prompt.lower()
        color_override = any(
            k in p_lower
            for k in (
                "لون", "الوان", "ألوان", "خلفية", "الخلفية", "براند", "هوية",
                "color", "colour", "palette", "background", "brand",
            )
        )
        observations: list[str] = []
        for key, label in (
            ("canvas_ratio", "Canvas ratio"),
            ("background", "Background"),
            ("composition", "Composition"),
            ("layout_pattern", "Layout pattern"),
            ("typography", "Typography appearance"),
            ("tone", "Tone"),
            ("color_mood", "Color mood"),
            ("text_density", "Text density"),
            ("lighting", "Lighting"),
        ):
            if color_override and key in {"background", "color_mood"}:
                continue
            value = _as_text(style.get(key))
            if value:
                observations.append(f"- {label}: {value}")
        if not color_override:
            palette = style.get("palette")
            if isinstance(palette, list) and palette:
                colors = [str(color)[:40] for color in palette[:16] if str(color).strip()]
                if colors:
                    observations.append("- Palette, preserve order/relative prominence: " + ", ".join(colors))
        else:
            observations.append("- Color & Background: STRICT OVERRIDE — follow the user brief's specified brand colors and background completely; do NOT use reference colors or reference background.")
        visible_text = style.get("visible_text")
        if isinstance(visible_text, list) and visible_text:
            exact = [str(text)[:300] for text in visible_text[:24] if str(text).strip()]
            if exact:
                observations.append("- Exact visible text, preserve verbatim: " + " | ".join(exact))
        elements = _format_elements(style.get("elements"))
        if elements:
            observations.append("- Element-by-element layout notes:")
            observations.extend(f"  {line}" for line in elements)
        brands = style.get("brand_elements")
        if isinstance(brands, list) and brands:
            observations.append("- Visible brand elements to retain only as shown: " + ", ".join(str(x)[:100] for x in brands[:16]))
        if observations:
            instruction.extend(["STRUCTURED REFERENCE OBSERVATIONS (secondary to the attached image):", *observations])

    return "\n".join(instruction)


__all__ = ["compose_reference_faithful_prompt"]
