"""Layer 3: TaskSpecs. This is where most of the product lives.

A TaskSpec binds a prompt template (loaded from ``prompts/``, versioned, with
named slots) to an output schema and a tier. Adding a capability means adding a
template + an output model + one TaskSpec entry -- no new agent class.

Every output model inherits ``TaskOutputBase``, which carries ``evidence`` and
``unavailable_sources``. That is the structural counterpart to the prompt rule
"never invent a statistic": if a number did not come from a tool result or from
the user, it must not appear, and any source that could not be reached has to be
declared here so the runtime can surface it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.schemas.common import Evidence, Tier

_logger = get_logger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

#: Placeholder syntax inside templates. Deliberately not ``{}`` so JSON
#: examples and CSS-ish text in a prompt cannot collide with formatting.
SLOT_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


# ------------------------------------------------------------ output models --


class TaskOutputBase(BaseModel):
    """Fields every task output carries, whatever the task."""

    summary: str = ""
    #: Where each claim came from. Tool name + detail.
    evidence: list[Evidence] = Field(default_factory=list)
    #: Sources that could not be reached. The runtime surfaces these to the user.
    unavailable_sources: list[str] = Field(default_factory=list)
    notes: str = ""


# -- researcher ----------------------------------------------------------------


class CompetitorProfile(BaseModel):
    name: str
    positioning: str = ""
    offers: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    source: str = ""


class CompetitorResearchOutput(TaskOutputBase):
    competitors: list[CompetitorProfile] = Field(default_factory=list)
    common_patterns: list[str] = Field(default_factory=list)
    differentiation_opportunities: list[str] = Field(default_factory=list)


class AudienceSegment(BaseModel):
    name: str
    description: str = ""
    pain_points: list[str] = Field(default_factory=list)
    motivations: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)


class MarketAnalysisOutput(TaskOutputBase):
    market_summary: str = ""
    segments: list[AudienceSegment] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    pricing_observations: list[str] = Field(default_factory=list)


class Gap(BaseModel):
    title: str
    detail: str = ""
    who_it_serves: str = ""
    evidence_source: str = ""
    opportunity_size: str = Field(
        default="not available",
        description="Only filled when a tool result or user input provides it.",
    )


class MarketGapsOutput(TaskOutputBase):
    gaps: list[Gap] = Field(default_factory=list)
    recommended_focus: list[str] = Field(default_factory=list)


# -- strategist ----------------------------------------------------------------


class ChannelPlan(BaseModel):
    channel: str
    role: str = ""
    budget_share: float | None = Field(default=None, ge=0.0, le=1.0)
    kpis: list[str] = Field(default_factory=list)


class MarketingStrategyOutput(TaskOutputBase):
    positioning: str = ""
    value_propositions: list[str] = Field(default_factory=list)
    messaging_pillars: list[str] = Field(default_factory=list)
    channels: list[ChannelPlan] = Field(default_factory=list)
    primary_kpis: list[str] = Field(default_factory=list)
    ninety_day_phases: list[str] = Field(default_factory=list)


class CampaignStrategyOutput(TaskOutputBase):
    objective: str = ""
    hypothesis: str = ""
    audiences: list[str] = Field(default_factory=list)
    budget_split: list[ChannelPlan] = Field(default_factory=list)
    creatives_needed: list[str] = Field(default_factory=list)
    timeline_weeks: int | None = None
    success_metrics: list[str] = Field(default_factory=list)


class ContentItem(BaseModel):
    week: int
    channel: str
    format: str = ""
    topic: str = ""
    goal: str = ""
    cta: str = ""


class ContentCalendarOutput(TaskOutputBase):
    cadence: str = ""
    items: list[ContentItem] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)


# -- creative ------------------------------------------------------------------


class CreativeBriefOutput(TaskOutputBase):
    concept: str = ""
    headline: str = ""
    subheadline: str = ""
    visual_direction: str = ""
    style_spec: dict[str, object] = Field(default_factory=dict)
    ctas: list[str] = Field(default_factory=list)
    variations: list[str] = Field(default_factory=list)
    reference_analysis_id: str | None = None


class Scene(BaseModel):
    order: int
    visual: str = ""
    voiceover: str = ""
    on_screen_text: str = ""
    duration_seconds: float | None = None


class CommercialScriptOutput(TaskOutputBase):
    hook: str = ""
    scenes: list[Scene] = Field(default_factory=list)
    voiceover: str = ""
    cta: str = ""
    duration_seconds: float | None = None


class ImagePromptOutput(TaskOutputBase):
    prompt: str = ""
    negative_prompt: str = ""
    style_spec: dict[str, object] = Field(default_factory=dict)
    aspect_ratio: str = "1:1"
    reference_analysis_id: str | None = None
    reference_notes: str = ""


class Shot(BaseModel):
    order: int
    description: str = ""
    camera: str = ""
    duration_seconds: float | None = None


class VideoPromptOutput(TaskOutputBase):
    prompt: str = ""
    shots: list[Shot] = Field(default_factory=list)
    voiceover: str = ""
    music_direction: str = ""
    duration_seconds: float | None = None
    aspect_ratio: str = "9:16"


class AdCopyOutput(TaskOutputBase):
    primary_texts: list[str] = Field(default_factory=list)
    headlines: list[str] = Field(default_factory=list)
    descriptions: list[str] = Field(default_factory=list)
    ctas: list[str] = Field(default_factory=list)


class AdVariation(BaseModel):
    angle: str = ""
    headline: str = ""
    primary_text: str = ""
    cta: str = ""
    style_notes: str = ""


class AdVariationsOutput(TaskOutputBase):
    variations: list[AdVariation] = Field(default_factory=list)
    based_on: str = ""


# -- analyst -------------------------------------------------------------------


class PerformanceFinding(BaseModel):
    title: str
    detail: str = ""
    severity: Literal["info", "warning", "critical"] = "info"
    metric: str | None = None
    value: float | None = None
    comparison: str | None = None


class PerformanceAnalysisOutput(TaskOutputBase):
    headline_metrics: dict[str, float] = Field(default_factory=dict)
    findings: list[PerformanceFinding] = Field(default_factory=list)
    fatigue_signals: list[str] = Field(default_factory=list)
    recommended_next_step: str = ""


# -- optimizer -----------------------------------------------------------------


class OptimizationProposalOutput(TaskOutputBase):
    action_type: str = ""
    title: str = ""
    rationale: str = ""
    payload: dict[str, object] = Field(default_factory=dict)
    estimated_impact: str = "not available"
    #: When variations are needed, Salma does NOT generate them. She returns an
    #: action_card that the orchestrator routes to layla.
    needs_creative_variations: bool = False
    creative_brief_for_layla: str = ""


TASK_OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "competitor_research": CompetitorResearchOutput,
    "market_analysis": MarketAnalysisOutput,
    "market_gaps": MarketGapsOutput,
    "marketing_strategy": MarketingStrategyOutput,
    "campaign_strategy": CampaignStrategyOutput,
    "content_calendar": ContentCalendarOutput,
    "creative_brief": CreativeBriefOutput,
    "commercial_script": CommercialScriptOutput,
    "image_prompt": ImagePromptOutput,
    "video_prompt": VideoPromptOutput,
    "ad_copy": AdCopyOutput,
    "ad_variations": AdVariationsOutput,
    "performance_analysis": PerformanceAnalysisOutput,
    "optimization_proposal": OptimizationProposalOutput,
}


# -------------------------------------------------------------- prompt store --


class PromptTemplate:
    """A versioned prompt template with named slots."""

    def __init__(self, name: str, version: str, body: str) -> None:
        self.name = name
        self.version = version
        self.body = body
        self.slots = sorted(set(SLOT_RE.findall(body)))

    def render(self, values: dict[str, str]) -> str:
        missing = [s for s in self.slots if s not in values]
        if missing:
            raise KeyError(f"prompt '{self.name}' is missing slots: {missing}")

        def _sub(match: re.Match[str]) -> str:
            return values[match.group(1)]

        return SLOT_RE.sub(_sub, self.body)


_template_cache: dict[str, PromptTemplate] = {}


def load_prompt(task_name: str) -> PromptTemplate:
    """Load ``prompts/<task_name>.md``. Cached; version comes from the header."""
    if task_name in _template_cache:
        return _template_cache[task_name]
    path = PROMPT_DIR / f"{task_name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt template for task '{task_name}' at {path}")
    raw = path.read_text(encoding="utf-8")
    version = "1"
    body = raw
    header = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if header:
        for line in header.group(1).splitlines():
            if line.lower().startswith("version:"):
                version = line.split(":", 1)[1].strip()
        body = raw[header.end():]
    template = PromptTemplate(task_name, version, body.strip())
    _template_cache[task_name] = template
    return template


def available_prompt_tasks() -> list[str]:
    return sorted(p.stem for p in PROMPT_DIR.glob("*.md"))


# ----------------------------------------------------------------- TaskSpec ----


class TaskSpec(BaseModel):
    """A prompt template + output schema + tier. Everything a task needs."""

    model_config = {"frozen": True}

    name: str
    #: Short human label for the UI.
    label: str = ""
    prompt_template: str = Field(description="rendered prompt body, loaded from prompts/")
    prompt_version: str = "1"
    output_model: type[BaseModel]
    tier: Tier = Tier.STRONG
    requires_brand_context: bool = True
    #: Which persona owns this task (documentation/validation only).
    owner_persona: str = ""
    description: str = ""

    def render_prompt(self, values: dict[str, str]) -> str:
        return PromptTemplate(self.name, self.prompt_version, self.prompt_template).render(values)


def _build(name: str, *, label: str, tier: Tier, owner: str, requires_brand_context: bool = True) -> TaskSpec:
    template = load_prompt(name)
    return TaskSpec(
        name=name,
        label=label,
        prompt_template=template.body,
        prompt_version=template.version,
        output_model=TASK_OUTPUT_MODELS[name],
        tier=tier,
        requires_brand_context=requires_brand_context,
        owner_persona=owner,
        description=f"{label} ({owner})",
    )


#: The task catalog. Tiers are deliberate: classification/cheap synthesis on
#: "fast", anything the user will read on "strong".
TASK_SPECS: dict[str, TaskSpec] = {
    "competitor_research": _build(
        "competitor_research", label="Competitor research", tier=Tier.STRONG, owner="researcher"
    ),
    "market_analysis": _build(
        "market_analysis", label="Market analysis", tier=Tier.STRONG, owner="researcher"
    ),
    "market_gaps": _build("market_gaps", label="Market gaps", tier=Tier.STRONG, owner="researcher"),
    "marketing_strategy": _build(
        "marketing_strategy", label="Marketing strategy", tier=Tier.STRONG, owner="omar"
    ),
    "campaign_strategy": _build(
        "campaign_strategy", label="Campaign strategy", tier=Tier.STRONG, owner="omar"
    ),
    "content_calendar": _build(
        "content_calendar", label="Content calendar", tier=Tier.STRONG, owner="omar"
    ),
    "creative_brief": _build(
        "creative_brief", label="Creative brief", tier=Tier.STRONG, owner="layla"
    ),
    "commercial_script": _build(
        "commercial_script", label="Commercial script", tier=Tier.STRONG, owner="layla"
    ),
    "image_prompt": _build("image_prompt", label="Image prompt", tier=Tier.STRONG, owner="layla"),
    "video_prompt": _build("video_prompt", label="Video prompt", tier=Tier.STRONG, owner="layla"),
    "ad_copy": _build("ad_copy", label="Ad copy", tier=Tier.STRONG, owner="layla"),
    "ad_variations": _build(
        "ad_variations", label="Ad variations", tier=Tier.STRONG, owner="layla"
    ),
    "performance_analysis": _build(
        "performance_analysis", label="Performance analysis", tier=Tier.STRONG, owner="karim"
    ),
    "optimization_proposal": _build(
        "optimization_proposal", label="Optimization proposal", tier=Tier.STRONG, owner="salma"
    ),
}


def get_task(name: str) -> TaskSpec:
    from app.core.errors import NotFoundError

    spec = TASK_SPECS.get(name)
    if spec is None:
        raise NotFoundError(
            f"task '{name}' is not in the catalog",
            detail={"available": sorted(TASK_SPECS)},
        )
    return spec


def all_tasks() -> list[TaskSpec]:
    return [TASK_SPECS[name] for name in sorted(TASK_SPECS)]


def validate_task_catalog() -> None:
    """Every declared task must have a template and an output model."""
    problems: list[str] = []
    for name in sorted(TASK_OUTPUT_MODELS):
        if name not in TASK_SPECS:
            problems.append(f"output model '{name}' has no TaskSpec")
    for name, spec in TASK_SPECS.items():
        if name not in TASK_OUTPUT_MODELS:
            problems.append(f"TaskSpec '{name}' has no output model")
        if not spec.prompt_template.strip():
            problems.append(f"TaskSpec '{name}' has an empty prompt template")
    if problems:
        raise RuntimeError("task catalog is inconsistent: " + "; ".join(problems))


# ------------------------------------------------------------- build_prompt ----

#: How each tone changes the writing. Persona data, not prompt engineering
#: scattered across templates.
TONE_STYLE: dict[str, str] = {
    "bold": (
        "Write boldly and directly. Short sentences. Strong claims, but only claims "
        "you can support. No hedging, no filler."
    ),
    "playful": (
        "Write with warmth and a light touch. Conversational, a little witty, never "
        "cutesy. Still precise."
    ),
    "minimal": (
        "Write as little as possible. Every sentence must earn its place. Prefer "
        "fragments over paragraphs. Numbers over adjectives."
    ),
    "professional": (
        "Write in a measured, professional register. Clear structure, no slang, no "
        "exclamation marks."
    ),
}

LANGUAGE_INSTRUCTION: dict[str, str] = {
    "en": "Write all prose in English.",
    "ar": (
        "Write all prose in natural, Egyptian-friendly Modern Standard Arabic. "
        "Use clear, warm, direct Arabic the way a Cairo marketing team actually "
        "writes. Keep every JSON key in English exactly as the schema requires, "
        "and keep proper nouns, brand names and platform names in Latin script. "
        "Numbers stay as digits."
    ),
}


def _persona_style_block(persona: Any) -> str:
    tone_value = str(getattr(getattr(persona, "tone", None), "value", "professional"))
    style = TONE_STYLE.get(tone_value, TONE_STYLE["professional"])
    return (
        f"ROLE\n{getattr(persona, 'role', 'Assistant')} - {getattr(persona, 'name', 'Agent')}\n\n"
        f"PERSONA STYLE ({tone_value})\n{style}\n\n"
        f"ABOUT THIS PERSONA\n{getattr(persona, 'description', '')}"
    )


def _brand_context_block(brand_context: Any) -> str:
    from app.core.memory.store import format_context_for_prompt

    if brand_context is None:
        return "BRAND CONTEXT\n(no brand context was retrieved for this run)"
    return format_context_for_prompt(brand_context)


def _render_value(value: Any) -> str:
    import json

    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _inputs_block(inputs: dict[str, Any], user_message: str) -> str:
    lines: list[str] = []
    if user_message:
        lines.append(f"User's request: {user_message}")
    if inputs:
        lines.append("Structured inputs:")
        for key in sorted(inputs):
            value = inputs[key]
            if value in (None, "", [], {}):
                continue
            lines.append(f"- {key}: {_render_value(value)}")
    if not lines:
        lines.append("(no additional inputs were provided)")
    return "\n".join(lines)


def _tool_results_block(tool_results: str) -> str:
    if not tool_results or not tool_results.strip():
        return (
            "TOOL RESULTS\n(no tools have been called yet - if you need data, call a "
            "tool; if no tool can provide it, declare it unavailable rather than "
            "inventing it)"
        )
    return f"TOOL RESULTS\n{tool_results.strip()}"


def build_prompt(
    persona: Any,
    task: TaskSpec,
    *,
    brand_context: Any = None,
    tool_results: str = "",
    inputs: dict[str, Any] | None = None,
    user_message: str = "",
) -> str:
    """Assemble a task prompt. THE ONLY place prompts are built.

    Order, always:
        1. role + persona style
        2. brand context block
        3. task instructions (rendered template)
        4. tool results
        5. output-schema instruction

    Improving this function improves every task at once, which is the point.
    """
    from app.core.llm.base import schema_instruction

    tone_value = str(getattr(getattr(persona, "tone", None), "value", "professional"))
    values = {
        "persona_role": getattr(persona, "role", "Assistant"),
        "persona_name": getattr(persona, "name", "Agent"),
        "persona_style": TONE_STYLE.get(tone_value, TONE_STYLE["professional"]),
        "brand_context": _brand_context_block(brand_context),
        "inputs": _inputs_block(inputs or {}, user_message),
        "tool_results": _tool_results_block(tool_results),
        "task_name": task.name,
    }
    body = task.render_prompt(values)

    language = str(getattr(getattr(persona, "language", None), "value", "en"))
    language_block = LANGUAGE_INSTRUCTION.get(language, LANGUAGE_INSTRUCTION["en"])

    parts = [
        _persona_style_block(persona),
        body,
        f"LANGUAGE\n{language_block}",
        schema_instruction(task.output_model),
    ]
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


__all__ = [
    "PROMPT_DIR",
    "TASK_OUTPUT_MODELS",
    "TASK_SPECS",
    "AdCopyOutput",
    "AdVariation",
    "AdVariationsOutput",
    "AudienceSegment",
    "CommercialScriptOutput",
    "CompetitorProfile",
    "CompetitorResearchOutput",
    "ContentCalendarOutput",
    "ContentItem",
    "CreativeBriefOutput",
    "Gap",
    "ImagePromptOutput",
    "MarketAnalysisOutput",
    "MarketGapsOutput",
    "MarketingStrategyOutput",
    "OptimizationProposalOutput",
    "CampaignStrategyOutput",
    "PerformanceAnalysisOutput",
    "PerformanceFinding",
    "PromptTemplate",
    "Scene",
    "Shot",
    "TaskOutputBase",
    "TaskSpec",
    "VideoPromptOutput",
    "all_tasks",
    "available_prompt_tasks",
    "get_task",
    "load_prompt",
    "validate_task_catalog",
]
