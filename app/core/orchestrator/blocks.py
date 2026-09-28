"""Compose the frontend's ``Message.blocks`` from agent results.

Two rules, both structural:

1. Every block is validated against the shared discriminated union before it is
   returned. ``validate_block`` is the single gate.
2. Anything unknown or failed degrades to a ``text`` or ``error`` block. A
   malformed agent result must never produce a 500.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.core.logging import get_logger
from app.schemas.blocks import (
    ActionCard,
    BaseBlock,
    Block,
    ContextItem,
    ContextUsedBlock,
    ErrorBlock,
    Finding,
    FindingsBlock,
    MessageWithBlocks,
    MetricBlock,
    QuestionWithOptionsBlock,
    RecommendationBlock,
    TaskProgressBlock,
    TextBlock,
)
from app.schemas.common import (
    BlockKind,
    Evidence,
    RecommendationStatus,
    TaskStatus,
    ToolStatus,
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------- validation --


def validate_block(block: dict[str, Any] | BaseBlock) -> BaseBlock:
    """Validate one block against the shared union. Raises on a bad block.

    Callers that must not fail use ``safe_block`` instead.
    """
    wrapper = MessageWithBlocks.model_validate([block])
    return wrapper.blocks[0]


def safe_block(block: dict[str, Any] | BaseBlock, *, fallback_text: str = "") -> BaseBlock:
    """Validate a block, degrading to text/error instead of raising."""
    try:
        return validate_block(block)
    except ValidationError as exc:
        _logger.warning("block_validation_failed", extra={"error": str(exc)[:300]})
        return TextBlock(text=fallback_text or "I could not format that result.")


def validate_blocks(blocks: list[dict[str, Any] | BaseBlock]) -> list[dict[str, Any]]:
    """Validate a list of blocks, dropping (and replacing) any that fail."""
    out: list[dict[str, Any]] = []
    for block in blocks:
        try:
            validated = validate_block(block)
            out.append(validated.model_dump(mode="json", by_alias=True))
        except ValidationError as exc:
            _logger.warning("block_dropped", extra={"error": str(exc)[:300]})
            out.append(
                TextBlock(text="(part of the response could not be displayed)").model_dump(
                    mode="json", by_alias=True
                )
            )
    return out


# ------------------------------------------------------------- block builders --


def text_block(text: str) -> dict[str, Any]:
    return TextBlock(text=text).model_dump(mode="json", by_alias=True)


def error_block(
    message: str, *, code: str | None = None, unavailable: list[str] | None = None
) -> dict[str, Any]:
    return ErrorBlock(
        message=message,
        code=code,
        retryable=False,
        unavailable_sources=unavailable or [],
    ).model_dump(mode="json", by_alias=True)


def findings_block(title: str, findings: list[Finding]) -> dict[str, Any]:
    return FindingsBlock(title=title, findings=findings).model_dump(mode="json", by_alias=True)


def metric_block(label: str, value: float, **kwargs: Any) -> dict[str, Any]:
    return MetricBlock(label=label, value=value, **kwargs).model_dump(mode="json", by_alias=True)


def action_card(
    *,
    action_type: str,
    label: str,
    target_persona: str | None = None,
    task: str | None = None,
    inputs: dict[str, Any] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    return ActionCard(
        action_type=action_type,
        label=label,
        description=description,
        target_persona=target_persona,
        task=task,
        inputs=inputs or {},
    ).model_dump(mode="json", by_alias=True)


def recommendation_block(
    *,
    recommendation_id: str,
    action_type: str,
    title: str,
    rationale: str = "",
    payload: dict[str, Any] | None = None,
    evidence: list[Evidence] | None = None,
    estimated_impact: str | None = None,
    status: str = RecommendationStatus.PENDING.value,
) -> dict[str, Any]:
    return RecommendationBlock(
        recommendation_id=recommendation_id,
        action_type=action_type,
        title=title,
        rationale=rationale,
        payload=payload or {},
        evidence=evidence or [],
        status=RecommendationStatus(status) if status in RecommendationStatus._value2member_map_ else RecommendationStatus.PENDING,
        requires_approval=True,
        estimated_impact=estimated_impact,
    ).model_dump(mode="json", by_alias=True)


def task_progress_block(
    *,
    task_id: str,
    task_name: str,
    persona_id: str,
    status: str = TaskStatus.RUNNING.value,
    steps: list[dict[str, Any]] | None = None,
    progress: float = 0.0,
    incomplete: bool = False,
    message: str | None = None,
) -> dict[str, Any]:
    return TaskProgressBlock(
        task_id=task_id,
        task_name=task_name,
        persona_id=persona_id,
        status=TaskStatus(status) if status in TaskStatus._value2member_map_ else TaskStatus.RUNNING,
        steps=steps or [],
        progress=progress,
        incomplete=incomplete,
        message=message,
    ).model_dump(mode="json", by_alias=True)


def context_used_block(
    items: list[ContextItem], *, token_estimate: int = 0, truncated: bool = False
) -> dict[str, Any]:
    return ContextUsedBlock(
        items=items, token_estimate=token_estimate, truncated=truncated
    ).model_dump(mode="json", by_alias=True)


def question_block(question: str, options: list[dict[str, Any]], missing: list[str]) -> dict[str, Any]:
    return QuestionWithOptionsBlock(
        question=question, options=options, missing_info=missing
    ).model_dump(mode="json", by_alias=True)


# ------------------------------------------------------------- composition ----


def _evidence_from_output(output: dict[str, Any]) -> list[Evidence]:
    raw = output.get("evidence") or []
    out: list[Evidence] = []
    for item in raw:
        if isinstance(item, dict):
            try:
                out.append(Evidence.model_validate(item))
            except ValidationError:
                continue
    return out


def compose_blocks(
    agent_result: dict[str, Any],
    *,
    persona_id: str,
    task_name: str,
    task_id: str | None = None,
    brand_context: Any = None,
) -> list[dict[str, Any]]:
    """Turn one AgentResult into a list of validated blocks.

    Never raises. Anything it cannot interpret becomes a text or error block.
    """
    blocks: list[dict[str, Any]] = []
    output = agent_result.get("output") or {}
    unavailable = list(agent_result.get("unavailable_sources") or [])
    incomplete = bool(agent_result.get("incomplete"))
    error = agent_result.get("error")

    # 1. Always lead with a text summary.
    summary = str(output.get("summary") or "").strip()
    if not summary:
        if error:
            summary = f"I could not complete {task_name}: {error}"
        elif incomplete:
            summary = (
                f"I ran out of budget before finishing {task_name}. "
                f"Reason: {agent_result.get('incomplete_reason') or 'cap reached'}."
            )
        else:
            summary = f"{task_name} finished."
    blocks.append(text_block(summary))

    # 2. Unavailability is surfaced explicitly, never hidden.
    if unavailable:
        blocks.append(
            text_block(
                "Some sources were unavailable, so parts of this are incomplete:\n"
                + "\n".join(f"- {u}" for u in unavailable)
                + "\n\nPaste or upload the data and I will fold it in."
            )
        )

    # 3. Task-specific blocks.
    try:
        if task_name in ("competitor_research", "market_analysis", "market_gaps"):
            blocks.extend(_research_blocks(task_name, output))
        elif task_name in ("marketing_strategy", "campaign_strategy", "content_calendar"):
            blocks.append(_strategy_block(task_name, output))
        elif task_name in ("creative_brief", "ad_copy", "ad_variations"):
            blocks.extend(_creative_blocks(task_name, output))
        elif task_name in ("commercial_script", "image_prompt", "video_prompt"):
            blocks.append(_creative_block(task_name, output))
        elif task_name == "performance_analysis":
            blocks.extend(_analysis_blocks(output))
        elif task_name == "optimization_proposal":
            blocks.extend(_optimization_blocks(output, unavailable))
    except Exception as exc:  # pragma: no cover - degrade, never 500
        _logger.warning("block_composition_failed", extra={"error": str(exc)})
        blocks.append(error_block("I could not format part of that result."))

    # 4. A hard error gets an explicit error block too.
    if error:
        blocks.append(error_block(str(error), code="agent_error"))

    # 5. Context used.
    if brand_context is not None:
        try:
            items = [
                ContextItem(
                    kind=item.kind,
                    label=item.label,
                    detail=item.detail,
                    similarity=item.similarity,
                    memory_id=item.memory_id,
                )
                for item in list(brand_context.items) + list(brand_context.learnings)
            ]
            if items:
                blocks.append(
                    context_used_block(
                        items,
                        token_estimate=int(getattr(brand_context, "token_estimate", 0) or 0),
                        truncated=bool(getattr(brand_context, "truncated", False)),
                    )
                )
        except Exception:  # pragma: no cover
            pass

    # 6. Validate everything on the way out.
    return validate_blocks(blocks)


def _research_blocks(task_name: str, output: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[Finding] = []
    evidence = _evidence_from_output(output)

    if task_name == "competitor_research":
        for comp in output.get("competitors") or []:
            if not isinstance(comp, dict):
                continue
            findings.append(
                Finding(
                    title=str(comp.get("name") or "Competitor"),
                    detail=str(comp.get("positioning") or ""),
                    source=str(comp.get("source") or "competitor_research"),
                )
            )
    elif task_name == "market_gaps":
        for gap in output.get("gaps") or []:
            if not isinstance(gap, dict):
                continue
            findings.append(
                Finding(
                    title=str(gap.get("title") or "Gap"),
                    detail=str(gap.get("detail") or ""),
                    source=str(gap.get("evidence_source") or "market_gaps"),
                )
            )
    else:
        for segment in output.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            findings.append(
                Finding(
                    title=str(segment.get("name") or "Segment"),
                    detail=str(segment.get("description") or ""),
                    source="market_analysis",
                )
            )

    blocks: list[dict[str, Any]] = []
    if findings:
        blocks.append(
            findings_block(f"{task_name.replace('_', ' ').title()} findings", findings)
        )
    for key in ("differentiation_opportunities", "opportunities", "recommended_focus"):
        values = output.get(key)
        if isinstance(values, list) and values:
            blocks.append(text_block("**" + key.replace("_", " ").title() + "**\n" + "\n".join(
                f"- {v}" for v in values if isinstance(v, str)
            )))
            break
    if evidence:
        blocks.append(
            text_block(
                "**Evidence**\n"
                + "\n".join(f"- {e.source}: {e.detail}" for e in evidence if e.detail)
            )
        )
    return blocks


def _strategy_block(task_name: str, output: dict[str, Any]) -> dict[str, Any]:
    parts: list[str] = []
    if task_name == "marketing_strategy":
        if output.get("positioning"):
            parts.append(f"**Positioning**\n{output['positioning']}")
        if output.get("value_propositions"):
            parts.append("**Value propositions**\n" + "\n".join(
                f"- {v}" for v in output["value_propositions"] if isinstance(v, str)))
        if output.get("channels"):
            rows = [
                f"- {c.get('channel')}: {c.get('role', '')}"
                for c in output["channels"]
                if isinstance(c, dict)
            ]
            if rows:
                parts.append("**Channels**\n" + "\n".join(rows))
        if output.get("ninety_day_phases"):
            parts.append("**90-day plan**\n" + "\n".join(
                f"{i + 1}. {v}" for i, v in enumerate(output["ninety_day_phases"])
                if isinstance(v, str)))
    elif task_name == "campaign_strategy":
        if output.get("objective"):
            parts.append(f"**Objective**\n{output['objective']}")
        if output.get("hypothesis"):
            parts.append(f"**Hypothesis**\n{output['hypothesis']}")
        if output.get("creatives_needed"):
            parts.append("**Creatives needed**\n" + "\n".join(
                f"- {v}" for v in output["creatives_needed"] if isinstance(v, str)))
    else:
        if output.get("cadence"):
            parts.append(f"**Cadence**\n{output['cadence']}")
        items = output.get("items") or []
        if items:
            rows = [
                f"- Week {i.get('week')} | {i.get('channel')} | {i.get('format', '')} | "
                f"{i.get('topic', '')}"
                for i in items
                if isinstance(i, dict)
            ]
            parts.append("**Schedule**\n" + "\n".join(rows))
    if not parts:
        return text_block(str(output.get("summary") or "Strategy produced."))
    return text_block("\n\n".join(parts))


def _creative_blocks(task_name: str, output: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if task_name == "ad_copy":
        for label, key in (
            ("Headlines", "headlines"),
            ("Primary texts", "primary_texts"),
            ("Descriptions", "descriptions"),
            ("CTAs", "ctas"),
        ):
            values = output.get(key)
            if isinstance(values, list) and values:
                blocks.append(text_block(
                    f"**{label}**\n" + "\n".join(f"- {v}" for v in values if isinstance(v, str))
                ))
        return blocks or [text_block("No copy produced.")]

    if task_name == "ad_variations":
        for i, variation in enumerate(output.get("variations") or [], 1):
            if not isinstance(variation, dict):
                continue
            blocks.append(text_block(
                f"**Variation {i} - {variation.get('angle', '')}**\n"
                f"Headline: {variation.get('headline', '')}\n"
                f"Text: {variation.get('primary_text', '')}\n"
                f"CTA: {variation.get('cta', '')}"
            ))
        return blocks or [text_block("No variations produced.")]

    # creative_brief
    parts = []
    if output.get("concept"):
        parts.append(f"**Concept**\n{output['concept']}")
    if output.get("headline"):
        parts.append(f"**Headline**\n{output['headline']}")
    if output.get("visual_direction"):
        parts.append(f"**Visual direction**\n{output['visual_direction']}")
    return [text_block("\n\n".join(parts) or "Brief produced.")]


def _creative_block(task_name: str, output: dict[str, Any]) -> dict[str, Any]:
    if task_name == "commercial_script":
        rows = []
        for scene in output.get("scenes") or []:
            if isinstance(scene, dict):
                rows.append(
                    f"{scene.get('order')}. {scene.get('visual', '')} "
                    f"({scene.get('duration_seconds', '?')}s)"
                )
        body = "\n\n".join(
            filter(
                None,
                [
                    f"**Hook**\n{output.get('hook')}" if output.get("hook") else "",
                    f"**Shots**\n" + "\n".join(rows) if rows else "",
                    f"**Voiceover**\n{output.get('voiceover')}" if output.get("voiceover") else "",
                    f"**CTA**\n{output.get('cta')}" if output.get("cta") else "",
                ],
            )
        )
        return text_block(body or "Script produced.")
    if task_name == "image_prompt":
        return text_block(
            "**Image prompt**\n"
            + str(output.get("prompt") or "")
            + (f"\n\n**Negative prompt**\n{output['negative_prompt']}"
               if output.get("negative_prompt") else "")
        )
    rows = []
    for shot in output.get("shots") or []:
        if isinstance(shot, dict):
            rows.append(f"{shot.get('order')}. {shot.get('description', '')}")
    return text_block(
        "**Video prompt**\n"
        + str(output.get("prompt") or "")
        + (("\n\n**Shots**\n" + "\n".join(rows)) if rows else "")
        + (f"\n\n**Voiceover**\n{output['voiceover']}" if output.get("voiceover") else "")
    )


def _analysis_blocks(output: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    metrics = output.get("headline_metrics") or {}
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                blocks.append(metric_block(str(key), float(value)))
    findings = output.get("findings") or []
    if findings:
        blocks.append(findings_block(
            "Findings",
            [
                Finding(
                    title=str(f.get("title") or "Finding"),
                    detail=str(f.get("detail") or ""),
                    source=str(f.get("metric") or "performance_analysis"),
                )
                for f in findings
                if isinstance(f, dict)
            ],
        ))
    return blocks


def _optimization_blocks(
    output: dict[str, Any], unavailable: list[str]
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    action_type = str(output.get("action_type") or "")

    if action_type == "needs_data":
        blocks.append(error_block(
            "I do not have the performance data needed to make a specific "
            "recommendation.",
            code="needs_data",
            unavailable=unavailable,
        ))
        return blocks

    if output.get("needs_creative_variations"):
        # Salma never generates creatives. She hands the work to layla via an
        # action card the user clicks.
        blocks.append(
            action_card(
                action_type="run_task",
                label="Generate new creative variations",
                target_persona="layla",
                task="ad_variations",
                inputs={
                    "brief": str(output.get("creative_brief_for_layla") or ""),
                    "campaign_id": None,
                },
                description=str(output.get("rationale") or ""),
            )
        )
    return blocks


__all__ = [
    "action_card",
    "compose_blocks",
    "context_used_block",
    "error_block",
    "findings_block",
    "metric_block",
    "question_block",
    "recommendation_block",
    "safe_block",
    "task_progress_block",
    "text_block",
    "validate_block",
    "validate_blocks",
]
