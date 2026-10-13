"""Deterministic, readable Markdown presentation for structured agent reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


_LABELS_AR = {
    "summary": "الخلاصة",
    "details": "التفاصيل",
    "market_overview": "نظرة عامة على السوق",
    "competitor_summary": "تحليل المنافسين",
    "brand": "العلامة",
    "positioning": "التموضع",
    "strengths": "نقاط القوة",
    "weaknesses_or_gaps": "نقاط الضعف أو الفجوات المرصودة",
    "pricing_insights": "ملاحظات الأسعار",
    "offer_insights": "العروض",
    "shipping_insights": "الشحن والتوصيل",
    "search_insights": "الظهور في البحث",
    "social_insights": "الحضور الاجتماعي",
    "opportunities_for_target_brand": "فرص للنشاط",
    "risks": "المخاطر",
    "evidence": "الشواهد",
    "claim": "المعلومة المرصودة",
    "source_url": "المصدر",
    "observed_text": "النص المرصود",
    "source_type": "نوع المصدر",
    "missing_inputs": "معلومات مطلوبة قبل الاعتماد",
    "assumptions": "افتراضات يجب تأكيدها",
    "objective": "الهدف",
    "audience": "الجمهور المستهدف",
    "target_segments": "شرائح الجمهور",
    "value_propositions": "القيمة المقترحة",
    "key_messages": "الرسائل الأساسية",
    "channels": "القنوات المقترحة",
    "channel": "القناة",
    "role": "دورها",
    "budget_share_pct": "حصة الميزانية المقترحة",
    "placements": "أماكن الظهور",
    "rationale": "المبرر",
    "phases": "مراحل التنفيذ المقترحة",
    "name": "المرحلة",
    "duration_days": "المدة بالأيام",
    "focus": "التركيز",
    "actions": "الإجراءات",
    "kpis": "مؤشرات النجاح",
    "success_metrics": "مؤشرات النجاح",
    "metric": "المؤشر",
    "target": "المستهدف",
    "creative_direction": "التوجه الإبداعي",
    "cta_direction": "الدعوة إلى الإجراء",
    "creatives_needed": "المواد الإبداعية المطلوبة",
    "headline": "العنوان",
    "primary_text": "النص الإعلاني",
    "description": "الوصف",
    "call_to_action": "الدعوة إلى الإجراء",
    "aspect_ratio": "نسبة الأبعاد",
    "negative_prompt": "ما ينبغي تجنبه في الصورة",
    "image_prompt": "برومبت الصورة",
    "video_prompt": "برومبت الفيديو",
    "music_direction": "توجيه الموسيقى والصوت",
    "variants": "النسخ الإعلانية",
    "angle": "زاوية الرسالة",
    "script": "السيناريو",
    "timecode": "التوقيت",
    "duration_seconds": "المدة بالثواني",
    "visual": "الصورة",
    "camera": "الكاميرا",
    "voiceover": "التعليق الصوتي",
    "audio_or_dialogue": "الصوت أو الحوار",
    "on_screen_text": "النص الظاهر",
    "date_range": "الفترة الزمنية",
    "campaign_id": "معرّف الحملة",
    "metrics": "المقاييس",
    "breakdowns": "التقسيمات",
    "title": "العنوان",
    "action_type": "نوع الإجراء المقترح",
    "proposed_change": "التغيير المقترح",
    "estimated_impact": "الأثر المتوقع",
    "requires_approval": "يتطلب موافقة قبل التنفيذ",
    "recommendation_id": "معرّف التوصية",
    "preserved_elements": "العناصر التي ستبقى",
    "proposed_changes": "التغييرات المقترحة",
    "questions_for_user": "أسئلة للتوضيح",
    "notes": "ملاحظات",
    "cadence": "وتيرة النشر",
    "items": "العناصر",
    "themes": "الموضوعات",
    "week": "الأسبوع",
    "format": "الصيغة",
    "topic": "الموضوع",
    "goal": "الغاية",
    "cta": "الدعوة إلى الإجراء",
    "status": "الحالة",
}

_LABELS_EN = {
    "summary": "Summary",
    "details": "Details",
    "market_overview": "Market overview",
    "competitor_summary": "Competitor analysis",
    "brand": "Brand",
    "positioning": "Positioning",
    "strengths": "Observed strengths",
    "weaknesses_or_gaps": "Observed weaknesses or gaps",
    "pricing_insights": "Pricing insights",
    "offer_insights": "Offer insights",
    "shipping_insights": "Shipping and delivery",
    "search_insights": "Search visibility",
    "social_insights": "Social presence",
    "opportunities_for_target_brand": "Opportunities for the business",
    "risks": "Risks",
    "evidence": "Evidence",
    "claim": "Observed claim",
    "source_url": "Source",
    "observed_text": "Observed text",
    "source_type": "Source type",
    "missing_inputs": "Needed before approval",
    "assumptions": "Assumptions to confirm",
    "objective": "Objective",
    "audience": "Target audience",
    "target_segments": "Target segments",
    "value_propositions": "Value propositions",
    "key_messages": "Key messages",
    "channels": "Suggested channels",
    "channel": "Channel",
    "role": "Role",
    "budget_share_pct": "Suggested budget share",
    "placements": "Placements",
    "rationale": "Rationale",
    "phases": "Suggested phases",
    "name": "Name",
    "duration_days": "Duration (days)",
    "focus": "Focus",
    "actions": "Actions",
    "kpis": "Success metrics",
    "success_metrics": "Success metrics",
    "metric": "Metric",
    "target": "Target",
    "creative_direction": "Creative direction",
    "cta_direction": "Call to action",
    "creatives_needed": "Creative assets needed",
    "headline": "Headline",
    "primary_text": "Primary text",
    "description": "Description",
    "call_to_action": "Call to action",
    "aspect_ratio": "Aspect ratio",
    "negative_prompt": "Negative prompt",
    "image_prompt": "Image prompt",
    "video_prompt": "Video prompt",
    "music_direction": "Music and audio direction",
    "variants": "Ad variations",
    "angle": "Message angle",
    "script": "Script",
    "timecode": "Timecode",
    "duration_seconds": "Duration (seconds)",
    "visual": "Visual",
    "camera": "Camera",
    "voiceover": "Voiceover",
    "audio_or_dialogue": "Audio or dialogue",
    "on_screen_text": "On-screen text",
    "date_range": "Date range",
    "campaign_id": "Campaign ID",
    "metrics": "Metrics",
    "breakdowns": "Breakdowns",
    "title": "Title",
    "action_type": "Proposed action type",
    "proposed_change": "Proposed change",
    "estimated_impact": "Estimated impact",
    "requires_approval": "Requires approval before execution",
    "recommendation_id": "Recommendation ID",
    "preserved_elements": "Elements to preserve",
    "proposed_changes": "Proposed changes",
    "questions_for_user": "Questions for the user",
    "notes": "Notes",
    "cadence": "Cadence",
    "items": "Items",
    "themes": "Themes",
    "week": "Week",
    "format": "Format",
    "topic": "Topic",
    "goal": "Goal",
    "cta": "Call to action",
    "status": "Status",
}


def _mapping(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(mode="json", exclude_none=True)
        except Exception:
            return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except (TypeError, ValueError):
                return None
            return dict(parsed) if isinstance(parsed, Mapping) else None
    return None


def _label(key: str, arabic: bool) -> str:
    labels = _LABELS_AR if arabic else _LABELS_EN
    if key in labels:
        return labels[key]
    return key.replace("_", " ").strip().capitalize()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (Mapping, list, tuple)):
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "null", "n/a"}:
        return ""
    return text


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _append_bullets(lines: list[str], values: Any, arabic: bool) -> None:
    for value in _as_list(values):
        text = _clean(value)
        if text:
            lines.append(f"- {text}")


def _append_value_section(
    lines: list[str],
    key: str,
    value: Any,
    *,
    arabic: bool,
    heading_level: int = 4,
) -> None:
    if value is None or value == "" or value == [] or value == {}:
        return
    label = _label(key, arabic)
    mapping = _mapping(value)
    if mapping is not None:
        lines.append(f"{'#' * heading_level} {label}")
        for inner_key, inner_value in mapping.items():
            _append_value_section(
                lines,
                inner_key,
                inner_value,
                arabic=arabic,
                heading_level=min(heading_level + 1, 6),
            )
        return
    if isinstance(value, (list, tuple)):
        values = list(value)
        if not values:
            return
        lines.append(f"{'#' * heading_level} {label}")
        if all(_mapping(item) is None for item in values):
            _append_bullets(lines, values, arabic)
            return
        for index, item in enumerate(values, start=1):
            row = _mapping(item)
            if row is None:
                text = _clean(item)
                if text:
                    lines.append(f"- {text}")
                continue
            title_key = next(
                (candidate for candidate in ("brand", "channel", "metric", "title", "name", "claim", "week") if _clean(row.get(candidate))),
                None,
            )
            title = _clean(row.get(title_key)) if title_key else ""
            if title_key == "week":
                title = f"{_label('week', arabic)} {title}"
            if not title:
                title = f"{'عنصر' if arabic else 'Item'} {index}"
            lines.append(f"- **{title}**")
            for inner_key, inner_value in row.items():
                if inner_key == title_key or inner_value is None or inner_value == "" or inner_value == [] or inner_value == {}:
                    continue
                if isinstance(inner_value, (list, tuple)):
                    flattened = "; ".join(_clean(part) for part in inner_value if _clean(part))
                    if flattened:
                        lines.append(f"  - **{_label(inner_key, arabic)}:** {flattened}")
                elif isinstance(inner_value, Mapping):
                    flattened = "; ".join(
                        f"{_label(str(nested_key), arabic)}: {_clean(nested_value)}"
                        for nested_key, nested_value in inner_value.items()
                        if _clean(nested_value)
                    )
                    if flattened:
                        lines.append(f"  - **{_label(inner_key, arabic)}:** {flattened}")
                else:
                    text = _clean(inner_value)
                    if text:
                        lines.append(f"  - **{_label(inner_key, arabic)}:** {text}")
        return
    text = _clean(value)
    if text:
        lines.append(f"**{label}:** {text}")


def _unique(values: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, Mapping):
            identity = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str)
        else:
            identity = _clean(value)
        if identity and identity not in seen:
            seen.add(identity)
            output.append(value)
    return output


def _append_channel_plans(lines: list[str], value: Any, arabic: bool) -> None:
    plans = [row for item in _as_list(value) if (row := _mapping(item)) is not None]
    if not plans:
        _append_value_section(lines, "channels", value, arabic=arabic)
        return
    lines.append(f"#### {_label('channels', arabic)}")
    for plan in plans:
        channel = _clean(plan.get("channel")) or ("قناة" if arabic else "Channel")
        share = _clean(plan.get("budget_share_pct"))
        heading = f"**{channel}**"
        if share:
            suffix = "حصة مقترحة" if arabic else "suggested share"
            heading += f" — {suffix}: {share}%"
        lines.append(f"- {heading}")
        for key in ("role", "placements", "rationale"):
            item = plan.get(key)
            if isinstance(item, (list, tuple)):
                item = ", ".join(_clean(part) for part in item if _clean(part))
            text = _clean(item)
            if text:
                lines.append(f"  - **{_label(key, arabic)}:** {text}")


def _append_strategy_section(
    lines: list[str],
    data: dict[str, Any],
    *,
    title: str,
    arabic: bool,
    include_summary: bool = True,
    skip: tuple[str, ...] = (),
) -> None:
    lines.append(f"### {title}")
    summary = _clean(data.get("summary")) if include_summary else ""
    if summary:
        lines.append(summary)
    order = (
        "objective",
        "audience",
        "positioning",
        "target_segments",
        "value_propositions",
        "key_messages",
        "phases",
        "channels",
        "kpis",
        "success_metrics",
        "creative_direction",
        "cta_direction",
        "creatives_needed",
        "risks",
        "evidence",
        "assumptions",
        "missing_inputs",
    )
    for key in order:
        if key in skip or key not in data:
            continue
        value = data.get(key)
        if key == "channels":
            _append_channel_plans(lines, value, arabic)
        elif key == "kpis":
            rows = [row for item in _as_list(value) if (row := _mapping(item)) is not None]
            if rows:
                lines.append(f"#### {_label(key, arabic)}")
                for row in rows:
                    metric = _clean(row.get("metric"))
                    target = _clean(row.get("target"))
                    if metric:
                        suffix = f" — {_label('target', arabic)}: {target}" if target else (" — المستهدف غير محدد" if arabic else " — target not set")
                        lines.append(f"- **{metric}**{suffix}")
            else:
                _append_value_section(lines, key, value, arabic=arabic)
        else:
            _append_value_section(lines, key, value, arabic=arabic)


def _append_evidence(lines: list[str], values: Any, arabic: bool, heading_level: int = 4) -> None:
    entries = _as_list(values)
    if not entries:
        return
    lines.append(f"{'#' * heading_level} {_label('evidence', arabic)}")
    for item in entries:
        row = _mapping(item)
        if row is None:
            text = _clean(item)
            if text:
                lines.append(f"- {text}")
            continue
        claim = _clean(row.get("claim"))
        observed = _clean(row.get("observed_text"))
        source = _clean(row.get("source_url"))
        source_type = _clean(row.get("source_type"))
        extras = [
            f"{_label(str(key), arabic)}: {_clean(value)}"
            for key, value in row.items()
            if key not in {"claim", "observed_text", "source_url", "source_type"}
            and _clean(value)
        ]
        label = claim or observed or source_type or ("دليل" if arabic else "Evidence")
        if source.startswith(("https://", "http://")):
            label += f" — [{('المصدر' if arabic else 'source')}]({source})"
        lines.append(f"- {label}")
        if observed and observed != label and observed != claim:
            lines.append(f"  - **{_label('observed_text', arabic)}:** {observed}")
        for detail in extras:
            lines.append(f"  - {detail}")


def _format_research(data: dict[str, Any], arabic: bool) -> str:
    title = "تحليل السوق والمنافسين" if arabic else "Market and competitor research"
    lines = [f"## {title}"]
    order = (
        "market_overview",
        "pricing_insights",
        "offer_insights",
        "shipping_insights",
        "search_insights",
        "social_insights",
    )
    for key in order:
        if key in data:
            _append_value_section(lines, key, data[key], arabic=arabic, heading_level=3)
    competitors = data.get("competitor_summary")
    if competitors:
        lines.append(f"### {_label('competitor_summary', arabic)}")
        for index, item in enumerate(_as_list(competitors), start=1):
            row = _mapping(item)
            if row is None:
                text = _clean(item)
                if text:
                    lines.append(f"#### {text}")
                continue
            brand = _clean(row.get("brand")) or (f"المنافس {index}" if arabic else f"Competitor {index}")
            lines.append(f"#### {brand}")
            for key in ("positioning", "strengths", "weaknesses_or_gaps"):
                if row.get(key):
                    _append_value_section(lines, key, row[key], arabic=arabic, heading_level=5)
            if row.get("evidence"):
                _append_evidence(lines, row["evidence"], arabic, heading_level=5)
    for key in ("opportunities_for_target_brand", "risks"):
        if data.get(key):
            _append_value_section(lines, key, data[key], arabic=arabic, heading_level=3)
    _append_evidence(lines, data.get("evidence"), arabic, heading_level=3)
    for key in ("assumptions", "missing_inputs"):
        if data.get(key):
            _append_value_section(lines, key, data[key], arabic=arabic, heading_level=3)
    return "\n".join(lines)


def _format_creative(data: dict[str, Any], arabic: bool) -> str:
    title = "المقترح الإبداعي" if arabic else "Creative proposal"
    lines = [f"## {title}"]
    deliverable = _clean(data.get("deliverable_type"))
    if deliverable:
        label = "نوع المخرج" if arabic else "Deliverable"
        lines.append(f"**{label}:** {deliverable.replace('_', ' ')}")
    summary = _clean(data.get("summary"))
    if summary:
        lines.extend([f"### {_label('summary', arabic)}", summary])
    for key in ("headline", "primary_text", "description", "call_to_action", "image_prompt", "video_prompt", "negative_prompt", "aspect_ratio", "music_direction"):
        if key in data:
            _append_value_section(lines, key, data[key], arabic=arabic, heading_level=3)
    variants = data.get("variants")
    if variants:
        lines.append(f"### {_label('variants', arabic)}")
        for index, item in enumerate(_as_list(variants), start=1):
            row = _mapping(item)
            if row is None:
                text = _clean(item)
                if text:
                    lines.append(f"#### {index}")
                    lines.append(text)
                continue
            lines.append(f"#### {('النسخة' if arabic else 'Option')} {index}")
            for key in ("angle", "headline", "primary_text", "description", "call_to_action"):
                if row.get(key):
                    _append_value_section(lines, key, row[key], arabic=arabic, heading_level=5)
    script = data.get("script")
    if script:
        lines.append(f"### {_label('script', arabic)}")
        for index, item in enumerate(_as_list(script), start=1):
            row = _mapping(item)
            if row is None:
                text = _clean(item)
                if text:
                    lines.append(f"- {text}")
                continue
            timing = _clean(row.get("timecode"))
            if not timing and _clean(row.get("duration_seconds")):
                timing = f"{_clean(row['duration_seconds'])} sec"
            lines.append(f"#### {timing or (f'لقطة {index}' if arabic else f'Shot {index}')}")
            for key in ("visual", "camera", "audio_or_dialogue", "on_screen_text"):
                if row.get(key):
                    _append_value_section(lines, key, row[key], arabic=arabic, heading_level=5)
    for key in ("details", "preserved_elements", "proposed_changes", "reference_style_spec", "questions_for_user", "missing_inputs"):
        if data.get(key):
            _append_value_section(lines, key, data[key], arabic=arabic, heading_level=3)
    return "\n".join(lines)


def _format_omar(data: dict[str, Any], arabic: bool) -> str:
    title = "خطة واستراتيجية التسويق" if arabic else "Marketing strategy and campaign plan"
    lines = [f"## {title}"]
    marketing = _mapping(data.get("marketing_strategy"))
    campaign = _mapping(data.get("campaign_strategy"))
    if marketing:
        _append_strategy_section(
            lines,
            marketing,
            title=("الاستراتيجية التسويقية" if arabic else "Marketing strategy"),
            arabic=arabic,
            skip=("evidence", "assumptions", "missing_inputs"),
        )
    if campaign:
        _append_strategy_section(
            lines,
            campaign,
            title=("خطة الحملة" if arabic else "Campaign plan"),
            arabic=arabic,
            include_summary=True,
            skip=("channels", "evidence", "assumptions", "missing_inputs"),
        )
    if not marketing and not campaign:
        _append_value_section(lines, "summary", data.get("summary"), arabic=arabic, heading_level=3)
        _append_value_section(lines, "details", data.get("details"), arabic=arabic, heading_level=3)
    evidence = _unique(
        _as_list(marketing.get("evidence"))
        + _as_list(campaign.get("evidence"))
        + _as_list(data.get("evidence"))
    )
    assumptions = _unique(
        _as_list(marketing.get("assumptions"))
        + _as_list(campaign.get("assumptions"))
        + _as_list(data.get("assumptions"))
    )
    missing = _unique(
        _as_list(marketing.get("missing_inputs"))
        + _as_list(campaign.get("missing_inputs"))
        + _as_list(data.get("missing_inputs"))
    )
    _append_evidence(lines, evidence, arabic, heading_level=3)
    for key, values in (("assumptions", assumptions), ("missing_inputs", missing)):
        if values:
            _append_value_section(lines, key, values, arabic=arabic, heading_level=3)
    return "\n".join(lines)


def _format_performance(data: dict[str, Any], arabic: bool) -> str:
    title = "تقرير أداء الحملة" if arabic else "Campaign performance report"
    lines = [f"## {title}"]
    _append_value_section(lines, "summary", data.get("summary"), arabic=arabic, heading_level=3)
    _append_value_section(lines, "details", data.get("details"), arabic=arabic, heading_level=3)
    for key in ("date_range", "campaign_id"):
        if data.get(key):
            _append_value_section(lines, key, data[key], arabic=arabic, heading_level=3)
    if data.get("metrics"):
        _append_value_section(lines, "metrics", data["metrics"], arabic=arabic, heading_level=3)
    if data.get("breakdowns"):
        _append_value_section(lines, "breakdowns", data["breakdowns"], arabic=arabic, heading_level=3)
    _append_evidence(lines, data.get("evidence"), arabic, heading_level=3)
    if data.get("missing_inputs"):
        _append_value_section(lines, "missing_inputs", data["missing_inputs"], arabic=arabic, heading_level=3)
    return "\n".join(lines)


def _format_optimization(data: dict[str, Any], arabic: bool) -> str:
    title = "خطة تحسين مقترحة" if arabic else "Proposed optimization plan"
    lines = [f"## {title}"]
    for key in ("summary", "title", "action_type", "rationale", "proposed_change", "estimated_impact", "evidence", "missing_inputs"):
        if key == "evidence":
            _append_evidence(lines, data.get(key), arabic, heading_level=3)
        elif data.get(key) is not None:
            _append_value_section(lines, key, data[key], arabic=arabic, heading_level=3)
    if data.get("requires_approval"):
        lines.append("**يتطلب موافقتك قبل التنفيذ.**" if arabic else "**Your approval is required before execution.**")
    return "\n".join(lines)


def _format_generic(data: dict[str, Any], agent: str, arabic: bool) -> str:
    names_ar = {"nour": "نور", "omar": "عمر", "layla": "ليلى", "karim": "كريم", "salma": "سلمى"}
    title = names_ar.get(agent, agent.title()) if arabic else f"{agent.title()} report"
    lines = [f"## {title}"]
    for key, value in data.items():
        if key in {"status", "business_id", "profile_source", "generated_assets"}:
            continue
        if key == "evidence":
            _append_evidence(lines, value, arabic, heading_level=3)
        else:
            _append_value_section(lines, key, value, arabic=arabic, heading_level=3)
    return "\n".join(lines)


def format_agent_report(agent: str, output: Any, *, arabic: bool) -> str:
    """Render one specialist's structured output as readable Markdown, not JSON."""
    data = _mapping(output)
    if data is None:
        from app.ai.text_utils import extract_clean_text
        return extract_clean_text(output).strip()
    if agent == "nour":
        return _format_research(data, arabic)
    if agent == "omar":
        return _format_omar(data, arabic)
    if agent == "layla":
        return _format_creative(data, arabic)
    if agent == "karim":
        return _format_performance(data, arabic)
    if agent == "salma":
        return _format_optimization(data, arabic)
    return _format_generic(data, agent, arabic)


def _flatten_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value or "")


def _unverified_claim_warnings(creative: Any, evidence: list[Any], arabic: bool) -> list[str]:
    copy_text = _flatten_text(creative).casefold()
    evidence_text = _flatten_text(evidence).casefold()
    claim_groups = [
        ("الضمان", ("ضمان", "warranty", "guarantee")),
        ("أصالة قطع الغيار", ("أصلية", "اصليه", "original parts", "genuine parts")),
        ("المقارنة بالتوكيل", ("التوكيل", "الوكيل", "dealership", "dealer")),
        ("ادعاء الأفضلية أو التوفير", ("الأفضل", "افضل", "أوفر", "ارخص", "best", "cheaper")),
        ("أجهزة أو جودة موصوفة بأنها الأحدث/الأعلى", ("أحدث الأجهزة", "أحدث الاجهزة", "أعلى جودة", "high quality", "latest equipment")),
    ]
    warnings: list[str] = []
    for label, terms in claim_groups:
        if any(term in copy_text for term in terms) and not any(term in evidence_text for term in terms):
            if arabic:
                warnings.append(f"تحقق من ادعاء {label} قبل النشر؛ لم يظهر له توثيق ضمن الشواهد المرفقة بهذه الخطة.")
            else:
                warnings.append(f"Verify the {label} claim before publishing; it is not documented in the evidence attached to this plan.")
    return warnings


def format_campaign_response(omar_output: Any, layla_output: Any, *, arabic: bool) -> str:
    """Present Omar's strategy and Layla's draft as one concise campaign brief."""
    omar = _mapping(omar_output) or {}
    layla = _mapping(layla_output) or {}
    marketing = _mapping(omar.get("marketing_strategy")) or {}
    campaign = _mapping(omar.get("campaign_strategy")) or {}
    title = "خطة الحملة الإعلانية" if arabic else "Advertising campaign plan"
    lines = [f"## {title}"]

    if marketing:
        _append_strategy_section(
            lines,
            marketing,
            title=("الاستراتيجية التسويقية — عمر" if arabic else "Marketing strategy — Omar"),
            arabic=arabic,
            skip=("evidence", "assumptions", "missing_inputs"),
        )
    elif _clean(omar.get("summary")):
        section_title = "استراتيجية عمر" if arabic else "Omar's strategy"
        lines.extend([f"### {section_title}", _clean(omar["summary"])])

    if campaign:
        campaign_skip = ("placements", "evidence", "assumptions", "missing_inputs")
        if marketing.get("channels"):
            campaign_skip = (*campaign_skip, "channels")
        _append_strategy_section(
            lines,
            campaign,
            title=("خطة الحملة" if arabic else "Campaign plan"),
            arabic=arabic,
            skip=campaign_skip,
        )

    if not marketing and not campaign and omar.get("details"):
        _append_value_section(lines, "details", omar["details"], arabic=arabic, heading_level=3)

    if layla:
        creative_title = "مسودة الإعلان — ليلى" if arabic else "Ad draft — Layla"
        lines.append(f"### {creative_title}")
        headline = _clean(layla.get("headline"))
        primary_text = _clean(layla.get("primary_text"))
        description = _clean(layla.get("description"))
        call_to_action = _clean(layla.get("call_to_action"))
        aspect_ratio = _clean(layla.get("aspect_ratio"))
        if headline:
            lines.append(f"**{_label('headline', arabic)}:** {headline}")
        if primary_text:
            lines.extend([f"#### {_label('primary_text', arabic)}", primary_text])
        if description:
            lines.append(f"**{_label('description', arabic)}:** {description}")
        if call_to_action:
            lines.append(f"**{_label('call_to_action', arabic)}:** {call_to_action}")
        if aspect_ratio:
            lines.append(f"**{_label('aspect_ratio', arabic)}:** {aspect_ratio}")
        for key in ("details", "missing_inputs", "questions_for_user"):
            if layla.get(key):
                _append_value_section(lines, key, layla[key], arabic=arabic, heading_level=4)
    elif isinstance(layla_output, str) and layla_output.strip():
        lines.extend([f"### {'مسودة الإعلان — ليلى' if arabic else 'Ad draft — Layla'}", layla_output.strip()])

    evidence = _unique([
        *_as_list(marketing.get("evidence")),
        *_as_list(campaign.get("evidence")),
        *_as_list(omar.get("evidence")),
    ])
    assumptions = _unique([
        *_as_list(marketing.get("assumptions")),
        *_as_list(campaign.get("assumptions")),
        *_as_list(omar.get("assumptions")),
    ])
    missing = _unique([
        *_as_list(marketing.get("missing_inputs")),
        *_as_list(campaign.get("missing_inputs")),
        *_as_list(omar.get("missing_inputs")),
        *_as_list(layla.get("missing_inputs")),
        *_as_list(layla.get("questions_for_user")),
    ])
    if evidence:
        _append_evidence(lines, evidence, arabic, heading_level=3)
    if assumptions:
        _append_value_section(lines, "assumptions", assumptions, arabic=arabic, heading_level=3)
    if missing:
        _append_value_section(lines, "missing_inputs", missing, arabic=arabic, heading_level=3)

    claim_warnings = _unverified_claim_warnings(layla_output, evidence, arabic)
    if claim_warnings:
        lines.append(f"### {'مراجعة قبل النشر' if arabic else 'Review before publishing'}")
        _append_bullets(lines, claim_warnings, arabic)

    displayed_channels = marketing.get("channels") or campaign.get("channels")
    has_suggested_split = any(
        _clean(row.get("budget_share_pct"))
        for row in (_mapping(item) or {} for item in _as_list(displayed_channels))
    )
    if arabic:
        lines.append("> **حالة التنفيذ:** هذه خطة ومسودة نصية فقط؛ لم يتم إنشاء أو نشر حملة أو إعلان على Meta.")
        if has_suggested_split:
            lines.append("> نسب القنوات، إن وُجدت، مقترحات أولية وليست توزيعًا معتمدًا للميزانية.")
    else:
        lines.append("> **Execution status:** This is a plan and copy draft only; no Meta campaign or ad was created or published.")
        if has_suggested_split:
            lines.append("> Any channel percentages shown are initial suggestions, not an approved budget allocation.")
    return "\n".join(line for line in lines if line.strip())


__all__ = ["format_agent_report", "format_campaign_response"]
