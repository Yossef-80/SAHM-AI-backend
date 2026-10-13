"""Prompt-only interview gates: checklist first, one approval card, no media generation."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ai.agents.orchestrator.brain import (
    _answered_creative_prompt_fields,
    _creative_prompt_blocks,
    _creative_prompt_facts_missing_from_history,
    _creative_prompt_user_evidence,
    _fallback_prompt_questions,
    _is_explicit_prompt_approval,
    _merge_creative_prompt_fact_updates,
    _normalize_creative_evidence_text,
    _prepare_creative_prompt_turn,
    _update_creative_prompt_user_facts,
)
from app.schemas.blocks import QuestionWithOptionsBlock, TextBlock


def test_incomplete_prompt_request_gets_one_grouped_checklist_not_approval() -> None:
    output = SimpleNamespace(
        summary="Not enough details yet.",
        details=[],
        missing_inputs=[],
        questions_for_user=[],
        preserved_elements=[],
        proposed_changes=[],
        ready_for_approval=True,
    )
    blocks, stage = _creative_prompt_blocks(
        output,
        context={"prompt_type": "image"},
        prompt="أريد برومبت قوي لتوليد صورة، اسأليني عن التفاصيل الناقصة.",
        stage="collecting_details",
        history=[],
    )

    assert stage == "collecting_details"
    assert len(blocks) == 1 and isinstance(blocks[0], TextBlock)
    # Wording is deliberately friendly now; the guarantee is structural: one
    # grouped checklist and no approval card on a collecting-details turn.
    assert "ناقصني كام تفصيلة صغيرة" in blocks[0].text
    assert "أكتب البرومبت النهائي" not in blocks[0].text


def test_repeated_checklist_only_asks_for_unanswered_slots() -> None:
    first_request = (
        "ليلى ساعديني على إعداد برومبت قوي لتوليد صورة. ابدئي بقائمة واحدة مجمّعة "
        "بالتفاصيل الناقصة، ثم اعرضي خطة لما سأحافظ عليه وما سأغيّره قبل البرومبت النهائي. "
        "موجزي المبدئي: تصميم سوشيال ميديا إعلاني لفتيس MG ZS."
    )
    repeated_checklist = [
        "ما المنتج أو الموضوع، وما الهدف من الصورة/الفيديو؟",
        "من الجمهور وأين سيُستخدم؟ اذكر المنصة والمقاس أو المدة إن كانا محددين.",
        "ما النص الظاهر أو العرض أو الدعوة للإجراء المعتمدة؟",
        "ما الأسلوب أو الألوان المطلوبة، وما العناصر التي يجب تجنبها؟",
    ]
    answer = (
        "1. فتيس MG ZS\n"
        "2. أصحاب سيارات MG الذين يحتاجون صيانة للفتيس أو تغييره\n"
        "3. عايز نص كرييتف\n"
        "4. الأحمر والأبيض"
    )
    output = SimpleNamespace(
        summary="Not enough details yet.",
        details=[],
        missing_inputs=repeated_checklist,
        questions_for_user=repeated_checklist,
        preserved_elements=[],
        proposed_changes=[],
        ready_for_approval=False,
    )

    blocks, stage = _creative_prompt_blocks(
        output,
        context={"prompt_type": "image"},
        prompt=answer,
        stage="collecting_details",
        history=[
            {"role": "user", "content": first_request},
            {
                "role": "assistant",
                "content": "## التفاصيل الناقصة\n" + "\n".join(repeated_checklist),
            },
        ],
    )

    # The subject and the objective are known, so the server presents a plan and
    # proposes the cosmetic gaps instead of opening a second round of questions.
    # The guarantee this test protects is unchanged: nothing already answered is
    # asked again.
    assert stage == "awaiting_approval"
    assert isinstance(blocks[1], QuestionWithOptionsBlock)
    reply = blocks[0].text
    assert "ما المنتج أو الموضوع" not in reply
    assert "من الجمهور" not in reply
    assert "ما الألوان المطلوبة" not in reply
    assert "النص الظاهر" not in reply
    assert "عناصر أو ادعاءات يجب تجنبها" not in reply


def test_explicit_none_answers_close_copy_offer_cta_and_avoidance_slots() -> None:
    first_request = "أريد برومبت صورة إعلان لفتيس MG ZS على السوشيال ميديا."
    first_answer = (
        "فتيس MG ZS؛ الجمهور أصحاب سيارات MG؛ ألوان أحمر وأبيض؛ أريد نص كرييتف وأسلوبًا عصريًا."
    )
    follow_up_answer = (
        "Instagram مقاس 4:5. بدون نص أو عرض أو CTA. لا توجد عناصر أو ادعاءات يجب تجنبها."
    )
    history = [
        {"role": "user", "content": first_request},
        {"role": "assistant", "content": "## التفاصيل الناقصة\nأسئلة Layla السابقة"},
        {"role": "user", "content": first_answer},
        {"role": "assistant", "content": "## التفاصيل الناقصة\nالأسئلة المتبقية"},
    ]
    output = SimpleNamespace(
        summary="Not enough details yet.",
        details=[],
        missing_inputs=[],
        questions_for_user=[],
        preserved_elements=[],
        proposed_changes=[],
        ready_for_approval=False,
    )

    blocks, stage = _creative_prompt_blocks(
        output,
        context={"prompt_type": "image"},
        prompt=follow_up_answer,
        stage="collecting_details",
        history=history,
    )

    # Everything was answered, so the reply is a plan, not another checklist.
    # The guarantee this test protects: an explicit "none" really closes the slot.
    assert stage == "awaiting_approval"
    assert isinstance(blocks[1], QuestionWithOptionsBlock)
    plan_text = blocks[0].text
    assert "النص الظاهر" not in plan_text
    assert "العرض الترويجي" not in plan_text
    assert "عناصر أو ادعاءات يجب تجنبها" not in plan_text


def test_complete_prompt_plan_has_single_approval_question() -> None:
    output = SimpleNamespace(
        summary="صورة مربعة لمنتج القهوة الباردة، بإضاءة صباحية هادئة.",
        details=["زجاجة في المقدمة وخلفية بسيطة."],
        missing_inputs=[],
        questions_for_user=[],
        preserved_elements=[],
        proposed_changes=["استخدام خلفية محايدة بدل ازدحام العناصر."],
        ready_for_approval=True,
    )
    blocks, stage = _creative_prompt_blocks(
        output,
        context={"prompt_type": "image"},
        prompt="أريد صورة إعلان لمنتج القهوة الباردة بأسلوب بسيط.",
        stage="collecting_details",
        history=[],
    )

    assert stage == "awaiting_approval"
    assert len(blocks) == 2
    assert isinstance(blocks[0], TextBlock)
    assert isinstance(blocks[1], QuestionWithOptionsBlock)
    assert "Do you approve" not in blocks[0].text
    assert "ملخص الخطة المقترحة" in blocks[0].text
    assert "اقتراح قابل للتعديل" in blocks[0].text
    assert blocks[1].question in {
        "دي الخطة اللي وصلت لها من كلامك. أكتب البرومبت النهائي لـصورة؟ ولو عايز أي تعديل قولّي عليه.",
        "Here is the plan I put together from what you told me. Shall I write the final image prompt? Tell me any change you want.",
    }


def test_approval_without_a_pending_plan_stays_in_details_stage() -> None:
    result = _prepare_creative_prompt_turn(
        {"workflow": "creative_prompt", "prompt_type": "image", "creative_prompt_stage": "collecting_details"},
        "موافق، اكتب البرومبت",
    )

    assert result["creative_prompt_stage"] == "collecting_details"
    assert result["creative_prompt_approval_without_plan"] is True


def test_explicit_approval_advances_only_a_plan_that_is_waiting_for_approval() -> None:
    result = _prepare_creative_prompt_turn(
        {"workflow": "creative_prompt", "prompt_type": "video", "creative_prompt_stage": "awaiting_approval"},
        "موافق، اكتب البرومبت",
    )

    assert result["creative_prompt_stage"] == "approved"


def test_approved_turn_returns_prompt_text_without_generating_media() -> None:
    output = SimpleNamespace(
        summary="فيديو رأسي قصير بمشاهد المنتج كما ورد في الملخص.",
        details=["مدة 15 ثانية، مع تعليق صوتي معتمد فقط."],
        missing_inputs=[],
        questions_for_user=[],
        preserved_elements=[],
        proposed_changes=[],
        ready_for_approval=True,
        video_prompt="Vertical 9:16 video; use only confirmed details from the brief.",
        negative_prompt="",
        script=[],
        music_direction="",
        generation_requested=False,
        generated_assets=[],
    )
    blocks, stage = _creative_prompt_blocks(
        output,
        context={"prompt_type": "video"},
        prompt="موافق، اكتب البرومبت",
        stage="approved",
        history=[],
    )

    assert stage == "prompt_ready"
    assert len(blocks) == 1 and isinstance(blocks[0], TextBlock)
    assert "Vertical 9:16 video" in blocks[0].text
    assert "لم يتم توليد صورة أو فيديو" in blocks[0].text


def test_egyptian_short_yes_is_approval_only_while_plan_is_pending() -> None:
    assert _is_explicit_prompt_approval("اه") is True

    collecting = _prepare_creative_prompt_turn(
        {"workflow": "creative_prompt", "prompt_type": "image", "creative_prompt_stage": "collecting_details"},
        "اه",
    )
    assert collecting["creative_prompt_stage"] == "collecting_details"
    assert collecting["creative_prompt_approval_without_plan"] is True

    awaiting = _prepare_creative_prompt_turn(
        {"workflow": "creative_prompt", "prompt_type": "image", "creative_prompt_stage": "awaiting_approval"},
        "اه",
    )
    assert awaiting["creative_prompt_stage"] == "approved"

    reminder = _prepare_creative_prompt_turn(
        {"workflow": "creative_prompt", "prompt_type": "image", "creative_prompt_stage": "awaiting_approval"},
        "بعتلك كل ده",
    )
    assert reminder["creative_prompt_stage"] == "awaiting_approval"
    assert reminder["creative_prompt_use_defaults"] is True


def test_reminder_while_plan_is_pending_does_not_restart_checklist_or_approve() -> None:
    context = _prepare_creative_prompt_turn(
        {"workflow": "creative_prompt", "prompt_type": "image", "creative_prompt_stage": "awaiting_approval"},
        "بعتلك كل ده",
    )
    output = SimpleNamespace(
        summary="Not enough details yet.",
        details=[],
        missing_inputs=["ما المقاس؟", "ما الأسلوب؟"],
        questions_for_user=["هل تريد تحديد المقاس؟"],
        preserved_elements=[],
        proposed_changes=[],
        ready_for_approval=False,
    )

    blocks, stage = _creative_prompt_blocks(
        output,
        context=context,
        prompt="بعتلك كل ده",
        stage="awaiting_approval",
        history=[{"role": "user", "content": "إعلان لفتيس MG ZS."}],
    )

    assert stage == "awaiting_approval"
    assert len(blocks) == 2
    assert isinstance(blocks[0], TextBlock)
    assert isinstance(blocks[1], QuestionWithOptionsBlock)
    assert "الخطة المعروضة سابقًا" in blocks[0].text
    assert "التفاصيل الناقصة" not in blocks[0].text


def test_no_more_details_does_not_bypass_missing_subject_or_objective() -> None:
    context = _prepare_creative_prompt_turn(
        {"workflow": "creative_prompt", "prompt_type": "image", "creative_prompt_stage": "collecting_details"},
        "بعتلك كل ده",
    )
    output = SimpleNamespace(
        summary="خطة إعلان جاهزة.",
        details=["خلفية بسيطة."],
        missing_inputs=[],
        questions_for_user=[],
        preserved_elements=[],
        proposed_changes=[],
        ready_for_approval=True,
    )

    blocks, stage = _creative_prompt_blocks(
        output,
        context=context,
        prompt="بعتلك كل ده",
        stage="collecting_details",
        history=[{"role": "user", "content": "أريد برومبت لصورة."}],
    )

    assert stage == "collecting_details"
    assert len(blocks) == 1 and isinstance(blocks[0], TextBlock)
    assert "المنتج أو الموضوع" in blocks[0].text
    assert "الهدف" in blocks[0].text
    assert "هل تعتمد هذه الخطة" not in blocks[0].text


def test_reported_multiturn_transcript_reaches_prompt_after_ah_without_reasking() -> None:
    kickoff = (
        "ليلى، ساعديني على إعداد برومبت قوي لتوليد صورة إعلان لفتيس MG ZS. "
        "ابدئي بقائمة واحدة مجمّعة بالتفاصيل الناقصة؛ لا تولّدي صورة."
    )
    details = (
        "1. الصورة هستخدمها في الإعلانات\n"
        "2. فيسبوك\n"
        "3. عايز نص كرييتف\n"
        "4. الأحمر والأبيض"
    )
    history = [
        {"role": "user", "content": kickoff},
        {"role": "assistant", "content": "ما الهدف؟ وما المنصة والمقاس؟ وما النص والعرض؟ وما الألوان؟"},
        {"role": "user", "content": details},
        {
            "role": "assistant",
            "content": "ما المقاس أو موضع النشر؟ وما النص أو العرض أو CTA؟ وهل توجد ادعاءات يجب تجنبها؟",
        },
    ]

    answered = _answered_creative_prompt_fields(kickoff + "\n" + details, {})
    assert {"subject", "objective", "specific_platform", "copy", "palette"} <= answered
    assert "style" not in answered  # “creative text” is copy intent, not visual-style evidence.

    context = _prepare_creative_prompt_turn(
        {"workflow": "creative_prompt", "prompt_type": "image", "creative_prompt_stage": "collecting_details"},
        "بعتلك كل ده",
    )
    assert context["creative_prompt_stage"] == "collecting_details"
    assert context["creative_prompt_use_defaults"] is True

    plan_output = SimpleNamespace(
        summary="إعلان لفتيس MG ZS على فيسبوك بألوان الأحمر والأبيض، مع مقاس 1:1 وأسلوب سينمائي فاخر كاقتراحين قابلين للتعديل.",
        details=[
            "المؤكد: الغرض إعلاني، والمنصة فيسبوك، والألوان الأحمر والأبيض.",
            "مقترح قابل للتعديل: صياغة خيارين أو ثلاثة للنص الظاهر؛ لم يُحدّد مقاس للصورة.",
        ],
        missing_inputs=[
            "ما موضع النشر أو المقاس/نسبة الأبعاد؟",
            "ما النص الظاهر أو العرض أو الدعوة للإجراء المعتمدة؟",
            "هل توجد عناصر أو ادعاءات يجب تجنبها؟",
        ],
        questions_for_user=["هل تريد نسبة 1:1 وأسلوبًا سينمائيًا فاخرًا؟"],
        preserved_elements=[],
        proposed_changes=["استخدام الألوان التي طلبها المستخدم فقط."],
        ready_for_approval=True,
    )
    blocks, stage = _creative_prompt_blocks(
        plan_output,
        context=context,
        prompt="بعتلك كل ده",
        stage=context["creative_prompt_stage"],
        history=history,
    )

    assert stage == "awaiting_approval"
    assert len(blocks) == 2
    assert isinstance(blocks[0], TextBlock)
    assert isinstance(blocks[1], QuestionWithOptionsBlock)
    plan_text = blocks[0].text
    assert "الخطة المقترحة" in plan_text
    assert "1:1" in plan_text and "سينمائي فاخر" in plan_text
    assert "ليس معلومة مؤكدة" in plan_text
    assert "التفاصيل الناقصة" not in plan_text
    assert "ما موضع النشر" not in plan_text
    assert "ما النص الظاهر" not in plan_text

    approved_context = _prepare_creative_prompt_turn(
        {**context, "creative_prompt_stage": stage},
        "اه",
    )
    assert approved_context["creative_prompt_stage"] == "approved"

    final_output = SimpleNamespace(
        summary="تم اعتماد الخطة وتقديم البرومبت النهائي. ما زالت هناك تفاصيل ناقصة.",
        details=[],
        missing_inputs=["ما المقاس؟"],
        questions_for_user=["هل تريد تحديد المقاس قبل المتابعة؟"],
        preserved_elements=[],
        proposed_changes=[],
        ready_for_approval=False,
        image_prompt="Create an image prompt for an MG ZS gearbox advertisement using only the confirmed red-and-white palette and Facebook placement; do not infer service claims or an aspect ratio.",
        negative_prompt="No unsupported offers, service claims, or benefits.",
        generation_requested=False,
        generated_assets=[],
    )
    final_blocks, final_stage = _creative_prompt_blocks(
        final_output,
        context=approved_context,
        prompt="اه",
        stage=approved_context["creative_prompt_stage"],
        history=history + [
            {"role": "assistant", "content": plan_text},
            {"role": "user", "content": "اه"},
        ],
    )

    assert final_stage == "prompt_ready"
    assert len(final_blocks) == 1 and isinstance(final_blocks[0], TextBlock)
    assert "Create an image prompt for an MG ZS gearbox advertisement" in final_blocks[0].text
    assert "تم اعتماد الخطة وتقديم البرومبت النهائي" not in final_blocks[0].text
    assert "هل تريد تحديد المقاس" not in final_blocks[0].text
    assert "لم يتم توليد صورة أو فيديو" in final_blocks[0].text


def test_egyptian_engine_overhaul_terms_match_without_heyb2a_partial_match() -> None:
    overhaul = _answered_creative_prompt_fields("عايز اعمل لعمرة الموتور", {})
    spelled_variant = _answered_creative_prompt_fields("عايز إعلان لعمرة الماتور", {})
    false_pronoun = _answered_creative_prompt_fields("الخدمة هيبقى شكلها جذاب", {})

    assert "subject" in overhaul
    assert "subject" in spelled_variant
    assert "subject" not in false_pronoun


def test_egyptian_same_theme_colors_copy_and_contact_details_are_reconciled() -> None:
    context = {
        "reference_analysis_context": {
            "source_name": "saved-reference.png",
            "summary": "A red-and-white automotive service advertisement.",
            "style": {"canvas_ratio": "4:5", "palette": ["red", "white"]},
        }
    }
    user_details = (
        "نفس الثيم ونفس الألوان؛ عايز نص كرييتف، وضيف الموقع ورقم التليفون."
    )

    answered = _answered_creative_prompt_fields(user_details, context)

    assert {"copy", "style", "palette", "reference", "format"} <= answered


def test_unsupplied_offer_or_cta_does_not_block_a_prompt_plan() -> None:
    answered = {
        "subject",
        "objective",
        "audience",
        "specific_platform",
        "format",
        "copy",
        "style",
        "palette",
        "avoid",
    }

    assert _fallback_prompt_questions(
        kind="image",
        has_reference=False,
        arabic=True,
        answered=answered,
    ) == []


def test_prompt_user_facts_survive_history_compaction_and_explicit_new_brief_resets() -> None:
    context = {
        "workflow": "creative_prompt",
        "prompt_type": "image",
        "creative_prompt_stage": "collecting_details",
    }
    kickoff = "ليلى، أريد برومبت صورة إعلان لعمرة الموتور."
    details = "نفس الثيم والألوان؛ عايز نص كرييتف، وضيف الموقع ورقم التليفون."

    _update_creative_prompt_user_facts(context, kickoff, [], kind="image")
    stored_facts = _update_creative_prompt_user_facts(
        context, details, [], kind="image"
    )
    evidence = _creative_prompt_user_evidence(
        "بعتلك كل ده", [], kind="image", context=context
    )
    answered = _answered_creative_prompt_fields(evidence, {})
    missing_from_recent_history = _creative_prompt_facts_missing_from_history(
        stored_facts, [], "بعتلك كل ده"
    )

    assert stored_facts == [kickoff, details]
    assert {"subject", "objective", "copy", "style", "palette", "reference"} <= answered
    assert missing_from_recent_history == [kickoff, details]

    context["creative_prompt_confirmed_facts"] = {
        "subject": {"value": "old subject", "status": "confirmed", "source": "user"}
    }
    context["creative_prompt_unresolved_fields"] = ["palette"]
    fresh_request = "عايز برومبت صورة جديدة لمنتج تاني."
    reset_facts = _update_creative_prompt_user_facts(
        context, fresh_request, [], kind="image"
    )
    assert reset_facts == [fresh_request]
    assert "creative_prompt_confirmed_facts" not in context
    assert "creative_prompt_unresolved_fields" not in context


def test_compacted_legacy_chat_seeds_user_facts_from_stored_turns() -> None:
    kickoff = "ليلى، أريد برومبت صورة إعلان لعمرة الموتور."
    details = "نفس الثيم والألوان؛ عايز نص كرييتف، وضيف الموقع ورقم التليفون."
    context = {
        "workflow": "creative_prompt",
        "prompt_type": "image",
        "creative_prompt_evidence_history": [
            {"role": "user", "content": kickoff},
            {"role": "assistant", "content": "## التفاصيل الناقصة\\nأجيبي عمّا لم تذكريه."},
            {"role": "user", "content": details},
        ],
    }

    facts = _update_creative_prompt_user_facts(
        context, "بعتلك كل ده", [], kind="image"
    )
    answered = _answered_creative_prompt_fields(
        _creative_prompt_user_evidence("بعتلك كل ده", [], kind="image", context=context),
        {},
    )

    assert facts == [kickoff, details, "بعتلك كل ده"]
    assert {"subject", "objective", "copy", "style", "palette", "reference"} <= answered


def test_reference_analysis_is_shown_once_and_kept_for_plan_review() -> None:
    kickoff = "عايز برومبت صورة إعلان لعمرة الموتور، بنفس ثيم الصورة المرجعية."
    reference = {
        "source_name": "saved-reference.png",
        "summary": "An automotive service creative with a centered subject.",
        "style": {
            "canvas_ratio": "4:5",
            "composition": "العنصر الرئيسي في المنتصف",
            "background": "خلفية فاتحة",
            "palette": ["أحمر", "أبيض"],
        },
    }
    context = {
        "prompt_type": "image",
        "reference_analysis_id": "ref-123",
        "reference_analysis_context": reference,
    }
    incomplete = SimpleNamespace(
        summary="Not enough details yet.",
        details=[],
        missing_inputs=[],
        questions_for_user=[],
        preserved_elements=[],
        proposed_changes=[],
        ready_for_approval=False,
    )

    first_blocks, first_stage = _creative_prompt_blocks(
        incomplete,
        context=context,
        prompt=kickoff,
        stage="collecting_details",
        history=[],
    )

    # A kickoff that already names the subject, the objective and the reference
    # is enough to show a plan; the reference analysis travels with it.
    assert first_stage == "awaiting_approval"
    first_text = first_blocks[0].text
    # The plan branch heads the same section "تحليل الصورة المرجعية عبر VLM"; both
    # headings count as "shown" for the shown-once key.
    assert "ما استخرجته من الصورة" in first_text or "تحليل الصورة المرجعية عبر VLM" in first_text
    assert "saved-reference.png" in first_text
    assert context["creative_prompt_reference_shown_key"] == "id:ref-123"
    assert context["reference_analysis_context"] == reference

    user_answer = (
        "نفس الثيم والألوان، وعايز نص كرييتف. ضيف الموقع ورقم التليفون، "
        "وشخص مصري مبتسم باصص للكاميرا."
    )
    plan = SimpleNamespace(
        summary="إعلان لعمرة الموتور يحافظ على الهوية البصرية ويضيف الشخص المطلوب.",
        details=["صياغة 2-3 خيارات نصية قصيرة، مع استخدام بيانات التواصل التي ذكرها المستخدم."],
        missing_inputs=[],
        questions_for_user=[],
        preserved_elements=["التكوين ولوحة الألوان المرصودة في المرجع."],
        proposed_changes=["إضافة شخص مصري مبتسم ينظر مباشرة إلى الكاميرا."],
        ready_for_approval=True,
    )
    history = [
        {"role": "user", "content": kickoff},
        {"role": "assistant", "content": first_text},
    ]
    second_blocks, second_stage = _creative_prompt_blocks(
        plan,
        context=context,
        prompt=user_answer,
        stage="collecting_details",
        history=history,
    )

    assert second_stage == "awaiting_approval"
    plan_text = second_blocks[0].text
    assert "ما استخرجته من الصورة" not in plan_text
    assert "تحليل الصورة المرجعية عبر VLM" not in plan_text
    assert "saved-reference.png" not in plan_text
    assert "ما أقترح الحفاظ عليه" in plan_text
    assert "التغييرات المقترحة" in plan_text
    assert "التكوين ولوحة الألوان" in plan_text
    assert "شخص مصري مبتسم" in plan_text
    assert isinstance(second_blocks[1], QuestionWithOptionsBlock)

    # An older active conversation may not have the newly persisted key yet;
    # recover it from the earlier assistant observation instead of repeating it.
    legacy_context = {
        key: value
        for key, value in context.items()
        if key != "creative_prompt_reference_shown_key"
    }
    legacy_blocks, _ = _creative_prompt_blocks(
        plan,
        context=legacy_context,
        prompt=user_answer,
        stage="collecting_details",
        history=history,
    )
    assert "ما استخرجته من الصورة" not in legacy_blocks[0].text
    assert legacy_context["creative_prompt_reference_shown_key"] == "id:ref-123"

    new_reference_context = {
        **context,
        "reference_analysis_id": "ref-456",
        "reference_analysis_context": {
            **reference,
            "source_name": "updated-reference.png",
        },
    }
    new_reference_blocks, _ = _creative_prompt_blocks(
        incomplete,
        context=new_reference_context,
        prompt=kickoff,
        stage="collecting_details",
        history=[],
    )
    assert "updated-reference.png" in new_reference_blocks[0].text
    assert new_reference_context["creative_prompt_reference_shown_key"] == "id:ref-456"


def test_semantic_fact_updates_require_current_user_quote_and_apply_corrections() -> None:
    context = {
        "creative_prompt_confirmed_facts": {
            "palette": {
                "value": "red",
                "status": "confirmed",
                "evidence": "الأحمر",
                "source": "user",
            }
        }
    }
    output = {
        "prompt_fact_updates": [
            {
                "field": "palette",
                "value": "blue and white",
                "status": "correction",
                "evidence": "قصدي الأزرق والأبيض",
            },
            {
                "field": "style",
                "value": "cinematic",
                "status": "confirmed",
                "evidence": "cinematic",
            },
        ]
    }

    facts, unresolved = _merge_creative_prompt_fact_updates(
        context,
        output,
        "مش الأحمر، قصدي الأزرق والأبيض.",
    )

    assert unresolved == []
    assert facts["palette"] == {
        "value": "blue and white",
        "status": "correction",
        "evidence": "قصدي الأزرق والأبيض",
        "source": "user",
    }
    assert "style" not in facts  # The supposed quote is absent from the latest turn.
    assert "palette" in _answered_creative_prompt_fields("", context)


def test_unmarked_conflicting_semantic_update_is_not_silently_promoted() -> None:
    context = {
        "creative_prompt_confirmed_facts": {
            "palette": {
                "value": "red",
                "status": "confirmed",
                "evidence": "red",
                "source": "user",
            }
        }
    }

    facts, unresolved = _merge_creative_prompt_fact_updates(
        context,
        {
            "prompt_fact_updates": [
                {
                    "field": "palette",
                    "value": "blue",
                    "status": "confirmed",
                    "evidence": "blue",
                }
            ]
        },
        "Blue could work.",
    )

    assert unresolved == ["palette"]
    assert facts["palette"]["status"] == "uncertain"
    assert "palette" not in _answered_creative_prompt_fields("", context)


def test_semantic_negation_closes_slot_and_uncertainty_stays_unresolved() -> None:
    context: dict[str, object] = {}
    facts, unresolved = _merge_creative_prompt_fact_updates(
        context,
        {
            "prompt_fact_updates": [
                {
                    "field": "offer",
                    "value": "none",
                    "status": "negated",
                    "evidence": "مش عايز عرض",
                }
            ]
        },
        "مش عايز عرض.",
    )
    assert unresolved == []
    assert facts["offer"]["status"] == "negated"
    assert "offer" in _answered_creative_prompt_fields("", context)

    _merge_creative_prompt_fact_updates(
        context,
        {
            "prompt_fact_updates": [
                {
                    "field": "palette",
                    "value": "red or blue; user is unsure",
                    "status": "uncertain",
                    "evidence": "مش متأكد أحمر ولا أزرق",
                }
            ]
        },
        "مش متأكد أحمر ولا أزرق.",
    )
    assert context["creative_prompt_unresolved_fields"] == ["palette"]
    assert "palette" not in _answered_creative_prompt_fields("", context)


def test_uncertain_semantic_fact_is_asked_again_after_history_compaction() -> None:
    confirmed = {
        field: {
            "value": field,
            "status": "confirmed",
            "evidence": field,
            "source": "user",
        }
        for field in (
            "subject",
            "objective",
            "audience",
            "specific_platform",
            "format",
            "copy",
            "style",
            "avoid",
        )
    }
    confirmed["palette"] = {
        "value": "not sure between red and blue",
        "status": "uncertain",
        "evidence": "مش متأكد أحمر ولا أزرق",
        "source": "user",
    }
    context = {
        "prompt_type": "image",
        "creative_prompt_confirmed_facts": confirmed,
        "creative_prompt_unresolved_fields": ["palette"],
    }
    output = SimpleNamespace(
        summary="Not enough details yet.",
        details=[],
        missing_inputs=[],
        questions_for_user=[],
        preserved_elements=[],
        proposed_changes=[],
        ready_for_approval=False,
    )

    blocks, stage = _creative_prompt_blocks(
        output,
        context=context,
        prompt="بعتلك كل ده",
        stage="collecting_details",
        history=[],
    )

    assert stage == "collecting_details"
    assert len(blocks) == 1 and isinstance(blocks[0], TextBlock)
    assert "ما الألوان المطلوبة؟" in blocks[0].text


# --------------------------------------------------------------------------- #
# Regression: Arabic punctuation must not silently discard a cited fact.
#
# `_merge_creative_prompt_fact_updates` drops any update whose `evidence` is not
# a substring of the latest user turn. Before `_normalize_creative_evidence_text`
# the comparison only folded letters, so an Arabic comma/question mark in the
# user's message never matched the ASCII punctuation the model quoted back -- and
# the interview appeared to forget an answer it had already been given.
# --------------------------------------------------------------------------- #


def _facts_for(
    user_turn: str, evidence: str, *, field: str = "style", value: str = "value"
) -> dict[str, dict[str, str]]:
    context: dict[str, object] = {}
    facts, _ = _merge_creative_prompt_fact_updates(
        context,
        {
            "prompt_fact_updates": [
                {"field": field, "value": value, "status": "confirmed", "evidence": evidence}
            ]
        },
        user_turn,
    )
    return facts


def test_arabic_punctuation_variant_still_matches_the_quoted_evidence() -> None:
    cases = [
        # (user wrote, model quoted back)
        ("عايز الصورة بالليل، والألوان دافئة", "عايز الصورة بالليل, والألوان دافئة"),
        ("عايز الصورة بالليل, والألوان دافئة", "عايز الصورة بالليل، والألوان دافئة"),
        ("مين الجمهور المستهدف؟", "مين الجمهور المستهدف?"),
        ("مين الجمهور المستهدف?", "مين الجمهور المستهدف؟"),
        ("الألوان؛ دافئة", "الألوان; دافئة"),
    ]
    for user_turn, evidence in cases:
        facts = _facts_for(user_turn, evidence)
        assert facts.get("style", {}).get("status") == "confirmed", (
            f"evidence {evidence!r} was dropped against turn {user_turn!r}"
        )


def test_quote_dash_and_invisible_directional_marks_are_folded() -> None:
    cases = [
        ("عايز «ستايل مودرن»", '"ستايل مودرن"'),      # guillemets -> ASCII double
        ("عايز “ستايل مودرن”", '"ستايل مودرن"'),      # curly double -> ASCII double
        ("ستايل ‘مودرن’", "'مودرن'"),                  # single quotes survive
        ("ستايل مودرن—فاخر", "ستايل مودرن-فاخر"),      # em dash -> hyphen
        ("ستايل\u200f مودرن", "ستايل مودرن"),          # RLM in the user's turn only
        ("ستايل\u200bمودرن", "ستايل مودرن"),           # zero-width space must not weld words
        ("ستايل مودرن\u200f", "ستايل مودرن"),          # trailing RLM
    ]
    for user_turn, evidence in cases:
        facts = _facts_for(user_turn, evidence)
        assert facts.get("style", {}).get("status") == "confirmed", (
            f"evidence {evidence!r} was dropped against turn {user_turn!r}"
        )


def test_quote_families_are_not_cross_folded() -> None:
    # A single-quoted fragment must not match inside a double-quoted phrase:
    # cross-folding would let the model cite something the user never said.
    assert _normalize_creative_evidence_text("“ستايل مودرن”") == '"ستايل مودرن"'
    assert _normalize_creative_evidence_text("‘ستايل مودرن’") == "'ستايل مودرن'"
    facts = _facts_for("عايز “ستايل مودرن”", "'مودرن'")
    assert "style" not in facts


def test_normalize_creative_evidence_text_folds_punctuation_and_whitespace() -> None:
    assert _normalize_creative_evidence_text("أ، ب؟ ج؛") == "ا, ب? ج;"
    assert _normalize_creative_evidence_text("«اقتباس»") == '"اقتباس"'
    assert _normalize_creative_evidence_text("كلمة\u00a0تانية") == "كلمه تانيه"
    assert _normalize_creative_evidence_text("  متعدد   مسافات  ") == "متعدد مسافات"
    # Letter folding is inherited from the shared helper (ة -> ه, إ -> ا).
    assert _normalize_creative_evidence_text("إحنا بنبيع عبايات") == "احنا بنبيع عبايات"
    assert _normalize_creative_evidence_text("قِطعة") == "قطعه"


def test_regex_metacharacters_stay_literal_in_evidence_normalization() -> None:
    # `؟` folds to `?`, which is a regex quantifier. Evidence matching is a plain
    # substring check, so it must stay literal rather than be compiled.
    facts = _facts_for("المقاس؟", "المقاس?")
    assert facts.get("style", {}).get("status") == "confirmed"

    # A genuine mismatch must still be rejected -- folding must not make
    # everything match.
    facts = _facts_for("عايز الصورة بالليل", "عايز الصورة بالنهار")
    assert "style" not in facts


def test_rejected_updates_are_counted_and_logged_without_user_text(caplog) -> None:
    context: dict[str, object] = {}
    with caplog.at_level("WARNING", logger="app.ai.agents.orchestrator.brain"):
        _merge_creative_prompt_fact_updates(
            context,
            {
                "prompt_fact_updates": [
                    {
                        "field": "palette",
                        "value": "red",
                        "status": "confirmed",
                        "evidence": "نص مش موجود خالص",
                    },
                    {
                        "field": "not_a_real_field",
                        "value": "x",
                        "status": "confirmed",
                        "evidence": "موجود",
                    },
                ]
            },
            "موجود",
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "creative_prompt_fact_updates_rejected" in message for message in messages
    ), messages
    # The user's own wording and the model's quote must never reach the logs.
    for record in caplog.records:
        dumped = str(record.__dict__)
        assert "نص مش موجود خالص" not in dumped
        assert "موجود" not in dumped.replace("creative_prompt_fact_updates_rejected", "")
    assert context.get("creative_prompt_confirmed_facts") == {}



# --- Layla must stay dynamic and friendly, not strict -------------------------
#
# The interview used an exact-match whitelist for approvals and a hardcoded
# vocabulary for "already answered". Both made Layla re-ask questions the user had
# answered and re-present a plan they had already approved, so a real conversation
# could loop at the approval gate.


@pytest.mark.parametrize(
    "message",
    [
        "موافق، اكتب البرومبت",   # the literal UI option label
        "موافق",
        "موافق يا ليلى",
        "موافق، اكتبه",
        "اوك موافق",
        "تمام يا ليلى",
        "ايوه اكتب البرومبت",
        "اعتمد الخطة",
        "موافق على الخطة",
        "يلا بينا",
        "ماشي",
        "ابدئي",
        "go ahead",
        "yes",
        "approve",
        "write the prompt",
    ],
)
def test_approval_is_recognised_in_natural_wording(message: str) -> None:
    from app.ai.agents.orchestrator.brain import _is_explicit_prompt_approval

    assert _is_explicit_prompt_approval(message) is True, message


@pytest.mark.parametrize(
    "message",
    [
        "لا",
        "لأ",
        "مش موافق",
        "موافق بس غيّر اللون",   # an approval word with an edit attached is a revision
        "استنى شوية",
        "غيّر الخلفية",
        "عدّل الألوان",
        "موافق؟",                 # asking, not approving
        "no",
        "wait, change the color",
        "edit the plan",
        "قطع غيار mg",            # an ordinary brief answer is never an approval
    ],
)
def test_revisions_refusals_and_questions_are_not_approvals(message: str) -> None:
    from app.ai.agents.orchestrator.brain import _is_explicit_prompt_approval

    assert _is_explicit_prompt_approval(message) is False, message


def test_spare_parts_and_bare_nothing_answer_their_slots() -> None:
    """"قطع غيار" is a subject and "لا يوجد" answers "anything to avoid?"."""
    from app.ai.agents.orchestrator.brain import _answered_creative_prompt_fields

    ctx: dict[str, object] = {"business_profile_context": {}}
    assert "subject" in _answered_creative_prompt_fields("قطع غيار mg", dict(ctx))
    assert "subject" in _answered_creative_prompt_fields("قطع غيار", dict(ctx))
    assert "avoid" in _answered_creative_prompt_fields("لا يوجد", dict(ctx))
    assert "avoid" in _answered_creative_prompt_fields("مفيش", dict(ctx))
    assert "avoid" in _answered_creative_prompt_fields("none", dict(ctx))


def test_a_stated_exclusion_still_counts_as_a_restriction() -> None:
    """Anchoring matters: "لا يوجد شعار" is a real restriction, not "nothing"."""
    from app.ai.agents.orchestrator.brain import _answered_creative_prompt_fields

    ctx: dict[str, object] = {"business_profile_context": {}}
    for text in ("لا يوجد شعار", "بدون شعار", "ممنوع", "لا تظهر الموتور"):
        assert "avoid" in _answered_creative_prompt_fields(text, dict(ctx)), text


def test_checklist_wording_is_friendly_not_refusing() -> None:
    """The interview must invite the user in, not announce what it refuses to do."""
    source = Path(__file__).resolve().parents[1] / "app/ai/agents/orchestrator/brain.py"
    text = source.read_text(encoding="utf-8")
    for stale in (
        "لن أكتب البرومبت النهائي، ولن أطلب الموافقة على خطة بعد",
        "التفاصيل الناقصة — أجب عمّا لم تذكره بالفعل",
        "أحتاج توضيح هذه النقاط",
    ):
        assert stale not in text, f"strict wording is back: {stale}"


def test_a_written_brief_answers_subject_and_objective_together() -> None:
    """Pasting a brief that states the goal must not be followed by "what is the objective?".

    The user's real transcript: they pasted "أهمية قطع الغيار المضمونة. لماذا تختار قطع
    الغيار المضمونة لسيارتك MG؟ ... استفسر عن توفر قطع الغيار لسيارتك" and Layla still
    asked for the objective, because an awareness objective is written out rather than
    named -- the vocabulary only knew إعلان/حملة/تسويق.
    """
    from app.ai.agents.orchestrator.brain import (
        _answered_creative_prompt_fields,
        _fallback_prompt_questions,
    )

    brief = (
        "أهمية قطع الغيار المضمونة. لماذا تختار قطع الغيار المضمونة لسيارتك MG؟ "
        "استخدام قطع الغيار المناسبة والمضمونة يحمي محرك وأجزاء سيارتك MG من الأعطال المفاجئة. "
        "احصل على قطع غيار أصلية وموثوقة مع ضمان معتمد في MG corner. "
        "استفسر عن توفر قطع الغيار لسيارتك"
    )
    answered = _answered_creative_prompt_fields(brief, {"business_profile_context": {}})
    assert "subject" in answered, answered
    assert "objective" in answered, answered

    questions = _fallback_prompt_questions(
        kind="video", has_reference=False, arabic=True, answered=answered
    )
    assert not any("الهدف" in q for q in questions), questions
    assert not any("المنتج أو الموضوع" in q for q in questions), questions


@pytest.mark.parametrize(
    "text",
    [
        "أهمية الصيانة الدورية",
        "لماذا تختار قطع الغيار المضمونة",
        "ليه تختار القطع الأصلية",
        "مميزات العمرة الكاملة",
        "فوائد استخدام الزيت الأصلي",
        "توعية بأهمية الفحص",
        "الفرق بين الأصلي والتقليد",
        "why choose genuine parts",
        "the benefits of a full overhaul",
    ],
)
def test_awareness_objectives_are_recognised(text: str) -> None:
    from app.ai.agents.orchestrator.brain import _answered_creative_prompt_fields

    assert "objective" in _answered_creative_prompt_fields(
        text, {"business_profile_context": {}}
    ), text


def test_a_brief_without_any_goal_still_asks_for_the_objective() -> None:
    """The guardrail stays: a bare product name does not invent an objective."""
    from app.ai.agents.orchestrator.brain import (
        _answered_creative_prompt_fields,
        _fallback_prompt_questions,
    )

    answered = _answered_creative_prompt_fields("فتيس MG", {"business_profile_context": {}})
    assert "objective" not in answered, answered
    questions = _fallback_prompt_questions(
        kind="image", has_reference=False, arabic=True, answered=answered
    )
    assert any("الهدف" in q for q in questions), questions


@pytest.mark.parametrize(
    "item",
    ["camera_motion", "transitions", "audio/dialogue", "on_screen_text", "end_frame", "cta"],
)
def test_raw_internal_field_names_never_reach_the_user(item: str) -> None:
    """A real model echoed `camera_motion` / `transitions` straight into the Arabic checklist.

    Observed live against Groq's openai/gpt-oss-120b: the list read
    "5. للفيديو: ما المدة…  6. camera_motion  7. transitions".
    """
    from app.ai.agents.orchestrator.brain import _is_raw_field_name_question

    assert _is_raw_field_name_question(item) is True, item


@pytest.mark.parametrize(
    "item",
    [
        "ما المدة المطلوبة؟",
        "How long should the video be?",
        "عدد المشاهد",
        "end card",
        "ما الذي تريد الحفاظ عليه من المرجع؟",
    ],
)
def test_real_questions_are_not_mistaken_for_field_names(item: str) -> None:
    from app.ai.agents.orchestrator.brain import _is_raw_field_name_question

    assert _is_raw_field_name_question(item) is False, item
