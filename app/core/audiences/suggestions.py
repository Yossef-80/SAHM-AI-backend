"""Grounded audience-angle suggestions through the current app.ai provider."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.agents.omar.service import _provider_http_status
from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.ai.security import redact_sensitive_data
from app.core.errors import LLMOutputError, LLMProviderError, NotFoundError
from app.db.models import BrandProfile, Business
from app.schemas.audiences import AudienceSuggestionRequest, AudienceSuggestionResponse

_AUDIENCE_SUGGESTION_PROMPT = """You are Sahm's audience-planning assistant.
Suggest practical audience angles from the supplied business profile and campaign brief.
Respond in the requested language. You are producing editable hypotheses, not confirmed
facts and not a launch-ready Meta audience.

Rules:
- Use only the supplied business/profile/brief facts. Do not invent product features,
  customer demographics, income, audience sizes, market share, performance, or conversion
  outcomes.
- Return 2 or 3 clearly different options. Set approach to `broad` for a broad option and
  `targeting_hypothesis` for categorized targeting ideas. Include one broad option with no
  targeting search terms when evidence is limited; do not force detailed targeting.
- For supported interest, behavior, job title, employer, education, life-event, industry, income,
  family-status, device, or operating-system ideas, return up to 12 short, categorized phrases in
  targeting_search_terms. Use only these exact category keys: interests, behaviors, job_titles,
  employers, education_majors, education_schools, life_events, industries, income,
  family_statuses, devices, operating_systems. Each item is a phrase to search in Meta.
  These are search phrases only: never fabricate Meta IDs, audience sizes, or claim that a phrase
  is available in Meta. Keep the legacy interest_search_terms field empty unless compatibility
  requires it.
- Suggest work, education, life-event, income, and family-status categories only when the supplied
  profile/brief directly supports them. Never infer sensitive personal attributes; prefer broad,
  interest, or behavior ideas when evidence is limited.
- Do not infer or change the user-selected countries. Do not narrow age or gender; this flow
  keeps those settings broad unless the user supplied evidence for a restriction.
- Explain why each option may fit, but do not promise results or assert that it is the best.
- If key details are missing, say what is missing rather than filling it with guesses.
- Do not call tools, browse, create/save audiences, create campaigns, or write to Meta.
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(", ", ": "))


async def generate_audience_suggestions(
    db: AsyncSession,
    *,
    business_id: str,
    request: AudienceSuggestionRequest,
) -> AudienceSuggestionResponse:
    business = await db.get(Business, business_id)
    if business is None:
        raise NotFoundError("business not found")
    profile = await db.scalar(
        select(BrandProfile).where(BrandProfile.business_id == business_id).limit(1)
    )
    context = {
        "language": request.language,
        "campaign": {
            "name": request.campaign_name,
            "objective": request.objective,
            "brief": request.brief or "",
            "user_selected_countries": request.countries,
        },
        "business_profile": {
            "name": str(business.name or "")[:200],
            "industry": str(business.industry or "")[:120],
            "country": str(business.country or "")[:2],
            "target_audience": str(profile.target_audience or "")[:1500] if profile else "",
            "key_offers": [str(item)[:200] for item in (profile.key_offers or [])[:12]] if profile else [],
            "goals": [str(item)[:200] for item in (profile.goals or [])[:12]] if profile else [],
        },
    }
    safe_context = redact_sensitive_data(context)
    provider = get_ai_settings().llm_provider
    try:
        runnable = get_llm(provider=provider).structured(AudienceSuggestionResponse)
        output = await runnable.ainvoke(
            [
                SystemMessage(content=_AUDIENCE_SUGGESTION_PROMPT),
                HumanMessage(
                    content="Draft audience ideas from this JSON. Phrases are not Meta IDs.\n"
                    + _json(safe_context)
                ),
            ]
        )
    except Exception as exc:
        status_code = _provider_http_status(exc)
        detail: dict[str, object] = {"stage": "audience_suggestions"}
        if status_code is not None:
            detail["provider_status_code"] = status_code
        raise LLMProviderError(
            "Audience suggestions are temporarily unavailable; you can still choose a saved or broad audience.",
            detail=detail,
            retryable=status_code is None or status_code in {408, 425, 429, 500, 502, 503, 504},
            status_code=status_code,
        ) from exc

    try:
        result = (
            output
            if isinstance(output, AudienceSuggestionResponse)
            else AudienceSuggestionResponse.model_validate(output)
        )
    except Exception as exc:
        raise LLMOutputError(
            "The audience assistant returned suggestions in an unexpected format.",
            detail={"stage": "audience_suggestions"},
        ) from exc

    if not result.suggestions:
        raise LLMOutputError(
            "The audience assistant did not return any suggestions.",
            detail={"stage": "audience_suggestions"},
        )
    return result
