"""Narrow adapters from the shared typed tools into the app.ai LangGraphs.

The domain handlers remain in ``app.core.tools``. This file only adapts their
Pydantic contracts and ToolContext to LangChain tools so the conversation graphs
can use the same implementations without duplicating business logic.
"""

from __future__ import annotations

import json
import re
import time
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from app.ai.loggers.callbacks import AIToolAuditCallback
from app.ai.security import redact_credential_text, redact_sensitive_data
from app.core.logging import get_logger
from app.schemas.common import ToolSideEffect

_logger = get_logger(__name__)

# Keep this allowlist explicit. It mirrors the supported read/draft capabilities
# for each conversation agent and deliberately excludes every external-write tool.
_AGENT_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "nour": ("get_social_profile", "get_keywords", "get_competitor_ads"),
    "layla": ("analyze_reference_creative", "generate_image"),
    "karim": ("list_meta_campaigns", "get_campaign_insights", "get_breakdowns"),
    "salma": ("get_campaign_insights", "get_breakdowns", "create_recommendation"),
}
_CORE_TOOL_NAMES = {name for names in _AGENT_TOOL_NAMES.values() for name in names}
_REFERENCE_INTENT = re.compile(
    r"(?i)(?:\breference\b|\binspiration\b|\bstyle\s+like\b|\buse\s+this\s+as\s+a\s+reference\b|"
    r"مرجع|كمرجع|استلهام|إلهام|زي\s+الصورة|مثل\s+الصورة|على\s+نمط)"
)
_GENERATE_INTENT = re.compile(
    # English imperatives, then the Arabic forms people actually type. The Arabic
    # side is deliberately broad about clitics -- "اعملي الصورة", "صمميلي صورة",
    # "صمميها", "طلّعيها" are all one request -- because a miss here is expensive:
    # the image tool silently refuses, and the user pays for a prompt they already
    # approved instead of getting the picture. The noun list is what stops this
    # from firing on an unrelated "اعملي حملة".
    r"(?i)(?:\bgenerate\s+(?:an?\s+)?image\b|\bcreate\s+(?:an?\s+)?image\b|"
    r"\brender\s+(?:an?\s+)?image\b|\bmake\s+(?:an?\s+)?image\b|"
    r"\b(?:generate|create|make|produce)\s+(?:the\s+)?(?:picture|design)\b|"
    # Masdar/verbal-noun phrasing ("توليد صورة", "إنشاء صورة", "تصميم صورة") has no
    # imperative verb for the branch below to latch onto, and it is how a brief is
    # usually titled.
    r"(?:توليد|انتاج|إنتاج|إنشاء|انشاء|صياغة|صناعة)\s*(?:ال)?"
    r"(?:صور(?:ة|تين)?|تصميم|بوستر|بنر|بانر|إعلان|اعلان|كرييتف|كرياتيف)|"
    r"(?:اعمل|أعمل|إعمل|اعملى|صمم|صمّم|صممى|نفذ|نفّذ|ولد|ولّد|انشئ|أنشئ|انشاء|اطلع|طلّع|طلع|سوي|سوّي|رندر|حوّل|حول)"
    r"(?:يها|ها|يه|ه|ي|لي|لى|لنا|لهم|وها)?"
    r"\s*(?:لي\s*)?(?:ال)?"
    r"(?:صور(?:ة|ه|تين)?|تصميم|ديزاين|بوست(?:ر)?|بنر|بانر|إعلان|اعلان|أعلان|كرييتف|كرياتيف|فيزوال|فيسوال)"
    r"|"
    # Bare object-pronoun form: "صمميها" / "اعمليها" / "نفذيها". The thing being
    # referred to is the deliverable already on the table, so the verb alone
    # carries the intent -- unless the object is spelled out as something that is
    # explicitly not an image.
    # The trailing ``\b`` is load-bearing: without it the engine can satisfy the
    # pronoun with the lone ``ه`` of "اعملها" and then the lookahead inspects the
    # leftover "ا حملة" -- which is not the blocked word, so "اعملها حملة" used to
    # read as an image request.
    r"(?:صمم|صمّم|نفذ|نفّذ|ولد|ولّد|اعمل|سوي|سوّي|طلع|طلّع|اطلع)(?:يها|ها|يه|ه)\b"
    r"(?!\s*(?:حملة|حمله|استراتيجية|إستراتيجية|تقرير|خطة|نص|كوبي|كابشن|ايميل|إيميل|سكريبت|جدول|بريف|برومبت|فيديو|ريل|موشن|انيميشن|أنيميشن|صوت|موسيقى))"
    r")"
)
_PROMPT_ONLY = re.compile(
    r"(?i)(?:\bprompt\s+only\b|\bwrite\s+(?:me\s+)?a\s+prompt\b|"
    r"\bimage\s+prompt\b|\bvideo\s+prompt\b|اكتب\s+(?:لي\s+)?(?:برومبت|prompt)|"
    r"برومبت\s+فقط|برومبت\s+بس|صياغة\s+البرومبت|"
    r"من\s+غير\s+(?:ما\s+)?(?:صورة|تصميم|تولّد|تولد|تعمل)|من\s+دون\s+(?:صورة|تصميم))"
)


def _tool_map() -> dict[str, Any]:
    # Imports are lazy so app.ai can be imported without registering unrelated
    # tool integrations until a graph needs these adapters.
    from app.core.tools.registry import ALL_TOOLS

    return {item.name: item for item in ALL_TOOLS}


def _unavailable(tool: Any, reason: str) -> str:
    output = tool.output_model.unavailable(
        source=f"agent_bridge:{tool.name}",
        reason=reason,
    )
    return json.dumps(output.model_dump(mode="json"), ensure_ascii=False)


#: Words that send the user back to the picture they already chose: "اعمل تصميم
#: زي ده بالظبط" is about the active reference. A request for something new
#: ("من الأول", "فكرة تانية") is not, and must never inherit it silently.
_DEICTIC_REFERENCE = re.compile(
    r"(?:ز[يى]|ذ[يى])\s*(?:ده|دا|دي|دى|كده|كدا)"
    r"|(?:ز[يى]|على\s*غرار|نفس)\s*(?:الصورة|التصميم|الشكل|الستايل|اللوجو|الهوية)"
    r"|ز[يى]\s*اللي\s*(?:بعتته|فوق|تحت|عجبك|عملته)"
    r"|\b(?:like|same as|similar to|based on)\s+(?:this|that|the\s+(?:above|reference|image|design|one))\b",
    re.IGNORECASE,
)


#: Tools that produce a paid picture. One per turn is the product rule ("كفاية
#: نسخة واحدة"): the user asked for an image, not for a bill for two.
_IMAGE_TOOLS = frozenset({"generate_image", "generate_image_draft_tool"})


def _image_already_produced(config: Any) -> bool:
    """Whether this very run already generated a picture.

    Asked of the run's tool audit, which is one object for the whole graph run --
    including the second attempt after a provider refusal. Only *completed* calls
    count: the call asking this question has already started, so a started-set
    check would refuse every first image.
    """
    callbacks = (config or {}).get("callbacks")
    if isinstance(callbacks, (list, tuple)):
        handlers = list(callbacks)
    else:
        handlers = list(getattr(callbacks, "handlers", []) or [])
    for handler in handlers:
        if not isinstance(handler, AIToolAuditCallback):
            continue
        if set(handler.tools_completed()) & _IMAGE_TOOLS:
            return True
    return False


def _context_reference_default(name: str, payload: Any, context: Any) -> Any:
    """Give ``generate_image`` the conversation's reference when the model forgot it.

    The model is told the active ``reference_analysis_id`` in its prompt, but a
    small model does occasionally drop it. Dropping it silently produced a
    from-scratch image the user immediately rejected as "مش زي اللي بعتّه" --
    a wrong picture is worse than a missing one, so this fills the gap only when
    all three of these hold: the tool is ``generate_image``, the model passed no
    reference at all, and the user's own words point back at the picture.
    """
    if name != "generate_image" or payload is None:
        return payload
    if getattr(payload, "reference_analysis_id", None) or getattr(
        payload, "reference_image_url", None
    ):
        return payload
    supplied = getattr(context, "user_supplied", {}) or {}
    if not isinstance(supplied, dict):
        return payload
    reference_id = str(supplied.get("reference_analysis_id") or "").strip()
    latest = str(supplied.get("latest_user_message") or "")
    if not reference_id or not _DEICTIC_REFERENCE.search(latest):
        return payload
    try:
        payload = payload.model_copy(update={"reference_analysis_id": reference_id})
    except Exception:  # noqa: BLE001 - an unexpected payload shape stays untouched
        return payload
    _logger.info(
        "generate_image_reference_defaulted",
        extra={"reference_analysis_id": reference_id},
    )
    return payload


def _validate_explicit_intent(name: str, context: Any) -> str | None:
    """Block generation/reference analysis unless the user asked for that action."""
    supplied = getattr(context, "user_supplied", {}) or {}
    latest = str(supplied.get("latest_user_message") or "") if isinstance(supplied, dict) else ""
    # An explicit picker beats the keyword guess. The regexes below are a
    # fallback for clients that send no mode, not the contract: without this a
    # design request phrased as "عايز تصميم للبوتاجاز ده" matched _GENERATE_INTENT
    # nowhere and generation was refused even though the user had asked for it.
    mode = (
        str(supplied.get("creative_mode") or "").strip().lower()
        if isinstance(supplied, dict)
        else ""
    )

    if (
        name == "analyze_reference_creative"
        and mode not in {"prompt", "design"}
        and not _REFERENCE_INTENT.search(latest)
    ):
        return (
            "No explicit reference-creative request was detected. Do not analyze a final uploaded ad. "
            "Ask the user whether this asset is a reference before requesting analysis."
        )
    if name == "generate_image":
        # `prompt` mode must still refuse: the user asked for wording, not pixels.
        if mode == "prompt" or (
            mode != "design"
            and (not _GENERATE_INTENT.search(latest) or _PROMPT_ONLY.search(latest))
        ):
            return (
                "Image generation was not explicitly requested. Return the prompt/brief and ask what "
                "changes the user wants; do not create a placeholder or image asset."
            )
    return None


async def _validate_tool_action(name: str, payload: BaseModel, context: Any) -> str | None:
    """Enforce approval, tenant, and no-mock/no-implicit-generation rules."""
    supplied = getattr(context, "user_supplied", {}) or {}
    if isinstance(supplied, dict) and supplied.get("competitor_ads_only"):
        if name != "get_competitor_ads":
            return "This restricted ToolContext is only valid for the explicitly requested competitor-Ad-Library lookup."
    elif name == "get_competitor_ads":
        return "The competitor-Ad-Library lookup requires its explicit user-request context."

    if name == "analyze_reference_creative":
        upload_id = getattr(payload, "upload_id", None)
        if upload_id:
            # The allow-list has to accept a dot: a stored reference carries its
            # extension (`reference_<hex>.png`). Without it the id this codebase
            # itself hands out was rejected as "invalid" before the file was even
            # opened -- so a perfectly good reference could never be analysed.
            # Traversal is still refused here as well as in the tool, which
            # resolves the path and re-checks the upload root.
            candidate = str(upload_id)
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", candidate) or ".." in candidate:
                return "The reference upload id is invalid; no file was opened."
        if not upload_id and not getattr(payload, "image_url", None):
            return "Provide a reference upload or public image URL explicitly."

    if name == "generate_image":
        cfg = context.require_settings()
        from app.core.tools.settings import resolve_tool_settings

        resolved = await resolve_tool_settings(context)
        provider = str(resolved.image_provider or cfg.image_provider or "mock").strip().lower()
        image_key = resolved.image_api_key or cfg.image_api_key
        supplied = getattr(context, "user_supplied", {}) or {}
        latest = str(supplied.get("latest_user_message") or "") if isinstance(supplied, dict) else ""
        explicitly_mock = bool(re.search(r"(?i)\bmock\b|placeholder|نموذج تجريبي", latest))
        allow_mock = bool(supplied.get("allow_mock_assets", False)) if isinstance(supplied, dict) else False
        if provider == "mock" and not explicitly_mock and not allow_mock:
            return (
                "No real image provider is configured. Mock placeholders are disabled by default; "
                "return the prompt and ask the user to configure/select an image provider."
            )
        if provider == "openai" and not image_key:
            return "The selected image provider is missing its workspace credential; no request was sent."
        if provider not in {"openai", "mock"}:
            return f"The selected image provider '{provider}' is not enabled in this build."

    if name == "create_recommendation":
        evidence = getattr(payload, "evidence", None) or []
        if not evidence or any(
            not str(getattr(item, "source", "") or "").strip()
            or not str(getattr(item, "detail", "") or "").strip()
            for item in evidence
        ):
            return "A recommendation requires at least one source-linked evidence item; no draft was saved."
        campaign_id = getattr(payload, "campaign_id", None) or getattr(context, "campaign_id", None)
        if not campaign_id:
            return "Choose a campaign before creating a recommendation; no draft was saved."
        context_campaign = getattr(context, "campaign_id", None)
        if context_campaign and getattr(payload, "campaign_id", None) and payload.campaign_id != context_campaign:
            return "The requested campaign does not match the conversation context; no draft was saved."
        from app.db.models import Campaign

        campaign = await context.db.get(Campaign, campaign_id)
        if campaign is None or campaign.business_id != context.business_id:
            return "The campaign was not found in this workspace; no draft was saved."
    return None


def _core_tool_context(configurable: dict[str, Any]) -> Any | None:
    """The ToolContext this run's core tools must use, or None.

    Two sources, in this order:

    1. ``tool_context`` -- an explicitly constructed context. The competitor-Ads
       route builds one directly because it must be *restricted* to that one
       lookup.
    2. The per-run keys the orchestrator always attaches (``tool_db``,
       ``business_id``, ...). Without this every allowlisted core tool answered
       "This conversation run did not provide a workspace tool context" -- which
       is exactly why Karim could not read a single campaign in chat while the
       campaigns page showed campaigns with spend and metrics. The tools were
       never broken; they simply never received a workspace.
    """
    context = configurable.get("tool_context")
    if context is not None:
        return context

    db = configurable.get("tool_db")
    if db is None:
        return None

    from app.core.tools.base import ToolContext

    core_llm = configurable.get("core_llm")
    user_supplied = configurable.get("user_supplied")
    return ToolContext(
        workspace_id=str(configurable.get("workspace_id") or ""),
        business_id=str(configurable.get("business_id") or ""),
        db=db,
        llm=core_llm,
        fast_llm=configurable.get("fast_llm") or core_llm,
        request_id=str(configurable.get("request_id") or "ai-chat"),
        conversation_id=configurable.get("conversation_id"),
        persona_id=str(configurable.get("agent") or "") or None,
        campaign_id=str(configurable.get("campaign_id") or "") or None,
        creative_id=str(configurable.get("creative_id") or "") or None,
        insight_id=str(configurable.get("insight_id") or "") or None,
        user_supplied=dict(user_supplied) if isinstance(user_supplied, dict) else {},
    )


def _make_langchain_tool(core_tool: Any) -> StructuredTool:
    async def _invoke(config: RunnableConfig, **kwargs: Any) -> str:
        configurable = (config or {}).get("configurable") or {}
        context = _core_tool_context(configurable)
        if context is None:
            return _unavailable(
                core_tool,
                "This conversation run did not provide a workspace tool context; no data or action was attempted.",
            )

        intent_error = _validate_explicit_intent(core_tool.name, context)
        if intent_error:
            return _unavailable(core_tool, intent_error)

        started = time.monotonic()
        payload: BaseModel | None = None
        result: BaseModel | None = None
        status = "error"
        error_text: str | None = None
        try:
            payload = core_tool.input_model.model_validate(kwargs)
            # One image per turn. A second call in the same run is refused *before*
            # the provider is reached, so a duplicate can never be billed.
            if core_tool.name in _IMAGE_TOOLS and _image_already_produced(config):
                _logger.info(
                    "image_generation_duplicate_refused",
                    extra={"tool": core_tool.name, "reason": "one image per turn"},
                )
                return _unavailable(
                    core_tool,
                    "An image was already generated in this turn. Refer to that one "
                    "instead of generating another; do not call this tool again.",
                )
            # The conversation's reference is a default, not an override: an
            # explicit choice by the model always wins.
            payload = _context_reference_default(core_tool.name, payload, context)
            policy_reason = await _validate_tool_action(core_tool.name, payload, context)
            if policy_reason:
                result = core_tool.output_model.unavailable(
                    source=f"agent_bridge:{core_tool.name}",
                    reason=policy_reason,
                )
                status = "blocked"
            else:
                result = await core_tool.handler(payload, context)
                if not isinstance(result, core_tool.output_model):
                    result = core_tool.output_model.model_validate(result)
                status_value = getattr(result, "status", "ok")
                status = str(getattr(status_value, "value", status_value))
            serialized = json.dumps(
                redact_sensitive_data(result.model_dump(mode="json")),
                ensure_ascii=False,
                default=str,
            )
        except Exception as exc:  # keep a tool failure from taking down the whole graph
            error_text = f"{type(exc).__name__}: {redact_credential_text(str(exc))}"
            serialized = _unavailable(
                core_tool,
                "The tool could not complete this request. No result should be inferred.",
            )
        finally:
            try:
                from app.core.tools.base import log_tool_call

                await log_tool_call(
                    context,
                    tool=core_tool.name,
                    tool_input=payload or kwargs,
                    output_summary=(serialized if "serialized" in locals() else ""),
                    status=status,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    error=error_text,
                )
            except Exception:
                # Tool-call telemetry must never turn an otherwise valid result
                # into a failed conversation response.
                pass
        return serialized

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=core_tool.name,
        description=core_tool.description,
        args_schema=core_tool.input_model,
    )


def get_core_tools_for_agent(agent_name: str) -> list[StructuredTool]:
    """Return adapters for one agent's explicitly allowlisted core tools."""
    requested = _AGENT_TOOL_NAMES.get(agent_name, ())
    registry = _tool_map()
    tools: list[StructuredTool] = []
    for name in requested:
        core_tool = registry.get(name)
        if core_tool is None:
            raise RuntimeError(f"AI bridge tool '{name}' is not registered")
        if core_tool.side_effect == ToolSideEffect.EXTERNAL_WRITE:
            raise RuntimeError(f"AI bridge refuses external-write tool '{name}'")
        tools.append(_make_langchain_tool(core_tool))
    return tools


def persist_ai_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """Persist app.ai-native tool inputs/results alongside main chat messages.

    Core tools already record their own invocation through ``log_tool_call``;
    submit tools are represented by the assistant response and are intentionally
    not duplicated here. Tools using LangGraph's injected state/tool-call ID are
    left untouched; the graph-level audit callback records those calls without
    breaking their Command/state-injection contract.
    """
    def _uses_graph_injection(tool_obj: BaseTool) -> bool:
        function = getattr(tool_obj, "func", None)
        if function is None:
            return False
        try:
            hints = get_type_hints(function, include_extras=True)
        except (NameError, TypeError):
            return False
        for annotation in hints.values():
            if get_origin(annotation) is not Annotated:
                continue
            metadata = get_args(annotation)[1:]
            if any(getattr(item, "__name__", "") in {"InjectedToolCallId", "InjectedState"} for item in metadata):
                return True
        return False

    wrapped: list[BaseTool] = []
    for original in tools:
        if (
            original.name in _CORE_TOOL_NAMES
            or original.name.startswith("submit_")
            or _uses_graph_injection(original)
        ):
            wrapped.append(original)
            continue
        async def _invoke(config: RunnableConfig, _tool: BaseTool = original, **kwargs: Any) -> Any:
            context = ((config or {}).get("configurable") or {}).get("tool_context")
            started = time.monotonic()
            status = "ok"
            error: str | None = None
            try:
                output = await _tool.ainvoke(kwargs, config=config)
                summary = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
                summary = str(redact_sensitive_data(summary))[:2000]
                if isinstance(summary, str) and summary.lstrip().startswith("{"):
                    try:
                        result_obj = json.loads(summary)
                        status = str(result_obj.get("status", "ok")) if isinstance(result_obj, dict) else "ok"
                    except ValueError:
                        pass
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {redact_credential_text(str(exc))}"
                raise
            finally:
                if context is not None:
                    try:
                        from app.core.tools.base import log_tool_call

                        await log_tool_call(
                            context,
                            tool=_tool.name,
                            tool_input=redact_sensitive_data(kwargs),
                            output_summary=(summary if "summary" in locals() else ""),
                            status=status,
                            duration_ms=int((time.monotonic() - started) * 1000),
                            error=error,
                        )
                    except Exception:
                        pass
            return output

        adapted = StructuredTool.from_function(
            coroutine=_invoke,
            name=original.name,
            description=original.description,
            args_schema=original.args_schema,
        )
        wrapped.append(adapted)
    return wrapped


__all__ = ["get_core_tools_for_agent", "persist_ai_tools"]
