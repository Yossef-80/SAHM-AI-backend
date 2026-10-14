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

from app.ai.security import redact_credential_text, redact_sensitive_data
from app.schemas.common import ToolSideEffect

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
    r"(?i)(?:\bgenerate\s+(?:an?\s+)?image\b|\bcreate\s+(?:an?\s+)?image\b|"
    r"\brender\s+(?:an?\s+)?image\b|\bmake\s+(?:an?\s+)?image\b|"
    r"انشئ\s+صورة|أنشئ\s+صورة|ولّد\s+صورة|ولد\s+صورة|اعمل\s+صورة|طلع\s+الصورة)"
)
_PROMPT_ONLY = re.compile(
    r"(?i)(?:\bprompt\s+only\b|\bwrite\s+(?:me\s+)?a\s+prompt\b|"
    r"\bimage\s+prompt\b|\bvideo\s+prompt\b|اكتب\s+(?:لي\s+)?(?:برومبت|prompt)|"
    r"برومبت\s+فقط|صياغة\s+البرومبت)"
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
        if upload_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(upload_id)):
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


def _make_langchain_tool(core_tool: Any) -> StructuredTool:
    async def _invoke(config: RunnableConfig, **kwargs: Any) -> str:
        configurable = (config or {}).get("configurable") or {}
        context = configurable.get("tool_context")
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
