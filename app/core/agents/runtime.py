"""Layer 5: the generic agent runtime.

ONE loop, configured by persona objects. No per-persona code.

    run_agent(persona, task_name, inputs, ctx) -> AgentResult

Loop:
  1. validate the task is on the persona's allowlist
  2. fetch brand context if the task needs it
  3. build the prompt (build_prompt -- the only prompt assembler)
  4. up to max_iterations: LLM -> tool calls (allowlisted only) -> repeat
  5. validate the final structured output (one repair retry)
  6. report the autonomy policy's outcome
  7. return AgentResult

The autonomy policy is enforced in **two** places, both before any handler runs:
the candidate tool list is filtered (a ``SUGGEST`` persona is never *offered* a
durable-write tool) and ``_run_one_tool`` re-checks it (a ``SUGGEST`` persona is
never able to *run* one, even if the model asks for it anyway). Step 6 only
reports what already happened; see ``blocks_draft_write`` and D1.

Hard caps (iterations, tokens, wall clock) always produce a partial result with
``incomplete=True`` -- a run never hangs.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.core.agents.persona import Persona
from app.core.agents.tasks import TASK_SPECS, TaskSpec, build_prompt, get_task
from app.core.errors import (
    LLMError,
    PolicyViolation,
    TaskNotAllowedError,
    ToolNotAllowedError,
    ToolValidationError,
)
from app.core.llm.base import LLMClient, LLMResult, Msg, ToolCall, format_validation_error
from app.core.llm.router import get_llm
from app.core.logging import get_logger, redact
from app.core.tools.base import Tool, ToolContext, log_tool_call
from app.core.tools.registry import get_tools_for
from app.core.usage import RunUsageTracker, check_budget, record_usage
from app.schemas.agent import AgentEvent, AgentResult, ToolCallRecord
from app.schemas.common import Autonomy, ToolStatus, canonicalize_url
from app.schemas.tools import BrandContext, MemoryScope

_logger = get_logger(__name__)


def _llm_failure_message(exc: LLMError) -> str:
    """Surface provider validation errors without exposing request secrets."""
    status = getattr(exc, "status_code", None)
    detail = getattr(exc, "detail", {})
    body = detail.get("body") if isinstance(detail, dict) else None
    provider_message = ""
    if isinstance(body, str):
        try:
            payload = json.loads(body)
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            if isinstance(error, dict):
                provider_message = str(error.get("message") or error.get("code") or "")
            elif error:
                provider_message = str(error)
        except (TypeError, ValueError):
            pass

    message = exc.message
    if status is not None:
        message = f"{message} (HTTP {status})"
    if provider_message:
        message = f"{message}: {provider_message[:300]}"
    return message


# ------------------------------------------------------------------- options --


@dataclass
class RunOptions:
    """Hard caps for one run. All values come from config by default."""

    max_iterations: int = 6
    max_tokens_per_run: int = 60_000
    timeout_seconds: float = 180.0
    daily_token_budget: int = 2_000_000
    #: Set False in tests that want to skip tool calls entirely.
    allow_tool_calls: bool = True
    temperature: float = 0.4
    max_tokens_per_call: int = 2000

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RunOptions":
        cfg = settings or get_settings()
        return cls(
            max_iterations=cfg.max_agent_iterations,
            max_tokens_per_run=cfg.max_tokens_per_run,
            timeout_seconds=cfg.agent_run_timeout_seconds,
            daily_token_budget=cfg.daily_token_budget,
        )


@dataclass
class RunDependencies:
    """Injected collaborators. Everything the loop touches, nothing ambient."""

    llm: LLMClient
    db: Any
    settings: Settings
    workspace_id: str = "default"
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:16]}")


# ------------------------------------------------------------ autonomy policy --

#: What each autonomy level is allowed to persist. Enforced here, never in a
#: prompt. See docs/DECISIONS.md.
_AUTONOMY_DESCRIPTION = {
    Autonomy.SUGGEST: "proposal only; nothing durable is created from the output",
    Autonomy.AUTO_DRAFT: "may persist drafts; can never trigger an external write",
    Autonomy.ASK_FIRST: "any side effect requires an approval record before execution",
}


def describe_autonomy(autonomy: Autonomy) -> str:
    return _AUTONOMY_DESCRIPTION[autonomy]


def blocks_draft_write(persona: Persona, tool_name: str) -> bool:
    """True when this persona must not run ``tool_name`` because it writes durably.

    This is the *gate*, and it runs before the handler does. D1 in the roadmap was
    that the policy was computed after the run loop had already executed the
    write-capable handlers, so a ``SUGGEST`` persona -- documented as "proposal
    only; nothing durable is created from the output" -- still persisted a
    ``learning`` row while the result claimed "nothing durable created". The fix
    is not a better report; it is refusing the write.
    """
    return persona.autonomy == Autonomy.SUGGEST and tool_name in _DRAFT_WRITE_TOOLS


def _apply_autonomy_policy(
    persona: Persona,
    task: TaskSpec,
    *,
    tool_calls: list[ToolCallRecord],
    output: dict[str, Any] | None,
) -> tuple[list[str], bool]:
    """Report what the run actually wrote. Never decides after the fact.

    Returns ``(persisted, requires_approval)``.

    - suggest: nothing durable -- because the handlers never ran
      (:func:`blocks_draft_write`). This is a *report* of an empty set rather
      than a decision made too late.
    - auto_draft: drafts may be persisted. External writes are structurally
      impossible (the registry refuses external_write tools), so there is nothing
      to guard against here beyond recording what was written.
    - ask_first: anything with a side effect stays pending and the result is
      flagged as requiring an approval record before execution.

    D1 regression guard: if a ``SUGGEST`` persona somehow produced a call to one
    of the draft-write tools, that is a bug in the gate, not something to paper
    over with a summary -- so it raises instead of quietly returning ``[]``.
    """
    persisted: list[str] = []
    requires_approval = False
    # ``status == "blocked"`` means the gate refused it and the handler never ran,
    # so such a call is evidence the policy worked, not evidence it was broken.
    side_effecting = [
        tc
        for tc in tool_calls
        if tc.tool in _DRAFT_WRITE_TOOLS and tc.status != "blocked"
    ]

    if persona.autonomy == Autonomy.SUGGEST:
        if side_effecting:
            raise PolicyViolation(
                "a suggest-autonomy persona reached a durable-write tool; the "
                "autonomy gate in _run_one_tool should have blocked it",
                detail={"tools": sorted({tc.tool for tc in side_effecting})},
            )
        return [], False

    if persona.autonomy == Autonomy.AUTO_DRAFT:
        if output is not None:
            persisted.append(f"task_output:{task.name}")
        for tc in side_effecting:
            persisted.append(f"tool:{tc.tool}")
        return persisted, False

    # ask_first
    requires_approval = True
    for tc in side_effecting:
        persisted.append(f"pending:{tc.tool}")
    return persisted, requires_approval


#: Tools whose handler writes to our own database.
_DRAFT_WRITE_TOOLS = frozenset(
    {"save_learning", "create_recommendation", "analyze_reference_creative", "generate_image"}
)


# ------------------------------------------------------------------- helpers --


def _tool_result_summary(output: Any) -> str:
    """Short, log-friendly rendering of a tool output."""
    try:
        if hasattr(output, "model_dump"):
            data = output.model_dump(mode="json", exclude_none=True)
        elif isinstance(output, dict):
            data = output
        else:
            return str(output)[:500]
        status = data.get("status", "ok")
        source = data.get("source", "")
        reason = data.get("unavailable_reason")
        base = f"status={status} source={source}"
        if reason:
            base += f" reason={reason}"
        # Include the most informative field without dumping everything.
        for key in ("results", "ads", "reviews", "keywords", "items", "assets", "findings"):
            if isinstance(data.get(key), list):
                base += f" {key}={len(data[key])}"
                break
        else:
            if isinstance(data.get("summary"), str) and data["summary"]:
                base += f" summary={data['summary'][:160]}"
        return base[:600]
    except Exception:  # pragma: no cover - summarising must never fail
        return "(summary unavailable)"


def _collect_unavailable(
    tool_calls: list[ToolCallRecord], outputs: list[Any]
) -> list[str]:
    """Names of sources that came back unavailable, de-duplicated."""
    unavailable: list[str] = []
    for record, output in zip(tool_calls, outputs):
        status = getattr(output, "status", None)
        if status == ToolStatus.UNAVAILABLE:
            source = getattr(output, "source", "") or record.tool
            reason = getattr(output, "unavailable_reason", "") or ""
            entry = f"{record.tool} ({source}): {reason}" if reason else f"{record.tool} ({source})"
            if entry not in unavailable:
                unavailable.append(entry)
    return unavailable


_RESEARCH_DATA_TOOLS = frozenset(
    {"web_search", "analyze_website", "get_social_profile", "get_reviews", "get_keywords", "get_competitor_ads"}
)


def _has_research_data(tool_calls: list[ToolCallRecord], outputs: list[Any]) -> bool:
    """Whether a research tool returned actual records rather than status alone."""
    for record, output in zip(tool_calls, outputs):
        if record.tool not in _RESEARCH_DATA_TOOLS or record.status not in ("ok", "partial"):
            continue
        try:
            data = output.model_dump(mode="json", exclude_none=True) if hasattr(output, "model_dump") else output
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if any(data.get(key) for key in ("results", "ads", "reviews", "keywords", "profile", "findings")):
            return True
        if record.tool == "analyze_website" and any(data.get(key) for key in ("title", "description", "headings")):
            return True
    return False


def _competitor_research_evidence(
    tool_calls: list[ToolCallRecord], outputs: list[Any]
) -> list[dict[str, Any]]:
    """Keep source-backed results available if the model cannot synthesize JSON."""
    evidence: list[dict[str, Any]] = []
    for record, output in zip(tool_calls, outputs):
        if record.status not in ("ok", "partial"):
            continue
        try:
            payload = output.model_dump(mode="json", exclude_none=True)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        provider_source = str(payload.get("source") or record.tool)
        source = (
            provider_source
            if provider_source.startswith(record.tool)
            else f"{record.tool}:{provider_source}"
        )
        if record.tool == "web_search":
            results = payload.get("results")
            for result in (results[:6] if isinstance(results, list) else []):
                if not isinstance(result, dict):
                    continue
                title = str(result.get("title") or "Search result").strip()
                snippet = str(result.get("snippet") or "").strip()
                evidence.append(
                    {
                        "source": source,
                        "detail": f"{title}: {snippet}"[:700],
                        "url": canonicalize_url(result.get("url")) if result.get("url") else None,
                    }
                )
        elif record.tool == "analyze_website":
            parts = [
                str(payload.get("title") or "").strip(),
                str(payload.get("description") or "").strip(),
                str(payload.get("summary") or "").strip(),
            ]
            offers = payload.get("offers") or []
            if isinstance(offers, list):
                parts.extend(
                    str(item.get("name") or item.get("detail") or "").strip()
                    for item in offers[:4]
                    if isinstance(item, dict)
                )
            detail = " — ".join(part for part in parts if part)
            if detail:
                evidence.append(
                    {
                        "source": source,
                        "detail": detail[:900],
                        "url": canonicalize_url(payload.get("url")) if payload.get("url") else None,
                    }
                )
    return evidence


def _tool_results_block(tool_calls: list[ToolCallRecord], outputs: list[Any]) -> str:
    """Render a bounded tool-results block for the final prompt."""
    if not tool_calls:
        return ""
    # Search results can contain long snippets and large website extracts. Keep
    # enough evidence for synthesis while protecting providers with low TPM
    # limits from oversized final requests.
    max_chars = 10_000
    lines: list[str] = []
    remaining = max_chars
    for record, output in zip(tool_calls, outputs):
        try:
            payload = (
                output.model_dump(mode="json", exclude_none=True)
                if hasattr(output, "model_dump")
                else output
            )
        except Exception:  # pragma: no cover
            payload = {"summary": record.output_summary}
        import json

        payload = _compact_tool_payload(record.tool, payload)
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
        section = f"### {record.tool}\n{rendered}"
        if len(section) > remaining:
            marker = "\n[Tool results truncated to fit the model context budget.]"
            section = section[: max(0, remaining - len(marker))] + marker
        if section:
            lines.append(section)
            remaining -= len(section) + 2
        if remaining <= 0:
            break
    return "\n\n".join(lines)


def _compact_tool_payload(tool_name: str, payload: Any) -> Any:
    """Keep provider responses useful while bounding their LLM-facing size."""
    if not isinstance(payload, dict) or tool_name != "web_search":
        return payload
    compact = dict(payload)
    results = compact.get("results")
    if isinstance(results, list):
        compact["results"] = [
            {
                **({key: value for key, value in item.items() if key != "snippet"}),
                "snippet": str(item.get("snippet") or "")[:500],
            }
            for item in results[:4]
            if isinstance(item, dict)
        ]
    return compact


def _bound_tool_transcript(messages: list[Msg], max_chars: int = 7_000) -> list[Msg]:
    """Keep recent tool evidence and compact older tool replies for next calls."""
    remaining = max_chars
    bounded = list(messages)
    for index in range(len(bounded) - 1, -1, -1):
        msg = bounded[index]
        if msg.role != "tool":
            continue
        content = msg.content or ""
        if len(content) <= remaining:
            remaining -= len(content)
            continue
        if remaining > 120:
            clipped = content[: remaining - 100] + " [older tool evidence compacted]"
            remaining = 0
        else:
            clipped = '{"note":"Earlier tool result retained for final synthesis."}'
        bounded[index] = Msg(
            role="tool",
            content=clipped,
            tool_call_id=msg.tool_call_id,
            name=msg.name,
        )
    return bounded


# ----------------------------------------------------------------- the loop --


async def _execute(
    persona: Persona,
    task_name: str,
    inputs: dict[str, Any],
    ctx: ToolContext,
    deps: RunDependencies,
    options: RunOptions,
    *,
    user_message: str = "",
) -> AsyncGenerator[AgentEvent, None]:
    """The single implementation of the run loop. Yields streaming events.

    The final event is always ``completed`` or ``failed`` and carries the
    AgentResult in ``data['result']``.
    """
    started = time.monotonic()
    request_id = deps.request_id
    tracker = RunUsageTracker(max_tokens_per_run=options.max_tokens_per_run)
    tool_calls: list[ToolCallRecord] = []
    tool_outputs: list[Any] = []
    unavailable: list[str] = []
    partial_notes: list[str] = []
    messages: list[Msg] = []
    incomplete = False
    incomplete_reason: str | None = None
    output_model_data: dict[str, Any] | None = None
    error: str | None = None

    def _result(**overrides: Any) -> AgentResult:
        base = dict(
            task=task_name,
            persona_id=persona.id,
            output=output_model_data or {},
            tool_calls_made=tool_calls,
            unavailable_sources=unavailable,
            usage=tracker.total,
            request_id=request_id,
            incomplete=incomplete,
            incomplete_reason=incomplete_reason,
            error=error,
        )
        base.update(overrides)
        return AgentResult(**base)

    # -- 1. validate the task ------------------------------------------------
    if not persona.can_run_task(task_name):
        raise TaskNotAllowedError(
            f"persona '{persona.id}' may not run task '{task_name}'",
            detail={"allowed_tasks": persona.allowed_tasks},
        )
    task: TaskSpec = get_task(task_name)

    yield AgentEvent(
        type="step_started",
        step="validate",
        detail=f"{persona.name} starting {task_name}",
    )

    # -- budget guard --------------------------------------------------------
    await check_budget(ctx.db, deps.workspace_id, options.daily_token_budget)

    # -- 2. brand context ----------------------------------------------------
    brand_context: BrandContext | None = None
    if task.requires_brand_context:
        yield AgentEvent(type="step_started", step="brand_context", detail="loading brand memory")
        try:
            from app.core.memory.store import BrandMemoryStore

            store = BrandMemoryStore(ctx.db)
            brand_context = await store.build_context(
                business_id=ctx.business_id,
                scope=MemoryScope(query=user_message or task_name, top_k=6),
                task_name=task_name,
            )
            yield AgentEvent(
                type="tool_result",
                step="brand_context",
                detail=f"{len(brand_context.items)} memory items, "
                f"{len(brand_context.learnings)} learnings",
            )
        except Exception as exc:
            _logger.warning("brand_context_failed", extra={"error": str(exc)})
            brand_context = None

    # -- 3. tools + prompt ---------------------------------------------------
    tools: list[Tool] = get_tools_for(persona) if options.allow_tool_calls else []
    if task_name == "competitor_research":
        # A first-pass competitor report needs live search and page extraction.
        # Hiding unrelated researcher tools reduces schema tokens and keeps
        # GPT-OSS focused on evidence gathering rather than task selection.
        tools = [tool for tool in tools if tool.name in {"web_search", "analyze_website"}]
    if persona.autonomy == Autonomy.SUGGEST:
        # Defence in depth with ``_run_one_tool``: a suggest-only persona is not
        # even *shown* the durable-write tools, so it cannot plan around the
        # gate by planning for it.
        tools = [t for t in tools if t.name not in _DRAFT_WRITE_TOOLS]
    tool_specs = [t.to_spec() for t in tools]
    tool_by_name = {t.name: t for t in tools}

    system = build_prompt(
        persona,
        task,
        brand_context=brand_context,
        tool_results="",
        inputs=inputs,
        user_message=user_message,
    )
    messages = [Msg(role="user", content=user_message or f"Run the {task_name} task.")]

    # -- 4. the loop ---------------------------------------------------------
    for iteration in range(1, options.max_iterations + 1):
        if time.monotonic() - started > options.timeout_seconds:
            incomplete = True
            incomplete_reason = (
                f"run exceeded the {options.timeout_seconds:.0f}s wall-clock cap"
            )
            yield AgentEvent(type="failed", step="timeout", detail=incomplete_reason)
            break
        if tracker.exceeded:
            incomplete = True
            incomplete_reason = (
                f"run exceeded the {options.max_tokens_per_run:,} token cap"
            )
            yield AgentEvent(type="failed", step="token_cap", detail=incomplete_reason)
            break

        yield AgentEvent(
            type="step_started", step=f"llm_call_{iteration}", detail=f"iteration {iteration}"
        )

        try:
            request_messages = _bound_tool_transcript(messages)
            _logger.info(
                "agent_llm_context_size task=%s iteration=%s system_chars=%s transcript_chars=%s tool_count=%s",
                task_name,
                iteration,
                len(system),
                sum(len(msg.content or "") for msg in request_messages),
                len(tool_specs),
            )
            result = await deps.llm.complete(
                system=system,
                messages=request_messages,
                tools=tool_specs or None,
                response_schema=None,  # tools first, structured output last
                tier=task.tier.value,  # type: ignore[arg-type]
                # Tool selection needs only a short function call. Reserving
                # the full 2k completion budget on every round consumes Groq's
                # TPM allowance before the final structured answer.
                max_tokens=min(options.max_tokens_per_call, 1200),
                temperature=options.temperature,
            )
        except LLMError as exc:
            failure = _llm_failure_message(exc)
            if (
                task_name == "competitor_research"
                and tool_calls
                and getattr(exc, "status_code", None) == 400
                and "parsing failed" in failure.lower()
            ):
                # Groq may reject a malformed GPT-OSS tool call after earlier
                # searches already succeeded. Keep those results and switch to
                # final synthesis instead of discarding all gathered evidence.
                _logger.warning(
                    "agent_tool_call_parse_failed_using_collected_results task=%s completed_tools=%s",
                    task_name,
                    len(tool_calls),
                )
                incomplete = True
                incomplete_reason = (
                    "Stopped gathering additional sources after the model provider "
                    "rejected a tool call; this report uses the results collected so far."
                )
                partial_notes.append(incomplete_reason)
                result = LLMResult(model=getattr(deps.llm, "model", ""))
            else:
                error = f"LLM call failed: {failure}"
                yield AgentEvent(type="failed", step="llm_call", detail=error)
                break

        tracker.add(result.usage)
        await record_usage(
            ctx.db,
            business_id=ctx.business_id,
            workspace_id=deps.workspace_id,
            usage=result.usage,
            persona_id=persona.id,
            tier=task.tier.value,
            # Label the billable act, not just the model: the usage screen breaks
            # spend down by action, and an unlabelled row is money nobody can
            # explain.
            action=f"task.{task_name}.step",
        )

        # The provider abstraction returns whole completions, not token streams,
        # so a token event carries the full text. When a streaming provider call
        # is added, this becomes a per-chunk yield with no other change.
        if result.text:
            yield AgentEvent(type="token", token=result.text, step=f"llm_call_{iteration}")

        # -- tool calls ------------------------------------------------------
        if result.tool_calls and options.allow_tool_calls:
            messages.append(
                Msg(role="assistant", content=result.text, tool_calls=result.tool_calls)
            )
            for call in result.tool_calls:
                yield AgentEvent(
                    type="tool_called",
                    step=call.name,
                    detail="starting",
                    data={"status": "running"},
                )
                record, output = await _run_one_tool(
                    call, persona=persona, ctx=ctx, tool_by_name=tool_by_name,
                    request_id=request_id,
                )
                tool_calls.append(record)
                tool_outputs.append(output)
                yield AgentEvent(
                    type="tool_result",
                    step=record.tool,
                    detail=record.output_summary,
                    data={"status": record.status, "error": record.error},
                )
                import json

                messages.append(
                    Msg(
                        role="tool",
                        content=json.dumps(
                            _compact_tool_payload(
                                record.tool,
                                output.model_dump(mode="json", exclude_none=True)
                                if hasattr(output, "model_dump")
                                else {"summary": record.output_summary},
                            ),
                            ensure_ascii=False,
                            default=str,
                        ),
                        tool_call_id=call.id,
                        name=record.tool,
                    )
                )

            # Refresh the prompt with the accumulated tool results.
            system = build_prompt(
                persona,
                task,
                brand_context=brand_context,
                # During tool execution, the provider already sees each
                # validated result once in its tool-role message. Duplicating
                # the growing transcript in system prompt made multi-search
                # requests exceed Groq's TPM limit before synthesis.
                tool_results="",
                inputs=inputs,
                user_message=user_message,
            )
            has_successful_search = any(
                record.tool == "web_search" and record.status == "ok"
                for record in tool_calls
            )
            if not (
                task_name == "competitor_research"
                and has_successful_search
                and len(tool_calls) >= 2
            ):
                continue
            # A search plus one follow-up result is enough for a first-pass
            # competitor report. Synthesize now instead of spending another
            # tool-selection call (which is where GPT-OSS has returned malformed
            # tool-call output under load).
            _logger.info(
                "competitor_research_synthesizing task=%s completed_tools=%s",
                task_name,
                len(tool_calls),
            )
            result = LLMResult(model=getattr(deps.llm, "model", ""))

        # -- no tool calls: ask for the final structured output ---------------
        if not result.tool_calls:
            messages.append(Msg(role="assistant", content=result.text))
            yield AgentEvent(
                type="step_started", step="structured_output", detail="validating output"
            )
            try:
                final_system = build_prompt(
                    persona,
                    task,
                    brand_context=brand_context,
                    tool_results=_tool_results_block(tool_calls, tool_outputs),
                    inputs=inputs,
                    user_message=user_message,
                )
                final_token_budget = (
                    max(options.max_tokens_per_call, 2500)
                    if task_name == "competitor_research"
                    else options.max_tokens_per_call
                )
                final = await deps.llm.complete(
                    system=final_system,
                    messages=list(messages),
                    # Keep the tool definitions in the transcript for the
                    # provider's final synthesis call, while the OpenAI
                    # adapter sets tool_choice="none" for structured output.
                    # GPT-OSS rejects a generated tool call if schemas vanish
                    # after the model has already used them in this turn.
                    tools=tool_specs or None,
                    response_schema=task.output_model,
                    tier=task.tier.value,  # type: ignore[arg-type]
                    max_tokens=final_token_budget,
                    temperature=0.0,
                )
                tracker.add(final.usage)
                await record_usage(
                    ctx.db,
                    business_id=ctx.business_id,
                    workspace_id=deps.workspace_id,
                    usage=final.usage,
                    persona_id=persona.id,
                    tier=task.tier.value,
                    action=f"task.{task_name}.final",
                )
                if final.parsed is not None:
                    output_model_data = final.parsed.model_dump(mode="json")
                    if task_name == "competitor_research" and _has_research_data(tool_calls, tool_outputs):
                        if not output_model_data.get("evidence"):
                            collected = _competitor_research_evidence(tool_calls, tool_outputs)
                            if collected:
                                output_model_data["evidence"] = collected
                else:
                    incomplete = True
                    incomplete_reason = "model produced no parseable structured output"
            except LLMError as exc:
                failure = _llm_failure_message(exc)
                if task_name == "competitor_research" and _has_research_data(tool_calls, tool_outputs):
                    evidence = _competitor_research_evidence(tool_calls, tool_outputs)
                    arabic = bool(re.search(r"[\u0600-\u06ff]", user_message))
                    rate_limited = "429" in failure or "rate limit" in failure.lower()
                    if arabic:
                        summary = (
                            f"نجح البحث في جمع {len(evidence)} نتيجة من المصادر، لكن تعذّر إكمال التحليل "
                            + (
                                "بسبب حدّ الطلبات لدى مزوّد النموذج. "
                                if rate_limited
                                else "لأن النموذج لم يُرجع تقريرًا منسقًا صالحًا. "
                            )
                            + "أعرض المصادر التي جُمعت أدناه دون "
                            "استنتاجات غير موثقة."
                        )
                    else:
                        summary = (
                            f"The search collected {len(evidence)} source results, but the final analysis "
                            + (
                                "was rate-limited by the model provider. "
                                if rate_limited
                                else "could not be returned as a valid structured report. "
                            )
                            + "The sources are shown below without "
                            "unsupported conclusions."
                        )
                    output_model_data = {
                        "summary": summary,
                        "evidence": evidence,
                        "unavailable_sources": [],
                        "notes": "Source collection succeeded; structured synthesis was unavailable.",
                        "competitors": [],
                        "common_patterns": [],
                        "differentiation_opportunities": [],
                    }
                    incomplete = True
                    incomplete_reason = (
                        "Provider rate limit prevented the final research synthesis."
                        if rate_limited
                        else "The model could not return a valid structured research report."
                    )
                    _logger.warning(
                        "competitor_research_returning_collected_evidence",
                        extra={"evidence_count": len(evidence), "failure": failure[:200]},
                    )
                else:
                    error = f"structured output failed: {failure}"
                    incomplete = True
                    incomplete_reason = error
            break
    else:
        # Loop exhausted without a final answer.
        incomplete = True
        incomplete_reason = (
            f"hit the {options.max_iterations}-iteration cap without a final answer"
        )
        yield AgentEvent(type="failed", step="iteration_cap", detail=incomplete_reason)

    # -- 5. unavailable sources ---------------------------------------------
    unavailable = _collect_unavailable(tool_calls, tool_outputs)
    unavailable.extend(note for note in partial_notes if note not in unavailable)
    if (
        task_name == "competitor_research"
        and output_model_data is not None
        and not _has_research_data(tool_calls, tool_outputs)
    ):
        reason = (
            "No research source returned usable competitor data. Configure web search "
            "in Settings → Connectors or provide competitor links/results, then retry."
        )
        if reason not in unavailable:
            unavailable.append(reason)
        # Enforce the task prompt's no-source rule in code as well: the model
        # must not fill the report with plausible but unverified brands or stats.
        output_model_data = {
            "summary": reason,
            "evidence": [],
            "unavailable_sources": unavailable,
            "notes": "No market claims generated because no research records were returned.",
            "competitors": [],
            "common_patterns": [],
            "differentiation_opportunities": [],
        }
        incomplete = True
        incomplete_reason = reason
    if unavailable:
        yield AgentEvent(
            type="tool_result",
            step="unavailable_sources",
            detail="; ".join(unavailable),
        )

    # -- 6. autonomy policy -------------------------------------------------
    persisted, requires_approval = _apply_autonomy_policy(
        persona, task, tool_calls=tool_calls, output=output_model_data
    )

    if incomplete:
        yield AgentEvent(
            type="failed",
            step="incomplete",
            detail=incomplete_reason or "run incomplete",
        )

    final_result = _result(persisted=persisted, incomplete=incomplete)
    yield AgentEvent(
        type="completed" if not error else "failed",
        step="done",
        detail=f"{task_name} finished in {int((time.monotonic() - started) * 1000)}ms",
        data={
            "result": final_result.model_dump(mode="json"),
            "requires_approval": requires_approval,
            "persisted": persisted,
        },
    )


def _status_value(status: object) -> str:
    """The wire value of a ToolStatus, tolerating plain strings and None."""
    if status is None:
        return ToolStatus.OK.value
    if isinstance(status, ToolStatus):
        return status.value
    return str(status)


async def _run_one_tool(
    call: ToolCall,
    *,
    persona: Persona,
    ctx: ToolContext,
    tool_by_name: dict[str, Tool],
    request_id: str,
) -> tuple[ToolCallRecord, Any]:
    """Validate and run a single tool call, honouring the allowlist."""
    started = time.monotonic()
    persona_name = persona.name if hasattr(persona, "name") else str(persona)

    # Autonomy gate (D1) runs BEFORE anything is executed, and before the
    # allowlist check, because the two answer different questions: "may this
    # persona run any tool like this" vs "may this persona create something
    # durable". Ordering matters only for the reason string.
    if blocks_draft_write(persona, call.name):
        reason = (
            f"'{call.name}' would create durable data, and {persona.name} is set "
            "to suggest-only. It was not run; the suggestion is in the output."
        )
        _logger.warning(
            "tool_blocked_by_autonomy",
            extra={
                "persona": persona.id,
                "autonomy": persona.autonomy.value,
                "tool": call.name,
                "request_id": request_id,
            },
        )
        return (
            ToolCallRecord(
                tool=call.name,
                input=dict(call.arguments),
                output_summary=reason,
                status="blocked",
                duration_ms=0,
                error=reason,
            ),
            {"status": "blocked", "detail": reason},
        )

    # Allowlist check happens BEFORE anything is executed.
    if not persona.can_use_tool(call.name):
        reason = (
            f"tool '{call.name}' is not on {persona.name}'s allowlist and was not run"
        )
        _logger.warning(
            "tool_blocked_by_allowlist",
            extra={"persona": persona.id, "tool": call.name, "request_id": request_id},
        )
        return (
            ToolCallRecord(
                tool=call.name,
                input=dict(call.arguments),
                output_summary=reason,
                status="blocked",
                duration_ms=0,
                error=reason,
            ),
            {"status": "blocked", "detail": reason},
        )

    tool = tool_by_name.get(call.name)
    if tool is None:
        reason = f"tool '{call.name}' is not registered"
        return (
            ToolCallRecord(
                tool=call.name,
                input=dict(call.arguments),
                output_summary=reason,
                status="blocked",
                duration_ms=0,
                error=reason,
            ),
            {"status": "blocked", "detail": reason},
        )

    tool_logger = _logger

    try:
        payload = tool.input_model.model_validate(call.arguments)
    except ValidationError as exc:
        reason = f"invalid input for '{call.name}': {format_validation_error(exc)}"
        tool_logger.warning(
            f"[TOOL CALL - INVALID] '{call.name}': {reason}",
            extra={"tool": call.name, "arguments": redact(dict(call.arguments)), "error": reason},
        )
        return (
            ToolCallRecord(
                tool=call.name,
                input=dict(call.arguments),
                output_summary=reason,
                status="error",
                duration_ms=0,
                error=reason,
            ),
            {"status": "error", "detail": reason},
        )

    tool_logger.info(
        f"[TOOL CALL - START] Executing '{call.name}' (persona: {persona_name})",
        extra={
            "tool": call.name,
            "call_id": call.id,
            "persona": persona_name,
            "arguments": redact(dict(call.arguments)),
        },
    )

    try:
        output = await tool.handler(payload, ctx)
        raw_status = getattr(output, "status", None)
        status = _status_value(raw_status)
        summary = _tool_result_summary(output)
        duration = int((time.monotonic() - started) * 1000)
        tool_logger.info(
            f"[TOOL RESULT - SUCCESS] '{call.name}' completed in {duration}ms (status: {status})",
            extra={
                "tool": call.name,
                "call_id": call.id,
                "persona": persona_name,
                "status": status,
                "duration_ms": duration,
                "summary": summary,
                "output": redact(output.model_dump(mode="json") if hasattr(output, "model_dump") else str(output)),
            },
        )
        await log_tool_call(
            ctx,
            tool=call.name,
            tool_input=payload,
            output_summary=summary,
            status=status,
            duration_ms=duration,
        )
        return (
            ToolCallRecord(
                tool=call.name,
                input=payload.model_dump(mode="json"),
                output_summary=summary,
                status=status,
                duration_ms=duration,
            ),
            output,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        duration = int((time.monotonic() - started) * 1000)
        tool_logger.error(
            f"[TOOL RESULT - ERROR] '{call.name}' failed in {duration}ms: {reason}",
            extra={
                "tool": call.name,
                "call_id": call.id,
                "persona": persona_name,
                "duration_ms": duration,
                "error": reason,
                "request_id": request_id,
            },
        )
        return (
            ToolCallRecord(
                tool=call.name,
                input=dict(call.arguments),
                output_summary=reason,
                status="error",
                duration_ms=duration,
                error=reason,
            ),
            {"status": "error", "detail": reason},
        )


# ------------------------------------------------------------- public entry --


async def run_agent(
    persona: Persona,
    task_name: str,
    inputs: dict[str, Any],
    ctx: ToolContext,
    *,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    options: RunOptions | None = None,
    user_message: str = "",
) -> AgentResult:
    """Run one task to completion and return the result.

    ``ctx`` carries the db session, llm client and request id. ``llm`` may be
    passed to override the router (tests do this).
    """
    cfg = settings or get_settings()
    opts = options or RunOptions.from_settings(cfg)
    deps = RunDependencies(
        llm=llm or ctx.llm or get_llm("strong", cfg),
        db=ctx.db,
        settings=cfg,
        workspace_id=ctx.workspace_id,
        request_id=ctx.request_id,
    )
    result: AgentResult | None = None
    async for event in _execute(persona, task_name, inputs, ctx, deps, opts,
                               user_message=user_message):
        if event.type in ("completed", "failed") and event.data and "result" in event.data:
            result = AgentResult.model_validate(event.data["result"])
    if result is None:  # pragma: no cover - _execute always yields a terminal event
        raise PolicyViolation("agent run produced no result")
    return result


async def run_agent_stream(
    persona: Persona,
    task_name: str,
    inputs: dict[str, Any],
    ctx: ToolContext,
    *,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    options: RunOptions | None = None,
    user_message: str = "",
) -> AsyncGenerator[AgentEvent, None]:
    """Streaming variant. Yields step_started / tool_called / tool_result /
    token / completed / failed events so the frontend can render TaskProgress and
    streaming text."""
    cfg = settings or get_settings()
    opts = options or RunOptions.from_settings(cfg)
    deps = RunDependencies(
        llm=llm or ctx.llm or get_llm("strong", cfg),
        db=ctx.db,
        settings=cfg,
        workspace_id=ctx.workspace_id,
        request_id=ctx.request_id,
    )
    async for event in _execute(persona, task_name, inputs, ctx, deps, opts,
                               user_message=user_message):
        yield event


__all__ = [
    "RunDependencies",
    "RunOptions",
    "blocks_draft_write",
    "describe_autonomy",
    "run_agent",
    "run_agent_stream",
]
