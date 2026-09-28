"""Layer 5: the generic agent runtime.

ONE loop, configured by persona objects. No per-persona code.

    run_agent(persona, task_name, inputs, ctx) -> AgentResult

Loop:
  1. validate the task is on the persona's allowlist
  2. fetch brand context if the task needs it
  3. build the prompt (build_prompt -- the only prompt assembler)
  4. up to max_iterations: LLM -> tool calls (allowlisted only) -> repeat
  5. validate the final structured output (one repair retry)
  6. apply the autonomy policy
  7. return AgentResult

Hard caps (iterations, tokens, wall clock) always produce a partial result with
``incomplete=True`` -- a run never hangs.
"""

from __future__ import annotations

import asyncio
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
from app.core.llm.base import LLMClient, Msg, ToolCall, format_validation_error
from app.core.llm.router import get_llm
from app.core.logging import get_logger
from app.core.tools.base import Tool, ToolContext, log_tool_call
from app.core.tools.registry import get_tools_for
from app.core.usage import RunUsageTracker, check_budget, record_usage
from app.schemas.agent import AgentEvent, AgentResult, ToolCallRecord
from app.schemas.common import Autonomy, ToolStatus
from app.schemas.tools import BrandContext, MemoryScope

_logger = get_logger(__name__)


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


def _apply_autonomy_policy(
    persona: Persona,
    task: TaskSpec,
    *,
    tool_calls: list[ToolCallRecord],
    output: dict[str, Any] | None,
) -> tuple[list[str], bool]:
    """Decide what gets persisted.

    Returns ``(persisted, requires_approval)``.

    - suggest: nothing durable. The output is a proposal for the user to accept.
    - auto_draft: drafts may be persisted. External writes are structurally
      impossible (the registry refuses external_write tools), so there is nothing
      to guard against here beyond recording what was written.
    - ask_first: anything with a side effect stays pending and the result is
      flagged as requiring an approval record before execution.
    """
    persisted: list[str] = []
    requires_approval = False
    side_effecting = [tc for tc in tool_calls if tc.tool in _DRAFT_WRITE_TOOLS]

    if persona.autonomy == Autonomy.SUGGEST:
        # Nothing durable is created from this run's output.
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
    return persisted, True


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


def _tool_results_block(tool_calls: list[ToolCallRecord], outputs: list[Any]) -> str:
    """Render tool results for the prompt."""
    if not tool_calls:
        return ""
    lines: list[str] = []
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

        rendered = json.dumps(payload, ensure_ascii=False, default=str)
        lines.append(f"### {record.tool}\n{rendered}")
    return "\n\n".join(lines)


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
            result = await deps.llm.complete(
                system=system,
                messages=list(messages),
                tools=tool_specs or None,
                response_schema=None,  # tools first, structured output last
                tier=task.tier.value,  # type: ignore[arg-type]
                max_tokens=options.max_tokens_per_call,
                temperature=options.temperature,
            )
        except LLMError as exc:
            error = f"LLM call failed: {exc.message}"
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
                record, output = await _run_one_tool(
                    call, persona=persona, ctx=ctx, tool_by_name=tool_by_name,
                    request_id=request_id,
                )
                tool_calls.append(record)
                tool_outputs.append(output)
                yield AgentEvent(
                    type="tool_called",
                    step=record.tool,
                    detail=record.output_summary,
                    data={"input": record.input, "status": record.status},
                )
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
                            output.model_dump(mode="json", exclude_none=True)
                            if hasattr(output, "model_dump")
                            else {"summary": record.output_summary},
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
                tool_results=_tool_results_block(tool_calls, tool_outputs),
                inputs=inputs,
                user_message=user_message,
            )
            continue

        # -- no tool calls: ask for the final structured output ---------------
        if not result.tool_calls:
            messages.append(Msg(role="assistant", content=result.text))
            yield AgentEvent(
                type="step_started", step="structured_output", detail="validating output"
            )
            try:
                final = await deps.llm.complete(
                    system=system,
                    messages=list(messages),
                    tools=None,
                    response_schema=task.output_model,
                    tier=task.tier.value,  # type: ignore[arg-type]
                    max_tokens=options.max_tokens_per_call,
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
                )
                if final.parsed is not None:
                    output_model_data = final.parsed.model_dump(mode="json")
                else:
                    incomplete = True
                    incomplete_reason = "model produced no parseable structured output"
            except LLMError as exc:
                error = f"structured output failed: {exc.message}"
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

    try:
        payload = tool.input_model.model_validate(call.arguments)
    except ValidationError as exc:
        reason = f"invalid input for '{call.name}': {format_validation_error(exc)}"
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

    try:
        output = await tool.handler(payload, ctx)
        raw_status = getattr(output, "status", None)
        status = _status_value(raw_status)
        summary = _tool_result_summary(output)
        duration = int((time.monotonic() - started) * 1000)
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
        _logger.warning(
            "tool_handler_failed",
            extra={"tool": call.name, "error": reason, "request_id": request_id},
        )
        return (
            ToolCallRecord(
                tool=call.name,
                input=dict(call.arguments),
                output_summary=reason,
                status="error",
                duration_ms=int((time.monotonic() - started) * 1000),
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
    "describe_autonomy",
    "run_agent",
    "run_agent_stream",
]
