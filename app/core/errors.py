"""Typed error hierarchy.

Every failure that crosses a layer boundary is one of these. API handlers map
them to consistent JSON error bodies; the agent runtime converts
``ToolUnavailable`` into a user-visible message rather than a crash.
"""

from __future__ import annotations


class SahmError(Exception):
    """Base class for every deliberate error in the system."""

    code: str = "sahm_error"
    http_status: int = 500

    def __init__(self, message: str, *, detail: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


# --------------------------------------------------------------- LLM layer --


class LLMError(SahmError):
    """Base for provider-level failures."""

    code = "llm_error"
    http_status = 502


class LLMOutputError(LLMError):
    """Structured output could not be produced/validated after one repair retry."""

    code = "llm_output_error"


class LLMProviderError(LLMError):
    """Provider returned an error, timed out, or exhausted retries."""

    code = "llm_provider_error"

    def __init__(
        self,
        message: str,
        *,
        detail: dict[str, object] | None = None,
        retryable: bool = True,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, detail=detail)
        #: Whether the caller should back off and try again.
        self.retryable = retryable
        self.status_code = status_code


class LLMNotConfiguredError(LLMError):
    """The configured provider has no API key / is not wired up."""

    code = "llm_not_configured"
    http_status = 503


# -------------------------------------------------------------- tool layer --


class ToolError(SahmError):
    """Base for tool failures."""

    code = "tool_error"
    http_status = 500


class ToolUnavailable(ToolError):
    """An external source is unreachable, rate limited, or has no data.

    Tools must raise/return this instead of fabricating data. The runtime turns
    it into a user-visible limitation message.
    """

    code = "tool_unavailable"
    http_status = 503


class ToolValidationError(ToolError):
    """Tool input failed the tool's own input_model validation."""

    code = "tool_validation_error"
    http_status = 422


class ToolNotFoundError(ToolError):
    """A tool name was requested that is not in the registry."""

    code = "tool_not_found"
    http_status = 404


# ------------------------------------------------------------ policy layer --


class PolicyViolation(SahmError):
    """An agent tried something its persona/autonomy policy forbids."""

    code = "policy_violation"
    http_status = 403


class ToolNotAllowedError(PolicyViolation):
    """Persona attempted to call a tool outside its allowlist."""

    code = "tool_not_allowed"


class TaskNotAllowedError(PolicyViolation):
    """Persona attempted a task outside its allowlist."""

    code = "task_not_allowed"


class ExternalWriteBlockedError(PolicyViolation):
    """An agent tried to reach an external_write tool."""

    code = "external_write_blocked"


class ApprovalRequired(SahmError):
    """A side effect needs a human approval record before it may execute."""

    code = "approval_required"
    http_status = 409


class ApprovalNotFoundError(SahmError):
    code = "approval_not_found"
    http_status = 404


class ApprovalStateError(SahmError):
    """Approval is in a state that forbids this transition."""

    code = "approval_state_error"
    http_status = 409


# ------------------------------------------------------------------- cost ----


class TokenBudgetExceeded(SahmError):
    """Per-workspace daily token budget exhausted."""

    code = "token_budget_exceeded"
    http_status = 429


# ------------------------------------------------------------------- data ----


class NotFoundError(SahmError):
    code = "not_found"
    http_status = 404


class ValidationFailed(SahmError):
    code = "validation_failed"
    http_status = 422


# -------------------------------------------------------------- security ----


class UnauthorizedError(SahmError):
    code = "unauthorized"
    http_status = 401


class ForbiddenError(SahmError):
    code = "forbidden"
    http_status = 403


class BlockedTargetError(SahmError):
    """SSRF guard: the requested URL resolves to a private/reserved range."""

    code = "blocked_target"
    http_status = 400


class UploadRejectedError(SahmError):
    """Uploaded file failed type/size validation."""

    code = "upload_rejected"
    http_status = 413
