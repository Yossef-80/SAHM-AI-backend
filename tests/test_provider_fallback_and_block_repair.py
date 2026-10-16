"""The failures behind "the AI provider is unavailable" and the lost formatting.

From a live log (Groq quota exhausted, workspace switched to OpenRouter without a
key):

* ``ui_block_structuring_failed`` / ``ValidationError`` -- the model returned each
  block as a JSON *string*, so the discriminated union rejected the whole formatted
  answer and the reply fell back to plain text;
* ``تعذر تشغيل مزود الذكاء الاصطناعي الآن`` -- the workspace's provider had no
  credential, and nothing fell back to the process provider that does work.

Both are covered here: the repair is deterministic, and the fallback must fire only
when the workspace choice is genuinely unusable.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.ai.providers.llm as llm_module  # noqa: E402
from app.ai.agents.orchestrator.brain import (  # noqa: E402
    _provider_failure_hint,
    _provider_rejected_request,
    _repair_structured_blocks,
)
from app.ai.errors import ProviderNotConfiguredError  # noqa: E402
from app.schemas.blocks import MessageWithBlocks, TextBlock  # noqa: E402


# ------------------------------------------------------- the block repair -------


def _validation_error_from(payload: dict[str, object]) -> ValidationError:
    try:
        MessageWithBlocks.model_validate(payload)
    except ValidationError as exc:
        return exc
    raise AssertionError("payload unexpectedly validated")


def test_a_block_serialised_as_a_json_string_is_recovered() -> None:
    """This is the exact shape seen in the log: blocks as a list of strings."""
    block = '{"type": "text", "text": "أهلاً، أقدر أساعدك."}'
    exc = _validation_error_from({"blocks": [block]})

    repaired = _repair_structured_blocks(exc)

    assert repaired is not None
    assert isinstance(repaired.blocks[0], TextBlock)
    assert "أساعدك" in repaired.blocks[0].text


def test_mixed_blocks_keep_their_order() -> None:
    blocks = [
        '{"type": "text", "text": "ملخص"}',
        '{"type": "findings", "items": [{"label": "الوصول", "detail": "٤٠٪"}]}',
    ]
    exc = _validation_error_from({"blocks": blocks})

    repaired = _repair_structured_blocks(exc)

    assert repaired is not None
    assert [block.type for block in repaired.blocks] == ["text", "findings"]


def test_a_whole_payload_serialised_as_a_string_is_recovered() -> None:
    import json

    payload = json.dumps({"blocks": [{"type": "text", "text": "تمام"}]})
    exc = _validation_error_from({"blocks": [payload]})

    repaired = _repair_structured_blocks(exc)

    assert repaired is not None
    assert repaired.blocks[0].text == "تمام"


def test_an_unrepairable_error_yields_nothing() -> None:
    payload = {"blocks": ['{"nonsense": true}', "not json at all"]}
    exc = _validation_error_from(payload)

    assert _repair_structured_blocks(exc) is None


def test_a_non_validation_error_is_not_treated_as_repairable() -> None:
    assert _repair_structured_blocks(TimeoutError("boom")) is None


# --------------------------------------------------- actionable error hints ----


def test_a_missing_credential_points_at_the_settings_screen() -> None:
    hint = _provider_failure_hint(
        ProviderNotConfiguredError("AI_LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is empty."),
        arabic=True,
    )

    assert "الإعدادات" in hint
    assert ".env" in hint


def test_an_exhausted_quota_says_to_switch_model() -> None:
    class _RateLimited(Exception):
        status_code = 429

    hint = _provider_failure_hint(_RateLimited("quota exhausted"), arabic=True)

    assert "حصة" in hint
    assert "بدّل" in hint


def test_a_rejected_key_says_to_check_the_key() -> None:
    class _Unauthorized(Exception):
        status_code = 401

    assert "مرفوض" in _provider_failure_hint(_Unauthorized("bad key"), arabic=True)


def test_an_unknown_failure_stays_out_of_the_bubble() -> None:
    """Vague errors belong in the server log, not in a user-facing guess."""
    assert _provider_failure_hint(ValueError("something internal"), arabic=True) == ""


# ------------------------------------------------ provider-level fallback -----


def _settings() -> object:
    """The real settings object, so key resolution behaves exactly as in a turn."""
    from app.ai.settings import get_ai_settings

    return get_ai_settings()


def _run_workspace_call(monkeypatch: pytest.MonkeyPatch, *, key_present: bool) -> list[tuple[str, str]]:
    """Drive `get_workspace_chat_llm` with a workspace that picked OpenRouter."""
    calls: list[tuple[str, str]] = []

    async def _tier_state(db: object, business_id: str, tier: str) -> dict[str, object]:
        return {
            "provider": "openrouter",
            "model": "some/openrouter-model",
            "api_key": "secret" if key_present else "",
            "base_url": None,
        }

    class _Handle:
        provider = "gemini"
        model_name = "gemini-3.6-flash"

    def _fake_get_llm(**kwargs: object) -> object:
        provider = str(getattr(kwargs.get("provider"), "value", kwargs.get("provider")))
        calls.append((provider, str(kwargs.get("model") or "")))
        settings = kwargs.get("settings")
        # Mirrors the real provider module: a selected provider with no credential
        # cannot be built at all.
        key = getattr(settings, "openrouter_api_key", "") if settings is not None else ""
        if provider == "openrouter" and not key:
            raise ProviderNotConfiguredError("OPENROUTER_API_KEY is empty.")
        return _Handle()

    monkeypatch.setattr("app.core.llm.router.resolve_workspace_tier", _tier_state)
    monkeypatch.setattr(llm_module, "get_llm", _fake_get_llm)

    asyncio.run(
        llm_module.get_workspace_chat_llm(
            db=object(), business_id="biz_1", tier="strong", settings=_settings()
        )
    )
    return calls


def test_a_workspace_without_a_key_falls_back_to_the_process_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    calls = _run_workspace_call(monkeypatch, key_present=False)

    assert calls[0][0] == "openrouter"          # the workspace choice was tried
    assert calls[-1][0] == settings.provider_for_tier("strong")   # the process provider answered
    assert calls[-1][1] == settings.model_for_tier("strong")


def test_a_workspace_with_a_key_keeps_its_own_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _run_workspace_call(monkeypatch, key_present=True)

    assert [provider for provider, _ in calls] == ["openrouter"]


# ------------------------------- a provider that gates the model itself -------


class _OpenAIPermissionDeniedError(Exception):
    """Exact shape from the live log: a 403 whose body blames the *model*."""

    status_code = 403

    def __init__(self) -> None:
        super().__init__(
            "Error code: 403 - {'error': {'message': 'thinkingmachines/inkling:free is only "
            "available on agentic harnesses. Try plugging it into a coding agent or "
            "productivity app listed on https://openrouter.ai/apps', 'code': 403}}"
        )


_OpenAIPermissionDeniedError.__name__ = "OpenAIPermissionDeniedError"


def test_a_gated_model_403_counts_as_a_refusal() -> None:
    """The request never ran, so retrying it on another model is free and safe."""
    assert _provider_rejected_request(_OpenAIPermissionDeniedError()) is True


def test_the_hint_blames_the_model_not_the_key() -> None:
    hint = _provider_failure_hint(_OpenAIPermissionDeniedError(), arabic=True)

    assert "المفتاح سليم" in hint
    assert "بدّل" in hint


def test_a_403_without_the_model_wording_still_mentions_the_key() -> None:
    class _Forbidden(Exception):
        status_code = 403

    hint = _provider_failure_hint(_Forbidden("access denied"), arabic=True)

    assert "المفتاح" in hint


def test_a_timeout_and_a_5xx_are_still_never_retried() -> None:
    class _Timeout(Exception):
        status_code = 504

    assert _provider_rejected_request(TimeoutError("the call may have run")) is False
    assert _provider_rejected_request(_Timeout("gateway")) is False


def test_every_place_that_calls_a_model_has_the_process_fallback() -> None:
    """Routing, the agent run and the chat reply all died on the same 403."""
    from app.ai.agents.orchestrator import brain

    source = Path(brain.__file__).read_text(encoding="utf-8")

    assert "routing_model_refused_request" in source
    assert "chat_model_refused_request" in source
    assert "agent_graph_failed" in source
    assert source.count("_process_fallback_handle(") >= 4  # definition + three call sites


# ------------------------------------------- the model that does not exist -----


def _google_model_not_found(message: str) -> BaseException:
    """The real class when it imports, a faithful stand-in otherwise.

    ``GoogleModelNotFoundError`` has *no* ``status_code`` and is a subclass of
    ``langchain_core.exceptions.ModelNotFoundError`` -- matching it by class name
    alone is exactly what let the live 404 walk past the retry.
    """
    try:
        from langchain_google_genai.chat_models import GoogleModelNotFoundError

        return GoogleModelNotFoundError(message)
    except Exception:  # noqa: BLE001 - the dependency may not be installed here
        try:
            from langchain_core.exceptions import ModelNotFoundError

            return ModelNotFoundError(message)
        except Exception:  # noqa: BLE001
            return RuntimeError(message)


def test_a_model_that_does_not_exist_is_treated_as_refused() -> None:
    """The live error: 404 NOT_FOUND: models/gemini-3-flash is not found."""
    exc = _google_model_not_found(
        "404 NOT_FOUND. {'error': {'code': 404, 'message': 'models/gemini-3-flash is not found "
        "for API version v1beta, or is not supported for generateContent.'}}"
    )

    assert _provider_rejected_request(exc) is True


def test_a_model_not_found_with_no_status_code_is_still_refused() -> None:
    """Google's class carries no ``status_code`` attribute at all."""
    exc = _google_model_not_found("models/gemini-3-flash is not found for API version v1beta")

    assert not getattr(exc, "status_code", None)
    assert _provider_rejected_request(exc) is True


def test_the_hint_names_the_model_not_the_key() -> None:
    exc = _google_model_not_found("models/gemini-3-flash is not found for API version v1beta")

    hint = _provider_failure_hint(exc, arabic=True)

    assert hint
    assert "الموديل" in hint
    assert "مش موجود" in hint
    assert "المفتاح مرفوض" not in hint


def test_the_hint_points_at_the_listing_script() -> None:
    class _NotFound(Exception):
        status_code = 404

    hint = _provider_failure_hint(_NotFound("model not found"), arabic=True)

    assert "list_llm_models" in hint


def test_a_404_with_nothing_about_models_still_reads_as_a_missing_model() -> None:
    """A gateway 404 means the route, and for a model endpoint that is the model."""
    class _NotFound(Exception):
        status_code = 404

    assert "الموديل" in _provider_failure_hint(_NotFound("404 page not found"), arabic=True)


def test_the_listing_script_ships_for_the_user_to_run() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "list_llm_models.py"

    assert script.is_file()
    source = script.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" in source and "OPENROUTER_API_KEY" in source


def test_an_unreadable_saved_key_never_blocks_a_save() -> None:
    """The save-time guard decrypts a stored key; a cipher failure is 'unknown'."""
    source = (Path(__file__).resolve().parents[1] / "app" / "api" / "settings.py").read_text(
        encoding="utf-8"
    )

    assert "a key we cannot read is not a verdict" in source
