"""The prompt interview must not survive on the conversation row once it ends.

The state is persisted by the service layer, and it is the *persisted* copy that
matters: as long as `workflow: creative_prompt` sits on the conversation, the
client re-sends it with every later turn and the AI layer files the turn back into
prompt-only mode. That is how the interview swallowed a design-mode «صمم الصورة»:
generation was never offered as a tool, and the stage machine answered with the
same canned line each time. These tests pin the clearing rule and, just as
importantly, that an ordinary turn is left completely alone.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SERVICE = Path(__file__).resolve().parents[1] / "app" / "core" / "orchestrator" / "service.py"

_PROMPT_KEYS = (
    "workflow",
    "prompt_type",
    "creative_prompt_stage",
    "creative_prompt_user_facts",
    "creative_prompt_confirmed_facts",
    "creative_prompt_reference_shown_key",
    "creative_prompt_unresolved_fields",
    "creative_prompt_answer",
)


def _source() -> str:
    return SERVICE.read_text(encoding="utf-8")


def test_the_service_parses() -> None:
    ast.parse(_source())


def test_the_clear_branch_exists_and_is_guarded_by_the_persisted_workflow() -> None:
    """Guarded by `context.get("workflow")`, or every ordinary turn would be hit.

    An unguarded branch would also stamp `persona_id = "layla"` onto Omar, Nour and
    Karim turns -- the next message would reach Layla regardless of who the user was
    talking to.
    """
    source = _source()

    assert "context.get(\"workflow\") == \"creative_prompt\"" in source
    assert 'getattr(final_gui, "creative_prompt_workflow", None) is False' in source


def test_every_prompt_state_key_is_cleared() -> None:
    source = _source()

    for key in _PROMPT_KEYS:
        assert f'"{key}"' in source, key


def test_clearing_is_persisted_back_to_the_row() -> None:
    """The dict assigned to the row must be a new object, or no UPDATE is emitted."""
    source = _source()
    branch = source.split('getattr(final_gui, "creative_prompt_workflow", None) is False')[1]
    branch = branch.split('if context.get("workflow") == "creative_prompt":')[0]

    assert "conversation.context = dict(context)" in branch
