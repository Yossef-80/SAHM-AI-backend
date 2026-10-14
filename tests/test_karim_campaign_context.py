"""Karim could not name a campaign the workspace had actually run.

Two independent defects produced the same visible symptom -- "the agent does not
know which campaign I mean, even though the campaigns page shows spend":

1. ``get_core_tools_for_agent`` needs a ``ToolContext``. The orchestrator only
   built one for the competitor-Ad-Library route, so ``list_meta_campaigns`` /
   ``get_campaign_insights`` / ``get_breakdowns`` answered
   "This conversation run did not provide a workspace tool context" on every
   ordinary Karim turn. The tools were never broken; they never received a
   workspace.
2. Nothing told the agent which campaigns exist in *this* workspace. The names
   live in the ``campaigns`` table, and the chat never read it, so the model had
   to guess a name it had never seen.

These tests pin both halves: the tools get the run's context, and the agent gets
a bounded, read-only ``[CURRENT_CAMPAIGN_CONTEXT]`` block with the exact ids.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.agents.orchestrator import brain
from app.db.models import Business, Campaign

# --------------------------------------------------------------- test doubles --


class _FakeRouter:
    def __init__(self, decision: brain.RoutingDecision) -> None:
        self.decision = decision

    async def ainvoke(self, _messages: Any) -> brain.RoutingDecision:
        return self.decision


class _FakeChat:
    def __init__(self, decision: brain.RoutingDecision) -> None:
        self.router = _FakeRouter(decision)

    def with_structured_output(self, schema: type[Any]) -> Any:
        assert schema is brain.RoutingDecision
        return self.router


class _RecordingGraph:
    """Stands in for Karim's compiled graph and keeps what it was called with."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((payload, config))
        return {
            "messages": [
                SimpleNamespace(
                    name="submit_performance_report",
                    content={"summary": "ok", "missing_inputs": []},
                    tool_calls=[],
                )
            ]
        }


@pytest.fixture()
def campaign_workspace(db_session: AsyncSession) -> AsyncSession:
    """A business with three campaigns, one of them linked to Meta."""
    db_session.add(
        Business(
            id="biz_k",
            workspace_id="ws_k",
            name="Cairo Coffee Co",
            industry="F&B",
            country="EG",
            language="ar",
            currency="EGP",
        )
    )
    db_session.add_all(
        [
            Campaign(
                id="cmp_ramadan",
                business_id="biz_k",
                name="رمضان 2026",
                status="active",
                meta_campaign_id="120252033253890323",
            ),
            Campaign(
                id="cmp_office",
                business_id="biz_k",
                name="Office Subscription Push",
                status="paused",
            ),
            Campaign(id="cmp_other", business_id="biz_k", name="Weekend Brunch", status="draft"),
        ]
    )
    # A second tenant's campaign must never appear in the block.
    db_session.add(
        Business(id="biz_other", workspace_id="ws_other", name="Other Co", country="EG", currency="EGP")
    )
    db_session.add(
        Campaign(id="cmp_leak", business_id="biz_other", name="Someone Else's Campaign", status="active")
    )
    return db_session


# ---------------------------------------------------- 1. the context block --


async def test_campaign_context_block_lists_workspace_campaigns(
    campaign_workspace: AsyncSession,
) -> None:
    block = await brain._campaign_context_message(campaign_workspace, "biz_k", "cmp_office")

    assert block is not None
    assert "[CURRENT_CAMPAIGN_CONTEXT]" in block
    # Every workspace campaign, with the exact id the tools need.
    assert "رمضان 2026" in block and "id=cmp_ramadan" in block
    assert "Office Subscription Push" in block and "id=cmp_office" in block
    assert "Weekend Brunch" in block
    # The focused campaign is called out, so "this campaign" resolves.
    assert "currently looking at this campaign: Office Subscription Push (id=cmp_office)" in block
    # The Meta id travels too: it is what get_campaign_insights actually reads.
    assert "meta_campaign_id=120252033253890323" in block
    # Tenant isolation and the no-guessing rule.
    assert "Someone Else's Campaign" not in block
    assert "never invent a campaign id" in block.lower()


async def test_campaign_context_block_absents_itself_when_there_is_nothing_to_show(
    db_session: AsyncSession,
) -> None:
    db_session.add(Business(id="biz_empty", workspace_id="ws_e", name="Empty", country="EG", currency="EGP"))
    await db_session.flush()
    assert await brain._campaign_context_message(db_session, "biz_empty", None) is None
    assert await brain._campaign_context_message(None, "biz_empty", None) is None


async def test_campaign_context_block_survives_a_broken_lookup(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A context lookup must never fail the turn it is decorating."""

    class _Boom:
        async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("database is down")

    assert await brain._campaign_context_message(_Boom(), "biz_k", None) is None


# ------------------------------------------- 2. the tools get their context --


async def test_karim_read_tools_receive_the_run_context(
    campaign_workspace: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No explicit ToolContext, only the orchestrator's per-run keys."""
    from app.ai.tools.core_bridge import get_core_tools_for_agent

    seen: dict[str, Any] = {}

    async def _fake_list(db: Any, business_id: str, settings: Any, payload: Any) -> Any:
        from app.schemas.tools import ListMetaCampaignsOutput, MetaCampaignSummary, ToolStatus

        seen["business_id"] = business_id
        seen["search"] = payload.search
        return ListMetaCampaignsOutput(
            status=ToolStatus.OK,
            source="meta_marketing_api",
            account_id="act_1",
            campaigns=[
                MetaCampaignSummary(meta_campaign_id="120252033253890323", name="رمضان 2026")
            ],
        )

    monkeypatch.setattr(
        "app.core.tools.meta_read.list_meta_campaigns_for_workspace", _fake_list
    )

    tools = {tool.name: tool for tool in get_core_tools_for_agent("karim")}
    result = await tools["list_meta_campaigns"].ainvoke(
        {"search": "رمضان"},
        config={
            "configurable": {
                "thread_id": "t1",
                "agent": "karim",
                "business_id": "biz_k",
                # exactly what brain._graph_config now attaches:
                "tool_db": campaign_workspace,
                "workspace_id": "ws_k",
                "core_llm": None,
                "request_id": "req-1",
                "conversation_id": "conv-1",
                "campaign_id": "cmp_ramadan",
                "user_supplied": {"latest_user_message": "ايه أخبار حملة رمضان؟"},
            }
        },
    )

    assert "did not provide a workspace tool context" not in result, result
    assert seen["business_id"] == "biz_k", "the handler must see the run's business"
    assert seen["search"] == "رمضان"
    assert "120252033253890323" in result


async def test_karim_insights_tool_resolves_a_sahm_campaign_id(
    campaign_workspace: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Sahm campaign id becomes its Meta id instead of "not found"."""
    from app.ai.tools.core_bridge import get_core_tools_for_agent

    captured: dict[str, Any] = {}

    class _FakeGraph:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
            captured["path"] = path
            captured["params"] = params
            return {
                "data": [
                    {
                        "date_start": "2026-09-01",
                        "date_stop": "2026-09-01",
                        "spend": "120.50",
                        "impressions": "1000",
                        "clicks": "25",
                    }
                ]
            }

    monkeypatch.setattr("app.ai.marketing.meta_ads.HttpMetaGraphClient", _FakeGraph)
    monkeypatch.setattr(
        "app.core.tools.meta_read._get_meta_token", AsyncMock(return_value="tok-1")
    )

    tools = {tool.name: tool for tool in get_core_tools_for_agent("karim")}
    result = await tools["get_campaign_insights"].ainvoke(
        {"campaign_id": "cmp_ramadan"},
        config={
            "configurable": {
                "thread_id": "t2",
                "agent": "karim",
                "business_id": "biz_k",
                "tool_db": campaign_workspace,
                "workspace_id": "ws_k",
                "request_id": "req-2",
            }
        },
    )

    assert "was not found" not in result, result
    assert captured["path"] == "120252033253890323/insights", captured
    assert '"status": "ok"' in result


# ------------------------------- 3. the agent is handed the campaign list --


async def test_karim_turn_receives_the_campaign_list_and_the_focused_campaign(
    campaign_workspace: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision = brain.RoutingDecision(agent="karim", reason="campaign performance question")
    chat = _FakeChat(decision)
    handle = SimpleNamespace(chat=chat, structured=chat.with_structured_output)
    karim = _RecordingGraph()

    monkeypatch.setattr(brain, "get_ai_settings", lambda: SimpleNamespace(llm_provider="mock"))
    monkeypatch.setattr(brain, "get_llm", lambda **_kwargs: handle)
    monkeypatch.setattr(brain, "karim_graph", karim)
    monkeypatch.setattr(brain, "compress_history_messages", lambda _history: [])

    await brain.orchestrate(
        "إيه أخبار حملة رمضان؟",
        history=[],
        business_id="biz_k",
        workspace_id="ws_k",
        db=campaign_workspace,
        core_llm=None,
        conversation_id="conv-1",
        request_id="req-3",
        context={"campaign_id": "cmp_ramadan"},
    )

    payload, config = karim.calls[0]
    system_texts = [
        str(message.content)
        for message in payload["messages"]
        if getattr(message, "type", "") == "system" or message.__class__.__name__ == "SystemMessage"
    ]
    joined = "\n".join(system_texts)
    assert "[CURRENT_CAMPAIGN_CONTEXT]" in joined, system_texts
    assert "رمضان 2026" in joined and "cmp_ramadan" in joined

    configurable = config["configurable"]
    assert configurable["campaign_id"] == "cmp_ramadan"
    assert configurable["tool_db"] is campaign_workspace
    assert configurable["workspace_id"] == "ws_k"
    assert configurable["user_supplied"]["latest_user_message"] == "إيه أخبار حملة رمضان؟"


async def test_a_non_campaign_agent_turn_does_not_read_the_campaign_table(
    campaign_workspace: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routing to Nour must not pay for a campaign query (or leak the list)."""
    decision = brain.RoutingDecision(agent="omar", reason="strategy question")
    chat = _FakeChat(decision)
    handle = SimpleNamespace(chat=chat, structured=chat.with_structured_output)

    calls: list[str] = []

    async def _spy(db: Any, business_id: str, campaign_id: str | None) -> str | None:
        calls.append(business_id)
        return "should not be used"

    monkeypatch.setattr(brain, "get_ai_settings", lambda: SimpleNamespace(llm_provider="mock"))
    monkeypatch.setattr(brain, "get_llm", lambda **_kwargs: handle)
    monkeypatch.setattr(brain, "omar_graph", _RecordingGraph())
    monkeypatch.setattr(brain, "compress_history_messages", lambda _history: [])
    monkeypatch.setattr(brain, "_campaign_context_message", _spy)

    await brain.orchestrate(
        "اعمل استراتيجية تسويق",
        history=[],
        business_id="biz_k",
        workspace_id="ws_k",
        db=campaign_workspace,
        conversation_id="conv-2",
        request_id="req-4",
        context={},
    )

    assert calls == [], "the campaign table is read only for campaign-scoped agents"


# --------------------------- 4. the task path ("Analyze this campaign") ------


async def test_analyze_campaign_task_learns_the_campaign_name(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The campaign page's "Analyze" action sends only `campaignId`.

    The task prompt used to receive that opaque id and nothing else, so the agent
    could not say which campaign it had analysed -- or confirm it was the right
    one. The route now enriches the inputs from the workspace row (read-only).
    """
    from app.api import tasks as tasks_module

    captured: dict[str, Any] = {}

    async def _capture(*_args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    # The background run is what would call the agent; capture its inputs instead.
    # Patch the name the route actually calls (it is imported into that module).
    monkeypatch.setattr(tasks_module, "run_task_in_background", _capture)

    register = await api_client.post(
        "/api/v1/auth/register",
        json={
            "email": "karim-task@example.com",
            "password": "correct horse battery staple",
            "fullName": "Owner Person",
            "workspaceName": "Karim Co",
        },
    )
    assert register.status_code == 201, register.text
    body = register.json()
    headers = {"Authorization": f"Bearer {body['accessToken']}"}
    business_id = body["workspaces"][0]["businessId"]

    # Seed the campaign directly: this test is about the task route, not the wizard.
    from app.db.models import Campaign
    from app.db.session import get_sessionmaker

    maker = get_sessionmaker()
    async with maker() as db:
        db.add(
            Campaign(
                id="cmp_from_ui",
                business_id=business_id,
                name="حملة العروض الصيفية",
                status="active",
                meta_campaign_id="120252033617460323",
                budget=1500.0,
                currency="EGP",
            )
        )
        await db.commit()

    # persona_id / task are query parameters (the body carries inputs only) --
    # exactly the shape the frontend's startTask uses.
    resp = await api_client.post(
        "/api/v1/tasks",
        headers=headers,
        params={"persona_id": "karim", "task": "performance_analysis"},
        json={"inputs": {"campaignId": "cmp_from_ui"}},
    )
    assert resp.status_code in (200, 201, 202), resp.text

    inputs = captured.get("inputs") or {}
    assert inputs.get("campaign_id") == "cmp_from_ui"
    assert inputs.get("campaign_name") == "حملة العروض الصيفية"
    assert inputs.get("meta_campaign_id") == "120252033617460323"
    assert inputs.get("campaign_status") == "active"
    assert inputs.get("campaign_budget") == 1500.0


async def test_insights_payload_reports_the_campaign_name(
    campaign_workspace: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Karim must be able to label the numbers with the campaign's name."""
    from app.core.tools.base import ToolContext
    from app.core.tools.meta_read import _handle_get_campaign_insights
    from app.schemas.tools import GetCampaignInsightsInput

    class _FakeGraph:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def get(self, _path: str, _params: dict[str, Any]) -> dict[str, Any]:
            return {
                "data": [
                    {
                        "date_start": "2026-09-01",
                        "date_stop": "2026-09-02",
                        "spend": "200",
                        "impressions": "2000",
                        "clicks": "40",
                    }
                ]
            }

    monkeypatch.setattr("app.ai.marketing.meta_ads.HttpMetaGraphClient", _FakeGraph)
    monkeypatch.setattr("app.core.tools.meta_read._get_meta_token", AsyncMock(return_value="tok"))

    ctx = ToolContext(
        workspace_id="ws_k",
        business_id="biz_k",
        db=campaign_workspace,
        llm=None,  # type: ignore[arg-type]
        request_id="req-5",
        campaign_id="cmp_ramadan",
    )
    payload = await _handle_get_campaign_insights(
        GetCampaignInsightsInput(campaign_id="cmp_ramadan"), ctx
    )

    assert payload.status == "ok", payload
    assert payload.campaign_name == "رمضان 2026"
    assert payload.meta_campaign_id == "120252033253890323"
    assert "رمضان 2026" in payload.evidence[0].detail
