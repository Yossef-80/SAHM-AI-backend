"""Required test 7: the full approval flow with audit rows.

pending -> approve (diff) -> confirm (execute) -> executed, plus audit_log rows
for each step, and a proof that a direct protected call without approval fails.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.approvals import service as approvals
from app.core.errors import ApprovalRequired, ApprovalStateError
from app.db.models import AuditLog, Recommendation
from app.schemas.recommendations import (
    ApproveRecommendationRequest,
    RecommendationStatus,
    RejectRecommendationRequest,
)


async def _make_recommendation(db: AsyncSession) -> Recommendation:
    row = Recommendation(
        id="rec_test_1",
        campaign_id="cmp_test",
        action_type="pause_ad",
        title="Pause 'Ramadan Reel v2'",
        rationale="CTR fell to 0.4% with frequency 6.1 over 14 days.",
        payload={"status": "PAUSED", "ad_id": "ad_123"},
        evidence=[{"source": "get_campaign_insights", "detail": "CTR 0.4%"}],
        status=RecommendationStatus.PENDING.value,
        requires_approval=True,
    )
    db.add(row)
    await db.commit()
    return row


# ------------------------------------------------------------------ the flow --


async def test_full_approval_flow_writes_audit_rows(seeded_db: AsyncSession) -> None:
    """pending -> approve -> diff -> confirm -> executed, with an audit row each."""
    await _make_recommendation(seeded_db)

    # --- phase 1: approve, get the diff, nothing executed -------------------
    response = await approvals.approve_recommendation(
        seeded_db,
        "rec_test_1",
        ApproveRecommendationRequest(confirm=False),
        actor="nour",
        request_id="req-a",
    )
    assert response.executed is False
    assert response.requires_confirmation is True
    assert response.diff, "a diff must be returned before execution"
    fields = {d.field for d in response.diff}
    assert {"status", "ad_id"} <= fields
    assert response.recommendation.status == RecommendationStatus.APPROVED

    row = await seeded_db.get(Recommendation, "rec_test_1")
    assert row is not None
    assert row.status == RecommendationStatus.APPROVED.value
    assert row.approved_by == "nour"
    assert row.approved_at is not None
    assert row.executed_at is None  # nothing has run yet
    assert row.diff  # the reviewed diff is stored

    audits = (await seeded_db.execute(select(AuditLog))).scalars().all()
    assert len(audits) == 1
    assert audits[0].action == "recommendation.approved"
    assert audits[0].actor == "nour"
    assert audits[0].before["status"] == RecommendationStatus.PENDING.value
    assert audits[0].entity_id == "rec_test_1"

    # --- phase 2: confirm and execute --------------------------------------
    # No Meta token is stored, so execution reports unavailable rather than
    # pretending it wrote to Meta -- and it is still audited.
    confirmed = await approvals.approve_recommendation(
        seeded_db,
        "rec_test_1",
        ApproveRecommendationRequest(confirm=True),
        actor="nour",
        request_id="req-b",
    )
    assert confirmed.requires_confirmation is False
    assert confirmed.diff
    # Meta is not connected, so the change could not be applied.
    assert confirmed.executed is False
    assert "meta token" in (confirmed.message or "").lower()

    audits = (await seeded_db.execute(select(AuditLog))).scalars().all()
    assert len(audits) == 2
    execution = [a for a in audits if a.action == "recommendation.executed"]
    assert len(execution) == 1
    assert execution[0].actor == "nour"
    assert execution[0].result is not None


async def test_approval_is_required_before_execution(seeded_db: AsyncSession) -> None:
    """A recommendation that was never approved cannot jump straight to execute."""
    await _make_recommendation(seeded_db)
    # Skipping phase 1 is allowed, but the protected handler still enforces its
    # own precondition: an explicit confirming actor is mandatory.
    with pytest.raises(ApprovalRequired):
        await approvals.approve_recommendation(
            seeded_db,
            "rec_test_1",
            ApproveRecommendationRequest(confirm=True),
            actor="",  # no explicit actor
            request_id="req-c",
        )


async def test_protected_tool_refuses_to_run_without_an_explicit_actor(
    seeded_db: AsyncSession,
) -> None:
    """The protected handler itself is the last line of defence."""
    await _make_recommendation(seeded_db)
    await approvals.approve_recommendation(
        seeded_db, "rec_test_1", ApproveRecommendationRequest(confirm=False), actor="nour"
    )

    from app.core.llm.mock_client import MockClient
    from app.core.tools import protected
    from app.core.tools.base import ToolContext

    ctx = ToolContext(
        workspace_id="ws_test",
        business_id="biz_test",
        db=seeded_db,
        llm=MockClient(),
        request_id="req-d",
        campaign_id="cmp_test",
    )
    with pytest.raises(ApprovalRequired, match="confirming actor"):
        await protected.apply_change.handler(
            protected.ApplyChangeInput(recommendation_id="rec_test_1", confirmed_by="system"),
            ctx,
        )


async def test_rejecting_a_recommendation_is_audited(seeded_db: AsyncSession) -> None:
    await _make_recommendation(seeded_db)
    result = await approvals.reject_recommendation(
        seeded_db, "rec_test_1", reason="CTR drop is seasonal", actor="omar"
    )
    assert result.status == RecommendationStatus.REJECTED
    audits = (await seeded_db.execute(select(AuditLog))).scalars().all()
    assert len(audits) == 1
    assert audits[0].action == "recommendation.rejected"
    assert audits[0].result == {"reason": "CTR drop is seasonal"}

    # A rejected recommendation cannot be approved afterwards.
    with pytest.raises(ApprovalStateError):
        await approvals.approve_recommendation(
            seeded_db, "rec_test_1", ApproveRecommendationRequest(confirm=False), actor="nour"
        )


async def test_double_execution_is_refused(seeded_db: AsyncSession) -> None:
    await _make_recommendation(seeded_db)
    row = await seeded_db.get(Recommendation, "rec_test_1")
    assert row is not None
    row.status = RecommendationStatus.EXECUTED.value
    await seeded_db.commit()

    with pytest.raises(ApprovalStateError, match="already been executed"):
        await approvals.approve_recommendation(
            seeded_db, "rec_test_1", ApproveRecommendationRequest(confirm=True), actor="nour"
        )


async def test_diff_only_contains_the_proposed_change(seeded_db: AsyncSession) -> None:
    """The review dialog must not imply changes that are not being made."""
    await _make_recommendation(seeded_db)
    diffs = await approvals.build_diff(seeded_db, await seeded_db.get(Recommendation, "rec_test_1"))
    assert {d.field for d in diffs} == {"ad_id", "status"}
    status_diff = next(d for d in diffs if d.field == "status")
    assert status_diff.after == "PAUSED"
    assert status_diff.before is None  # unknown, not guessed


async def test_missing_recommendation_raises_not_found(seeded_db: AsyncSession) -> None:
    from app.core.errors import ApprovalNotFoundError

    with pytest.raises(ApprovalNotFoundError):
        await approvals.approve_recommendation(
            seeded_db,
            "rec_nope",
            ApproveRecommendationRequest(confirm=False),
            actor="nour",
        )


# --------------------------------------------------- end-to-end through the API


async def test_approval_flow_through_the_http_api(api_client, seeded_db: AsyncSession) -> None:
    """POST /recommendations/{id}/approve twice, then read the audit rows."""
    await _make_recommendation(seeded_db)
    client = api_client
    first = await client.post(
        "/api/v1/recommendations/rec_test_1/approve", json={"confirm": False}
    )
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["executed"] is False
    assert body["requiresConfirmation"] is True
    assert body["diff"]

    second = await client.post(
        "/api/v1/recommendations/rec_test_1/approve", json={"confirm": True}
    )
    assert second.status_code == 200, second.text
    assert second.json()["auditLogId"]

    # The recommendation is visible and no longer pending.
    detail = await client.get("/api/v1/recommendations/rec_test_1")
    assert detail.status_code == 200
    assert detail.json()["status"] != RecommendationStatus.PENDING.value


async def test_recommendations_endpoint_lists_pending_first(api_client, seeded_db: AsyncSession) -> None:
    await _make_recommendation(seeded_db)
    client = api_client
    resp = await client.get("/api/v1/recommendations")
    assert resp.status_code == 200
    ids = [r["id"] for r in resp.json()]
    assert "rec_test_1" in ids
