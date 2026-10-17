"""Regression tests for ``POST /creatives/generate`` and a saved reference.

The endpoint used to answer **500** to every request that carried a
``referenceAnalysisId``: the saved analysis was pushed onto the request with
``model_copy(update=...)`` -- which skips validation -- so ``payload.style_spec``
was a plain ``dict`` by the time the generation input was built, and the first
``.model_dump()`` on it raised
``AttributeError: 'dict' object has no attribute 'model_dump'``. The UI, with no
readable response, reported it as "could not reach the backend".

These tests pin the fixed behaviour: the call completes, the style the caller
sends is the validated one that reaches the image provider, a stored style that
no longer parses degrades to "no style" instead of failing the request, and a
reference that belongs to another workspace is still ignored.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import BrandMemoryItem

#: ``Origin`` matters: a browser request is what the user actually made, and the
#: CORS middleware's answer is part of what the client can read.
HEADERS = {"X-Workspace-Id": "biz_test", "Origin": "http://localhost:3000"}

#: The shape ``save_reference_analysis`` writes: a JSON blob, not a StyleSpec.
REFERENCE_STYLE = {
    "canvas_ratio": "1:1",
    "palette": ["#0f172a", "#f59e0b"],
    "composition": "centered product shot",
}


async def _seed_reference(
    db: AsyncSession,
    *,
    reference_id: str = "mem_ref_a",
    business_id: str = "biz_test",
    style: object = REFERENCE_STYLE,
) -> None:
    db.add(
        BrandMemoryItem(
            id=reference_id,
            business_id=business_id,
            kind="reference_analysis",
            text="A saved reference analysis.",
            metadata_={"source_name": "saved.png", "summary": "saved", "style": style},
        )
    )
    await db.commit()


async def _generate(api_client: object, **overrides: object) -> object:
    body: dict[str, object] = {
        "prompt": "عايز اعمل عرض لبوست سوشيال ميديا للاعلادات بهوك يشد علي الميكاترونيك",
        "aspectRatio": "1:1",
        "campaignId": None,
        "numImages": 1,
    }
    body.update(overrides)
    return await api_client.post("/api/v1/creatives/generate", headers=HEADERS, json=body)  # type: ignore[attr-defined]


def test_style_spec_payload_accepts_both_shapes() -> None:
    """The helper that keeps a raw dict from reaching ``.model_dump()``."""
    from app.api.creatives import _style_spec_payload
    from app.schemas.creatives import StyleSpec

    validated = StyleSpec(palette=["#ffffff"], composition="centered")
    assert _style_spec_payload(validated)["palette"] == ["#ffffff"]
    assert _style_spec_payload({"palette": ["#ffffff"]})["palette"] == ["#ffffff"]
    assert _style_spec_payload(None) == {}
    # A stored style is untrusted JSON: anything that is not a StyleSpec shape is
    # "no style", never an exception on the request path.
    assert _style_spec_payload("earthy tones") == {}
    assert _style_spec_payload({"palette": "not-a-list"}) == {}


async def test_reference_analysis_id_no_longer_500s(
    api_client: object, seeded_db: AsyncSession
) -> None:
    """The exact request the UI sent: a saved reference, no explicit style."""
    await _seed_reference(seeded_db)

    resp = await _generate(api_client, referenceAnalysisId="mem_ref_a")  # type: ignore[arg-type]

    assert resp.status_code == 200, resp.text  # type: ignore[attr-defined]
    body = resp.json()  # type: ignore[attr-defined]
    assert "error" not in body
    # Offline the reference image itself is not retained, so the honest answer is
    # "unavailable" with a reason -- but it is an *answer*, not a stack trace.
    assert body["status"] in {"ok", "partial", "unavailable"}


async def test_unparseable_saved_style_degrades_instead_of_failing(
    api_client: object, seeded_db: AsyncSession
) -> None:
    """A reference whose stored style is corrupt must not take the endpoint down."""
    await _seed_reference(seeded_db, style={"palette": "not-a-list"})

    resp = await _generate(api_client, referenceAnalysisId="mem_ref_a")  # type: ignore[arg-type]

    assert resp.status_code == 200, resp.text  # type: ignore[attr-defined]


async def test_another_workspaces_reference_is_still_ignored(
    api_client: object, seeded_db: AsyncSession
) -> None:
    """Naming someone else's reference id stays a no-op, not a read across the line."""
    await _seed_reference(seeded_db, reference_id="mem_ref_other", business_id="biz_other")

    resp = await _generate(api_client, referenceAnalysisId="mem_ref_other")  # type: ignore[arg-type]

    assert resp.status_code == 200, resp.text  # type: ignore[attr-defined]


async def test_style_spec_from_the_body_reaches_the_generated_asset(
    api_client: object, seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the style spec: the image provider is actually given it."""
    from app.config import get_settings

    # The mock provider is the only one that can run offline, and it is opt-in so
    # that a mock asset can never masquerade as a generated one.
    monkeypatch.setenv("IMAGE_ALLOW_MOCK_ASSETS", "true")
    get_settings.cache_clear()

    resp = await _generate(  # type: ignore[arg-type]
        api_client,
        styleSpec={"palette": ["#0f172a", "#f59e0b"], "composition": "centered"},
    )

    assert resp.status_code == 200, resp.text  # type: ignore[attr-defined]
    body = resp.json()  # type: ignore[attr-defined]
    assert body["status"] == "ok", body
    assert len(body["assets"]) == 1
    asset = body["assets"][0]
    assert asset["provider"] == "mock"
    # Tool outputs are plain models with no alias generator, so an asset arrives
    # snake_case inside the camelCase envelope.
    assert asset["style_spec"]["palette"] == ["#0f172a", "#f59e0b"]
    # status/source describe the generation and come from the output: reading
    # them off the asset record used to be a 500 on this (success) path.
    assert asset["status"] == "ok"
    assert asset["source"] == "generate_image:mock"


@pytest.mark.asyncio
async def test_delete_creative_reference_success(api_client: object, seeded_db: AsyncSession) -> None:
    await _seed_reference(seeded_db, reference_id="mem_delete_target", business_id="biz_test")
    # Verify it exists
    resp_list = await api_client.get("/api/v1/creatives/references", headers=HEADERS)  # type: ignore[attr-defined]
    assert resp_list.status_code == 200
    assert any(r["id"] == "mem_delete_target" for r in resp_list.json())

    # Delete it
    resp_del = await api_client.delete("/api/v1/creatives/references/mem_delete_target", headers=HEADERS)  # type: ignore[attr-defined]
    assert resp_del.status_code == 204

    # Verify it's gone
    resp_after = await api_client.get("/api/v1/creatives/references", headers=HEADERS)  # type: ignore[attr-defined]
    assert resp_after.status_code == 200
    assert not any(r["id"] == "mem_delete_target" for r in resp_after.json())


@pytest.mark.asyncio
async def test_delete_creative_reference_cross_workspace_isolated(
    api_client: object, seeded_db: AsyncSession
) -> None:
    await _seed_reference(seeded_db, reference_id="mem_foreign_ref", business_id="biz_foreign")
    # Tenancy invariant: accessing or deleting another workspace's resource returns 404
    resp = await api_client.delete("/api/v1/creatives/references/mem_foreign_ref", headers=HEADERS)  # type: ignore[attr-defined]
    assert resp.status_code == 404
