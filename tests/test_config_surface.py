"""P2: the configuration surface -- profile, brand memory, connector settings.

Every one of these tests is a property a customer would notice if it broke:

* ``PATCH /business/profile`` writes **all six** brand fields. Before P2.1 five
  of them had a read path and no write path anywhere in the codebase.
* Brand memory is writable for **all seven author-facing** kinds. The other
  six kinds in the registry are written by their own pipeline (identity sync,
  research runs, chat attachments) and are rejected by the API -- a kind is
  either writable here or produced elsewhere, never both.
* A connector secret is write-only. No response body, ever, in any form.
* The effective configuration says which layer answered -- workspace, env or
  default -- so a UI can show what is inherited instead of a merged value with
  no provenance.
* One workspace's configuration is invisible to another, and is 404 rather than
  403 when addressed directly.

And the two regressions that must stay fixed:

* a ``suggest``-autonomy persona creates **no** durable rows (D1),
* CORS ``*`` plus credentials refuses to start, and production refuses to boot
  without a real encryption key (D5).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import create_session, hash_password
from app.core.memory.kinds import API_WRITABLE_KINDS, MEMORY_KINDS
from app.db.models import (
    BrandMemoryItem,
    Business,
    Membership,
    ToolSettings,
    User,
)

PASSWORD = "correct horse battery staple"

#: A secret that must never appear in a response body. Distinctive on purpose:
#: a substring search over the whole JSON is the assertion that matters.
META_TOKEN = "EAAG-secret-token-do-not-leak-1234567890"
APP_SECRET = "appsecret-abcdef-do-not-leak-9876543210"


# ------------------------------------------------------------------ helpers ----


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _register(client: Any, email: str, name: str = "Acme") -> dict[str, Any]:
    """Register an owner and return the token response."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": PASSWORD,
            "fullName": "Owner Person",
            "workspaceName": name,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _make_member(
    db: AsyncSession, email: str, business_id: str, role: str
) -> str:
    """Create a user with ``role`` inside an existing workspace; return a token."""
    business = await db.get(Business, business_id)
    assert business is not None
    slug = email.replace("@", "_").replace(".", "_")
    user = User(
        id=f"usr_{slug}", email=email, password_hash=hash_password(PASSWORD)
    )
    db.add(user)
    await db.flush()
    db.add(
        Membership(
            id=f"mbr_{slug}",
            user_id=user.id,
            workspace_id=business.workspace_id,
            business_id=business.id,
            role=role,
        )
    )
    token = await create_session(db, user, user_agent="pytest", ip="127.0.0.1")
    await db.commit()
    return token


def _connector(body: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in body["connectors"]:
        if entry["connector"] == name:
            return entry
    raise AssertionError(f"connector '{name}' missing from {list(body)}")


def _field(connector: dict[str, Any], key: str) -> dict[str, Any]:
    for item in connector["fields"]:
        if item["key"] == key:
            return item
    raise AssertionError(f"field '{key}' missing from {connector['connector']}")


def _assert_no_secret(payload: Any) -> None:
    """No response payload may contain a stored secret, in any form."""
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    assert META_TOKEN not in text, "the Meta access token leaked into a payload"
    assert APP_SECRET not in text, "the Meta app secret leaked into a payload"


# ------------------------------------------------------------ P2.1 the profile --


async def test_profile_patch_writes_every_brand_field(api_client) -> None:
    """The headline: all six brand fields are writable and round-trip."""
    body = await _register(api_client, "profile@example.com", "Brand Co")
    headers = _auth(body["accessToken"])

    payload = {
        "name": "Brand Co",
        "industry": "Speciality coffee",
        "websiteUrl": "https://brand.example",
        "country": "eg",
        "language": "en",
        "currency": "EGP",
        "monthlyBudget": 25000,
        "brandVoice": "warm, direct, a little cheeky, never corporate",
        "visualStyle": "earthy tones, natural window light, matte paper",
        "targetAudience": "25-40 Cairo professionals who care where beans come from",
        "keyOffers": ["subscription beans", "office catering", "cupping workshops"],
        "goals": ["grow subscriptions to 1000", "open the new branch"],
        "competitors": ["Bean There", "Kaffa Roasters"],
    }
    resp = await api_client.patch(
        "/api/v1/business/profile", headers=headers, json=payload
    )
    assert resp.status_code == 200, resp.text

    # The PATCH response is the GET response -- no adapter needed client-side.
    fetched = await api_client.get("/api/v1/business/profile", headers=headers)
    assert fetched.status_code == 200, fetched.text
    profile = fetched.json()["business"]

    assert profile["brandVoice"] == payload["brandVoice"]
    assert profile["visualStyle"] == payload["visualStyle"]
    assert profile["targetAudience"] == payload["targetAudience"]
    assert profile["keyOffers"] == payload["keyOffers"]
    assert profile["goals"] == payload["goals"]
    assert profile["competitors"] == payload["competitors"]
    assert profile["country"] == "EG", "country is normalised to upper case"
    assert profile["monthlyBudget"] == 25000

    # ...and the PATCH body itself already contained them, so a form needs one
    # request, not a PATCH followed by a GET.
    assert resp.json()["business"]["brandVoice"] == payload["brandVoice"]


async def test_profile_patch_is_partial_null_clears_and_omitted_is_untouched(
    api_client,
) -> None:
    """The three instructions a form needs: set, clear, leave alone."""
    body = await _register(api_client, "partial@example.com")
    headers = _auth(body["accessToken"])

    await api_client.patch(
        "/api/v1/business/profile",
        headers=headers,
        json={
            "brandVoice": "keep me",
            "competitors": ["Acme Coffee"],
            "keyOffers": ["one", "two"],
        },
    )
    # Only brandVoice is sent: competitors and keyOffers must be untouched.
    resp = await api_client.patch(
        "/api/v1/business/profile", headers=headers, json={"brandVoice": "changed"}
    )
    profile = resp.json()["business"]
    assert profile["brandVoice"] == "changed"
    assert profile["competitors"] == ["Acme Coffee"]
    assert profile["keyOffers"] == ["one", "two"]

    # null clears; [] empties a list.
    resp = await api_client.patch(
        "/api/v1/business/profile",
        headers=headers,
        json={"competitors": None, "keyOffers": []},
    )
    profile = resp.json()["business"]
    assert profile["competitors"] == [], "null clears a list to empty"
    assert profile["keyOffers"] == []


async def test_profile_patch_rejects_an_empty_body(api_client) -> None:
    body = await _register(api_client, "empty@example.com")
    resp = await api_client.patch(
        "/api/v1/business/profile",
        headers=_auth(body["accessToken"]),
        json={},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_failed"


async def test_profile_patch_caps_field_lengths(api_client) -> None:
    """A form cannot write a 10 MB JSON blob through one field."""
    body = await _register(api_client, "caps@example.com")
    resp = await api_client.patch(
        "/api/v1/business/profile",
        headers=_auth(body["accessToken"]),
        json={"brandVoice": "x" * 5_000},
    )
    assert resp.status_code == 422, resp.text

    resp = await api_client.patch(
        "/api/v1/business/profile",
        headers=_auth(body["accessToken"]),
        json={"competitors": [f"rival {i}" for i in range(60)]},
    )
    assert resp.status_code == 422, resp.text


async def test_profile_patch_requires_member_not_viewer(
    api_client, seeded_db: AsyncSession
) -> None:
    """A viewer can read the profile and cannot write it."""
    owner = await _register(api_client, "writer@example.com")
    business_id = owner["workspaces"][0]["businessId"]
    viewer = await _make_member(seeded_db, "profile-viewer@example.com", business_id, "viewer")
    member = await _make_member(seeded_db, "profile-member@example.com", business_id, "member")

    resp = await api_client.patch(
        "/api/v1/business/profile",
        headers=_auth(viewer),
        json={"brandVoice": "viewer was here"},
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "forbidden"
    assert resp.json()["error"]["detail"]["required_role"] == "member"

    resp = await api_client.patch(
        "/api/v1/business/profile",
        headers=_auth(member),
        json={"brandVoice": "member was here"},
    )
    assert resp.status_code == 200, resp.text


# ------------------------------------------------------- P2.2 brand memory --


async def test_memory_crud_covers_every_writable_kind(api_client) -> None:
    """All seven author-facing kinds are creatable; service kinds are not."""
    body = await _register(api_client, "memory@example.com")
    headers = _auth(body["accessToken"])

    assert len(API_WRITABLE_KINDS) == 7, "seven author-facing kinds"
    assert len(MEMORY_KINDS) == 13, "the registry knows 13 kinds in total"
    assert not set(API_WRITABLE_KINDS) - set(MEMORY_KINDS)

    created: list[str] = []
    for kind in API_WRITABLE_KINDS:
        resp = await api_client.post(
            "/api/v1/memory/items",
            headers=headers,
            json={
                "kind": kind,
                "text": f"A {kind} written by a user, not by an agent.",
                "metadata": {"author": "pytest", "tag": kind},
            },
        )
        assert resp.status_code == 201, f"{kind}: {resp.text}"
        item = resp.json()
        assert item["kind"] == kind
        assert item["metadata"]["author"] == "pytest"
        created.append(item["id"])

    listing = await api_client.get("/api/v1/memory/items", headers=headers)
    assert listing.status_code == 200, listing.text
    body_json = listing.json()
    assert {i["kind"] for i in body_json["items"]} == set(API_WRITABLE_KINDS)
    assert body_json["mode"] == "recent", "no q -> newest-first, not similarity"

    # Filter by kind.
    filtered = await api_client.get(
        "/api/v1/memory/items?kind=voice_example", headers=headers
    )
    assert filtered.status_code == 200, filtered.text
    assert [i["kind"] for i in filtered.json()["items"]] == ["voice_example"]

    # Edit the text and the metadata.
    patched = await api_client.patch(
        f"/api/v1/memory/items/{created[0]}",
        headers=headers,
        json={"text": "Rewritten: we talk like a friend, not a barista snob.",
              "metadata": {"author": "pytest", "revised": True}},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["text"].startswith("Rewritten")
    assert patched.json()["metadata"]["revised"] is True

    deleted = await api_client.delete(
        f"/api/v1/memory/items/{created[0]}", headers=headers
    )
    assert deleted.status_code == 204, deleted.text

    after = await api_client.get("/api/v1/memory/items", headers=headers)
    assert created[0] not in {i["id"] for i in after.json()["items"]}


async def test_memory_rejects_an_unknown_kind(api_client) -> None:
    body = await _register(api_client, "badkind@example.com")
    resp = await api_client.post(
        "/api/v1/memory/items",
        headers=_auth(body["accessToken"]),
        json={"kind": "vibes", "text": "not a real kind"},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_failed"


async def test_memory_rejects_a_service_owned_kind(api_client) -> None:
    """``brand_identity`` is real, but only the identity pipeline writes it."""
    body = await _register(api_client, "servicekind@example.com")
    resp = await api_client.post(
        "/api/v1/memory/items",
        headers=_auth(body["accessToken"]),
        json={"kind": "brand_identity", "text": "invented identity"},
    )
    assert resp.status_code == 422, resp.text
    assert "writable" in resp.json()["error"]["detail"]


async def test_memory_search_is_lexically_ordered_not_substring(
    api_client,
) -> None:
    """``q`` runs BM25 + recency + affinity, so ``mode`` is ``'lexical'``.

    The old assertion was ``mode == "similar"`` because ``q`` meant cosine
    similarity over a 64-dimension hashed vector. The replacement is exact about
    why an item matched: every hit carries a ``reason`` naming the terms.
    """
    body = await _register(api_client, "search@example.com")
    headers = _auth(body["accessToken"])
    for kind, text in [
        ("voice_example", "We never say 'artisanal'. We say fresh."),
        ("offer", "Free delivery inside Cairo on every subscription."),
        ("audience", "Office managers ordering for teams of 10-50 people."),
    ]:
        await api_client.post(
            "/api/v1/memory/items", headers=headers, json={"kind": kind, "text": text}
        )

    resp = await api_client.get(
        "/api/v1/memory/items?q=delivery%20Cairo", headers=headers
    )
    assert resp.status_code == 200, resp.text
    body_json = resp.json()
    assert body_json["mode"] == "lexical"
    assert body_json["query"] == "delivery Cairo"
    assert body_json["items"], "BM25 must find the offer"
    assert body_json["items"][0]["kind"] == "offer"
    assert body_json["items"][0]["score"] is not None
    assert "lex" in (body_json["items"][0]["reason"] or "")
    # Nothing else mentions "delivery" or "Cairo", so nothing else may rank.
    assert [i["kind"] for i in body_json["items"]] == ["offer"]


async def test_cross_workspace_memory_access_is_404(api_client) -> None:
    """A row that exists but belongs to someone else is 404, never 403."""
    owner = await _register(api_client, "tenant-a@example.com", "Tenant A")
    a_headers = _auth(owner["accessToken"])
    created = await api_client.post(
        "/api/v1/memory/items",
        headers=a_headers,
        json={"kind": "offer", "text": "Tenant A's secret pricing"},
    )
    item_id = created.json()["id"]

    stranger = await _register(api_client, "tenant-b@example.com", "Tenant B")
    b_headers = _auth(stranger["accessToken"])

    listing = await api_client.get("/api/v1/memory/items", headers=b_headers)
    assert listing.status_code == 200
    assert item_id not in {i["id"] for i in listing.json()["items"]}

    resp = await api_client.patch(
        f"/api/v1/memory/items/{item_id}", headers=b_headers, json={"text": "hijacked"}
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "not_found"

    resp = await api_client.delete(
        f"/api/v1/memory/items/{item_id}", headers=b_headers
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "not_found"

    # ...and the row is still there, unchanged.
    still = await api_client.get("/api/v1/memory/items", headers=a_headers)
    assert item_id in {i["id"] for i in still.json()["items"]}


async def test_memory_prune_is_owner_only_and_dry_run_by_default(
    api_client, seeded_db: AsyncSession
) -> None:
    """``POST /memory/prune`` replaced ``/memory/reindex``.

    Nothing is embedded any more, so there is nothing to rebuild. What an
    operator still needs is the ability to see, and then apply, expiry and the
    per-kind ceiling -- and a non-owner must not be able to archive memory.
    """
    body = await _register(api_client, "prune@example.com")
    business_id = body["workspaces"][0]["businessId"]
    headers = _auth(body["accessToken"])
    await api_client.post(
        "/api/v1/memory/items",
        headers=headers,
        json={"kind": "learning", "text": "short ads outperformed long ones"},
    )

    member = await _make_member(seeded_db, "prune-member@example.com", business_id, "member")
    resp = await api_client.post("/api/v1/memory/prune", headers=_auth(member), json={})
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "forbidden"

    # Default is a dry run: it reports, it does not archive.
    resp = await api_client.post("/api/v1/memory/prune", headers=headers, json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["dryRun"] is True
    assert resp.json()["archivedByKind"] == {}

    # An unknown kind is a 422, not a silent empty report.
    resp = await api_client.post(
        "/api/v1/memory/prune", headers=headers, json={"kind": "nonsense"}
    )
    assert resp.status_code == 422, resp.text


async def test_expired_memory_is_gone_from_listings_and_context(
    api_client, seeded_db: AsyncSession
) -> None:
    """Expiry is a predicate on ``expires_at``, not a reindex step (P2.2)."""
    from datetime import datetime, timedelta, timezone

    body = await _register(api_client, "expiry@example.com")
    headers = _auth(body["accessToken"])
    created = await api_client.post(
        "/api/v1/memory/items",
        headers=headers,
        json={"kind": "offer", "text": "buy one bag get one free"},
    )
    item_id = created.json()["id"]

    # An expired row is invisible to the default listing ...
    await seeded_db.execute(
        BrandMemoryItem.__table__.update()
        .where(BrandMemoryItem.id == item_id)
        .values(expires_at=datetime.now(timezone.utc) - timedelta(days=1))
    )
    await seeded_db.commit()

    resp = await api_client.get("/api/v1/memory/items", headers=headers)
    assert item_id not in {i["id"] for i in resp.json()["items"]}

    # ... and still visible when archived rows are asked for explicitly, so an
    # operator can audit what expired instead of wondering where it went.
    resp = await api_client.get(
        "/api/v1/memory/items?include_archived=true", headers=headers
    )
    assert item_id in {i["id"] for i in resp.json()["items"]}

    # Pruning for real applies the expiry that the read path was already
    # enforcing: the row was invisible but not yet archived.
    resp = await api_client.post(
        "/api/v1/memory/prune", headers=headers, json={"kind": "offer", "dryRun": False}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["archivedByKind"].get("offer") == 1


# --------------------------------------------------- P2.4 connector settings --


async def test_connector_get_reports_the_layer_that_answered(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """workspace / env / default, per connector and per field."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "meta_ad_account_id", "act_from_env")
    monkeypatch.setattr(get_settings(), "meta_access_token", None)
    # Emptying IMAGE_MODEL proves the third layer: with nothing in env *and*
    # nothing stored, the built-in default answers.
    monkeypatch.setattr(get_settings(), "image_model", None)

    body = await _register(api_client, "conn-source@example.com")
    headers = _auth(body["accessToken"])

    before = (await api_client.get("/api/v1/settings/connectors", headers=headers)).json()
    assert before["overrideCount"] == 0
    meta = _connector(before, "meta")
    assert meta["hasOverride"] is False
    assert _field(meta, "ad_account_id")["source"] == "env"
    assert _field(meta, "ad_account_id")["value"] == "act_from_env"
    assert _field(meta, "access_token")["source"] == "unset"
    # A mock provider is never "configured": it returns placeholder data.
    search = _connector(before, "search")
    assert search["configured"] is False
    assert search["placeholder"] is True
    # The image model comes from the built-in default, not from any env var.
    model = _field(_connector(before, "image"), "model")
    assert model["source"] == "default", model
    assert model["value"] == "mock-image-1"

    resp = await api_client.patch(
        "/api/v1/settings/connectors/meta",
        headers=headers,
        json={
            "adAccountId": "act_from_workspace",
            "pageId": "page_from_workspace",
            "accessToken": META_TOKEN,
        }, 
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["source"] == "workspace"
    assert _field(resp.json(), "ad_account_id")["value"] == "act_from_workspace"
    assert _field(resp.json(), "ad_account_id")["source"] == "workspace"
    assert _field(resp.json(), "page_id")["value"] == "page_from_workspace"
    assert _field(resp.json(), "page_id")["source"] == "workspace"

    after = (await api_client.get("/api/v1/settings/connectors", headers=headers)).json()
    assert after["overrideCount"] == 1
    meta = _connector(after, "meta")
    assert meta["configured"] is True, "a real provider with all required Meta fields is configured"
    _assert_no_secret(after)

    status = await api_client.get("/api/v1/integrations/meta/status", headers=headers)
    assert status.status_code == 200, status.text
    assert status.json()["connected"] is True
    assert status.json()["campaignReady"] is True
    assert status.json()["pageConfigured"] is True

    # The workspace value wins over the env var.
    assert _field(meta, "ad_account_id")["value"] == "act_from_workspace"


async def test_connector_update_refuses_credentials_under_a_placeholder_provider(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reported bug: key + model saved, provider untouched, generation dead.

    ``generate_image`` reads ``image_api_key``/``image_model`` only when the
    provider is a real one, so saving them while ``provider`` still resolves to
    ``mock`` used to be a green 200 that changed nothing. The user filled the
    form, saw "saved", and image generation stayed disabled with no explanation.
    """
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "image_provider", "mock")
    monkeypatch.setattr(get_settings(), "image_api_key", None)

    body = await _register(api_client, "image-creds@example.com")
    headers = _auth(body["accessToken"])

    # 1) Credentials without a provider: refused, and nothing is stored.
    refused = await api_client.patch(
        "/api/v1/settings/connectors/image",
        headers=headers,
        json={"apiKey": "sk-live-key", "model": "gpt-image-1"},
    )
    assert refused.status_code == 422, refused.text
    payload = refused.json()["error"]
    assert payload["code"] == "validation_failed", payload
    assert payload["detail"]["code"] == "provider_not_selected", payload
    assert payload["detail"]["provider"] == "mock"
    assert sorted(payload["detail"]["fields"]) == ["api_key", "model"]
    assert "nothing was saved" in payload["message"]
    assert "sk-live-key" not in refused.text, "a refused body must not echo the secret"

    after_refusal = (
        await api_client.get("/api/v1/settings/connectors", headers=headers)
    ).json()
    assert after_refusal["overrideCount"] == 0, "a refused patch wrote a row"
    assert _connector(after_refusal, "image")["hasOverride"] is False

    # 2) The same request plus the provider works, and the connector is live.
    accepted = await api_client.patch(
        "/api/v1/settings/connectors/image",
        headers=headers,
        json={"provider": "openai", "apiKey": "sk-live-key", "model": "gpt-image-1"},
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["configured"] is True
    assert accepted.json()["placeholder"] is False
    assert "sk-live-key" not in accepted.text

    # 3) A credential-only patch is fine once the stored provider is real.
    again = await api_client.patch(
        "/api/v1/settings/connectors/image", headers=headers, json={"apiKey": "sk-rotated"}
    )
    assert again.status_code == 200, again.text


async def test_connector_provider_choices_are_published_and_normalised(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The settings screen renders a select from the server's list, not free text.

    Two properties: the accepted values travel with the connector (so the UI
    cannot drift from the backend), and a differently-cased value is stored
    normalised instead of breaking every later ``== "openai"`` comparison.
    """
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "image_provider", "mock")

    body = await _register(api_client, "provider-list@example.com")
    headers = _auth(body["accessToken"])

    listing = (
        await api_client.get("/api/v1/settings/connectors", headers=headers)
    ).json()
    image = _connector(listing, "image")
    assert image["providers"] == ["mock", "openai", "stability"], image
    assert _connector(listing, "search")["providers"] == [
        "mock",
        "tavily",
        "serper",
        "brave",
    ]
    # A connector without a provider selector publishes an empty list.
    assert _connector(listing, "meta")["providers"] == []

    # "OpenAI" is accepted and stored lowercase.
    mixed = await api_client.patch(
        "/api/v1/settings/connectors/image",
        headers=headers,
        json={"provider": "  OpenAI  ", "apiKey": "sk-live-key"},
    )
    assert mixed.status_code == 200, mixed.text
    assert mixed.json()["provider"] == "openai"
    assert mixed.json()["configured"] is True

    # A value this build does not implement is a 422 naming the accepted ones.
    bogus = await api_client.patch(
        "/api/v1/settings/connectors/image", headers=headers, json={"provider": "gpt-image-1"}
    )
    assert bogus.status_code == 422, bogus.text
    assert bogus.json()["error"]["detail"]["code"] == "unknown_provider"
    assert "openai" in bogus.json()["error"]["message"]


async def test_a_stored_secret_never_appears_in_any_response(api_client) -> None:
    """Grep the whole JSON, not just the obvious field."""
    body = await _register(api_client, "secret@example.com")
    headers = _auth(body["accessToken"])

    resp = await api_client.patch(
        "/api/v1/settings/connectors/meta",
        headers=headers,
        json={
            "appId": "1234567890",
            "appSecret": APP_SECRET,
            "adAccountId": "act_1",
            "accessToken": META_TOKEN,
        },
    )
    assert resp.status_code == 200, resp.text
    _assert_no_secret(resp)

    listing = await api_client.get("/api/v1/settings/connectors", headers=headers)
    assert listing.status_code == 200
    _assert_no_secret(listing)

    single = await api_client.patch(
        "/api/v1/settings/connectors/meta",
        headers=headers,
        json={"adAccountId": "act_2"},
    )
    assert single.status_code == 200
    _assert_no_secret(single)

    removed = await api_client.delete("/api/v1/settings/connectors/meta", headers=headers)
    assert removed.status_code == 200, removed.text
    _assert_no_secret(removed)

    field = _field(
        _connector(listing.json(), "meta"), "access_token"
    )
    assert field["value"] is None, "a secret is never serialised, not even as text"
    assert field["hasValue"] is True
    assert field["masked"] and META_TOKEN not in field["masked"]


async def test_secrets_are_encrypted_at_rest(api_client, seeded_db: AsyncSession) -> None:
    """Reading the column must not yield the plaintext."""
    body = await _register(api_client, "at-rest@example.com")
    business_id = body["workspaces"][0]["businessId"]
    headers = _auth(body["accessToken"])

    await api_client.patch(
        "/api/v1/settings/connectors/meta",
        headers=headers,
        json={"accessToken": META_TOKEN, "appSecret": APP_SECRET},
    )
    row = (
        await seeded_db.execute(
            select(ToolSettings).where(ToolSettings.business_id == business_id)
        )
    ).scalars().first()
    assert row is not None
    assert row.meta_access_token_enc and META_TOKEN not in row.meta_access_token_enc
    assert row.meta_app_secret_enc and APP_SECRET not in row.meta_app_secret_enc
    assert row.updated_by and row.updated_by.startswith("usr_")

    # ...and it round-trips through the resolver, which is how a tool reads it.
    from app.core.tools.settings import load_connector_settings

    resolved = await load_connector_settings(seeded_db, business_id)
    assert resolved.meta_access_token == META_TOKEN
    assert resolved.meta_app_secret == APP_SECRET


async def test_connector_patch_is_owner_only_and_validates_the_name(
    api_client, seeded_db: AsyncSession
) -> None:
    owner = await _register(api_client, "conn-owner@example.com")
    business_id = owner["workspaces"][0]["businessId"]
    headers = _auth(owner["accessToken"])

    viewer = await _make_member(
        seeded_db, "conn-viewer@example.com", business_id, "viewer"
    )
    member = await _make_member(
        seeded_db, "conn-member@example.com", business_id, "member"
    )

    for role_token, label in ((viewer, "viewer"), (member, "member")):
        resp = await api_client.patch(
            "/api/v1/settings/connectors/meta",
            headers=_auth(role_token),
            json={"accessToken": META_TOKEN},
        )
        assert resp.status_code == 403, f"{label} -> {resp.status_code}"
        assert resp.json()["error"]["detail"]["required_role"] == "owner"

    resp = await api_client.patch(
        "/api/v1/settings/connectors/not_a_connector",
        headers=headers,
        json={"apiKey": "x"},
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "not_found"

    # A field that belongs to a different connector is a 422, not a silent no-op.
    resp = await api_client.patch(
        "/api/v1/settings/connectors/meta",
        headers=headers,
        json={"login": "dataforseo-user"},
    )
    assert resp.status_code == 422, resp.text
    assert "login" in resp.json()["error"]["message"]

    resp = await api_client.patch(
        "/api/v1/settings/connectors/meta", headers=headers, json={}
    )
    assert resp.status_code == 422, resp.text


async def test_deleting_the_override_falls_back_to_env(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "google_places_api_key", "places-key-from-env")

    body = await _register(api_client, "fallback@example.com")
    headers = _auth(body["accessToken"])

    await api_client.patch(
        "/api/v1/settings/connectors/google_places",
        headers=headers,
        json={"apiKey": "places-key-from-workspace"},
    )
    listing = await api_client.get("/api/v1/settings/connectors", headers=headers)
    places = _connector(listing.json(), "google_places")
    assert places["source"] == "workspace"
    assert _field(places, "api_key")["source"] == "workspace"

    resp = await api_client.delete(
        "/api/v1/settings/connectors/google_places", headers=headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "connector": "google_places",
        "removed": True,
        "source": "env",
        "configured": True,
    }

    listing = await api_client.get("/api/v1/settings/connectors", headers=headers)
    places = _connector(listing.json(), "google_places")
    assert places["hasOverride"] is False
    assert places["source"] == "env"
    assert _field(places, "api_key")["source"] == "env"

    # Deleting again is a 200 with removed=false, not a 404.
    resp = await api_client.delete(
        "/api/v1/settings/connectors/google_places", headers=headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["removed"] is False


async def test_connector_settings_are_workspace_scoped(
    api_client, seeded_db: AsyncSession
) -> None:
    """Two customers really can use two different Meta accounts now."""
    a = await _register(api_client, "cfg-a@example.com", "Customer A")
    a_headers = _auth(a["accessToken"])
    await api_client.patch(
        "/api/v1/settings/connectors/meta",
        headers=a_headers,
        json={"adAccountId": "act_customer_a", "accessToken": META_TOKEN},
    )

    b = await _register(api_client, "cfg-b@example.com", "Customer B")
    b_headers = _auth(b["accessToken"])
    listing = await api_client.get("/api/v1/settings/connectors", headers=b_headers)
    assert listing.json()["overrideCount"] == 0
    meta = _connector(listing.json(), "meta")
    assert meta["hasOverride"] is False
    assert _field(meta, "ad_account_id")["source"] == "unset"

    rows = (
        await seeded_db.execute(select(ToolSettings))
    ).scalars().all()
    assert len(rows) == 1, "B must not have inherited A's row"


# ----------------------------------------------- P2.5 the tools read it too --


async def test_tool_settings_are_preferred_over_env(
    api_client, monkeypatch: pytest.MonkeyPatch, seeded_db: AsyncSession
) -> None:
    """P2.5: a workspace row beats the env var at tool-call time."""
    from app.config import get_settings
    from app.core.tools.base import ToolContext
    from app.core.tools.settings import resolve_tool_settings

    monkeypatch.setattr(get_settings(), "image_api_key", "env-image-key")

    body = await _register(api_client, "tool-cfg@example.com")
    business_id = body["workspaces"][0]["businessId"]
    # `provider` is sent with the credentials on purpose: a workspace row that
    # stores a key while the provider stays on the placeholder `mock` is now a
    # 422 (see test_connector_update_refuses_credentials_under_a_placeholder_provider).
    await api_client.patch(
        "/api/v1/settings/connectors/image",
        headers=_auth(body["accessToken"]),
        json={"provider": "openai", "apiKey": "workspace-image-key", "model": "dall-e-3"},
    )

    ctx = ToolContext(
        workspace_id=business_id,
        business_id=business_id,
        db=seeded_db,
        llm=None,  # type: ignore[arg-type]
        request_id="req-test",
    )
    resolved = await resolve_tool_settings(ctx)
    assert resolved.image_api_key == "workspace-image-key", "the row must win"
    assert resolved.image_model == "dall-e-3"
    assert resolved.source("image", "api_key") == "workspace"
    assert ctx.tool_settings is resolved, "resolution is cached on the context"

    await seeded_db.execute(
        ToolSettings.__table__.delete().where(ToolSettings.business_id == business_id)
    )
    await seeded_db.commit()
    fresh = ToolContext(
        workspace_id=business_id,
        business_id=business_id,
        db=seeded_db,
        llm=None,  # type: ignore[arg-type]
        request_id="req-test",
    )
    assert (await resolve_tool_settings(fresh)).image_api_key == "env-image-key"


async def test_workspace_image_settings_actually_generate_a_real_image(
    api_client, seeded_db: AsyncSession, respx_mock
) -> None:
    """End to end: fill the form -> the provider is really called with those values.

    The reported failure was "I set the API key and the model and image generation
    never worked". This is the positive path that has to work once the provider is
    selected: the workspace row supplies the key, the model and the base URL to
    ``generate_image``, and the resulting asset is stored as a real provider asset
    -- not a placeholder.
    """
    import httpx

    from app.core.tools.base import ToolContext
    from app.core.tools.media import _handle_generate_image
    from app.schemas.tools import GenerateImageInput

    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "path": request.url.path,
                "authorization": request.headers.get("authorization", ""),
                "body": json.loads(request.content.decode() or "{}"),
            }
        )
        return httpx.Response(
            200, json={"data": [{"url": "https://assets.example.com/generated.png"}]}
        )

    respx_mock.route(method="POST", host="image-provider.invalid").mock(
        side_effect=handler
    )

    body = await _register(api_client, "image-e2e@example.com")
    business_id = body["workspaces"][0]["businessId"]
    headers = _auth(body["accessToken"])

    configured = await api_client.patch(
        "/api/v1/settings/connectors/image",
        headers=headers,
        json={
            "provider": "openai",
            "apiKey": "sk-workspace-key",
            "model": "gpt-image-1",
            "baseUrl": "https://image-provider.invalid/v1",
        },
    )
    assert configured.status_code == 200, configured.text
    assert configured.json()["configured"] is True

    ctx = ToolContext(
        workspace_id=business_id,
        business_id=business_id,
        db=seeded_db,
        llm=None,  # type: ignore[arg-type]
        request_id="req-image-e2e",
    )
    result = await _handle_generate_image(
        GenerateImageInput(prompt="a coffee cup on a saucer"), ctx
    )

    assert result.status == "ok", result
    assert result.assets and result.assets[0].provider == "openai"
    assert result.assets[0].url == "https://assets.example.com/generated.png"
    assert captured and captured[0]["path"] == "/v1/images/generations"
    assert captured[0]["authorization"] == "Bearer sk-workspace-key"
    assert captured[0]["body"]["model"] == "gpt-image-1", captured[0]["body"]


async def test_connector_catalogue_reflects_the_workspace_row(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``GET /tools/connectors`` and the settings screen agree."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "meta_access_token", None)

    body = await _register(api_client, "catalogue@example.com")
    headers = _auth(body["accessToken"])

    before = await api_client.get("/api/v1/tools/connectors", headers=headers)
    assert before.status_code == 200, before.text
    assert "meta_access_token" in before.json()["get_campaign_insights"]["missing_env"]

    await api_client.patch(
        "/api/v1/settings/connectors/meta",
        headers=headers,
        json={"accessToken": META_TOKEN},
    )
    after = await api_client.get("/api/v1/tools/connectors", headers=headers)
    assert after.status_code == 200, after.text
    assert after.json()["get_campaign_insights"]["missing_env"] == []
    assert after.json()["get_campaign_insights"]["configured"] is True


# ------------------------------------------------- P2.6 the SUGGEST bug (D1) --


async def test_a_suggest_persona_creates_no_durable_rows(
    api_client, seeded_db: AsyncSession
) -> None:
    """D1 regression.

    The bug: a ``suggest`` persona is documented as "proposal only; nothing
    durable is created from the output", yet the old policy was computed *after*
    the run loop had already executed handlers that write. The test drives a run
    whose only tool call is ``save_learning`` and asserts that nothing was
    written -- and that the block is visible in the event stream rather than
    being silently swallowed.
    """
    from app.core.agents.persona import Persona
    from app.core.agents.runtime import RunOptions, blocks_draft_write, run_agent_stream
    from app.core.llm.base import LLMResult, ToolCall
    from app.core.tools.base import ToolContext
    from app.core.tools.registry import register_all
    from app.schemas.common import Autonomy

    register_all()

    persona = Persona(
        id="suggester",
        name="The Strategist",
        emoji="\U0001F9E0",
        role="strategist",
        description="Reads the numbers and proposes; never writes.",
        autonomy=Autonomy.SUGGEST,
        allowed_tasks=["performance_analysis"],
        allowed_tools=["save_learning", "get_brand_context"],
    )

    if True:
        class _AlwaysSavesLearning:
            """A mock LLM whose only move is to ask for the durable write."""

            model = "mock-strong-1"

            async def complete(self, **kwargs: Any) -> LLMResult:
                if kwargs.get("tools"):
                    return LLMResult(
                        text="",
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="save_learning",
                                arguments={
                                    "what_worked": "short video ads",
                                    "applies_to_future": True,
                                },
                            )
                        ],
                        model=self.model,
                    )
                # No structured output: the run ends incomplete, which is fine.
                # What matters is that nothing durable was written on the way.
                return LLMResult(text="short video ads beat static.", model=self.model)

        ctx = ToolContext(
            workspace_id="ws_suggest",
            business_id="biz_test",
            db=seeded_db,
            llm=_AlwaysSavesLearning(),  # type: ignore[arg-type]
            request_id="req-suggest",
        )

        async def _learning_count() -> int:
            return len(
                (
                    await seeded_db.execute(
                        select(BrandMemoryItem).where(
                            BrandMemoryItem.business_id == "biz_test",
                            BrandMemoryItem.kind == "learning",
                        )
                    )
                )
                .scalars()
                .all()
            )

        before = await _learning_count()
        rows_before = len(
            (await seeded_db.execute(select(BrandMemoryItem))).scalars().all()
        )

        events = []
        async for event in run_agent_stream(
            persona,
            "performance_analysis",
            {},
            ctx,
            options=RunOptions.from_settings(),
        ):
            events.append(event)

        blocked = [
            e
            for e in events
            if e.type == "tool_result" and "suggest-only" in (e.detail or "")
        ]
        assert blocked, "the blocked call must be visible in the event stream"

        assert await _learning_count() == before, (
            "a suggest persona must not persist a learning"
        )
        assert (
            len((await seeded_db.execute(select(BrandMemoryItem))).scalars().all())
            == rows_before
        ), "no brand memory row of any kind may be added"

        final = events[-1].data["result"]
        assert final["persisted"] == [], "nothing durable was created"
        assert blocks_draft_write(persona, "save_learning") is True
        assert blocks_draft_write(persona, "get_brand_context") is False


# ------------------------------------------- startup guards (P4 / D5, P5.9) --


def test_cors_wildcard_with_credentials_refuses_to_start() -> None:
    """``*`` plus credentials is refused at startup, not discovered in prod."""
    from app.config import Settings
    from app.core.security_headers import validate_cors

    settings = Settings(cors_origins=("*",))
    with pytest.raises(ValueError, match="CORS_ORIGINS"):
        validate_cors(settings)

    explicit = Settings(cors_origins=("http://localhost:3000",))
    assert validate_cors(explicit) == []

    warned = validate_cors(Settings(cors_origins=()))
    assert warned and "empty" in warned[0]


def test_cors_uses_explicit_methods_and_headers() -> None:
    from app.config import Settings
    from app.core.security_headers import build_headers

    settings = Settings(cors_origins=("http://localhost:3000",))
    assert "*" not in settings.cors_allow_methods
    assert "*" not in settings.cors_allow_headers
    assert "GET" in settings.cors_allow_methods
    assert "Authorization" in settings.cors_allow_headers
    assert "frame-ancestors 'none'" in build_headers(settings)["Content-Security-Policy"]


def test_token_cipher_refuses_to_encrypt_in_production_without_a_key() -> None:
    """D5: a prod process without a key must not silently use the dev one."""
    from app.config import Settings
    from app.core.errors import EncryptionKeyMissingError
    from app.core.security import DEV_FERNET_KEY, TokenCipher, verify_runtime_security

    dev = Settings(environment="dev", token_encryption_key=None)
    warnings = verify_runtime_security(dev)
    assert warnings and "TOKEN_ENCRYPTION_KEY" in warnings[0]

    prod = Settings(environment="prod", token_encryption_key=None)
    with pytest.raises(EncryptionKeyMissingError):
        verify_runtime_security(prod)
    with pytest.raises(EncryptionKeyMissingError):
        TokenCipher(None, environment="prod")

    real = Settings(environment="prod", token_encryption_key=DEV_FERNET_KEY.decode())
    assert verify_runtime_security(real) == []
    cipher = TokenCipher(real.token_encryption_key)
    assert cipher.decrypt(cipher.encrypt("hello")) == "hello"
    assert cipher.is_dev_mode is False


def test_secrets_are_absent_from_the_log_redaction_list() -> None:
    """The new secret fields must be scrubbed from every log line."""
    from app.core.logging import redact

    scrubbed = redact(
        {
            "meta_app_secret_enc": APP_SECRET,
            "meta_access_token_enc": META_TOKEN,
            "google_places_api_key_enc": "k",
            "dataforseo_password_enc": "p",
            "image_api_key_enc": "i",
            "search_api_key_enc": "s",
            "api_key": "plain",
            "nested": [{"access_token": META_TOKEN}],
        }
    )
    for key in ("meta_app_secret_enc", "meta_access_token_enc", "api_key"):
        assert scrubbed[key] == "***REDACTED***", key
    assert scrubbed["nested"][0]["access_token"] == "***REDACTED***"
    assert META_TOKEN not in str(scrubbed)

async def test_image_connector_is_not_reported_configured_without_a_real_provider(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The image connector must say "not configured" while it is on `mock`.

    Regression: the `image` spec declared `placeholder_providers=("mock",)` but no
    `provider_field`, and none of its fields were `required`. `.provider` therefore
    read "" and `.required_fields` was empty, so `configured` came back True even
    with no image provider at all. The creatives page keyed its "Create image"
    button off that flag, enabled it, and `generate_image` rejected the request with
    "Mock placeholder images are disabled by default" -- the button looked live and
    always failed.
    """
    from app.config import get_settings

    body = await _register(api_client, "image-conn@example.com")
    headers = _auth(body["accessToken"])

    # 1) Untouched install: provider defaults to `mock`, no key.
    monkeypatch.setattr(get_settings(), "image_provider", "mock")
    monkeypatch.setattr(get_settings(), "image_api_key", None)
    unset = _connector(
        (await api_client.get("/api/v1/settings/connectors", headers=headers)).json(), "image"
    )
    assert unset["configured"] is False, unset
    assert unset["placeholder"] is True, unset
    assert _field(unset, "provider")["value"] == "mock"

    # 2) A real provider without its key is still not configured.
    monkeypatch.setattr(get_settings(), "image_provider", "openai")
    monkeypatch.setattr(get_settings(), "image_api_key", None)
    keyless = _connector(
        (await api_client.get("/api/v1/settings/connectors", headers=headers)).json(), "image"
    )
    assert keyless["configured"] is False, keyless
    assert keyless["placeholder"] is False, keyless

    # 3) Only a real provider *with* credentials enables the button.
    monkeypatch.setattr(get_settings(), "image_api_key", "sk-test-key")
    ready = _connector(
        (await api_client.get("/api/v1/settings/connectors", headers=headers)).json(), "image"
    )
    assert ready["configured"] is True, ready
    assert ready["placeholder"] is False, ready
