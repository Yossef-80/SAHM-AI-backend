"""Regression tests for the two ``brand_profiles`` field families.

The bug these lock down
-----------------------
``PATCH /api/v1/business/profile`` normalised ``None`` -> ``[]`` for **every**
member of ``_PROFILE_FIELDS`` (``app/api/business.py``). That normalisation is
right for the three JSON list columns and wrong for the three ``Text`` columns,
and the difference is invisible until the driver binds the value:

* ``brand_voice`` / ``visual_style`` / ``target_audience`` are ``Text`` and
  nullable (``app/db/models.py:197-199``). Storing ``[]`` there binds a Python
  list into a text parameter and the driver raises
  ``sqlite3.InterfaceError: Error binding parameter 3 - type 'list' is not
  supported``. The client saw a 500 the moment a user cleared a text field.
* ``key_offers`` / ``goals`` / ``competitors`` are ``JSON`` with
  ``default=list`` (``app/db/models.py:200-202``) and are read unconditionally
  as ``list(profile.goals)`` (``app/api/business.py:185-186``). A NULL there is
  a bug waiting to happen, so ``null`` and ``[]`` mean the same thing.

These tests assert on the **stored row**, not just on the HTTP response, because
the response cannot tell ``[]`` apart from ``NULL`` for the list fields and the
whole bug lived below the serialisation layer.

The two families and their instructions:

===================  ==================  =========================
Sent                 ``Text`` column    ``JSON`` list column
===================  ==================  =========================
omitted (key absent) untouched           untouched
``null``             SQL ``NULL``       ``[]``
``[]``               rejected (422)     ``[]``
===================  ==================  =========================
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import BrandProfile

PASSWORD = "correct horse battery staple"

#: The three nullable ``Text`` columns -- ``null`` means SQL NULL.
TEXT_FIELDS = ("brandVoice", "visualStyle", "targetAudience")
#: The three non-nullable JSON list columns -- ``null`` means ``[]``.
LIST_FIELDS = ("keyOffers", "goals", "competitors")


# ------------------------------------------------------------------ helpers ----


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _register(client: Any, email: str) -> dict[str, Any]:
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": PASSWORD,
            "fullName": "Owner Person",
            "workspaceName": "Field Family Co",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _stored(db: AsyncSession, business_id: str) -> BrandProfile:
    """Read the profile row straight out of the database, bypassing identity map.

    ``expire_on_commit=False`` means the session still holds the instance the
    request mutated; asserting on that object would test the ORM, not the
    database, which is exactly where the bug lived.
    """
    db.expire_all()
    row = (
        await db.execute(
            select(BrandProfile).where(BrandProfile.business_id == business_id)
        )
    ).scalars().first()
    assert row is not None, "the PATCH must create the brand_profiles row"
    return row


async def _seeded_owner(client: Any, db: AsyncSession, email: str):
    """Register an owner and give every profile field a non-empty value."""
    body = await _register(client, email)
    business_id = body["workspaces"][0]["businessId"]
    resp = await client.patch(
        "/api/v1/business/profile",
        headers=_auth(body["accessToken"]),
        json={
            "brandVoice": "warm, direct, a bit cheeky",
            "visualStyle": "earthy tones, natural light",
            "targetAudience": "25-40 Cairo professionals",
            "keyOffers": ["subscription beans", "office catering"],
            "goals": ["grow subscriptions"],
            "competitors": ["Bean There"],
        },
    )
    assert resp.status_code == 200, resp.text
    return _auth(body["accessToken"]), business_id


# ------------------------------------------------- family 1: Text / null ----


@pytest.mark.parametrize("field", TEXT_FIELDS)
async def test_null_on_a_text_field_stores_sql_null(
    api_client, db_session: AsyncSession, field: str
) -> None:
    """The live bug: clearing a text field with ``null`` returned 500.

    Storing ``NULL`` is the only correct answer here -- the column is nullable
    ``Text`` and there is no meaningful "empty list" for a brand voice.
    """
    headers, business_id = await _seeded_owner(
        api_client, db_session, f"{field}-null@example.com"
    )

    resp = await api_client.patch(
        "/api/v1/business/profile", headers=headers, json={field: None}
    )

    assert resp.status_code == 200, (
        "clearing a text field with null must not 500: " + resp.text
    )
    assert resp.json()["business"][field] is None

    row = await _stored(db_session, business_id)
    column = {
        "brandVoice": "brand_voice",
        "visualStyle": "visual_style",
        "targetAudience": "target_audience",
    }[field]
    assert getattr(row, column) is None, (
        f"{column} must be SQL NULL, not an empty list"
    )


@pytest.mark.parametrize("field", LIST_FIELDS)
async def test_null_on_a_list_field_stores_empty_list(
    api_client, db_session: AsyncSession, field: str
) -> None:
    """``null`` on a list column means "clear it" and is stored as ``[]``.

    Never ``NULL``: the column is non-nullable JSON with a list default and the
    read path calls ``list(profile.goals)`` unconditionally.
    """
    headers, business_id = await _seeded_owner(
        api_client, db_session, f"{field}-null@example.com"
    )

    resp = await api_client.patch(
        "/api/v1/business/profile", headers=headers, json={field: None}
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["business"][field] == []

    row = await _stored(db_session, business_id)
    column = {"keyOffers": "key_offers", "goals": "goals", "competitors": "competitors"}[field]
    value = getattr(row, column)
    assert value is not None, f"{column} must never be stored as NULL"
    assert value == [], f"null on {column} must store [], got {value!r}"


@pytest.mark.parametrize("field", LIST_FIELDS)
async def test_empty_list_on_a_list_field_stores_empty_list(
    api_client, db_session: AsyncSession, field: str
) -> None:
    """``[]`` and ``null`` are the same instruction for a list column."""
    headers, business_id = await _seeded_owner(
        api_client, db_session, f"{field}-empty@example.com"
    )

    resp = await api_client.patch(
        "/api/v1/business/profile", headers=headers, json={field: []}
    )

    assert resp.status_code == 200, resp.text
    row = await _stored(db_session, business_id)
    column = {"keyOffers": "key_offers", "goals": "goals", "competitors": "competitors"}[field]
    assert getattr(row, column) == [], f"{column} must store []"


# ------------------------------------------------- family 3: omitted keys ----


async def test_an_omitted_key_leaves_every_column_untouched(
    api_client, db_session: AsyncSession
) -> None:
    """Absent key != ``null``. Sending one field must not wipe the other five.

    This is the distinction ``model_fields_set`` exists for (``app/schemas/
    business.py::BusinessProfileUpdateRequest.provided``); a plain ``None``
    default would make "not sent" and "sent null" indistinguishable.
    """
    headers, business_id = await _seeded_owner(
        api_client, db_session, "omitted@example.com"
    )

    resp = await api_client.patch(
        "/api/v1/business/profile",
        headers=headers,
        json={"brandVoice": "only this one changed"},
    )
    assert resp.status_code == 200, resp.text

    profile = resp.json()["business"]
    assert profile["brandVoice"] == "only this one changed"
    assert profile["visualStyle"] == "earthy tones, natural light"
    assert profile["targetAudience"] == "25-40 Cairo professionals"
    assert profile["keyOffers"] == ["subscription beans", "office catering"]
    assert profile["goals"] == ["grow subscriptions"]
    assert profile["competitors"] == ["Bean There"]

    row = await _stored(db_session, business_id)
    assert row.brand_voice == "only this one changed"
    assert row.visual_style == "earthy tones, natural light"
    assert row.target_audience == "25-40 Cairo professionals"
    assert row.key_offers == ["subscription beans", "office catering"]
    assert row.goals == ["grow subscriptions"]
    assert row.competitors == ["Bean There"]


# ------------------------------------- the two families in one request ----


async def test_both_families_cleared_in_a_single_request(
    api_client, db_session: AsyncSession
) -> None:
    """The real UI case: "Clear all brand fields" sends null for all six.

    Before the fix this single request was a 500 as soon as it reached the
    first ``Text`` column.
    """
    headers, business_id = await _seeded_owner(
        api_client, db_session, "clearall@example.com"
    )

    resp = await api_client.patch(
        "/api/v1/business/profile",
        headers=headers,
        json={
            "brandVoice": None,
            "visualStyle": None,
            "targetAudience": None,
            "keyOffers": None,
            "goals": None,
            "competitors": None,
        },
    )
    assert resp.status_code == 200, resp.text

    profile = resp.json()["business"]
    for field in TEXT_FIELDS:
        assert profile[field] is None, f"{field} must read back as null"
    for field in LIST_FIELDS:
        assert profile[field] == [], f"{field} must read back as an empty list"

    row = await _stored(db_session, business_id)
    assert row.brand_voice is None and row.visual_style is None
    assert row.target_audience is None
    assert row.key_offers == [] and row.goals == [] and row.competitors == []
