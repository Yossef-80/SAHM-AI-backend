"""Schemas for ``/api/v1/settings/connectors`` (P2.4).

The masking rule is structural, not conventional
-----------------------------------------------
A secret is declared as ``SecretStr`` here. That is not decoration: pydantic
renders ``SecretStr`` as ``**********`` in ``repr``/``str``, so an accidental
``logger.info("%s", payload)`` or a traceback frame cannot leak the value. The
API layer additionally never puts a secret into a response body -- it emits
``hasValue`` and a mask -- so a "grep the response for the key" test is a real
guarantee rather than a promise.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, SecretStr

from app.schemas.common import ApiModel

#: Same cap as ``FieldSpec.max_length`` defaults; enforced again per field by the
#: connector spec so a new connector cannot quietly widen it.
MAX_VALUE = 500
#: Access tokens are long. Capped, but generously.
MAX_TOKEN = 4_000


class ConnectorFieldOut(ApiModel):
    """One resolved field of one connector.

    ``value`` is ``None`` for every secret, unconditionally. ``hasValue`` and
    ``masked`` are what a form renders.
    """

    key: str
    label: str
    secret: bool = False
    required: bool = False
    max_length: int = MAX_VALUE
    description: str = ""
    #: Non-secret fields only.
    value: str | None = None
    has_value: bool = False
    #: ``"••••••••"`` when a secret exists, else ``None``. Never the secret.
    masked: str | None = None
    #: ``workspace`` | ``env`` | ``default`` | ``unset``.
    source: str = "unset"


class ConnectorOut(ApiModel):
    """One connector's effective configuration, with its provenance."""

    connector: str
    label: str
    #: Dominant layer for this connector: ``workspace`` if anything is
    #: overridden, else ``env``, else ``default``, else ``unset``.
    source: str = "unset"
    #: True when this workspace has its own row for this connector.
    has_override: bool = False
    #: True when every required field is present and a real provider is chosen.
    configured: bool = False
    provider: str | None = None
    #: True when the chosen provider returns placeholder data rather than live data.
    placeholder: bool = False
    fields: list[ConnectorFieldOut] = Field(default_factory=list)
    notes: str = ""


class ConnectorSettingsResponse(ApiModel):
    """``GET /settings/connectors`` body."""

    business_id: str
    environment: str
    #: Number of connectors with a stored workspace row.
    override_count: int = 0
    connectors: list[ConnectorOut] = Field(default_factory=list)
    #: True when no secret key is configured for this deployment (dev/test).
    encryption_key_configured: bool = True


class ConnectorUpdateRequest(ApiModel):
    """``PATCH /settings/connectors/{connector}`` body.

    One flat body for all five connectors rather than five request models: the
    connector is in the path, so the field keys are unambiguous inside it, and a
    single model keeps the OpenAPI document readable. The handler rejects a key
    that does not belong to the addressed connector with a 422 naming the
    accepted keys -- silently ignoring it would make a typo look like a save.

    ``null`` or ``""`` **clears** a stored value; omitting a key leaves the
    stored value alone.
    """

    # -- meta --
    app_id: str | None = Field(default=None, max_length=120)
    app_secret: SecretStr | None = None
    ad_account_id: str | None = Field(default=None, max_length=120)
    page_id: str | None = Field(default=None, max_length=120)
    access_token: SecretStr | None = None
    # -- google places / image (base URL override) --
    base_url: str | None = Field(default=None, max_length=MAX_VALUE)
    # -- google places / image / search --
    api_key: SecretStr | None = None
    # -- dataforseo / image / search --
    provider: str | None = Field(default=None, max_length=30)
    login: str | None = Field(default=None, max_length=200)
    password: SecretStr | None = None
    # -- image --
    model: str | None = Field(default=None, max_length=80)

    def sent(self) -> dict[str, Any]:
        """The keys the caller actually sent, secrets unwrapped.

        ``SecretStr`` is unwrapped here and nowhere else, so the encryption call
        in the handler is the single place a plaintext secret is used.
        """
        out: dict[str, Any] = {}
        for name in self.model_fields_set:
            value = getattr(self, name)
            if isinstance(value, SecretStr):
                value = value.get_secret_value()
            out[name] = value
        return out


class ConnectorDeleteResponse(ApiModel):
    """``DELETE /settings/connectors/{connector}`` body.

    200 rather than 204 because the caller needs to know what the connector fell
    back to, and that answer only exists after the delete.
    """

    connector: str
    removed: bool
    #: Where the connector now reads from.
    source: str = "env"
    configured: bool = False


__all__ = [
    "MAX_TOKEN",
    "MAX_VALUE",
    "ConnectorDeleteResponse",
    "ConnectorFieldOut",
    "ConnectorOut",
    "ConnectorSettingsResponse",
    "ConnectorUpdateRequest",
]