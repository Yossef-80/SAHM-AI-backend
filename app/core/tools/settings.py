"""P2.4/P2.5: per-workspace connector configuration.

Why this file exists
--------------------
Before P2 every provider credential was a process-wide env var, so two customers
could never use two different Meta ad accounts: whoever set ``META_APP_ID`` last
set it for everybody. This module makes the credential *data* -- stored per
(business, connector), encrypted at rest, editable by an owner over HTTP -- while
keeping the env var as the platform-level fallback so dev, CI and the existing
scripts keep working untouched.

Three layers, resolved in this order
------------------------------------
1. ``workspace`` -- the ``tool_settings`` row for this business.
2. ``env``       -- ``Settings``, i.e. whatever the process was started with.
3. ``default``   -- the built-in literal in :data:`CONNECTOR_SPECS`.

Every resolution reports which layer answered, so ``GET /settings/connectors``
can tell a UI what is *inherited* and what is *overridden* instead of showing
one merged value with no provenance.

What is deliberately NOT here
-----------------------------
* **No HTTP layer.** The router is ``app/api/settings.py``; this file only
  resolves. That keeps the tool handlers, which must not know about FastAPI,
  on exactly one code path to their configuration.
* **No plaintext secrets in the resolve result.** :class:`ConnectorSettings`
  deliberately carries plaintext *inside the process* -- a tool has to sign a
  real request with it -- but it is never serialised. The API layer renders
  ``hasValue`` / ``masked`` instead, and ``TokenCipher`` decryption happens here
  and nowhere else so there is one place to audit.

Used by
-------
``research.py``, ``meta_read.py``, ``media.py`` and ``protected.py`` call
:func:`resolve_tool_settings` (async, cached on the ``ToolContext``) instead of
reading ``ctx.require_settings()`` directly, so a workspace row wins over the env
var with no change to how the handlers are invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.security import TokenCipher
from app.db.models import ToolSettings as ToolSettingsRow

_logger = get_logger(__name__)

#: Where a resolved value came from. Reported verbatim to the UI.
#:
#: ``env`` means "the ``Settings`` object answered". ``Settings`` is itself
#: env-backed, so a field that has a *declared default* (``IMAGE_MODEL="mock-image-1"``)
#: also reports ``env`` -- the value really is coming from the process's
#: configuration, not from a literal in this file. ``default`` is reserved for
#: the built-in literals in :data:`CONNECTOR_SPECS`, which is the only layer that
#: cannot be changed without a code edit. The distinction matters when debugging
#: "why is this value what it is", so it is kept rather than collapsed.
SOURCE_WORKSPACE = "workspace"
SOURCE_ENV = "env"
SOURCE_DEFAULT = "default"
SOURCE_UNSET = "unset"

#: Never returned over HTTP in place of a secret.
SECRET_MASK = "••••••••"


# ------------------------------------------------------------------- the map --


@dataclass(frozen=True)
class FieldSpec:
    """One configurable field of one connector.

    ``db_column`` is nullable for a field that has no column (none today, but it
    keeps the map forward-compatible) and ``env`` is a tuple because one API key
    has three possible env vars: ``search_api_key`` resolves to whichever of
    ``TAVILY_API_KEY`` / ``SERPER_API_KEY`` / ``BRAVE_API_KEY`` the selected
    provider owns.
    """

    key: str
    label: str
    db_column: str | None
    env: tuple[str, ...] = ()
    secret: bool = False
    max_length: int = 500
    required: bool = False
    default: str | None = None
    description: str = ""
    #: Closed set of accepted values, published to the client so a form can
    #: render a select. Empty means free text. Enforced on write by the settings
    #: API, the same way ``allowed_providers`` is: a quality of "HGIH" must be a
    #: rejected request, not a saved value that the image call then ignores.
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConnectorSpec:
    """Everything the settings surface knows about one connector."""

    name: str
    label: str
    fields: tuple[FieldSpec, ...]
    #: Field key of the provider selector, when the connector has a choice.
    provider_field: str | None = None
    #: Providers that return placeholder data, so their key is never required.
    placeholder_providers: tuple[str, ...] = ("mock",)
    #: Accepted values of ``provider_field``, in the order a UI should list them.
    #: Empty means "this connector has no provider selector". Enforced on write by
    #: ``app/api/settings.py`` and published to the client as ``ConnectorOut.providers``
    #: so the settings screen can render a select instead of a free-text box --
    #: a typo in this field is otherwise indistinguishable from a saved setup.
    allowed_providers: tuple[str, ...] = ()
    notes: str = ""

    def field(self, key: str) -> FieldSpec | None:
        for spec in self.fields:
            if spec.key == key:
                return spec
        return None

    @property
    def required_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(spec for spec in self.fields if spec.required)


#: The five connectors a workspace can configure. This extends -- deliberately,
#: rather than replaces -- ``app/core/tools/connectors.py``, which stays the
#: authority on *which tool talks to which URL*. This map is the authority on
#: *which credentials the user can fill in*.
CONNECTOR_SPECS: dict[str, ConnectorSpec] = {
    "meta": ConnectorSpec(
        name="meta",
        label="Meta Marketing API",
        fields=(
            FieldSpec(
                key="app_id",
                label="App ID",
                db_column="meta_app_id",
                env=("META_APP_ID",),
                max_length=120,
                description="Your Meta app's numeric id.",
            ),
            FieldSpec(
                key="app_secret",
                label="App secret",
                db_column="meta_app_secret_enc",
                env=("META_APP_SECRET",),
                secret=True,
                max_length=200,
                description="Write-only. Never returned by any endpoint.",
            ),
            FieldSpec(
                key="ad_account_id",
                label="Ad account ID",
                db_column="meta_ad_account_id",
                env=("META_AD_ACCOUNT_ID",),
                max_length=120,
                required=True,
                description="The act_<id> this workspace manages.",
            ),
            FieldSpec(
                key="page_id",
                label="Facebook Page ID",
                db_column="meta_page_id",
                env=("META_PAGE_ID",),
                max_length=120,
                required=True,
                description="The Page that will publish the approved ads for this workspace.",
            ),
            FieldSpec(
                key="access_token",
                label="Access token",
                db_column="meta_access_token_enc",
                env=("META_ACCESS_TOKEN",),
                secret=True,
                max_length=1000,
                required=True,
                description="Write-only. Required for every Meta read.",
            ),
            FieldSpec(
                key="base_url",
                label="Graph base URL",
                db_column="base_url",
                env=("META_GRAPH_BASE_URL",),
                max_length=500,
                description="Repoint at a sandbox or a proxy.",
            ),
        ),
        notes=(
            "Admin-managed only. The token is used for Meta reads and approved ad-plan writes; "
            "the ad account ID and Facebook Page ID are required for campaign creation. "
            "No self-service OAuth flow is provided here."
        ),
    ),
    "google_places": ConnectorSpec(
        name="google_places",
        label="Google Places",
        fields=(
            FieldSpec(
                key="api_key",
                label="API key",
                db_column="google_places_api_key_enc",
                env=("GOOGLE_PLACES_API_KEY",),
                secret=True,
                max_length=200,
                required=True,
                description="Write-only. Required for real reviews.",
            ),
            FieldSpec(
                key="base_url",
                label="Places base URL",
                db_column="google_places_base_url",
                env=("PLACES_BASE_URL",),
                max_length=500,
            ),
        ),
        notes="Used by get_reviews. Without a key the tool returns labelled mock data.",
    ),
    "dataforseo": ConnectorSpec(
        name="dataforseo",
        label="DataForSEO (keywords)",
        provider_field="provider",
        fields=(
            FieldSpec(
                key="provider",
                label="Provider",
                db_column="keyword_provider",
                env=("KEYWORD_PROVIDER",),
                max_length=30,
                default="mock",
            ),
            FieldSpec(
                key="login",
                label="Login",
                db_column="dataforseo_login",
                env=("DATAFORSEO_LOGIN",),
                max_length=200,
                required=True,
            ),
            FieldSpec(
                key="password",
                label="Password",
                db_column="dataforseo_password_enc",
                env=("DATAFORSEO_PASSWORD",),
                secret=True,
                max_length=200,
                required=True,
                description="Write-only. Used as HTTP basic auth.",
            ),
        ),
        placeholder_providers=("mock",),
        allowed_providers=("mock", "dataforseo"),
        notes="Used by get_keywords.",
    ),
    "image": ConnectorSpec(
        name="image",
        label="Image generation",
        # Without provider_field the connector could not tell `mock` from a real
        # provider: `.provider` read "" and `.required_fields` was empty, so
        # `configured` came back True for a workspace that had no image provider
        # at all. The UI then enabled "Create image" and generate_image rejected
        # it with "Mock placeholder images are disabled by default".
        provider_field="provider",
        fields=(
            FieldSpec(
                key="provider",
                label="Provider",
                db_column="image_provider",
                env=("IMAGE_PROVIDER",),
                max_length=30,
                default="mock",
            ),
            FieldSpec(
                key="api_key",
                label="API key",
                db_column="image_api_key_enc",
                env=("IMAGE_API_KEY",),
                secret=True,
                required=True,
                max_length=200,
                description="Required unless the provider is 'mock'.",
            ),
            FieldSpec(
                key="model",
                label="Model",
                db_column="image_model",
                env=("IMAGE_MODEL",),
                max_length=80,
                default="mock-image-1",
            ),
            FieldSpec(
                key="base_url",
                label="Image base URL",
                db_column="base_url",
                env=("IMAGE_BASE_URL",),
                max_length=500,
            ),
            FieldSpec(
                key="quality",
                label="Quality",
                db_column="image_quality",
                env=("IMAGE_QUALITY",),
                max_length=16,
                # `auto` lets the provider decide, which is the honest default:
                # GPT Image's own default is `high`, the expensive one, and
                # nobody should pay for it by not knowing the setting exists.
                default="auto",
                description="How much effort the image model spends. GPT Image models only.",
                choices=("auto", "low", "medium", "high"),
            ),
        ),
        # `mock` is a placeholder: it satisfies no real generation, so a key set
        # while it is selected is dead weight (see `_validate_provider_selection`).
        allowed_providers=("mock", "openai", "stability"),
        notes="Used by generate_image.",
    ),
    "search": ConnectorSpec(
        name="search",
        label="Web search",
        provider_field="provider",
        fields=(
            FieldSpec(
                key="provider",
                label="Provider",
                db_column="search_provider",
                env=("SEARCH_PROVIDER",),
                max_length=30,
                default="mock",
            ),
            FieldSpec(
                key="api_key",
                label="API key",
                db_column="search_api_key_enc",
                env=("TAVILY_API_KEY", "SERPER_API_KEY", "BRAVE_API_KEY"),
                secret=True,
                max_length=200,
                description=(
                    "Write-only. Whichever of TAVILY/SERPER/BRAVE the chosen "
                    "provider uses."
                ),
            ),
        ),
        allowed_providers=("mock", "tavily", "serper", "brave"),
        notes="Used by web_search.",
    ),
}

#: Connectors that may have a stored row, in the order the UI should show them.
CONNECTOR_ORDER: tuple[str, ...] = tuple(CONNECTOR_SPECS)


def connector_spec(connector: str) -> ConnectorSpec | None:
    """The spec for ``connector``, or None when the name is unknown.

    Unknown names are a 404 at the API boundary, never a 422: the client asked
    for a connector that does not exist in this build.
    """
    return CONNECTOR_SPECS.get(connector)


# ---------------------------------------------------------------- resolution --


def _settings_value(settings: Settings, env_name: str) -> str | None:
    """Read one env-named setting off ``Settings``.

    ``SEARCH_PROVIDER`` is the spelling a user edits in ``.env``; the field is
    ``search_provider``. pydantic-settings is case-insensitive on input, so the
    lowercase form always exists.
    """
    value = getattr(settings, env_name.lower(), None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class ResolvedField:
    """One field after the three layers were merged."""

    spec: FieldSpec
    value: str | None
    source: str

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def has_value(self) -> bool:
        return bool(self.value)


@dataclass(frozen=True)
class ResolvedConnector:
    """One connector after the three layers were merged."""

    spec: ConnectorSpec
    fields: tuple[ResolvedField, ...]

    def field(self, key: str) -> ResolvedField | None:
        for item in self.fields:
            if item.key == key:
                return item
        return None

    @property
    def value_of(self) -> dict[str, str | None]:
        return {item.key: item.value for item in self.fields}

    @property
    def provider(self) -> str:
        """The selected provider, or "" when this connector has no selector."""
        if self.spec.provider_field is None:
            return ""
        found = self.field(self.spec.provider_field)
        return (found.value if found else None) or ""

    @property
    def is_placeholder(self) -> bool:
        return self.provider in self.spec.placeholder_providers

    @property
    def missing_required(self) -> tuple[str, ...]:
        """Keys of required fields that have no value at all.

        Used by the settings API to explain *why* a connector is not
        ``configured`` instead of only reporting the boolean.
        """
        missing: list[str] = []
        for spec in self.spec.required_fields:
            resolved = self.field(spec.key)
            if resolved is None or not resolved.has_value:
                missing.append(spec.key)
        return tuple(missing)

    @property
    def configured(self) -> bool:
        """True when this connector can return *real* data.

        A placeholder provider (``mock``) is deliberately **not** configured:
        the UI must be able to say "this is demo data" rather than "connected",
        and its keys are genuinely irrelevant. With a real provider selected,
        every field the connector needs must be present.
        """
        if self.spec.provider_field and not self.provider:
            return False
        if self.is_placeholder:
            return False
        return not self.missing_required

    @property
    def source(self) -> str:
        """The dominant layer: workspace if anything was overridden, else env."""
        sources = {item.source for item in self.fields}
        if SOURCE_WORKSPACE in sources:
            return SOURCE_WORKSPACE
        if SOURCE_ENV in sources:
            return SOURCE_ENV
        if SOURCE_DEFAULT in sources:
            return SOURCE_DEFAULT
        return SOURCE_UNSET


def resolve_connector(
    spec: ConnectorSpec,
    row: ToolSettingsRow | None,
    settings: Settings,
    *,
    decrypt: TokenCipher | None = None,
) -> ResolvedConnector:
    """Merge the workspace row, the process env and the built-in default.

    ``decrypt`` is passed in (rather than built here) so the cipher -- and
    therefore the Fernet key -- is constructed once per request by the caller
    that already holds ``Settings``.
    """
    cipher = decrypt or TokenCipher(settings.token_encryption_key)
    resolved: list[ResolvedField] = []
    for field_spec in spec.fields:
        value: str | None = None
        source = SOURCE_UNSET

        stored = (
            getattr(row, field_spec.db_column, None)
            if row is not None and field_spec.db_column
            else None
        )
        if field_spec.secret:
            if stored:
                try:
                    value = cipher.decrypt(str(stored)) or None
                except Exception as exc:  # pragma: no cover - rotated key
                    _logger.warning(
                        "tool_setting_secret_undecryptable",
                        extra={"connector": spec.name, "field": field_spec.key},
                    )
                    _logger.debug("decrypt error detail", extra={"error": str(exc)})
                    value = None
                if value:
                    source = SOURCE_WORKSPACE
        elif stored is not None and str(stored).strip():
            value = str(stored).strip()
            source = SOURCE_WORKSPACE

        if value is None:
            for env_name in field_spec.env:
                candidate = _settings_value(settings, env_name)
                if candidate:
                    value = candidate
                    source = SOURCE_ENV
                    break

        if value is None and field_spec.default is not None:
            value = field_spec.default
            source = SOURCE_DEFAULT

        resolved.append(ResolvedField(spec=field_spec, value=value, source=source))
    return ResolvedConnector(spec=spec, fields=tuple(resolved))


@dataclass(frozen=True)
class ConnectorSettings:
    """The resolved configuration for one business, ready for tool handlers.

    Built once per request and cached on the ``ToolContext``. The plain
    attributes are what ``research.py`` / ``meta_read.py`` / ``media.py`` read,
    which is why they are spelled exactly like the ``Settings`` fields they
    shadow: the handlers changed one call, not twenty.
    """

    business_id: str
    connectors: dict[str, ResolvedConnector] = field(default_factory=dict)

    # -- lookups --
    def connector(self, name: str) -> ResolvedConnector | None:
        return self.connectors.get(name)

    def value(self, connector: str, key: str) -> str | None:
        """Effective value of one field, or None."""
        resolved = self.connectors.get(connector)
        if resolved is None:
            return None
        found = resolved.field(key)
        return found.value if found else None

    def source(self, connector: str, key: str) -> str:
        """Which layer answered for one field."""
        resolved = self.connectors.get(connector)
        if resolved is None:
            return SOURCE_UNSET
        found = resolved.field(key)
        return found.source if found else SOURCE_UNSET

    # -- Meta --
    @property
    def meta_access_token(self) -> str | None:
        return self.value("meta", "access_token")

    @property
    def meta_app_id(self) -> str | None:
        return self.value("meta", "app_id")

    @property
    def meta_app_secret(self) -> str | None:
        return self.value("meta", "app_secret")

    @property
    def meta_ad_account_id(self) -> str | None:
        return self.value("meta", "ad_account_id")

    @property
    def meta_page_id(self) -> str | None:
        return self.value("meta", "page_id")

    @property
    def meta_graph_base_url(self) -> str | None:
        return self.value("meta", "base_url")

    # -- Google Places --
    @property
    def google_places_api_key(self) -> str | None:
        return self.value("google_places", "api_key")

    @property
    def places_base_url(self) -> str | None:
        return self.value("google_places", "base_url")

    # -- DataForSEO --
    @property
    def keyword_provider(self) -> str | None:
        return self.value("dataforseo", "provider")

    @property
    def dataforseo_login(self) -> str | None:
        return self.value("dataforseo", "login")

    @property
    def dataforseo_password(self) -> str | None:
        return self.value("dataforseo", "password")

    # -- Images --
    @property
    def image_provider(self) -> str | None:
        return self.value("image", "provider")

    @property
    def image_api_key(self) -> str | None:
        return self.value("image", "api_key")

    @property
    def image_model(self) -> str | None:
        return self.value("image", "model")

    @property
    def image_quality(self) -> str | None:
        return self.value("image", "quality")

    @property
    def image_base_url(self) -> str | None:
        return self.value("image", "base_url")

    # -- Search --
    @property
    def search_provider(self) -> str | None:
        return self.value("search", "provider")

    @property
    def search_api_key(self) -> str | None:
        return self.value("search", "api_key")


EMPTY_SETTINGS = ConnectorSettings(business_id="")


async def load_connector_settings(
    db: AsyncSession,
    business_id: str,
    settings: Settings | None = None,
) -> ConnectorSettings:
    """Read every stored connector row for one business and resolve it.

    One query for all five connectors rather than one per connector: a tool call
    that needs three of them would otherwise pay three round trips.
    """
    cfg = settings or get_settings()
    stmt = select(ToolSettingsRow).where(ToolSettingsRow.business_id == business_id)
    rows = (await db.execute(stmt)).scalars().all()
    by_name = {row.connector: row for row in rows}

    cipher = TokenCipher(cfg.token_encryption_key)
    resolved = {
        name: resolve_connector(spec, by_name.get(name), cfg, decrypt=cipher)
        for name, spec in CONNECTOR_SPECS.items()
    }
    return ConnectorSettings(business_id=business_id, connectors=resolved)


async def resolve_tool_settings(ctx: Any) -> ConnectorSettings:
    """Resolve (and cache on the context) the connector config for a tool call.

    The seam every tool handler goes through. Caching on ``ToolContext`` matters:
    an agent run makes many tool calls in one request, and decrypting six Fernet
    values per call would be pure waste.
    """
    cached = getattr(ctx, "tool_settings", None)
    if isinstance(cached, ConnectorSettings):
        return cached
    resolved = await load_connector_settings(
        ctx.db, ctx.business_id, ctx.settings or get_settings()
    )
    ctx.tool_settings = resolved
    return resolved


__all__ = [
    "CONNECTOR_ORDER",
    "CONNECTOR_SPECS",
    "EMPTY_SETTINGS",
    "SECRET_MASK",
    "SOURCE_DEFAULT",
    "SOURCE_ENV",
    "SOURCE_UNSET",
    "SOURCE_WORKSPACE",
    "ConnectorSettings",
    "ConnectorSpec",
    "FieldSpec",
    "ResolvedConnector",
    "ResolvedField",
    "connector_spec",
    "load_connector_settings",
    "resolve_connector",
    "resolve_tool_settings",
]