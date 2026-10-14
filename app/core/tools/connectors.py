"""One map of every tool to the external service it talks to.

Why this file exists
--------------------
Before this, the destination of each tool was buried inside the handler modules
(``research.py``, ``meta_read.py``, ``media.py``). Asking "which URL does this
tool hit, and can I change it?" meant opening five files and reading the string
concatenations. This is the single place that answers it.

It is data only -- no imports from the tool modules, no behaviour. The
:func:`connector_for` helper is what ``GET /api/v1/tools`` reads so the HTTP
catalogue reports the same thing this file says.

``url_template`` may contain these placeholders, resolved by
:func:`resolve_url`:

===========================  ====================================================
``{base}``                   ``Settings.meta_graph_base_url``
``{version}``                ``Settings.meta_graph_version``
``{places}``                 ``Settings.places_base_url``
``{llm}``                    ``Settings.llm_base_url`` or the OpenAI default
===========================  ====================================================

``configurable_via`` names the env var when a URL can be changed without a code
edit; ``None`` means the URL is hardcoded in its handler and changing it requires
editing that module.

P2.4 extended this file rather than adding a parallel map: the URL/provider map
above stays the single place that knows where a tool points, and the
per-workspace *credentials* map lives next door in
:mod:`app.core.tools.settings`. When a resolved ``ConnectorSettings`` is passed to
:func:`connector_for`, ``missing_env`` becomes "required fields still missing
after the workspace row was merged", so ``GET /api/v1/tools/connectors`` reports
the same truth as ``GET /api/v1/settings/connectors``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import Settings

#: The OpenAI default used when ``LLM_BASE_URL`` is unset.
DEFAULT_LLM_URL = "https://api.openai.com/v1/chat/completions"

#: Hardcoded hostnames that are not currently exposed through Settings.
#: Kept here so the mapping below can say "hardcoded" without a second lookup.
HARDCODED = {
    "tavily": "https://api.tavily.com/search",
    "serper": "https://google.serper.dev/search",
    "brave": "https://api.search.brave.com/res/v1/web/search",
    "dataforseo": (
        "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live"
    ),
    "openai_images": "https://api.openai.com/v1/images/generations",
}


@dataclass(frozen=True)
class Connector:
    """What one tool talks to, and whether you can repoint it."""

    #: ``none`` | ``web_search`` | ``website`` | ``meta_ads`` | ``meta_read``
    #: | ``google_places`` | ``keywords`` | ``images`` | ``llm`` | ``database``
    provider: str
    #: URL or URL template. ``None`` for tools that never leave the process.
    url_template: str | None = None
    #: The env var that repoints it, or ``None`` when hardcoded.
    configurable_via: str | None = None
    #: Extra env vars this connector needs before it returns real data.
    required_env: tuple[str, ...] = ()
    #: Env var that selects which provider is actually used, when there is a choice.
    selected_by: str | None = None
    #: Alternative providers, each with its URL template.
    alternatives: dict[str, str] = field(default_factory=dict)
    #: When True the live URL is ``alternatives[<selected_by>]``, not ``url_template``.
    selected_url: bool = False
    notes: str = ""

    def resolve(self, settings: Settings, tool_settings: Any = None) -> dict[str, Any]:
        """Concrete JSON-safe view: real URL, config key, and what is missing.

        ``tool_settings`` is an optional ``app.core.tools.settings.ConnectorSettings``.
        When supplied, credential checks consult the merged workspace+env+
        default value instead of the env var alone, which is what makes a
        workspace row actually visible in the tool catalogue.
        """
        url = None
        if self.selected_url and self.selected_by:
            chosen = self._selected_provider(settings, tool_settings)
            template = self.alternatives.get(str(chosen), "")
            url = resolve_url(template, settings) if template else None
        elif self.url_template:
            url = resolve_url(self.url_template, settings)
        missing = self._missing(settings, tool_settings)
        return {
            "provider": self.provider,
            "url": url,
            "url_template": self.url_template,
            "configurable_via": self.configurable_via,
            "selected_by": self.selected_by,
            "required_env": list(self.required_env),
            "missing_env": missing,
            "configured": bool(url) and not missing,
            "alternatives": {
                name: resolve_url(tpl, settings)
                for name, tpl in self.alternatives.items()
            },
            "notes": self.notes,
        }

    def _selected_provider(self, settings: Settings, tool_settings: Any) -> Any:
        """The chosen provider, preferring a workspace row over the env var."""
        connector_name, field_key = _PROVIDER_SELECTION.get(
            self.selected_by or "", (None, None)
        )
        if tool_settings is not None and connector_name and field_key:
            chosen = tool_settings.value(connector_name, field_key)
            if chosen:
                return chosen
        return _setting(settings, self.selected_by or "")

    def _missing(self, settings: Settings, tool_settings: Any) -> list[str]:
        """Required credentials that are still absent after the merge.

        The names are still *env var* names, because that is what the rest of
        the catalogue and the operator's runbook speak; the difference is that a
        name is no longer reported missing when the workspace supplied it.
        """
        missing: list[str] = []
        for key in self.required_env:
            connector_name, field_key = _CREDENTIAL_LOOKUP.get(key.upper(), (None, None))
            value = (
                tool_settings.value(connector_name, field_key)
                if tool_settings is not None and connector_name and field_key
                else None
            )
            # ConnectorSettings already merges env, but a key that is not in the
            # lookup (or a caller that passed no tool_settings) still has to be
            # checked against Settings, or every connector would report missing.
            if not value:
                value = _setting(settings, key)
            if not value:
                missing.append(key)
        return missing


#: Required env var -> (connector, field) in ``app.core.tools.settings``. Used by
#: ``Connector._missing`` to honour a workspace row without a second map.
_CREDENTIAL_LOOKUP: dict[str, tuple[str, str]] = {
    "META_ACCESS_TOKEN": ("meta", "access_token"),
    "META_AD_ACCOUNT_ID": ("meta", "ad_account_id"),
    "GOOGLE_PLACES_API_KEY": ("google_places", "api_key"),
    "DATAFORSEO_LOGIN": ("dataforseo", "login"),
    "DATAFORSEO_PASSWORD": ("dataforseo", "password"),
}

#: Provider selector env var -> (connector, field). Same purpose for the two
#: connectors whose URL is chosen by a provider name.
_PROVIDER_SELECTION: dict[str, tuple[str, str]] = {
    "SEARCH_PROVIDER": ("search", "provider"),
    "IMAGE_PROVIDER": ("image", "provider"),
    "KEYWORD_PROVIDER": ("dataforseo", "provider"),
}


def _setting(settings: Settings, key: str) -> Any:
    """Read a Settings field by name, accepting the ENV VAR spelling.

    ``selected_by`` is documented with env var names (``SEARCH_PROVIDER``) because
    that is what a user edits, while the field is ``search_provider``. pydantic
    settings are case-insensitive on input, so the lowercase form always exists.
    """
    if hasattr(settings, key):
        return getattr(settings, key)
    return getattr(settings, key.lower(), None)


def resolve_url(template: str, settings: Settings) -> str:
    """Expand ``{base}`` / ``{version}`` / ``{places}`` / ``{llm}`` placeholders."""
    return (
        template.replace("{base}", settings.meta_graph_base_url.rstrip("/"))
        .replace("{version}", settings.meta_graph_version)
        .replace("{places}", settings.places_base_url.rstrip("/"))
        .replace("{llm}", settings.llm_base_url or DEFAULT_LLM_URL)
    )


#: Tool name -> the service it reaches. ``ALL_TOOLS`` (17) plus the two protected
#: tools, which no agent may call but which still talk to Meta.
CONNECTORS: dict[str, Connector] = {
    # ---------------------------------------------------------- research ------
    "web_search": Connector(
        provider="web_search",
        url_template="{web_search}",
        configurable_via=None,
        required_env=(),
        selected_by="SEARCH_PROVIDER",
        selected_url=True,
        alternatives={
            "mock": "mock://labelled-demo-results",
            "tavily": HARDCODED["tavily"],
            "serper": HARDCODED["serper"],
            "brave": HARDCODED["brave"],
        },
        notes=(
            "URL is chosen by SEARCH_PROVIDER but hardcoded per provider in "
            "app/core/tools/research.py. A provider with a missing key falls "
            "through to 'unavailable', never to mock."
        ),
    ),
    "analyze_website": Connector(
        provider="website",
        url_template=None,
        notes=(
            "Fetches a URL supplied by the caller. SSRF-guarded by "
            "app/core/security.py; there is no fixed host."
        ),
    ),
    "brand_report": Connector(
        provider="composite_research",
        url_template=None,
        notes=(
            "Legacy brand_mcp wrapper that may consult several dynamic public sources "
            "and optional Apify actors; there is no single fixed endpoint. The evidence-first "
            "app.ai.tools.brand_research path is separately exercised with offline fixtures."
        ),
    ),
    "full_analysis": Connector(
        provider="local_analysis",
        url_template=None,
        notes=(
            "Runs analysis over previously collected brand_mcp data; optional NLP "
            "dependencies are local and there is no fixed endpoint."
        ),
    ),
    "route_file_to_tool": Connector(
        provider="local_file_router",
        url_template=None,
        notes=(
            "Legacy MCP file-routing wrapper. Only use uploaded/supplied files; do not "
            "pass arbitrary server paths. The app.ai file router accepts supplied content."
        ),
    ),
    "get_social_profile": Connector(
        provider="meta_ads",
        url_template="{base}/{version}/{platform}",
        configurable_via="META_GRAPH_BASE_URL",
        required_env=("meta_access_token",),
        notes="Reads META_ACCESS_TOKEN directly from the environment.",
    ),
    "get_reviews": Connector(
        provider="google_places",
        url_template="{places}/findplacefromtext/json",
        configurable_via="PLACES_BASE_URL",
        required_env=("google_places_api_key",),
        alternatives={"details": "{places}/details/json"},
        notes="Returns a deterministic mock in dev/test when the key is absent.",
    ),
    "get_keywords": Connector(
        provider="keywords",
        url_template=HARDCODED["dataforseo"],
        configurable_via=None,
        selected_by="KEYWORD_PROVIDER",
        required_env=("dataforseo_login", "dataforseo_password"),
        alternatives={"mock": "mock://deterministic-volumes"},
        notes="Hardcoded in app/core/tools/research.py; basic-auth credentials.",
    ),
    "get_competitor_ads": Connector(
        provider="meta_ads",
        url_template="{base}/{version}/ads_archive",
        configurable_via="META_GRAPH_BASE_URL",
        required_env=("meta_access_token",),
        notes="Only supports countries in the tool's allowlist; falls back to user-supplied ads.",
    ),
    # ------------------------------------------------------------ memory ------
    "get_brand_context": Connector(
        provider="database",
        url_template=None,
        notes="Local only. Reads brand_memory_items + brand_profiles via pgvector or in-Python cosine.",
    ),
    "save_learning": Connector(
        provider="database",
        url_template=None,
        notes="Local only. Writes one brand_memory_items row of kind 'learning'.",
    ),
    # -------------------------------------------------------------- meta ------
    "list_meta_campaigns": Connector(
        provider="meta_read",
        url_template="{base}/{version}/act_<connected_ad_account>/campaigns",
        configurable_via="META_GRAPH_BASE_URL",
        required_env=("meta_access_token",),
        notes=(
            "Read-only campaign discovery. The ad account ID and token resolve from the connected "
            "Meta Integration/workspace settings; the displayed URL uses a dynamic account placeholder."
        ),
    ),
    "get_campaign_insights": Connector(
        provider="meta_read",
        url_template="{base}/{version}/{meta_campaign_id}/insights",
        configurable_via="META_GRAPH_BASE_URL",
        required_env=("meta_access_token",),
        notes="Reads the encrypted Integration token, not the env var, for this path.",
    ),
    "get_breakdowns": Connector(
        provider="meta_read",
        url_template="{base}/{version}/{meta_campaign_id}/insights",
        configurable_via="META_GRAPH_BASE_URL",
        required_env=("meta_access_token",),
        notes="Same insights endpoint, with a breakdown= query parameter.",
    ),
    # ------------------------------------------------------------- media ------
    "analyze_reference_creative": Connector(
        provider="llm",
        url_template="{llm}",
        configurable_via="LLM_BASE_URL",
        notes="Fetches the supplied image over the network, then calls the strong-tier LLM for vision.",
    ),
    "generate_image": Connector(
        provider="images",
        url_template=HARDCODED["openai_images"],
        configurable_via=None,
        selected_by="IMAGE_PROVIDER",
        required_env=(),
        alternatives={"mock": "mock://svg-data-url"},
        notes=(
            "Hardcoded in app/core/tools/media.py. IMAGE_BASE_URL exists but is never "
            "read. IMAGE_PROVIDER=stability returns 'unavailable' -- not implemented."
        ),
    ),
    # ---------------------------------------------------------- records ------
    "create_recommendation": Connector(
        provider="database",
        url_template=None,
        notes="Local only. Creates a pending recommendations row awaiting human approval.",
    ),
    # ------------------------------------------- protected (never agent-call) --
    "apply_change": Connector(
        provider="meta_write",
        url_template="{base}/{version}/{object_id}",
        configurable_via="META_GRAPH_BASE_URL",
        required_env=("meta_access_token",),
        notes="PROTECTED. Reachable only via the two-phase approval flow.",
    ),
    "publish_campaign": Connector(
        provider="meta_write",
        url_template="{base}/{version}/{campaign_object_id}",
        configurable_via="META_GRAPH_BASE_URL",
        required_env=("meta_access_token",),
        notes="PROTECTED. Reachable only via the two-phase approval flow.",
    ),
}


def connector_for(
    tool_name: str, settings: Settings, tool_settings: Any = None
) -> dict[str, Any]:
    """Connector view for one tool. Unknown tools get an honest 'unknown' entry."""
    connector = CONNECTORS.get(tool_name)
    if connector is None:  # pragma: no cover - guarded by a test
        return {
            "provider": "unknown",
            "url": None,
            "url_template": None,
            "configurable_via": None,
            "selected_by": None,
            "required_env": [],
            "missing_env": [],
            "configured": False,
            "alternatives": {},
            "notes": "No connector is registered for this tool.",
        }
    return connector.resolve(settings, tool_settings)


def summary(settings: Settings, tool_settings: Any = None) -> dict[str, Any]:
    """Whole connector map, for ``GET /api/v1/connectors``."""
    return {
        name: connector_for(name, settings, tool_settings) for name in sorted(CONNECTORS)
    }


__all__ = [
    "CONNECTORS",
    "_CREDENTIAL_LOOKUP",
    "_PROVIDER_SELECTION",
    "DEFAULT_LLM_URL",
    "HARDCODED",
    "Connector",
    "connector_for",
    "resolve_url",
    "summary",
]