"""AGENTS.md is the entry point for anyone (or any agent) changing this code.

A map that points at files which no longer exist is worse than no map: the reader
follows it, finds nothing, and concludes the codebase is stale. So the map is
tested like code.

What is checked
--------------
Every ``backticked`` token in AGENTS.md (and the two root guides it chains to,
README.md and CONTRIBUTING.md) that *looks like a repository path* must resolve
to a real file or directory.

What is deliberately skipped
----------------------------
* globs (``test_creative_*.py``), templates (``<agent>/graph.py``), and anything
  containing ``*``, ``<`` or ``{}`` -- those are patterns, not paths;
* bare module names with no slash (``brain.py``, ``store.py``): they are used as
  shorthand in the task router where the directory is given in the same row;
* URLs, ``URL``, ``extra={...}`` and other code fragments.

The test also asserts that it actually checked a meaningful number of paths, so
a refactor that accidentally stops parsing the document fails instead of passing
vacuously.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Documents that must stay navigable. AGENTS.md is the map; the other two are
#: the doors into it and are checked for the same reason.
GUIDES = ("AGENTS.md", "README.md", "CONTRIBUTING.md")

_PATH_TOKEN = re.compile(r"`([^`\n]+)`")
_SUFFIXES = (".py", ".md", ".toml", ".yml", ".yaml", ".json", ".sql", ".cfg", ".ini")

#: A few real, load-bearing paths that must be *checked* even though they are
#: written without a directory prefix elsewhere. Keeps the parse honest.
_MIN_PATHS_CHECKED = 40


def _candidate_paths(document: str) -> set[str]:
    """Backticked tokens that are plausibly repository paths."""
    found: set[str] = set()
    for token in _PATH_TOKEN.findall(document):
        token = token.strip()
        if not token or any(ch in token for ch in "*<>{}|$"):
            continue
        if token.startswith(("http", "mailto:", "#", "app://")):
            continue
        if token in {"URL", "extra={...}"}:
            continue
        if "/" not in token:
            continue  # shorthand: the directory is given in the same table row
        if not token.endswith(_SUFFIXES) and not token.endswith("/"):
            continue
        found.add(token.rstrip("/"))
    return found


def _resolves(token: str) -> bool:
    """True when ``token`` names something real.

    The module map writes paths relative to ``app/`` (``core/memory/store.py``)
    because repeating ``app/`` in every row is noise; the task router writes them
    from the repository root (``app/core/memory/store.py``). Both must resolve,
    and so must a suffix such as ``layla/service.py`` under a nested agent
    package. A moved file stops matching any of these, which is the point.
    """
    if (REPO_ROOT / token).exists():
        return True
    if (REPO_ROOT / "app" / token).exists():
        return True
    suffix = "/" + token
    return any(
        path.as_posix().endswith(suffix)
        for path in (REPO_ROOT / "app").rglob(token.rsplit("/", 1)[-1])
    )


@pytest.mark.parametrize("guide", GUIDES)
def test_every_path_in_the_map_exists(guide: str) -> None:
    document = (REPO_ROOT / guide).read_text(encoding="utf-8")
    candidates = _candidate_paths(document)

    missing = sorted(token for token in candidates if not _resolves(token))

    assert candidates, f"{guide} mentions no repository paths at all -- the parser or the doc is broken"
    assert not missing, (
        f"{guide} points at paths that do not exist:\n  "
        + "\n  ".join(missing)
        + "\nUpdate the map in the same commit that moves the file."
    )


def test_the_map_is_substantial_enough_to_be_worth_checking() -> None:
    """Guards the guard: a stubbed AGENTS.md must not make this suite green."""
    document = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    checked = _candidate_paths(document)

    assert len(checked) >= _MIN_PATHS_CHECKED, (
        f"only {len(checked)} paths parsed from AGENTS.md; the task router and module "
        "map are supposed to reference the real tree"
    )


def test_the_map_covers_every_top_level_package() -> None:
    """A new top-level directory must be added to the module map, not just exist."""
    document = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    expected = {"app/", "tests/", "docs/", "mcp_servers/", "scripts/"}
    missing = {entry for entry in expected if entry not in document}

    assert not missing, f"AGENTS.md does not mention: {sorted(missing)}"


def test_agents_md_links_resolve() -> None:
    """The markdown links inside AGENTS.md must not rot."""
    document = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    broken = []
    for target in re.findall(r"\]\(([^)#][^)]*)\)", document):
        if target.startswith(("http", "mailto:", "#")):
            continue
        if not (REPO_ROOT / target.split("#")[0]).exists():
            broken.append(target)

    assert not broken, f"AGENTS.md has broken links: {broken}"
