"""Check whether this machine can reach the Supabase database, and how.

Why this script exists
----------------------
Supabase shows you four values on the project page (project URL, publishable
key, secret key, JWKS URL). None of them is a database connection string: the
app talks to Postgres via SQLAlchemy/asyncpg, which needs the *database
password* from Settings -> Database.

It also hides two traps that cost real time:

1. The direct host ``db.<ref>.supabase.co`` is often **IPv6-only** (no A
   record). On an IPv4-only machine DNS simply fails, which looks like
   "Supabase is down" when it is not.
2. The pooler needs the **tenant username** ``postgres.<ref>``, not ``postgres``,
   otherwise Supavisor answers "no tenant identifier provided".

Verified for the project configured here: the pooler region is ``eu-west-1``.

Usage (from the ``sahm-ai backend`` folder)::

    python scripts\check_supabase.py
    python scripts\check_supabase.py --url https://yyyy.supabase.co

It never prints a key, only whether one was found and whether it worked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_URL = "https://fmvwhrduiwloqravbeqi.supabase.co"
KNOWN_REGION = "eu-west-1"
REST_TIMEOUT = 20
TCP_TIMEOUT = 12

REGIONS = [
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1", "eu-central-2",
    "eu-north-1", "ap-southeast-1", "ap-southeast-2", "ap-northeast-1",
    "ap-northeast-2", "ap-south-1", "sa-east-1", "ca-central-1",
]


def _env(name: str) -> str | None:
    """Read from the process env first, then the project .env, never printing it."""
    value = os.environ.get(name)
    if value:
        return value
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return None


def project_ref(url: str) -> str:
    """Extract ``xxxx`` from ``https://xxxx.supabase.co``."""
    host = url.split("//", 1)[-1].split("/", 1)[0]
    return host.split(".", 1)[0]


def check_rest(url: str) -> bool:
    """Is the project reachable and does the secret key authenticate?"""
    key = _env("SUPABASE_SECRET_KEY")
    if not key:
        print("  [skip] no SUPABASE_SECRET_KEY found in the environment or .env")
        return False
    request = urllib.request.Request(
        f"{url.rstrip('/')}/rest/v1/",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REST_TIMEOUT) as resp:
            spec = json.load(resp)
        version = (spec.get("info") or {}).get("version", "?")
        print(f"  [ok]   REST API reachable, Postgres {version}")
        return True
    except urllib.error.HTTPError as exc:
        print(f"  [FAIL] REST returned HTTP {exc.code} - key rejected or project paused")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
    return False


def check_direct_host(ref: str) -> bool:
    """The direct host is usually IPv6-only. Report that clearly, not as 'down'."""
    host = f"db.{ref}.supabase.co"
    print(f"3. Direct host ({host})")
    try:
        infos = socket.getaddrinfo(host, 5432, proto=socket.IPPROTO_TCP)
    except Exception as exc:  # noqa: BLE001
        print(f"  [info] does not resolve from this machine ({type(exc).__name__})")
        print("         expected when the project is IPv6-only -- use the pooler instead")
        return False
    families = {family.name for family, *_ in infos}
    print(f"  [ok]   resolves to: {', '.join(sorted(families))}")
    if "AF_INET6" in families and "AF_INET" not in families:
        print("         IPv6-only: unusable from an IPv4-only machine. Use the pooler.")
        return False
    try:
        sock = socket.create_connection((host, 5432), timeout=TCP_TIMEOUT)
        sock.close()
        print("  [ok]   TCP 5432 open on the direct host")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] TCP 5432 {type(exc).__name__}")
        return False


def check_pooler(ref: str) -> bool:
    """Is outbound 5432 possible at all, and does this project have a region?"""
    print("4. Supavisor pooler (the IPv4 route that actually works)")
    host = f"aws-0-{KNOWN_REGION}.pooler.supabase.com"
    try:
        sock = socket.create_connection((host, 5432), timeout=TCP_TIMEOUT)
        sock.close()
        print(f"  [ok]   {KNOWN_REGION}:5432 accepts TCP - egress is NOT filtered")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {KNOWN_REGION}:5432 {type(exc).__name__} - egress may be filtered")
        return False
    return asyncio.run(find_region(ref))


async def find_region(ref: str) -> bool:
    """Which region recognises this tenant? No password required to find out."""
    import asyncpg

    user = f"postgres.{ref}"

    async def probe(region: str) -> bool:
        try:
            conn = await asyncio.wait_for(
                asyncpg.connect(
                    host=f"aws-0-{region}.pooler.supabase.com",
                    port=5432,
                    user=user,
                    password="probe-not-the-real-password",
                    database="postgres",
                    ssl="require",
                    timeout=8,
                ),
                timeout=12,
            )
            await conn.close()
            return True
        except asyncpg.InvalidPasswordError:
            return True  # tenant exists; we merely guessed the password
        except Exception:  # noqa: BLE001
            return False

    async def sweep() -> str | None:
        for region in REGIONS:
            found = await probe(region)
            if found:
                print(f"  [ok]   region {region} recognises tenant {user}")
                return region
        return None

    region = await sweep()
    if region is None:
        print(f"  [FAIL] no region recognises tenant {user}")
        return False
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Supabase project URL")
    args = parser.parse_args(argv)

    ref = project_ref(args.url)

    print(f"Supabase project: {ref}")
    print()
    print("1. Project alive over HTTPS?")
    rest_ok = check_rest(args.url)
    print()
    direct_ok = check_direct_host(ref)
    print()
    pooler_ok = check_pooler(ref)
    print()

    print("=" * 70)
    if rest_ok and (direct_ok or pooler_ok):
        print("READY. Uncomment and fill in the DATABASE_URL line in .env:")
        print(
            f"  DATABASE_URL=postgresql+asyncpg://postgres.{ref}:<PASSWORD>"
            f"@aws-0-{KNOWN_REGION}.pooler.supabase.com:5432/postgres?ssl=require"
        )
        print()
        print("Note the three details that are easy to get wrong:")
        print(f"  1. username is postgres.{ref}  (the tenant, not plain 'postgres')")
        print("  2. ?ssl=require  (asyncpg does not understand 'sslmode')")
        print("  3. the pooler host, because the direct host is IPv6-only")
        print()
        print("Then:")
        print("  python -m alembic -c app/db/migrations/alembic.ini upgrade head")
        print("  python scripts\\seed_demo_data.py --reset")
    elif rest_ok:
        print("The project is up but no Postgres route works from this machine.")
        print("  - If the direct host is IPv6-only and you have no IPv6 route, the")
        print("    pooler is the only way; it needs TCP 5432 open outbound.")
        print("  - Ask whoever runs this network to allow TCP 5432 outbound.")
    else:
        print("The project itself is not answering. Check the URL and the key.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))