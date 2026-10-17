# syntax=docker/dockerfile:1
#
# Sahm AI Marketing OS - backend image (P5.1).
#
# Build from the backend folder (the one whose name contains a SPACE):
#   cd "sahm-ai backend"
#   docker build -t sahm-backend:latest .
#
# NOTE ON PATHS: every COPY/WORKDIR below is a POSIX path *inside* the image, so
# the space in the build context directory never reaches the Dockerfile. You only
# need quotes in the *shell* command that names the build context.
#
# ---------------------------------------------------------------------------
# MIGRATIONS PACKAGING DECISION (P5.1) - why there is no `pip install .`
# ---------------------------------------------------------------------------
# The trap: `app/db/migrations/` has no `__init__.py`, and pyproject.toml uses
#     [tool.setuptools.packages.find] include = ["app*"]
# which only discovers real *packages*. So an installed wheel/site-packages copy
# of the app ships NO alembic.ini, NO env.py, NO script.py.mako and NO
# versions/ directory -- and
#     alembic -c app/db/migrations/alembic.ini upgrade head
# cannot work from an installed distribution at all.
#
# Adding `__init__.py` is NOT sufficient on its own: alembic.ini declares
#     script_location = app/db/migrations
# `script_location` is resolved relative to the *current working directory*, not
# relative to site-packages. Even with the package markers present, an installed
# copy only resolves if the process happens to be CWD'd into site-packages.
#
# DECISION: option (b) -- ship `app/` as plain source and run from the source
# tree. Stage 1 builds a venv with the *third-party* dependencies only (no
# `pip install .` of this project, so there is no wheel to lose files in), and
# stage 2 copies the venv plus the `app/` source tree with WORKDIR set to the
# directory that makes every relative path in alembic.ini resolve. This makes
# `alembic -c app/db/migrations/alembic.ini` and `python -c "import app"` behave
# identically in the image and on a developer machine.
#
# Belt-and-braces: `app/db/migrations/__init__.py` and
# `app/db/migrations/versions/__init__.py` ARE added (see the COPY comment in
# stage 2). They cost nothing, and if anyone later does build a wheel the
# migration tree travels with it instead of silently vanishing. They are not
# load-bearing for the image - the WORKDIR is.
#
# Verified: stage 2 contains /srv/app/app/db/migrations/alembic.ini,
# env.py, script.py.mako and versions/ (containing cfe093781e86_initial_schema.py),
# and `python -m alembic -c app/db/migrations/alembic.ini current` works from
# WORKDIR /srv/app.
#
# `scripts/` is deliberately NOT copied: the runtime image carries the service
# and its migration history, not the operational toolbox (`check_migrations.py`,
# `db_admin.py`, `seed_demo_data.py`). Run those on the host or from the compose
# `migrate` service, which mounts the repository.

# =========================================================== stage 1: builder ==
FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Runtime-only dependency set. Deliberately NOT `pip install .` -- see the
# migrations packaging decision above. Only third-party wheels land in the venv.
COPY pyproject.toml /tmp/pyproject.toml

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install \
      "fastapi>=0.115" \
      "uvicorn[standard]>=0.30" \
      "pydantic>=2.7" \
      "pydantic-settings>=2.3" \
      "SQLAlchemy>=2.0.30" \
      "alembic>=1.13" \
      "aiosqlite>=0.20" \
      "asyncpg>=0.29" \
      "langgraph>=0.2.60" \
      "httpx>=0.27" \
      "Pillow>=10.0.0" \
      "cryptography>=42" \
      "python-multipart>=0.0.9" \
      "beautifulsoup4>=4.12" \
      "greenlet>=3.5.6"

# ============================================================ stage 2: runtime ==
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:${PATH}" \
    ENVIRONMENT=dev

# curl is NOT installed: the healthcheck below uses the interpreter that is
# already in the venv, which keeps the runtime image small and removes a
# dependency on a distro package that may vanish between slim releases.
# libpq is not needed either: asyncpg is a pure-python client (it speaks the
# wire protocol itself), so there is no libpq to ship.

# Non-root user. `useradd --system` gives a uid with no login shell and no
# home-directory ownership games; the app never writes anywhere except
# ./var/uploads, which is owned below.
RUN useradd --system --create-home --uid 10001 --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

# WORKDIR /srv/app is what makes alembic work: from here, the relative
# `script_location = app/db/migrations` resolves to
# /srv/app/app/db/migrations, and `prepend_sys_path = .` puts /srv/app on
# sys.path so `import app` finds the source tree.
WORKDIR /srv/app

# The application, as source (see the packaging decision above).
COPY app/ /srv/app/app/
# Explicit and redundant on purpose: the task contract requires the migration
# tree in the runtime image, and naming it here means a future refactor that
# drops it from the general COPY fails loudly instead of shipping an image
# whose `alembic upgrade head` cannot run.
COPY app/db/migrations/ /srv/app/app/db/migrations/
COPY pyproject.toml /srv/app/pyproject.toml

# `var/` is in .dockerignore, so create the one directory the app writes to.
RUN mkdir -p /srv/app/var/uploads && chown -R app:app /srv/app

USER app

EXPOSE 8000

# HEALTHCHECK hits `/health` -- the *liveness* endpoint (`app/main.py:295`).
# It opens no database connection and does no I/O, which is the whole point:
# a transient Postgres blip must not make Docker restart a perfectly healthy
# process and turn a 30-second outage into a crash loop.
# The readiness endpoint `/health/ready` (`app/main.py:309`) is the one that
# checks the database and the encryption key, and it belongs to the orchestrator
# / load balancer as a readiness gate -- NOT to Docker's liveness probe.
# `docker-compose.yml` and `docs/RUNNING_BACKEND.md` use `/health` too.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# ---------------------------------------------------------------------------
# Production command (P5.4). `--reload` MUST NEVER appear here or in any
# production start command:
#
#   * it spawns a supervisor + a reloader parent, so `--workers N` is ignored
#     and you silently run ONE process - losing the concurrency you bought;
#   * the watcher consumes CPU and holds the source tree's mtimes;
#   * every file write in the container restarts the app, so a log write or a
#     volume mount that touches the tree becomes a self-inflicted outage loop.
#
# `--proxy-headers --forwarded-allow-ips` are required because the rate limiter
# (`app/core/ratelimit.py`) keys on `request.client.host`. Without them every
# request appears to come from the load balancer and one client can exhaust the
# bucket for everyone. `*` is correct *only* because the app is reachable solely
# through that proxy; restrict it to the proxy CIDR if it is not.
# `--workers 2` is an explicit floor, not a default: size it as
#   2 x CPU + 1, then re-check pool_size x workers against the database's
#   connection limit (see docs/PRODUCTION_ROADMAP.md §3.3).
# ---------------------------------------------------------------------------
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--forwarded-allow-ports", "443"]