# Running the backend

This runbook applies to the `backend/` directory in the Sahm workspace. The
health checks below are also used by the Docker image and local Compose stack.

## 1. Start locally

From the repository root, install the backend dependencies and start the API:

```bash
cd backend
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Configure required environment variables in the deployment environment or a
local `.env` file. Never commit credentials. For an offline development run,
select the existing mock providers; this does not change the production
provider/model selection.

## 2. Liveness and readiness

The liveness endpoint confirms only that the API process is responding. It is
intentionally independent of Postgres, Meta, search, and model providers:

```bash
curl http://127.0.0.1:8000/health
```

The `Dockerfile` `HEALTHCHECK` and the API service in `docker-compose.yml` both
probe `/health`. A failed liveness check means the process should be restarted.

`/health/ready` is a separate readiness check. It can report unavailable when
required dependencies are not ready, without turning a temporary dependency
problem into a container restart loop. Probe it during rollout or from an
orchestrator readiness gate, not as the Docker liveness check.

## 3. Stop locally

Stop the `uvicorn` process with `Ctrl-C`. The health endpoints are read-only;
they do not perform database migrations or modify Meta assets.
