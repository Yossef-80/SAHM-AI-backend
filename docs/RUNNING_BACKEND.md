# Running the backend

> **Audience:** Anyone starting the API locally, in Docker, or in an environment.
> The health checks below are the same ones the Docker image and Compose stack use.

## 1. Start locally

From the repository root (this repository **is** the backend):

```bash
uv sync                              # or: pip install -e .
cp .env.example .env
uv run alembic upgrade head          # create the schema
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API boots with no credentials, but the chat needs a model: `.env.example`
selects `AI_LLM_PROVIDER=ollama`. For a fully offline run set it to `mock` — every
route then works and returns deterministic output, with no key and no server.
Switching to a real provider means changing that one variable (plus its key);
no code path changes.

### Docker

```bash
docker compose up --build           # API + database
docker compose logs -f api
```

Compose starts a one-shot `migrate` service that runs `alembic upgrade head`
before the API service starts; the API image itself deliberately contains no
migration step (so a rolling restart can never mutate the schema). The image
probes `/health` for liveness.

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

## 4. Logs

The API logs one JSON object per line with `request_id`, `workspace_id`,
`business_id`, `method`, `path`, `status`, `duration_ms` and DB query counts.
Values that look like credentials are redacted before they reach a handler. To
follow a single request end to end, grep its `request_id`.

## 5. Common startup failures

| Message | Cause | Fix |
| :--- | :--- | :--- |
| `insecure_runtime_configuration` | `TOKEN_ENCRYPTION_KEY` is unset | set a real key outside dev |
| `ProviderNotConfigured` | provider selected, key missing | set the key, or use `mock` |
| `no such table` | migrations not applied | `alembic upgrade head` |
| CORS failure in the browser | origin not in `CORS_ORIGINS` | add the frontend origin |
