# Contributing

> **Audience:** Anyone opening a pull request against this repository.
> **Read first:** [`AGENTS.md`](AGENTS.md) — the task router that tells you which
> files own which behaviour, so you do not have to read the tree. Then
> [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) and
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5 for the playbooks.

---

## 1. Get set up

```bash
git clone <repository-url>
cd SAHM-AI-backend
uv sync                    # or: python -m venv .venv && pip install -e ".[dev]"
cp .env.example .env       # defaults are offline-safe; no keys required
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

Before you start a branch, make sure the baseline is green:

```bash
uv run pytest -q
```

---

## 2. Branching

| Branch | Purpose |
| :--- | :--- |
| `main` | What is deployed. Protected. |
| `feat/<short-name>` | A feature or a tool. Branch from `main`, open the PR against `main`. |
| `fix/<short-name>` | A defect fix. |
| `docs/<short-name>` | Documentation only. |

```bash
git checkout -b feat/facebook-scraper-tool
```

One branch, one idea. A PR that adds a tool **and** refactors the orchestrator
cannot be reviewed.

---

## 3. What a good change looks like here

The codebase has a few non-negotiables. A PR that breaks one of them will not be
merged regardless of how well it works:

1. **Multi-tenancy is a parameter.** Every query is scoped by the `business_id` /
   `workspace_id` from the auth dependency — never from the request body.
2. **No agent can cause an external write.** Tools with
   `side_effect="external_write"` are rejected by the registry. Anything that
   changes a live ad account goes through a recommendation and a human approval.
3. **The UI contract is typed.** New response shapes are Pydantic schemas and UI
   blocks, not ad-hoc dictionaries.
4. **Memory goes through the store.** `BrandMemoryStore` owns dedupe, TTL and token
   caps; constructing `BrandMemoryItem` by hand bypasses all three.
5. **The reply language is decided once.** Never re-derive it from the prompt text;
   see [`docs/LANGUAGE_POLICY.md`](docs/LANGUAGE_POLICY.md).
6. **Fail closed and say so.** A missing key raises a typed error; a missing fact
   becomes `missing_inputs`, never an invented number.

---

## 4. Environment variables

If your change needs a new setting:

1. add it to **both** settings classes if both layers read it
   (`app/config.py`, `app/ai/settings.py`) — they are separate objects by design;
2. document it in `.env.example` with a comment explaining the effect of changing
   it, not just its name;
3. give it a safe default so a fresh clone still boots.

Never commit `.env`, a real API key, or a customer's data. Logs are redacted by
`app/core/logging.py`, but that is a safety net, not a licence.

---

## 5. Tests

- Every bug fix gets a test that fails before the fix and passes after it.
- Contract-level tests are preferred to implementation-level ones: assert the
  behaviour a caller depends on (the block shape, the language, the 404 for a
  foreign workspace), not the internal call order.
- No test may reach a live provider, the open web, or Meta. Use the existing fakes
  in `tests/`.
- Run the whole suite, not just your file:

```bash
uv run pytest -q
```

---

## 6. Commit messages

Conventional Commits, one line, imperative mood:

```
feat: add tiktok hashtag analyser for Nour
fix: keep Arabic answers when the UI language is Arabic
docs: rewrite the memory architecture reference
refactor: replace the three post-turn memory passes with one consolidator
test: pin the mid-conversation language switch
```

The body explains **why** the change exists and what it makes impossible. If the
change has a trade-off, write it down — the next reader will need it.

---

## 7. Pull request checklist

- [ ] The description says what changed and why, in that order.
- [ ] `uv run pytest -q` is green.
- [ ] New behaviour has a test; changed behaviour has an updated test.
- [ ] New settings are in both settings classes and in `.env.example`.
- [ ] Schema changes come with an Alembic migration, and the migration was run
      forward and back at least once.
- [ ] No `.env`, key, token, or customer data in the diff.
- [ ] Documentation updated in the same PR — especially
      `docs/MEMORY_ARCHITECTURE.md`, `docs/LANGUAGE_POLICY.md` or
      `docs/ARCHITECTURE.md` if the change touches those contracts.
- [ ] If the PR changes a user-visible answer, one sentence on the before/after.

---

## 8. Review

A reviewer checks, in this order:

1. **Correctness** — does it do what the description claims?
2. **Boundaries** — tenancy, side effects, language, typed outputs.
3. **Failure modes** — what happens with no key, no data, a slow provider, a
   second tenant?
4. **Cost** — an extra LLM call per turn is a product decision, not an
   implementation detail. Say so in the PR.
5. **Readability** — the next engineer should not need the PR to understand the
   file.

Review comments are about the code. If a discussion needs more than two rounds,
take it to an issue and land the smallest correct version first.

---

## 9. Code style

- `ruff` for lint and format (`uv run ruff check .`), line length 100.
- Type hints on public functions; `from __future__ import annotations` at the top
  of new modules.
- Comments explain *why*. The code already says *what*.
- No dead code, no commented-out blocks, no `TODO` without an owner and an issue.
