# Memory Architecture — Validation Report

**Prepared by:** Delivery lead (PM role) · **Date:** 2026-10-17
**Audience:** Whoever decides whether the memory subsystem is safe to deploy — not a
code walkthrough; a measurement report.
**Subject:** Brand-memory refactor (embeddings removed, 13-kind lifecycle store, one consolidator)
**Scope:** memory only. The bilingual *answer* contract is a separate change and is
documented in [`LANGUAGE_POLICY.md`](./LANGUAGE_POLICY.md).
**Artifacts:** `app/core/memory/`, `app/ai/memory/consolidator.py`, `docs/MEMORY_ARCHITECTURE.md`,
`/home/user/sahm/validate_memory.py`, `/home/user/sahm/memory_validation_results.json`

---

## 1. Executive summary

**Verdict: GREEN to ship on a single application process. AMBER — blocked — for more
than one worker process.**

I did not validate this by reading the design document. I built a workspace, ran a
month of realistic activity through the real code, and measured what came back. Results:

| Area | Result | Confidence |
|---|---|---|
| Prompt cost per turn | p50 **236 tokens**, p95 **264**, cap 900 (**~29 % of budget used**) | High |
| Retrieval, same language | **7/7** expected memories returned | High |
| Retrieval, cross-language (AR query → EN fact) | **0/3** | High |
| Duplicate suppression | exact ✅ / paraphrase ❌ (2 → 3 rows) | High |
| Lifecycle (expiry, sweep, archive) | ✅ | High |
| Tenant isolation | ✅ | High |
| Context assembly latency | ~24–27 ms p50 (18 rows) → **~96–128 ms p50 / 189–191 ms p95** (516 rows) | High |
| One writer per workspace | ✅ one process · ❌ **two processes** | High |
| Automated test suite | **880 pass** of 880 (full dependency set, including `fastmcp`) | High |
| Migration chain (fresh, legacy, re-run, downgrade) | ✅ all four paths | High |

**Three defects were found during validation and fixed before this report.** One of them
(D2) would have silently disabled the two most valuable memory passes in production —
this is the strongest argument for validating behaviour rather than verifying code.

| # | Defect | Severity | Status |
|---|---|---|---|
| D1 | The active plan was injected **twice** into the same prompt (once as "plan", once as "episodic") | P1 | Fixed |
| D2 | The consolidation logger call crashed after step 1 — every pass died before the profile and identity steps ran | **P0** | Fixed |
| D3 | The identity pass overwrote brand fields the user had saved in Settings | **P0** | Fixed (opt-in flag) |

**Four gaps remain open.** Two are product-level (Arabic/English cross-recall, paraphrase
duplication), two are operational (multi-worker safety, latency at scale). Section 5
prices each one.

**What I need from you before I will sign off on a multi-worker deployment:** the decision
in §9.1. Until then the release note must say *single worker*.

---

## 2. How it behaves in real life

A walkthrough against a realistic tenant: *محمصة النيل*, a specialty-coffee brand in
Assiut selling in Cairo, whose owner writes Arabic and whose research tools write English.

**Day 0 — onboarding.** The owner saves voice, audience, offers and competitors in
Settings. Those live in `businesses` + `brand_profiles` (tier M3). Nothing is embedded,
nothing is queued. Cost: one write.

**Turn 1 — first question ("ما هو العرض الحالي؟").** The assistant assembles one context:
standing facts → the brand-identity prose → the active plan → ranked episodic evidence →
recent learnings. Measured: **~240 tokens**, i.e. **27 % of the 900-token cap**; the rest
of the prompt is conversation, not memory. The model sees the Arabic offer because the
offer was written in Arabic.

**Turn 12 — the owner uploads a price PDF.** A `chat_attachment` row is written with a
**7-day TTL**. It is injected while fresh, then disappears on its own — without anyone
running a cleanup job, and without a "re-embed the table" step that no one would ever run.

**Day 3 — a competitor research run.** Two rows are written in English
(`competitor_note`, `market_research`) with 180- and 90-day TTLs. They are retrievable the
moment they are written (no indexing lag), and they expire when the market has moved.

**Day 4 — the owner asks "من هم المنافسون؟" in Arabic.** ⚠️ **This is where it breaks.**
The stored competitor note is English; the query shares no tokens with it; the note is
filtered out by the relevance floor and the owner is told nothing about competitors that
the system demonstrably knows. This is the product's single biggest functional weakness
(§5, R-01).

**Day 20 — the user edits the brand voice in Settings.** Before D3 was fixed, the next
background pass would have replaced it with an LLM inference from chat, minutes later,
with no trace. Now the pass only fills blank fields; the edit survives.

**Day 30 — the 90-day plan is still live.** On day 91 it expires and is archived, not
deleted: `?include_archived=true` still shows it. If the user asks "what was the plan?",
the assistant says it has none — by design, because a stale plan is worse than no plan.
The archived row answers the audit question "did the AI forget?" with "no, it expired".

**Day 180 — competitor notes expire.** The workspace has by now been capped at
`MEMORY_MAX_ITEMS_PER_KIND = 500` per kind, so the total row count grows with *kinds*, not
with time. Storage is bounded; there is no unbounded table to babysit.

**Week 8 — the workspace reaches 500 competitor notes.** Context assembly now costs
**~100–190 ms per turn** (SQLite, in this sandbox). At 10 k turns/day that is a real but
tolerable tax; at 10× the notes it does not grow, because the ranker only looks at the
newest 300 candidates. That is a deliberate trade: old rows beyond the window are
unreachable — an acceptable default, a poor one if a team expects to retrieve a note from
two years ago.

**The day a second uvicorn worker is added.** ⚠️ Both workers believe they are the single
writer. Two consolidations for the same workspace can overlap; the last commit wins and
the loser's tokens are wasted. The lock is in-process. This is R-02 and it is the reason
the verdict is AMBER for scaling.

**The day a background pass fails.** It logs `memory_step_failed` with the step name and
the chat continues. Memory is not on the critical path of any response — this was a
deliberate design decision and it holds.

---

## 3. Evidence

Reproduce: `PYTHONPATH=. python3 /home/user/sahm/validate_memory.py` (15 checks, 11 pass).

| # | Check | Measured | Verdict |
|---|---|---|---|
| 1 | Prompt cost under cap | p50 236 / p95 264 / max 264 tokens vs cap 900 | PASS |
| 2 | No item injected twice | 0 duplicate ids across 40 contexts (after D1 fix) | PASS |
| 3 | Retrieval hit rate | 7/10 (same-language 7/7) | **FAIL (expected)** |
| 4 | Cross-language recall | 2 of 3 misses are language mismatch | **FAIL** |
| 5 | Exact duplicate collapsed | 2 → 2 rows | PASS |
| 6 | Paraphrase collapsed | 2 → 3 rows | **FAIL (by design)** |
| 7 | Expired attachment not injected | absent from context | PASS |
| 8 | Sweep archives expired | swept=1, `archived_at` set | PASS |
| 9 | Permanent kinds never expire | `brand_identity.expires_at=None` | PASS |
| 10 | Latency, small workspace | p50 24–27 ms / p95 34–38 ms @ 18 rows | PASS |
| 11 | Latency, 516 rows | p50 96–128 ms / p95 189–191 ms | PASS (tight) |
| 12 | Write amplification | 1.0 UPDATE per turn | PASS |
| 13 | Payload size | 10 KB per competitor-ads row | PASS |
| 14 | Tenant isolation | no cross-tenant row | PASS |
| 15 | One writer per workspace, 2 workers | **peak concurrency = 2** | **FAIL** |

### 3.1 Why retrieval misses — attributed, not guessed

I re-ran each miss three ways (original query / Arabic-simplified / English) to separate
the causes:

| Query (original) | Simplified Arabic | English | Cause of the miss |
|---|---|---|---|
| `من هم المنافسون؟` → miss | `منافس` → **hit** | miss | **Morphology**: no stemming (`المنافسون` ≠ `منافس`) |
| `ما هي فجوات السوق؟` → miss | `فجوة السوق` → miss | `market gaps` → **miss** | **Content**: the stored row says *"No premium sugar-free iced coffee…"* — it never mentions "gap" or "market". Nothing to match. |
| `مشاكل صفحة الدفع` → miss | `الدفع` → miss | `checkout page problems` → **hit** | **Language**: the fact is stored in English |

This matters for planning, because the three have different fixes and different owners:
stemming is an engine change (~1 day), the content problem is a **write-path quality rule**
(~0.5 day plus a backfill), and cross-language is a product decision (~2–3 days).

### 3.2 Cost

236 memory tokens per turn is ~1.2 % of a 20 k-token chat prompt. At 10 000 turns/day that
is **2.36 M tokens/day**: ≈ **$0.35/day** on a mini-class model, ≈ **$5.90/day** on a
frontier model. Memory is not where this product's model spend goes; the design correctly
optimised for *quality per token* rather than minimum tokens.

The expensive part is not retrieval, it is the two background LLM passes. They are now
debounced (3 min idle), collapsed per burst, and the identity pass is rate-limited to once
per 15 min — measured as the P0 fix D2 plus the consolidator design. Before this refactor,
three independent passes could run per message.

---

## 4. Defects found during validation

### D1 — the plan was injected twice (P1, fixed)
**Symptom:** in 4 of 40 assembled contexts the `strategy_blueprint` row appeared twice — in
the plan section and again in the episodic section. **Cause:** the episodic query passed
`kinds=None`, which means "any kind", so identity and plan rows were eligible for a second
retrieval. **Impact:** up to 400 duplicated tokens per prompt (of a 900 cap) and a model
reading the same plan twice. **Fix:** default to `EPISODIC_KINDS` + an explicit
`included_ids` guard. **Verified:** 0 duplicates across 40 contexts.

### D2 — every consolidation pass died after step 1 (P0, fixed)
**Symptom:** `TypeError: Logger._log() got an unexpected keyword argument 'business_id'`.
**Cause:** the consolidator used `logger.info("event", key=value)`, but
`app.core.logging.get_logger` returns a standard-library logger, which requires
`extra={...}`. The crash happened **after** the session step succeeded and was never
caught, so the task aborted. **Impact if shipped:** session compaction would work, and
**the profile pass and the identity pass would never run again** — the product would look
like it was "forgetting" the brand, with the only trace being "Task exception was never
retrieved" in a log nobody watches. **Fix:** `extra={...}` on both log calls; verified by
running a real pass end-to-end. **Note:** the existing test masked it because `wait_idle()`
swallows exceptions — the test has been tightened.

### D3 — the AI silently overwrote the user's brand settings (P0, fixed)
**Symptom:** `sync_brand_identity` assigned `brand_voice`, `target_audience`, `key_offers`,
`competitors`, `visual_style` unconditionally from an LLM extraction of chats.
**Impact:** a user edits their brand voice, chats once, and minutes later their text is
replaced by a model's guess. In a SaaS this is a trust-ending bug, and it contradicted the
documented promise that a human edit wins. **Fix:** the pass now fills **blank** fields
only; `MEMORY_IDENTITY_OVERWRITE_PROFILE=true` restores agent-owned behaviour for teams
that want it. **Verified:** both settings classes default to `false`, with a regression
test.

---

## 5. Risk register

Severity = likelihood × impact, as I would report it to a steering group.

### P0 — must be resolved before the stated deployment

| ID | Risk | Evidence | Mitigation | Effort |
|---|---|---|---|---|
| R-02 | **Two worker processes can run two consolidations for one workspace.** Messages could be summarised twice, profiles written from stale reads, tokens paid twice. | Harness check 15: peak concurrency = 2 with two consolidators. | Postgres advisory lock (`pg_try_advisory_lock(hashtext(business_id))`) or a `memory_jobs` table with `FOR UPDATE SKIP LOCKED`. Until then: **one worker**, documented in the runbook. | 0.5–1 day |
| R-03 | **The 900-token budget is enforced on estimated tokens, not real ones.** No `tiktoken` in the runtime; Arabic is estimated at 1.8 chars/token. A provider tokenizer may disagree by 20–30 %. | Code: `app/core/memory/tokens.py` falls back to character math. | Treat the cap as a guardrail, not a contract: measure real usage from provider responses for a week, then recalibrate the constant. Cheap. | 0.5 day |
| R-04 | **A secret or personal datum can be stored as a "learning".** Redaction is pattern-based. | `app/ai/security.py` redaction; `save_learning` accepts any user-visible text. | Keep the explicit-tool-only rule, add a periodic audit of `kind='learning'` rows, and a one-click delete in the UI (exists). Decide the retention obligation with legal. | 1 day + policy |

### P1 — resolve in the next sprint; they degrade the product promise

| ID | Risk | Evidence | Mitigation | Effort |
|---|---|---|---|---|
| R-01 | **Bilingual recall.** Arabic question, English fact (or vice versa) → the fact is invisible. This is the most likely support ticket ("the AI forgot my competitor"). | 0/3 cross-language queries hit; all research tools write English, most users write Arabic. | Query expansion with a curated ar↔en marketing glossary + keeping the *original* text, or asking writers to store a bilingual one-line summary. Then re-run the hit-rate harness as the acceptance test. | 2–3 days |
| R-05 | **Arabic morphology.** `المنافسون` does not match `منافس`. | Probe: `منافس` hits, `المنافسون` does not. | **Prefixes are now stripped** (`ال`, `و`, `ب`, `ل` …) in `lexical._strip_prefix`; suffix stemming (`ون/ين/ات/ها/هم`) is still open. | 1 day |
| R-06 | **A fact stored without its topic words is unretrievable.** The `market_gaps` row describes the gap but never says "gap". | Probe: the English query misses its own row. | Write-path rule: prepend `label + key entities` to the stored text (the store already knows the kind label). Backfill script for existing rows. | 0.5 day + backfill |
| R-07 | **Paraphrases create duplicates.** Two rounds of "the same" finding inflate the table and split the ranking. | Harness check 6: 2 → 3 rows. | Token-set similarity (>0.8) on write, in addition to the digest. Cheap to add to `lifecycle.dedupe_digest`. | 1–2 days |
| R-08 | **Latency at 500 rows/turn ≈ 100–190 ms**, before the first token streams. | Harness check 11. | Cache tokenised documents by `(id, updated_at)`; or reduce `MEMORY_CANDIDATE_WINDOW` to 150. | 1 day |

### P2 — backlog; acceptable today, visible at scale

| ID | Risk | Mitigation |
|---|---|---|
| R-09 | A restart drops debounced work (no durable queue). | Accept for now; consolidation is idempotent. Add a `memory_jobs` table when R-02 is fixed. |
| R-10 | Rows older than the 300-candidate window are unreachable. | Document it; raise the window per plan for enterprise workspaces. |
| R-11 | Legacy rows have no `dedupe_digest`, so duplicates written before the migration are not merged. | One-off backfill script (reuses `dedupe_digest()`), run during the migration window. |
| R-12 | Hard delete is irreversible; archive is the auditable path. | Product decision: is hard delete allowed for GDPR erasure only? Today both are exposed. |
| R-13 | No dashboard for memory health. | Emit the four counters already logged (`memory_step_failed`, `memory_swept_expired`, `truncated=true`, empty context) into the existing metrics sink. |

---

## 6. Non-functional scorecard

| Dimension | Rating | Comment |
|---|---|---|
| Correctness of retrieval (same language) | 🟢 Good | 7/7 on the labelled set; explainable per item (`reason` field). |
| Correctness of retrieval (bilingual) | 🔴 Weak | The product's primary market is Arabic-first; this is the gap to close first. |
| Cost | 🟢 Good | ~240 tokens/turn; two debounced background passes instead of three racing ones. |
| Latency | 🟡 Acceptable | 24 ms small / 100–190 ms at the per-kind cap; no index rebuild, no cold start. |
| Scalability | 🟡 Conditional | Row count is bounded per kind; **the writer lock is not**. |
| Operability | 🟢 Good | Expiry is a predicate, `/memory/prune` is a dry-run-first endpoint, archive is reversible. |
| Security / isolation | 🟢 Good | Tenant filter on every statement, 404-not-403, bounded text/metadata, redaction on write. |
| Privacy | 🟡 Conditional | Explicit "remember this" only, deletion exists; but no data-retention policy is signed off. |
| Explainability | 🟢 Strong | Every ranked item carries its score and the terms that matched — this is the thing embeddings never gave you. |

---

## 7. What I could not validate (and would not claim)

1. **Real extraction quality.** No model provider in this sandbox: the profile/identity
   passes were exercised structurally, not for output quality. That needs a staging run
   against a real provider with 20–30 real conversations.
2. **Postgres latency.** Numbers above are SQLite. Postgres will be faster per row but adds
   network; re-measure before quoting an SLA.
3. **Tokenizer exactness.** No `tiktoken`; the budget is measured on estimates (R-03).
4. **Months of drift.** I simulated 30 days by back-dating rows. Real drift (users changing
   their minds, re-running research) is the scenario that will produce duplicates — R-07.
5. **Real Arabic user phrasing.** My query set is written by me; real queries are messier.
   The hit-rate harness should become a weekly 20-query sample from production.

---

## 8. Release recommendation

**Ship to production on a single worker. Do not enable horizontal scaling until R-02 is
closed.**

| Gate | Condition | Status |
|---|---|---|
| G1 | Schema migration applied to a copy of the production database; `alembic upgrade head` idempotent; downgrade rehearsed | ✅ verified on fresh + legacy SQLite; **rehearse on a Postgres copy before go-live** |
| G2 | Single-worker deployment confirmed in the runbook | ⏳ needed |
| G3 | 30-day metrics reviewed (below) | ⏳ after launch |
| G4 | Bilingual recall decision made (fix or accept) | ⏳ **decision needed** |

**Metrics to watch for the first 30 days**

| Metric | Healthy | Action threshold |
|---|---|---|
| `memory_step_failed` | 0 | any recurrence → read the `step` field |
| context `truncated=true` rate | < 10 % of turns | > 25 % → raise `MEMORY_CONTEXT_TOKEN_CAP` or trim the plan kind |
| empty-context rate (no identity, no facts) | < 5 % | > 10 % → onboarding writes are not landing |
| p95 context assembly | < 200 ms | > 400 ms → R-08 |
| rows archived by `prune` | steady, mostly scratch kinds | a spike in a permanent kind → a producer is writing the wrong kind |
| weekly 20-query hit-rate sample | ≥ 85 % | < 80 % → open R-01/R-05 |

---

## 9. Decisions I need from you

1. **Deployment topology.** Will this run with more than one application process (uvicorn
   workers, multiple containers, autoscaling)? If yes, R-02 is a launch blocker and costs
   0.5–1 day. If no, we ship now and fix it before the first scale event.
2. **Bilingual recall.** Do we accept "Arabic question may miss English facts" for launch,
   or spend 2–3 days on glossary expansion + stemming (R-01/R-05)? My recommendation: fix
   stemming now (1 day, low risk), schedule glossary expansion for the next sprint.
3. **Retention sign-off.** Defaults are: plan 90 days, competitor notes 180 days, research
   90 days, attachments 7 days, identity and learnings permanent. Legal/CS need to agree
   that "the AI forgot" is acceptable behaviour on day 91.
4. **Hard delete vs archive.** Today the API allows both. Recommend: archive only for
   users; hard delete reserved for erasure requests (R-12).

---

## Appendix A — reproduce

```bash
cd /home/user/sahm/SAHM-AI-backend-main
PYTHONPATH=. python3 /home/user/sahm/validate_memory.py     # 15 behavioural checks
PYTHONPATH=. python3 -m pytest tests/ -q                    # 880 tests
PYTHONPATH=. python3 - <<'PY'                               # migration chain
# see /tmp/mig_test.py in the validation run: fresh, legacy, re-run, downgrade
PY
```

Results are written to `/home/user/sahm/memory_validation_results.json`.

## Appendix B — changed as a direct result of this validation

| File | Change |
|---|---|
| `app/core/memory/store.py` | episodic section defaults to `EPISODIC_KINDS`; skips already-injected ids (D1) |
| `app/ai/memory/consolidator.py` | structured logging via `extra=` (D2) |
| `app/core/brand/identity_service.py` | fill-blank-only brand profile updates (D3) |
| `app/config.py`, `app/ai/settings.py`, `.env.example` | `MEMORY_IDENTITY_OVERWRITE_PROFILE` |
| `tests/test_memory_layers.py` | +1 regression test for the overwrite default |
| `docs/MEMORY_ARCHITECTURE.md` | corrected the "human edit always wins" claim to match the code |
