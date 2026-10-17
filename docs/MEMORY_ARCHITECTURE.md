# Brand Memory Architecture

> **Audience:** Engineers changing retrieval, retention or prompt assembly — and
> reviewers who want to know why there is no vector database.
>
> This document is authoritative for how Sahm stores, ranks, budgets, expires
> and injects memory. Where it disagrees with any other document (including
> [`ORCHESTRATOR.md`](./ORCHESTRATOR.md), which covers the routing and the
> post-turn workers), this file wins.
>
> Code: `app/core/memory/` (design + store), `app/ai/memory/` (LLM passes),
> `app/api/memory.py` (operator API), `app/db/models.py` (`BrandMemoryItem`).

---

## 1. What problem this solves, in one paragraph

A marketing assistant is only useful if it already knows the brand: the offers,
the tone, the audience, the competitor it lost to last month, the plan agreed
two weeks ago, and the fact that a "short UGC reel" worked while a static carousel
did not. That knowledge has to survive a restart, survive a *thousand* tenants,
fit in a prompt of a few hundred tokens, and go stale on purpose when it stops
being true. It must also never be a place where one workspace can read another's
data, or where an unbounded blob of text can be smuggled into a system prompt.

The previous implementation tried to do this with a vector column. It did not
work, and section 9 explains exactly why, because "we removed the embeddings" is
only a good decision if the replacement is demonstrably better.

---

## 2. The memory tiers

Six tiers, ordered from shortest-lived to longest-lived. Each tier has exactly
one storage location and exactly one writer.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ M0  WORKING      the in-flight turn (LangGraph state, no persistence)        │
│ M1  SESSION      messages + conversation.summary        ← compaction pass    │
│ M2  PROFILE      Profile.user_context_summary (≤300 tok) ← LTM pass          │
│ M3  IDENTITY     BrandProfile + Business (structured)   ← Settings page /    │
│                                                          identity pass       │
│ M4  BRAND        brand_memory_items (13 kinds)          ← tools + services   │
│ M5  SCRATCH      brand_memory_items with a short TTL    ← attachments, refs  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### M0 — Working memory
The single turn in progress. It lives in the agent graph state and in
`messages`. Nothing is persisted specially, nothing is ranked. If a value only
matters for this response, it does not belong in M1–M5.

### M1 — Session memory
`conversations.summary` plus the last `MEMORY_RETAIN_RECENT_TURNS` raw turns.
While the conversation's token count is under `SESSION_MEMORY_MAX_TOKENS` the
model gets raw turns; past it, the **compaction** step rewrites older turns into
`conversation.summary` and keeps the tail verbatim.

*Why a rolling summary and not "keep everything":* a 40-turn Arabic
conversation is ~30k tokens; sending it every turn is both expensive and worse
for accuracy ("lost in the middle"). The summary is scoped to the conversation,
so it cannot leak into another one.

### M2 — Profile memory (the "brand brain")
One row per business: `profiles.user_context_summary`. It is the distilled,
permanent paragraph that every agent gets: what the brand is, what it sells, who
it sells to, what has been tried. Hard cap `MEMORY_GLOBAL_PROFILE_MAX_TOKENS`
(default 300). Written only by the **profile pass** (section 4).

### M3 — Identity memory (structured)
`businesses` + `brand_profiles`: name, industry, market, currency, brand voice,
visual style, target audience, key offers, goals, competitors. This is the
canonical form of the facts in M2, and it is what the Settings page edits. The
identity pass may *propose* updates to it; a human edit always wins.

### M4 — Brand memory items (the episodic record)
`brand_memory_items`, one row per fact/insight/artifact. This is where research
output, competitor notes, reference analyses, the active strategy blueprint and
explicit learnings live. Every row carries the lifecycle columns in section 5.
Retrieval from this tier is what section 6 describes.

### M5 — Scratch memory
Same table, but with a short TTL (`chat_attachment`: 7 days,
`market_research` / `market_gaps`: 90, `website_analysis`: 120). These exist to make the *next few*
conversations coherent (e.g. "the PDF I uploaded yesterday"), not to be part of
the brand's permanent record. They expire automatically and are pruned first.

---

## 3. Memory kinds — one registry, thirteen rows

`app/core/memory/kinds.py` is the only place that enumerates kinds. Before it,
seven kinds were listed in the API and thirteen were written by the code; six of
them (`brand_identity`, `strategy_blueprint`, `market_research`, `market_gaps`,
`website_analysis`, `chat_attachment`) were **written but never retrievable**,
because retrieval only searched the original seven.

Every kind declares:

| field | meaning |
|---|---|
| `layer` | `identity` / `plan` / `episodic` — which prompt section it can occupy |
| `ttl_days` | `None` = permanent; otherwise the item expires and is archived |
| `importance` | prior weight 0..1 before any per-item signal |
| `max_tokens` | per-item cap applied **on write** |
| `tasks` | task groups this kind is useful for (affinity bonus) |
| `api_writable` | whether `POST /memory/items` may create it |

| kind | layer | ttl | importance | max tok | written by |
|---|---|---|---|---|---|
| `brand_identity` | identity | — | 0.85 | 200 | identity pass, Settings sync |
| `voice_example` | identity | — | 0.70 | 120 | API, tools |
| `visual_style` | identity | — | 0.70 | 120 | API, tools |
| `offer` | identity | — | 0.70 | 120 | API, tools |
| `audience` | identity | — | 0.70 | 120 | API, tools |
| `strategy_blueprint` | plan | 90d | 0.80 | 400 | strategy tool, LTM pass |
| `competitor_note` | episodic | 180d | 0.55 | 150 | competitor tools |
| `market_research` | episodic | 90d | 0.50 | 150 | research API/tools |
| `market_gaps` | episodic | 90d | 0.55 | 150 | research API/tools |
| `website_analysis` | episodic | 120d | 0.45 | 150 | website tool |
| `reference_analysis` | episodic | 365d | 0.50 | 150 | reference analysis |
| `chat_attachment` | episodic | 7d | 0.10 | 40 | upload pipeline |
| `learning` | learning | — | 0.60 | 120 | `save_learning`, API |

Rules that fall out of the table:

* **A permanent kind must earn its place.** Only identity facts, the active plan
  and explicit learnings are permanent. Everything derived from a single
  research run expires, because the market moves and a stale "competitor X
  charges 999 EGP" is worse than no note at all.
* **`strategy_blueprint` is a singleton.** "The plan" is state, not an event;
  `BrandMemoryStore.upsert_singleton` overwrites the one active row. Appending a
  new plan per run would make five plans compete for 135 tokens.
* **`brand_identity` is service-owned.** It exists so agents can read the
  identity as prose without re-joining three tables; the API refuses to create
  it (`api_writable=False`), because a hand-written row would disagree with
  `brand_profiles`.

---

## 4. The write path

### 4.1 Synchronous writes (tools, API, services)

All writes go through `BrandMemoryStore.write_item(...)`, which:

1. **validates the kind** against the registry (an unknown kind is a bug, not a
   silent row);
2. **truncates to `spec.max_tokens`** (or `MEMORY_ITEM_MAX_TOKENS`, whichever is
   smaller) — a memory item is injected verbatim into a prompt, so an unbounded
   row is a prompt-injection surface;
3. **computes `dedupe_digest`** = `sha256(kind|business_id|campaign_id|normalise(text))`
   and, when `MEMORY_DEDUPE_ENABLED`, updates the existing row instead of adding
   a near-duplicate ("we sell cold brew" said five ways is one fact);
4. **stamps `expires_at`** from the kind's TTL, so expiry is an indexed predicate
   rather than Python arithmetic over the table;
5. **sets `importance`** from the kind prior (or the explicit value);
6. **sweeps** expired rows when `MEMORY_SWEEP_ON_WRITE` is on.

Producers that construct `BrandMemoryItem` directly (identity sync, research
API, chat attachments, reference analysis, competitor-ads cache) call
`memory_defaults(kind, ...)` so they get the same TTL/importance/dedupe
treatment. A row written without it would never expire and would outlive every
row the store writes.

### 4.2 Background consolidation — one writer, not three

Before: after every chat turn the API started **three** independent background
passes — session compaction, a debounced LTM rewrite, and a brand-identity
extraction. Nothing serialised them. Two rapid turns could interleave the LTM
and identity passes, each reading `profiles`, editing it and committing; last
writer won, and both completions were paid for. Repro: two messages 2 s apart
produced three profile writes and two identity writes in under a minute.

Now: `app/ai/memory/consolidator.py` — `memory_consolidator.schedule(...)`:

```
schedule(business_id, conversation_id)
   │
   ├─ per-business asyncio.Lock  ── one pass in flight, ever
   │
   ├─ 1. SESSION     compact if over SESSION_MEMORY_MAX_TOKENS (immediate)
   │
   ├─ sleep(MEMORY_LTM_DEBOUNCE_SECONDS)      ← collapse a burst into one pass
   │
   ├─ 2. PROFILE     IntelligentMemoryCompressor.update_global_profile()
   │                 (skips itself when no message is newer than its watermark)
   │
   └─ 3. IDENTITY    BrandIntelligenceAgent.sync_brand_identity()
                     (at most once per MEMORY_IDENTITY_MIN_INTERVAL_SECONDS)
```

Each step is isolated: a failure is logged as `memory_step_failed` with its step
name and the next step still runs. The whole thing runs after the response is
streamed, so memory can never fail a chat turn.

**One writer per tier** is the invariant. If you find yourself adding a fourth
pass, add it as a step here instead.

---

## 5. Lifecycle: freshness, expiry, pruning

| column | purpose |
|---|---|
| `created_at` | when the fact was learned (recency signal) |
| `updated_at` | when it was last rewritten (dedupe hit or singleton update) |
| `expires_at` | materialised TTL; `NULL` = permanent |
| `archived_at` | soft delete — out of retrieval, still auditable |
| `importance` | ranking input, default from the kind |
| `pinned` | never filtered by relevance; survives pruning |
| `use_count`, `last_used_at` | proof an item was injected into a prompt |
| `dedupe_digest` | digest of the normalised content |

Three mechanisms, in increasing order of aggression:

1. **Expiry** (`sweep_expired`): `expires_at <= now` → `archived_at = now`,
   reason `expired`. Runs on write (cheap, indexed) and on `/memory/prune`.
2. **Per-kind ceiling** (`prune_over_cap`): above
   `MEMORY_MAX_ITEMS_PER_KIND`, archive the worst-ranked unpinned rows. Ranking
   for pruning is `(pinned, importance, use_count, created_at)` — a note that
   has been injected twenty times beats a fresher one that never matched.
3. **Manual archive/delete** via the API; deletes are hard, archives are
   reversible.

Nothing is ever silently dropped. An expired row stays visible under
`?include_archived=true`, which is what makes "why did the AI forget X?" an
answerable question instead of a mystery.

---

## 6. The read path

### 6.1 No embeddings: what replaced them

Retrieval is a deterministic score over rows already scoped by
(business, kind, active, campaign):

```
score = ( w_affinity · affinity
        + w_lexical  · lexical
        + w_recency  · recency
        + w_importance · importance ) / Σw
        + pinned_boost  (if pinned)
        + campaign_boost (if it belongs to the campaign in scope)
```

Defaults: `w_affinity 0.20`, `w_lexical 0.40`, `w_recency 0.25`,
`w_importance 0.15`; `pinned_boost 0.35`; `campaign_boost 0.10`;
`phrase_boost 0.10`; `min_relevance 0.12`.

* **lexical** — BM25 (`k1=1.5`, `b=0.75`) over an Arabic/English-normalised
  token stream, normalised to 0..1. Normalisation (`app/core/memory/lexical.py`)
  strips harakat and tatweel, folds `أإآٱ→ا`, `ىئ→ي`, `ة→ه`, `ؤ→و`, and strips
  the clitics `وال بال كال فال لل ال و ب ك ف` so `بالإعلانات` matches `إعلانات`.
  This matters: Arabic is the primary language of the product and the older
  `[a-z0-9']+` tokenizer returned **zero tokens** for Arabic text.
* **recency** — `0.5 ** (age_days / MEMORY_RECENCY_HALF_LIFE_DAYS)`; a 30-day-old
  item scores half a fresh one.
* **affinity** — 1 when the kind lists the current task group (research,
  strategy, creative, analytics, optimization), else 0. A `market_research` note
  should not outrank an `offer` when the agent is writing copy.
* **importance** — the kind prior, or an author-set value.
* **`min_relevance`** — with a query present, rows scoring below it are dropped
  **unless pinned**. This is what stops "what is our offer?" from dragging in a
  competitor note that happens to share the word "price".

Every hit carries a human-readable `reason()`
(`lex=0.81 rec=0.42 imp=0.70 aff=1`), and the API returns it, so a bad ranking is
debuggable from a response body instead of a code reading.

### 6.2 Prompt assembly

`BrandMemoryStore.build_context()` produces one `BrandContext` under a hard
token budget (`MEMORY_CONTEXT_TOKEN_CAP`, default 900):

```
1. standing facts     Business + BrandProfile (identity fields, offers, goals)
2. distilled facts    Profile.user_context_summary            (M2)
3. identity memories  brand_identity, voice, style, offers, audience
                      budget: MEMORY_IDENTITY_SHARE × cap  (default 405)
4. active plan        the one strategy_blueprint
                      budget: MEMORY_PLAN_SHARE × cap      (default 135)
5. episodic evidence  top-k ranked research/competitor/reference notes
6. learnings          up to MEMORY_RECENT_LEARNINGS, ranked
```

Ordering is by **layer, not by score**: identity facts are always present before
one strong episodic match can consume the budget. Each section stops adding when
its share or the remaining total is exhausted, and `truncated=True` is set so the
caller can say "this is not everything" instead of pretending it is.

Retrieved ids are recorded via `touch_items(...)` (tenant-scoped), feeding
`use_count`/`last_used_at` for pruning.

### 6.3 What the agent sees

`get_brand_context` (AI-layer tool) reads — in order of authority — the Settings
profile supplied by the conversation bridge, the stored `brand_profiles` row, the
distilled `user_context_summary`, and recent learnings. It answers `partial` with
`unavailable_reason` when nothing is stored; it never invents brand facts.

The DB is authoritative. The LangGraph `BaseStore` path in
`app/ai/memory/brand_memory.py` is kept **only** as a fallback for an agent graph
with no database attached (offline scripts, unit tests). It is not a second
source of truth: writes go to Postgres whenever a `tool_db` is present.

---

## 7. Multi-tenancy and safety

* Every query is filtered by `business_id`; there is no cross-workspace read path.
* Id-addressed routes return **404, not 403**, for another workspace's row — a 403
  confirms the row exists and turns ids into an enumeration oracle.
* `touch_items` is tenant-scoped so a stray id cannot bump a counter elsewhere.
* Text is capped per kind at write time and again at assembly time; the API
  additionally caps `text` (4 000 chars) and `metadata` (32 keys / 8 KB).
* Secrets are redacted by `app/ai/security.py` before a learning is stored.
* "Remember this" is an explicit tool call (`save_learning`); the background
  passes only distil what the user already said.

---

## 8. Configuration

Every knob is a `MEMORY_*` environment variable, present identically on
`app.config.Settings` and `app.ai.settings.AISettings`, and documented in
`.env.example`. `MemoryPolicy.from_settings()` reads either object by attribute
name, so the store, the ranker and the assembler can never disagree.

| variable | default | effect |
|---|---|---|
| `MEMORY_ENABLED` | `true` | master switch: no brand memory is read or written |
| `SESSION_MEMORY_MAX_TOKENS` | `10000` | history ceiling before compaction |
| `MEMORY_SESSION_MAX_TOKENS` | — | alias of the above |
| `MEMORY_RETAIN_RECENT_TURNS` | `2` | raw turns kept after a summary |
| `MEMORY_LTM_DEBOUNCE_SECONDS` | `180` | idle time before the profile pass |
| `BRAND_MEMORY_TOKEN_LIMIT` | `10000` | ceiling for the distilled brand facts |
| `MEMORY_GLOBAL_PROFILE_MAX_TOKENS` | `300` | cap on `user_context_summary` |
| `MEMORY_IDENTITY_SYNC_ENABLED` | `true` | run the identity pass at all |
| `MEMORY_IDENTITY_MIN_INTERVAL_SECONDS` | `900` | at most one identity pass per interval |
| `MEMORY_IDENTITY_OVERWRITE_PROFILE` | `false` | allow the identity pass to replace user-saved profile fields |
| `MEMORY_CONTEXT_TOKEN_CAP` | `900` | total tokens memory may inject per prompt |
| `MEMORY_CONTEXT_TOP_K` | `6` | candidates per retrieval section |
| `MEMORY_RECENT_LEARNINGS` | `3` | learnings considered per prompt |
| `MEMORY_IDENTITY_SHARE` | `0.45` | share of the cap reserved for identity |
| `MEMORY_PLAN_SHARE` | `0.15` | share reserved for the active plan |
| `MEMORY_CANDIDATE_WINDOW` | `300` | newest rows ranked per query |
| `MEMORY_RECENCY_HALF_LIFE_DAYS` | `30` | recency decay |
| `MEMORY_WEIGHT_AFFINITY` | `0.20` | kind/task fit weight |
| `MEMORY_WEIGHT_LEXICAL` | `0.40` | BM25 weight |
| `MEMORY_WEIGHT_RECENCY` | `0.25` | freshness weight |
| `MEMORY_WEIGHT_IMPORTANCE` | `0.15` | importance weight |
| `MEMORY_PINNED_BOOST` | `0.35` | additive score for pinned rows |
| `MEMORY_CAMPAIGN_BOOST` | `0.10` | additive score for in-scope campaign rows |
| `MEMORY_PHRASE_BOOST` | `0.10` | additive score for an exact phrase match |
| `MEMORY_MIN_RELEVANCE` | `0.12` | drop rows below this lexical score (unless pinned) |
| `MEMORY_DEDUPE_ENABLED` | `true` | update-in-place on duplicate content |
| `MEMORY_DEFAULT_TTL_DAYS` | `180` | TTL when a kind declares none |
| `MEMORY_MAX_ITEMS_PER_KIND` | `500` | per-kind ceiling before pruning |
| `MEMORY_ITEM_MAX_TOKENS` | `120` | per-item cap ceiling |
| `MEMORY_SWEEP_ON_WRITE` | `true` | sweep expired rows on write |

Weights are normalised by their sum, so only ratios matter. The removed
variables — `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`,
`VECTOR_BACKEND` — are no longer read; a deployment with them set boots fine and
ignores them.

---

## 9. Why the embeddings were removed

The old design stored a `VECTOR(64)` per row and ranked with cosine similarity.
Three things were wrong with it, and only one of them is about quality:

1. **There was no embedding model.** `EMBEDDING_PROVIDER` defaulted to `mock`,
   which produced a deterministic *hashed bag of words*. Cosine similarity of a
   hashed bag of words is a worse keyword matcher than BM25 (no IDF, no length
   normalisation) wearing a vector's clothes. Switching to a real provider meant
   a network call per write, a per-provider dimension, and a backfill of every
   row whenever the model changed — the `POST /memory/reindex` endpoint existed
   only to pay that bill.
2. **It made the table unportable.** The column was a pgvector type; the local
   SQLite database had no such type; the initial migration ran
   `CREATE EXTENSION vector`, which fails on SQLite and on a Postgres role
   without extension privileges. Two storage backends and two migration paths
   had to be reasoned about for a feature that was never on in production.
3. **It hid a bug instead of exposing it.** Six kinds were written and never
   retrieved (section 3). With vectors this looked like "the model needs better
   embeddings"; without them it is a one-line filter that no longer matches.

What replaced it is deterministic, inspectable, testable offline, has no cold
start, no per-write cost, and no reindex step. For short, dense, bilingual,
kind-tagged marketing memory — where "does this mention Cairo or القاهرة" is the
question actually being asked — BM25 plus recency plus kind affinity retrieves at
least as well. If semantic recall is ever needed for long-form documents, the
answer is a **separate** document index, not a vector column bolted onto the
memory table.

---

## 10. Operations

```bash
# what would expiry/pruning archive, per kind (dry run, owner only)
curl -X POST /api/v1/memory/prune -d '{"dry_run": true}'

# apply it
curl -X POST /api/v1/memory/prune -d '{"dry_run": false}'

# inspect what expired (archived rows are kept, not deleted)
curl '/api/v1/memory/items?include_archived=true'

# why did this row rank here?
curl '/api/v1/memory/items?q=خصم' # each item carries score + reason
```

Schema changes ship in `app/db/migrations/versions/b3f1c2d4e5a6_brand_memory_lifecycle.py`,
which drops the leftover `embedding` column, adds the lifecycle columns and the
four composite indexes, and is idempotent (it inspects the live schema first, so
it is a no-op on a database built by `create_all()`).

**Signals to watch** (`logs/`):

| log line | healthy | response |
|---|---|---|
| `memory_step_failed` | absent | read `step`; profile/identity failures are isolated |
| `ltm_consolidation_skipped_no_new_messages` | frequent | normal — the watermark is doing its job |
| `memory_swept_expired` with a growing count | steady | check a producer is not writing with the wrong kind |
| `brand_context` `truncated=true` | occasional | raise `MEMORY_CONTEXT_TOKEN_CAP` or trim the plan |

**When memory looks wrong**, in the order worth checking: (1) is the row there
and is its `kind` correct? (2) does it match the query lexically — Arabic
normalisation means `الإعلانات` should match `اعلانات`; (3) is it expired or
archived? (4) is `min_relevance` too high for a short query? (5) is the budget
already consumed by identity facts?

---

## 11. Testing

`tests/test_memory_layers.py` covers the properties this document claims:

* the registry is total (every kind has a spec; no writer uses an unknown kind);
* Arabic normalisation folds the variants and strips clitics;
* BM25 orders an exact match above a partial one, and returns 0 for no overlap;
* recency decays by half per half-life;
* pinned rows survive `min_relevance`; unpinned low-relevance rows do not;
* dedupe updates in place; TTL stamps `expires_at` and the sweep archives it;
* `upsert_singleton` keeps exactly one plan row;
* assembly respects the token cap and puts identity before episodic;
* cross-tenant reads and writes return nothing / 404;
* the two settings classes expose identical memory knobs (no drift).
