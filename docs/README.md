# Documentation

> Sahm is an autonomous marketing department for small businesses: five specialist
> AI agents, one shared brand memory, in Arabic and English.
> Start at [`../README.md`](../README.md) for the one-page version; come back here
> when you need depth.

---

## Read by goal

### "I need to explain Sahm to a non-technical audience"

| Document | What it gives you |
| :--- | :--- |
| [`NON_TECHNICAL_PRESENTATION_GUIDE.md`](NON_TECHNICAL_PRESENTATION_GUIDE.md) | The business story, an 8-slide deck, and a demo script with no code in it. |
| [`PRODUCT_OVERVIEW.md`](PRODUCT_OVERVIEW.md) | What the product is, the problem it solves, who it is for, and what "done" looks like. |
| [`PRESENTATION_AND_DEMO_GUIDE.md`](PRESENTATION_AND_DEMO_GUIDE.md) | The 10-slide pitch, speaking scripts, live demo steps, and answers to the awkward questions. |

### "I need to understand every feature"

[`PROJECT_FEATURES_AND_CAPABILITIES.md`](PROJECT_FEATURES_AND_CAPABILITIES.md) —
feature by feature: what the user sees, what the backend does, which tables are
touched, which agent owns it.

### "I am a reviewer reading the code"

| Document | What it gives you |
| :--- | :--- |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design, the data-flow diagram, the schema, the full API table, security model, extension playbooks, runbook. |
| [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md) | How brand memory works: 13 kinds, lifecycle, BM25 + recency + affinity ranking, token budgets, and why there is no vector store. |
| [`ORCHESTRATOR.md`](ORCHESTRATOR.md) | Routing, the specialist graphs, the UI-block bridge, the background workers. |
| [`LANGUAGE_POLICY.md`](LANGUAGE_POLICY.md) | The bilingual contract: which language every answer uses, and where that is decided. |
| [`DATABASE_GUIDE.md`](DATABASE_GUIDE.md) | Tables, relationships, conventions for writing to them. |
| [`../README.md`](../README.md) §4 | The design decisions worth arguing about. |

### "I am going to change the code"

| Document | What it gives you |
| :--- | :--- |
| [`../AGENTS.md`](../AGENTS.md) | **Start here.** Task router (task → files → tests), module map with status, invariants, recipes. Built so neither a person nor an agent has to scan the tree. |
| [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) | Tool and agent internals, safe zones in the tree, how to add a tool, prompt tuning, failure modes. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) §5 | Playbooks: new tool, new provider, new migration. |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Branches, commits, PR checklist, review rules. |
| [`RUNNING_BACKEND.md`](RUNNING_BACKEND.md) | Run it locally, health checks, containers. |
| [`AD_CREATIVE_FLOW.md`](AD_CREATIVE_FLOW.md) | The competitor-ad → vision → prompt → image flow, and its safety rules. |

### "I need to operate it"

[`RUNNING_BACKEND.md`](RUNNING_BACKEND.md) and the runbook section of
[`ARCHITECTURE.md`](ARCHITECTURE.md) §6. Memory retention is configured and
audited through `POST /memory/prune` (see `MEMORY_ARCHITECTURE.md` §9).

---

## The whole set

```
AGENTS.md                                    # the code map (root, not in docs/)
docs/
├── README.md                                # you are here
│
├── ── product & storytelling ───────────────────────────────────────────
├── PRODUCT_OVERVIEW.md                      # vision, problem, scope, market
├── PROJECT_FEATURES_AND_CAPABILITIES.md     # every feature, end to end
├── NON_TECHNICAL_PRESENTATION_GUIDE.md      # business-language deck + demo
├── PRESENTATION_AND_DEMO_GUIDE.md           # technical pitch + live demo
│
├── ── engineering ──────────────────────────────────────────────────────
├── ARCHITECTURE.md                          # design, schema, API, runbook
├── ORCHESTRATOR.md                          # routing, graphs, workers
├── MEMORY_ARCHITECTURE.md                   # brand memory, in detail
├── MEMORY_VALIDATION_REPORT.md              # PM review of the memory rebuild
├── LANGUAGE_POLICY.md                       # bilingual answer contract
├── DATABASE_GUIDE.md                        # tables and conventions
├── DEVELOPER_GUIDE.md                       # tools, agents, safe zones
├── AD_CREATIVE_FLOW.md                      # competitor ad → prompt → image
└── RUNNING_BACKEND.md                       # local run, health, containers
```

---

## Conventions in this folder

- Every document states its **audience** at the top. If your audience is not
  listed, the document is probably not for you — use the tables above.
- **Code wins.** When a document and the code disagree, the code is right and the
  document is a bug. The docs are checked against the tree in the same PR that
  changes behaviour.
- Paths are **repository-relative**. No absolute paths, no `file:///` links, no
  machine-specific directories.
- One idea per section, tables over prose for anything enumerable, and an honest
  "what this does not do" wherever a limitation exists.
