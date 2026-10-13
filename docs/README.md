# 📚 SAHM AI — Documentation Portal & Knowledge Base

> Welcome to the comprehensive documentation suite for **SAHM AI (سهم)** — the autonomous Multi-Agent Marketing Operating System designed for SMBs, e-commerce brands, and agencies in Egypt and the MENA region.

---

## 🗺️ Documentation Directory Map

This folder (`backend/docs/`) contains 7 core technical, commercial, and presentation manuals designed to help you **understand every feature in the project**, **deliver non-technical and executive presentations**, **improve and extend the codebase**, and **run live software demonstrations**:

```
SAHM-AI-backend/docs/
│
├── 📖 README.md                                      # You are here: Master Portal & Reading Paths
│
├── 📢 NON_TECHNICAL_PRESENTATION_GUIDE.md           # Non-Technical Presentation, Pitch Deck & Storytelling
│                                                     # (The "AI Tool Chaos" problem, connected intelligence, 8 slides)
│
├── 🎤 PRESENTATION_AND_DEMO_GUIDE.md                 # Full Presentation, Pitch Deck & Live Demo Playbook
│                                                     # (10-slide deck, step-by-step live demo scripts, Q&A answers)
│
├── 🚀 TEAM_PROJECT_SCOPE_AND_ASSESSMENT.md          # Project Scope, Commercial Vision & State Assessment
│                                                     # (What SAHM is, problems solved, current architecture)
│
├── 🌟 PROJECT_FEATURES_AND_CAPABILITIES.md           # End-to-End Features & Capabilities Manual
│                                                     # (Deep-dive into all 10 platform feature modules)
│
├── 🛠️ TEAM_TOOLS_AND_AGENTS_GUIDE.md                 # Tools & Agents Engineering Playbook
│                                                     # (9 Research tools, LangGraph agents, Playwright, Apify, GA4)
│
├── 🧠 TEAM_BRAND_MEMORY_AND_ORCHESTRATOR_ARCHITECTURE.md # Central Brain & Cognitive Architecture
│                                                     # (4-tier memory, token bloat prevention, Orchestrator brain)
│
└── ⚙️ SYSTEM_ARCHITECTURE_AND_ENGINEERING_PLAYBOOK.md # System Design, ERD Models, APIs & Security
                                                      # (Database ERD, REST API spec, Fernet encryption, runbook)
```

---

## 🎯 Recommended Reading Paths by Goal

Choose the reading path that fits your current objective:

### Path A: "I need to give a presentation to business owners, clients, or non-technical judges"
1. Start with [**`NON_TECHNICAL_PRESENTATION_GUIDE.md`**](file:///D:/NTI-Final-Project/SAHM-AI-backend/docs/NON_TECHNICAL_PRESENTATION_GUIDE.md) for the storytelling narrative ("The AI Tool Chaos Trap" vs. "The Connected Intelligence Flywheel").
2. Use the **8-Slide Ready-to-Use Deck** with exact speaking scripts in business language.
3. Use the **Non-Technical Live Demo Script** (Ahmed's Apparel Brand story) to demonstrate Tie House research, 5-day calendar scheduling, and Egyptian ad hooks without showing a single line of code.

### Path B: "I need to understand every feature and what SAHM offers"
1. Read [**`TEAM_PROJECT_SCOPE_AND_ASSESSMENT.md`**](file:///D:/NTI-Final-Project/SAHM-AI-backend/docs/TEAM_PROJECT_SCOPE_AND_ASSESSMENT.md) for the commercial vision, target audience, and the 5 specialized AI marketers.
2. Read [**`PROJECT_FEATURES_AND_CAPABILITIES.md`**](file:///D:/NTI-Final-Project/SAHM-AI-backend/docs/PROJECT_FEATURES_AND_CAPABILITIES.md) for a detailed walkthrough of all 10 features (Research Hub, Assistant Chat, Content Calendar, Campaigns, Creatives, GA4 Analytics, and Safe Approvals).

### Path C: "I need technical depth, investor Q&A, and full pitch deck coverage"
1. Open [**`PRESENTATION_AND_DEMO_GUIDE.md`**](file:///D:/NTI-Final-Project/SAHM-AI-backend/docs/PRESENTATION_AND_DEMO_GUIDE.md).
2. Copy the **10-Slide Pitch Deck** into your presentation slides.
3. Review the **10 Q&A Questions & Answers** to prepare for challenging technical, business, and AI safety inquiries.

### Path D: "I am an engineer looking to improve or build new features"
1. Read [**`TEAM_TOOLS_AND_AGENTS_GUIDE.md`**](file:///D:/NTI-Final-Project/SAHM-AI-backend/docs/TEAM_TOOLS_AND_AGENTS_GUIDE.md) to learn how agents and tools are implemented in LangGraph.
2. Read [**`TEAM_BRAND_MEMORY_AND_ORCHESTRATOR_ARCHITECTURE.md`**](file:///D:/NTI-Final-Project/SAHM-AI-backend/docs/TEAM_BRAND_MEMORY_AND_ORCHESTRATOR_ARCHITECTURE.md) to understand the 4-tier memory hierarchy, token bloat rules, and deterministic routing.
3. Consult [**`SYSTEM_ARCHITECTURE_AND_ENGINEERING_PLAYBOOK.md`**](file:///D:/NTI-Final-Project/SAHM-AI-backend/docs/SYSTEM_ARCHITECTURE_AND_ENGINEERING_PLAYBOOK.md) for database ERDs, REST API contracts, security protocols, and step-by-step developer recipes.

---

## ⚡ Quick Summary of SAHM AI's Core Capabilities

- **Autonomous Multi-Agent Squad:** Nour (Researcher), Omar (Strategist), Layla (Creative), Karim (Analyst), and Salma (Optimizer) collaborating under a deterministic Orchestrator.
- **The Connected Intelligence Flywheel:** Search connects to Strategy; Strategy connects to Creatives; Creatives connect to the Calendar; Analytics connect to Approvals. All powered by one shared Brand Memory.
- **Real-Time Market Evidence:** Live web crawling, Google Maps reviews extraction via Playwright, Facebook Ad Library scraping via Apify, and Google Analytics 4 performance reporting.
- **Ground-Truth Entity Resolver:** Auto-corrects brand typos (e.g. *"Tiee Hose"* ➔ *"Tie House"*) and rejects fake queries before running searches.
- **Unified Brand Identity Hub:** Central living brand profile (`/brand-identity`) with color palette tokens, strategic essence (voice, tagline, archetype), target audience, SWOT analysis, and 1-click sync from chats.
- **Automated Content Calendar:** Direct multi-day social media schedule generation saved automatically into PostgreSQL.
- **Zero-Token-Bloat Memory:** Dual-layer storage (380-word reports for humans, ~100-token facts for AI prompts) reducing context window consumption by over 90%.
- **Human-in-the-Loop Safety:** Two-phase approval workflow with before/after diffs and immutable audit logging before executing ad account changes.
