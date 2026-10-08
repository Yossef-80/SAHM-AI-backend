# Git Workflow & Best Practices for Tool Developers

Welcome! All new tool development happens under the `tools-development` branch (or feature branches branching off from it). 

Follow this guide to easily access the branch, write your tools, and submit your code without breaking the main backend.

---

## 1. How to Access the Branch

If you haven't already, clone the repository to your local machine:
```bash
git clone https://github.com/Yossef-80/SAHM-AI-backend.git
cd SAHM-AI-backend
```

Fetch all the latest branches from the remote repository:
```bash
git fetch origin
```

Switch to the tools development branch:
```bash
git checkout tools-development
```

Make sure your local branch is up to date:
```bash
git pull origin tools-development
```

---

## 2. Best Practices for Developing Your Tool

When building a new tool (following the instructions in `DEVELOPER_TOOL_GUIDE.md`), do **not** code directly on `tools-development`. Create your own feature branch!

### Step A: Create a Feature Branch
Name it something obvious, like `feat/facebook-scraper-tool`:
```bash
git checkout -b feat/your-tool-name
```

### Step B: Build and Test Locally
1. Add your tool file in `app/ai/tools/`.
2. Add your tool to the target agent's graph in `app/ai/agents/<agent_name>/graph.py`.
3. Test it locally using `uv run python -m uvicorn app.main:app`.

### Step C: Update the Environment Templates
If your tool requires a new API key (e.g., `OPENAI_API_KEY`), you **must**:
1. Add the variable to `app/ai/settings.py`.
2. Add a blank placeholder to `.env.example` (e.g., `OPENAI_API_KEY=`).
*CRITICAL: Never commit your actual `.env` file containing real API keys!*

---

## 3. How to Push Your Tool

Once your tool is working perfectly, it's time to push it to GitHub.

Stage your changed files:
```bash
git add app/ai/tools/your_tool.py
git add app/ai/agents/nour/graph.py
git add app/ai/settings.py
git add .env.example
```

Commit your changes with a clear message:
```bash
git commit -m "feat: add facebook ad scraper tool for Nour"
```

Push your feature branch to GitHub:
```bash
git push -u origin feat/your-tool-name
```

---

## 4. Submitting for Review (Pull Requests)

After you push your branch:
1. Go to the GitHub repository in your web browser.
2. Click **"Compare & pull request"**.
3. Set the "base" branch to `tools-development` (NOT `main`!).
4. Add a brief description of what your tool does and which API keys the Tech Lead needs to add to their `.env` to test it.

Once the Tech Lead reviews and tests your tool, they will merge it into `tools-development`, and eventually into the live `main` branch.
