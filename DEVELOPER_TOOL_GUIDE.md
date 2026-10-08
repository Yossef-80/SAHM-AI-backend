# Developer Guide: Building AI Tools for SAHM OS

Welcome to the SAHM AI Tool Development team! We recently upgraded our entire AI engine to a modern **LangChain & LangGraph** architecture. 

This guide will explain exactly how you should build tools, how they interact with our database, how the UI displays them, and how to submit your work so it can be merged and tested instantly.

---

## 1. The Architecture (What you need to know)
* **The Agents:** Our AI agents (Nour, Omar, Layla, etc.) are built using LangGraph. They are located in `app/ai/agents/`.
* **The UI Bridge:** The React frontend *does not* read raw tool outputs. It only reads strict JSON UI Blocks (`text`, `findings`, `action_card`). 
* **Your Tools:** Tools are "weapons" you give to the agents. If you build a tool that scrapes Facebook, the agent uses your tool, reads the data, and then *the agent* formats it for the frontend. You do not need to worry about React UI Blocks inside your tool!

## 2. Types of Tools

There are two types of tools you can build. You must understand which one you are building:

### Type A: "Read-Only" Tools (Data Gatherers)
These tools go out to the internet, fetch data, and return it to the AI as a string or JSON. 
* **Database:** They **do not** write to the database.
* **UI Display:** The UI never sees this data directly. The AI reads it, thinks about it, and writes a summary to the user.
* *Example:* A tool that fetches a competitor's SEO ranking.

### Type B: "Action" Tools (Database Writers)
These tools take final decisions from the AI and save them as tangible entities in our system.
* **Database:** They **must** write to tables like `Campaign`, `Insight`, `Creative`, or `Recommendation` using SQLAlchemy.
* **UI Display:** Once saved to the DB, the frontend will automatically fetch them from standard REST APIs (e.g., the Campaigns dashboard). 
* *Example:* A tool called `create_ad_campaign` that saves a new marketing strategy to the database.

---

## 3. How to Build a Tool (Step-by-Step)

All new tools should be created in the `app/ai/tools/` directory.

### Step 1: Define the Input Schema (Pydantic)
You must define exactly what arguments the AI needs to pass to your tool. Use descriptions so the AI knows what to do!
```python
from pydantic import BaseModel, Field

class FetchCompetitorAdsInput(BaseModel):
    competitor_name: str = Field(description="The name of the competitor brand.")
    platform: str = Field(description="The platform to search (e.g., 'facebook', 'tiktok')")
```

### Step 2: Write the Tool Function
Use the `@tool` decorator from LangChain.
```python
from langchain_core.tools import tool
import httpx

@tool(args_schema=FetchCompetitorAdsInput)
async def fetch_competitor_ads(competitor_name: str, platform: str) -> str:
    \"\"\"Fetches the currently active ads for a given competitor.\"\"\"
    # Put your API logic here!
    # return a string or JSON dictionary
    return f"Found 3 active ads for {competitor_name} on {platform}."
```

### Step 3 (For Type B Tools): Injecting the Database
If your tool needs to save data (like an `Insight`), require `db: AsyncSession` in your function signature. The system will automatically inject it.
```python
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Insight
import uuid

@tool(args_schema=SaveInsightInput)
async def save_insight(title: str, detail: str, db: AsyncSession) -> str:
    \"\"\"Saves a market insight.\"\"\"
    new_insight = Insight(id=f"ins_{uuid.uuid4().hex[:12]}", title=title, detail=detail)
    db.add(new_insight)
    await db.commit()
    return "Insight saved successfully."
```

---

## 4. Delivery Instructions (How to Submit Your PR)

When you finish building your tool, you must follow these exact steps so the Tech Lead can pull your branch and test it instantly:

1. **Create the Tool File:** Place your code in `app/ai/tools/<your_tool_name>.py`.
2. **Update Settings:** If your tool requires a new API key (e.g., `FACEBOOK_API_KEY`), add it to `app/ai/settings.py` inside the `AISettings` class, AND add an empty placeholder to `.env.example`.
3. **Register the Tool:** Go to the agent that should use your tool (e.g., `app/ai/agents/nour/graph.py`). Import your tool and add it to the `tools = [...]` list.
4. **Push:** Commit your code and push it to your assigned branch.

*Do not commit your personal `.env` file with real API keys!*
