import json
import ast
from typing import Any, Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from app.ai.providers.llm import get_llm
from app.ai.providers.search import get_search_provider
from app.ai.settings import get_ai_settings

def clean_raw_string(s: str) -> str:
    s = s.strip()
    if (s.startswith("{'type'") or s.startswith("[{'type'") or 
        s.startswith('{"type"') or s.startswith('[{"type"')):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                texts = [item.get("text", "") for item in parsed if isinstance(item, dict)]
                if texts:
                    s = "\n".join(texts)
            elif isinstance(parsed, dict) and "text" in parsed:
                s = str(parsed["text"])
        except Exception:
            try:
                data = json.loads(s)
                if isinstance(data, list):
                    texts = [item.get("text", "") for item in data if isinstance(item, dict)]
                    if texts:
                        s = "\n".join(texts)
                elif isinstance(data, dict) and "text" in data:
                    s = str(data["text"])
            except Exception:
                pass
    s = s.replace("\ufffd", "")
    return s.strip()

def extract_clean_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        extracted = "\n".join([p for p in parts if p.strip()])
        return clean_raw_string(extracted)
    if isinstance(content, dict):
        if "text" in content:
            return clean_raw_string(str(content["text"]))
        return clean_raw_string(str(content))
    return clean_raw_string(str(content))

print("Testing cleaner on corrupted DB sample:")
corrupted = "{'type': 'text', 'text': '### Overview & Positioning\\n* **Brand Heritage:** Founded in 1995'}"
cleaned = extract_clean_text(corrupted)
print("Cleaned:")
print(cleaned)
assert cleaned.startswith("### Overview"), "Failed to clean corrupted dict!"
print("Assertion passed!")
