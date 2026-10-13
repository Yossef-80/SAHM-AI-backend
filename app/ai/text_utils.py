"""Robust text extraction and cleansing utilities across AI modules.

Prevents serialized Python dict/list leaks (e.g. from LangChain Google GenAI
parts containing thought/signatures) into user-facing chat blocks, memory, or reports.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


def clean_raw_string(s: str) -> str:
    """Extract clean readable text from serialized Python dicts, lists, or JSON strings."""
    if not s:
        return ""
    s = s.strip()

    # Check if the string itself is a serialized Python dict / list of dicts
    if (
        s.startswith("{'type'")
        or s.startswith("[{'type'")
        or s.startswith('{"type"')
        or s.startswith('[{"type"')
        or (s.startswith("[{") and "'text':" in s)
        or (s.startswith("[{") and '"text":' in s)
    ):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                texts = [
                    item.get("text", "")
                    for item in parsed
                    if isinstance(item, Mapping) and "text" in item
                ]
                if texts:
                    s = "\n".join(str(t) for t in texts if t)
            elif isinstance(parsed, Mapping) and "text" in parsed:
                s = str(parsed["text"])
        except Exception:
            try:
                data = json.loads(s)
                if isinstance(data, list):
                    texts = [
                        item.get("text", "")
                        for item in data
                        if isinstance(item, Mapping) and "text" in item
                    ]
                    if texts:
                        s = "\n".join(str(t) for t in texts if t)
                elif isinstance(data, Mapping) and "text" in data:
                    s = str(data["text"])
            except Exception:
                pass

    # Fallback regex if text remains encapsulated in {'type': 'text', 'text': ...}
    if "{'type': 'text'" in s or '{"type": "text"' in s or ("'extras'" in s and "'text'" in s):
        matches = re.findall(
            r"['\"]text['\"]\s*:\s*['\"](.*?)['\"](?:\s*,\s*['\"][a-zA-Z_]+['\"]|\s*})",
            s,
            re.DOTALL,
        )
        if matches:
            s = "\n".join(matches)

    # Clean replacement characters or leftover byte anomalies
    s = s.replace("\ufffd", "").replace("\\n", "\n")
    return s.strip()


def extract_clean_text(content: Any) -> str:
    """Extract strictly plain markdown text from any LangChain LLM content structure."""
    if content is None:
        return ""
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(extract_clean_text(item["content"]))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            elif hasattr(item, "content"):
                parts.append(extract_clean_text(getattr(item, "content")))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        extracted = "\n".join([p for p in parts if str(p).strip()])
        return clean_raw_string(extracted)
    if isinstance(content, Mapping):
        if "text" in content:
            return clean_raw_string(str(content["text"]))
        if "content" in content:
            return extract_clean_text(content["content"])
        return clean_raw_string(str(content))
    return clean_raw_string(str(content))
