"""Providers: the ONLY place that knows about concrete backends.

    providers/
    ├── _loader.py   # enum -> module lazy import (shared)
    ├── llm/         # get_llm()            : ollama.py | openai.py | anthropic.py | mock.py
    ├── search/      # get_search_provider(): tavily.py | mock.py | none.py
    └── db/          # get_checkpointer() / get_store(): memory.py | sqlite.py | postgres.py

The enum value in ``.env`` decides which module is imported. Tools and agents
only ever call the ``get_*`` factories, so swapping a backend never touches them.
"""
