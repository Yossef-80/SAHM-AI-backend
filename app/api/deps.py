"""Shared API dependencies."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.llm.base import LLMClient
from app.core.llm.router import get_llm
from app.db.session import get_session

SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_llm_client() -> AsyncIterator[LLMClient]:
    """The strong-tier client for request-scoped use."""
    yield get_llm("strong")


LLMDep = Annotated[LLMClient, Depends(get_llm_client)]


def request_id_of(request: Request) -> str:
    """The request id set by middleware, or a fresh one."""
    return getattr(request.state, "request_id", None) or f"req_{id(request):x}"


RequestIdDep = Annotated[str, Depends(request_id_of)]
