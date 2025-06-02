#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""

"""

from typing import Any, Optional

from pydantic import BaseModel, Field, PrivateAttr

class CacheHandler:
    """Callback handler for tool usage."""

    _cache: PrivateAttr = {}

    def __init__(self):
        self._cache = {}

    def add(self, tool, input, output):
        input = input.strip()
        self._cache[f"{tool}-{input}"] = output

    def read(self, tool, input) -> Optional[str]:
        input = input.strip()
        return self._cache.get(f"{tool}-{input}")

class CacheHit(BaseModel):
    """Cache Hit Object."""

    class Config:
        arbitrary_types_allowed = True

    # Making it Any instead of AgentAction to avoind
    # pydantic v1 vs v2 incompatibility, langchain should
    # soon be updated to pydantic v2
    action: Any = Field(description="Action taken")
    cache: CacheHandler = Field(description="Cache Handler for the tool")

