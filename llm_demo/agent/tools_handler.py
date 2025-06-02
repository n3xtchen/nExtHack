#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""

"""

from langchain.tools import Tool
from pydantic import BaseModel, ConfigDict, Field

from typing import Any, Dict
from langchain.callbacks.base import BaseCallbackHandler

from agent.cache import CacheHandler

class CacheTools(BaseModel):
    """Default tools to hit the cache."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Hit Cache"
    cache_handler: CacheHandler = Field(
        description="Cache Handler for the crew",
        default=CacheHandler(),
    )

    def tool(self):
        return Tool.from_function(
            func=self.hit_cache,
            name=self.name,
            description="Reads directly from the cache",
        )

    def hit_cache(self, key):
        split = key.split("tool:")
        tool = split[1].split("|input:")[0].strip()
        tool_input = split[1].split("|input:")[1].strip()
        return self.cache_handler.read(tool, tool_input)

class ToolsHandler(BaseCallbackHandler):
    """Callback handler for tool usage."""

    last_used_tool: Dict[str, Any] = {}
    cache: CacheHandler

    def __init__(self, cache: CacheHandler = None, **kwargs: Any):
        """Initialize the callback handler."""
        self.cache = cache
        super().__init__(**kwargs)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        name = serialized.get("name")
        if name not in ["invalid_tool", "_Exception"]:
            tools_usage = {
                "tool": name,
                "input": input_str,
            }
            self.last_used_tool = tools_usage

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        if (
            "is not a valid tool" not in output
            and "Invalid or incomplete response" not in output
            and "Invalid Format" not in output
        ):
            if self.last_used_tool["tool"] != CacheTools().name:
                self.cache.add(
                    tool=self.last_used_tool["tool"],
                    input=self.last_used_tool["input"],
                    output=output,
                )
