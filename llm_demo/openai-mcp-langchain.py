#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""
使用 langraph react agent 调用 mcp

参考：
- https://medium.com/ideaboxai/model-context-protocol-with-langchain-agent-client-f9562d2790b3
- https://github.com/langchain-ai/langchain-mcp-adapters
"""

import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# 让 LLM 生成要调度的代码
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4.1", model_provider="openai")

async def main():
    async with MultiServerMCPClient({
        "weather": {
            "command": "python",
            "args": ["./mcp-stdio-get-weather.py"],
            "transport": "stdio"
        }
    }) as client:
        agent = create_react_agent(
            llm,
            client.get_tools()
        )
        query = "What's the weather like in Paris today?"
        # weather_response = await agent.ainvoke(
        #     {"messages": [{"role": "user", "content": query}]}
        # )
        # print(weather_response)
        
        inputs = {"messages": [("user", query)]}
        async for s in agent.astream(inputs):
            print(s)
            # message = s["mssages"][-1]
            # if isinstance(message, tuple):
            #     print(message)
            # else:
            #     message.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())
