#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""

"""

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
        weather_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what is the weather in Paris today?"}]}
        )
        print(weather_response)


if __name__ == "__main__":
    asyncio.run(main())
