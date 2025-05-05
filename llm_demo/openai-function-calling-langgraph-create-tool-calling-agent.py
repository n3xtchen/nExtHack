#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""
使用 langgraph react agent 调用 functions

参考 
- https://python.langchain.com/docs/how_to/migrate_agent/
- https://blog.csdn.net/qq_33431368/article/details/141183890
"""

# 让 LLM 生成要调度的代码

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4.1", model_provider="openai")

# 工具定义和绑定
from langchain_core.tools import tool

@tool
def get_weather(latitude:float, longitude:float):
    """Get current temperature for provided coordinates in celsius.

    Args:
        latitude: first number
        longitude: second number
    """
    # response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    # data = response.json()
    # return data['current']['temperature_2m']
    return 22.2

tools = [get_weather]


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

from langchain.agents import AgentExecutor, create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools)

query = "What's the weather like in Paris today?"
for step in agent_executor.stream({"input": query}):
    print(step)
