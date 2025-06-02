#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""
使用 langchain 调用 functions

参考 https://python.langchain.com/docs/how_to/tool_results_pass_to_model/
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
llm_with_tools = llm.bind_tools(tools)

# 工具选择
from langchain_core.messages import HumanMessage

query = "What's the weather like in Paris today?"

messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(query)

messages.append(ai_msg)

# 工具调用
for tool_call in ai_msg.tool_calls:
    selected_tool = {"get_weather": get_weather}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

for msg in messages:
    print(msg)

# 最终结果<
result = llm_with_tools.invoke(messages)

print(result)
