#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""
使用 langgraph react agent 调用 functions

参考 https://langchain-ai.github.io/langgraph/how-tos/tool-calling/
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

from langgraph.prebuilt import ToolNode
tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)

from langgraph.graph import StateGraph, MessagesState, START, END


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()

query = "What's the weather like in Paris today?"
for chunk in app.stream(
    {"messages": [("human", query)]}, stream_mode="values"
):
    chunk["messages"][-1].pretty_print()
