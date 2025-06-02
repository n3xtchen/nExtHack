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
def get_weather(latitude:float, longitude:float) -> float:
    """Get current temperature for provided coordinates in celsius.

    参数:
        latitude: 纬度，必须是浮点数，例如48.8566
        longitude: 经度，必须是浮点数，例如2.3522

    示例: get_weather(latitude=48.8566, longitude=2.3522)
    """
    # response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    # data = response.json()
    # return data['current']['temperature_2m']
    return 22.2

tools = [get_weather, ]

from langchain_core.prompts import PromptTemplate

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action(in dictionary format)
If there's an error, re run the tool with corrected input\
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

from langchain.agents import AgentExecutor, create_react_agent

# action input 解析

import re
from typing import Union
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish

FINAL_ANSWER_ACTION = "Final Answer:"

class ReActActionInputParser(ReActSingleInputOutputParser):
    """
    将 action input 解析成字典
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            tool_input = tool_input.strip('"')

            print("xx", action, "y", tool_input, eval(tool_input))
            return AgentAction(action, eval(tool_input), text)

        return super().parse(text)



agent = create_react_agent(llm, tools, prompt=prompt, output_parser=ReActActionInputParser())
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = "What's the weather like in Paris today?"
for step in agent_executor.stream({"input": query}):
    print(step)
