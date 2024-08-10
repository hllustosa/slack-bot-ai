from __future__ import annotations

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable

from project.llm.gpt import get_selected_tool


class AnswerSlackMessageUseCase:
    def __init__(self, llm_with_tools: Runnable[LanguageModelInput, BaseMessage]):
        self.llm_with_tools = llm_with_tools

    def execute(self, query: str | list[str | dict]):
        messages = [HumanMessage(query)]
        ai_msg = self.llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        for tool_call in ai_msg.tool_calls:
            selected_tool = get_selected_tool(tool_call)
            tool_output = selected_tool.invoke(tool_call['args'])
            messages.append(
                ToolMessage(
                    tool_output, tool_call_id=tool_call['id'],
                ),
            )

        return messages
