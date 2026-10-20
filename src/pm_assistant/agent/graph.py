"""The top-level agent: one LLM that decides, by native tool calling, which tool (if any) to use.

There is no router: the graph is the standard model <-> tools loop. The model sees one read-only
knowledge tool (the RAG + KG graph) and every action tool discovered from the MCP server.
Two middlewares wrap the existing nodes (they add no node and make no routing decision):

- tool calls are made one at a time, so a question about a change is answered after the change;
- a failing tool (crashed or stuck action server, ...) becomes an error message to the model
  instead of an exception that aborts the turn.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolErrorMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from pm_assistant.config import load_settings, make_chat_model
from pm_assistant.info.tool import default_info_tool

from .mcp_tools import load_action_tools
from .prompts import SYSTEM_PROMPT


class SequentialToolCalls(AgentMiddleware):
    """Ask the model for at most one tool call per step (no parallel tool use)."""

    def wrap_model_call(self, request, handler):
        return handler(self._sequential(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._sequential(request))

    @staticmethod
    def _sequential(request):
        return request.override(
            model_settings={**request.model_settings, "parallel_tool_calls": False}
        )


def root_cause(exc: BaseException) -> BaseException:
    """The innermost exception of an (anyio) exception group, otherwise the exception itself."""
    try:
        return root_cause(exc.exceptions[0])
    except AttributeError:
        return exc


def report_tool_error(exc: Exception, request: Any) -> str:
    cause = root_cause(exc)
    return f"Tool '{request.tool_call['name']}' failed: {type(cause).__name__}: {cause}"


def build_agent(*, model: BaseChatModel, tools: Sequence[BaseTool]) -> CompiledStateGraph:
    return create_agent(
        model,
        list(tools),
        system_prompt=SYSTEM_PROMPT,
        middleware=[SequentialToolCalls(), ToolErrorMiddleware(report_tool_error)],
        name="pm_agent",
    )


async def _build_graph(settings) -> CompiledStateGraph:
    actions, knowledge = await asyncio.gather(
        load_action_tools(settings), asyncio.to_thread(default_info_tool, settings)
    )
    return build_agent(model=make_chat_model(settings.model), tools=[knowledge, *actions])


_GRAPHS: dict[str, CompiledStateGraph] = {}


async def make_graph() -> CompiledStateGraph:
    """Graph factory (used by langgraph.json, the CLI, Gradio and the optional MCP front end).

    Action tools are discovered from the MCP server once per process and configuration (a new
    server tool is picked up on the next start). The server also creates the store on its first
    start, so this process only ever reads it.
    """
    settings = load_settings()
    key = repr(settings)
    return _GRAPHS.get(key) or _GRAPHS.setdefault(key, await _build_graph(settings))


async def chat_turn(graph: Any, history: list, text: str) -> tuple[list, str]:
    """Send one user message with the conversation so far; return the new history and reply."""
    result = await graph.ainvoke({"messages": [*history, HumanMessage(text)]})
    messages = result["messages"]
    return messages, str(messages[-1].text)
