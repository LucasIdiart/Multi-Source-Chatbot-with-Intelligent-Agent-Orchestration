"""Gradio chat (stock ChatInterface, no custom frontend). Passive: forwards messages.

Run with the `pm-gradio` script (ui/ has no `if __name__ == "__main__"` guard: it is
AST-checked to contain no conditionals at all).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import gradio as gr
from langchain_core.messages import convert_to_messages

from pm_assistant.agent.graph import chat_turn

GraphFactory = Callable[[], Awaitable[Any]]


def make_respond(graph_factory: GraphFactory):
    async def respond(message: str, history: list[dict]) -> str:
        # Gradio history is OpenAI-style {"role", "content"} dicts; LangChain converts them.
        _, reply = await chat_turn(await graph_factory(), convert_to_messages(history), message)
        return reply

    return respond


def build_demo(graph_factory: GraphFactory | None = None) -> gr.ChatInterface:
    from pm_assistant.agent.graph import make_graph

    return gr.ChatInterface(
        make_respond(graph_factory or make_graph),
        title="Northwind PM Assistant",
        description="Ask about people, projects and policies, or ask it to assign, create or "
        "update work. The agent decides whether to search the knowledge base or act.",
    )


def main() -> None:
    build_demo().launch(show_error=True)  # surface the cause (missing key, overload, ...)
