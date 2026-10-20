"""Gradio chat (stock ChatInterface, no custom frontend). Passive: forwards messages.

Each reply is preceded by collapsible "tool" bubbles (Gradio's built-in message metadata) that
show which tool the agent chose and with which arguments — the knowledge base or an MCP action —
so the route the agent picked is visible. They are display-only and are not sent back as history.

Run with the `pm-gradio` script (ui/ has no `if __name__ == "__main__"` guard: it is
AST-checked to contain no conditionals at all).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import gradio as gr
from langchain_core.messages import ToolMessage, convert_to_messages

from pm_assistant.agent.graph import chat_turn

GraphFactory = Callable[[], Awaitable[Any]]
STATUS_SUFFIX = {"error": " — failed"}


def conversation(history: list[dict]) -> list:
    """Gradio history without the display-only tool bubbles (they carry a metadata title)."""
    return convert_to_messages(
        [{"role": t["role"], "content": t["content"]} for t in history
         if not (t.get("metadata") or {}).get("title")]
    )


def tool_bubbles(new_messages: list) -> list[gr.ChatMessage]:
    """One collapsible bubble per tool call the agent made in this turn."""
    status = {m.tool_call_id: m.status for m in new_messages if isinstance(m, ToolMessage)}
    return [
        gr.ChatMessage(
            role="assistant",
            content=json.dumps(call["args"], ensure_ascii=False, indent=2),
            metadata={"title": f"🔧 {call['name']}"
                      f"{STATUS_SUFFIX.get(status.get(call['id'], 'success'), '')}"},
        )
        for m in new_messages
        for call in getattr(m, "tool_calls", None) or []
    ]


def make_respond(graph_factory: GraphFactory):
    async def respond(message: str, history: list[dict]) -> list[gr.ChatMessage]:
        sent = conversation(history)
        messages, reply = await chat_turn(await graph_factory(), sent, message)
        new = messages[len(sent) + 1:]
        return [*tool_bubbles(new), gr.ChatMessage(role="assistant", content=reply)]

    return respond


def build_demo(graph_factory: GraphFactory | None = None) -> gr.ChatInterface:
    from pm_assistant.agent.graph import make_graph

    return gr.ChatInterface(
        make_respond(graph_factory or make_graph),
        title="Northwind PM Assistant",
        description="Ask about people, projects and policies, or ask it to assign, create or "
        "update work. The agent decides whether to search the knowledge base or act — the 🔧 "
        "boxes show the tool it chose.",
    )


def main() -> None:
    build_demo().launch(show_error=True)  # surface the cause (missing key, overload, ...)
