"""Optional: expose the whole assistant as an MCP server for MCP clients (VS Code, Copilot, ...).

The client sends a message; the agent inside still decides on its own whether to query the
knowledge base or call an action tool on the separate action server.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

server = FastMCP(
    "pm-assistant",
    instructions="Project-management assistant for Northwind Labs. Send it questions or requests.",
    log_level="WARNING",
)


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
async def ask(
    message: Annotated[str, Field(description="A question or a request, in natural language.")],
) -> str:
    """Ask the PM assistant a question (answered from documents + knowledge graph) or request
    a change (executed through the action MCP server). Each call is a new conversation."""
    from pm_assistant.agent.graph import chat_turn, make_graph

    _, reply = await chat_turn(await make_graph(), [], message)
    return reply


def main() -> None:
    # Build the agent before serving: importing the info stack (numpy, via rank_bm25) while the
    # stdio transport's reader thread is blocked on stdin deadlocks on Windows.
    from pm_assistant.agent.graph import make_graph

    asyncio.run(make_graph())
    server.run()


if __name__ == "__main__":
    main()
