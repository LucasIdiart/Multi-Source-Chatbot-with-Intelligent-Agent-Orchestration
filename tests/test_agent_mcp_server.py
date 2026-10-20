"""The optional MCP front end answers over real stdio (it used to deadlock on Windows)."""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pm_assistant.config import REPO_ROOT


def test_ask_returns_over_stdio_within_a_bounded_time(settings):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "pm_assistant.agent_mcp_server"],
        cwd=str(REPO_ROOT),
        # No API key on purpose: the model call fails fast, so the call must come back as an
        # MCP error instead of hanging.
        env={"PM_DB_PATH": str(settings.db_path), "PYTHONIOENCODING": "utf-8"},
    )

    async def run():
        async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await asyncio.wait_for(
                session.call_tool("ask", {"message": "Who leads PRJ-01?"}), timeout=120
            )
            return tools, result

    tools, result = asyncio.run(run())
    assert [t.name for t in tools.tools] == ["ask"]
    assert result.isError is True
    assert result.content
