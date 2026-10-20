"""Discover the action tools from the MCP server(s) at startup.

Whatever tools the configured server exposes become agent tools, with the name, description and
JSON schema the server publishes. Nothing here (or in the agent) lists tool names, so adding a
tool to the server is enough for the agent to start using it.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from pm_assistant.config import REPO_ROOT, Settings, load_settings


def mcp_connections(settings: Settings) -> dict[str, dict[str, Any]]:
    """stdio connection to the action server. MCP stdio does not inherit the parent environment,
    so the store location is passed explicitly; a read timeout keeps a stuck server from
    hanging the conversation."""
    command, *args = settings.mcp_server_cmd
    return {
        "pm_actions": {
            "transport": "stdio",
            "command": command,
            "args": args,
            "cwd": str(REPO_ROOT),
            "env": {
                "PM_DB_PATH": str(settings.db_path),
                "PM_SEED_DIR": str(settings.seed_dir),
                "PYTHONIOENCODING": "utf-8",
            },
            "session_kwargs": {"read_timeout_seconds": timedelta(seconds=settings.mcp_timeout_s)},
        }
    }


DISCOVERY_ATTEMPTS = 3


async def load_action_tools(settings: Settings | None = None) -> list[BaseTool]:
    """List the server's tools. Retried: a stdio session can fail transiently on start/close
    (e.g. under load on Windows), and one bad spawn should not take the whole agent down."""
    settings = settings or load_settings()
    client = MultiServerMCPClient(mcp_connections(settings))
    for attempt in range(1, DISCOVERY_ATTEMPTS):
        with contextlib.suppress(Exception):
            return await client.get_tools()
        await asyncio.sleep(0.5 * attempt)
    return await client.get_tools()
