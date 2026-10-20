"""An MCP server whose only tool kills the process mid-call (simulates a transport failure)."""

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("crashing", log_level="WARNING")


@mcp.tool()
def archive_project(project: str) -> str:
    """Archive a project."""
    os._exit(1)


if __name__ == "__main__":
    mcp.run()
