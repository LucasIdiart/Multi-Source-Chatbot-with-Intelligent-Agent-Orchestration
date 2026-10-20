"""An MCP server whose only tool never answers (simulates a stuck action server)."""

import time

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hanging", log_level="WARNING")


@mcp.tool()
def archive_project(project: str) -> str:
    """Archive a project."""
    time.sleep(3600)
    return "never"


if __name__ == "__main__":
    mcp.run()
