"""Offline test setup: no API key, a fresh seeded database per test, default MCP server."""

from __future__ import annotations

import ast
import importlib.util
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import fake_synth  # noqa: E402
from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402

import pm_assistant  # noqa: E402
from pm_assistant.agent.mcp_tools import mcp_connections  # noqa: E402
from pm_assistant.config import load_settings  # noqa: E402
from pm_assistant.info.subgraph import default_info_subgraph  # noqa: E402
from pm_assistant.info.tool import make_info_tool  # noqa: E402
from pm_assistant.store.seed import ensure_db  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
PKG = Path(pm_assistant.__file__).parent


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    """Isolated settings: no API key, no developer overrides, a fresh seeded store."""
    for var in [v for v in os.environ if v.startswith("PM_")] + ["ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PM_DB_PATH", str(tmp_path / "pm.db"))
    settings = load_settings()
    ensure_db(settings.db_path, settings.seed_dir)
    return settings


@pytest.fixture
def db(settings):
    """Query helper over the test database."""

    def query(sql: str, *params):
        conn = sqlite3.connect(settings.db_path)
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    return query


async def with_mcp_session(settings, fn):
    """Run fn(session) against the action server, launched exactly as the agent launches it."""
    async with MultiServerMCPClient(mcp_connections(settings)).session("pm_actions") as session:
        return await fn(session)


def info_tool(settings, synth=None):
    """The production knowledge tool with an offline synthesizer."""
    return make_info_tool(default_info_subgraph(settings, synth_model=synth or fake_synth()))


def imports_of(package: str) -> set[str]:
    """Every module a package imports, including `from x import y` submodules, resolved."""
    found: set[str] = set()
    for path in (PKG / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found |= {a.name for a in node.names}
            if isinstance(node, ast.ImportFrom):
                mod = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), f"pm_assistant.{package}"
                ) if node.level else node.module
                found |= {mod, *(f"{mod}.{a.name}" for a in node.names)}
    return found


def violates(deps: set[str], forbidden: tuple[str, ...]) -> list[str]:
    return sorted(d for d in deps for f in forbidden if d == f or d.startswith(f + "."))
