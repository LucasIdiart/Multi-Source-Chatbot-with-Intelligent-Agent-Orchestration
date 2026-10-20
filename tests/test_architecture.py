"""EC2 separation + EC4 no manual routing, checked structurally on the source and the graph."""

import ast
import sqlite3

import pytest
from conftest import PKG, imports_of, violates
from fakes import scripted
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from pm_assistant.agent.graph import build_agent
from pm_assistant.agent.prompts import SYSTEM_PROMPT
from pm_assistant.store.read import connect_ro


def test_import_scanner_sees_every_import_form():
    deps = imports_of("mcp_server")
    assert "pm_assistant.store.write" in deps  # `from pm_assistant.store import write`
    assert "pm_assistant.store.seed.ensure_db" in deps


def test_action_server_never_touches_the_informational_side():
    assert not violates(imports_of("mcp_server"), ("pm_assistant.info", "pm_assistant.agent",
                                                   "pm_assistant.store.read"))


def test_informational_side_never_touches_actions_or_mcp():
    assert not violates(imports_of("info"), ("mcp", "langchain_mcp_adapters",
                                             "pm_assistant.mcp_server", "pm_assistant.store.write",
                                             "pm_assistant.store.seed", "sqlite3"))


def test_agent_and_ui_have_no_direct_data_access():
    for package in ("agent", "ui"):
        assert not violates(imports_of(package), ("sqlite3", "pm_assistant.store",
                                                  "pm_assistant.mcp_server")), package


def test_knowledge_graph_connection_is_read_only(settings):
    conn = connect_ro(settings.db_path)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("INSERT INTO clients(id, name, industry) VALUES ('C-99', 'x', 'y')")
    conn.close()


def test_seed_fallback_connection_is_read_only(settings, tmp_path):
    conn = connect_ro(tmp_path / "missing.db", settings.seed_dir)
    assert conn.execute("SELECT COUNT(*) FROM people").fetchone()[0] == 12
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM people")
    assert not (tmp_path / "missing.db").exists()


DECISION_CODE = [p for d in ("agent", "ui") for p in (PKG / d).rglob("*.py")]
TEXT_METHODS = {"startswith", "endswith", "find", "lower", "casefold", "search", "match",
                "fullmatch", "index", "count", "split", "partition", "__contains__"}


@pytest.mark.parametrize("path", DECISION_CODE, ids=lambda p: p.name)
def test_decision_code_has_no_conditional_or_text_matching(path):
    problems = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.If | ast.IfExp | ast.Match | ast.Compare):
            problems.append(f"line {node.lineno}: {type(node).__name__}")
        if isinstance(node, ast.Import | ast.ImportFrom):
            mods = {a.name for a in node.names} | {getattr(node, "module", None)}
            problems += [f"line {node.lineno}: imports {m}"
                         for m in mods & {"re", "fnmatch", "difflib"}]
        if isinstance(node, ast.Attribute) and node.attr in TEXT_METHODS:
            problems.append(f"line {node.lineno}: .{node.attr}()")
    assert not problems, f"{path.name}: {problems}"


@tool
def any_tool(x: str) -> str:
    """Placeholder tool."""
    return x


def test_agent_graph_is_only_the_model_tools_loop():
    agent = build_agent(model=scripted(AIMessage("hi")), tools=[any_tool])
    g = agent.get_graph()
    assert set(g.nodes) - {"__start__", "__end__"} == {"model", "tools"}
    conditional = {(e.source, e.target) for e in g.edges if e.conditional}
    # All framework edges: model -> tools when it emits tool calls, -> END when it answers,
    # tools -> model with the results; model -> model is create_agent's own edge for
    # middleware that wraps the model call (our SequentialToolCalls). None inspects user text.
    assert conditional <= {("model", "tools"), ("model", "__end__"), ("tools", "model"),
                           ("model", "model")}
    assert agent.checkpointer is None


def test_system_prompt_names_no_tool():
    names = ["ask_knowledge_base", "assign_contractor", "end_assignment", "create_task",
             "update_task_status", "trigger_workflow"]
    assert not [n for n in names if n in SYSTEM_PROMPT]
