"""EC5 — a new MCP tool is picked up with zero changes to the agent's code; failures of the
action server reach the model as tool errors instead of crashing the turn."""

import asyncio
import json
import sys
import time

from conftest import FIXTURES, info_tool
from fakes import RunRecorder, scripted, tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from pm_assistant.agent.graph import build_agent
from pm_assistant.agent.mcp_tools import load_action_tools
from pm_assistant.config import load_settings


def point_agent_at(monkeypatch, server: str):
    monkeypatch.setenv("PM_MCP_SERVER_CMD", json.dumps([sys.executable, str(FIXTURES / server)]))
    return load_settings()


def run_one_call(settings, name, args, reply="Done."):
    tools = asyncio.run(load_action_tools(settings))
    model = scripted(tool_call(name, args), AIMessage(reply))
    agent = build_agent(model=model, tools=[info_tool(settings), *tools])
    recorder = RunRecorder()
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage("do it")]},
                                    config={"callbacks": [recorder]}))
    return tools, model, recorder, out["messages"]


def test_new_server_tool_is_discovered_bound_and_executed(monkeypatch, db):
    settings = point_agent_at(monkeypatch, "extended_mcp_server.py")
    tools, model, recorder, messages = run_one_call(
        settings, "escalate_risk",
        {"project": "Fleet Telemetry Platform", "severity": "high",
         "summary": "Backfill will slip past 28 October"},
        reply="Escalated.",
    )
    by_name = {t.name: t for t in tools}
    assert by_name["escalate_risk"].args_schema["properties"]["severity"]["enum"] == [
        "low", "medium", "high"
    ]
    assert "escalate_risk" in model.bound_tool_names  # the model was offered the new tool
    assert recorder.count("escalate_risk") == 1
    assert messages[-1].text == "Escalated."
    assert db("SELECT tool FROM action_log ORDER BY id") == [("escalate_risk",)]
    assert db("SELECT name, project_id FROM workflow_runs") == [("escalation", "PRJ-02")]


def test_transport_failure_becomes_a_tool_error_not_a_crash(monkeypatch):
    settings = point_agent_at(monkeypatch, "crashing_mcp_server.py")
    *_, messages = run_one_call(settings, "archive_project", {"project": "PRJ-04"}, "Sorry.")
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_msg.status == "error"
    assert "archive_project" in str(tool_msg.content) and "Connection closed" in str(
        tool_msg.content)
    assert messages[-1].text == "Sorry."


def test_stuck_server_times_out_into_a_tool_error(monkeypatch):
    monkeypatch.setenv("PM_MCP_TIMEOUT_S", "2")
    settings = point_agent_at(monkeypatch, "hanging_mcp_server.py")
    started = time.monotonic()
    *_, messages = run_one_call(settings, "archive_project", {"project": "PRJ-04"}, "Sorry.")
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    # The exact cause text depends on how the stdio session is torn down (read timeout or the
    # closed stream behind it); what matters is a bounded wait that ends in a tool error.
    assert tool_msg.status == "error"
    assert "archive_project" in str(tool_msg.content)
    assert messages[-1].text == "Sorry."
    assert time.monotonic() - started < 30
