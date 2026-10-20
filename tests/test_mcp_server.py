"""EC2 — actions go through our MCP server, over the real stdio protocol, with correct schemas."""

import asyncio
import json
import sqlite3

import pytest
from conftest import with_mcp_session

from pm_assistant.store import write

ACTION_TOOLS = {"assign_contractor", "end_assignment", "create_task", "update_task_status",
                "trigger_workflow"}
DESTRUCTIVE = {"assign_contractor", "end_assignment", "update_task_status"}


def session_run(settings, fn):
    return asyncio.run(with_mcp_session(settings, fn))


def test_lists_exactly_the_action_tools_with_json_schemas(settings):
    tools = session_run(settings, lambda s: s.list_tools())
    by_name = {t.name: t for t in tools.tools}
    assert set(by_name) == ACTION_TOOLS
    for tool in by_name.values():
        schema = tool.inputSchema
        assert schema["type"] == "object"
        assert schema["properties"] and schema["required"]
        assert tool.description
        assert tool.outputSchema and tool.outputSchema["type"] == "object"
        assert tool.annotations.readOnlyHint is False
        assert tool.annotations.destructiveHint is (tool.name in DESTRUCTIVE)
    status = by_name["update_task_status"].inputSchema["properties"]["status"]
    assert status["enum"] == ["todo", "in_progress", "blocked", "done"]
    workflow = by_name["trigger_workflow"].inputSchema["properties"]["workflow"]
    assert workflow["enum"] == ["kickoff", "offboarding", "escalation"]
    alloc = by_name["assign_contractor"].inputSchema["properties"]["allocation_pct"]
    assert (alloc["minimum"], alloc["maximum"]) == (1, 100)
    assert set(by_name["create_task"].inputSchema["required"]) == {"project", "title"}
    assert by_name["create_task"].inputSchema["properties"]["title"]["maxLength"] == 200


def test_exposes_no_read_tools(settings):
    tools = session_run(settings, lambda s: s.list_tools())
    prefixes = ("get_", "list_", "search_", "find_", "query_", "read_")
    assert not [t.name for t in tools.tools if t.name.startswith(prefixes)]


def test_call_by_name_mutates_the_store_and_returns_structured_content(settings, db):
    result = session_run(
        settings,
        lambda s: s.call_tool(
            "assign_contractor",
            {"person": "Kenji Sato", "project": "Fraud Detection Engine",
             "role": "backend engineer", "allocation_pct": 40},
        ),
    )
    assert result.isError is False
    assert result.structuredContent["record"]["person_id"] == "P-011"
    assert db("SELECT allocation_pct FROM assignments WHERE person_id='P-011' "
              "AND project_id='PRJ-03'") == [(40,)]
    assert db("SELECT tool FROM action_log") == [("assign_contractor",)]


def test_all_actions_work_by_id(settings, db):
    async def calls(s):
        return [
            await s.call_tool("create_task", {"project": "PRJ-03", "title": "Feature store PoC",
                                              "assignee": "P-005", "due_date": "2026-12-01"}),
            await s.call_tool("update_task_status", {"task": "T-003", "status": "in_progress"}),
            await s.call_tool("trigger_workflow", {"workflow": "kickoff", "project": "PRJ-03"}),
            await s.call_tool("end_assignment", {"person": "P-007", "project": "PRJ-02"}),
        ]

    results = session_run(settings, calls)
    assert [r.isError for r in results] == [False, False, False, False]
    assert db("SELECT title, assignee_id FROM tasks WHERE id='T-011'") == [
        ("Feature store PoC", "P-005")
    ]
    assert db("SELECT status FROM tasks WHERE id='T-003'") == [("in_progress",)]
    assert results[1].structuredContent["record"]["previous_status"] == "blocked"
    assert db("SELECT COUNT(*) FROM assignments WHERE person_id='P-007' AND project_id='PRJ-02'"
              ) == [(0,)]
    assert db("SELECT name, status FROM workflow_runs ORDER BY id") == [
        ("kickoff", "queued"), ("offboarding", "queued")
    ]


def test_updating_an_assignment_keeps_the_previous_values(settings):
    result = session_run(settings, lambda s: s.call_tool(
        "assign_contractor", {"person": "Ben Carter", "project": "Fleet Telemetry Platform",
                              "role": "architect", "allocation_pct": 50}))
    assert result.structuredContent["record"]["previous"] == {"role": "tech lead",
                                                              "allocation_pct": 70}


def test_rule_and_schema_violations_are_tool_errors(settings, db):
    async def calls(s):
        return [
            await s.call_tool("assign_contractor", {"person": "Nobody Here", "project": "PRJ-01",
                                                    "role": "dev", "allocation_pct": 10}),
            # Diego is already 100% on Fleet Telemetry.
            await s.call_tool("assign_contractor", {"person": "P-004", "project": "PRJ-03",
                                                    "role": "data engineer", "allocation_pct": 20}),
            await s.call_tool("assign_contractor", {"person": "P-011", "project": "PRJ-03",
                                                    "role": "dev", "allocation_pct": 500}),
            await s.call_tool("update_task_status", {"task": "T-001", "status": "finished"}),
            await s.call_tool("end_assignment", {"person": "P-005", "project": "PRJ-01"}),
        ]

    results = session_run(settings, calls)
    assert [r.isError for r in results] == [True] * 5
    text = json.dumps([r.content[0].text for r in results])
    assert "Unknown person" in text and "exceeds" in text and "not assigned" in text
    assert db("SELECT COUNT(*) FROM action_log") == [(0,)]


@pytest.mark.parametrize(
    "ref, expected",
    [("P-011", "P-011"), ("Kenji Sato", "P-011"), ("kenji", "P-011"), ("SATO", "P-011")],
)
def test_names_resolve_by_whole_words(settings, ref, expected):
    conn = write.connect_rw(settings.db_path)
    assert write._require(conn, "people", ref, "person")["id"] == expected


@pytest.mark.parametrize("ref", ["Al", "Martin", "Emil", "%", "_", "k_nji", "y"])
def test_fragments_and_wildcards_never_resolve(settings, ref):
    conn = write.connect_rw(settings.db_path)
    with pytest.raises(write.StoreError, match="Unknown"):
        write._require(conn, "people", ref, "person")


def test_ambiguous_references_list_the_candidates(settings):
    conn = write.connect_rw(settings.db_path)
    with pytest.raises(write.StoreError, match="matches several.*T-"):
        write._require(conn, "tasks", "audit", "task")  # T-002 and T-003
    assert write._require(conn, "projects", "Fraud Detection", "project")["id"] == "PRJ-03"


def test_capacity_check_and_write_are_atomic_across_connections(settings, db):
    """A second writer waits for the first transaction, then sees its allocation."""
    first, second = write.connect_rw(settings.db_path), write.connect_rw(settings.db_path)
    second.execute("PRAGMA busy_timeout=0")
    with write.transaction(first):
        first.execute("UPDATE assignments SET allocation_pct = 90 WHERE person_id='P-011'")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            write.assign_contractor(second, "P-011", "PRJ-05", "backend engineer", 40)
    with pytest.raises(write.StoreError, match="exceeds"):
        write.assign_contractor(second, "P-011", "PRJ-05", "backend engineer", 40)
    assert db("SELECT SUM(allocation_pct) FROM assignments WHERE person_id='P-011'") == [(90,)]


def test_closed_projects_reject_changes(settings, db):
    conn = write.connect_rw(settings.db_path)
    conn.execute("UPDATE projects SET status='closed' WHERE id='PRJ-04'")
    for action in (lambda: write.create_task(conn, "PRJ-04", "New work"),
                   lambda: write.trigger_workflow(conn, "kickoff", "PRJ-04"),
                   lambda: write.assign_contractor(conn, "P-008", "PRJ-04", "dev", 10),
                   lambda: write.end_assignment(conn, "P-008", "PRJ-04"),
                   lambda: write.update_task_status(conn, "T-008", "todo")):
        with pytest.raises(write.StoreError, match="closed"):
            action()
    assert db("SELECT COUNT(*) FROM action_log") == [(0,)]


@pytest.mark.parametrize(
    "table, ref, expected",
    [("people", "Kenji Sato (P-011)", "P-011"),
     ("projects", "Fraud Detection Engine (PRJ-03)", "PRJ-03"),
     ("projects", "the Cloud Migration project", "PRJ-05"),
     ("tasks", "T-004", "T-004")],
)
def test_labels_the_system_produces_resolve(settings, table, ref, expected):
    conn = write.connect_rw(settings.db_path)
    assert write._require(conn, table, ref, "record")["id"] == expected


def test_an_id_wins_over_a_title_that_looks_like_an_id(settings):
    conn = write.connect_rw(settings.db_path)
    write.create_task(conn, "PRJ-01", "T-004")
    assert write._require(conn, "tasks", "T-004", "task")["id"] == "T-004"


def test_a_full_title_wins_over_an_id_mentioned_inside_it(settings, db):
    conn = write.connect_rw(settings.db_path)
    created = write.create_task(conn, "PRJ-01", "Unblock T-003 with legal sign-off")
    moved = write.update_task_status(conn, "Unblock T-003 with legal sign-off", "done")
    assert moved["id"] == created["id"] == "T-011"
    assert db("SELECT status FROM tasks WHERE id='T-003'") == [("blocked",)]


@pytest.mark.parametrize("table, ref", [("people", "Kenji Sato (P-001)"),
                                        ("tasks", "Design HIPAA audit log schema (T-004)")])
def test_a_name_and_id_that_disagree_are_rejected(settings, table, ref):
    conn = write.connect_rw(settings.db_path)
    with pytest.raises(write.StoreError, match="does not match the name"):
        write._require(conn, table, ref, "record")
