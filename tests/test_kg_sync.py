"""Actions executed through MCP are visible to the Knowledge Graph on the next question."""

import asyncio

from conftest import with_mcp_session

from pm_assistant.info.kg import build_kg, retrieve_facts
from pm_assistant.store.read import connect_ro, fetch_graph_data


def call(settings, name, args):
    return asyncio.run(with_mcp_session(settings, lambda s: s.call_tool(name, args)))


def free_capacity(settings, person):
    conn = connect_ro(settings.db_path)
    G = build_kg(fetch_graph_data(conn))
    conn.close()
    return G.nodes[f"person:{person}"]["attrs"]["free_capacity_pct"]


def test_assignment_appears_in_graph_and_capacity_is_derived(settings):
    assert free_capacity(settings, "P-011") == 50
    result = call(settings, "assign_contractor", {"person": "Kenji Sato", "project": "PRJ-03",
                                                  "role": "backend engineer",
                                                  "allocation_pct": 40})
    assert not result.isError
    facts = retrieve_facts("Who works on the Fraud Detection Engine?", settings.db_path)
    assert {
        "subject": "Kenji Sato (P-011)",
        "predicate": "ASSIGNED_TO {'role': 'backend engineer', 'allocation_pct': 40}",
        "object": "Fraud Detection Engine (PRJ-03)",
    } in facts
    assert free_capacity(settings, "P-011") == 10


def test_ending_an_assignment_frees_capacity_and_queues_offboarding(settings):
    call(settings, "end_assignment", {"person": "Kenji Sato", "project": "Fleet Telemetry"})
    assert free_capacity(settings, "P-011") == 100
    facts = retrieve_facts("offboarding workflow for Fleet Telemetry Platform", settings.db_path)
    assert any(t["predicate"] == "HAS_WORKFLOW" and "offboarding" in t["object"] for t in facts)


def test_task_status_change_appears_in_graph(settings):
    call(settings, "update_task_status", {"task": "T-003", "status": "in_progress"})
    facts = retrieve_facts("status of the HIPAA audit log schema task", settings.db_path)
    assert {"subject": "Design HIPAA audit log schema (T-003)", "predicate": "status",
            "object": "in_progress"} in facts
