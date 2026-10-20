"""EC3 — the agent picks the route by tool calling; nothing inspects the user's text.

The scripted model stands in for the LLM's decision. The same user text produces different
routes depending only on what the model decides, which shows the graph has no other router.
Real-model routing is checked by scripts/smoke.py (gate smoke-real).
"""

import asyncio

from conftest import info_tool
from fakes import RunRecorder, scripted, tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from pm_assistant.agent.graph import build_agent
from pm_assistant.agent.mcp_tools import load_action_tools

TEXT = "Kenji Sato and the Fraud Detection Engine"


def agent_with(settings, model):
    return build_agent(model=model,
                       tools=[info_tool(settings), *asyncio.run(load_action_tools(settings))])


def run(agent, text):
    recorder = RunRecorder()
    out = asyncio.run(
        agent.ainvoke({"messages": [HumanMessage(text)]}, config={"callbacks": [recorder]})
    )
    return out["messages"], recorder


def test_info_decision_runs_the_rag_kg_graph_and_no_action(settings, db):
    model = scripted(tool_call("ask_knowledge_base", {"question": TEXT}), AIMessage("answer"))
    messages, rec = run(agent_with(settings, model), TEXT)
    assert rec.count("info_subgraph") == 1
    assert rec.count("assign_contractor") == 0
    assert db("SELECT COUNT(*) FROM action_log") == [(0,)]
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_msg.name == "ask_knowledge_base" and tool_msg.status == "success"


def test_action_decision_calls_the_mcp_server_and_not_the_info_graph(settings, db):
    args = {"person": "Kenji Sato", "project": "Fraud Detection Engine",
            "role": "backend engineer", "allocation_pct": 40}
    model = scripted(tool_call("assign_contractor", args), AIMessage("done"))
    _, rec = run(agent_with(settings, model), TEXT)
    assert rec.count("info_subgraph") == 0
    assert rec.count("assign_contractor") == 1
    assert db("SELECT role FROM assignments WHERE person_id='P-011' AND project_id='PRJ-03'") == [
        ("backend engineer",)
    ]
    assert db("SELECT tool FROM action_log") == [("assign_contractor",)]


def test_same_text_different_decisions_different_routes(settings):
    info_model = scripted(tool_call("ask_knowledge_base", {"question": TEXT}), AIMessage("a"))
    act_model = scripted(
        tool_call("trigger_workflow", {"workflow": "kickoff", "project": "PRJ-03"}),
        AIMessage("b"),
    )
    no_tool_model = scripted(AIMessage("just chatting"))
    _, rec_info = run(agent_with(settings, info_model), TEXT)
    _, rec_act = run(agent_with(settings, act_model), TEXT)
    _, rec_none = run(agent_with(settings, no_tool_model), TEXT)
    assert (rec_info.count("info_subgraph"), rec_info.count("trigger_workflow")) == (1, 0)
    assert (rec_act.count("info_subgraph"), rec_act.count("trigger_workflow")) == (0, 1)
    assert (rec_none.count("info_subgraph"), rec_none.count("trigger_workflow")) == (0, 0)


def test_mixed_request_change_then_question_in_one_turn(settings, db):
    model = scripted(
        tool_call("assign_contractor", {"person": "P-011", "project": "PRJ-03",
                                        "role": "backend engineer", "allocation_pct": 30}, "c1"),
        tool_call("ask_knowledge_base", {"question": "Free capacity of Kenji Sato?"}, "c2"),
        AIMessage("Assigned Kenji; he has 20% free."),
    )
    messages, _ = run(agent_with(settings, model), "Assign Kenji, then tell me his capacity")
    assert [m.name for m in messages if isinstance(m, ToolMessage)] == [
        "assign_contractor", "ask_knowledge_base"
    ]
    knowledge = [m for m in messages if isinstance(m, ToolMessage)][1]
    assert knowledge.status == "success"
    assert db("SELECT COUNT(*) FROM action_log") == [(1,)]


def test_model_is_asked_for_one_tool_call_at_a_time(settings):
    model = scripted(AIMessage("hi"))
    run(agent_with(settings, model), "hello")
    assert model.bound_kwargs.get("parallel_tool_calls") is False


def test_action_error_is_reported_to_the_model_not_raised(settings):
    model = scripted(
        tool_call("assign_contractor", {"person": "P-004", "project": "PRJ-03",
                                        "role": "data engineer", "allocation_pct": 50}),
        AIMessage("Could not assign."),
    )
    messages, _ = run(agent_with(settings, model), "assign Diego")
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_msg.status == "error"
    assert "exceeds" in str(tool_msg.content)
    assert messages[-1].text == "Could not assign."
