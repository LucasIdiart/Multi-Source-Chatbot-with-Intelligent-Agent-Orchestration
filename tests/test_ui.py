"""EC6 — the UIs are passive: they forward messages to the agent graph and show its reply."""

import asyncio
import importlib
import json

from conftest import imports_of, violates
from langchain_core.messages import AIMessage, HumanMessage

from pm_assistant.config import REPO_ROOT
from pm_assistant.ui import cli, gradio_app


class GraphSpy:
    def __init__(self, fail_on: str | None = None):
        self.calls = []
        self.fail_on = fail_on

    async def ainvoke(self, state):
        self.calls.append(state["messages"])
        if state["messages"][-1].content == self.fail_on:
            raise RuntimeError("API overloaded")
        return {"messages": [*state["messages"], AIMessage("reply from agent")]}


def scripted_input(*lines):
    pending = iter(lines)

    async def read(prompt):
        try:
            return next(pending)
        except StopIteration:
            raise EOFError from None  # what input() raises at end of input

    return read


def test_cli_forwards_every_message_and_prints_the_reply(capsys):
    spy = GraphSpy()
    asyncio.run(cli.chat_loop(spy, read=scripted_input("who leads PRJ-01?", "assign Kenji")))
    assert [m[-1].content for m in spy.calls] == ["who leads PRJ-01?", "assign Kenji"]
    assert len(spy.calls[1]) == 3  # history (question, answer) + new message
    assert capsys.readouterr().out.count("reply from agent") == 2


def test_cli_survives_a_failing_turn_and_keeps_the_conversation(capsys):
    spy = GraphSpy(fail_on="")
    asyncio.run(cli.chat_loop(spy, read=scripted_input("hi", "", "still there?")))
    out = capsys.readouterr().out
    assert "[error] RuntimeError: API overloaded" in out
    assert [m[-1].content for m in spy.calls] == ["hi", "", "still there?"]
    assert len(spy.calls[2]) == 3  # the failed turn did not wipe the history


def test_gradio_forwards_history_and_message():
    spy = GraphSpy()

    async def factory():
        return spy

    respond = gradio_app.make_respond(factory)
    history = [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": [{"type": "text", "text": "hello"}],
                "metadata": None}]
    reply = asyncio.run(respond("who leads PRJ-01?", history))
    assert reply == "reply from agent" and type(reply) is str
    sent = spy.calls[0]
    assert [type(m) for m in sent] == [HumanMessage, AIMessage, HumanMessage]
    assert [m.text for m in sent] == ["hi", "hello", "who leads PRJ-01?"]
    # A reply fed back as history (Gradio API path) works again.
    again = asyncio.run(respond("and now?", [*history, {"role": "assistant", "content": reply}]))
    assert again == "reply from agent"


def test_gradio_demo_builds():
    assert gradio_app.build_demo(lambda: None) is not None


def test_ui_imports_no_tools_or_data():
    assert not violates(imports_of("ui"), ("pm_assistant.info", "pm_assistant.mcp_server",
                                           "pm_assistant.store", "langchain_mcp_adapters",
                                           "mcp"))


def load_spec(spec: str):
    module, attr = spec.split(":")
    return getattr(importlib.import_module(module), attr)


def test_langgraph_json_exposes_agent_and_info_pipeline():
    config = json.loads((REPO_ROOT / "langgraph.json").read_text(encoding="utf-8"))
    assert set(config["graphs"]) == {"pm_assistant", "info_pipeline"}
    factory = load_spec(config["graphs"]["pm_assistant"])
    agent = asyncio.run(factory())
    assert agent.checkpointer is None
    tools = agent.nodes["tools"].bound.tools_by_name
    assert set(tools) == {"ask_knowledge_base", "assign_contractor", "end_assignment",
                          "create_task", "update_task_status", "trigger_workflow"}
    assert asyncio.run(factory()) is agent  # discovered once per process, not per request
    info = load_spec(config["graphs"]["info_pipeline"])()
    assert "synthesize" in info.get_graph().nodes
