"""Test doubles: a scripted tool-calling chat model and a callback that records graph runs."""

from __future__ import annotations

import itertools
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedToolModel(GenericFakeChatModel):
    """Replays scripted AI messages (including tool calls) and records how tools were bound.

    Stands in for the LLM's decision: the graph code under test is identical either way.
    """

    bound_tool_names: list[str] = []
    bound_kwargs: dict[str, Any] = {}

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedToolModel:
        self.bound_tool_names = [t.name for t in tools]
        self.bound_kwargs = kwargs
        return self


def scripted(*messages: AIMessage) -> ScriptedToolModel:
    return ScriptedToolModel(messages=iter(messages))


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def fake_synth(answer: str = "synthesized answer [kg] [doc:x]") -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=itertools.repeat(AIMessage(answer)))


class RunRecorder(BaseCallbackHandler):
    """Collects (name, run_id, parent_run_id) for every chain and tool run."""

    def __init__(self) -> None:
        self.runs: list[tuple[str, UUID, UUID | None]] = []

    def _record(self, serialized, *, run_id, parent_run_id=None, **kw):
        self.runs.append((kw.get("name") or (serialized or {}).get("name", ""), run_id,
                          parent_run_id))

    def on_chain_start(self, serialized, inputs, **kw):
        self._record(serialized, **kw)

    def on_tool_start(self, serialized, input_str, **kw):
        self._record(serialized, **kw)

    def count(self, name: str) -> int:
        return sum(1 for n, _, _ in self.runs if n == name)

    def ancestors(self, name: str) -> list[str]:
        parents = {run_id: parent for _, run_id, parent in self.runs}
        names = {run_id: n for n, run_id, _ in self.runs}
        run_id = next(r for n, r, _ in self.runs if n == name)
        chain = []
        while (run_id := parents.get(run_id)) is not None:
            chain.append(names.get(run_id, "?"))
        return chain
