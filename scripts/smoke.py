"""Real-model smoke test: does Claude route each turn correctly on its own, with correct answers?

Runs one multi-turn conversation against the real agent (real Claude, real MCP server, a
throw-away copy of the database) and checks, per turn:

- information turns run the RAG + KG graph and no action; where it matters, the answer contains
  the fact that only fresh, complete retrieval can give (e.g. the post-assignment capacity);
- action turns call an MCP action tool and change the database;
- small talk uses no tool; tool calls arrive one per model step (no parallel calls).

The checks inspect the tool calls the model made and the resulting database/answer, never the
user's text. Exit code 1 on any mismatch. Usage (needs ANTHROPIC_API_KEY):
    uv run python scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[1]
INFO = "ask_knowledge_base"


class Trace(BaseCallbackHandler):
    """Counts informational-graph runs and what each retriever returned."""

    def __init__(self) -> None:
        self.names: dict = {}
        self.info_runs = 0
        self.chunks: list[list[str]] = []
        self.triples: list[int] = []

    def on_chain_start(self, serialized, inputs, *, run_id, **kw):
        self.names[run_id] = kw.get("name")
        self.info_runs += kw.get("name") == "info_subgraph"

    def on_chain_end(self, outputs, *, run_id, **kw):
        name = self.names.get(run_id)
        if name == "rag_retrieve":
            self.chunks.append([c["doc_id"] for c in outputs["chunks"]])
        if name == "kg_retrieve":
            self.triples.append(len(outputs["triples"]))


@dataclass
class Turn:
    kind: str  # info | action | chat | mixed
    text: str
    expect_tools: tuple[str, ...] = ()
    db_check: tuple[str, object] | None = None  # (sql, expected first row)
    answer_contains: tuple[str, ...] = ()
    expect_doc: str | None = None
    calls: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


CONVERSATION = [
    Turn("info", "Who leads the Cloud Migration project, and for which client is it?",
         answer_contains=("Farid", "Cedar Bank"), expect_doc="brief-cloud-migration"),
    Turn("info", "What does our policy say about contractors who charge more than 100 USD/hour?",
         answer_contains=("Ana Ruiz",), expect_doc="staffing-policy"),
    Turn("info", "And which people on that migration project are contractors?",
         answer_contains=("Farid",)),
    Turn("info", "Who has the most free capacity right now?", answer_contains=("Emily",)),
    Turn("action", "Please assign Kenji Sato to the Fraud Detection Engine as a backend engineer "
         "at 40%.", expect_tools=("assign_contractor",),
         db_check=("SELECT allocation_pct FROM assignments WHERE person_id='P-011' AND "
                   "project_id='PRJ-03'", (40,))),
    Turn("info", "How much free capacity does Kenji have now?", answer_contains=("10%",)),
    Turn("chat", "Thanks, that is all for now!"),
    Turn("mixed", "Mark the HIPAA audit log schema task as in progress, then tell me who is on "
         "the Patient Portal team.", expect_tools=("update_task_status", INFO),
         db_check=("SELECT status FROM tasks WHERE id='T-003'", ("in_progress",)),
         answer_contains=("Chloe",)),
    Turn("action", "Kenji is leaving the Fleet Telemetry Platform; take him off that project.",
         expect_tools=("end_assignment",),
         db_check=("SELECT COUNT(*) FROM assignments WHERE person_id='P-011' AND "
                   "project_id='PRJ-02'", (0,))),
    Turn("info", "¿Quién en toda la empresa sabe Kubernetes?", answer_contains=("Farid",)),
]
EXTENSION_TURN = Turn(
    "action", "Escalate a high-severity risk on the Fleet Telemetry Platform: the telemetry "
    "backfill is going to slip past its deadline.", expect_tools=("escalate_risk",),
    db_check=("SELECT COUNT(*) FROM action_log WHERE tool='escalate_risk'", (1,)),
)


def db_row(sql: str):
    conn = sqlite3.connect(os.environ["PM_DB_PATH"])
    try:
        return conn.execute(sql).fetchone()
    finally:
        conn.close()


def check(turn: Turn, trace: Trace, action_tools: set[str], answer: str, per_step: list[int]):
    actions = [c for c in turn.calls if c in action_tools]
    if turn.kind == "info":
        if INFO not in turn.calls or trace.info_runs < 1:
            turn.problems.append("expected the knowledge graph pipeline to run")
        if actions:
            turn.problems.append(f"unexpected actions {actions}")
        if not (trace.triples and max(trace.triples) >= 1):
            turn.problems.append(f"knowledge graph returned nothing: {trace.triples}")
    if turn.expect_doc and not any(turn.expect_doc in docs for docs in trace.chunks):
        turn.problems.append(f"document {turn.expect_doc} not retrieved: {trace.chunks}")
    if turn.kind == "chat" and turn.calls:
        turn.problems.append(f"small talk used tools {turn.calls}")
    for tool in turn.expect_tools:
        if tool not in turn.calls:
            turn.problems.append(f"expected a call to {tool}")
    if turn.kind == "mixed" and INFO in turn.calls and turn.expect_tools[0] in turn.calls:
        last_info = max(i for i, c in enumerate(turn.calls) if c == INFO)
        if turn.calls.index(turn.expect_tools[0]) > last_info:
            turn.problems.append("the question was answered before the change was made")
    if any(n > 1 for n in per_step):
        turn.problems.append(f"parallel tool calls in one step: {per_step}")
    if turn.db_check and db_row(turn.db_check[0]) != turn.db_check[1]:
        turn.problems.append(f"database: {db_row(turn.db_check[0])} != {turn.db_check[1]}")
    for expected in turn.answer_contains:
        if expected not in answer:
            turn.problems.append(f"answer does not mention {expected!r}")


async def run_conversation(turns: list[Turn]) -> None:
    from pm_assistant.agent.graph import make_graph

    graph = await make_graph()
    action_tools = set(graph.nodes["tools"].bound.tools_by_name) - {INFO}
    history: list = []
    for turn in turns:
        trace = Trace()
        result = await graph.ainvoke(
            {"messages": [*history, HumanMessage(turn.text)]}, config={"callbacks": [trace]}
        )
        new = [m for m in result["messages"][len(history) + 1:] if isinstance(m, AIMessage)]
        tool_calls = [tc for m in new for tc in m.tool_calls]
        turn.calls = [tc["name"] for tc in tool_calls]
        history = result["messages"]
        answer = history[-1].text
        check(turn, trace, action_tools, answer, [len(m.tool_calls) for m in new])
        status = "PASS" if not turn.problems else "FAIL"
        print(f"[{status}] {turn.kind:<6} {turn.text[:70]!r}\n         tools={turn.calls} "
              f"info_runs={trace.info_runs} docs={trace.chunks} triples={trace.triples}")
        for tc in tool_calls:
            print(f"         -> {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")
        print("         answer:", answer[:260].replace("\n", " "))
        for problem in turn.problems:
            print("         !!", problem)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2
    os.environ["PM_DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="pm-smoke-")) / "pm.db")
    print(f"model={os.environ.get('PM_MODEL', 'default')}  db={os.environ['PM_DB_PATH']}\n")
    asyncio.run(run_conversation(CONVERSATION))

    print("\n-- extension: same agent code, MCP server with one extra tool --")
    os.environ["PM_MCP_SERVER_CMD"] = json.dumps(
        [sys.executable, str(ROOT / "tests" / "fixtures" / "extended_mcp_server.py")]
    )
    asyncio.run(run_conversation([EXTENSION_TURN]))

    turns = [*CONVERSATION, EXTENSION_TURN]
    failed = [t for t in turns if t.problems]
    print(f"\n{len(turns) - len(failed)}/{len(turns)} turns routed and answered correctly")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
