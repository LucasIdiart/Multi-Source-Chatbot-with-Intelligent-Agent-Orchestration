"""Check a running `langgraph dev` server (the backend Agent Chat UI connects to).

Start the server first (PowerShell):
    $env:PYTHONIOENCODING="utf-8"; uv run langgraph dev --allow-blocking --no-browser
Then:  uv run python scripts/check_server.py [--url http://127.0.0.1:2024]

Sends one information question and one action request through the server's HTTP API and prints
which tools the agent chose. Exit code 1 if either route is missing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from langgraph_sdk import get_client

PROMPTS = [
    ("info", "Who leads the Patient Portal Revamp project?"),
    ("action", "Trigger the kickoff workflow for the Fraud Detection Engine project."),
]


async def run(url: str) -> int:
    client = get_client(url=url)
    graphs = sorted(a["graph_id"] for a in await client.assistants.search())
    print("graphs served:", graphs)
    failures = 0
    for kind, text in PROMPTS:
        thread = await client.threads.create()
        state = await client.runs.wait(
            thread["thread_id"], "pm_assistant",
            input={"messages": [{"role": "user", "content": text}]},
        )
        calls = [tc["name"] for m in state["messages"] for tc in m.get("tool_calls") or []]
        expected = {"info": lambda c: c == ["ask_knowledge_base"],
                    "action": lambda c: "trigger_workflow" in c}[kind](calls)
        failures += not expected
        print(f"[{'PASS' if expected else 'FAIL'}] {kind:<6} {text!r}\n         tools={calls}")
        print("         answer:", str(state["messages"][-1]["content"])[:200].replace("\n", " "))
    return 1 if failures else 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:2024")
    return asyncio.run(run(parser.parse_args().url))


if __name__ == "__main__":
    sys.exit(main())
