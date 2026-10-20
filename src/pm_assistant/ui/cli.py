"""Terminal chat. Passive: every message goes to the agent graph unchanged.

Exit with Ctrl+C, or end of input (Ctrl+Z then Enter on Windows, Ctrl+D elsewhere).
Run with the `pm-chat` script (ui/ has no `__main__` guard: it is AST-checked to contain no
conditionals at all).
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from pm_assistant.agent.graph import chat_turn


async def read_line(prompt: str) -> str:
    """input() on a daemon thread, so Ctrl+C can end the program while it waits for a line."""
    loop = asyncio.get_running_loop()
    line: asyncio.Future[str] = loop.create_future()

    def worker() -> None:
        try:
            loop.call_soon_threadsafe(line.set_result, input(prompt))
        except BaseException as exc:  # EOFError at end of input
            loop.call_soon_threadsafe(line.set_exception, exc)

    threading.Thread(target=worker, daemon=True).start()
    return await line


async def chat_loop(graph: Any, read: Callable[[str], Awaitable[str]] = read_line) -> None:
    history: list = []
    with contextlib.suppress(EOFError, KeyboardInterrupt, asyncio.CancelledError):
        while True:
            text = await read("you> ")
            try:
                history, reply = await chat_turn(graph, history, text)
            except Exception as exc:  # keep the conversation alive (API errors, empty input)
                reply = f"[error] {type(exc).__name__}: {exc}"
            print(f"assistant> {reply}\n", flush=True)


def main() -> None:
    from pm_assistant.agent.graph import make_graph

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    async def run() -> None:
        await chat_loop(await make_graph())

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())
