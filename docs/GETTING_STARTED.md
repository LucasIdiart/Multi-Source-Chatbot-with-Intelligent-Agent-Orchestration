# Getting started — run the PM Assistant and demo it

Step-by-step guide to install, run and demo the project. For the architecture and how each
evaluation criterion is proved, see the [README](../README.md).

## 1. Prerequisites

| Requirement | How to get it |
|---|---|
| [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager) | Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` · macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| An Anthropic API key | <https://console.anthropic.com/> |
| Git | <https://git-scm.com/> |

You do **not** need to install Python yourself: uv downloads Python 3.12 automatically.

## 2. Install

```bash
git clone https://github.com/LucasIdiart/Multi-Source-Chatbot-with-Intelligent-Agent-Orchestration.git
cd Multi-Source-Chatbot-with-Intelligent-Agent-Orchestration
uv sync
```

`uv sync` creates `.venv/` and installs the exact versions pinned in `uv.lock`.

## 3. Set the API key

The key is read only from the environment (never from a file).

| Shell | Command |
|---|---|
| PowerShell (current window) | `$env:ANTHROPIC_API_KEY="sk-ant-..."` |
| PowerShell (permanent, new windows) | `setx ANTHROPIC_API_KEY "sk-ant-..."` |
| bash / zsh | `export ANTHROPIC_API_KEY=sk-ant-...` |

Optional: `PM_MODEL` selects the Claude model (default `claude-sonnet-5-5`). All other settings
are listed in [`.env.example`](../.env.example).

## 4. Check the installation (no API key needed)

```bash
uv run pytest -q
```

Expected: all tests pass in about a minute. They use a scripted model and the real MCP server, so
they need no key and no network.

## 5. Run it — choose a chat UI

All three UIs are passive: they only forward your message to the agent and show its reply.

### Option A — Gradio (simplest; best for a screen-shared demo)

```bash
uv run pm-gradio
```

Open <http://127.0.0.1:7860>. Above each answer, a collapsible **🔧 box shows the tool the agent
chose** (`ask_knowledge_base` for information, an MCP action such as `assign_contractor`) and its
arguments; a turn with no box used no tool.

### Option B — LangChain Agent Chat UI (the challenge's recommended UI)

PowerShell:

```powershell
$env:PYTHONIOENCODING="utf-8"
uv run langgraph dev --allow-blocking
```

bash/zsh: `uv run langgraph dev --allow-blocking`

Then, in your regular browser, open <https://agentchat.vercel.app> and set:

- **Deployment URL:** `http://localhost:2024`
- **Assistant / Graph ID:** `pm_assistant`

The same command also opens **LangGraph Studio**, where the two graphs can be inspected visually:
`pm_assistant` (the agent: model ⇄ tools) and `info_pipeline` (RAG ‖ KG → synthesize).

`--allow-blocking` is required because the MCP stdio client makes a blocking call when it starts
the action server. `PYTHONIOENCODING` avoids a Unicode error in the Windows console.

### Option C — Terminal

```bash
uv run pm-chat
```

Type messages after `you>`. Exit with Ctrl+C.

## 6. Demo script

Reset the demo data first, so every demo starts from the same state:

```bash
uv run python -m pm_assistant.store.seed --reset
```

Send a warm-up message (the first one takes a few seconds longer: the agent discovers the MCP
tools), then go through this order:

| # | Message | What it shows |
|---|---|---|
| 1 | *Who leads the Cloud Migration project and for which client?* | Information route: documents (RAG) + knowledge graph (KG), with citations |
| 2 | *Who has the most free capacity right now?* | Aggregate question answered by the KG (Emily Okafor, 70%) |
| 3 | *What is our policy for contractors charging more than 100 USD an hour?* | Answer from a policy document (RAG) |
| 4 | *Assign Kenji Sato to the Fraud Detection Engine as backend engineer at 40%.* | Action route: MCP tool, using names instead of ids |
| 5 | *How much free capacity does Kenji have now?* | The KG reflects the action immediately (10%) |
| 6 | *Assign Diego Martins to Cloud Migration at 30%.* | Business rule rejects the action (Diego is at 100%); the agent explains why |
| 7 | *Mark the HIPAA audit log schema task as in progress, then tell me who is on that team.* | Mixed request: action first, then a question about the new state |
| 8 | *¿Quién en la empresa sabe Kubernetes?* | Works in other languages |
| 9 | *Thanks!* | Small talk: the agent decides to use no tool |

### Show that new MCP tools need no change to the agent

Stop the UI (Ctrl+C), point the agent at an MCP server that has one extra tool, and start it again.

PowerShell:

```powershell
$env:PM_MCP_SERVER_CMD='["python", "tests/fixtures/extended_mcp_server.py"]'
uv run pm-gradio
```

bash/zsh: `PM_MCP_SERVER_CMD='["python", "tests/fixtures/extended_mcp_server.py"]' uv run pm-gradio`

Ask: *Escalate a high-severity risk on Fleet Telemetry: the backfill will slip.* The agent calls
`escalate_risk`, a tool it had never seen, with no change to its code. Afterwards remove the
variable (`Remove-Item Env:PM_MCP_SERVER_CMD` in PowerShell, `unset PM_MCP_SERVER_CMD` in bash).

### Show the automated proof

```bash
uv run pytest -q                    # offline test-suite, no API key
uv run python scripts/smoke.py      # 11-turn conversation with real Claude, PASS/FAIL per turn
```

## 7. Optional extras

- **Inspect the MCP server on its own** (needs Node.js):
  `npx @modelcontextprotocol/inspector uv run pm-mcp-server` — lists the action tools with their
  JSON schemas and lets you call them by hand.
- **Use the assistant from VS Code / Copilot:** `.vscode/mcp.json` registers the whole agent as an
  MCP server (`pm-assistant`, one `ask` tool).

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ANTHROPIC_API_KEY is not set` / authentication error | Set the key in the same terminal that runs the app (section 3). With `setx`, open a new terminal. |
| `langgraph dev` fails with `UnicodeEncodeError` | Set `PYTHONIOENCODING=utf-8` before running it (Windows). |
| `langgraph dev` says port 2024 is in use | A previous server is still running. Windows: `Get-CimInstance Win32_Process \| Where-Object { $_.CommandLine -match 'langgraph\|spawn_main' } \| Stop-Process -Force` · macOS/Linux: `pkill -f langgraph` |
| Agent Chat UI says "Failed to connect" | Make sure `langgraph dev` is running and use a regular browser (some embedded browsers block websites from calling `localhost`). |
| Answers mention old assignments from earlier tests | Reset the data: `uv run python -m pm_assistant.store.seed --reset` |
| The first message is slow | Normal: the agent builds its graph and discovers the MCP tools once per process. Each action also starts a short-lived MCP server process (~1–2 s). |

Where the data lives: the SQLite store is at `%LOCALAPPDATA%\pm-assistant\pm.db` on Windows and
`~/.local/share/pm-assistant/pm.db` elsewhere (override with `PM_DB_PATH`). The documents are in
`data/corpus/` and the seed records in `data/seed/`.
