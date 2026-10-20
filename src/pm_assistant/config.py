"""Runtime settings, read from environment variables at call time (never at import time)."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # keeps the MCP server process from importing langchain at start-up
    from langchain_core.language_models import BaseChatModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "claude-sonnet-5-5"


@dataclass(frozen=True)
class Settings:
    model: str
    synth_model: str
    db_path: Path
    seed_dir: Path
    corpus_dir: Path
    mcp_server_cmd: tuple[str, ...]
    mcp_timeout_s: float
    rag_k: int
    kg_hops: int


def default_db_path() -> Path:
    """Outside the repo on purpose: synced folders (OneDrive, Dropbox) can lock SQLite files.

    Windows: %LOCALAPPDATA%/pm-assistant/pm.db; elsewhere $XDG_DATA_HOME or ~/.local/share.
    """
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    return Path(base or Path.home() / ".local" / "share") / "pm-assistant" / "pm.db"


def _path(var: str, default: Path) -> Path:
    return Path(os.environ.get(var) or default).resolve()


def load_settings() -> Settings:
    model = os.environ.get("PM_MODEL") or DEFAULT_MODEL
    raw_cmd = os.environ.get("PM_MCP_SERVER_CMD")
    cmd = json.loads(raw_cmd) if raw_cmd else [sys.executable, "-m", "pm_assistant.mcp_server"]
    return Settings(
        model=model,
        synth_model=os.environ.get("PM_SYNTH_MODEL") or model,
        db_path=_path("PM_DB_PATH", default_db_path()),
        seed_dir=_path("PM_SEED_DIR", REPO_ROOT / "data" / "seed"),
        corpus_dir=_path("PM_CORPUS_DIR", REPO_ROOT / "data" / "corpus"),
        mcp_server_cmd=tuple(str(part) for part in cmd),
        mcp_timeout_s=float(os.environ.get("PM_MCP_TIMEOUT_S") or 60),
        rag_k=int(os.environ.get("PM_RAG_K") or 4),
        kg_hops=int(os.environ.get("PM_KG_HOPS") or 2),
    )


def make_chat_model(model_name: str) -> BaseChatModel:
    """Build the Claude chat model. Constructing it does not need an API key; calling it does."""
    from langchain_anthropic import ChatAnthropic

    # No temperature override: current Claude models reject non-default sampling parameters.
    return ChatAnthropic(model=model_name, max_tokens=2048)
