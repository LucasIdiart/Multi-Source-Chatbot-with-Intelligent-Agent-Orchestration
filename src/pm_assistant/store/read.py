"""Read-only access to the store. Used exclusively by the informational graph (Knowledge Graph)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .seed import seeded_memory_db

TABLES = (
    "people",
    "person_skills",
    "clients",
    "projects",
    "project_skills",
    "assignments",
    "tasks",
    "workflow_runs",
)


def connect_ro(db_path: Path, seed_dir: Path | None = None) -> sqlite3.Connection:
    """Open the store in SQLite read-only mode: any write raises OperationalError.

    Before the action server has created the store (fresh machine), the seed data is served
    from an in-memory copy instead, so information questions work from the first run without
    the informational side ever writing to disk.
    """
    path = Path(db_path)
    if path.exists() or seed_dir is None:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    else:
        conn = seeded_memory_db(seed_dir)
        conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    return conn


def fetch_graph_data(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    return {
        table: [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]  # noqa: S608
        for table in TABLES
    }
