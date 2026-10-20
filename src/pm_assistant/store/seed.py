"""Create and seed the SQLite store from data/seed/*.json (idempotent, atomic)."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

from .schema import create_schema

SEED_FILES = ("clients", "people", "projects", "assignments", "tasks")


def load_seed(seed_dir: Path) -> dict[str, list[dict]]:
    return {
        name: json.loads((Path(seed_dir) / f"{name}.json").read_text(encoding="utf-8"))
        for name in SEED_FILES
    }


def _insert_seed(conn: sqlite3.Connection, seed: dict[str, list[dict]]) -> None:
    skills = {s["name"] for p in seed["people"] for s in p["skills"]}
    skills |= {s for p in seed["projects"] for s in p["required_skills"]}
    conn.executemany("INSERT INTO skills(name) VALUES (?)", [(s,) for s in sorted(skills)])
    conn.executemany(
        "INSERT INTO clients(id, name, industry) VALUES (:id, :name, :industry)", seed["clients"]
    )
    conn.executemany(
        "INSERT INTO people(id, name, title, seniority, employment_type, location, "
        "hourly_rate_usd, availability_pct) VALUES (:id, :name, :title, :seniority, "
        ":employment_type, :location, :hourly_rate_usd, :availability_pct)",
        seed["people"],
    )
    conn.executemany(
        "INSERT INTO person_skills(person_id, skill, level) VALUES (?, ?, ?)",
        [(p["id"], s["name"], s["level"]) for p in seed["people"] for s in p["skills"]],
    )
    conn.executemany(
        "INSERT INTO projects(id, name, client_id, status, start_date, end_date, lead_id, "
        "description) VALUES (:id, :name, :client_id, :status, :start_date, :end_date, "
        ":lead_id, :description)",
        seed["projects"],
    )
    conn.executemany(
        "INSERT INTO project_skills(project_id, skill) VALUES (?, ?)",
        [(p["id"], s) for p in seed["projects"] for s in p["required_skills"]],
    )
    conn.executemany(
        "INSERT INTO assignments(person_id, project_id, role, allocation_pct, assigned_at) "
        "VALUES (:person_id, :project_id, :role, :allocation_pct, :assigned_at)",
        seed["assignments"],
    )
    conn.executemany(
        "INSERT INTO tasks(id, project_id, title, status, assignee_id, due_date) "
        "VALUES (:id, :project_id, :title, :status, :assignee_id, :due_date)",
        seed["tasks"],
    )


def _build(conn: sqlite3.Connection, seed_dir: Path) -> sqlite3.Connection:
    conn.execute("PRAGMA foreign_keys=ON")
    create_schema(conn)
    with conn:
        _insert_seed(conn, load_seed(seed_dir))
    return conn


def seeded_memory_db(seed_dir: Path) -> sqlite3.Connection:
    """An in-memory database holding the seed data (nothing touches the disk)."""
    return _build(sqlite3.connect(":memory:"), seed_dir)


def ensure_db(db_path: Path, seed_dir: Path, *, reset: bool = False) -> Path:
    """Create the database from the seed if it does not exist (or if reset=True).

    The database is built in a temporary file and moved into place, so readers never see a
    half-seeded store. If another process created the store meanwhile, its copy is kept.
    """
    db_path = Path(db_path)
    if db_path.exists() and not reset:
        return db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(f"{db_path.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    try:
        _build(conn, seed_dir)
        conn.close()
        if reset or not db_path.exists():
            os.replace(tmp, db_path)
    finally:
        conn.close()  # before unlink: Windows cannot delete an open file
        tmp.unlink(missing_ok=True)
    return db_path


def main() -> None:
    from pm_assistant.config import load_settings

    parser = argparse.ArgumentParser(description="Create (or reset) the PM SQLite store.")
    parser.add_argument("--reset", action="store_true", help="rebuild from the seed files")
    args = parser.parse_args()
    settings = load_settings()
    path = ensure_db(settings.db_path, settings.seed_dir, reset=args.reset)
    print(f"database ready: {path}")


if __name__ == "__main__":
    main()
