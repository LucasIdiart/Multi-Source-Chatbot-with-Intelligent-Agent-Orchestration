"""Write access to the store. Used exclusively by the MCP server (the only action executor).

Every action runs inside one ``BEGIN IMMEDIATE`` transaction: the business-rule checks and the
write see the same data even when several MCP server processes run actions at the same time.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

TaskStatus = Literal["todo", "in_progress", "blocked", "done"]
Workflow = Literal["kickoff", "offboarding", "escalation"]
WORD_RE = re.compile(r"\w+")
ID_RE = re.compile(r"\b[A-Za-z]+-\d+\b")
# Generic words people add to a name ("the Cloud Migration project") that are not part of it.
FILLER = frozenset({"the", "project", "task", "person", "contractor"})


class StoreError(ValueError):
    """A business-rule or reference violation. Reported to the MCP client as a tool error."""


def today() -> str:
    return datetime.now(UTC).date().isoformat()


def connect_rw(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_path), isolation_level=None, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Take the write lock before reading, so check-then-write is atomic across processes."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _words(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.casefold()) if w not in FILLER]


def _require(conn: sqlite3.Connection, table: str, ref: str, label: str) -> sqlite3.Row:
    """Resolve a record by id, by exact name, or by whole words of its name.

    Users talk in names ("assign Kenji to Fraud Detection"); records are keyed by id. In order:
    a bare id ("T-004"); the exact full name or title (even if it contains another record's id,
    e.g. a task titled "Unblock T-003"); a name with its id ("Kenji Sato (P-011)"), accepted only
    when both point to the same record; whole words that are part of exactly one record's name.
    Fragments ("Al", "Martin"), a name and id that disagree, and ambiguous references are
    rejected (with the candidates), so an action never lands on a record the user did not name.
    """
    name_col = "title" if table == "tasks" else "name"
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608 (internal constants)
    ref = (ref or "").strip()
    ids = {i.casefold() for i in ID_RE.findall(ref)}
    by_id = [r for r in rows if r["id"].casefold() in ids]
    name_words = _words(ID_RE.sub(" ", ref))  # the reference without its ids
    if by_id and not name_words:  # a bare id
        return _single(by_id, ref, label, name_col)
    exact = [r for r in rows if _words(r[name_col]) == _words(ref)]
    if exact:
        return _single(exact, ref, label, name_col)
    if by_id:  # name + id: both must describe the same record
        consistent = [r for r in by_id if set(name_words) <= set(_words(r[name_col]))]
        if not consistent:
            named = ", ".join(f"{r['id']} ({r[name_col]})" for r in by_id)
            raise StoreError(f"'{ref}': the id points to {named}, which does not match the "
                             f"name given. Use either the id or the {label}'s name.")
        return _single(consistent, ref, label, name_col)
    matches = [r for r in rows if name_words and set(name_words) <= set(_words(r[name_col]))]
    return _single(matches, ref, label, name_col)


def _single(matches: list[sqlite3.Row], ref: str, label: str, name_col: str) -> sqlite3.Row:
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise StoreError(
            f"Unknown {label} '{ref}'. Use the {label} id or full name "
            f"(whole words of the name also work when they identify a single {label})."
        )
    options = ", ".join(f"{r['id']} ({r[name_col]})" for r in matches)
    raise StoreError(f"'{ref}' matches several {label}s: {options}. Be more specific.")


def log_action(conn: sqlite3.Connection, tool: str, args: dict, result: dict) -> None:
    conn.execute(
        "INSERT INTO action_log(tool, args_json, result_json) VALUES (?, ?, ?)",
        (tool, json.dumps(args, sort_keys=True), json.dumps(result, sort_keys=True)),
    )


def _open_project(conn: sqlite3.Connection, ref: str) -> sqlite3.Row:
    """A project that may still change: closed projects are read-only for every action."""
    project = _require(conn, "projects", ref, "project")
    if project["status"] == "closed":
        raise StoreError(f"Project {project['id']} is closed; it cannot be changed.")
    return project


def assign_contractor(
    conn: sqlite3.Connection, person_id: str, project_id: str, role: str, allocation_pct: int
) -> dict:
    with transaction(conn):
        person = _require(conn, "people", person_id, "person")
        project = _open_project(conn, project_id)
        current = conn.execute(
            "SELECT role, allocation_pct FROM assignments WHERE person_id = ? AND project_id = ?",
            (person["id"], project["id"]),
        ).fetchone()
        other = conn.execute(
            "SELECT COALESCE(SUM(allocation_pct), 0) FROM assignments "
            "WHERE person_id = ? AND project_id != ?",
            (person["id"], project["id"]),
        ).fetchone()[0]
        capacity = person["availability_pct"]
        if other + allocation_pct > capacity:
            raise StoreError(
                f"{person['name']} is already allocated {other}% elsewhere; adding "
                f"{allocation_pct}% exceeds their {capacity}% availability "
                f"(max assignable now: {max(capacity - other, 0)}%)."
            )
        record = {
            "person_id": person["id"],
            "person_name": person["name"],
            "project_id": project["id"],
            "project_name": project["name"],
            "role": role,
            "allocation_pct": allocation_pct,
            "assigned_at": today(),
            "previous": dict(current) if current else None,
        }
        conn.execute(
            "INSERT INTO assignments(person_id, project_id, role, allocation_pct, assigned_at) "
            "VALUES (:person_id, :project_id, :role, :allocation_pct, :assigned_at) "
            "ON CONFLICT(person_id, project_id) DO UPDATE SET role = excluded.role, "
            "allocation_pct = excluded.allocation_pct, assigned_at = excluded.assigned_at",
            record,
        )
        log_action(conn, "assign_contractor",
                   {"person": person_id, "project": project_id, "role": role,
                    "allocation_pct": allocation_pct}, record)
    return record


def end_assignment(conn: sqlite3.Connection, person_id: str, project_id: str) -> dict:
    """Remove a person from a project and queue the offboarding workflow for them."""
    with transaction(conn):
        person = _require(conn, "people", person_id, "person")
        project = _open_project(conn, project_id)
        current = conn.execute(
            "SELECT role, allocation_pct FROM assignments WHERE person_id = ? AND project_id = ?",
            (person["id"], project["id"]),
        ).fetchone()
        if current is None:
            raise StoreError(f"{person['name']} is not assigned to {project['name']}.")
        conn.execute("DELETE FROM assignments WHERE person_id = ? AND project_id = ?",
                     (person["id"], project["id"]))
        run = conn.execute(
            "INSERT INTO workflow_runs(name, project_id, payload_json) VALUES (?, ?, ?)",
            ("offboarding", project["id"],
             json.dumps({"person_id": person["id"], "person_name": person["name"],
                         "requested_on": today()})),
        )
        record = {"person_id": person["id"], "person_name": person["name"],
                  "project_id": project["id"], "project_name": project["name"],
                  "previous": dict(current), "offboarding_run_id": run.lastrowid}
        log_action(conn, "end_assignment", {"person": person_id, "project": project_id}, record)
    return record


def _next_task_id(conn: sqlite3.Connection) -> str:
    ids = [row[0] for row in conn.execute("SELECT id FROM tasks")]
    highest = max((int(i.split("-")[1]) for i in ids), default=0)
    return f"T-{highest + 1:03d}"


def create_task(
    conn: sqlite3.Connection,
    project_id: str,
    title: str,
    assignee_id: str | None = None,
    due_date: str | None = None,
) -> dict:
    if not title.strip():
        raise StoreError("title must not be empty.")
    if due_date is not None:
        try:
            datetime.strptime(due_date, "%Y-%m-%d")
        except ValueError as exc:
            raise StoreError(f"due_date must be a real YYYY-MM-DD date: '{due_date}'.") from exc
    with transaction(conn):
        project = _open_project(conn, project_id)
        assignee = assignee_id and _require(conn, "people", assignee_id, "person")["id"]
        record = {
            "id": _next_task_id(conn),
            "project_id": project["id"],
            "title": title.strip(),
            "status": "todo",
            "assignee_id": assignee,
            "due_date": due_date,
        }
        conn.execute(
            "INSERT INTO tasks(id, project_id, title, status, assignee_id, due_date) "
            "VALUES (:id, :project_id, :title, :status, :assignee_id, :due_date)",
            record,
        )
        log_action(conn, "create_task",
                   {"project": project_id, "title": title, "assignee": assignee_id,
                    "due_date": due_date}, record)
    return record


def update_task_status(conn: sqlite3.Connection, task_id: str, status: TaskStatus) -> dict:
    if status not in get_args(TaskStatus):
        raise StoreError(f"status must be one of {', '.join(get_args(TaskStatus))}.")
    with transaction(conn):
        task = _require(conn, "tasks", task_id, "task")
        _open_project(conn, task["project_id"])
        record = {"id": task["id"], "title": task["title"], "previous_status": task["status"],
                  "status": status}
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task["id"]))
        log_action(conn, "update_task_status", {"task": task_id, "status": status}, record)
    return record


def trigger_workflow(conn: sqlite3.Connection, name: Workflow, project_id: str) -> dict:
    if name not in get_args(Workflow):
        raise StoreError(f"workflow must be one of {', '.join(get_args(Workflow))}.")
    with transaction(conn):
        project = _open_project(conn, project_id)
        payload = {"project_name": project["name"], "requested_on": today()}
        cur = conn.execute(
            "INSERT INTO workflow_runs(name, project_id, payload_json) VALUES (?, ?, ?)",
            (name, project["id"], json.dumps(payload)),
        )
        record = {"run_id": cur.lastrowid, "workflow": name, "project_id": project["id"],
                  "project_name": project["name"], "status": "queued"}
        log_action(conn, "trigger_workflow", {"workflow": name, "project": project_id}, record)
    return record
