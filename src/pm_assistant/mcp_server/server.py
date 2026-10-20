"""MCP server exposing the PM *action* tools over stdio.

This is the only component that changes state. It deliberately exposes no read/query tools:
information questions are answered by the agent's RAG + Knowledge Graph pipeline, never here.
To add a capability, add another ``@mcp.tool`` below — the agent discovers it at startup with no
change to its decision logic.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from pm_assistant.config import load_settings
from pm_assistant.store import write
from pm_assistant.store.seed import ensure_db

mcp = FastMCP(
    "pm-actions",
    instructions=(
        "Action tools that create or modify project-management records (staffing assignments, "
        "tasks, workflow runs). Every call is validated and written to an audit log. People, "
        "projects and tasks can be referenced by id (P-001, PRJ-01, T-001) or by name."
    ),
    log_level="WARNING",  # logs go to stderr: stdout is the MCP protocol channel
)

# Creates new records only.
ADDITIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False,
                           openWorldHint=False)
# Overwrites or removes existing data (MCP clients may ask the user to confirm these).
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False,
                              openWorldHint=False)

PersonRef = Annotated[
    str, Field(min_length=2, max_length=80, description="Person id (e.g. P-003) or name.")
]
ProjectRef = Annotated[
    str, Field(min_length=2, max_length=80, description="Project id (e.g. PRJ-01) or name.")
]
TaskRef = Annotated[
    str, Field(min_length=2, max_length=200,
               description="Task id (e.g. T-004) or the words of its title.")
]


class ActionResult(BaseModel):
    """Outcome of an action tool. Failed actions are reported as MCP tool errors instead."""

    message: str = Field(description="Human-readable summary of what changed.")
    record: dict[str, Any] = Field(description="The record as stored after the change.")


def run_action(action: Callable[..., dict], /, **kwargs: Any) -> dict:
    """Open the store, run one write action, and turn rule violations into MCP tool errors."""
    settings = load_settings()
    ensure_db(settings.db_path, settings.seed_dir)
    conn = write.connect_rw(settings.db_path)
    try:
        return action(conn, **kwargs)
    except (write.StoreError, sqlite3.IntegrityError) as exc:
        raise ValueError(str(exc)) from exc
    finally:
        conn.close()


@mcp.tool(annotations=DESTRUCTIVE)
def assign_contractor(
    person: PersonRef,
    project: ProjectRef,
    role: Annotated[
        str, Field(min_length=2, max_length=80, description="Role, e.g. 'backend engineer'.")
    ],
    allocation_pct: Annotated[
        int, Field(ge=1, le=100, description="Share of a full working week, 1-100.")
    ],
) -> ActionResult:
    """Assign a person (employee or contractor) to a project, or change their existing
    assignment on it (the previous role and allocation are returned and audit-logged).

    Rejected if the total allocation would exceed the person's contracted availability.
    """
    rec = run_action(
        write.assign_contractor,
        person_id=person, project_id=project, role=role, allocation_pct=allocation_pct,
    )
    return ActionResult(
        message=f"{rec['person_name']} assigned to {rec['project_name']} as {rec['role']} "
        f"at {rec['allocation_pct']}%.",
        record=rec,
    )


@mcp.tool(annotations=DESTRUCTIVE)
def end_assignment(person: PersonRef, project: ProjectRef) -> ActionResult:
    """Remove a person from a project (frees their allocation) and queue the offboarding
    workflow for them."""
    rec = run_action(write.end_assignment, person_id=person, project_id=project)
    return ActionResult(
        message=f"{rec['person_name']} removed from {rec['project_name']}; offboarding "
        f"queued (run #{rec['offboarding_run_id']}).",
        record=rec,
    )


@mcp.tool(annotations=ADDITIVE)
def create_task(
    project: ProjectRef,
    title: Annotated[str, Field(min_length=3, max_length=200, description="Short task title.")],
    assignee: Annotated[
        str | None, Field(max_length=80, description="Optional person id or name.")
    ] = None,
    due_date: Annotated[
        str | None,
        Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Optional due date, YYYY-MM-DD."),
    ] = None,
) -> ActionResult:
    """Create a new task (status 'todo') in a project."""
    rec = run_action(
        write.create_task,
        project_id=project, title=title, assignee_id=assignee, due_date=due_date,
    )
    return ActionResult(message=f"Created {rec['id']}: {rec['title']}.", record=rec)


@mcp.tool(annotations=DESTRUCTIVE)
def update_task_status(
    task: TaskRef,
    status: Annotated[write.TaskStatus, Field(description="New task status.")],
) -> ActionResult:
    """Change the status of an existing task (the previous status is returned)."""
    rec = run_action(write.update_task_status, task_id=task, status=status)
    return ActionResult(
        message=f"{rec['id']} moved from {rec['previous_status']} to {rec['status']}.",
        record=rec,
    )


@mcp.tool(annotations=ADDITIVE)
def trigger_workflow(
    workflow: Annotated[write.Workflow, Field(description="Workflow to start for the project.")],
    project: ProjectRef,
) -> ActionResult:
    """Request an operational workflow (kickoff, offboarding or escalation) for a project.

    The request is recorded in the workflow queue with status 'queued' for the operations team;
    this tool does not run the workflow itself.
    """
    rec = run_action(write.trigger_workflow, name=workflow, project_id=project)
    return ActionResult(
        message=f"Workflow '{rec['workflow']}' queued for {rec['project_name']} "
        f"(run #{rec['run_id']}).",
        record=rec,
    )


def main() -> None:
    settings = load_settings()
    ensure_db(settings.db_path, settings.seed_dir)
    mcp.run()
