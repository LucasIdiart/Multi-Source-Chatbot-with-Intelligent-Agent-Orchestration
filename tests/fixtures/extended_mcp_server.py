"""The real action server plus one extra tool, to prove the agent adapts with no code changes."""

import json
from typing import Annotated, Literal

from pydantic import Field

from pm_assistant.mcp_server.server import (
    ADDITIVE,
    ActionResult,
    ProjectRef,
    main,
    mcp,
    run_action,
)
from pm_assistant.store import write


def _escalate(conn, project_id: str, severity: str, summary: str) -> dict:
    """One transaction: the escalation run and its audit entry are written together."""
    with write.transaction(conn):
        project = write._open_project(conn, project_id)
        run = conn.execute(
            "INSERT INTO workflow_runs(name, project_id, payload_json) VALUES (?, ?, ?)",
            ("escalation", project["id"], json.dumps({"severity": severity, "summary": summary})),
        )
        record = {"run_id": run.lastrowid, "project_id": project["id"],
                  "project_name": project["name"], "severity": severity, "summary": summary}
        write.log_action(conn, "escalate_risk", {"project": project_id, "severity": severity},
                         record)
    return record


@mcp.tool(annotations=ADDITIVE)
def escalate_risk(
    project: ProjectRef,
    severity: Annotated[Literal["low", "medium", "high"], Field(description="Risk severity.")],
    summary: Annotated[str, Field(min_length=5, max_length=500, description="What the risk is.")],
) -> ActionResult:
    """Escalate a delivery risk on a project to the engineering manager."""
    rec = run_action(_escalate, project_id=project, severity=severity, summary=summary)
    return ActionResult(message=f"Risk escalated on {rec['project_name']}.", record=rec)


if __name__ == "__main__":
    main()
