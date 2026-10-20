# Workflows Handbook

## Kickoff workflow
Triggered when a project moves from planned to active or when a new contractor joins a project.
It creates the project channel, schedules the kickoff meeting within five business days, requests
the client's data processing addendum for regulated engagements, and provisions repository access.

## Offboarding workflow
Triggered when someone leaves a project. Access to repositories and cloud accounts is revoked within
24 hours and the delivery lead receives a handover checklist.

## Escalation workflow
Triggered when a project is at risk: a blocked task older than one week on the critical path, a
missed client milestone, or a security finding rated high. It notifies the engineering manager and
the client's account owner and opens a risk review within two business days.

## Task status definitions
- todo: accepted but not started.
- in progress: someone is actively working on it.
- blocked: cannot progress because of an external dependency; the blocker must be written in the
  task comments.
- done: merged, deployed where applicable, and accepted by the delivery lead.
