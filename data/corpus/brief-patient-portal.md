# Project Brief — Patient Portal Revamp (PRJ-01)

## Goal
Replace Aurora Health's ten-year-old patient portal with an accessible, mobile-friendly React
application. Patients must be able to book appointments, read lab results and message their care team.

## Constraints
The portal must meet WCAG 2.2 AA. Every access to patient records must be written to an immutable
audit log to satisfy HIPAA requirements. The audit log design is the main technical risk.

## Milestones
- Accessible booking flow in staging by 7 November 2026.
- External accessibility audit by mid-November 2026.
- Public launch planned for February 2027.

## Open risks
The HIPAA audit log schema task is blocked: the client has not yet confirmed its log retention period
and nobody on the team currently owns compliance for this project. Adding someone with compliance
experience is recommended.
