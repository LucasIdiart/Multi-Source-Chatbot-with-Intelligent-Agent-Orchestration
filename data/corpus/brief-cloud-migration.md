# Project Brief — Cloud Migration (PRJ-05)

## Goal
Move Cedar Bank's core reporting workloads from the on-prem data centre to AWS by January 2027.

## Approach
Infrastructure is defined with Terraform modules and deployed to Kubernetes clusters. Data transfer
goes through an encrypted private link. A threat model of the data transfer must be approved by the
bank's security office before the first production cut-over.

## Status
The reporting VPC Terraform modules are in progress. The threat model is not started yet and is due
on 10 November 2026.
