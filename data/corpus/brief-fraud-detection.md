# Project Brief — Fraud Detection Engine (PRJ-03)

## Goal
Score card transactions for Cedar Bank in under 50 milliseconds and flag likely fraud, with
explanations that the bank's auditors can review.

## Scope
Gradient-boosted model plus rule overrides, a feature store fed from the transaction stream, and an
explanation report per flagged transaction. Model decisions must be reproducible for audits.

## Staffing needs
The project is planned to start on 15 November 2026. It needs a machine learning lead (already
assigned), a security engineer for threat modelling of the scoring API, and a second Python engineer
for the feature pipeline. Because Cedar Bank is a regulated fintech client, the team must include
someone with the compliance skill.

## Dependencies
The labelled fraud dataset must arrive from Cedar Bank before model training starts.
