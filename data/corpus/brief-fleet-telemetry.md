# Project Brief — Fleet Telemetry Platform (PRJ-02)

## Goal
Ingest GPS and engine telemetry from about 4,000 Bluefin Logistics trucks into a lakehouse so the
client can predict maintenance and optimise routes.

## Architecture
Devices publish events to a managed Kafka cluster on AWS. Spark structured streaming jobs running on
Kubernetes clean the events and write Delta tables. A device registry in PostgreSQL maps trucks to
sensors.

## Status
The device registry migration to PostgreSQL is done. The six-month backfill of historical engine
telemetry is in progress and is the critical path for the December 2026 delivery. The ingestion API
must sustain 20,000 events per second; load testing is scheduled for early November.

## Team notes
The data engineer is fully allocated to this project, so any new data work on other projects needs
a different person.
