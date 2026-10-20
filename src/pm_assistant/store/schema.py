"""SQLite schema for the PM records."""

from __future__ import annotations

import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    title TEXT NOT NULL,
    seniority TEXT NOT NULL,
    employment_type TEXT NOT NULL CHECK (employment_type IN ('employee', 'contractor')),
    location TEXT NOT NULL,
    hourly_rate_usd INTEGER NOT NULL,
    availability_pct INTEGER NOT NULL CHECK (availability_pct BETWEEN 1 AND 100)
);
CREATE TABLE IF NOT EXISTS skills (name TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS person_skills (
    person_id TEXT NOT NULL REFERENCES people(id),
    skill TEXT NOT NULL REFERENCES skills(name),
    level INTEGER NOT NULL CHECK (level BETWEEN 1 AND 5),
    PRIMARY KEY (person_id, skill)
);
CREATE TABLE IF NOT EXISTS clients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    industry TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    client_id TEXT NOT NULL REFERENCES clients(id),
    status TEXT NOT NULL CHECK (status IN ('planned', 'active', 'on_hold', 'closed')),
    start_date TEXT,
    end_date TEXT,
    lead_id TEXT REFERENCES people(id),
    description TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS project_skills (
    project_id TEXT NOT NULL REFERENCES projects(id),
    skill TEXT NOT NULL REFERENCES skills(name),
    PRIMARY KEY (project_id, skill)
);
CREATE TABLE IF NOT EXISTS assignments (
    person_id TEXT NOT NULL REFERENCES people(id),
    project_id TEXT NOT NULL REFERENCES projects(id),
    role TEXT NOT NULL,
    allocation_pct INTEGER NOT NULL CHECK (allocation_pct BETWEEN 1 AND 100),
    assigned_at TEXT NOT NULL,
    PRIMARY KEY (person_id, project_id)
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    title TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('todo', 'in_progress', 'blocked', 'done')),
    assignee_id TEXT REFERENCES people(id),
    due_date TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS workflow_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL CHECK (name IN ('kickoff', 'offboarding', 'escalation')),
    project_id TEXT NOT NULL REFERENCES projects(id),
    status TEXT NOT NULL DEFAULT 'queued',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS action_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
