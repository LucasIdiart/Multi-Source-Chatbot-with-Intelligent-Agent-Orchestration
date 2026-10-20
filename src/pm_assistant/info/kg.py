"""Knowledge Graph over the structured PM records (the KG half of the informational pipeline).

The graph is rebuilt from the read-only store on every query, so changes made through the MCP
action tools are visible immediately. Facts for a question are found by *retrieval*, never by
keyword rules, combining three signals:

1. entities the question names explicitly (an id, or every word of a name);
2. entities whose profile (label, attributes, relations) ranks highest under BM25;
3. individual facts that rank highest under BM25 — this is what answers aggregate questions
   ("who has the most free capacity?", "list the contractors") that name no entity.

Entities from 1 and 2 are expanded `hops` relations outward, with the attributes of every
entity reached.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import networkx as nx

from pm_assistant.store.read import connect_ro, fetch_graph_data

from .rag import BM25Index, tokenize

MAX_SEEDS = 5
MAX_FACT_HITS = 40
MAX_TRIPLES = 250  # ~5k tokens of evidence; relations are ordered before attributes
# Retrieval vocabulary for entity types, so "people", "staff" or "team" find person facts.
TYPE_TERMS = {
    "person": "person people staff team member colleague",
    "project": "project engagement",
    "client": "client customer account",
    "task": "task work item",
    "skill": "skill expertise",
    "workflow_run": "workflow run",
}


class Triple(TypedDict):
    subject: str
    predicate: str
    object: str


def _label(name: str, ident: str) -> str:
    return f"{name} ({ident})"


def build_kg(data: dict[str, list[dict]]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    allocated: dict[str, int] = {}
    for a in data["assignments"]:
        allocated[a["person_id"]] = allocated.get(a["person_id"], 0) + a["allocation_pct"]

    def entity(node: str, kind: str, name: str, ident: str, **attrs) -> None:
        G.add_node(node, type=kind, name=name, ident=ident, label=_label(name, ident), attrs=attrs)

    for p in data["people"]:
        used = allocated.get(p["id"], 0)
        entity(f"person:{p['id']}", "person", p["name"], p["id"], title=p["title"],
               seniority=p["seniority"], employment_type=p["employment_type"],
               location=p["location"], hourly_rate_usd=p["hourly_rate_usd"],
               contracted_availability_pct=p["availability_pct"], allocated_pct=used,
               free_capacity_pct=p["availability_pct"] - used)
    for c in data["clients"]:
        entity(f"client:{c['id']}", "client", c["name"], c["id"], industry=c["industry"])
    for pr in data["projects"]:
        entity(f"project:{pr['id']}", "project", pr["name"], pr["id"], status=pr["status"],
               start_date=pr["start_date"], end_date=pr["end_date"],
               description=pr["description"])
        G.add_edge(f"project:{pr['id']}", f"client:{pr['client_id']}", predicate="FOR_CLIENT")
        if pr["lead_id"]:
            G.add_edge(f"person:{pr['lead_id']}", f"project:{pr['id']}", predicate="LEADS")
    skills = {r["skill"] for r in data["person_skills"]}
    skills |= {r["skill"] for r in data["project_skills"]}
    for s in skills:
        G.add_node(f"skill:{s}", type="skill", name=s, ident=s, label=s, attrs={})
    for r in data["person_skills"]:
        G.add_edge(f"person:{r['person_id']}", f"skill:{r['skill']}", predicate="HAS_SKILL",
                   level=r["level"])
    for r in data["project_skills"]:
        G.add_edge(f"project:{r['project_id']}", f"skill:{r['skill']}", predicate="REQUIRES_SKILL")
    for a in data["assignments"]:
        G.add_edge(f"person:{a['person_id']}", f"project:{a['project_id']}",
                   predicate="ASSIGNED_TO", role=a["role"], allocation_pct=a["allocation_pct"])
    for t in data["tasks"]:
        entity(f"task:{t['id']}", "task", t["title"], t["id"], status=t["status"],
               due_date=t["due_date"])
        G.add_edge(f"project:{t['project_id']}", f"task:{t['id']}", predicate="HAS_TASK")
        if t["assignee_id"]:
            G.add_edge(f"person:{t['assignee_id']}", f"task:{t['id']}", predicate="ASSIGNEE_OF")
    for w in data["workflow_runs"]:
        node = f"workflow:{w['id']}"
        G.add_node(node, type="workflow_run", name=f"{w['name']} workflow run",
                   ident=f"#{w['id']}", label=f"{w['name']} workflow run #{w['id']}",
                   attrs={"status": w["status"], "created_at_utc": w["created_at"]})
        G.add_edge(f"project:{w['project_id']}", node, predicate="HAS_WORKFLOW")
    return G


def _edge_triple(G: nx.MultiDiGraph, u: str, v: str, data: dict) -> Triple:
    extras = {k: val for k, val in data.items() if k != "predicate"}
    predicate = f"{data['predicate']} {extras}" if extras else data["predicate"]
    return Triple(subject=G.nodes[u]["label"], predicate=predicate, object=G.nodes[v]["label"])


def attribute_triples(G: nx.MultiDiGraph, node: str) -> list[Triple]:
    label = G.nodes[node]["label"]
    return [Triple(subject=label, predicate="IS_A", object=G.nodes[node]["type"])] + [
        Triple(subject=label, predicate=key, object=str(value))
        for key, value in G.nodes[node]["attrs"].items()
    ]


def all_triples(G: nx.MultiDiGraph) -> list[Triple]:
    facts = [t for n in G.nodes for t in attribute_triples(G, n)]
    return facts + [_edge_triple(G, u, v, d) for u, v, d in G.edges(data=True)]


def _fact_text(t: Triple) -> str:
    obj = TYPE_TERMS.get(t["object"], t["object"])
    return f"{t['subject']} {t['predicate']} {obj}".replace("_", " ")


def _profile(G: nx.MultiDiGraph, node: str) -> str:
    """Searchable text for an entity: its type, label, attributes and neighbours."""
    attrs = G.nodes[node]
    # The entity's own name counts more than the names of its neighbours.
    parts = [TYPE_TERMS[attrs["type"]], *[attrs["label"]] * 3]
    parts += [f"{k} {v}" for k, v in attrs["attrs"].items()]
    parts += [f"{d['predicate']} {G.nodes[v]['label']}" for _, v, d in G.out_edges(node, data=True)]
    parts += [f"{d['predicate']} {G.nodes[u]['label']}" for u, _, d in G.in_edges(node, data=True)]
    return " ".join(str(p) for p in parts).replace("_", " ")


def mentioned_entities(G: nx.MultiDiGraph, question: str) -> list[str]:
    """Entities the question names: by id, or by every word of their name."""
    words = set(tokenize(question))

    def named(attrs: dict) -> bool:
        name_words = set(tokenize(attrs["name"]))
        return attrs["ident"].casefold() in words or bool(name_words) and name_words <= words

    return [n for n, a in G.nodes(data=True) if named(a)]


def link_entities(G: nx.MultiDiGraph, question: str, max_seeds: int = MAX_SEEDS) -> list[str]:
    """Named entities first, then the entities whose profile best matches the question."""
    nodes = list(G.nodes)
    index = BM25Index([tokenize(_profile(G, n)) for n in nodes])
    ranked = [nodes[i] for i, _ in index.top_k(question, max_seeds)]
    return list(dict.fromkeys([*mentioned_entities(G, question), *ranked]))


def neighborhood_triples(G: nx.MultiDiGraph, seeds: list[str], hops: int) -> list[Triple]:
    """Every relation within `hops` of the seeds, then the attributes of the entities reached,
    hop by hop: relations of hop 1, attributes of hop-1 entities, relations of hop 2, ...
    Relations come before attributes so multi-hop links survive the evidence cap."""
    triples: list[Triple] = []
    seen_edges: set[tuple[str, str, int]] = set()
    frontier, visited = list(seeds), set(seeds)
    for _ in range(hops):
        reached = []
        for node in frontier:
            edges = [*G.out_edges(node, keys=True, data=True),
                     *G.in_edges(node, keys=True, data=True)]
            for u, v, k, d in edges:
                if (u, v, k) in seen_edges:
                    continue
                seen_edges.add((u, v, k))
                triples.append(_edge_triple(G, u, v, d))
                other = v if u == node else u
                if other not in visited:
                    visited.add(other)
                    reached.append(other)
        triples += [t for n in reached for t in attribute_triples(G, n)]
        frontier = reached
    return triples


def retrieve_from_graph(G: nx.MultiDiGraph, question: str, hops: int = 2) -> list[Triple]:
    seeds = link_entities(G, question)
    facts = all_triples(G)
    index = BM25Index([tokenize(_fact_text(t)) for t in facts])
    fact_hits = [facts[i] for i, _ in index.top_k(question, MAX_FACT_HITS)]
    seed_attrs = [t for s in seeds for t in attribute_triples(G, s)]
    ordered = [*seed_attrs, *fact_hits, *neighborhood_triples(G, seeds, hops)]
    unique = list({(t["subject"], t["predicate"], t["object"]): t for t in ordered}.values())
    return unique[:MAX_TRIPLES]


def retrieve_facts(
    question: str, db_path: Path, hops: int = 2, seed_dir: Path | None = None
) -> list[Triple]:
    conn = connect_ro(db_path, seed_dir)
    try:
        data = fetch_graph_data(conn)
    finally:
        conn.close()
    return retrieve_from_graph(build_kg(data), question, hops)
