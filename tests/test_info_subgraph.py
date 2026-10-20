"""EC1 — informational queries flow through one centralized graph that combines RAG and KG."""

import pytest
from conftest import info_tool
from fakes import RunRecorder, fake_synth
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from pm_assistant.info.kg import retrieve_facts
from pm_assistant.info.rag import BM25Retriever, default_retriever, load_corpus
from pm_assistant.info.subgraph import default_info_subgraph, make_info_graph


def test_topology_runs_rag_and_kg_in_parallel_then_synthesizes(settings):
    g = default_info_subgraph(settings, synth_model=fake_synth()).get_graph()
    assert set(g.nodes) - {"__start__", "__end__"} == {"rag_retrieve", "kg_retrieve", "synthesize"}
    edges = {(e.source, e.target) for e in g.edges}
    assert {("__start__", "rag_retrieve"), ("__start__", "kg_retrieve")} <= edges
    assert {("rag_retrieve", "synthesize"), ("kg_retrieve", "synthesize")} <= edges
    assert all(not e.conditional for e in g.edges)


def fact(subject_prefix: str, predicate: str, obj: str):
    return lambda t: (t["subject"].startswith(subject_prefix) and t["predicate"] == predicate
                      and t["object"] == obj)


# question -> (a document that must be retrieved, facts the graph must return)
RELEVANCE = {
    "Who leads the Cloud Migration and what does it involve?": (
        "brief-cloud-migration",
        [fact("Farid Haddad (P-006)", "LEADS", "Cloud Migration (PRJ-05)")],
    ),
    "What is the policy for contractor hourly rates above 100 USD?": (
        "staffing-policy",
        [fact("Farid Haddad (P-006)", "hourly_rate_usd", "100")],
    ),
    # Aggregate questions name no entity: every person's fact must be reachable.
    "Who has the most free capacity right now?": (
        None,
        [fact("Emily Okafor (P-005)", "free_capacity_pct", "70"),
         fact("Laura Rossi (P-012)", "free_capacity_pct", "60")],
    ),
    "Which contractors have free capacity?": (
        None,
        [fact("Isabel Torres (P-009)", "employment_type", "contractor"),
         fact("Isabel Torres (P-009)", "free_capacity_pct", "40")],
    ),
    # Multi-hop: project -> assigned people -> their attributes.
    "Which people assigned to the Cloud Migration project are contractors?": (
        "brief-cloud-migration",
        [fact("Jamal Wright (P-010)", "employment_type", "employee"),
         fact("Farid Haddad (P-006)", "employment_type", "contractor")],
    ),
}


@pytest.mark.parametrize("question", RELEVANCE)
def test_both_retrievers_return_the_relevant_evidence(settings, question):
    doc, facts = RELEVANCE[question]
    state = default_info_subgraph(settings, synth_model=fake_synth()).invoke({"question": question})
    if doc:
        assert doc in [c["doc_id"] for c in state["chunks"]]
    for matches in facts:
        assert any(matches(t) for t in state["triples"]), question
    assert state["answer"]


def test_three_hop_question_keeps_the_linking_relations(settings):
    """client -> project -> assigned people -> skill: the assignment edges must survive the cap."""
    facts = retrieve_facts(
        "Which projects for regulated clients have nobody with the compliance skill on the team?",
        settings.db_path,
    )
    assert any(fact("Jamal Wright (P-010)", "HAS_SKILL {'level': 4}", "compliance")(t)
               for t in facts)
    assert any(t["subject"] == "Jamal Wright (P-010)" and t["predicate"].startswith("ASSIGNED_TO")
               and t["object"] == "Cloud Migration (PRJ-05)" for t in facts)


def test_aggregate_question_returns_every_person(settings):
    facts = retrieve_facts("Who has the most free capacity right now?", settings.db_path)
    people = {t["subject"] for t in facts if t["predicate"] == "free_capacity_pct"}
    assert len(people) == 12


@pytest.mark.parametrize("question", ["What did we decide in the last meeting?",
                                      "Summarize the latest meeting notes."])
def test_documents_are_found_by_their_file_name_words(settings, question):
    chunks = default_retriever(str(settings.corpus_dir)).retrieve(question, 4)
    assert "meeting-notes-2026-10-12" in [c["doc_id"] for c in chunks]


def test_unrelated_question_retrieves_nothing(settings):
    retriever = default_retriever(str(settings.corpus_dir))
    assert retriever.retrieve("zzqx vbnm plokij", 4) == []
    assert retrieve_facts("zzqx vbnm plokij", settings.db_path) == []


def test_plural_and_spanish_questions_still_match(settings):
    retriever = default_retriever(str(settings.corpus_dir))
    assert [c["doc_id"] for c in retriever.retrieve("What are our staffing policies?", 4)] == [
        "staffing-policy"] * 4
    assert retriever.retrieve("¿Quién sabe Kubernetes?", 4)


def test_synthesizer_receives_both_sources(settings):
    seen = []

    class Spy:
        def invoke(self, messages):
            seen.append(messages[-1].content)
            return fake_synth().invoke(messages)

    default_info_subgraph(settings, synth_model=Spy()).invoke(
        {"question": "Who leads the Cloud Migration?"}
    )
    assert "[doc:brief-cloud-migration]" in seen[0]
    assert "[kg] Farid Haddad (P-006) --LEADS--> Cloud Migration (PRJ-05)" in seen[0]


def test_tool_runs_the_subgraph_nested_in_its_trace(settings):
    recorder = RunRecorder()
    info_tool(settings).invoke({"question": "Who leads Patient Portal?"},
                               config={"callbacks": [recorder]})
    assert recorder.count("info_subgraph") == 1
    assert recorder.ancestors("synthesize")[:2] == ["info_subgraph", "ask_knowledge_base"]


def test_tool_reports_failures_as_tool_errors(settings):
    class Broken:
        def invoke(self, messages):
            raise RuntimeError("model overloaded")

    tool = info_tool(settings, synth=Broken())
    msg = tool.invoke(
        {"type": "tool_call", "name": tool.name, "id": "c1", "args": {"question": "x?"}}
    )
    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    assert "model overloaded" in msg.content


def test_works_before_the_store_exists_without_writing_it(settings, tmp_path, monkeypatch):
    missing = tmp_path / "not-created-yet" / "pm.db"
    monkeypatch.setenv("PM_DB_PATH", str(missing))
    facts = retrieve_facts("Who leads the Cloud Migration?", missing, seed_dir=settings.seed_dir)
    assert any(fact("Farid Haddad (P-006)", "LEADS", "Cloud Migration (PRJ-05)")(t) for t in facts)
    assert not missing.exists()


def test_each_run_starts_from_fresh_evidence_on_a_persisted_thread(settings):
    graph = default_info_subgraph(settings, synth_model=fake_synth()).copy(
        update={"checkpointer": InMemorySaver()}
    )
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"question": "Who leads the Cloud Migration?"}, config)
    second = graph.invoke({"question": "What is the policy for contractor rates?"}, config)
    assert "brief-cloud-migration" not in [c["doc_id"] for c in second["chunks"]]


def test_empty_corpus_and_empty_files_do_not_crash(tmp_path):
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    (tmp_path / "blank-start.md").write_text("\n## Heading\nSome text.", encoding="utf-8")
    assert [c["title"] for c in load_corpus(tmp_path)] == ["Heading / Heading"]
    assert BM25Retriever([]).retrieve("anything", 4) == []


def test_langgraph_studio_factory_builds_the_pipeline():
    g = make_info_graph().get_graph()
    assert {"rag_retrieve", "kg_retrieve", "synthesize"} <= set(g.nodes)
