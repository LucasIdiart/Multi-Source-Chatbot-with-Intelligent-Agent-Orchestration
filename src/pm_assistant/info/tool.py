"""Expose the informational graph to the agent as a single read-only tool."""

from __future__ import annotations

from langchain_core.tools import BaseTool, ToolException, tool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field


class KnowledgeQuery(BaseModel):
    question: str = Field(
        description=(
            "A standalone question in English, with names and context from the conversation "
            "resolved (e.g. 'What skills does Chloe Nguyen have?' rather than 'what about her?')."
        )
    )


def make_info_tool(subgraph: CompiledStateGraph) -> BaseTool:
    @tool("ask_knowledge_base", args_schema=KnowledgeQuery)
    def ask_knowledge_base(question: str) -> str:
        """Answer any question that seeks information about the organisation: people, skills,
        availability, projects, clients, assignments, tasks, workflows, policies and processes.
        Searches the document corpus (RAG) and the knowledge graph together and returns a cited
        answer. Read-only: it never changes any data."""
        try:
            return subgraph.invoke({"question": question})["answer"]
        except Exception as exc:  # surfaced to the model as a tool error, never a crash
            raise ToolException(f"Knowledge base unavailable: {exc}") from exc

    ask_knowledge_base.handle_tool_error = True
    return ask_knowledge_base


def default_info_tool(settings=None) -> BaseTool:
    from .subgraph import default_info_subgraph

    return make_info_tool(default_info_subgraph(settings))
