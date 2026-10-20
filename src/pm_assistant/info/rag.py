"""Document retrieval (the RAG half of the informational pipeline).

The default retriever is BM25: deterministic, offline and dependency-light. Anything that
implements ``Retriever`` (e.g. an embeddings-backed vector store) can be swapped in.
"""

from __future__ import annotations

import heapq
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol, TypedDict

from rank_bm25 import BM25Plus

MAX_CHUNK_CHARS = 800
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its me my of on or "
    "our the their them there they this to was we what when where which who whom why will with "
    "would you your".split()
)
TOKEN_RE = re.compile(r"\w+(?:-\w+)*")


class Chunk(TypedDict):
    doc_id: str
    title: str
    text: str
    score: float


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Chunk]: ...


def _fold(token: str) -> str:
    """Light English plural folding: policies→policy, processes→process, contractors→contractor."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "xes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (any script) without stopwords, plural-folded. A hyphenated word
    also yields its parts, so "over-allocated" matches "allocated" and "notes" in a file name."""
    words = TOKEN_RE.findall(text.casefold())
    tokens = words + [part for w in words for part in w.split("-")[1:] + w.split("-")[:1]
                      if "-" in w]
    return [_fold(t) for t in tokens if t and t not in STOPWORDS]


class BM25Index:
    """BM25 over tokenized documents. A document is a hit only if it shares a query term."""

    def __init__(self, docs: Sequence[list[str]]):
        self.docs = [set(d) for d in docs]
        # BM25Plus: IDF stays positive even for terms in half the documents (Okapi gives 0).
        self.bm25 = BM25Plus(docs) if docs else None

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        terms = set(tokenize(query))
        candidates = [i for i, doc in enumerate(self.docs) if terms & doc]
        if not candidates or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(list(terms))
        best = heapq.nlargest(k, candidates, key=lambda i: scores[i])
        return [(i, round(float(scores[i]), 3)) for i in best]


def _split(text: str) -> list[str]:
    parts, current = [], ""
    for para in text.split("\n\n"):
        if current and len(current) + len(para) > MAX_CHUNK_CHARS:
            parts.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    """Split every markdown file on '## ' headings, then into ~800-character pieces."""
    chunks: list[Chunk] = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            continue
        doc_title = text.splitlines()[0].lstrip("# ").strip()
        for section in re.split(r"\n(?=## )", text):
            if not section.strip():
                continue
            heading = section.strip().splitlines()[0].lstrip("# ").strip()
            for part in _split(section):
                chunks.append(
                    Chunk(doc_id=path.stem, title=f"{doc_title} / {heading}", text=part, score=0.0)
                )
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # The file name is indexed too: "meeting-notes-2026-10-12" is how users refer to a
        # document whose title says "Weekly Delivery Review".
        self.index = BM25Index(
            [tokenize(f"{c['doc_id']}\n{c['title']}\n{c['text']}") for c in chunks]
        )

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        return [Chunk(**{**self.chunks[i], "score": s}) for i, s in self.index.top_k(query, k)]


@lru_cache(maxsize=4)
def default_retriever(corpus_dir: str) -> BM25Retriever:
    return BM25Retriever(load_corpus(Path(corpus_dir)))
