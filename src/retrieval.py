"""Dense, lexical, and hybrid retrieval with a common traceable result contract."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from .day6.embeddings import MyEmbedder
from .day6.store import QdrantLocalStore


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


class Retriever(Protocol):
    name: str
    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]: ...


class LexicalRetriever:
    name = "tfidf"
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        counts = [Counter(tokenize(c["text"])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + n)) + 1 for term, n in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        raw = {term: n * self.idf[term] for term, n in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
        return {term: v / norm for term, v in raw.items()}

    @staticmethod
    def _matches(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        return all(chunk["metadata"].get(key) == value for key, value in (filters or {}).items())

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        qv = self._vector(Counter(tokenize(query))); found = []
        for chunk, vector in zip(self.chunks, self.vectors, strict=True):
            if not self._matches(chunk, filters): continue
            score = sum(v * vector.get(term, 0.0) for term, v in qv.items())
            if score > 0: found.append({**chunk, "score": score, "strategy": self.name})
        return sorted(found, key=lambda x: (-x["score"], x["id"]))[:top_k]


class DenseRetriever:
    name = "dense"
    def __init__(self, store_path: Path, model_name: str, strategy: str = "paragraph") -> None:
        self.embedder = MyEmbedder(model_name)
        self.store = QdrantLocalStore(store_path, self.embedder.dimensions)
        self.strategy = strategy

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        rows = self.store.search(self.strategy, self.embedder.embed_query(query), top_k=top_k, filters=filters)
        return [{"id": r["chunk_id"], "text": r["text"], "source_id": r["metadata"]["document_id"],
                 "source": r["metadata"]["source"], "page": r["metadata"]["page"],
                 "metadata": r["metadata"], "score": r["score"], "strategy": self.name} for r in rows]

    def close(self) -> None: self.store.close()


class HybridRetriever:
    name = "hybrid_rrf"
    def __init__(self, dense: DenseRetriever, lexical: LexicalRetriever, rrf_k: int = 60) -> None:
        self.dense, self.lexical, self.rrf_k = dense, lexical, rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        pool = max(top_k * 4, 12); merged: dict[str, dict[str, Any]] = {}
        for results in (self.dense.search(query, filters=filters, top_k=pool), self.lexical.search(query, filters=filters, top_k=pool)):
            for rank, row in enumerate(results, start=1):
                item = merged.setdefault(row["id"], {**row, "score": 0.0, "strategy": self.name})
                item["score"] += 1 / (self.rrf_k + rank)
        return sorted(merged.values(), key=lambda x: (-x["score"], x["id"]))[:top_k]
