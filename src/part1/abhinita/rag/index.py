"""Tasks 1.1 (store) and 1.2 (hybrid retrieval): ChromaDB dense index + BM25 keyword index.

* Dense: sentence-transformers embeddings (bge-small) stored in ChromaDB with the metadata the
  lab asks for (doc_id, page, figure_id, trust_level). Falls back to a numpy cosine search if
  chromadb is not installed.
* Keyword: a small self-contained BM25 (no extra dependency).
* Fusion: Reciprocal Rank Fusion of the two rankings.
* The index is READ-ONLY after build(): conversations never write to it (see Task 1.3).
* Low-trust items are stored (the lab wants trust_level stored) but are never returned.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

import numpy as np

from .config import EMBED_NAME, RRF_K, TOP_K_BM25, TOP_K_DENSE
from .schema import Item

log = logging.getLogger("rag.index")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def index_text(item: Item) -> str:
    """Text that represents an item in the indexes: its own text + any VLM figure transcription."""
    extra = item.meta.get("transcription", "")
    return f"{item.text}\n{extra}" if extra else item.text


class BM25:
    """Okapi BM25 over a fixed corpus."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.tf = [Counter(d) for d in docs]
        self.dl = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = self.dl.mean() if len(docs) else 1.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.docs))
        for t in set(query):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return s


class Embedder:
    """Thin wrapper so the rest of the code does not care which library embeds the text."""

    def __init__(self, name: str = EMBED_NAME):
        from sentence_transformers import SentenceTransformer  # imported lazily: heavy
        self.model = SentenceTransformer(name)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if is_query:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


class HybridIndex:
    def __init__(self, items: list[Item], embedder: Embedder | None = None, use_chroma: bool = True):
        self.items = items
        self.embedder = embedder
        self.by_id = {it.item_id: i for i, it in enumerate(items)}
        # Low-trust items stay in the store but are masked out of every search.
        self.searchable = np.array([it.trust != "low" for it in items])
        self.bm25 = BM25([tokenize(index_text(it)) for it in items])
        self.collection = None
        self.emb = None
        if embedder is not None:
            self.emb = embedder.encode([index_text(it) for it in items])
            if use_chroma:
                self._build_chroma()

    # ------------------------------------------------------------------ store
    def _build_chroma(self) -> None:
        try:
            import chromadb
            client = chromadb.Client()  # in-memory: rebuilt on every run, nothing persisted to git
            try:
                client.delete_collection("kb")
            except Exception:
                pass
            col = client.create_collection("kb", metadata={"hnsw:space": "cosine"})
            col.add(
                ids=[it.item_id for it in self.items],
                documents=[index_text(it) for it in self.items],
                embeddings=self.emb.tolist(),
                metadatas=[{"doc_id": it.doc_id, "page": it.page, "figure_id": it.label,
                            "trust_level": it.trust, "kind": it.kind} for it in self.items],
            )
            self.collection = col
        except Exception as e:  # fail soft: numpy cosine below gives identical results
            log.warning("ChromaDB unavailable (%s); using numpy cosine instead", e)

    def count(self) -> int:
        """Number of stored records - used to prove conversations never write to the base index."""
        return self.collection.count() if self.collection is not None else len(self.items)

    # ----------------------------------------------------------------- search
    def dense_ranking(self, query: str, k: int = TOP_K_DENSE) -> list[int]:
        if self.embedder is None:
            return []
        q = self.embedder.encode([query], is_query=True)
        if self.collection is not None:
            try:
                res = self.collection.query(query_embeddings=q.tolist(), n_results=min(k, len(self.items)),
                                            where={"trust_level": {"$ne": "low"}})
                return [self.by_id[i] for i in res["ids"][0]]
            except Exception as e:
                log.warning("Chroma query failed (%s); falling back to numpy", e)
        sims = self.emb @ q[0]
        sims = np.where(self.searchable, sims, -1e9)
        return list(np.argsort(-sims)[:k])

    def bm25_ranking(self, query: str, k: int = TOP_K_BM25) -> list[int]:
        s = self.bm25.scores(tokenize(query))
        s = np.where(self.searchable, s, -1e9)
        order = np.argsort(-s)[:k]
        return [int(i) for i in order if s[i] > 0]

    def search(self, query: str, mode: str = "hybrid") -> list[tuple[Item, float]]:
        """mode: 'hybrid' (dense + BM25, RRF) or 'dense' (baseline for the evaluation table)."""
        dense = self.dense_ranking(query)
        if mode == "dense" and dense:
            return [(self.items[i], 1.0 / (RRF_K + r + 1)) for r, i in enumerate(dense)]
        sparse = self.bm25_ranking(query)
        score: dict[int, float] = {}
        for ranking in (dense, sparse):
            for r, i in enumerate(ranking):
                score[i] = score.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        order = sorted(score, key=score.get, reverse=True)
        return [(self.items[i], score[i]) for i in order]
