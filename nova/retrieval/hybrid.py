"""Hybrid retriever combining vector cosine similarity and keyword overlap."""

import re
from typing import Any

import numpy as np

from nova.core.protocols import MemoryStore
from nova.core.schemas import Memory
from nova.retrieval.embedder import Embedder, deserialize_embedding


class HybridRetriever:
    """Hybrid retriever scoring memories by 0.6*cosine + 0.4*keyword overlap."""

    def __init__(self, store: MemoryStore, embedder: Embedder) -> None:
        """Initialize hybrid retriever with store and embedder."""
        self.store = store
        self.embedder = embedder

    def _get_candidates(
        self,
        include_history: bool,
        session_id: str | None,
    ) -> list[tuple[Memory, Any]]:
        """Retrieve candidate memories and their embedding blobs from the store."""
        if hasattr(self.store, "conn") and self.store.conn is not None:
            sql = "SELECT id, type, subject, value, reason, source_msg_id, source_quote, session_id, timestamp, status, supersedes_id, confidence, embedding FROM memories"
            clauses = []
            params: list[Any] = []
            if not include_history:
                clauses.append("status = 'current'")
            if session_id is not None:
                clauses.append("session_id = ?")
                params.append(session_id)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)

            rows = self.store.conn.execute(sql, params).fetchall()
            candidates: list[tuple[Memory, Any]] = []
            for r in rows:
                mem = self.store._row_to_memory(r)
                candidates.append((mem, r["embedding"]))
            return candidates

        return []

    def retrieve(
        self,
        query: str,
        k: int = 5,
        include_history: bool = False,
        session_id: str | None = None,
    ) -> list[Memory]:
        """Retrieve top-k memories using hybrid cosine and keyword overlap scoring."""
        candidates = self._get_candidates(include_history, session_id)
        if not candidates:
            return []

        q_vec = np.asarray(self.embedder.embed(query), dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        q_tokens = set(re.findall(r"\w+", query.lower()))

        scored: list[tuple[float, Memory]] = []
        for mem, emb_blob in candidates:
            if emb_blob is not None:
                cand_vec = deserialize_embedding(emb_blob)
            else:
                cand_vec = np.asarray(
                    self.embedder.embed(f"{mem.subject} {mem.value}"),
                    dtype=np.float32,
                )

            cand_norm = float(np.linalg.norm(cand_vec))
            if q_norm > 0.0 and cand_norm > 0.0:
                cosine = max(0.0, float(np.dot(q_vec, cand_vec) / (q_norm * cand_norm)))
            else:
                cosine = 0.0

            content_tokens = set(
                re.findall(r"\w+", f"{mem.subject} {mem.value} {mem.source_quote}".lower())
            )
            token_overlap = (
                len(q_tokens & content_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0
            )

            subj_words = set(re.findall(r"\w+", mem.subject.lower()))
            subj_match = (
                1.0 if (q_tokens & subj_words or mem.subject.lower() in query.lower()) else 0.0
            )
            keyword_score = min(1.0, 0.7 * token_overlap + 0.3 * subj_match)

            score = 0.6 * cosine + 0.4 * keyword_score
            scored.append((score, mem))

        scored.sort(key=lambda item: (-item[0], -item[1].timestamp.timestamp(), item[1].id))
        return [mem for _, mem in scored[:k]]
