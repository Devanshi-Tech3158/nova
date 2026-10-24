"""Anchor facade integrating extraction, storage, supersession, and retrieval."""

import logging
from typing import Any

from nova.core.protocols import Extractor, MemoryStore, Retriever
from nova.core.schemas import IngestReport, Message
from nova.extraction.quote_validator import validate_quotes
from nova.retrieval.embedder import serialize_embedding
from nova.temporal.supersession import apply as apply_supersession

logger = logging.getLogger("nova.pipeline")


class Anchor:
    """Anchor facade providing unified ingestion, query answering, and verification."""

    def __init__(
        self,
        store: MemoryStore,
        extractor: Extractor,
        retriever: Retriever,
        audit_log: Any = None,
    ) -> None:
        """Initialize Anchor with underlying store, extractor, and retriever."""
        self.store = store
        self.extractor = extractor
        self.retriever = retriever
        self.audit_log = audit_log
        self._ingested_msg_ids: set[str] = set()

    def ingest(self, session_id: str, messages: list[Message]) -> IngestReport:
        """Extract, validate, store, supersede, and embed memories from conversation turns."""
        if not messages:
            return IngestReport(added=0, superseded=0, dropped=0)

        normalized_messages = [
            msg
            if msg.session_id == session_id
            else msg.model_copy(update={"session_id": session_id})
            for msg in messages
        ]

        msg_ids = [m.id for m in normalized_messages]
        if msg_ids and all(m_id in self._ingested_msg_ids for m_id in msg_ids):
            logger.info(
                "ingest skipped: duplicate messages",
                extra={
                    "session_id": session_id,
                    "messages_count": len(messages),
                    "added": 0,
                    "superseded": 0,
                    "dropped": 0,
                },
            )
            return IngestReport(added=0, superseded=0, dropped=0)

        raw_memories = self.extractor.extract(normalized_messages)
        valid_memories, dropped_count = validate_quotes(raw_memories, normalized_messages)

        new_memories = [m for m in valid_memories if self.store.get(m.id) is None]

        if new_memories:
            self.store.add_many(new_memories)

        superseded_count = apply_supersession(self.store, valid_memories)

        if hasattr(self.retriever, "embedder") and new_memories:
            embedder = self.retriever.embedder
            for mem in new_memories:
                emb = embedder.embed(f"{mem.subject} {mem.value}")
                blob = serialize_embedding(emb)
                if hasattr(self.store, "conn") and self.store.conn is not None:
                    self.store.conn.execute(
                        "UPDATE memories SET embedding = ? WHERE id = ?;",
                        (blob, mem.id),
                    )

        for m_id in msg_ids:
            self._ingested_msg_ids.add(m_id)

        report = IngestReport(
            added=len(new_memories),
            superseded=superseded_count,
            dropped=dropped_count,
        )

        logger.info(
            "ingest completed",
            extra={
                "session_id": session_id,
                "messages_count": len(messages),
                "added": report.added,
                "superseded": report.superseded,
                "dropped": report.dropped,
            },
        )
        return report

    def ask(self, question: str, draft_answer: str | None = None) -> Any:
        """Answer question from verified memories (stub for Person 2)."""
        raise NotImplementedError("ask() will be implemented by Person 2")

    def verify(self, question: str, answer: str) -> Any:
        """Verify an answer against retrieved memories (stub for Person 2)."""
        raise NotImplementedError("verify() will be implemented by Person 2")

    def get_audit(self, audit_id: str) -> Any:
        """Retrieve audit record by id."""
        if self.audit_log is not None:
            return self.audit_log.get(audit_id)
        return None
