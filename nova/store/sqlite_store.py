"""SQLite-backed MemoryStore implementation."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from nova.core.schemas import Memory, MemoryStatus, MemoryType


class StoreError(Exception):
    """Exception raised for storage layer errors."""


class SqliteMemoryStore:
    """MemoryStore backed by SQLite with WAL mode and schema versioning."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        """Initialize SQLite store connection and run migrations."""
        self.db_path = str(db_path)
        try:
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON;")
            if self.db_path != ":memory:":
                self.conn.execute("PRAGMA journal_mode = WAL;")
                self.conn.execute("PRAGMA synchronous = NORMAL;")
            self.migrate()
        except sqlite3.Error as e:
            raise StoreError(f"Failed to initialize SQLite store: {e}") from e

    def migrate(self) -> None:
        """Run idempotent database migrations."""
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL
                    );
                """)
                row = self.conn.execute(
                    "SELECT MAX(version) AS current_version FROM schema_version;"
                ).fetchone()
                current_version = (
                    row["current_version"] if row and row["current_version"] is not None else 0
                )

                if current_version < 1:
                    self.conn.execute("""
                        CREATE TABLE IF NOT EXISTS memories (
                            id TEXT PRIMARY KEY,
                            type TEXT NOT NULL,
                            subject TEXT NOT NULL,
                            value TEXT NOT NULL,
                            reason TEXT,
                            source_msg_id TEXT NOT NULL,
                            source_quote TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            status TEXT NOT NULL DEFAULT 'current',
                            supersedes_id TEXT,
                            confidence REAL NOT NULL DEFAULT 1.0,
                            embedding BLOB
                        );
                    """)
                    self.conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_memories_subject_status
                        ON memories (subject, status);
                    """)
                    self.conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_memories_session_id
                        ON memories (session_id);
                    """)
                    now_iso = datetime.now(UTC).isoformat()
                    self.conn.execute(
                        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?);",
                        (1, now_iso),
                    )
        except sqlite3.Error as e:
            raise StoreError(f"Migration failed: {e}") from e

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert a database row into a Memory model."""
        ts = datetime.fromisoformat(row["timestamp"])
        return Memory(
            id=row["id"],
            type=MemoryType(row["type"]),
            subject=row["subject"],
            value=row["value"],
            reason=row["reason"],
            source_msg_id=row["source_msg_id"],
            source_quote=row["source_quote"],
            session_id=row["session_id"],
            timestamp=ts,
            status=MemoryStatus(row["status"]),
            supersedes_id=row["supersedes_id"],
            confidence=float(row["confidence"]),
        )

    def add(self, memory: Memory, embedding: bytes | None = None) -> None:
        """Insert a memory idempotently by id."""
        sql = """
            INSERT INTO memories (
                id, type, subject, value, reason, source_msg_id,
                source_quote, session_id, timestamp, status,
                supersedes_id, confidence, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING;
        """
        params = (
            memory.id,
            memory.type.value,
            memory.subject,
            memory.value,
            memory.reason,
            memory.source_msg_id,
            memory.source_quote,
            memory.session_id,
            memory.timestamp.isoformat(),
            memory.status.value,
            memory.supersedes_id,
            memory.confidence,
            embedding,
        )
        try:
            with self.conn:
                self.conn.execute(sql, params)
        except sqlite3.Error as e:
            raise StoreError(f"Failed to add memory: {e}") from e

    def add_many(
        self,
        memories: list[Memory],
        embeddings: list[bytes | None] | None = None,
    ) -> None:
        """Insert multiple memories in a single transaction."""
        sql = """
            INSERT INTO memories (
                id, type, subject, value, reason, source_msg_id,
                source_quote, session_id, timestamp, status,
                supersedes_id, confidence, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING;
        """
        try:
            with self.conn:
                for idx, mem in enumerate(memories):
                    emb = embeddings[idx] if embeddings and idx < len(embeddings) else None
                    params = (
                        mem.id,
                        mem.type.value,
                        mem.subject,
                        mem.value,
                        mem.reason,
                        mem.source_msg_id,
                        mem.source_quote,
                        mem.session_id,
                        mem.timestamp.isoformat(),
                        mem.status.value,
                        mem.supersedes_id,
                        mem.confidence,
                        emb,
                    )
                    self.conn.execute(sql, params)
        except sqlite3.Error as e:
            raise StoreError(f"Failed to add memories in batch: {e}") from e

    def get(self, memory_id: str) -> Memory | None:
        """Retrieve memory by id."""
        try:
            row = self.conn.execute(
                "SELECT * FROM memories WHERE id = ?;",
                (memory_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_memory(row)
        except sqlite3.Error as e:
            raise StoreError(f"Failed to get memory {memory_id}: {e}") from e

    def current_by_subject(self, subject: str) -> Memory | None:
        """Retrieve current active memory for a subject."""
        try:
            row = self.conn.execute(
                "SELECT * FROM memories WHERE subject = ? AND status = 'current' ORDER BY timestamp DESC LIMIT 1;",
                (subject,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_memory(row)
        except sqlite3.Error as e:
            raise StoreError(f"Failed to query current memory for {subject}: {e}") from e

    def list_by_session(self, session_id: str) -> list[Memory]:
        """List all memories for a session ordered by timestamp."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE session_id = ? ORDER BY timestamp ASC;",
                (session_id,),
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]
        except sqlite3.Error as e:
            raise StoreError(f"Failed to list memories for session {session_id}: {e}") from e

    def supersede(self, old_id: str, new_id: str) -> None:
        """Atomically supersede an older memory with a newer memory."""
        try:
            with self.conn:
                cur = self.conn.cursor()
                old_row = cur.execute(
                    "SELECT id FROM memories WHERE id = ?;",
                    (old_id,),
                ).fetchone()
                if old_row is None:
                    raise StoreError(f"Memory to supersede not found: {old_id}")
                new_row = cur.execute(
                    "SELECT id FROM memories WHERE id = ?;",
                    (new_id,),
                ).fetchone()
                if new_row is None:
                    raise StoreError(f"Replacement memory not found: {new_id}")
                cur.execute(
                    "UPDATE memories SET status = 'superseded' WHERE id = ?;",
                    (old_id,),
                )
                cur.execute(
                    "UPDATE memories SET supersedes_id = ? WHERE id = ?;",
                    (old_id, new_id),
                )
        except sqlite3.Error as e:
            raise StoreError(f"Supersede transaction failed: {e}") from e

    def close(self) -> None:
        """Close underlying database connection."""
        self.conn.close()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()
