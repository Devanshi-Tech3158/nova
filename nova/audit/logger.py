"""Audit log: one SQLite record per query (ids, verdicts, action, latency, tokens)."""

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from nova.core.config import get_settings
from nova.core.schemas import DecisionAction, Verdict

MAX_TEXT = 2000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    retrieved_ids TEXT NOT NULL,
    verdicts TEXT NOT NULL,
    decision_action TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL
)
"""
_COLS = (
    "id, timestamp, question, answer, retrieved_ids, verdicts, decision_action, "
    "latency_ms, prompt_tokens, completion_tokens"
)


class AuditError(Exception):
    """Raised when the audit log cannot be read or written."""


class AuditRecord(BaseModel):
    """One verification query and its outcome."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    question: str = ""
    answer: str = ""
    retrieved_ids: list[str] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    decision_action: DecisionAction = DecisionAction.ABSTAIN
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Timer:
    """Holds elapsed milliseconds once the timed block exits."""

    elapsed_ms: float = 0.0


@contextmanager
def timed() -> Iterator[Timer]:
    """Measure wall-clock milliseconds of the enclosed block."""
    timer = Timer()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.elapsed_ms = (time.perf_counter() - start) * 1000.0


def _secrets() -> list[str]:
    s = get_settings()
    return [k for k in (s.api_key, s.openai_api_key, s.anthropic_api_key) if k]


def _clean(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    return text[:MAX_TEXT]


class AuditLog:
    """SQLite-backed audit log in its own table of the configured database."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Open (and create if needed) the audit table; defaults to config db_path."""
        path = str(db_path if db_path is not None else get_settings().db_path)
        try:
            self.conn = sqlite3.connect(path, check_same_thread=False)
            self.conn.execute(_SCHEMA)
            self.conn.commit()
        except sqlite3.Error as e:
            raise AuditError(f"Failed to initialize audit log: {e}") from e

    def write(self, record: AuditRecord) -> str:
        """Persist the record (text truncated, secrets redacted) and return its id."""
        sec = _secrets()
        verdicts = [
            {
                **v.model_dump(mode="json"),
                "claim": _clean(v.claim, sec),
                "rationale": _clean(v.rationale, sec),
            }
            for v in record.verdicts
        ]
        row = (
            record.id,
            record.timestamp.isoformat(),
            _clean(record.question, sec),
            _clean(record.answer, sec),
            json.dumps(record.retrieved_ids),
            json.dumps(verdicts),
            record.decision_action.value,
            record.latency_ms,
            record.prompt_tokens,
            record.completion_tokens,
        )
        try:
            with self.conn:
                self.conn.execute(
                    f"INSERT INTO audit_log ({_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?)", row
                )
        except sqlite3.Error as e:
            raise AuditError(f"Failed to write audit record: {e}") from e
        return record.id

    def get(self, record_id: str) -> AuditRecord | None:
        """Return the record with this id, or None."""
        rows = self._query(f"SELECT {_COLS} FROM audit_log WHERE id = ?", (record_id,))
        return rows[0] if rows else None

    def recent(self, n: int = 20) -> list[AuditRecord]:
        """Return up to n records, newest first."""
        if n <= 0:
            return []
        sql = f"SELECT {_COLS} FROM audit_log ORDER BY timestamp DESC, rowid DESC LIMIT ?"
        return self._query(sql, (n,))

    def _query(self, sql: str, params: tuple[object, ...]) -> list[AuditRecord]:
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            raise AuditError(f"Failed to read audit log: {e}") from e
        return [
            AuditRecord(
                id=r[0],
                timestamp=datetime.fromisoformat(r[1]),
                question=r[2],
                answer=r[3],
                retrieved_ids=json.loads(r[4]),
                verdicts=[Verdict(**v) for v in json.loads(r[5])],
                decision_action=DecisionAction(r[6]),
                latency_ms=r[7],
                prompt_tokens=r[8],
                completion_tokens=r[9],
            )
            for r in rows
        ]
