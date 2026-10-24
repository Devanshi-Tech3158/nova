"""Core domain schemas and enumerations for Nova."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    """Enumeration of recognized memory types."""

    FACT = "fact"
    DECISION = "decision"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    COMMITMENT = "commitment"


class MemoryStatus(StrEnum):
    """Lifecycle status of a stored memory."""

    CURRENT = "current"
    SUPERSEDED = "superseded"


class VerdictLabel(StrEnum):
    """Verification label for a claim against evidence."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    OUTDATED = "outdated"
    UNSUPPORTED = "unsupported"


class DecisionAction(StrEnum):
    """Deterministic action chosen based on claim verdicts."""

    ANSWER = "answer"
    FLAG_CONFLICT = "flag_conflict"
    CORRECT = "correct"
    QUALIFY = "qualify"
    ABSTAIN = "abstain"


class Memory(BaseModel):
    """Stored atomic memory with provenance and supersession state."""

    id: str
    type: MemoryType
    subject: str
    value: str
    reason: str | None = None
    source_msg_id: str
    source_quote: str
    session_id: str
    timestamp: datetime
    status: MemoryStatus = MemoryStatus.CURRENT
    supersedes_id: str | None = None
    confidence: float = 1.0


class Message(BaseModel):
    """Conversation turn within a session."""

    id: str
    session_id: str
    role: str
    text: str
    timestamp: datetime


class Verdict(BaseModel):
    """Evaluation verdict for an individual atomic claim."""

    claim: str
    label: VerdictLabel
    memory_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class Decision(BaseModel):
    """Final decision determined from claim verdicts."""

    action: DecisionAction
    verdicts: list[Verdict] = Field(default_factory=list)
    final_answer: str | None = None


class IngestReport(BaseModel):
    """Summary counts returned after memory ingestion."""

    added: int = 0
    superseded: int = 0
    dropped: int = 0
