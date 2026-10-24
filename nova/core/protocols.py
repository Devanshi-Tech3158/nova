"""Protocol interfaces for core Nova components."""

from typing import Protocol, runtime_checkable

from nova.core.schemas import Memory, Message, Verdict


@runtime_checkable
class Extractor(Protocol):
    """Protocol for extracting structured memories from conversation messages."""

    def extract(self, messages: list[Message]) -> list[Memory]:
        """Extract memories from a sequence of messages."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for persistent memory storage with supersession support."""

    def add(self, memory: Memory) -> None:
        """Add a single memory to the store."""
        ...

    def add_many(self, memories: list[Memory]) -> None:
        """Add multiple memories to the store."""
        ...

    def get(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by id."""
        ...

    def current_by_subject(self, subject: str) -> Memory | None:
        """Retrieve the current active memory for a subject."""
        ...

    def list_by_session(self, session_id: str) -> list[Memory]:
        """List all memories associated with a session id."""
        ...

    def supersede(self, old_id: str, new_id: str) -> None:
        """Mark an older memory as superseded by a newer memory id."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Protocol for memory retrieval."""

    def retrieve(
        self,
        query: str,
        k: int = 5,
        include_history: bool = False,
        session_id: str | None = None,
    ) -> list[Memory]:
        """Retrieve relevant memories matching query."""
        ...


@runtime_checkable
class Verifier(Protocol):
    """Protocol for answer claim verification against memories."""

    def verify(
        self,
        question: str,
        answer: str,
        memories: list[Memory],
    ) -> list[Verdict]:
        """Verify claims in answer against retrieved memories."""
        ...
