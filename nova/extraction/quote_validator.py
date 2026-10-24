"""Validation of memory quotes against source message texts."""

import hashlib

from nova.core.schemas import Memory, Message


def generate_memory_id(
    session_id: str,
    msg_id: str,
    subject: str,
    value: str,
) -> str:
    """Compute deterministic 16-character SHA-1 id for a memory."""
    content = f"{session_id}|{msg_id}|{subject}|{value}"
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and lowercase text for substring comparison."""
    return " ".join(text.split()).lower()


def validate_quotes(
    memories: list[Memory],
    messages: list[Message],
) -> tuple[list[Memory], int]:
    """Filter memories keeping only those with verbatim normalized quotes in cited messages."""
    msg_map = {msg.id: msg.text for msg in messages}
    kept: list[Memory] = []
    dropped_count = 0
    seen_ids: set[str] = set()

    for mem in memories:
        if not mem.source_quote.strip():
            dropped_count += 1
            continue
        text = msg_map.get(mem.source_msg_id)
        if text is None:
            dropped_count += 1
            continue
        if normalize_whitespace(mem.source_quote) not in normalize_whitespace(text):
            dropped_count += 1
            continue
        mem_id = generate_memory_id(
            mem.session_id,
            mem.source_msg_id,
            mem.subject,
            mem.value,
        )
        if mem_id in seen_ids:
            dropped_count += 1
            continue
        seen_ids.add(mem_id)
        if mem.id != mem_id:
            mem = mem.model_copy(update={"id": mem_id})
        kept.append(mem)

    return kept, dropped_count
