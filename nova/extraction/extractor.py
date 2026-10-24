"""LLM-based memory extractor with windowing and validation."""

import json

from pydantic import ValidationError

from nova.core.llm import LLMClient
from nova.core.schemas import Memory, Message
from nova.extraction.prompts import EXTRACTION_PROMPT, REPAIR_PROMPT
from nova.extraction.quote_validator import generate_memory_id, validate_quotes


class ExtractionError(Exception):
    """Exception raised when memory extraction fails."""


def _clean_json_text(text: str) -> str:
    """Strip markdown code fence blocks if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


class LLMExtractor:
    """Extract structured memories using an LLM in 20-message windows."""

    WINDOW_SIZE = 20

    def __init__(self, llm: LLMClient) -> None:
        """Initialize extractor with an LLM client."""
        self.llm = llm

    def _build_window_prompt(self, messages: list[Message]) -> str:
        """Format a batch of conversation messages for extraction."""
        formatted_messages = "\n".join(f"[{msg.id}] ({msg.role}): {msg.text}" for msg in messages)
        return f"{EXTRACTION_PROMPT}\n\nConversation turns to analyze:\n{formatted_messages}"

    def _parse_items(
        self,
        text: str,
        msg_map: dict[str, Message],
    ) -> list[Memory]:
        """Parse JSON string and validate extracted memory schemas."""
        cleaned = _clean_json_text(text)
        data = json.loads(cleaned)
        if not isinstance(data, list):
            raise TypeError("LLM output must be a JSON array")

        parsed: list[Memory] = []
        for item in data:
            if not isinstance(item, dict):
                raise TypeError("Array item must be a JSON object")
            source_msg_id = str(item.get("source_msg_id", ""))
            msg = msg_map.get(source_msg_id)
            if msg is None:
                continue
            mem_id = generate_memory_id(
                session_id=msg.session_id,
                msg_id=source_msg_id,
                subject=str(item.get("subject", "")),
                value=str(item.get("value", "")),
            )
            mem = Memory(
                id=mem_id,
                type=item["type"],
                subject=item["subject"],
                value=item["value"],
                reason=item.get("reason"),
                source_msg_id=source_msg_id,
                source_quote=str(item.get("source_quote", "")),
                session_id=msg.session_id,
                timestamp=msg.timestamp,
                confidence=float(item.get("confidence", 1.0)),
            )
            parsed.append(mem)
        return parsed

    def _extract_window(self, window: list[Message]) -> list[Memory]:
        """Extract and validate memories from a single message window."""
        msg_map = {m.id: m for m in window}
        prompt = self._build_window_prompt(window)
        response = self.llm.complete(prompt, json=True)

        try:
            raw_memories = self._parse_items(response, msg_map)
        except (json.JSONDecodeError, ValueError, TypeError, ValidationError):
            repair_input = f"{REPAIR_PROMPT}\n\nInvalid response:\n{response}"
            repaired_response = self.llm.complete(repair_input, json=True)
            try:
                raw_memories = self._parse_items(repaired_response, msg_map)
            except (json.JSONDecodeError, ValueError, TypeError, ValidationError) as e:
                raise ExtractionError(f"Extraction failed after repair retry: {e}") from e

        kept, _ = validate_quotes(raw_memories, window)
        return kept

    def extract(self, messages: list[Message]) -> list[Memory]:
        """Extract validated memories across message windows."""
        if not messages:
            return []

        results: list[Memory] = []
        seen_ids: set[str] = set()

        for i in range(0, len(messages), self.WINDOW_SIZE):
            window = messages[i : i + self.WINDOW_SIZE]
            window_memories = self._extract_window(window)
            for mem in window_memories:
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    results.append(mem)

        return results
