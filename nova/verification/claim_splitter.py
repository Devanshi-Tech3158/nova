"""Split an answer into atomic claims."""

import json
import re

from pydantic import TypeAdapter, ValidationError

from nova.core.llm import LLMClient
from nova.verification.prompts import SPLIT_PROMPT, SPLIT_REPAIR_PROMPT

MAX_CLAIMS = 8
_CLAIMS = TypeAdapter(list[str])
_SENTENCE_END = re.compile(r"[.!?]+(?:\s+|$)")


class VerificationError(Exception):
    """Exception raised when claim splitting or judging fails."""


def _strip_fences(text: str) -> str:
    """Remove markdown code fences around JSON."""
    return re.sub(r"^```[a-z]*\s*|\s*```$", "", text.strip())


def _sentence_count(text: str) -> int:
    """Count sentences in text."""
    return len([p for p in _SENTENCE_END.split(text) if p.strip()])


def _parse(response: str) -> list[str]:
    """Parse and validate an LLM response as a list of claim strings."""
    claims = _CLAIMS.validate_python(json.loads(_strip_fences(response)))
    return [c.strip() for c in claims if c.strip()][:MAX_CLAIMS]


def split_claims(llm: LLMClient, answer: str) -> list[str]:
    """Split answer into at most 8 atomic claims; one sentence skips the LLM."""
    text = answer.strip()
    if not text:
        return []
    if _sentence_count(text) <= 1:
        return [text]
    response = llm.complete(f"{SPLIT_PROMPT}\n\nAnswer:\n{text}", json=True)
    try:
        return _parse(response)
    except (json.JSONDecodeError, ValidationError):
        retry_prompt = f"{SPLIT_REPAIR_PROMPT}\n\nInvalid response:\n{response}"
        repaired = llm.complete(retry_prompt, json=True)
        try:
            return _parse(repaired)
        except (json.JSONDecodeError, ValidationError) as e:
            raise VerificationError(f"Claim splitting failed after repair retry: {e}") from e
