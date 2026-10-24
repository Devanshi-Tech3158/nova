"""Tests for claim splitting."""

import json

import pytest

from nova.core.llm import FakeLLM
from nova.verification.claim_splitter import VerificationError, split_claims


def test_compound_answer_is_split() -> None:
    """Multi-sentence answers are split into the LLM's claims."""
    claims = ["The project uses Postgres.", "The deadline is Friday."]
    llm = FakeLLM([json.dumps(claims)])
    assert split_claims(llm, "We use Postgres. It ships Friday.") == claims
    assert len(llm.calls) == 1


def test_single_sentence_skips_llm() -> None:
    """One-sentence answers are returned as-is with no LLM call."""
    llm = FakeLLM()
    assert split_claims(llm, "The database is Postgres.") == ["The database is Postgres."]
    assert llm.calls == []


def test_bad_json_retries_once() -> None:
    """Invalid JSON triggers one repair call, then succeeds."""
    llm = FakeLLM(["not json", '["A is true."]'])
    assert split_claims(llm, "A is true. Hello there.") == ["A is true."]
    assert len(llm.calls) == 2


def test_bad_json_twice_raises() -> None:
    """Two invalid responses raise VerificationError."""
    llm = FakeLLM(["nope", '{"a": 1}'])
    with pytest.raises(VerificationError):
        split_claims(llm, "One thing. Another thing.")
    assert len(llm.calls) == 2


def test_empty_answer_returns_empty() -> None:
    """Empty or whitespace answers yield [] without an LLM call."""
    llm = FakeLLM()
    assert split_claims(llm, "") == []
    assert split_claims(llm, "   ") == []
    assert llm.calls == []


def test_claims_capped_at_eight() -> None:
    """Responses with more than 8 claims are truncated to 8."""
    llm = FakeLLM([json.dumps([f"claim {i}" for i in range(12)])])
    assert len(split_claims(llm, "One. Two. Three.")) == 8
