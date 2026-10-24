"""Verification service coordinating retrieval, claim splitting, judging, and audit."""

from collections.abc import Callable
from typing import Any

from nova.audit.logger import AuditLog, AuditRecord, timed
from nova.core.llm import LLMClient
from nova.core.protocols import Retriever
from nova.core.schemas import Decision, Memory, Verdict
from nova.verification.claim_splitter import split_claims
from nova.verification.decision import decide
from nova.verification.judge import LLMJudge


class AnchorVerifier:
    """Coordinates retrieval, claim splitting, LLM judging, and decision engine."""

    def __init__(
        self,
        retriever: Retriever,
        splitter: Callable[..., list[str]] | None = None,
        judge: Any = None,
        audit: AuditLog | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        """Initialize verifier with retriever, splitter, judge, audit log, and LLM."""
        self.retriever = retriever
        self.splitter = splitter or split_claims
        self.llm = llm or getattr(judge, "llm", None)
        self.judge = judge or (LLMJudge(self.llm) if self.llm else None)
        self.audit = audit

    def _split(self, answer: str) -> list[str]:
        """Split answer text into atomic claims."""
        if callable(self.splitter):
            try:
                return self.splitter(self.llm, answer)
            except TypeError:
                return self.splitter(answer)
        return [answer.strip()] if answer.strip() else []

    def _judge(
        self,
        question: str,
        claims: list[str],
        memories: list[Memory],
    ) -> list[Verdict]:
        """Judge claims against retrieved memories."""
        if hasattr(self.judge, "judge_claims"):
            return self.judge.judge_claims(question, claims, memories)
        if callable(self.judge):
            return self.judge(question, claims, memories)
        return []

    def verify(self, question: str, answer: str) -> Decision:
        """Verify an answer against retrieved memories through the full pipeline."""
        with timed() as timer:
            memories = self.retriever.retrieve(question, k=5, include_history=True)
            claims = self._split(answer)
            verdicts = self._judge(question, claims, memories)
            decision = decide(verdicts, memories)

        if self.audit is not None:
            record = AuditRecord(
                question=question,
                answer=answer,
                retrieved_ids=[m.id for m in memories],
                verdicts=verdicts,
                decision_action=decision.action,
                latency_ms=timer.elapsed_ms,
            )
            self.audit.write(record)

        return decision

    def ask(self, question: str, draft_answer: str | None = None) -> Decision:
        """Answer question strictly from retrieved memories if draft is omitted, then verify."""
        if draft_answer is None:
            memories = self.retriever.retrieve(question, k=5, include_history=False)
            mem_text = "\n".join(
                f"- {m.subject}: {m.value} (quote: {m.source_quote})" for m in memories
            )
            prompt = (
                "Answer the question using only these memories. If the memories do not contain the answer, say unknown.\n\n"
                f"Memories:\n{mem_text}\n\nQuestion: {question}"
            )
            if self.llm is None:
                raise ValueError("LLMClient required to answer questions without a draft answer")
            draft_answer = self.llm.complete(prompt)

        return self.verify(question, draft_answer)
