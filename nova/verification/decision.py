"""Deterministic decision engine: claim verdicts to a final action."""

from nova.core.schemas import (
    Decision,
    DecisionAction,
    Memory,
    MemoryStatus,
    Verdict,
    VerdictLabel,
)

ABSTAIN_TEXT = "Insufficient evidence in memory to answer."


def _by_id(memories: list[Memory]) -> dict[str, Memory]:
    return {m.id: m for m in memories}


def _quotes(verdict: Verdict, index: dict[str, Memory]) -> list[str]:
    return [f'"{index[i].source_quote}"' for i in verdict.memory_ids if i in index]


def _conflict(verdicts: list[Verdict], index: dict[str, Memory]) -> str:
    lines = []
    for v in verdicts:
        if v.label is VerdictLabel.CONTRADICTED:
            quotes = ", ".join(_quotes(v, index)) or "no stored source"
            lines.append(f'Claim "{v.claim}" conflicts with memory: {quotes}.')
    return "Conflict between sources. " + " ".join(lines)


def _current_and_old(memory: Memory, memories: list[Memory]) -> tuple[Memory, Memory | None]:
    """Resolve a memory to its current version and the version it replaced."""
    index = _by_id(memories)
    if memory.status is MemoryStatus.SUPERSEDED:
        successors = [m for m in memories if m.supersedes_id == memory.id]
        return (successors[0], memory) if successors else (memory, None)
    old = index.get(memory.supersedes_id) if memory.supersedes_id else None
    return memory, old


def _correction(verdicts: list[Verdict], memories: list[Memory]) -> str:
    index = _by_id(memories)
    lines = []
    for v in verdicts:
        if v.label is not VerdictLabel.OUTDATED:
            continue
        found = [index[i] for i in v.memory_ids if i in index]
        if not found:
            lines.append(f'Claim "{v.claim}" is outdated; no current memory found.')
            continue
        cur, old = _current_and_old(found[0], memories)
        replaced = f" It replaced {old.value}." if old else " It replaced the old value."
        lines.append(f'Claim "{v.claim}" is outdated. Current: {cur.value}.{replaced}')
    return " ".join(lines)


def _qualify(verdicts: list[Verdict], index: dict[str, Memory]) -> str:
    sup = [v for v in verdicts if v.label is VerdictLabel.SUPPORTED]
    uns = [v for v in verdicts if v.label is VerdictLabel.UNSUPPORTED]
    return (
        "Supported: " + "; ".join(v.claim for v in sup) + ". "
        "Unverified: " + "; ".join(v.claim for v in uns) + "."
    )


def _answer(verdicts: list[Verdict], index: dict[str, Memory]) -> str:
    claims = "; ".join(v.claim for v in verdicts)
    quotes = [q for v in verdicts for q in _quotes(v, index)]
    return claims + (" Sources: " + ", ".join(quotes) + "." if quotes else "")


def decide(verdicts: list[Verdict], memories: list[Memory]) -> Decision:
    """Pick the action by priority contradicted > outdated > unsupported > supported."""
    if not verdicts:
        return Decision(action=DecisionAction.ABSTAIN, verdicts=[], final_answer=ABSTAIN_TEXT)
    labels = {v.label for v in verdicts}
    index = _by_id(memories)
    if VerdictLabel.CONTRADICTED in labels:
        action, text = DecisionAction.FLAG_CONFLICT, _conflict(verdicts, index)
    elif VerdictLabel.OUTDATED in labels:
        action, text = DecisionAction.CORRECT, _correction(verdicts, memories)
    elif VerdictLabel.UNSUPPORTED in labels:
        if VerdictLabel.SUPPORTED in labels:
            action, text = DecisionAction.QUALIFY, _qualify(verdicts, index)
        else:
            action, text = DecisionAction.ABSTAIN, ABSTAIN_TEXT
    else:
        action, text = DecisionAction.ANSWER, _answer(verdicts, index)
    return Decision(action=action, verdicts=verdicts, final_answer=text)
