"""Table-driven tests for the deterministic decision engine."""

from datetime import datetime
from itertools import product

import pytest

from nova.core.schemas import (
    Decision,
    Memory,
    MemoryStatus,
    MemoryType,
    Verdict,
)
from nova.core.schemas import (
    DecisionAction as A,
)
from nova.core.schemas import (
    VerdictLabel as L,
)
from nova.verification.decision import ABSTAIN_TEXT, decide

T = datetime(2026, 1, 1)


def mem(id_: str, value: str, status=MemoryStatus.CURRENT, sup: str | None = None) -> Memory:
    return Memory(
        id=id_,
        type=MemoryType.DECISION,
        subject="db",
        value=value,
        source_msg_id="m",
        source_quote=f"use {value}",
        session_id="s",
        timestamp=T,
        status=status,
        supersedes_id=sup,
    )


OLD = mem("old", "mysql", MemoryStatus.SUPERSEDED)
NEW = mem("new", "postgres", sup="old")
MEMS = [OLD, NEW]


def v(label: L, ids: list[str] | None = None, claim: str | None = None) -> Verdict:
    return Verdict(claim=claim or f"c-{label.value}", label=label, memory_ids=ids or [])


def expected(labels: tuple[L, ...]) -> A:
    s = set(labels)
    if L.CONTRADICTED in s:
        return A.FLAG_CONFLICT
    if L.OUTDATED in s:
        return A.CORRECT
    if L.UNSUPPORTED in s:
        return A.QUALIFY if L.SUPPORTED in s else A.ABSTAIN
    return A.ANSWER


COMBOS = [c for n in (1, 2, 3) for c in product(L, repeat=n)]


@pytest.mark.parametrize("labels", COMBOS)
def test_every_label_combination(labels):
    verdicts = [v(x, ["new"]) for x in labels]
    d = decide(verdicts, MEMS)
    assert d.action is expected(labels)
    assert d.verdicts == verdicts
    assert d.final_answer
    if set(labels) == {L.SUPPORTED}:
        assert d.action is A.ANSWER
        assert "conflict" not in d.final_answer.lower()
        assert "unverified" not in d.final_answer.lower()


def test_empty_abstains():
    d = decide([], MEMS)
    assert d == Decision(action=A.ABSTAIN, verdicts=[], final_answer=ABSTAIN_TEXT)


def test_conflict_lists_both_sources():
    d = decide([v(L.CONTRADICTED, ["new", "ghost"], "uses mysql")], MEMS)
    assert 'Claim "uses mysql"' in d.final_answer
    assert '"use postgres"' in d.final_answer
    assert "ghost" not in d.final_answer


def test_conflict_without_stored_source():
    assert "no stored source" in decide([v(L.CONTRADICTED)], MEMS).final_answer


def test_correct_from_superseded_id():
    d = decide([v(L.OUTDATED, ["old"])], MEMS)
    assert "Current: postgres" in d.final_answer
    assert "replaced mysql" in d.final_answer


def test_correct_from_current_id():
    d = decide([v(L.OUTDATED, ["new"])], MEMS)
    assert "Current: postgres" in d.final_answer
    assert "replaced mysql" in d.final_answer


def test_correct_current_without_old_in_list():
    d = decide([v(L.OUTDATED, ["new"])], [NEW])
    assert "replaced the old value" in d.final_answer


def test_correct_superseded_without_successor():
    d = decide([v(L.OUTDATED, ["old"])], [OLD])
    assert "Current: mysql" in d.final_answer
    assert "replaced the old value" in d.final_answer


def test_correct_without_memory():
    assert "no current memory found" in decide([v(L.OUTDATED, ["x"])], MEMS).final_answer


def test_correct_skips_other_labels():
    d = decide([v(L.SUPPORTED, ["new"], "ok"), v(L.OUTDATED, ["old"], "x")], MEMS)
    assert "ok" not in d.final_answer


def test_qualify_text():
    d = decide([v(L.SUPPORTED, claim="a"), v(L.UNSUPPORTED, claim="b")], MEMS)
    assert d.final_answer == "Supported: a. Unverified: b."


def test_answer_cites_quotes():
    d = decide([v(L.SUPPORTED, ["new"], "uses postgres")], MEMS)
    assert d.final_answer == 'uses postgres Sources: "use postgres".'
