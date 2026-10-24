# CLAUDE.md: handoff for the next AI agent

Read `AGENTS.md` first. Its TOKEN RULES override everything here. This file records what has been
built so far, the exact interfaces, the assumptions made, and what is still unverified.

## Project in one paragraph
Nova / "Context Anchor": external, source-linked memory plus evidence-checked answers. Write path
extracts memories from conversations; read path retrieves them and verifies a draft answer against
them. The final action is decided by deterministic code, never by an LLM. Absence of evidence is
`unsupported`, never `contradicted`. Spec: `context_anchor_architecture.pdf` (v1.0, 24 Oct 2026).

## Who owns what (from the spec)
- Person 1: `core/`, `store/`, `extraction/`, `temporal/`, `retrieval/`, ingest, API, Postgres.
- Person 2 (verification side, this work): `nova/verification/*`, `nova/audit/*`, and `ask()` /
  `verify()` wiring in `nova/pipeline.py`. Phases on FakeLLM: 1 claim splitter, 2 judge,
  3 decision engine, 4 audit log + wiring.
- Person 3: `benchmark/`. Person 4: human-only work.
- Edit only your assigned directories. Never change `nova/core/` without asking the human;
  propose the change instead.

## Phase status
| Phase | Deliverable | Status |
|---|---|---|
| 1 | `nova/verification/claim_splitter.py`, `prompts.py` | Done, not committed |
| 2 | `nova/verification/judge.py` | Done, not committed |
| 3 | `nova/verification/decision.py` + `tests/test_decision.py` (table-driven, no LLM) | Done, not committed |
| 4a | `nova/audit/logger.py` + `tests/test_audit.py` | Done, not committed |
| 4b | Wire `Anchor.ask()` / `Anchor.verify()` (use `decide`, `AuditLog`, `timed`) | NOT started |

No git history exists in the uploaded zip, so nothing was committed. Suggested commits:
`phase-1: claim splitter`, `phase-2: evidence judge`, `phase-3: decision engine`, `phase-4: audit log`. Rule 6: one phase per session, one commit, stop.

## Core contracts used (read-only, from `nova/core/`)
- `LLMClient.complete(prompt: str, *, json: bool = False) -> str`.
- `FakeLLM(responses)`: `responses` is a list (popped in order), dict (prompt -> reply), or str;
  records `llm.calls` as `[{"prompt", "json"}]`. When the list runs out it returns `default_response`
  (`""`), which is invalid JSON and triggers the repair path.
- `Verifier` protocol: `verify(question, answer, memories: list[Memory]) -> list[Verdict]`.
- `Verdict(claim, label: VerdictLabel, memory_ids: list[str] = [], rationale: str = "")`.
- `VerdictLabel`: supported, contradicted, outdated, unsupported.
- `DecisionAction`: answer, flag_conflict, correct, qualify, abstain. `Decision(action, verdicts,
  final_answer)`.
- `Memory.status`: `MemoryStatus.CURRENT | SUPERSEDED`; `Memory.supersedes_id` links the successor
  to the memory it replaced.
- `Anchor.ask()` and `Anchor.verify()` in `nova/pipeline.py` are still stubs that raise
  `NotImplementedError`. Read the `Anchor.__init__` signature before wiring; this file does not
  document it.

## What was built

### `nova/verification/claim_splitter.py`
- `split_claims(llm, answer) -> list[str]`.
- Empty/whitespace answer -> `[]`, no LLM call. At most one sentence -> `[answer.strip()]`, no LLM
  call. Sentence count splits on `.`, `!`, `?` followed by whitespace or end of text.
- Otherwise one `llm.complete(..., json=True)`; output validated as `list[str]` with a Pydantic
  `TypeAdapter`. Invalid JSON or schema -> one repair call -> `VerificationError`.
- Claims are stripped, blanks dropped, and capped at 8 in code (`MAX_CLAIMS`).
- Defines `VerificationError` and `_strip_fences` (removes markdown fences around JSON).
- `VerificationError` lives here because it did not exist in `core/` (frozen). Move it to `core/`
  only with human approval, then update the imports in `judge.py` and the tests.

### `nova/verification/prompts.py`
- `SPLIT_PROMPT` (constant, 9 lines) and `SPLIT_REPAIR_PROMPT`.

### `nova/verification/judge.py`
- `LLMJudge(llm)`:
  - `judge_claims(question, claims, memories) -> list[Verdict]`: ONE LLM call for all claims.
  - `verify(question, answer, memories) -> list[Verdict]`: `split_claims` then `judge_claims`.
- Prompt gets numbered memories (`id`, `value`, `source_quote`, `status`, `supersedes_id`, all
  `json.dumps`-escaped) and 0-based numbered claims. Reply is
  `[{claim_idx, label, memory_ids, rationale}]`, validated with Pydantic (`_Item`).
- Invalid JSON or schema -> one repair call -> `VerificationError`.
- Code-side guard (`_check`), the part that must not be weakened:
  1. cited ids not in the input memories are dropped;
  2. supported/contradicted/outdated left with no valid ids -> `unsupported`;
  3. `outdated` is kept only if a cited memory has `status == SUPERSEDED` AND a CURRENT memory in
     the input has `supersedes_id` equal to it; otherwise -> `unsupported`. Never trust the LLM here;
  4. missing, out-of-range, or duplicate `claim_idx` -> `unsupported` (first duplicate wins);
  5. rationale truncated to 20 words.
- Empty memories: no LLM call. `judge_claims` returns one unsupported verdict per claim. `verify`
  returns ONE unsupported verdict for the whole answer (or `[]` for an empty answer), because
  splitting a multi-sentence answer would cost an LLM call. The decision outcome (abstain) is the
  same either way.
- The judge prompt constants (`JUDGE_PROMPT`, `JUDGE_REPAIR_PROMPT`) live in `judge.py`, not
  `prompts.py`, because the task named only `judge.py` and `test_judge.py`.
- `judge.py` imports `VerificationError`, `_strip_fences`, `split_claims` from `claim_splitter.py`
  (same directory, same owner).

### Tests (FakeLLM only, no real API)
- `tests/test_claim_splitter.py`, 6 tests: compound split, one-sentence fast path (no calls), bad
  JSON retry, bad JSON twice raises, empty -> `[]`, cap at 8.
- `tests/test_judge.py`, 10 tests: four labels, hallucinated id downgraded, outdated enforced in
  code (current memory cited, and superseded with no successor), missing/out-of-range
  `claim_idx`, bad JSON retry and raise, empty-memories short-circuit, `verify` batching (2 calls:
  split + judge).


### `nova/verification/decision.py` (phase 3)
- `decide(verdicts, memories) -> Decision`; pure, no LLM, `verdicts` returned unchanged.
- Priority: contradicted -> `flag_conflict`; outdated -> `correct`; unsupported -> `qualify` if >=1
  supported else `abstain`; all supported -> `answer`; empty list -> `abstain` (`ABSTAIN_TEXT`).
- Templates: conflict lists claim + memory `source_quote`s; correct states current value and the
  value it replaced; qualify gives "Supported: ... Unverified: ..."; answer lists claims + quotes.
- Outdated resolution: a superseded memory id maps to its successor (`supersedes_id == id`) in
  `memories`; a current id uses `supersedes_id` for the old value. Missing links use fallback text.
  Unknown ids are ignored.
- Tests: `tests/test_decision.py` covers all label combos (1-3 verdicts), empty list, every template
  branch; fully-supported answers never contain conflict/unverified text.
- NOT verified with real tools: pytest/ruff/pydantic were unavailable offline. Logic was run with
  throwaway shims in /tmp (95 cases passed, every body line hit). Run `ruff check` and
  `pytest -q -x` (with `--cov`) before committing.

### `nova/audit/logger.py` (phase 4a)
- `AuditRecord` (pydantic, lives here not in `core/`): id (uuid4 hex default), timestamp (UTC default),
  question, answer, retrieved_ids, verdicts, decision_action, latency_ms, prompt_tokens, completion_tokens.
- `AuditLog(db_path=None)`: defaults to `get_settings().db_path`; own table `audit_log`, parameterized SQL,
  own connection. `.write(record) -> id`, `.get(id) -> AuditRecord | None`, `.recent(n=20)` newest first
  (timestamp, then insert order). Duplicate id -> `AuditError`; sqlite errors -> `AuditError`.
- Stored question, answer, verdict claim/rationale are cut to 2000 chars (`MAX_TEXT`); any configured API key
  value found in them is replaced by `[redacted]`. No key field exists.
- `timed()` context manager yields a `Timer`; read `.elapsed_ms` after the block.
- Verdicts are stored as JSON, so `get()` returns them as `Verdict` objects. Tokens are caller-supplied.
- Verification: same as phase 3 (shims only, 8 tests passed, all body lines hit). Run real `ruff`/`pytest` first.

## Verification status: READ THIS
- The sandbox had no network, so `pydantic`, `pytest` and `ruff` could not be installed.
- All 16 tests above passed in a hand-written runner against a throwaway minimal pydantic stub
  (not in the repo). This is NOT a real run.
- Not yet run: real `pytest -q -x` (including the other 39 existing tests), `ruff check .`,
  `ruff format --check .`. CI runs both ruff commands (`.github/workflows/ci.yml`), so run
  `ruff format` on the new files first; formatting differences are possible.
- Real pydantic is stricter than the stub in places. If a test fails, check first: enum coercion
  of `label`, int coercion of `claim_idx`, and construction of `Memory` in `tests/test_judge.py`.
- `tests/test_judge.py` has two `# fmt: skip` markers on compact fixtures; remove them if
  `ruff format` is happy without them.
- Line length limit is 100 (`pyproject.toml`); new files stay within it.

## Next steps (not started)
- Phase 3, `nova/verification/decision.py`: pure function from `list[Verdict]` to `Decision`.
  Priority contradicted > outdated > unsupported > supported. Mapping:
  all supported -> `answer`; any contradicted -> `flag_conflict` (show both sources, no silent
  override); any outdated -> `correct` (current value, noting it replaced the old one); some
  supported plus some unsupported -> `qualify`; nothing supported or no memories -> `abstain`.
  Table-driven tests, no LLM, deterministic response templates with source quotes.
- Phase 4: `nova/audit/logger.py` (one record per query: retrieved ids, verdicts, action, latency,
  tokens; text truncated to 2000 chars; never store keys; logs record counts, not content), then
  implement `Anchor.verify()` and `Anchor.ask()` in `pipeline.py`. `ask()` with no draft makes one
  extra LLM call that answers strictly from retrieved memories, then verifies. Retrieval is
  `Retriever.retrieve(query, k=5, include_history=False, session_id=None)`.
- Spec lists `verification/service.py`; it does not exist yet.

## Rules that bite (from AGENTS.md, condensed)
- All LLM calls go through `LLMClient`. Tests use `FakeLLM` / `HashEmbedder` only.
- Every memory keeps a verbatim `source_quote`. Python 3.11, full type hints, one-line docstrings,
  no comments restating code, no unrequested features or files.
- Tests: `pytest -q -x 2>&1 | tail -20`. Max 3 fix attempts, then report the blocker.
- Final reply at most 5 lines: files changed, test result, assumptions.
