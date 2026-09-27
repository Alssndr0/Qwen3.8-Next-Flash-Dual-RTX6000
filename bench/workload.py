#!/usr/bin/env python3
"""
workload.py — the prompt generators loadgen.py fires.

Three kinds, selected with `--workload`. All keep the shape the protocol fixes:
a shared prefix that stays hot in the prefix cache plus a per-request cold tail,
seeded, so the prefix-cache hit lands near what production measures.

`filler` (the original, 2026-08-12)
    18,000 tokens drawn uniformly from a 38-word list, task "Summarize the above."
    Reproducible to the byte, and every run before 2026-09-13 used it. Keep it: it
    is the only way to re-run a stored arm on its own content.

`code` (2026-09-13, the default)
    A synthetic Python service repo, then an implementation task against it.

    Why it exists: `filler`'s content is word salad, and MTP acceptance is a property
    of content. On filler the drafter accepts 43% of draft positions; on our live
    opencode traffic it accepts 50%, and bilikaz measure 80-97% on dense code with the
    same method. So filler's `accept_len` ~2.3 is the prompt generator's number, not
    the engine's, and any arm that only pays off on structured content is invisible to
    the suite. See memory `bench-content-is-word-salad` and TUNING.md:282.

    The generated code is not semantically meaningful and does not need to be. What
    the drafter sees is the *statistical* structure of code — repeated identifiers,
    predictable punctuation and indentation, docstring and error-handling boilerplate
    — and that is what this reproduces.

`agentic` (2026-09-18)
    One turn of a multi-turn agent session against the same synthetic repo: a harness
    system prompt, a task, a tool loop (assistant tool_calls -> tool results) and a
    `tools` array, sent as chat `messages` rather than one user string. The shape is
    the production shape Prometheus on the head measured over the 46.6 h after the
    lpt1024 boot (2026-09-16 18:31 -> 09-18 17:05 UTC, 6.4k requests):

        prompt ~65.9k tokens (57% > 50k, 23% > 100k), prefix hit 94.2%
        uncached tokens per request mean 3.8k, median ~1.5k -- an agent turn appends
            one tool result on top of a session prefix that is already resident, and
            the engine recomputes the last 832-token block of every hit (MTP drops it)
        generation mean 535, tail to 2-5k; ~half of it reasoning; ~80% of turns end
            in a tool call; queue ~0, concurrency c=2-3 for 68% of request-time
        acceptance 2.65 tokens/step (71 / 53 / 42% per draft position) against
            3.15 on `code` -- the gap is content, and every decode number the suite
            produced on `code` is ~16% optimistic for what users get

    So `code`'s 18k prompt with a 300-token forced continuation cannot rank an arm for
    production; this kind can. Structure of one session (see `_build_agentic`):

        [system]  harness prompt, identical everywhere            |  shared, seed 0,
        [user]    the task                                          |  hot in the prefix
        [assistant tool_call / tool result] x N  "exploration"     |  cache
        [assistant tool_call / tool result] x (turn+1)  the session's own loops, seeded
                                                        by (tail_seed, session)

    Turn t of a session repeats turn t-1's messages verbatim and appends one loop, so
    the only cold tokens are that loop (plus the dropped block), exactly as in
    production. The assistant halves of the loops are canned, not the model's own
    replies: a rep must be byte-identical across arms for compare.py's pairing, and a
    sampled reply is not. Tool results are drawn from a skewed size distribution
    (`_RESULT_SIZES`) whose mean is `unique_tokens` and whose median is ~40% of it.

All are pure functions of (prompt_tokens, unique_tokens, index, tail_seed[, turn]),
with no clock and no `random` module: the LCG below is the same one `filler` has always
used, so a prompt cannot change under a Python upgrade.
"""

from __future__ import annotations

import json

# `code` is generated to a character budget and converted with this ratio. Measured
# 2026-09-13 against the live served tokenizer (`POST /tokenize`, Qwen3.8-Flash-Next-NVFP4)
# on a 201,677-char sample of this generator's own output: 48,335 tokens = 4.1725
# chars/token. Re-measure with `./workload.py --calibrate <url> <model>` if the generator
# or the tokenizer changes; loadgen warns when the served prompt_tokens drifts >5% from
# --prompt-tokens, which is the signal that this constant has gone stale.
CHARS_PER_TOKEN = 4.1725

KINDS = ("filler", "code", "agentic")

# `agentic` mixes three densities, measured 2026-09-19 against the served tokenizer
# (vLLM 0.29 /tokenize, hibrid48; `./workload.py --calibrate-agentic <url> <model>` on the
# head): the line-numbered reads and path:line grep hits that make up the tool results
# tokenise at 3.40-3.53 chars/token, not the 4.17 of plain code (the offline guess was
# 17% low); the tool-call and tools-schema JSON at 3.42-3.80; the English system/user
# text at 4.2-4.5. loadgen compares the served prompt_tokens with the builder's estimate
# and warns at >5% drift; re-measure after any tokenizer or template change.
CHARS_PER_TOKEN_TOOL = 3.49    # role=tool content (read_file / grep / pytest blend)
CHARS_PER_TOKEN_JSON = 3.6     # tool_calls function JSON and the tools schema
CHARS_PER_TOKEN_PROSE = 4.2    # system prompt and task text


class _Rng:
    """The house LCG. Deterministic across interpreters, unlike random.choice()."""

    def __init__(self, seed: int):
        self.x = (seed * 2654435761 + 1) % (2**31)

    def next(self) -> int:
        self.x = (self.x * 1103515245 + 12345) % (2**31)
        return self.x

    def pick(self, seq):
        return seq[self.next() % len(seq)]

    def below(self, n: int) -> int:
        return self.next() % n


# --------------------------------------------------------------------------- #
# filler — the original generator, byte-for-byte
# --------------------------------------------------------------------------- #

# Common short words, so one word lands near one token. The exact ratio does not
# matter: the run reports the prompt_tokens vllm actually counted.
WORDS = (
    "the quick brown fox jumps over a lazy dog while the parser reads tokens from "
    "one buffer and returns structured output to its caller after checking every "
    "field against the schema it loaded at startup time without any error"
).split()


def _filler(n_words: int, seed: int) -> str:
    """Deterministic word sequence — same (n, seed) always yields the same text."""
    out = []
    x = (seed * 2654435761 + 1) % (2**31)
    for _ in range(n_words):
        x = (x * 1103515245 + 12345) % (2**31)
        out.append(WORDS[x % len(WORDS)])
    return " ".join(out)


def _build_filler(prompt_tokens: int, unique_tokens: int, index: int, tail_seed: int) -> str:
    shared_n = max(prompt_tokens - unique_tokens, 0)
    parts = [_filler(shared_n, seed=0)] if shared_n else []
    if unique_tokens:
        parts.append(
            f"[request {tail_seed}:{index}] "
            + _filler(unique_tokens, seed=tail_seed * 100003 + index + 1)
        )
    parts.append("Summarize the above.")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# code — a synthetic Python service repo
# --------------------------------------------------------------------------- #

_DOMAINS = ("order", "invoice", "shipment", "account", "ledger", "session",
            "payment", "catalog", "tenant", "webhook", "subscription", "receipt")
_ADJ = ("pending", "settled", "stale", "archived", "draft", "locked", "partial")
_VERBS = ("resolve", "reconcile", "dispatch", "validate", "normalise", "expire",
          "flush", "rebuild", "annotate", "compact", "reindex", "settle")
_FIELDS = ("id", "created_at", "updated_at", "status", "amount_cents", "currency",
           "tenant_id", "retry_count", "checksum", "external_ref", "version")
_TYPES = ("str", "int", "float", "bool", "datetime", "Decimal", "UUID")
_ERRORS = ("ValueError", "KeyError", "TimeoutError", "LookupError", "RuntimeError")


def _module(rng: _Rng, name: str) -> str:
    """One plausible module: imports, a dataclass, a service class, free functions."""
    dom = rng.pick(_DOMAINS)
    cls = dom.capitalize()
    lines = [
        f'"""{name}.py — {rng.pick(_VERBS)} {dom} records for the {rng.pick(_DOMAINS)} service."""',
        "",
        "from __future__ import annotations",
        "",
        "import logging",
        "from dataclasses import dataclass, field",
        "from datetime import datetime, timezone",
        "from decimal import Decimal",
        "from typing import Iterable, Mapping, Sequence",
        "",
        f"from .errors import {cls}NotFound, {cls}Conflict",
        "from .telemetry import counter, timed",
        "",
        "log = logging.getLogger(__name__)",
        "",
        f"DEFAULT_PAGE_SIZE = {rng.below(400) + 50}",
        f"MAX_RETRIES = {rng.below(6) + 2}",
        f"BACKOFF_SECONDS = {rng.below(20) + 1}.0",
        "",
        "",
        "@dataclass(slots=True)",
        f"class {cls}Record:",
        f'    """A single {dom} row as it comes back from the store."""',
        "",
    ]
    # `id` and `status` always, the rest sampled without replacement: a dataclass with
    # the same field twice is not code any model has seen, and `is_terminal` below
    # reads `status`.
    chosen = ["id", "status"]
    pool = [f for f in _FIELDS if f not in chosen]
    for _ in range(rng.below(4) + 3):
        if not pool:
            break
        chosen.append(pool.pop(rng.below(len(pool))))
    for f_ in chosen:
        t = "str" if f_ in ("id", "status") else rng.pick(_TYPES)
        lines.append(f"    {f_}: {t} | None = None")
    lines += [
        "    tags: list[str] = field(default_factory=list)",
        "",
        "    def is_terminal(self) -> bool:",
        f'        return self.status in {{"{rng.pick(_ADJ)}", "{rng.pick(_ADJ)}"}}',
        "",
        "    def as_payload(self) -> dict[str, object]:",
        '        """Wire form; None fields are dropped so diffs stay small."""',
        "        out: dict[str, object] = {}",
        "        for key in self.__slots__:",
        "            value = getattr(self, key, None)",
        "            if value is None:",
        "                continue",
        "            if isinstance(value, datetime):",
        "                value = value.astimezone(timezone.utc).isoformat()",
        "            elif isinstance(value, Decimal):",
        "                value = str(value)",
        "            out[key] = value",
        "        return out",
        "",
        "",
        f"class {cls}Service:",
        f'    """Reads and writes {dom} records, with retry and telemetry."""',
        "",
        "    def __init__(self, store, clock=None, page_size: int = DEFAULT_PAGE_SIZE) -> None:",
        "        self._store = store",
        "        self._clock = clock or (lambda: datetime.now(timezone.utc))",
        "        self._page_size = page_size",
        "",
    ]
    for _ in range(rng.below(3) + 3):
        verb, err = rng.pick(_VERBS), rng.pick(_ERRORS)
        lines += [
            "    @timed",
            f"    def {verb}_{dom}(self, {dom}_id: str, *, force: bool = False) -> {cls}Record:",
            f'        """{verb.capitalize()} one {dom}, raising if it has already settled."""',
            f"        record = self._store.get({dom}_id)",
            "        if record is None:",
            f"            raise {cls}NotFound({dom}_id)",
            "        if record.is_terminal() and not force:",
            f'            raise {cls}Conflict(f"{dom} {{{dom}_id}} is {{record.status}}")',
            "        try:",
            f"            updated = self._store.apply({dom}_id, {{\"status\": \"{rng.pick(_ADJ)}\"}})",
            f"        except {err} as exc:",
            f'            log.warning("{verb} failed for %s: %s", {dom}_id, exc)',
            f'            counter("{dom}.{verb}.error").inc()',
            "            raise",
            "        record.updated_at = self._clock()",
            f'        counter("{dom}.{verb}.ok").inc()',
            "        return updated",
            "",
        ]
    lines += [
        f"    def iter_{dom}s(self, *, status: str | None = None) -> Iterable[{cls}Record]:",
        '        """Page through the store; the cursor is opaque and store-defined."""',
        "        cursor = None",
        "        while True:",
        "            page, cursor = self._store.page(cursor, self._page_size, status=status)",
        "            for row in page:",
        "                yield row",
        "            if cursor is None:",
        "                break",
        "",
        "",
        f"def summarise_{dom}s(rows: Sequence[{cls}Record]) -> Mapping[str, int]:",
        '    """Count rows by status; used by the nightly reconciliation report."""',
        "    totals: dict[str, int] = {}",
        "    for row in rows:",
        '        key = row.status or "unknown"',
        "        totals[key] = totals.get(key, 0) + 1",
        "    return totals",
        "",
    ]
    return "\n".join(lines)


_PREAMBLE = """You are an expert software engineer working inside an existing Python
repository. You have the full source of the relevant package below. Follow the
conventions already in the code: `from __future__ import annotations`, PEP 604
unions, dataclasses with `slots=True`, module-level `log`, and the `@timed` /
`counter(...)` telemetry helpers. Do not invent new dependencies. When you change a
file, return the complete file rather than a diff.

Repository: services/billing — the package below is `services/billing/core/`.
"""


def _repo_text(rng: _Rng, char_budget: int, prefix: str) -> str:
    """Emit modules up to the budget. `prefix` namespaces the file names.

    The final block is trimmed at a line boundary rather than letting a whole module
    overshoot: emitting modules until the budget is merely *exceeded* ran +4% long on
    every request, which is a systematic bias on prompt_tokens, not noise.
    """
    out, used, n = [], 0, 0
    while used < char_budget:
        name = f"{prefix}_{rng.pick(_VERBS)}_{n:02d}"
        body = _module(rng, name)
        block = f"\n# ---- services/billing/core/{name}.py " + "-" * 28 + f"\n\n{body}\n"
        if used + len(block) > char_budget:
            keep = block[: char_budget - used]
            cut = keep.rfind("\n")
            out.append(keep[:cut] + "\n" if cut > 0 else keep)
            break
        out.append(block)
        used += len(block)
        n += 1
    return "".join(out)


def _build_code(prompt_tokens: int, unique_tokens: int, index: int, tail_seed: int) -> str:
    shared_tokens = max(prompt_tokens - unique_tokens, 0)
    parts = [_PREAMBLE]
    if shared_tokens:
        # seed=0, exactly like filler's shared half: identical on every run and every
        # request, so it stays resident in the prefix cache the way opencode's real
        # ~11.8k system prompt does.
        budget = int(shared_tokens * CHARS_PER_TOKEN) - len(_PREAMBLE)
        parts.append(_repo_text(_Rng(0), max(budget, 0), "core"))
    if unique_tokens:
        # The cold half: this request's own module, never seen before.
        rng = _Rng(tail_seed * 100003 + index + 1)
        parts.append(f"\n# ---- request {tail_seed}:{index} ----\n")
        parts.append(_repo_text(rng, int(unique_tokens * CHARS_PER_TOKEN), f"req{index}"))
        target = f"req{index}_{_Rng(tail_seed * 100003 + index + 1).pick(_VERBS)}_00"
    else:
        target = "core_resolve_00"
    # The task is chosen to run well past --max-tokens of real code: at 300 tokens the
    # model is still writing the file, so `ignore_eos` never forces off-distribution
    # continuation the way it does on filler's "Summarize the above."
    parts.append(
        f"\n\nTask: in `{target}.py`, add a `retry_with_backoff` decorator that retries "
        f"on `TimeoutError` up to `MAX_RETRIES` times with exponential backoff starting at "
        f"`BACKOFF_SECONDS`, logging each attempt through the module `log` and incrementing "
        f'a `counter(\"...retry\")`. Apply it to every service method that touches the store. '
        f"Return the complete updated file."
    )
    return "".join(parts)


# --------------------------------------------------------------------------- #
# agentic — one turn of a tool-loop session, as chat messages + tools
# --------------------------------------------------------------------------- #

# The harness prompt. Generic on purpose: what matters to the engine is that it is
# long, English, identical across every session, and sits at the very front of the
# prefix cache -- the role opencode's ~11.8k system prompt plays in production.
_SYSTEM_PROMPT = """You are an autonomous software engineering agent operating inside a developer's
terminal. You are working in the repository `services/billing`, a Python 3.12 package
that reconciles orders, invoices, shipments and ledger entries for a multi-tenant
payments platform. You have a set of tools to inspect and change the repository and to
run commands in it. Use them; do not guess at file contents you have not read.

## How to work

- Start by understanding the task. Read the files it touches before proposing a change,
  and read the callers of anything whose signature you intend to alter.
- Prefer small, verifiable steps: read, then edit, then run the relevant tests, then
  continue. Do not batch several unrelated edits into one turn.
- When a tool result contradicts your expectation, stop and re-read rather than
  pressing on with the plan you had before the result arrived.
- Never fabricate output from a command you did not run. If a command fails, show the
  failure and decide what to do next on the basis of the actual error.
- Keep the user informed in short plain sentences. Explain what you found and what you
  are about to do; do not narrate every keystroke.
- Match the conventions already present in the code: `from __future__ import
  annotations`, PEP 604 unions, dataclasses with `slots=True`, module-level `log`, the
  `@timed` decorator and `counter(...)` telemetry helpers from `.telemetry`. Do not add
  dependencies. Do not reformat files you are not otherwise changing.
- Tests live under `services/billing/tests/` and run with `python -m pytest -q`. A change
  is not done until the tests that cover it pass, or you have explained why they cannot
  be run here.
- Do not commit, push, or rewrite history unless explicitly asked. Do not delete files
  or data directories. Do not modify anything outside the repository.

## Tool usage

- `read_file` returns the file with line numbers; use `offset` and `limit` for large
  files rather than reading them whole repeatedly.
- `edit_file` performs an exact-string replacement. The `old_string` must match exactly
  once; include enough surrounding lines to make it unique. Read the file first.
- `write_file` overwrites a whole file; use it only for new files.
- `grep` searches file contents with a regular expression and returns matching lines
  with their paths and line numbers. `glob` finds files by name pattern. `list_dir`
  lists one directory.
- `bash` runs a shell command in the repository root with a timeout. Quote paths.
  Prefer the dedicated file tools over `cat`, `sed` and `find`.
- `todo_write` records the plan you are following so the user can see progress; update
  it when a step completes.

## Environment

- Working directory: /home/dev/services/billing
- Platform: linux, Python 3.12, git repository on branch `feature/retry-backoff`
- The store behind `_store` is an in-memory fake in tests and a Postgres client in
  production; the fake lives in `services/billing/tests/fakes.py`.
- Telemetry helpers are no-ops under test.

Reply in the language of the user. When the task is complete, summarise what changed
and what you verified, in a few sentences, without repeating the diff.
"""

_TASK = """The nightly reconciliation job has started failing intermittently with
`TimeoutError` from the store when it walks large tenants, and the on-call runbook
says the fix is retry with backoff. Add a `retry_with_backoff` decorator to the core
package that retries on `TimeoutError` up to `MAX_RETRIES` times with exponential
backoff starting at `BACKOFF_SECONDS`, logging each attempt through the module `log`
and incrementing a `counter("...retry")`, and apply it to every service method that
touches the store. Keep the existing telemetry. Add tests. Start by finding every
service method that touches `_store` so we know the blast radius before editing.
"""

_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the repository and return its contents with line numbers.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root"},
            "offset": {"type": "integer", "description": "First line to return (1-based)"},
            "limit": {"type": "integer", "description": "Maximum number of lines to return"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file with the given contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace one exact occurrence of old_string with new_string in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence"}},
            "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command in the repository root and return stdout and stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "description": "Seconds before the command is killed"},
            "description": {"type": "string", "description": "What the command does, for the user"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search file contents with a regular expression; returns path:line:text.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"},
            "include": {"type": "string", "description": "Glob of files to search, e.g. *.py"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "glob",
        "description": "Find files whose path matches a glob pattern.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List the entries of one directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "todo_write",
        "description": "Replace the task list shown to the user.",
        "parameters": {"type": "object", "properties": {
            "todos": {"type": "array", "items": {"type": "object", "properties": {
                "content": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
                "required": ["content", "status"]}}},
            "required": ["todos"]}}},
]

# Tool-result sizes in tokens, before the `unique_tokens` scaling: three bands with the
# weights production shows for an agent turn -- most tool calls return a short grep,
# listing or file slice; some return a whole module; a few return a big file or a test
# log. Mean 2,310 at weight 1.0; median ~900. With the ~80-token canned call and the
# 832-token block the engine recomputes on every hit, that is ~3.2k mean / ~1.8k median
# uncached per request against production's 3.8k / ~1.5k.
_RESULT_SIZES = (  # (weight, low, high)
    (55, 300, 1000),
    (30, 1500, 3500),
    (15, 6000, 10000),
)
_RESULT_MEAN = sum(w * (lo + hi) / 2 for w, lo, hi in _RESULT_SIZES) / sum(
    w for w, _, _ in _RESULT_SIZES)


def _draw_result_tokens(rng: _Rng, unique_tokens: int) -> int:
    r = rng.below(100)
    acc = 0
    for w, lo, hi in _RESULT_SIZES:
        acc += w
        if r < acc:
            n = lo + rng.below(hi - lo + 1)
            break
    else:  # pragma: no cover -- weights sum to 100
        n = _RESULT_SIZES[0][1]
    return max(50, int(n * unique_tokens / _RESULT_MEAN))


def _with_line_numbers(text: str, start: int = 1) -> str:
    return "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(text.split("\n"), start))


def _tool_loop(rng: _Rng, n_tokens: int, call_id: str, prefix: str) -> tuple[dict, dict]:
    """One canned (assistant tool_call, tool result) pair of about n_tokens tokens.

    Three result flavours, all cut from this generator's own synthetic repo:
    read_file (a module with line numbers), grep (path:line:text hits), bash (a
    pytest run that fails, quoting generated lines). The assistant half carries no
    text and no reasoning -- opencode does not send reasoning back -- just the call.
    """
    kind = rng.pick(("read_file", "read_file", "grep", "bash"))
    budget = int(n_tokens * CHARS_PER_TOKEN_TOOL)
    name = f"{prefix}_{rng.pick(_VERBS)}_{rng.below(40):02d}"
    path = f"services/billing/core/{name}.py"
    if kind == "read_file":
        args = {"path": path}
        body = _module(rng, name)
        while len(body) < budget:
            body += "\n\n" + _module(rng, name)
        result = _with_line_numbers(body[:budget].rsplit("\n", 1)[0])
    elif kind == "grep":
        pat = rng.pick(("_store\\.", "@timed", "TimeoutError", "counter\\(", "def .*_id: str"))
        args = {"pattern": pat, "include": "*.py"}
        lines, used = [], 0
        while used < budget:
            mod = f"services/billing/core/{prefix}_{rng.pick(_VERBS)}_{rng.below(40):02d}.py"
            src = _module(rng, "m").split("\n")
            for i, line in enumerate(src, 1):
                if "_store" in line or "@timed" in line or "counter(" in line:
                    hit = f"{mod}:{i}:{line}"
                    lines.append(hit)
                    used += len(hit) + 1
                    if used >= budget:
                        break
        result = "\n".join(lines)
    else:
        args = {"command": "python -m pytest -q services/billing/tests -x",
                "description": "Run the billing test suite"}
        src = _module(rng, name).split("\n")
        out = ["============================= test session starts ==============================",
               "platform linux -- Python 3.12.6, pytest-8.3.3", "collected 148 items", ""]
        while sum(len(x) + 1 for x in out) < budget:
            fn = f"test_{rng.pick(_VERBS)}_{rng.pick(_DOMAINS)}"
            out += [f"____________________________ {fn} ____________________________", ""]
            for j in range(rng.below(12) + 6):
                out.append(f"    {src[(rng.below(len(src)))].strip()}")
            out += ["E   TimeoutError: store call exceeded 30s",
                    f"services/billing/tests/{fn}.py:{rng.below(200) + 10}: TimeoutError", ""]
        out.append("1 failed, 147 passed in 42.18s")
        result = "\n".join(out)[:budget]
    call = {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": kind, "arguments": json.dumps(args)}}]}
    reply = {"role": "tool", "tool_call_id": call_id, "content": result}
    return call, reply


def _est_tokens(messages: list[dict], tools: list[dict]) -> int:
    """Chars/token estimate of the rendered prompt; the served count replaces it."""
    n = 0
    for m in messages:
        if m["role"] == "tool":
            n += len(m["content"]) / CHARS_PER_TOKEN_TOOL + 8
        elif m.get("tool_calls"):
            n += len(json.dumps(m["tool_calls"][0]["function"])) / CHARS_PER_TOKEN_JSON + 12
        else:
            n += len(m["content"]) / CHARS_PER_TOKEN_PROSE + 6
    n += len(json.dumps(tools)) / CHARS_PER_TOKEN_JSON
    return int(n)


def _build_agentic(prompt_tokens: int, unique_tokens: int, index: int, tail_seed: int,
                   turn: int = 0) -> dict:
    """Turn `turn` of session `index`: messages + tools + the builder's token estimate.

    prompt_tokens  size of the turn-0 prompt; later turns grow by one loop each
    unique_tokens  mean tokens appended per turn (the cold part); 0 = no session loops,
                   which is only useful for prefix-cache experiments
    """
    messages = [{"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _TASK}]
    # Shared exploration, seed 0 on every run and every session: the loops any agent
    # on this task would have done first. Fills to the turn-0 budget less one session
    # loop, so the session's own first loop lands the prompt on prompt_tokens.
    fixed = _est_tokens(messages, _TOOLS)
    shared_budget = max(prompt_tokens - unique_tokens - fixed, 0)
    rng0, used, k = _Rng(0), 0, 0
    while used < shared_budget:
        want = min(_draw_result_tokens(rng0, max(unique_tokens, 1)), shared_budget - used)
        if want < 50:
            break
        call, reply = _tool_loop(rng0, want, f"call_shared_{k:03d}", "core")
        messages += [call, reply]
        used += _est_tokens([call, reply], [])
        k += 1
    # The session's own loops, seeded by (tail_seed, session): cold on first use, and
    # turn t contains turns 0..t-1's loops verbatim so they are hits.
    if unique_tokens:
        rng = _Rng(tail_seed * 100003 + index + 1)
        for t in range(turn + 1):
            want = _draw_result_tokens(rng, unique_tokens)
            call, reply = _tool_loop(rng, want, f"call_{tail_seed}_{index}_{t:02d}", f"s{index}")
            messages += [call, reply]
    return {"messages": messages, "tools": _TOOLS, "tool_choice": "auto",
            "est_tokens": _est_tokens(messages, _TOOLS)}


def build_turn(prompt_tokens: int, unique_tokens: int, index: int, tail_seed: int = 0,
               kind: str = "code", turn: int = 0) -> dict:
    """Request body fields for one request of any kind.

    `filler` and `code` are one user message; `agentic` is a whole chat with tools.
    Callers key the prefix cache on the returned messages exactly as built, so the list
    for turn t must be a strict prefix of the list for turn t+1 (test_loadgen checks).
    """
    if kind == "agentic":
        return _build_agentic(prompt_tokens, unique_tokens, index, tail_seed, turn)
    text = build_prompt(prompt_tokens, unique_tokens, index, tail_seed, kind)
    return {"messages": [{"role": "user", "content": text}], "est_tokens": None}


def contamination_hit_pct(kind: str, prompt_tokens: int, unique_tokens: int) -> float:
    """Above this cache-hit %, the cold tail was served from the prefix cache.

    `code`: the shared prefix is ~85% of the prompt; 92% was the hand-set line. `agentic`
    appends ~unique_tokens + a dropped 832-token block per turn on a ~prompt_tokens
    prompt, so the honest hit is ~94-95%; contamination reads ~99%. Halfway between.
    """
    if kind != "agentic":
        return 92.0
    honest = 100.0 * (1 - (unique_tokens + 832) / max(prompt_tokens, 1))
    return (honest + 99.5) / 2


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

_BUILDERS = {"filler": _build_filler, "code": _build_code}


def build_prompt(
    prompt_tokens: int,
    unique_tokens: int,
    index: int,
    tail_seed: int = 0,
    kind: str = "code",
) -> str:
    """Shared prefix + per-request cold tail, in the requested flavour.

    The shared part is seeded identically on every run, deliberately: it mimics
    opencode's ~11.8k system prompt, which really is identical across agents and stays
    hot in the prefix cache. The tail is seeded by tail_seed so a rep is reproducible
    while still being cold on first use — reusing indices across levels or runs leaves
    the tails in the prefix cache and reports ~99% hit, making prefill look far cheaper
    than it is in production.
    """
    try:
        return _BUILDERS[kind](prompt_tokens, unique_tokens, index, tail_seed)
    except KeyError:
        raise SystemExit(f"unknown --workload {kind!r}; expected one of {KINDS}") from None


def calibrate(url: str, model: str) -> float:
    """Measure chars/token for the `code` generator against a running server.

    Uses vLLM's /tokenize, which is CPU-side and does not touch the GPU — safe to run
    against the live serve. Prints the ratio to put in CHARS_PER_TOKEN.
    """
    import json
    import urllib.request

    sample = _repo_text(_Rng(0), 200_000, "cal")
    req = urllib.request.Request(
        url.rstrip("/") + "/tokenize",
        data=json.dumps({"model": model, "prompt": sample}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        n = json.load(resp)["count"]
    ratio = len(sample) / n
    print(f"chars={len(sample)} tokens={n} -> CHARS_PER_TOKEN = {ratio:.4f}")
    print(f"(file currently says {CHARS_PER_TOKEN})")
    return ratio


def calibrate_agentic(url: str, model: str) -> float:
    """Measure chars/token for one `agentic` turn against a running server's /tokenize.

    Posts the rendered chat (messages + tools) -- CPU-side, no GPU work. Prints the
    served count next to the builder's estimate. Tool results are ~85% of the prompt, so
    a drift is almost always CHARS_PER_TOKEN_TOOL; the per-piece breakdown that set the
    three constants on 2026-09-19 is in the header comment above them.
    """
    import urllib.request

    body = _build_agentic(65000, 2300, 1, 20260918, 0)
    est = body["est_tokens"]
    req = urllib.request.Request(
        url.rstrip("/") + "/tokenize",
        data=json.dumps({"model": model, "messages": body["messages"],
                         "tools": body["tools"], "add_generation_prompt": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        n = json.load(resp)["count"]
    print(f"served={n} estimated={est} -> estimate is {100 * (est / n - 1):+.1f}% off; "
          f"scale CHARS_PER_TOKEN_TOOL ({CHARS_PER_TOKEN_TOOL}) by {est / n:.4f}")
    return est / n


if __name__ == "__main__":  # ./workload.py code | head -40   |   ./workload.py agentic --turn 2
    import sys

    if "--calibrate-agentic" in sys.argv:
        i = sys.argv.index("--calibrate-agentic")
        rest = sys.argv[i + 1 :]
        calibrate_agentic(
            rest[0] if rest else "http://127.0.0.1:8000",
            rest[1] if len(rest) > 1 else "Qwen3.8-Flash-Next-NVFP4",
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "agentic":
        turn = int(sys.argv[sys.argv.index("--turn") + 1]) if "--turn" in sys.argv else 0
        body = build_turn(65000, 2300, 1, 20260918, "agentic", turn)
        print(json.dumps({k: v for k, v in body.items() if k != "tools"}, indent=1))
    elif "--calibrate" in sys.argv:
        i = sys.argv.index("--calibrate")
        rest = sys.argv[i + 1 :]
        calibrate(
            rest[0] if rest else "http://127.0.0.1:8000",
            rest[1] if len(rest) > 1 else "Qwen3.8-Flash-Next-NVFP4",
        )
    else:
        k = sys.argv[1] if len(sys.argv) > 1 else "code"
        print(build_prompt(18000, 2700, 1, 20260913, k))
