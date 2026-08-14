"""Single det (Determined CLI) invocation path: wrapper resolution, process
invocation, state parsing, and bounded polling.

Authority: Determined-status utility task spec (2026-08-14). This module is
the ONE place in the ADAGS repo that decides how to locate and run the `det`
executable; both `scripts/det_monitor.py` and `scripts/submit_apollo.py`
route every `det` call through :func:`resolve_det_command`, :func:`run_det`,
and :func:`check_invocation` here rather than shelling out independently.

Torch-free and side-effect-free at import time: pure functions and frozen
dataclasses only. NOTHING in this module prints -- only `scripts/det_monitor.py`'s
`cmd_*` functions print (mirroring `scripts/submit_apollo.py`'s convention).

Style: fail-closed, matching `scripts/submit_apollo.py`. Every user-facing
failure raises a `depth_visibility.errors.ContractError` or one of the
subclasses defined there (`DetInvocationError`, `DetParseError`,
`DetUnknownStateError`).

VERIFIED against a live cluster (Determined CLI 0.38.1 / master 0.38.0,
http://determined.intern.denayer.be:8080) on 2026-08-14. Every parsing
decision below traces to an actual probe of that cluster, not an assumption
-- see each function's docstring for what was observed and why it shapes the
parser. Two notable departures from a naive reading of the CLI:

  * `det experiment describe <id> --json` returns a JSON ARRAY containing
    one object (keys `config`, `experiment`, `jobSummary`), not a bare
    object. :func:`parse_experiment_state` unwraps this.
  * `det task list --json` returns a JSON OBJECT keyed by
    "<taskId>.<allocationNumber>" whose records carry NO `state` field at
    all in this CLI version -- the subcommand only ever lists currently
    active allocations, so presence IS the only state evidence it offers,
    and there is no `det task describe`. :func:`parse_task_state` documents
    the resulting design.

Also verified: `det` writes a version-skew warning to stderr on every
invocation ("Master version ... is less than CLI version ..."). Non-empty
stderr is therefore never treated as a failure signal anywhere in this
module -- only a nonzero return code is.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from depth_visibility.errors import (
    ContractError,
    DetInvocationError,
    DetParseError,
    DetUnknownStateError,
)

__all__ = [
    "DetCommand",
    "DetInvocation",
    "DetStatus",
    "PollObservation",
    "PollResult",
    "TERMINAL_STATES",
    "NONTERMINAL_STATES",
    "resolve_det_command",
    "run_det",
    "check_invocation",
    "classify_state",
    "parse_experiment_state",
    "parse_task_state",
    "parse_version_output",
    "validate_experiment_id",
    "validate_task_id",
    "experiment_status",
    "task_status",
    "poll_until_terminal",
]


# ---------------------------------------------------------------------------
# Wrapper resolution
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DetCommand:
    """A resolved, classified path to the `det` CLI, ready to prefix an argv."""

    resolved_path: str
    kind: str  # "executable" | "python-script"
    interpreter: str | None
    argv_prefix: tuple[str, ...]


#: Suffixes classified as a native executable without inspecting file bytes.
_EXECUTABLE_SUFFIXES = frozenset({".exe", ".com", ".bat", ".cmd"})
_PE_MAGIC = b"MZ"
_ELF_MAGIC = b"\x7fELF"
_SHEBANG_READ_BYTES = 4096


def resolve_det_command(
    explicit_path: str | None = None,
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> DetCommand:
    """Locate and classify the `det` executable.

    If `explicit_path` is given it is used verbatim (must exist as a file).
    Otherwise `which("det")` resolves it from PATH; `None` raises a loud
    :class:`ContractError` naming PATH as the thing to fix.

    Classification inspects the FILE, not just its name, in this order:

      1. suffix in {.exe, .com, .bat, .cmd} -> "executable".
      2. first bytes are the PE magic (`MZ`) or ELF magic (`\\x7fELF`) ->
         "executable" (this is the verified shape of
         `C:\\Users\\sucar\\AppData\\Roaming\\Python\\Python312\\Scripts\\det.exe`
         on this machine: a 108395-byte PE binary with no extensionless
         wrapper alongside it).
      3. the file starts with `#!` and that shebang line contains
         "python" (case-insensitive), OR the suffix is `.py` ->
         "python-script", `interpreter=sys.executable`.
      4. otherwise: raise :class:`ContractError` describing exactly what
         was found (first line and magic bytes) -- never guess.

    `NEVER use shell=True` -- this function only classifies a path; it does
    not invoke it.
    """

    if explicit_path is not None:
        candidate = str(explicit_path)
        if not os.path.isfile(candidate):
            raise ContractError(
                f"explicit det path does not exist or is not a regular file: {candidate}"
            )
    else:
        found = which("det")
        if found is None:
            raise ContractError(
                "the `det` CLI was not found on PATH (shutil.which('det') returned None); "
                "install the Determined CLI, or pass an explicit path"
            )
        candidate = str(found)
        if not os.path.isfile(candidate):
            raise ContractError(
                f"`det` resolved via PATH to {candidate!r} but that is not a regular file"
            )

    suffix = Path(candidate).suffix.lower()
    if suffix in _EXECUTABLE_SUFFIXES:
        return DetCommand(
            resolved_path=candidate, kind="executable", interpreter=None, argv_prefix=(candidate,)
        )

    try:
        with open(candidate, "rb") as handle:
            head = handle.read(_SHEBANG_READ_BYTES)
    except OSError as exc:
        raise ContractError(
            f"could not read det candidate to classify it: {candidate} ({exc})"
        ) from exc

    if head[:2] == _PE_MAGIC or head[:4] == _ELF_MAGIC:
        return DetCommand(
            resolved_path=candidate, kind="executable", interpreter=None, argv_prefix=(candidate,)
        )

    shebang_has_python = False
    if head.startswith(b"#!"):
        shebang_line = head.split(b"\n", 1)[0]
        shebang_has_python = "python" in shebang_line.decode("utf-8", "replace").lower()

    if shebang_has_python or suffix == ".py":
        return DetCommand(
            resolved_path=candidate,
            kind="python-script",
            interpreter=sys.executable,
            argv_prefix=(sys.executable, candidate),
        )

    first_line_preview = head.split(b"\n", 1)[0][:120].decode("utf-8", "replace")
    magic_preview = head[:8].hex()
    raise ContractError(
        f"cannot classify det candidate {candidate!r}: not a recognised executable suffix "
        f"{sorted(_EXECUTABLE_SUFFIXES)}, not PE/ELF magic, no python shebang, and suffix is "
        f"not .py. first line: {first_line_preview!r}; first 8 bytes (hex): {magic_preview!r}"
    )


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DetInvocation:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


def run_det(
    command: DetCommand,
    args: Sequence[str],
    *,
    timeout_seconds: float | None = 120.0,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> DetInvocation:
    """Run `det <args>` via `command.argv_prefix` and capture the result.

    Always passes a list argv with `shell=False`; never a shell string.
    Catches ONLY `OSError` and `subprocess.TimeoutExpired` and re-raises as
    :class:`DetInvocationError` (chained with `from exc`), including the
    full argv and the original exception text. No other exception type is
    caught here.

    Does NOT raise on a nonzero return code, and does NOT treat non-empty
    stderr as failure -- the live `det` CLI writes a version-skew warning to
    stderr on every call, success or not. Callers decide via
    :func:`check_invocation`.

    `timeout_seconds=None` disables the timeout entirely (passed through as
    `timeout=None` to `subprocess.run`), matching Python's own default. This
    is a deliberate widening of the spec's plain `float` type: it lets a
    long-running call such as `det experiment logs -f` (used by
    `scripts/submit_apollo.py`'s `cmd_logs`, which had no timeout at all
    before this refactor) keep its original unbounded-wait behaviour while
    every new caller in `scripts/det_monitor.py` keeps a sane 120s default.
    """

    argv = tuple(command.argv_prefix) + tuple(str(a) for a in args)
    start = time.monotonic()
    try:
        result = runner(
            list(argv),
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except OSError as exc:
        raise DetInvocationError(
            f"failed to invoke det (argv={list(argv)!r}): {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DetInvocationError(
            f"det invocation timed out after {timeout_seconds}s (argv={list(argv)!r}): {exc}"
        ) from exc
    duration = time.monotonic() - start
    return DetInvocation(
        argv=argv,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_seconds=duration,
    )


def check_invocation(inv: DetInvocation) -> None:
    """Raise :class:`DetInvocationError` if `inv.returncode != 0`.

    Quotes returncode, stderr, AND stdout (a `det` failure sometimes puts
    the actionable message on stdout, sometimes on stderr -- verified: `det
    experiment describe 999999 --json` puts "experiment '999999' not found"
    on stderr with empty stdout, but this is not guaranteed for every
    subcommand). Never triggered by stderr content alone.
    """

    if inv.returncode != 0:
        raise DetInvocationError(
            f"det invocation failed (argv={list(inv.argv)!r}, returncode={inv.returncode}); "
            f"stderr={inv.stderr!r}; stdout={inv.stdout!r}"
        )


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------


TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "CANCELED",
        "CANCELLED",
        "ERROR",
        "DELETED",
        "STATE_COMPLETED",
        "STATE_CANCELED",
        "STATE_ERROR",
        "STATE_DELETED",
    }
)
NONTERMINAL_STATES = frozenset(
    {
        "ACTIVE",
        "PAUSED",
        "QUEUED",
        "PULLING",
        "STARTING",
        "RUNNING",
        "STOPPING_COMPLETED",
        "STOPPING_CANCELED",
        "STOPPING_ERROR",
        "STATE_ACTIVE",
        "STATE_QUEUED",
        "STATE_PULLING",
        "STATE_STARTING",
        "STATE_RUNNING",
        "STATE_PAUSED",
    }
)


def _strip_state_prefix(value: str) -> str:
    return value[len("STATE_") :] if value.startswith("STATE_") else value


_TERMINAL_NORMALIZED = frozenset(_strip_state_prefix(s) for s in TERMINAL_STATES)
_NONTERMINAL_NORMALIZED = frozenset(_strip_state_prefix(s) for s in NONTERMINAL_STATES)


def classify_state(state: str) -> str:
    """Return "terminal" or "nonterminal" for a `det`-reported state string.

    Normalises case and an optional leading `STATE_` prefix (both prefixed
    and unprefixed spellings appear across `det` subcommands/versions -- the
    live `experiment describe` probe returned `STATE_COMPLETED`). An
    unrecognised state ALWAYS raises :class:`DetUnknownStateError`; it is
    never defaulted to nonterminal (a silently-defaulted unknown state
    would let a truly-finished experiment appear to poll forever).
    """

    if not isinstance(state, str) or not state.strip():
        raise DetUnknownStateError(f"state is not a non-empty string: {state!r}")
    normalized = _strip_state_prefix(state.strip().upper())
    if normalized in _TERMINAL_NORMALIZED:
        return "terminal"
    if normalized in _NONTERMINAL_NORMALIZED:
        return "nonterminal"
    raise DetUnknownStateError(f"unrecognized det state: {state!r}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _truncated(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def parse_experiment_state(stdout: str) -> str:
    """Parse `det experiment describe <id> --json` output and return the state.

    VERIFIED 2026-08-14 (det 0.38.1 / master 0.38.0, experiment 14,
    COMPLETED): the live output is a JSON ARRAY with exactly one object
    whose keys are `config`, `experiment`, `jobSummary` -- NOT a bare
    object, even though only one experiment ID was given. This function
    unwraps a single-element list before applying the spec's documented
    lookup order: `payload["experiment"]["state"]`, then
    `payload["state"]`. A missing/empty/non-string state, or output that
    is not JSON at all, always raises :class:`DetParseError` -- never
    returns `None` or `""`.
    """

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise DetParseError(
            f"det experiment describe output is not valid JSON: {_truncated(stdout)!r}"
        ) from exc

    if isinstance(payload, list):
        if not payload:
            raise DetParseError(
                f"det experiment describe returned an empty JSON array: {_truncated(stdout)!r}"
            )
        payload = payload[0]

    if not isinstance(payload, dict):
        raise DetParseError(
            f"det experiment describe output is not a JSON object (or array of one): "
            f"{_truncated(stdout)!r}"
        )

    state: Any = None
    experiment_block = payload.get("experiment")
    if isinstance(experiment_block, dict):
        state = experiment_block.get("state")
    if not isinstance(state, str) or not state:
        state = payload.get("state")
    if not isinstance(state, str) or not state:
        raise DetParseError(
            f"det experiment describe output has no usable state at "
            f"experiment.state or state: {_truncated(stdout)!r}"
        )
    return state


def parse_task_state(stdout: str, task_id: str) -> str:
    """Parse `det task list --json` output and return the state for `task_id`.

    VERIFIED 2026-08-14 (det 0.38.1 / master 0.38.0, two live active tasks
    -- one JupyterLab, one Command): the output is a JSON OBJECT keyed by
    `"<taskId>.<allocationNumber>"` (e.g.
    `"4fe6d93e-113c-4038-a79d-0ac8c9ae16bb.1"`), each value a record
    carrying its own `taskId` field. This function tolerates BOTH that
    shape and a plain list-of-records shape, matching on a record's
    `taskId` (or `id`) field equal to `task_id`.

    Critically, NEITHER live record carried a `state` key at all -- `det
    task list` in this CLI version has no `--json` field for state, and
    there is no `det task describe` subcommand to fall back to. Empirically
    it only ever lists currently-ACTIVE allocations. Given that reality:

      * if a matching record DOES carry a non-empty `state` string (a
        future CLI version might add one), it is returned verbatim;
      * otherwise, since a listed record is by construction a live
        allocation, the synthetic state `"ACTIVE"` is returned (a member
        of :data:`NONTERMINAL_STATES`);
      * if NO record matches `task_id`, this raises :class:`DetParseError`
        rather than inferring a terminal state -- absence from an
        active-only listing is not evidence the task ever reached a
        terminal state (it may never have existed, or the query may have
        failed silently upstream); "never default an unknown state to
        nonterminal" extends here to "never default an absence to
        terminal" for the same reason.
    """

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise DetParseError(
            f"det task list output is not valid JSON: {_truncated(stdout)!r}"
        ) from exc

    records: list[dict[str, Any]]
    if isinstance(payload, dict):
        records = [v for v in payload.values() if isinstance(v, dict)]
    elif isinstance(payload, list):
        records = [v for v in payload if isinstance(v, dict)]
    else:
        raise DetParseError(
            f"det task list output is neither a JSON object nor a JSON array: "
            f"{_truncated(stdout)!r}"
        )

    matches = [
        record
        for record in records
        if str(record.get("taskId") or record.get("id") or "") == task_id
    ]
    if not matches:
        raise DetParseError(
            f"task {task_id!r} not found in `det task list` output (this CLI's task list "
            "only shows currently-active allocations; absence is not evidence of a "
            f"terminal state): {_truncated(stdout)!r}"
        )

    state = matches[0].get("state")
    if isinstance(state, str) and state:
        return state
    return "ACTIVE"


def parse_version_output(stdout: str) -> dict[str, str]:
    """Parse `det version` output and return `{"client_version": ..., "master_version": ...}`.

    VERIFIED 2026-08-14: `det version` (no `--json` flag exists on this
    subcommand in 0.38.1) emits YAML, e.g.::

        client:
          version: 0.38.1
        master:
          ...
          version: 0.38.0
        master_address: http://...

    Parsed with PyYAML via a LOCAL import (PyYAML is already a repo
    dependency -- see `utils/hide_reveal_poc.py`,
    `tests/test_elgs_configs.py` -- but importing it locally keeps this
    module importable even in an environment that lacks it, unless this
    specific function is called).
    """

    import yaml  # local import: see docstring

    try:
        payload = yaml.safe_load(stdout)
    except yaml.YAMLError as exc:
        raise DetParseError(
            f"det version output is not valid YAML: {_truncated(stdout)!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise DetParseError(
            f"det version output is not a YAML mapping: {_truncated(stdout)!r}"
        )

    client_block = payload.get("client")
    master_block = payload.get("master")
    client_version = client_block.get("version") if isinstance(client_block, dict) else None
    master_version = master_block.get("version") if isinstance(master_block, dict) else None
    if not client_version or not master_version:
        raise DetParseError(
            f"det version output is missing client.version or master.version: "
            f"{_truncated(stdout)!r}"
        )
    return {"client_version": str(client_version), "master_version": str(master_version)}


# ---------------------------------------------------------------------------
# Identifier validation -- never interpolate untrusted values
# ---------------------------------------------------------------------------


_EXPERIMENT_ID_RE = re.compile(r"^[0-9]+$")
_TASK_ID_RE = re.compile(r"^[0-9a-fA-F][0-9a-fA-F-]{7,63}$")


def validate_experiment_id(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    if not _EXPERIMENT_ID_RE.match(text):
        raise ContractError(
            f"invalid experiment id (must match {_EXPERIMENT_ID_RE.pattern!r}): {value!r}"
        )
    return text


def validate_task_id(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    if not _TASK_ID_RE.match(text):
        raise ContractError(
            f"invalid task id (must match {_TASK_ID_RE.pattern!r}): {value!r}"
        )
    return text


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DetStatus:
    kind: str  # "experiment" | "task"
    identifier: str
    state: str
    terminal: bool
    observed_at_utc: str
    invocation: DetInvocation


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def experiment_status(
    command: DetCommand,
    experiment_id: Any,
    *,
    timeout_seconds: float | None = 120.0,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> DetStatus:
    """`det experiment describe <id> --json` -> validated, parsed, classified status."""

    exp_id = validate_experiment_id(experiment_id)
    invocation = run_det(
        command,
        ["experiment", "describe", exp_id, "--json"],
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    check_invocation(invocation)
    state = parse_experiment_state(invocation.stdout)
    terminal = classify_state(state) == "terminal"
    return DetStatus(
        kind="experiment",
        identifier=exp_id,
        state=state,
        terminal=terminal,
        observed_at_utc=_utc_now_iso(),
        invocation=invocation,
    )


def task_status(
    command: DetCommand,
    task_id: Any,
    *,
    timeout_seconds: float | None = 120.0,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> DetStatus:
    """`det task list --json` -> validated, parsed, classified status for `task_id`.

    See :func:`parse_task_state` for why this is the CLI form that
    actually works in 0.38.1 (there is no `det task describe`).
    """

    tid = validate_task_id(task_id)
    invocation = run_det(
        command, ["task", "list", "--json"], timeout_seconds=timeout_seconds, runner=runner
    )
    check_invocation(invocation)
    state = parse_task_state(invocation.stdout, tid)
    terminal = classify_state(state) == "terminal"
    return DetStatus(
        kind="task",
        identifier=tid,
        state=state,
        terminal=terminal,
        observed_at_utc=_utc_now_iso(),
        invocation=invocation,
    )


# ---------------------------------------------------------------------------
# Bounded polling
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PollObservation:
    poll_index: int
    status: DetStatus | None
    failure: str | None  # populated ONLY when tolerate_transient allowed a retry
    changed: bool  # state differs from the previous SUCCESSFUL observation


@dataclasses.dataclass(frozen=True)
class PollResult:
    observations: tuple[PollObservation, ...]
    final_status: DetStatus | None
    reached_terminal: bool
    exhausted: bool  # ran out of polls without reaching terminal


def poll_until_terminal(
    command: DetCommand,
    kind: str,
    identifier: Any,
    *,
    interval_seconds: float,
    max_polls: int,
    tolerate_transient: int = 0,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout_seconds: float | None = 120.0,
) -> PollResult:
    """Poll `kind`/`identifier` until a terminal state or `max_polls` is exhausted.

    No long-lived shell or background thread: each poll is one full
    synchronous `det` invocation. `sleeper(interval_seconds)` is called
    BETWEEN polls only (never before the first poll, never after a poll
    that reaches terminal or is the last allowed poll) -- inject a fake
    sleeper in tests so nothing actually sleeps.

    On an invocation/parse/unknown-state failure (anything raising
    `ContractError` -- `DetInvocationError`, `DetParseError`, and
    `DetUnknownStateError` all are one): if `tolerate_transient` budget
    remains, this records a :class:`PollObservation` with `status=None,
    failure=<message>` and consumes one unit of budget; otherwise it
    RE-RAISES. A failed observation is never recorded as `changed=False`
    "unchanged" -- `changed` is only ever computed between two SUCCESSFUL
    observations, so a caller can structurally tell "state didn't change"
    apart from "the last poll failed".

    `clock` is accepted for dependency-injection parity with `sleeper` (a
    caller wanting deterministic timing under test can inject one); it is
    not currently used to bound polling since `max_polls` already bounds
    it, and `DetInvocation.duration_seconds` (via `time.monotonic()`
    directly in :func:`run_det`) already carries per-call timing.
    """

    if kind == "experiment":
        def fetch() -> DetStatus:
            return experiment_status(
                command, identifier, timeout_seconds=timeout_seconds, runner=runner
            )
    elif kind == "task":
        def fetch() -> DetStatus:
            return task_status(
                command, identifier, timeout_seconds=timeout_seconds, runner=runner
            )
    else:
        raise ContractError(
            f"poll_until_terminal: unknown kind {kind!r} (expected 'experiment' or 'task')"
        )

    observations: list[PollObservation] = []
    last_successful_state: str | None = None
    remaining_transient_budget = int(tolerate_transient)
    reached_terminal = False

    for poll_index in range(int(max_polls)):
        if poll_index > 0:
            sleeper(interval_seconds)
        try:
            status = fetch()
        except ContractError as exc:
            if remaining_transient_budget > 0:
                remaining_transient_budget -= 1
                observations.append(
                    PollObservation(
                        poll_index=poll_index, status=None, failure=str(exc), changed=False
                    )
                )
                continue
            raise

        changed = last_successful_state is not None and status.state != last_successful_state
        observations.append(
            PollObservation(poll_index=poll_index, status=status, failure=None, changed=changed)
        )
        last_successful_state = status.state
        if status.terminal:
            reached_terminal = True
            break

    final_status = None
    for obs in reversed(observations):
        if obs.status is not None:
            final_status = obs.status
            break

    return PollResult(
        observations=tuple(observations),
        final_status=final_status,
        reached_terminal=reached_terminal,
        exhausted=not reached_terminal,
    )
