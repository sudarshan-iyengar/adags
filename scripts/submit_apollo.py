#!/usr/bin/env python3
"""EL-GS Determined submission wrapper for Apollo (DRAFT).

Authority: research-wiki/operations/elgs-m0-m1-implementation-plan.md S10.2
("Execution design"). This is the "new scripts/submit_apollo.py" that S10.1
finds no existing launcher to extend -- there is no recorded `det`
invocation, commit-isolated code path, immutable config record, or
Determined-aware ledger anywhere in the repo before this file.

Responsibilities (S10.2 items 1-7), each backed by a pure, independently
testable function:

  * execution closure           resolve_execution_set / check_execution_closure
  * isolated commit context     materialize_context
  * config identity              canonical_config_hash
  * provenance manifest          build_manifest
  * duplicate-submission guard   claim_cell
  * append-only history          append_ledger
  * placeholder rendering        render_template
  * in-container self-check      runtime_assertions

`submit`, `preflight`, `status`, `logs`, `cancel`, and `audit` are thin CLI
orchestrations over those functions; only `submit`, `preflight`, `status`,
`logs`, and `cancel` invoke the `det` CLI (never during CPU unit tests, which
exercise the pure functions directly).

Style: fail-closed. Every user-facing failure raises a
`depth_visibility.errors.ContractError` (or a subclass) with an actionable
message; nothing here prints from a library function -- only the CLI (`cmd_*`)
functions print.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.artifacts import (  # noqa: E402
    atomic_write_json_immutable,
    build_inventory,
    write_terminal_last,
)
from depth_visibility.canonical import canonical_json_bytes, sha256_file  # noqa: E402
from depth_visibility.errors import ArtifactError, ContractError, SchemaError  # noqa: E402
from elgs import determined  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Directories whose contents are execution-relevant for an EL-GS Apollo run
#: (S10.2 item 1). Matches the plan's declared set verbatim.
EXECUTION_DIRS: tuple[str, ...] = (
    "elgs",
    "scene",
    "gaussian_renderer",
    "utils",
    "arguments",
    "depth_visibility",
)

#: Individual execution-relevant files, always in the set regardless of the
#: run's named config(s).
EXECUTION_FILES: tuple[str, ...] = (
    "main.py",
    "scripts/submit_apollo.py",
    "det_exp_apollo.yaml",
)

#: The only scripts the experiment template may exec (fail-closed allowlist;
#: M1 adds the tracks-preprocessing and census-cell entrypoints alongside the
#: trainer). Non-``main.py`` entrypoints receive ONLY the caller's
#: ``--extra-arg`` tokens -- their CLIs are self-contained -- while the named
#: ``--config`` still enters the closure set, the manifest, and the runtime
#: config-hash contract (for census cells the config IS the frozen prereg).
ALLOWED_ENTRYPOINT_SCRIPTS: tuple[str, ...] = (
    "main.py",
    "scripts/build_elgs_tracks.py",
    "scripts/build_m1_census.py",
    "scripts/apply_cycle3_gate.py",
    "scripts/build_absence_diagnostic.py",
    "scripts/build_coverage_bounding_pair.py",
    "scripts/build_a0b_camera_mapping.py",
    "scripts/indep_coverage_recompute.py",
    "scripts/imvid_pilot_prepare.py",
    "scripts/imvid_sparse_init.py",
    "scripts/imvid_build_initialization.py",
    "scripts/imvid_verify_pinhole.py",
    "scripts/fetch_immersive_scene.py",
    "scripts/benchmark_elgs_q.py",
    "scripts/eval_diva360_heldout.py",
    "scripts/audit_mechanism.py",
    "scripts/flow_plumbing_smoke.py",
    "scripts/verify_flow_vjp_runtime.py",
    "scripts/eval_lrv1_event.py",
    "scripts/estimate_episodes.py",
    "scripts/consolidate_packets.py",
    "scripts/falsify_b2_edit.py",
    "scripts/payload_headroom.py",
)

#: The one shared, mutable, historical worktree path. No evidence-bearing
#: code is ever allowed to execute from here (S10.2 item 2).
FORBIDDEN_WORKTREE = "/apollo/users/sri/proj_adags/repo/adags"

# Owner decision (recorded in the Apollo execution-authority wiki page):
# all EL-GS experiments live in this workspace/project, created-if-absent
# at preflight (plan §16b).
WORKSPACE_NAME = "adags"
PROJECT_NAME = "elgs"

#: Filename of the run manifest materialize_context/submit write into the
#: git-archive context -- "the only non-repo file" per S10.2 item 2.
MANIFEST_FILENAME = "elgs_run_manifest.json"

#: Mirrors ``main.py``'s ``DEFAULT_MAX_TRAIN_ITERATIONS``. Substituted into
#: the template's ``ADAGS_MAX_ITERATIONS`` so an unraised ceiling renders to
#: exactly the value main.py would have used with the variable unset.
#: ``tests/test_submit_apollo.py`` asserts the two stay equal; duplicated
#: rather than imported because importing main.py pulls in torch.
DEFAULT_MAX_TRAIN_ITERATIONS = 6000

MANIFEST_SCHEMA = "elgs-apollo-run-manifest-v1"
CLAIM_SCHEMA = "elgs-apollo-claim-v1"
LEDGER_SCHEMA = "elgs-apollo-ledger-v1"
TERMINAL_SCHEMA = "elgs-apollo-terminal-v1"

_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")
_EXPERIMENT_ID_RE = re.compile(r"[Ee]xperiment(?:\s+ID)?\s*#?\s*(\d+)")


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------


def _utc_now_stamp() -> str:
    """Return an ISO-8601 UTC timestamp with a literal ``Z`` offset."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _relative_repo_path(repo_root: Path, raw: os.PathLike[str] | str) -> str:
    """Return ``raw`` as a POSIX path relative to ``repo_root``.

    Raises :class:`ContractError` if ``raw`` does not resolve under
    ``repo_root`` (a config path outside the repo cannot be part of a
    commit-isolated execution set).
    """

    candidate = Path(raw)
    absolute = candidate if candidate.is_absolute() else (repo_root / candidate)
    try:
        relative = absolute.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ContractError(f"path escapes repo root {repo_root}: {raw}") from exc
    return relative.as_posix()


def _run_git(repo_root: Path, *args: str) -> str:
    """Run ``git <args>`` in ``repo_root`` and return RAW (unstripped) stdout.

    Deliberately does not ``.strip()`` the output: ``git status --porcelain``
    status codes are column-position-sensitive (` M path`, `??path` with a
    leading space that IS the "unmodified in index" column), and stripping
    the whole blob eats that leading space and shifts every column by one.
    Callers that want a single trimmed token (a commit SHA, a branch name)
    strip explicitly at the call site instead.
    """

    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ContractError(
            f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def git_rev_parse(repo_root: os.PathLike[str] | str, rev: str = "HEAD") -> str:
    """Return the full commit SHA for ``rev``."""

    return _run_git(Path(repo_root), "rev-parse", rev).strip()


def git_current_branch(repo_root: os.PathLike[str] | str) -> str:
    """Return the current branch name (``HEAD`` if detached)."""

    return _run_git(Path(repo_root), "rev-parse", "--abbrev-ref", "HEAD").strip()


# ---------------------------------------------------------------------------
# Execution closure (S10.2 item 1)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ClosureReport:
    """Outcome of an execution-closure check against ``git status``.

    ``dirty_inside`` and ``dirty_outside`` are sorted tuples of repo-relative
    POSIX paths reported dirty or untracked by ``git status --porcelain``,
    split by whether they fall under the declared execution set.
    """

    execution_set: tuple[str, ...]
    dirty_inside: tuple[str, ...]
    dirty_outside: tuple[str, ...]
    evidence_bearing: bool
    dirty_smoke: bool


def resolve_execution_set(
    repo_root: os.PathLike[str] | str,
    config_paths: Iterable[os.PathLike[str] | str],
    entrypoint_script: str | None = None,
) -> list[str]:
    """Return the declared execution-relevant path set (S10.2 item 1).

    Always includes :data:`EXECUTION_DIRS` and :data:`EXECUTION_FILES`, plus
    every path in ``config_paths`` normalized relative to ``repo_root``, plus
    ``entrypoint_script`` when one is named. Every entry must exist under
    ``repo_root`` or this raises :class:`ContractError` (a missing declared
    path is repo drift, not a dirty-tree condition, so it fails closed here
    rather than being silently skipped by the closure check).

    ``entrypoint_script`` closes a gap recorded in
    ``research-wiki/operations/block-2026-08-23-live-state-and-budget.md``
    section 9: ``scripts/`` is outside :data:`EXECUTION_DIRS` apart from this
    wrapper, so a DIRTY working copy of a non-``main.py`` entrypoint did not
    block a submission that used it. The container runs the committed
    snapshot, so the result was not wrong -- it was silently misleading,
    which is exactly what the closure check exists to prevent for
    ``main.py``. Only the entrypoint actually used enters the set, so
    editing an unrelated script never blocks an unrelated cell.
    """

    root = Path(repo_root)
    entries: set[str] = set(EXECUTION_DIRS) | set(EXECUTION_FILES)
    for raw in config_paths:
        entries.add(_relative_repo_path(root, raw))
    if entrypoint_script is not None:
        entries.add(_relative_repo_path(root, entrypoint_script))
    missing = sorted(entry for entry in entries if not (root / entry).exists())
    if missing:
        raise ContractError(
            f"declared execution-relevant path(s) are missing from the repo: {missing}"
        )
    return sorted(entries)


def _porcelain_paths(repo_root: Path) -> list[str]:
    """Parse ``git status --porcelain`` into a flat list of touched paths.

    Rename/copy lines (``R  old -> new`` / ``C  old -> new``) contribute
    both sides; every path is stripped of a trailing ``/`` (git reports an
    untracked directory as a single ``dirname/`` entry) and any quoting git
    applies to paths containing unusual characters.
    """

    output = _run_git(repo_root, "status", "--porcelain")
    raw_paths: list[str] = []
    for line in output.splitlines():
        if not line:
            continue
        if len(line) < 4:
            raise ContractError(f"unparseable `git status --porcelain` line: {line!r}")
        rest = line[3:]
        if " -> " in rest:
            old, _, new = rest.partition(" -> ")
            raw_paths.append(old)
            raw_paths.append(new)
        else:
            raw_paths.append(rest)
    return [path.strip().strip('"').rstrip("/") for path in raw_paths]


def _entry_covers(entry: str, path: str) -> bool:
    """True if declared ``entry`` and reported ``path`` denote overlapping content.

    Covers three cases: exact match; ``entry`` is a directory containing
    ``path``; and ``path`` is a directory git collapsed that contains the
    (file-level) ``entry`` -- e.g. git reports an entirely untracked
    ``configs/elgs/`` as one line even though the execution set names an
    individual file inside it.
    """

    return entry == path or path.startswith(entry + "/") or entry.startswith(path + "/")


def check_execution_closure(
    repo_root: os.PathLike[str] | str,
    execution_set: Iterable[str],
    *,
    dirty_smoke: bool = False,
    exploratory: bool = False,
) -> ClosureReport:
    """Classify the working tree against the declared execution set (S10.2 item 1).

    Any dirty or untracked path inside ``execution_set`` refuses the run
    with :class:`ContractError` unless ``dirty_smoke`` is set, in which case
    the run is permitted but the returned report carries
    ``evidence_bearing=False`` (S10.2 item 1's ``--dirty-smoke`` override).
    Dirty/untracked paths outside the set (e.g. the two user-owned
    ``research-wiki/`` files) are never a refusal reason; they are only
    reported in ``dirty_outside`` for the manifest to record as excluded.
    """

    root = Path(repo_root)
    entries = sorted(set(execution_set))
    dirty_inside: set[str] = set()
    dirty_outside: set[str] = set()
    for path in _porcelain_paths(root):
        if any(_entry_covers(entry, path) for entry in entries):
            dirty_inside.add(path)
        else:
            dirty_outside.add(path)

    if dirty_inside and not dirty_smoke:
        raise ContractError(
            "execution closure violated -- these declared execution-relevant paths "
            f"are dirty or untracked: {sorted(dirty_inside)}. Commit or discard them, "
            "or pass --dirty-smoke for a deliberately non-evidence-bearing run."
        )

    # `evidence_bearing` was previously derived SOLELY from working-tree
    # cleanliness, which conflates two independent things: whether the
    # code is reproducible, and whether the run is scientifically
    # claim-bearing. A clean tree could therefore never be declared
    # exploratory, and nothing stopped an evidence-bearing ledger line
    # from consuming unfrozen smoke-tier constants. `--exploratory` lets
    # a run say what it IS. It can only ever REMOVE evidence-bearing
    # status, never grant it: a dirty tree stays non-evidence-bearing
    # regardless.
    return ClosureReport(
        execution_set=tuple(entries),
        dirty_inside=tuple(sorted(dirty_inside)),
        dirty_outside=tuple(sorted(dirty_outside)),
        evidence_bearing=(not dirty_inside) and not exploratory,
        dirty_smoke=bool(dirty_smoke),
    )


# ---------------------------------------------------------------------------
# Isolated commit materialization (S10.2 item 2)
# ---------------------------------------------------------------------------


def _safe_extract_tar(tar: tarfile.TarFile, out_dir: Path) -> None:
    out_root = out_dir.resolve()
    for member in tar.getmembers():
        member_path = (out_dir / member.name).resolve()
        if member_path != out_root and not str(member_path).startswith(str(out_root) + os.sep):
            raise ContractError(f"archive member escapes context directory: {member.name}")
    try:
        tar.extractall(path=out_dir, filter="data")  # Python >= 3.12, PEP 706
    except TypeError:
        # Older Python without the `filter` kwarg; membership was already
        # validated above so a plain extract is safe.
        tar.extractall(path=out_dir)


def materialize_context(
    repo_root: os.PathLike[str] | str,
    commit: str,
    out_dir: os.PathLike[str] | str,
) -> str:
    """Materialize ``commit`` into ``out_dir`` via ``git archive`` and return its SHA-256.

    ``out_dir`` must exist and be empty. Streams ``git archive --format=tar
    <commit>`` to a temporary file, hashes it with
    :func:`depth_visibility.canonical.sha256_file` (reused, not
    reimplemented), then extracts it into ``out_dir`` with path-traversal
    checking. The caller is responsible for writing the run manifest into
    ``out_dir`` afterward -- this function only produces the code snapshot.
    """

    root = Path(repo_root)
    target = Path(out_dir)
    if not target.is_dir():
        raise ContractError(f"materialize_context target is not a directory: {target}")
    if any(target.iterdir()):
        raise ContractError(f"materialize_context target directory is not empty: {target}")

    descriptor, tmp_name = tempfile.mkstemp(prefix="elgs-archive-", suffix=".tar")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            result = subprocess.run(
                ["git", "-C", str(root), "archive", "--format=tar", str(commit)],
                stdout=handle,
                stderr=subprocess.PIPE,
            )
        if result.returncode != 0:
            raise ContractError(
                f"git archive {commit} failed (exit {result.returncode}): "
                f"{result.stderr.decode('utf-8', 'replace').strip()}"
            )
        archive_sha256 = sha256_file(tmp_path)
        with tarfile.open(tmp_path, mode="r:") as tar:
            _safe_extract_tar(tar, target)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    return archive_sha256


# ---------------------------------------------------------------------------
# Config identity (S10.2 item 3)
# ---------------------------------------------------------------------------


def canonical_config_hash(
    paths: Iterable[os.PathLike[str] | str],
    *,
    relative_to: os.PathLike[str] | str | None = None,
) -> str:
    """Return a stable SHA-256 identity over a set of config files.

    JSON files are parsed and re-encoded with
    :func:`depth_visibility.canonical.canonical_json_bytes` (so formatting
    and key order never change the identity); YAML files are hashed as raw
    bytes (so formatting DOES change the identity -- YAML has no canonical
    form in this repo). ``paths`` order does not matter; the per-file
    ``(path, sha256)`` pairs are sorted by path before the combining hash is
    taken, so the result is deterministic regardless of iteration order.

    The caller must pass paths relative to the same logical root at every
    call site (repo root at submission time, context root at runtime) so a
    submission-time hash and a runtime re-hash of "the same" config agree.
    """

    entries: list[tuple[str, str]] = []
    for raw in paths:
        source = Path(raw)
        if not source.is_file():
            raise ContractError(f"config path does not exist: {source}")
        suffix = source.suffix.lower()
        if suffix == ".json":
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SchemaError(f"config is not valid JSON: {source}") from exc
            body = canonical_json_bytes(payload)
        elif suffix in (".yaml", ".yml"):
            body = source.read_bytes()
        else:
            raise ContractError(f"unsupported config extension for canonical hashing: {source}")
        # The entry KEY must be root-independent, or a submission-time
        # hash under a temp context dir can never match the runtime
        # re-hash under /run/determined/workdir (the second S1 smoke
        # caught exactly this: identical bytes, different absolute
        # path strings, different combined hash).
        key = (
            source.resolve().relative_to(Path(relative_to).resolve()).as_posix()
            if relative_to is not None
            else Path(raw).as_posix()
        )
        entries.append((key, sha256(body).hexdigest()))
    entries.sort(key=lambda item: item[0])
    combined = canonical_json_bytes(
        {"config_files": [{"path": path, "sha256": digest} for path, digest in entries]}
    )
    return sha256(combined).hexdigest()


# ---------------------------------------------------------------------------
# Provenance manifest (S10.2 item 4)
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    repo_root: os.PathLike[str] | str,
    commit: str,
    branch: str,
    config_paths: Iterable[os.PathLike[str] | str],
    image_ref: str,
    pool: str,
    slots: int,
    seed: int,
    run_dir: str,
    wrapper_argv: Sequence[str],
    evidence_bearing: bool,
    projected_gpu_hours: float,
    dataset_manifest_path: os.PathLike[str] | str | None = None,
    prereg_dir: os.PathLike[str] | str | None = None,
    submitter_host: str | None = None,
    utc_stamp: str | None = None,
    hash_root: os.PathLike[str] | str | None = None,
    entrypoint_script: str = "main.py",
) -> dict[str, Any]:
    """Build the pre-submission run manifest (S10.2 item 4).

    Every field the plan requires: commit SHA, branch, config path+hash
    pairs (raw file SHA-256, for audit) plus one combined canonical hash
    (for runtime re-verification, S10.2 item 3), every
    ``configs/elgs/prereg_*.json`` hash, image ref, pool, slots, seed, an
    optional dataset manifest hash (``None`` is valid for a non-DiVa-360
    smoke run), the output run dir, the wrapper's own argv and file SHA-256,
    submitter hostname, a UTC stamp, the closure check's ``evidence_bearing``
    flag, and the caller-supplied GPU-hour projection. Experiment/task IDs
    and terminal state are NOT included here -- they are recorded after
    submission (into a sibling ``<claim>.experiment`` file and, at
    completion, ``terminal.json`` via :func:`cmd_audit`).
    """

    root = Path(repo_root)
    # Hashes must describe the bytes that TRAVEL. When the caller has
    # materialized the git-archive context, hash_root points at it —
    # otherwise a Windows CRLF worktree hashes differently from the
    # LF archive the container re-verifies (the exact fail-closed
    # mismatch the first S1 smoke caught).
    files_root = Path(hash_root) if hash_root is not None else root
    config_entries = []
    for raw in config_paths:
        relative = _relative_repo_path(root, raw)
        config_entries.append(
            {"path": relative, "sha256": sha256_file(files_root / relative)}
        )
    config_entries.sort(key=lambda item: item["path"])

    prereg_root = (
        Path(prereg_dir) if prereg_dir is not None else files_root / "configs" / "elgs"
    )
    prereg_entries = []
    if prereg_root.is_dir():
        for path in sorted(prereg_root.glob("prereg_*.json")):
            prereg_entries.append(
                {"path": _relative_repo_path(files_root, path), "sha256": sha256_file(path)}
            )

    dataset_manifest = None
    if dataset_manifest_path is not None:
        dataset_path = Path(dataset_manifest_path)
        if not dataset_path.is_file():
            raise ContractError(f"dataset manifest does not exist: {dataset_path}")
        dataset_manifest = {"path": str(dataset_path), "sha256": sha256_file(dataset_path)}

    wrapper_candidate = files_root / "scripts" / "submit_apollo.py"
    wrapper_file = (
        wrapper_candidate if wrapper_candidate.is_file() else Path(__file__).resolve()
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "commit": str(commit),
        "branch": str(branch),
        "config_files": config_entries,
        "config_canonical_hash": canonical_config_hash(
            (files_root / entry["path"] for entry in config_entries),
            relative_to=files_root,
        ),
        "prereg_files": prereg_entries,
        "image_ref": str(image_ref),
        "pool": str(pool),
        "slots": int(slots),
        "seed": int(seed),
        "dataset_manifest": dataset_manifest,
        "run_dir": str(run_dir),
        "wrapper": {
            "argv": [str(item) for item in wrapper_argv],
            "file_sha256": sha256_file(wrapper_file),
        },
        "submitter_host": submitter_host or socket.gethostname(),
        "utc_stamp": utc_stamp or _utc_now_stamp(),
        "evidence_bearing": bool(evidence_bearing),
        "projected_gpu_hours": float(projected_gpu_hours),
        "entrypoint_script": str(entrypoint_script),
    }


# ---------------------------------------------------------------------------
# Duplicate-submission guard (S10.2 item 5)
# ---------------------------------------------------------------------------


def claim_cell(
    claims_dir: os.PathLike[str] | str,
    cell_name: str,
    retry: int,
    *,
    payload: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically claim ``<cell_name>__r<retry>.json`` under ``claims_dir``.

    Reuses :func:`depth_visibility.artifacts.atomic_write_json_immutable`
    for the O_EXCL-style immutable write (S10.2 item 5) rather than
    reimplementing it. Raises :class:`ContractError` with an actionable
    message if the claim already exists (race-safe duplicate-submission
    guard: the claim is written *before* ``det e create`` runs).
    """

    directory = Path(claims_dir)
    directory.mkdir(parents=True, exist_ok=True)
    claim_path = directory / f"{cell_name}__r{int(retry)}.json"
    body = {
        "schema": CLAIM_SCHEMA,
        "cell_name": str(cell_name),
        "retry": int(retry),
        "claimed_at_utc": _utc_now_stamp(),
        **dict(payload or {}),
    }
    try:
        atomic_write_json_immutable(claim_path, body)
    except ArtifactError as exc:
        raise ContractError(
            f"cell already claimed (EEXIST): {claim_path} -- inspect the claim file and "
            "`det experiment list` to resolve, then resubmit with a new --retry counter"
        ) from exc
    return claim_path


# ---------------------------------------------------------------------------
# Append-only ledger (S10.2 item 5)
# ---------------------------------------------------------------------------


def append_ledger(ledger_path: os.PathLike[str] | str, record: Mapping[str, Any]) -> None:
    """Append one JSON line (schema ``elgs-apollo-ledger-v1``) to ``ledger_path``.

    A single O_APPEND write of one line, mirroring
    ``scripts/submit_phase9_depth_visibility.sh``'s ``JOB_LEDGER.jsonl``
    shape. ``record["schema"]`` is always forced to :data:`LEDGER_SCHEMA`,
    even if the caller supplied a different value, so every ledger line is
    unambiguous to a downstream reader regardless of caller discipline.
    """

    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {**dict(record), "schema": LEDGER_SCHEMA}
    try:
        encoded = json.dumps(
            body, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError("ledger record is not finite JSON data") from exc
    line = (encoded + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# Template rendering (S10.2 item 6)
# ---------------------------------------------------------------------------


def render_template(
    template_path: os.PathLike[str] | str,
    substitutions: Mapping[str, str],
) -> str:
    """Render ``{{PLACEHOLDER}}`` markers in ``template_path`` and verify completeness.

    Raises :class:`ContractError` if the template references a placeholder
    absent from ``substitutions``, or if any ``{{...}}``-shaped text
    (original or reintroduced by a substituted value) survives in the
    rendered output -- det_exp_apollo.yaml is never valid to submit with a
    live placeholder in it.
    """

    text = Path(template_path).read_text(encoding="utf-8")

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in substitutions:
            raise ContractError(f"template placeholder has no substitution: {{{{{key}}}}}")
        return str(substitutions[key])

    rendered = _PLACEHOLDER_RE.sub(_replace, text)
    leftover = sorted(set(_PLACEHOLDER_RE.findall(rendered)))
    if leftover:
        raise ContractError(f"rendered template still contains placeholder(s): {leftover}")
    return rendered


# ---------------------------------------------------------------------------
# Entrypoint-side self-check (S10.2 item 2, requirement 3)
# ---------------------------------------------------------------------------


def _refuse_forbidden_worktree(current_dir: str) -> None:
    """Raise if ``current_dir`` is (or is inside) :data:`FORBIDDEN_WORKTREE`."""

    normalized = os.path.realpath(current_dir).replace("\\", "/")
    forbidden = os.path.realpath(FORBIDDEN_WORKTREE).replace("\\", "/")
    if normalized == forbidden or normalized.startswith(forbidden + "/"):
        raise ContractError(
            f"refusing to execute from the shared mutable worktree ({FORBIDDEN_WORKTREE}); "
            "EL-GS runs must execute from a materialized git-archive context (S10.2 item 2)"
        )


def _assert_path_within_root(candidate: os.PathLike[str] | str, root: os.PathLike[str] | str, label: str) -> None:
    """Raise unless ``candidate`` resolves under ``root`` (``os.path.commonpath``)."""

    candidate_real = os.path.realpath(candidate)
    root_real = os.path.realpath(root)
    if os.path.commonpath([candidate_real, root_real]) != root_real:
        raise ContractError(
            f"{label} does not resolve under the execution context root: "
            f"{candidate_real} is not inside {root_real}"
        )


def runtime_assertions(*, cwd: str | None = None, manifest_filename: str = MANIFEST_FILENAME) -> None:
    """In-container self-check the entrypoint runs before ``main.py`` (S10.2 item 2/3).

    Raises :class:`ContractError` -- never returns a falsy status -- if any
    of the following hold:

      * the process is executing from :data:`FORBIDDEN_WORKTREE`;
      * ``elgs`` or ``main`` do not resolve under the current working
        directory (i.e. the process is not running from the isolated
        git-archive context Determined uploaded); or
      * the merged run config's canonical hash (recomputed with
        :func:`canonical_config_hash`) disagrees with the manifest written
        into the context at submission time.

    This function is only meaningfully callable inside the baked Apollo
    runtime image, where ``elgs``/``main`` and their transitive (CUDA)
    dependencies are importable -- it is intentionally never exercised by
    the CPU unit test suite. :func:`_refuse_forbidden_worktree` and
    :func:`_assert_path_within_root` carry the pure, directly-tested logic
    behind it.
    """

    current_dir = os.path.realpath(cwd if cwd is not None else os.getcwd())
    _refuse_forbidden_worktree(current_dir)

    import elgs as _elgs_module  # local import: only available in-container
    import main as _main_module  # local import: only available in-container

    _assert_path_within_root(_elgs_module.__file__, current_dir, "elgs.__file__")
    _assert_path_within_root(_main_module.__file__, current_dir, "main.__file__")

    manifest_path = Path(current_dir) / manifest_filename
    if not manifest_path.is_file():
        raise ContractError(f"run manifest is missing from the context root: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_hash = manifest.get("config_canonical_hash")
    config_files = manifest.get("config_files") or []
    if recorded_hash and config_files:
        recomputed = canonical_config_hash(
            (Path(current_dir) / entry["path"] for entry in config_files),
            relative_to=current_dir,
        )
        if recomputed != recorded_hash:
            raise ContractError(
                "merged run config hash disagrees with the submission-time manifest "
                f"(recorded {recorded_hash}, recomputed {recomputed}); refusing to train"
            )

    # Make the RUN DIR self-describing. The manifest is written into the
    # uploaded context, so before this nothing ever placed it beside the
    # artifacts and `cmd_audit` -- which requires it there -- could never
    # succeed on any run. Copying it here, after validation and before
    # any training, means a run directory always records the commit,
    # config hash, image and evidence-bearing status that produced it.
    # The write is O_EXCL: a second task writing into the same run dir
    # fails loudly rather than silently overwriting another run's
    # provenance (the template's "never a silent second run under the
    # same experiment ID" requirement).
    run_dir = os.environ.get("ADAGS_RUN_DIR", "").strip()
    if run_dir:
        target = Path(run_dir) / manifest_filename
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json_immutable(target, manifest)


# ---------------------------------------------------------------------------
# CLI: det invocation helpers (never used by the CPU test suite)
# ---------------------------------------------------------------------------


def _det_version() -> str:
    command = determined.resolve_det_command()
    invocation = determined.run_det(command, ["--version"])
    determined.check_invocation(invocation)
    return invocation.stdout.strip() or invocation.stderr.strip()


def _parse_experiment_id(stdout: str) -> int:
    match = _EXPERIMENT_ID_RE.search(stdout)
    if not match:
        raise ContractError(f"could not parse an experiment ID from `det e create` output: {stdout!r}")
    return int(match.group(1))


def _build_run_id(*, cell: str, seed: int, commit: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{cell}_{seed}_{commit[:7]}"


def _build_entrypoint_args(
    *,
    config_path: str,
    run_dir: str,
    extra_args: Sequence[str],
    entrypoint_script: str = "main.py",
) -> str:
    if entrypoint_script not in ALLOWED_ENTRYPOINT_SCRIPTS:
        raise ContractError(
            f"entrypoint script {entrypoint_script!r} is not in the allowlist "
            f"{ALLOWED_ENTRYPOINT_SCRIPTS}"
        )
    if entrypoint_script == "main.py":
        tokens = ["--config", config_path, "--model_path", run_dir, *extra_args]
    else:
        tokens = list(extra_args)
        if not tokens:
            raise ContractError(
                f"entrypoint {entrypoint_script!r} takes its whole CLI from "
                "--extra-arg tokens; none were given"
            )
    return " ".join(shlex.quote(str(token)) for token in tokens)


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def cmd_preflight(args: argparse.Namespace) -> int:
    version = _det_version()
    print(f"det CLI: {version}")

    repo_root = Path(args.repo_root).resolve()
    execution_set = resolve_execution_set(
        repo_root, [args.config], entrypoint_script=args.entrypoint_script
    )
    closure = check_execution_closure(
        repo_root, execution_set, dirty_smoke=args.dirty_smoke,
        exploratory=args.exploratory,
    )
    print("execution closure: " + ("OK (evidence-bearing)" if closure.evidence_bearing else "DIRTY-SMOKE (not evidence-bearing)"))
    if closure.dirty_outside:
        print(f"excluded (dirty/untracked, outside execution set): {list(closure.dirty_outside)}")

    stub_substitutions = {
        "NAME": args.cell,
        "POOL": args.pool,
        "IMAGE_REF": args.image_ref,
        "ENTRYPOINT_SCRIPT": args.entrypoint_script,
        "ENTRYPOINT_ARGS": "--config placeholder --model_path placeholder",
        "RUN_DIR": "placeholder",
        "MAX_TRAIN_ITERATIONS": str(args.max_train_iterations),
    }
    render_template(args.template, stub_substitutions)
    print(f"template renders cleanly: {args.template}")

    if args.dataset_manifest:
        if not Path(args.dataset_manifest).is_file():
            raise ContractError(f"dataset manifest is missing: {args.dataset_manifest}")
        print(f"dataset manifest present: {args.dataset_manifest}")

    claims_dir = Path(args.claims_dir)
    claims_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(claims_dir, os.W_OK):
        raise ContractError(f"claims directory is not writable: {claims_dir}")
    print(f"claims directory writable: {claims_dir}")

    ledger_dir = Path(args.ledger).parent
    ledger_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(ledger_dir, os.W_OK):
        raise ContractError(f"ledger directory is not writable: {ledger_dir}")
    print(f"ledger directory writable: {ledger_dir}")

    ensure_workspace_and_project()
    return 0


def ensure_workspace_and_project() -> None:
    """Create-or-verify the recorded Determined workspace/project
    (owner decision, recorded in the Apollo execution-authority wiki
    page): workspace WORKSPACE_NAME, project PROJECT_NAME. `det`
    returns nonzero for an already-existing name; that outcome is
    accepted, any other failure surfaces.

    Not routed through :func:`elgs.determined.check_invocation`: this
    function's nonzero-exit handling is tri-state (created / already
    exists / real failure), which `check_invocation`'s blanket
    any-nonzero-is-fatal contract cannot express. It still goes through
    :func:`elgs.determined.resolve_det_command` and
    :func:`elgs.determined.run_det` for the actual process invocation, so
    there is still exactly one place in the repo that decides how to find
    and launch `det`.
    """
    command = determined.resolve_det_command()

    # POSITIVE verification, not error-string matching. The previous
    # revision treated a nonzero `create` as "already exists" only when
    # the output contained the literal "already exists"; the master
    # actually answers a duplicate workspace with "avoid names equal to
    # other workspaces (case-insensitive)", so preflight FAILED on the
    # workspace M0 itself created. Asking whether the thing exists is
    # both correct and immune to the master's phrasing.
    probe = determined.run_det(command, ["workspace", "describe", WORKSPACE_NAME])
    if probe.returncode == 0:
        print(f"exists: workspace {WORKSPACE_NAME}")
    else:
        created = determined.run_det(command, ["workspace", "create", WORKSPACE_NAME])
        if created.returncode != 0:
            raise ContractError(
                f"det workspace create {WORKSPACE_NAME} failed: "
                f"{created.stderr.strip() or created.stdout.strip()}"
            )
        print(f"created: workspace {WORKSPACE_NAME}")

    listing = determined.run_det(
        command, ["project", "list", WORKSPACE_NAME, "--json"]
    )
    existing: set[str] = set()
    if listing.returncode == 0:
        try:
            existing = {
                str(entry.get("name")) for entry in json.loads(listing.stdout or "[]")
            }
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"could not parse `det project list {WORKSPACE_NAME} --json`: {exc}"
            ) from exc
    if PROJECT_NAME in existing:
        print(f"exists: project {WORKSPACE_NAME}/{PROJECT_NAME}")
        return
    created = determined.run_det(
        command, ["project", "create", WORKSPACE_NAME, PROJECT_NAME]
    )
    if created.returncode != 0:
        raise ContractError(
            f"det project create {WORKSPACE_NAME} {PROJECT_NAME} failed: "
            f"{created.stderr.strip() or created.stdout.strip()}"
        )
    print(f"created: project {WORKSPACE_NAME}/{PROJECT_NAME}")


def cmd_submit(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()

    # 1. det CLI on PATH + `det --version`.
    print(f"det CLI: {_det_version()}")

    commit = git_rev_parse(repo_root)
    branch = git_current_branch(repo_root)
    run_id = _build_run_id(cell=args.cell, seed=args.seed, commit=commit)
    run_dir = f"{args.runs_root.rstrip('/')}/{run_id}"
    config_relative = _relative_repo_path(repo_root, args.config)

    substitutions = {
        "NAME": args.cell,
        "POOL": args.pool,
        "IMAGE_REF": args.image_ref,
        "ENTRYPOINT_SCRIPT": args.entrypoint_script,
        "ENTRYPOINT_ARGS": _build_entrypoint_args(
            config_path=config_relative,
            run_dir=run_dir,
            extra_args=args.extra_arg,
            entrypoint_script=args.entrypoint_script,
        ),
        "RUN_DIR": run_dir,
        "MAX_TRAIN_ITERATIONS": str(args.max_train_iterations),
    }

    # 2. template renders.
    rendered_config = render_template(args.template, substitutions)

    # 3. execution closure -- including the entrypoint actually used, so a
    # dirty non-main.py entrypoint refuses the run instead of silently
    # executing its committed twin.
    execution_set = resolve_execution_set(
        repo_root, [args.config], entrypoint_script=args.entrypoint_script
    )
    closure = check_execution_closure(
        repo_root, execution_set, dirty_smoke=args.dirty_smoke,
        exploratory=args.exploratory,
    )

    # 4. claim (before materialization/submission -- race-safe duplicate guard).
    claim_path = claim_cell(
        args.claims_dir,
        args.cell,
        args.retry,
        payload={
            "pool": args.pool,
            "commit": commit,
            "run_dir": run_dir,
            "evidence_bearing": closure.evidence_bearing,
            "dirty_smoke": closure.dirty_smoke,
        },
    )

    with tempfile.TemporaryDirectory(prefix="elgs-submit-") as tmp:
        context_dir = Path(tmp) / "context"
        context_dir.mkdir()

        # 5. materialize.
        archive_sha256 = materialize_context(repo_root, commit, context_dir)

        # 6. manifest, written O_EXCL into the context; run-dir path
        # recorded. hash_root = the context: hashes describe the bytes
        # that travel, not the (possibly CRLF) worktree view.
        manifest = build_manifest(
            repo_root=repo_root,
            commit=commit,
            branch=branch,
            config_paths=[args.config],
            image_ref=args.image_ref,
            pool=args.pool,
            slots=1,
            seed=args.seed,
            run_dir=run_dir,
            wrapper_argv=sys.argv,
            evidence_bearing=closure.evidence_bearing,
            projected_gpu_hours=args.projected_gpu_hours,
            dataset_manifest_path=args.dataset_manifest,
            hash_root=context_dir,
            entrypoint_script=args.entrypoint_script,
        )
        manifest["archive_sha256"] = archive_sha256
        manifest["rendered_config_sha256"] = sha256(rendered_config.encode("utf-8")).hexdigest()
        manifest["claim_path"] = str(claim_path)
        manifest["dirty_outside_execution_set"] = list(closure.dirty_outside)
        atomic_write_json_immutable(context_dir / MANIFEST_FILENAME, manifest)

        if args.dry_run:
            print(json.dumps(manifest, indent=2, sort_keys=True))
            print(f"[dry-run] would submit with rendered config:\n{rendered_config}")
            return 0

        rendered_config_path = Path(tmp) / "det_exp_apollo.rendered.yaml"
        rendered_config_path.write_text(rendered_config, encoding="utf-8")

        result = subprocess.run(
            ["det", "e", "create", str(rendered_config_path), str(context_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ContractError(
                f"`det e create` failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        experiment_id = _parse_experiment_id(result.stdout)

        # Claim files are immutable; the experiment ID goes into a sibling file.
        experiment_claim_path = claim_path.parent / f"{claim_path.name}.experiment"
        atomic_write_json_immutable(
            experiment_claim_path,
            {"claim": str(claim_path), "experiment_id": experiment_id, "recorded_at_utc": _utc_now_stamp()},
        )

        append_ledger(
            args.ledger,
            {
                "event": "submitted",
                "cell_name": args.cell,
                "retry": args.retry,
                "experiment_id": experiment_id,
                "run_dir": run_dir,
                "commit": commit,
                "pool": args.pool,
                "evidence_bearing": closure.evidence_bearing,
                "projected_gpu_hours": args.projected_gpu_hours,
                "timestamp_utc": _utc_now_stamp(),
            },
        )
        print(f"Submitted {args.cell} as Determined experiment {experiment_id}")
        print(f"Run dir: {run_dir}")
        print(f"Ledger: {args.ledger}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    command = determined.resolve_det_command()
    invocation = determined.run_det(
        command, ["experiment", "describe", str(args.experiment_id), "--json"]
    )
    determined.check_invocation(invocation)
    print(invocation.stdout)
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    command = determined.resolve_det_command()
    det_args = ["experiment", "logs", str(args.experiment_id)]
    if args.follow:
        det_args.append("-f")
    # timeout_seconds=None: the original bare subprocess.run call here had
    # no timeout at all, and `-f` follows logs indefinitely -- preserve
    # that unbounded-wait behaviour rather than inheriting run_det's 120s
    # default (which would turn a normal `--follow` session into a
    # DetInvocationError after two minutes).
    invocation = determined.run_det(command, det_args, timeout_seconds=None)
    determined.check_invocation(invocation)
    print(invocation.stdout)
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    command = determined.resolve_det_command()
    invocation = determined.run_det(command, ["experiment", "kill", str(args.experiment_id)])
    determined.check_invocation(invocation)
    append_ledger(
        args.ledger,
        {
            "event": "cancelled",
            "experiment_id": int(args.experiment_id),
            "reason": args.reason or "",
            "timestamp_utc": _utc_now_stamp(),
        },
    )
    print(f"Cancelled experiment {args.experiment_id}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ContractError(f"run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    terminal_path = run_dir / "terminal.json"
    if terminal_path.exists():
        print(f"terminal.json already sealed: {terminal_path}")
        return 0

    experiment_id = args.experiment_id or manifest.get("experiment_id")
    if not experiment_id:
        raise ContractError(
            "no experiment ID available -- pass --experiment-id or ensure the manifest/claim records one"
        )
    command = determined.resolve_det_command()
    invocation = determined.run_det(
        command, ["experiment", "describe", str(experiment_id), "--json"]
    )
    determined.check_invocation(invocation)
    # elgs.determined.parse_experiment_state, not a local json.loads(...).get(...):
    # `det experiment describe --json` was VERIFIED (2026-08-14, live cluster) to
    # return a JSON ARRAY containing one object, not a bare object -- the
    # original `described.get("experiment", {})` here would raise
    # AttributeError ('list' object has no attribute 'get') against the real
    # CLI. Routing through the shared, live-verified parser fixes that
    # latent bug as a direct consequence of there being one det-invocation
    # and one state-parsing path.
    state = determined.parse_experiment_state(invocation.stdout)

    inventory = build_inventory(run_dir, exclude=(MANIFEST_FILENAME, "terminal.json"))
    payload = {
        "schema": TERMINAL_SCHEMA,
        "experiment_id": int(experiment_id),
        "state": state,
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_inventory": inventory,
        "audited_at_utc": _utc_now_stamp(),
    }
    write_terminal_last(terminal_path, payload, required_paths=[manifest_path])
    print(f"Wrote {terminal_path}")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def _add_common_submission_args(parser: argparse.ArgumentParser, *, require_projected_hours: bool) -> None:
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="ADAGS repo checkout root")
    parser.add_argument("--cell", required=True, help="cell/experiment name (ledger + claim key)")
    parser.add_argument("--pool", required=True, choices=("hopper", "dgx"), help="Determined resource pool")
    parser.add_argument("--image-ref", required=True, help="image reference; @sha256:... for evidence-bearing runs")
    parser.add_argument("--config", required=True, help="run config path passed to main.py --config")
    parser.add_argument("--seed", required=True, type=int, help="run seed")
    parser.add_argument("--retry", type=int, default=0, help="retry counter for the claim key")
    parser.add_argument(
        "--runs-root",
        default=os.environ.get("ADAGS_RUNS_ROOT", "/apollo/users/sri/proj_adags/runs/elgs"),
        help="run output root on shared Apollo storage",
    )
    parser.add_argument(
        "--claims-dir",
        default=os.environ.get(
            "ADAGS_ELGS_CLAIMS_DIR",
            str(Path(os.environ.get("ADAGS_PROJECT_ROOT", str(REPO_ROOT))) / "agent-control" / "elgs-apollo" / "claims"),
        ),
        help="O_EXCL claim directory on shared Apollo storage",
    )
    parser.add_argument(
        "--ledger",
        default=os.environ.get(
            "ADAGS_ELGS_LEDGER",
            str(Path(os.environ.get("ADAGS_PROJECT_ROOT", str(REPO_ROOT))) / "agent-control" / "elgs-apollo" / "experiment-ledger.jsonl"),
        ),
        help="append-only experiment-ledger.jsonl path",
    )
    parser.add_argument(
        "--template", default=str(REPO_ROOT / "det_exp_apollo.yaml"), help="det_exp_apollo.yaml template path"
    )
    parser.add_argument("--dataset-manifest", default=None, help="dataset manifest path (optional for a smoke run)")
    parser.add_argument(
        "--dirty-smoke",
        action="store_true",
        help="permit a dirty/untracked execution-relevant tree; stamps evidence_bearing=false",
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help=(
            "declare this run EXPLORATORY: stamps evidence_bearing=false even "
            "on a clean tree. Required for any run consuming smoke-tier "
            "constants, and the only way to say 'not claim-bearing' about "
            "reproducible code"
        ),
    )
    parser.add_argument(
        "--projected-gpu-hours",
        type=float,
        required=require_projected_hours,
        default=0.0,
        help="projected GPU-hours for this run (recorded in the manifest; guard against the M0/M1 ceilings)",
    )
    parser.add_argument(
        "--entrypoint-script",
        default="main.py",
        help=(
            "script the experiment execs (allowlist: "
            f"{', '.join(ALLOWED_ENTRYPOINT_SCRIPTS)}); non-main.py entrypoints "
            "take their whole CLI from --extra-arg tokens. Also enters the "
            "execution-closure set, so a dirty entrypoint refuses the run"
        ),
    )
    parser.add_argument(
        "--max-train-iterations",
        type=int,
        default=DEFAULT_MAX_TRAIN_ITERATIONS,
        help=(
            "value for main.py's ADAGS_MAX_ITERATIONS train-iteration guard. "
            "The default reproduces main.py's own unset-environment behaviour "
            "exactly; raise it only for a deliberately long run (e.g. a "
            "checkpoint continuation past the default ceiling)"
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="submit_apollo.py",
        description="EL-GS Determined submission wrapper for Apollo (draft; see S10.2).",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    preflight = subparsers.add_parser("preflight", help="smallest-sufficient pre-submission checks")
    _add_common_submission_args(preflight, require_projected_hours=False)
    preflight.set_defaults(func=cmd_preflight)

    submit = subparsers.add_parser("submit", help="submit one EL-GS Determined experiment")
    _add_common_submission_args(submit, require_projected_hours=True)
    submit.add_argument("--extra-arg", action="append", default=[], help="extra entrypoint CLI token (repeatable)")
    submit.add_argument("--dry-run", action="store_true", help="stop before `det e create`; print the manifest")
    submit.set_defaults(func=cmd_submit)

    status = subparsers.add_parser("status", help="`det experiment describe --json`")
    status.add_argument("--experiment-id", required=True)
    status.set_defaults(func=cmd_status)

    logs = subparsers.add_parser("logs", help="`det experiment logs`")
    logs.add_argument("--experiment-id", required=True)
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(func=cmd_logs)

    cancel = subparsers.add_parser("cancel", help="`det experiment kill` + ledger `cancelled` event")
    cancel.add_argument("--experiment-id", required=True)
    cancel.add_argument("--reason", default=None)
    cancel.add_argument(
        "--ledger",
        default=os.environ.get(
            "ADAGS_ELGS_LEDGER",
            str(Path(os.environ.get("ADAGS_PROJECT_ROOT", str(REPO_ROOT))) / "agent-control" / "elgs-apollo" / "experiment-ledger.jsonl"),
        ),
    )
    cancel.set_defaults(func=cmd_cancel)

    audit = subparsers.add_parser("audit", help="re-audit a run dir; seal terminal.json if absent")
    audit.add_argument("--run-dir", required=True)
    audit.add_argument("--experiment-id", default=None)
    audit.set_defaults(func=cmd_audit)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "ClosureReport",
    "append_ledger",
    "build_manifest",
    "canonical_config_hash",
    "check_execution_closure",
    "claim_cell",
    "git_current_branch",
    "git_rev_parse",
    "materialize_context",
    "render_template",
    "resolve_execution_set",
    "runtime_assertions",
]


if __name__ == "__main__":
    raise SystemExit(main())
