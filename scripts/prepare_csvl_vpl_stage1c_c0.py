#!/usr/bin/env python3
"""Freeze exact source, command, config, and input bindings for Stage 1C Gate C0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.canonical import sha256_file
from depth_visibility.interval_admission import STAGE1B_FREEZE_COMMIT, validate_config


DEFAULT_MATRIX = REPO_ROOT / "configs/depth_visibility/phase9_csvl_vpl_stage1c_c0_v1.json"
DEFAULT_RUN_ID = "P9-VPL-S1C-C0-CUT-S20260726"


def expand(value: str) -> str:
    result = os.path.expandvars(value)
    if "$" in result:
        raise RuntimeError(f"unresolved environment variable: {value}")
    return result


def git(*arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def write_new_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to replace immutable Stage-1C C0 authority: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def verify_reference(reference: dict[str, Any]) -> None:
    path = Path(expand(str(reference["path"]))).resolve()
    if not path.is_file():
        raise RuntimeError(f"bound input is missing: {path}")
    if sha256_file(path) != reference["sha256"]:
        raise RuntimeError(f"bound input hash mismatch: {path}")
    if reference["schema"] == "phase9-terminal-manifest-v1":
        terminal = json.loads(path.read_text(encoding="utf-8"))
        if terminal.get("run_id") != reference["producer_run_id"]:
            raise RuntimeError(f"bound terminal producer mismatch: {path}")
        if terminal.get("status") != "completed" or terminal.get("exit_code") != 0:
            raise RuntimeError(f"bound terminal is not completed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    matrix_path = Path(args.matrix).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matches = [value for value in matrix["runs"] if value["run_id"] == args.run_id]
    if len(matches) != 1:
        raise RuntimeError(f"run entry count is {len(matches)}")
    entry = matches[0]
    if entry["action"] != "admit-stage1c-interval":
        raise RuntimeError("Stage-1C C0 authority received the wrong action")
    commit = git("rev-parse", "HEAD")
    if subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", STAGE1B_FREEZE_COMMIT, commit],
        check=False,
    ).returncode != 0:
        raise RuntimeError("current commit does not descend from the frozen Stage-1B commit")
    for reference in [*entry["input_artifacts"], *entry["external_inputs"]]:
        verify_reference(reference)
    config_path = (REPO_ROOT / entry["configuration"]["base_config_path"]).resolve()
    validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    if sha256_file(config_path) != entry["configuration"]["base_config_sha256"]:
        raise RuntimeError("Stage-1C C0 config hash differs from the run matrix")

    output_root = Path(expand(entry["storage"]["output_root"])).resolve()
    for output in entry["expected_outputs"]:
        path = Path(expand(output["path"]))
        if path.exists():
            raise RuntimeError(f"immutable Stage-1C C0 output already exists: {path}")
    authority_root = output_root.parent.parent / "authority"
    implementation_path = authority_root / "exact-worktree-implementation.json"
    command_path = authority_root / "command-registry.json"
    execution_path = output_root / "resolved-execution.json"
    if any(path.exists() for path in (implementation_path, command_path, execution_path)):
        raise RuntimeError("Stage-1C C0 authority already exists; inspect it instead of replacing it")

    source_paths = sorted((REPO_ROOT / "depth_visibility").glob("*.py")) + [
        REPO_ROOT / "scripts/run_phase9_depth_visibility.py",
        REPO_ROOT / "scripts/run_phase9_depth_visibility_job.sh",
        REPO_ROOT / "scripts/submit_phase9_depth_visibility.sh",
        REPO_ROOT / "scripts/prepare_csvl_vpl_stage1c_c0.py",
        config_path,
        matrix_path,
        REPO_ROOT / "tests/test_depth_visibility_stage1.py",
        REPO_ROOT / "tests/test_depth_visibility_stage1b.py",
        REPO_ROOT / "tests/test_depth_visibility_stage1c.py",
        REPO_ROOT / "tests/test_depth_visibility_canonical.py",
        REPO_ROOT / "research-wiki/operations/phase9-csvl-vpl-stage1b-result.md",
        REPO_ROOT / "research-wiki/operations/phase9-csvl-vpl-stage1c-authority.md",
    ]
    source_files = []
    seen = set()
    for path in source_paths:
        resolved = path.resolve()
        relative = resolved.relative_to(REPO_ROOT).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        if not resolved.is_file():
            raise RuntimeError(f"Stage-1C C0 source file is missing: {relative}")
        source_files.append({
            "path": relative,
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        })
    implementation = {
        "schema_version": "phase9-stage1c-c0-exact-worktree-implementation-v1",
        "run_id": args.run_id,
        "binding_mode": "exact_worktree_files_no_commit_claim",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": commit,
        "stage1b_freeze_commit": STAGE1B_FREEZE_COMMIT,
        "worktree_clean_claim": False,
        "worktree_status_at_binding": git("status", "--short"),
        "source_files": source_files,
        "matrix_path": matrix_path.relative_to(REPO_ROOT).as_posix(),
        "matrix_sha256": sha256_file(matrix_path),
        "interpretation": "Stage-1C C0 exact bytes are bound without claiming a clean worktree; unrelated pre-existing changes are preserved outside the source binding.",
    }
    write_new_json(implementation_path, implementation)
    implementation_sha = sha256_file(implementation_path)

    argv = [expand(value) if "$" in value else value for value in entry["command"]["argv_template"]]
    argv_sha = hashlib.sha256(json.dumps(argv, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    command_registry = {
        "schema_version": "phase9-stage1c-c0-command-registry-v1",
        "run_id": args.run_id,
        "stage1b_freeze_commit": STAGE1B_FREEZE_COMMIT,
        "implementation_manifest_sha256": implementation_sha,
        "matrix_sha256": sha256_file(matrix_path),
        "launcher_sha256": sha256_file(REPO_ROOT / entry["command"]["launcher_path"]),
        "job_wrapper_sha256": sha256_file(REPO_ROOT / "scripts/run_phase9_depth_visibility_job.sh"),
        "argv": argv,
        "argv_sha256": argv_sha,
        "scheduler": entry["scheduler"],
    }
    write_new_json(command_path, command_registry)
    command_sha = sha256_file(command_path)

    merged_identity = {
        "base_config_sha256": sha256_file(config_path),
        "derived_config_sha256": None,
        "training": entry.get("training"),
        "seeds": entry.get("seeds"),
    }
    merged_sha = hashlib.sha256(json.dumps(merged_identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    configuration = dict(entry["configuration"])
    configuration.update({
        "derived_config_sha256": None,
        "resolved_merged_config_sha256": merged_sha,
        "status": "exact_worktree_binding_inputs_resolved",
    })
    execution = {
        "schema_version": "phase9-resolved-execution-v1",
        "run_id": args.run_id,
        "action": entry["action"],
        "resolved_argv": argv,
        "resolved_argv_sha256": argv_sha,
        "launcher_sha256": sha256_file(REPO_ROOT / entry["command"]["launcher_path"]),
        "implementation": {
            "commit": commit,
            "implementation_manifest_path": str(implementation_path),
            "implementation_manifest_sha256": implementation_sha,
            "command_registry_path": str(command_path),
            "command_registry_sha256": command_sha,
        },
        "configuration": configuration,
        "expected_outputs": entry["expected_outputs"],
        "input_artifacts": entry["input_artifacts"],
        "external_inputs": entry["external_inputs"],
        "scheduler": entry["scheduler"],
        "conditional": False,
        "registered_predicate": entry["pre_observation_promotion_predicate"],
        "input_binding_status": "resolved_exact_all_inputs",
    }
    write_new_json(execution_path, execution)
    print(json.dumps({
        "run_id": args.run_id,
        "stage1b_freeze_commit": STAGE1B_FREEZE_COMMIT,
        "execution_manifest": str(execution_path),
        "execution_manifest_sha256": sha256_file(execution_path),
        "implementation_manifest": str(implementation_path),
        "implementation_manifest_sha256": implementation_sha,
        "source_file_count": len(source_files),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
