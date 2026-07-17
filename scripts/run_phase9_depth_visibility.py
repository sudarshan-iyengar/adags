#!/usr/bin/env python3
"""Registered Phase 9 execution entrypoint.

Every substantial action is intended for Slurm. The entrypoint validates the
registered run/action binding, writes outputs beneath the registered run root,
and seals terminal.json last. Unsupported or incomplete scientific actions fail
closed instead of emitting placeholder success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.annotation import (
    build_empty_annotation_packet,
    load_json,
    validate_empty_annotation_packet,
)
from depth_visibility.baselines import validate_baseline_registry
from depth_visibility.canonical import sha256_file
from depth_visibility.da3_adapter import (
    load_da3,
    run_analytic_conformance,
    run_two_group_conformance,
    verify_model_authority,
)
from depth_visibility.evaluator import (
    artifact_reference,
    load_run_entry,
    terminal_manifest,
    validate_execution_manifest,
    resolved_python_argv,
    write_json_atomic,
)
from depth_visibility.errors import ContractError, ProvenanceError
from depth_visibility.fixtures import two_plane_track_pixels
from depth_visibility.groups import enumerate_anchor_groups
from depth_visibility.ledger import build_target_frame_ledger
from depth_visibility.n3v import compute_r_scene, load_scene_index, validate_split_binding


DEFAULT_MATRIX = REPO_ROOT / "configs/depth_visibility/phase9_run_matrix_v1.json"
DEFAULT_CONFIG = REPO_ROOT / "configs/depth_visibility/csvl_isr_v1.json"
DEFAULT_WINDOWS = REPO_ROOT / "configs/depth_visibility/annotation_windows_v1.json"
DEFAULT_BASELINES = REPO_ROOT / "configs/depth_visibility/r031_baselines_v1.json"
DEFAULT_SCHEMA_BUNDLE = REPO_ROOT / "configs/depth_visibility/phase9_schema_bundle_v1.json"
CYCLE_RELATIVE = Path("runs/phase9-depth-visibility-capacity/cycle-v1")


def _expand(value: str) -> Path:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ProvenanceError(f"unresolved environment variable in path: {value}")
    return Path(expanded)


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _expected_path(entry: dict[str, Any], schema: str) -> Path:
    matches = [item for item in entry["expected_outputs"] if item["schema"] == schema]
    if len(matches) != 1:
        raise ContractError(f"{entry['run_id']} expects {len(matches)} outputs with schema {schema}")
    return _expand(matches[0]["path"])


def _output_root(entry: dict[str, Any]) -> Path:
    return _expand(entry["storage"]["output_root"])


def _scientific_file_ref(path: Path, schema: str, run_id: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "schema": schema,
        "producer_run_id": run_id,
        "sha256": sha256_file(path),
    }


def action_static(entry: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import unittest

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_depth_visibility*.py",
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise ContractError(
            f"static suite failed: failures={len(result.failures)} errors={len(result.errors)}"
        )
    payload = {
        "test_count": int(result.testsRun),
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "suite": "tests/test_depth_visibility*.py",
    }
    return [], payload


def action_synthetic(entry: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conformance = dict(run_analytic_conformance())
    layers, witnesses = two_plane_track_pixels()
    ledger = build_target_frame_ledger(
        scene="analytic_two_plane",
        frame=0,
        scored_target="cam00",
        track_pixels=layers,
        visible_witnesses=witnesses,
        provenance={
            "physical_ancestry": ["cam01", "cam02", "cam03", "cam04", "cam05", "cam06", "cam07"],
            "dependencies": [],
        },
    )
    states = {
        layer["track_id"]: layer["state"]
        for pixel in ledger["regions"]
        for layer in [pixel]
    }
    if "visible" not in states.values() or "occluded" not in states.values():
        raise ContractError("two-plane synthetic fixture did not recover both depth states")
    report_path = _output_root(entry) / "synthetic-report.json"
    report = {
        "schema_version": "phase9-synthetic-report-v1",
        "run_id": entry["run_id"],
        "analytic_camera": conformance,
        "ledger_id": ledger["ledger_id"],
        "states": states,
    }
    write_json_atomic(report_path, report)
    return [_scientific_file_ref(report_path, "phase9-synthetic-report-v1", entry["run_id"])], report


def _processed_size(native_width: int, native_height: int, target: int = 504, patch: int = 14) -> tuple[int, int]:
    # Mirror the pinned upper-bound resize dimensions without importing DA3.
    scale = target / float(max(native_width, native_height))
    width = max(1, int(round(native_width * scale)))
    height = max(1, int(round(native_height * scale)))

    def nearest_multiple(value: int) -> int:
        down = (value // patch) * patch
        up = down + patch
        return max(1, up if abs(up - value) <= abs(value - down) else down)

    return nearest_multiple(width), nearest_multiple(height)


def _pinned_da3_processed_intrinsics(
    intrinsic: Any,
    native_width: int,
    native_height: int,
    processed_width: int,
    processed_height: int,
) -> np.ndarray:
    """Independently mirror the pinned DA3 InputProcessor row scaling."""

    if min(native_width, native_height, processed_width, processed_height) <= 0:
        raise ContractError("DA3 intrinsic resize dimensions must be positive")
    result = np.asarray(intrinsic, dtype=np.float64).copy()
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ContractError("DA3 intrinsic authority must be a finite 3x3 matrix")
    result[0, :] *= processed_width / float(native_width)
    result[1, :] *= processed_height / float(native_height)
    return result


def _select_conformance_groups(records: list[Any], r_scene: float, config: dict[str, Any]) -> tuple[str, tuple[tuple[str, ...], ...]]:
    # Apply the preregistered all-anchor then repeated-lower-anchor rule.
    by_camera = {record.camera_id: record for record in records}
    if len(by_camera) != len(records) or "cam00" in by_camera:
        raise ProvenanceError("A03 requires one frame-0 record per training camera and no cam00")
    grouping = config["grouping"]
    groups = enumerate_anchor_groups(
        by_camera,
        r_scene,
        maximum_cameras=int(grouping["maximum_cameras"]),
        maximum_optical_axis_angle_degrees=float(grouping["maximum_optical_axis_angle_degrees"]),
        minimum_center_distance_rscene=float(grouping["minimum_center_distance_rscene"]),
        minimum_second_singular_value_rscene=float(grouping["minimum_second_singular_value_rscene"]),
    )
    unique = tuple(sorted(set(groups)))
    eligible = [
        camera_id
        for camera_id in sorted(by_camera)
        if sum(camera_id in group for group in unique) >= 2
    ]
    if not eligible:
        raise ProvenanceError("no cut frame-0 training camera appears in two valid groups")
    anchor_camera_id = eligible[0]
    selected = tuple(group for group in unique if anchor_camera_id in group)[:2]
    if len(selected) != 2:
        raise ProvenanceError("selected A03 anchor lacks two distinct valid groups")
    return anchor_camera_id, selected


def _matrix_output_path(matrix_path: Path, run_id: str, schema: str) -> Path:
    producer = load_run_entry(matrix_path, run_id)
    return _expected_path(producer, schema)


def _seed_conformance(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_da3_weight(args: argparse.Namespace, execution: dict[str, Any] | None) -> Path:
    if args.input:
        return Path(args.input).resolve()
    environment = os.environ.get("PHASE9_DA3_WEIGHT_PATH")
    if environment:
        return Path(environment).resolve()
    if execution is not None:
        for item in execution.get("external_inputs", []) + execution.get("inputs", []):
            if item.get("role") == "da3_weight" and item.get("path"):
                return _expand(item["path"]).resolve()
    raise ProvenanceError("DA3 weight path must be supplied by --input, PHASE9_DA3_WEIGHT_PATH, or the resolved execution manifest")


def _bound_a02_authority(
    execution: dict[str, Any] | None,
    matrix_path: Path,
) -> tuple[Path, dict[str, Any], str]:
    """Load DA3 authority only through the exact successful A02 terminal binding."""

    if execution is None:
        raise ProvenanceError("A03 requires a resolved execution manifest")
    matches = [
        item for item in execution.get("input_artifacts", [])
        if item.get("producer_run_id") == "P9-A02-DA3-WEIGHT-SHA"
        and item.get("schema") == "phase9-terminal-manifest-v1"
    ]
    if len(matches) != 1 or not str(matches[0].get("status", "")).startswith("resolved_exact"):
        raise ProvenanceError("A03 lacks one exact resolved A02 terminal input")
    terminal_ref = matches[0]
    terminal_path = _expand(str(terminal_ref["path"]))
    expected_terminal_sha = terminal_ref.get("sha256")
    if not terminal_path.is_file() or sha256_file(terminal_path) != expected_terminal_sha:
        raise ProvenanceError("A02 terminal bytes do not match the resolved A03 input")
    terminal = _json(terminal_path)
    if (
        terminal.get("schema_version") != "phase9-terminal-manifest-v1"
        or terminal.get("run_id") != "P9-A02-DA3-WEIGHT-SHA"
        or terminal.get("action") != "hash-da3"
        or terminal.get("status") != "completed"
        or terminal.get("exit_code") != 0
    ):
        raise ProvenanceError("A02 terminal is not a successful hash-da3 authority")
    authority_path = _matrix_output_path(
        matrix_path, "P9-A02-DA3-WEIGHT-SHA", "phase9-da3-authority-v1"
    ).resolve()
    produced = [
        item for item in terminal.get("produced_artifacts", [])
        if item.get("schema") == "phase9-da3-authority-v1"
        and item.get("producer_run_id") == "P9-A02-DA3-WEIGHT-SHA"
        and Path(str(item.get("path", ""))).resolve() == authority_path
    ]
    if len(produced) != 1 or not authority_path.is_file():
        raise ProvenanceError("A02 terminal does not bind exactly one DA3 authority artifact")
    authority_sha = sha256_file(authority_path)
    if produced[0].get("sha256") != authority_sha:
        raise ProvenanceError("DA3 authority bytes do not match the A02 terminal")
    return authority_path, _json(authority_path), str(expected_terminal_sha)


def action_hash_da3(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    weight = _find_da3_weight(args, execution)
    expected_bytes = int(_json(DEFAULT_CONFIG)["da3"]["weight_expected_bytes"])
    if not weight.is_file() or weight.stat().st_size != expected_bytes:
        raise ProvenanceError(f"DA3 weight size mismatch: expected {expected_bytes}, got {weight.stat().st_size if weight.exists() else 'missing'}")
    digest = sha256_file(weight)
    authority_path = _expected_path(entry, "phase9-da3-authority-v1")
    authority = {
        "schema_version": "phase9-da3-authority-v1",
        "run_id": entry["run_id"],
        "model_id": _json(DEFAULT_CONFIG)["da3"]["model_id"],
        "weight_bytes": expected_bytes,
        "weight_sha256": digest,
        "path_role": "read_only_model_weight",
        "da3_checkout_commit": _json(DEFAULT_CONFIG)["da3"]["checkout_commit"],
    }
    write_json_atomic(authority_path, authority)
    return [_scientific_file_ref(authority_path, "phase9-da3-authority-v1", entry["run_id"])], {"weight_bytes": expected_bytes, "weight_sha256": digest}


def action_da3_conformance(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Run the exact analytic plus two-group cut frame-0 A03 fixture.
    config = _json(DEFAULT_CONFIG)
    expected_scene = config["data"]["development_scene"]
    if args.scene != expected_scene or expected_scene != "cut_roasted_beef":
        raise ProvenanceError(f"A03 scene must be {expected_scene}")
    project_root = _expand("$WORK/proj_adags")
    data_root = project_root / "data/n3v"
    da3_checkout = Path(
        os.environ.get("PHASE9_DA3_REPO", str(project_root / "repo/depth-anything-3"))
    ).resolve()
    model_dir = Path(
        os.environ.get(
            "PHASE9_DA3_MODEL_DIR",
            str(project_root / "models/depth-anything/DA3NESTED-GIANT-LARGE-1.1"),
        )
    ).resolve()
    index = load_scene_index(
        data_root / expected_scene,
        scene=expected_scene,
        expose_test_images=False,
        hash_train_images=False,
        timestamp_tolerance_seconds=float(config["camera"]["timestamp_tolerance_seconds"]),
    )
    split_manifest = _json(REPO_ROOT / config["data"]["split_manifest"])
    split_binding_raw = validate_split_binding(index, split_manifest)
    split_binding = {key: dict(value) for key, value in split_binding_raw.items()}
    frame_records = [record for record in index.split("train") if record.frame == 0]
    r_scene = compute_r_scene(index.split("train"))
    anchor_camera_id, groups = _select_conformance_groups(frame_records, r_scene, config)
    records_by_camera = {record.camera_id: record for record in frame_records}

    matrix_path = Path(args.matrix).resolve()
    authority_path, authority, a02_terminal_sha = _bound_a02_authority(
        execution, matrix_path
    )
    if (
        authority.get("schema_version") != "phase9-da3-authority-v1"
        or authority.get("run_id") != "P9-A02-DA3-WEIGHT-SHA"
        or authority.get("model_id") != config["da3"]["model_id"]
        or not isinstance(authority.get("weight_sha256"), str)
    ):
        raise ProvenanceError("A02 DA3 authority is missing or incompatible")
    model_authority = verify_model_authority(
        model_dir,
        expected_weight_sha256=authority["weight_sha256"],
        hash_weights=True,
    )
    _seed_conformance(int(entry["seeds"]["training"]))
    model = load_da3(da3_checkout, model_dir, device="cuda")

    group_inputs = []
    for group in groups:
        records = [records_by_camera[camera_id] for camera_id in group]
        if any(record.image_path is None or record.camera_id == "cam00" for record in records):
            raise ProvenanceError("A03 group lacks training RGB or contains cam00")
        expected_k = []
        for record in records:
            processed_width, processed_height = _processed_size(record.width, record.height)
            expected_k.append(
                _pinned_da3_processed_intrinsics(
                    record.K,
                    record.width,
                    record.height,
                    processed_width,
                    processed_height,
                )
            )
        group_inputs.append(
            {
                "member_camera_ids": list(group),
                "images": [str(record.image_path) for record in records],
                "extrinsics_w2c": np.stack([record.w2c_opencv for record in records]),
                "intrinsics": np.stack([record.K for record in records]),
                "expected_processed_intrinsics": np.stack(expected_k),
            }
        )
    real = run_two_group_conformance(
        model,
        group_inputs,
        anchor_camera_id=anchor_camera_id,
        repeat_atol=float(config["da3"]["conformance_repeat_atol"]),
        repeat_rtol=float(config["da3"]["conformance_repeat_rtol"]),
    )
    payload = {
        "schema_version": "phase9-da3-conformance-v1",
        "scene": expected_scene,
        "frame": 0,
        "analytic": dict(run_analytic_conformance()),
        "real": real,
        "r_scene": r_scene,
        "split_binding": split_binding,
        "a02_authority_path": str(authority_path),
        "a02_authority_sha256": sha256_file(authority_path),
        "a02_terminal_sha256": a02_terminal_sha,
        "model_authority": model_authority,
        "cam00_rgb_opened": False,
    }
    return [], payload


def action_annotation_packet(entry: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(DEFAULT_WINDOWS)
    packet = build_empty_annotation_packet(
        manifest,
        manifest_path=str(DEFAULT_WINDOWS.relative_to(REPO_ROOT)),
    )
    validate_empty_annotation_packet(packet)
    packet_path = _expected_path(entry, "phase9-annotation-packet-v1")
    separation_path = _expected_path(entry, "phase9-r009-separation-proof-v1")
    write_json_atomic(packet_path, packet)
    separation = {
        "schema_version": "phase9-r009-separation-proof-v1",
        "run_id": entry["run_id"],
        "source_annotation_manifest_sha256": sha256_file(DEFAULT_WINDOWS),
        "source_r009_sha256": manifest["r009_exclusion"]["source_sha256"],
        "margin_frames_each_side": manifest["r009_exclusion"]["margin_frames_each_side"],
        "all_candidates_verified_disjoint": True,
        "window_count": len(manifest["windows"]),
    }
    write_json_atomic(separation_path, separation)
    refs = [
        _scientific_file_ref(packet_path, "phase9-annotation-packet-v1", entry["run_id"]),
        _scientific_file_ref(separation_path, "phase9-r009-separation-proof-v1", entry["run_id"]),
    ]
    return refs, {"human_fields_empty": True, "window_count": 54, "frame_review_rows": 594}


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _validate_runtime_execution(
    entry: dict[str, Any],
    execution: dict[str, Any],
    execution_path: Path,
) -> None:
    """Verify current bytes and argv against the immutable I01 execution binding."""

    argv = resolved_python_argv(
        execution,
        run_id=entry["run_id"],
        action=entry["action"],
        launcher_path=entry["command"]["launcher_path"],
        execution_manifest_path=str(execution_path),
    )
    if tuple(sys.argv) != argv[1:]:
        raise ProvenanceError("current process argv differs from resolved execution argv")
    launcher = (REPO_ROOT / entry["command"]["launcher_path"]).resolve()
    try:
        launcher.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ProvenanceError("registered launcher escapes repository root") from exc
    if sha256_file(launcher) != execution["launcher_sha256"]:
        raise ProvenanceError("current launcher bytes differ from I01 binding")
    implementation = execution["implementation"]
    if _git("rev-parse", "HEAD") != implementation.get("commit"):
        raise ProvenanceError("current Git commit differs from I01 binding")
    for path_key, sha_key, label in (
        ("implementation_manifest_path", "implementation_manifest_sha256", "implementation manifest"),
        ("command_registry_path", "command_registry_sha256", "command registry"),
    ):
        path = _expand(str(implementation.get(path_key, "")))
        if not path.is_file() or sha256_file(path) != implementation.get(sha_key):
            raise ProvenanceError(f"current {label} bytes differ from I01 binding")
    current_config = _resolved_config_binding(entry)
    for key in ("base_config_sha256", "derived_config_sha256", "resolved_merged_config_sha256"):
        if current_config.get(key) != execution["configuration"].get(key):
            raise ProvenanceError(f"current configuration differs from I01 binding: {key}")
    for section_name in ("input_artifacts", "external_inputs"):
        for reference in execution.get(section_name, []):
            path = _expand(str(reference.get("path", "")))
            if not path.is_file() or sha256_file(path) != reference.get("sha256"):
                raise ProvenanceError(
                    f"current {section_name} bytes differ from resolved binding: {path}"
                )


def _tracked_phase9_paths() -> list[Path]:
    prefixes = (
        "depth_visibility/",
        "tests/test_depth_visibility",
        "configs/depth_visibility/",
        "research-wiki/operations/phase9-",
        "research-wiki/objectives/depth-visibility-capacity-v1.md",
        "scripts/run_phase9_depth_visibility",
        "scripts/submit_phase9_depth_visibility.sh",
        "scripts/build_phase9_run_matrix.py",
    )
    paths = []
    for raw in _git("ls-files").splitlines():
        if any(raw.startswith(prefix) for prefix in prefixes):
            paths.append(REPO_ROOT / raw)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ProvenanceError(f"tracked Phase 9 source is missing: {missing}")
    return sorted(paths)



def _resolved_config_binding(registered: dict[str, Any]) -> dict[str, Any]:
    binding = dict(registered["configuration"])
    base_path = REPO_ROOT / binding["base_config_path"]
    base_sha = sha256_file(base_path)
    if base_sha != binding["base_config_sha256"]:
        raise ProvenanceError(f"base config hash mismatch for {registered['run_id']}")
    derived_path = binding.get("derived_config_path")
    derived_sha = None
    if derived_path is not None:
        derived_file = REPO_ROOT / derived_path
        derived_sha = sha256_file(derived_file)
        if derived_sha != binding.get("derived_config_sha256"):
            raise ProvenanceError(f"derived config hash mismatch for {registered['run_id']}")
    merged_identity = {
        "base_config_sha256": base_sha,
        "derived_config_sha256": derived_sha,
        "training": registered.get("training"),
        "seeds": registered.get("seeds"),
    }
    binding.update(
        {
            "base_config_sha256": base_sha,
            "derived_config_sha256": derived_sha,
            "resolved_merged_config_sha256": hashlib.sha256(
                json.dumps(merged_identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest(),
            "status": "static_bindings_resolved_inputs_pending_producers",
        }
    )
    return binding


def _write_execution_templates(
    matrix: dict[str, Any],
    *,
    implementation_path: Path,
    implementation_sha: str,
    command_registry_path: Path,
    command_registry_sha: str,
) -> int:
    """Write per-run immutable static bindings; producer hashes remain explicit."""

    count = 0
    for registered in matrix["runs"]:
        argv = [_expand(value).as_posix() if "$" in value else value for value in registered["command"]["argv_template"]]
        argv_sha = hashlib.sha256(
            json.dumps(argv, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        execution = {
            "schema_version": "phase9-resolved-execution-v1",
            "run_id": registered["run_id"],
            "action": registered["action"],
            "resolved_argv": argv,
            "resolved_argv_sha256": argv_sha,
            "launcher_sha256": sha256_file(REPO_ROOT / registered["command"]["launcher_path"]),
            "implementation": {
                "commit": _git("rev-parse", "HEAD"),
                "implementation_manifest_path": str(implementation_path),
                "implementation_manifest_sha256": implementation_sha,
                "command_registry_path": str(command_registry_path),
                "command_registry_sha256": command_registry_sha,
            },
            "configuration": _resolved_config_binding(registered),
            "expected_outputs": registered["expected_outputs"],
            "input_artifacts": registered["input_artifacts"],
            "external_inputs": registered["external_inputs"],
            "scheduler": registered["scheduler"],
            "conditional": registered["conditional"],
            "registered_predicate": registered["pre_observation_promotion_predicate"],
            "input_binding_status": ("resolved_exact_no_inputs" if not registered["input_artifacts"] and not registered["external_inputs"] else "resolve_exact_producer_hashes_before_submission"),
        }
        destination = _expand(registered["storage"]["output_root"]) / "resolved-execution.json"
        write_json_atomic(destination, execution)
        count += 1
    return count

def action_freeze_implementation(entry: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _git("status", "--porcelain"):
        raise ProvenanceError("I01 requires a clean tracked worktree")
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if head != upstream:
        raise ProvenanceError("I01 requires HEAD to equal the current upstream")
    tracked = [{"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)} for path in _tracked_phase9_paths()]
    matrix_sha = sha256_file(DEFAULT_MATRIX)
    config_sha = sha256_file(DEFAULT_CONFIG)
    schema_sha = sha256_file(DEFAULT_SCHEMA_BUNDLE)
    baseline_audit = validate_baseline_registry(_json(DEFAULT_BASELINES))
    implementation_path = _expected_path(entry, "phase9-implementation-freeze-v1")
    command_path = _expected_path(entry, "phase9-command-registry-v1")
    implementation = {
        "schema_version": "phase9-implementation-freeze-v1",
        "run_id": entry["run_id"],
        "branch": branch,
        "commit": head,
        "upstream_commit": upstream,
        "worktree_clean": True,
        "tracked_sources": tracked,
        "run_matrix_sha256": matrix_sha,
        "base_config_sha256": config_sha,
        "schema_bundle_sha256": schema_sha,
        "baseline_registry_sha256": sha256_file(DEFAULT_BASELINES),
        "baseline_registry_valid": baseline_audit["valid"],
        "environment_setup": {
            "path": "$WORK/proj_adags/exp_index/leonardo_env.sh",
            "sha256": sha256_file(_expand("$WORK/proj_adags/exp_index/leonardo_env.sh")),
        },
        "scheduler_authority": {
            "gpu_profile": "Leonardo boost: 4xA100 64GiB, 32 CPUs, node memory 514000 MiB, maximum one day",
            "gpu_memory_gib_required": 64,
        },
    }
    write_json_atomic(implementation_path, implementation)
    implementation_sha = sha256_file(implementation_path)

    matrix = _json(DEFAULT_MATRIX)
    commands = []
    for registered in matrix["runs"]:
        argv = [_expand(value).as_posix() if "$" in value else value for value in registered["command"]["argv_template"]]
        argv_sha = hashlib.sha256(json.dumps(argv, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
        commands.append(
            {
                "run_id": registered["run_id"],
                "action": registered["action"],
                "argv": argv,
                "argv_sha256": argv_sha,
                "launcher_sha256": sha256_file(REPO_ROOT / registered["command"]["launcher_path"]),
                "scheduler": registered["scheduler"],
            }
        )
    command_registry = {
        "schema_version": "phase9-command-registry-v1",
        "run_id": entry["run_id"],
        "implementation_freeze_sha256": implementation_sha,
        "run_matrix_sha256": matrix_sha,
        "commands": commands,
    }
    write_json_atomic(command_path, command_registry)
    resolved_execution_count = _write_execution_templates(
        matrix,
        implementation_path=implementation_path,
        implementation_sha=implementation_sha,
        command_registry_path=command_path,
        command_registry_sha=sha256_file(command_path),
    )
    refs = [
        _scientific_file_ref(implementation_path, "phase9-implementation-freeze-v1", entry["run_id"]),
        _scientific_file_ref(command_path, "phase9-command-registry-v1", entry["run_id"]),
    ]
    return refs, {"branch": branch, "commit": head, "resolved_command_count": len(commands), "resolved_execution_count": resolved_execution_count, "implementation_freeze_sha256": implementation_sha}


def action_decide(entry: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decision_path = _expected_path(entry, "phase9-decision-v1")
    payload = {
        "schema_version": "phase9-decision-v1",
        "run_id": entry["run_id"],
        "status": "not_evaluable",
        "reason": "decision handler requires stage-specific scored artifacts and must not infer them from terminal status",
        "registered_predicate": entry["pre_observation_promotion_predicate"],
    }
    write_json_atomic(decision_path, payload)
    return [_scientific_file_ref(decision_path, "phase9-decision-v1", entry["run_id"])], payload


def action_unimplemented(entry: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raise ContractError(
        f"registered action {entry['action']!r} has no admitted producer in this implementation; "
        "do not emit placeholder evidence"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--input")
    parser.add_argument("--scene")
    parser.add_argument("--allow-unresolved-local", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_path = Path(args.matrix).resolve()
    entry = load_run_entry(matrix_path, args.run_id)
    if entry["action"] != args.action:
        raise ProvenanceError(f"action mismatch: matrix={entry['action']} argv={args.action}")
    output_root = _output_root(entry)
    output_root.mkdir(parents=True, exist_ok=True)
    terminal_path = _expected_path(entry, "phase9-terminal-manifest-v1")
    execution_path = _expand(args.execution_manifest)
    execution = None
    execution_sha = "0" * 64
    if execution_path.is_file():
        execution = _json(execution_path)
        validate_execution_manifest(
            execution,
            run_id=args.run_id,
            action=args.action,
            run_entry=entry,
            require_resolved=args.action != "freeze-implementation",
        )
        _validate_runtime_execution(entry, execution, execution_path)
        execution_sha = sha256_file(execution_path)
    elif not (
        args.allow_unresolved_local
        and entry["scheduler"]["scheduler"] == "none"
        and args.action in {"static", "synthetic", "freeze-implementation"}
    ):
        raise ProvenanceError(f"resolved execution manifest is missing: {execution_path}")

    handlers = {
        "static": lambda: action_static(entry, args),
        "synthetic": lambda: action_synthetic(entry, args),
        "hash-da3": lambda: action_hash_da3(entry, args, execution),
        "da3-conformance": lambda: action_da3_conformance(entry, args, execution),
        "build-annotation-packet": lambda: action_annotation_packet(entry, args),
        "freeze-implementation": lambda: action_freeze_implementation(entry, args),
        "decide": lambda: action_decide(entry, args),
    }
    try:
        produced, payload = handlers.get(args.action, lambda: action_unimplemented(entry, args))()
        terminal = terminal_manifest(
            run_id=args.run_id,
            action=args.action,
            status="completed",
            exit_code=0,
            execution_manifest_sha256=execution_sha,
            produced_artifacts=produced,
            scientific_payload=payload,
        )
        write_json_atomic(terminal_path, terminal)
        print(json.dumps({"run_id": args.run_id, "status": "completed", "terminal": str(terminal_path)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"exception_type": type(exc).__name__, "message": str(exc)}
        terminal = terminal_manifest(
            run_id=args.run_id,
            action=args.action,
            status="failed",
            exit_code=1,
            execution_manifest_sha256=execution_sha,
            produced_artifacts=[],
            failure=failure,
        )
        try:
            write_json_atomic(terminal_path, terminal)
        except Exception:
            traceback.print_exc()
        print(json.dumps({"run_id": args.run_id, "status": "failed", "failure": failure}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
