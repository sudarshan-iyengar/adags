#!/usr/bin/env python3
"""Registered Phase 9 execution entrypoint.

Every substantial action is intended for Slurm. The entrypoint validates the
registered run/action binding, writes outputs beneath the registered run root,
and seals terminal.json last. Unsupported or incomplete scientific actions fail
closed instead of emitting placeholder success.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.annotation import (
    build_empty_annotation_packet,
    load_json,
    validate_empty_annotation_packet,
    validate_human_label_freeze,
)
from depth_visibility.artifacts import build_inventory, load_verified_array, write_canonical_array
from depth_visibility.baselines import validate_baseline_registry
from depth_visibility.capacity import (
    CapacityBank,
    apply_point_neutral_transaction,
    build_event_blind_capacity_targets,
    select_event_blind_donors,
)
from depth_visibility.canonical import domain_id, sha256_file
from depth_visibility.da3_adapter import (
    INFERENCE_ARGUMENTS,
    load_da3,
    processed_k_corner_error,
    relative_mad_maximum,
    repetition_delta_report,
    run_analytic_conformance,
    run_group,
    run_two_group_conformance,
    verify_model_authority,
)
from depth_visibility.fast_pilot import (
    anchor_duplicate_diagnostic,
    evaluate_frame_geometry,
    temporal_bin_transitions,
)
from depth_visibility.evaluator import (
    artifact_reference,
    decide_gate_a,
    load_run_entry,
    terminal_manifest,
    validate_execution_manifest,
    resolved_python_argv,
    write_json_atomic,
)
from depth_visibility.errors import ContractError, ProvenanceError, SchemaError
from depth_visibility.fixtures import two_plane_track_pixels
from depth_visibility.flow import validate_flow_manifest
from depth_visibility.schema import validate_payload
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
    produced = [
        item for item in terminal.get("produced_artifacts", [])
        if item.get("schema") == "phase9-da3-authority-v1"
        and item.get("producer_run_id") == "P9-A02-DA3-WEIGHT-SHA"
    ]
    if len(produced) != 1:
        raise ProvenanceError("A02 terminal does not bind exactly one DA3 authority artifact")
    authority_path = _expand(str(produced[0].get("path", ""))).resolve()
    if not authority_path.is_file():
        raise ProvenanceError("A02 DA3 authority artifact is missing")
    authority_sha = sha256_file(authority_path)
    if produced[0].get("sha256") != authority_sha:
        raise ProvenanceError("DA3 authority bytes do not match the A02 terminal")
    return authority_path, _json(authority_path), str(expected_terminal_sha)




P01_DA3_SIDECAR_RUN_ID = "P9-V5-P01-CUT-DA3-SIDECAR-S20260721"
P02_FLOW_SIDECAR_RUN_ID = "P9-V6-P02-CUT-FLOW-ADAPT-S20260721"


def _require_completed_terminal_input(
    execution: dict[str, Any] | None,
    *,
    producer_run_id: str,
    action: str,
) -> tuple[Path, dict[str, Any], str]:
    if execution is None:
        raise ProvenanceError(f"{producer_run_id} terminal requires a resolved execution manifest")
    matches = [
        item for item in execution.get("input_artifacts", [])
        if item.get("producer_run_id") == producer_run_id
        and item.get("schema") == "phase9-terminal-manifest-v1"
    ]
    if len(matches) != 1 or not str(matches[0].get("status", "")).startswith("resolved_exact"):
        raise ProvenanceError(f"P03 lacks one exact resolved terminal input for {producer_run_id}")
    terminal_ref = matches[0]
    terminal_path = _expand(str(terminal_ref["path"])).resolve()
    expected_sha = str(terminal_ref.get("sha256") or "")
    if not terminal_path.is_file() or sha256_file(terminal_path) != expected_sha:
        raise ProvenanceError(f"{producer_run_id} terminal bytes do not match the resolved P03 input")
    terminal = _json(terminal_path)
    if (
        terminal.get("schema_version") != "phase9-terminal-manifest-v1"
        or terminal.get("run_id") != producer_run_id
        or terminal.get("action") != action
        or terminal.get("status") != "completed"
        or terminal.get("exit_code") != 0
    ):
        raise ProvenanceError(f"{producer_run_id} terminal is not a successful {action} run")
    return terminal_path, terminal, expected_sha


def _bound_terminal_json_artifact(
    terminal: Mapping[str, Any],
    *,
    producer_run_id: str,
    schema: str,
) -> tuple[Path, dict[str, Any], str]:
    produced = [
        item for item in terminal.get("produced_artifacts", [])
        if item.get("schema") == schema and item.get("producer_run_id") == producer_run_id
    ]
    if len(produced) != 1:
        raise ProvenanceError(f"{producer_run_id} terminal does not bind exactly one {schema}")
    path = _expand(str(produced[0].get("path", ""))).resolve()
    if not path.is_file():
        raise ProvenanceError(f"{schema} artifact is missing: {path}")
    actual_sha = sha256_file(path)
    if produced[0].get("sha256") != actual_sha:
        raise ProvenanceError(f"{schema} bytes do not match the producer terminal")
    payload = _json(path)
    if payload.get("schema_version") != schema or payload.get("run_id") != producer_run_id:
        raise ProvenanceError(f"{schema} payload identity is incompatible with {producer_run_id}")
    return path, payload, actual_sha


def _bound_p03_inputs(execution: dict[str, Any] | None) -> dict[str, Any]:
    p01_terminal_path, p01_terminal, p01_terminal_sha = _require_completed_terminal_input(
        execution,
        producer_run_id=P01_DA3_SIDECAR_RUN_ID,
        action="produce-da3",
    )
    da3_manifest_path, da3_manifest, da3_manifest_sha = _bound_terminal_json_artifact(
        p01_terminal,
        producer_run_id=P01_DA3_SIDECAR_RUN_ID,
        schema="phase9-da3-sidecar-v1",
    )
    da3_arrays_path, da3_arrays, da3_arrays_sha = _bound_terminal_json_artifact(
        p01_terminal,
        producer_run_id=P01_DA3_SIDECAR_RUN_ID,
        schema="phase9-da3-array-inventory-v1",
    )
    p02_terminal_path, p02_terminal, p02_terminal_sha = _require_completed_terminal_input(
        execution,
        producer_run_id=P02_FLOW_SIDECAR_RUN_ID,
        action="adapt-flow",
    )
    flow_manifest_path, flow_manifest, flow_manifest_sha = _bound_terminal_json_artifact(
        p02_terminal,
        producer_run_id=P02_FLOW_SIDECAR_RUN_ID,
        schema="phase9-flow-manifest-v1",
    )
    return {
        "p01_terminal_path": p01_terminal_path,
        "p01_terminal_sha256": p01_terminal_sha,
        "da3_manifest_path": da3_manifest_path,
        "da3_manifest": da3_manifest,
        "da3_manifest_sha256": da3_manifest_sha,
        "da3_arrays_path": da3_arrays_path,
        "da3_arrays": da3_arrays,
        "da3_arrays_sha256": da3_arrays_sha,
        "p02_terminal_path": p02_terminal_path,
        "p02_terminal_sha256": p02_terminal_sha,
        "flow_manifest_path": flow_manifest_path,
        "flow_manifest": flow_manifest,
        "flow_manifest_sha256": flow_manifest_sha,
    }


def _select_sidecar_anchor_groups(
    group_records: list[Mapping[str, Any]],
) -> tuple[str, list[Mapping[str, Any]]]:
    by_members: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for record in group_records:
        members = tuple(str(value) for value in record.get("member_camera_ids", []))
        if not members or any(value == "cam00" for value in members):
            raise ProvenanceError("P03 sidecar group is missing train-camera membership or contains cam00")
        if members in by_members:
            raise ProvenanceError("P03 sidecar frame contains duplicate camera groups")
        by_members[members] = record
    unique = tuple(sorted(by_members))
    cameras = sorted({camera for group in unique for camera in group})
    eligible = [camera for camera in cameras if sum(camera in group for group in unique) >= 2]
    if not eligible:
        raise ProvenanceError("P03 cannot find a repeated train-camera anchor in sidecars")
    anchor_camera_id = eligible[0]
    selected_members = tuple(group for group in unique if anchor_camera_id in group)[:2]
    if len(selected_members) != 2:
        raise ProvenanceError("P03 selected anchor lacks two distinct sidecar groups")
    return anchor_camera_id, [by_members[members] for members in selected_members]


def _load_p01_group_prediction(
    sidecar_root: Path,
    group: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    refs = group.get("array_refs", {})
    required = {
        "depth", "confidence", "processed_intrinsics", "aligned_w2c",
        "expected_processed_intrinsics",
    }
    if not isinstance(refs, Mapping) or not required.issubset(refs):
        raise ProvenanceError("P03 DA3 sidecar group lacks required array references")

    def load(name: str) -> np.ndarray:
        ref = refs[name]
        if not isinstance(ref, Mapping):
            raise ProvenanceError(f"P03 DA3 array reference is not an object: {name}")
        return np.asarray(load_verified_array(sidecar_root / str(ref["path"]), ref))

    depth = load("depth")
    confidence = load("confidence")
    intrinsics = load("processed_intrinsics")
    extrinsics = load("aligned_w2c")
    expected_intrinsics = load("expected_processed_intrinsics")
    members = [str(value) for value in group.get("member_camera_ids", [])]
    if (
        depth.ndim != 3
        or confidence.shape != depth.shape
        or intrinsics.shape != (depth.shape[0], 3, 3)
        or expected_intrinsics.shape != intrinsics.shape
        or extrinsics.shape != (depth.shape[0], 4, 4)
        or len(members) != depth.shape[0]
    ):
        raise ProvenanceError("P03 DA3 sidecar arrays are not mutually aligned")
    prediction = {
        "depth": depth,
        "confidence": confidence,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }
    group_input = {
        "member_camera_ids": members,
        "expected_processed_intrinsics": expected_intrinsics,
    }
    return prediction, group_input


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
                "source_records": [
                    {
                        "camera_id": record.camera_id,
                        "image_path": str(record.image_path),
                        "image_sha256": record.image_sha256,
                        "file_stem": record.file_stem,
                        "time": float(record.time),
                    }
                    for record in records
                ],
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


def _build_real_group_inputs(
    frame_records: list[Any],
    groups: tuple[tuple[str, ...], ...],
) -> list[dict[str, Any]]:
    records_by_camera = {record.camera_id: record for record in frame_records}
    result = []
    for group in groups:
        records = [records_by_camera[camera_id] for camera_id in group]
        if any(record.image_path is None or record.camera_id == "cam00" for record in records):
            raise ProvenanceError("fast pilot group lacks training RGB or contains cam00")
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
        result.append(
            {
                "member_camera_ids": list(group),
                "images": [str(record.image_path) for record in records],
                "extrinsics_w2c": np.stack([record.w2c_opencv for record in records]),
                "intrinsics": np.stack([record.K for record in records]),
                "expected_processed_intrinsics": np.stack(expected_k),
                "source_records": [
                    {
                        "camera_id": record.camera_id,
                        "image_path": str(record.image_path),
                        "image_sha256": record.image_sha256,
                        "file_stem": record.file_stem,
                        "time": float(record.time),
                    }
                    for record in records
                ],
            }
        )
    return result


def _geometry_input_check(
    predictions: list[Mapping[str, Any]],
    group_inputs: list[dict[str, Any]],
    *,
    anchor_camera_id: str,
) -> Mapping[str, Any]:
    if len(predictions) != 2 or len(group_inputs) != 2:
        raise ContractError("fast geometry input check requires exactly two groups")
    anchor_depths = []
    k_errors = []
    for prediction, group in zip(predictions, group_inputs, strict=True):
        members = list(group["member_camera_ids"])
        if anchor_camera_id not in members:
            raise ContractError("fast geometry group is missing the shared anchor")
        anchor_depths.append(
            np.asarray(prediction["depth"])[members.index(anchor_camera_id)]
        )
        k_errors.append(
            processed_k_corner_error(
                np.asarray(group["expected_processed_intrinsics"]),
                np.asarray(prediction["intrinsics"]),
                int(np.asarray(prediction["depth"]).shape[1]),
                int(np.asarray(prediction["depth"]).shape[2]),
            )
        )
    return {
        "anchor_camera_id": anchor_camera_id,
        "anchor_cross_group_relative_mad_maximum": relative_mad_maximum(
            np.stack(anchor_depths)
        ),
        "processed_k_corner_error_maximum_pixels": max(k_errors),
        "processed_k_corner_error_by_group_pixels": k_errors,
    }


def _select_full_scene_groups(
    frame_records: list[Any], r_scene: float, config: dict[str, Any]
) -> tuple[tuple[str, ...], ...]:
    by_camera = {record.camera_id: record for record in frame_records}
    if len(by_camera) != len(frame_records) or "cam00" in by_camera:
        raise ProvenanceError("P01 requires one record per training camera and no cam00")
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
    if not unique:
        raise ProvenanceError("P01 found no complete train-camera DA3 groups")
    return unique


def _write_da3_group_sidecar(
    *,
    sidecar_root: Path,
    scene: str,
    frame: int,
    group_index: int,
    target_camera: str,
    group_input: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    member_camera_ids = [str(value) for value in group_input["member_camera_ids"]]
    source_records = [dict(item) for item in group_input.get("source_records", [])]
    if len(source_records) != len(member_camera_ids):
        raise ProvenanceError("P01 source-record provenance is incomplete")
    if any(not item.get("image_sha256") for item in source_records):
        raise ProvenanceError("P01 requires train-image SHA-256 bindings")
    identity = {
        "scene": str(scene),
        "frame": int(frame),
        "group_index": int(group_index),
        "target_camera": str(target_camera),
        "member_camera_ids": member_camera_ids,
    }
    group_id = domain_id("phase9-p01-da3-group-v1", identity)
    arrays_dir = sidecar_root / "arrays" / f"frame_{int(frame):06d}" / f"group_{int(group_index):04d}"
    arrays = {
        "input_intrinsics": np.asarray(group_input["intrinsics"]),
        "input_w2c": np.asarray(group_input["extrinsics_w2c"]),
        "expected_processed_intrinsics": np.asarray(group_input["expected_processed_intrinsics"]),
        "depth": np.asarray(prediction["depth"]),
        "confidence": np.asarray(prediction["confidence"]),
        "processed_intrinsics": np.asarray(prediction["intrinsics"]),
        "aligned_w2c": np.asarray(prediction["extrinsics"]),
        "processed_images": np.asarray(prediction["processed_images"]),
    }
    array_refs = {
        name: write_canonical_array(
            arrays_dir / f"{name}.npy",
            array,
            f"phase9-p01-da3-{name}",
            relative_to=sidecar_root,
        )
        for name, array in arrays.items()
    }
    depth = arrays["depth"]
    k_error = processed_k_corner_error(
        arrays["expected_processed_intrinsics"],
        arrays["processed_intrinsics"],
        int(depth.shape[1]),
        int(depth.shape[2]),
    )
    w2c_delta = float(np.max(np.abs(arrays["input_w2c"] - arrays["aligned_w2c"])))
    return {
        "group_id": group_id,
        **identity,
        "physical_ancestry": sorted(member_camera_ids),
        "source_records": source_records,
        "array_refs": array_refs,
        "processed_depth_shape": [int(value) for value in depth.shape],
        "processed_k_corner_error_maximum_pixels": float(k_error),
        "aligned_w2c_input_maximum_absolute_difference": w2c_delta,
    }


def action_produce_da3(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Produce immutable full-cut DA3 sidecars without opening cam00 RGB."""

    config = _json(DEFAULT_CONFIG)
    expected_scene = config["data"]["development_scene"]
    if entry.get("scene") != expected_scene or args.scene != expected_scene:
        raise ProvenanceError(f"P01 sidecar production is admitted only for {expected_scene}")
    if expected_scene != "cut_roasted_beef":
        raise ProvenanceError("P01 v5 is development-cut only")
    target_camera = str(config["data"]["test_camera"])
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
        hash_train_images=True,
        timestamp_tolerance_seconds=float(config["camera"]["timestamp_tolerance_seconds"]),
    )
    split_manifest = _json(REPO_ROOT / config["data"]["split_manifest"])
    split_binding = {
        key: dict(value)
        for key, value in validate_split_binding(index, split_manifest).items()
    }
    test_records = index.by_camera_frame("test")
    if any(record.image_path is not None for record in test_records.values()):
        raise ProvenanceError("P01 attempted to expose held-out cam00 RGB")
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

    manifest_path = _expected_path(entry, "phase9-da3-sidecar-v1")
    arrays_path = _expected_path(entry, "phase9-da3-array-inventory-v1")
    sidecar_root = manifest_path.parent
    train_records = index.split("train")
    frames = sorted({record.frame for record in train_records})
    r_scene = compute_r_scene(train_records)
    group_records: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    for frame in frames:
        frame_records = [record for record in train_records if record.frame == frame]
        groups = _select_full_scene_groups(list(frame_records), r_scene, config)
        group_inputs = _build_real_group_inputs(list(frame_records), groups)
        frame_group_ids: list[str] = []
        k_errors: list[float] = []
        w2c_errors: list[float] = []
        for group_index, group_input in enumerate(group_inputs):
            prediction = run_group(
                model,
                group_input["images"],
                group_input["extrinsics_w2c"],
                group_input["intrinsics"],
            )
            group_record = _write_da3_group_sidecar(
                sidecar_root=sidecar_root,
                scene=expected_scene,
                frame=frame,
                group_index=group_index,
                target_camera=target_camera,
                group_input=group_input,
                prediction=prediction,
            )
            group_records.append(group_record)
            frame_group_ids.append(group_record["group_id"])
            k_errors.append(float(group_record["processed_k_corner_error_maximum_pixels"]))
            w2c_errors.append(float(group_record["aligned_w2c_input_maximum_absolute_difference"]))
        frame_summaries.append(
            {
                "frame": int(frame),
                "target_camera": target_camera,
                "target_calibration_only": True,
                "group_count": len(frame_group_ids),
                "group_ids": frame_group_ids,
                "processed_k_corner_error_maximum_pixels": max(k_errors),
                "aligned_w2c_input_maximum_absolute_difference": max(w2c_errors),
            }
        )

    inventory = build_inventory(sidecar_root / "arrays")
    arrays_payload = {
        "schema_version": "phase9-da3-array-inventory-v1",
        "run_id": entry["run_id"],
        "scene": expected_scene,
        "array_root": "arrays",
        "file_count": len(inventory),
        "total_file_bytes": sum(int(item["bytes"]) for item in inventory),
        "files": inventory,
    }
    write_json_atomic(arrays_path, arrays_payload)
    manifest = {
        "schema_version": "phase9-da3-sidecar-v1",
        "run_id": entry["run_id"],
        "scene": expected_scene,
        "target_camera": target_camera,
        "cam00_rgb_opened": False,
        "label_dependent_gate_a": "not_evaluable",
        "methodology_status": "full_cut_da3_sidecar_production_no_csvl_scoring",
        "frames": frame_summaries,
        "groups": group_records,
        "frame_count": len(frames),
        "group_count": len(group_records),
        "r_scene": float(r_scene),
        "inference_arguments": dict(INFERENCE_ARGUMENTS),
        "split_binding": split_binding,
        "a02_authority_path": str(authority_path),
        "a02_authority_sha256": sha256_file(authority_path),
        "a02_terminal_sha256": a02_terminal_sha,
        "model_authority": model_authority,
        "array_inventory_sha256": sha256_file(arrays_path),
    }
    write_json_atomic(manifest_path, manifest)
    refs = [
        _scientific_file_ref(manifest_path, "phase9-da3-sidecar-v1", entry["run_id"]),
        _scientific_file_ref(arrays_path, "phase9-da3-array-inventory-v1", entry["run_id"]),
    ]
    return refs, {
        "frame_count": len(frames),
        "group_count": len(group_records),
        "array_file_count": arrays_payload["file_count"],
        "total_array_bytes": arrays_payload["total_file_bytes"],
        "cam00_rgb_opened": False,
        "label_dependent_gate_a": "not_evaluable",
    }


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _build_flow_record(
    *,
    scene: str,
    flow_path: Path,
    source_record: Any,
    target_record: Any,
    direction: str,
    generator_revision: str,
) -> dict[str, Any]:
    if source_record.camera_id != target_record.camera_id:
        raise ProvenanceError("P02 flow pairs must remain within one train camera")
    raw_sha = sha256_file(flow_path)
    with np.load(flow_path, allow_pickle=False) as payload:
        if set(payload.files) != {"flow", "mask"}:
            raise ProvenanceError(f"P02 flow NPZ has unexpected keys: {flow_path}")
        flow = np.asarray(payload["flow"])
        mask = np.asarray(payload["mask"])
    expected_hw = (int(source_record.height), int(source_record.width))
    if flow.dtype != np.float32 or flow.shape != (*expected_hw, 2) or not np.isfinite(flow).all():
        raise ProvenanceError(f"P02 flow array shape/dtype/finiteness mismatch: {flow_path}")
    if mask.dtype != np.bool_ or mask.shape != expected_hw:
        raise ProvenanceError(f"P02 flow validity mask shape/dtype mismatch: {flow_path}")
    if (source_record.image_sha256 is None) or (target_record.image_sha256 is None):
        raise ProvenanceError("P02 requires hashed train source and target RGB images")
    dt_seconds = float(target_record.time - source_record.time)
    validate_flow_manifest(
        {
            "source_camera": source_record.camera_id,
            "target_camera": target_record.camera_id,
            "source_image": source_record.file_stem,
            "target_image": target_record.file_stem,
            "source_frame": source_record.frame,
            "target_frame": target_record.frame,
            "direction": direction,
            "dt": dt_seconds,
            "height": source_record.height,
            "width": source_record.width,
            "units": "pixels",
            "pixel_centers": "integer",
            "sampling": "bilinear_align_corners_false",
            "validity_semantics": "true_means_sample_is_valid",
            "occlusion_semantics": "true_means_not_occluded",
            "generator_revision": generator_revision,
            "source_hashes": [source_record.image_sha256, target_record.image_sha256],
            "array_hash": raw_sha,
        }
    )
    return {
        "schema_version": "depth-visibility-flow-schema-v1",
        "scene": scene,
        "flow_npz_path": str(flow_path),
        "source_image": source_record.file_stem,
        "target_image": target_record.file_stem,
        "source_camera": source_record.camera_id,
        "target_camera": target_record.camera_id,
        "source_frame": int(source_record.frame),
        "target_frame": int(target_record.frame),
        "direction": direction,
        "dt_seconds": dt_seconds,
        "source_width": int(source_record.width),
        "source_height": int(source_record.height),
        "target_width": int(target_record.width),
        "target_height": int(target_record.height),
        "units": "pixels_at_source_resolution",
        "coordinate_convention": "integer_pixel_centers",
        "sampling": "bilinear_align_corners_false",
        "flow_key": "flow",
        "valid_key": "mask",
        "validity_semantics": "true_means_sample_is_valid",
        "occlusion_semantics": "true_means_not_occluded",
        "generator_name": "SEA-RAFT",
        "generator_revision": generator_revision,
        "source_hashes": [source_record.image_sha256, target_record.image_sha256],
        "array_hashes": {
            "npz_sha256": raw_sha,
            "flow_contiguous_sha256": _array_sha256(flow),
            "mask_contiguous_sha256": _array_sha256(mask.astype(np.uint8)),
        },
        "flow_dtype": str(flow.dtype),
        "valid_dtype": str(mask.dtype),
        "valid_pixel_fraction": float(mask.mean()),
    }


def action_adapt_flow(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seal existing SEA-RAFT flow sidecars without generating new flow."""

    config = _json(DEFAULT_CONFIG)
    scene = str(entry.get("scene") or "")
    if args.scene != scene or scene != "cut_roasted_beef":
        raise ProvenanceError("P02 v6 is admitted only for cut_roasted_beef")
    direction = str(config["evaluation"]["flow_direction"])
    if direction != "forward_t_to_t_plus_1":
        raise ProvenanceError("P02 raw SEA-RAFT sidecars require forward_t_to_t_plus_1 semantics")
    project_root = _expand("$WORK/proj_adags")
    scene_root = project_root / "data/n3v" / scene
    flow_root = scene_root / "flow"
    generator_path = project_root / "repo/SEA-RAFT/generate_dataset_flow.py"
    if not flow_root.is_dir() or not generator_path.is_file():
        raise ProvenanceError("P02 raw flow root or SEA-RAFT generator is missing")

    index = load_scene_index(
        scene_root,
        scene=scene,
        expose_test_images=False,
        hash_train_images=True,
        timestamp_tolerance_seconds=float(config["camera"]["timestamp_tolerance_seconds"]),
    )
    split_manifest = _json(REPO_ROOT / config["data"]["split_manifest"])
    split_binding = {
        key: dict(value)
        for key, value in validate_split_binding(index, split_manifest).items()
    }
    if any(record.image_path is not None for record in index.by_camera_frame("test").values()):
        raise ProvenanceError("P02 attempted to expose held-out cam00 RGB")

    by_camera_frame = index.by_camera_frame("train")
    train_cameras = sorted({camera for camera, _ in by_camera_frame})
    train_frames = sorted({frame for _, frame in by_camera_frame})
    if train_frames != list(range(min(train_frames), max(train_frames) + 1)):
        raise ProvenanceError("P02 train frames are not contiguous")
    raw_flow_paths = sorted(flow_root.glob("*.npz"))
    raw_flow_names = {path.name for path in raw_flow_paths}
    expected_names: set[str] = set()
    records: list[dict[str, Any]] = []
    valid_fractions: list[float] = []
    generator_revision = sha256_file(generator_path)
    for camera in train_cameras:
        for frame in train_frames[:-1]:
            source = by_camera_frame[(camera, frame)]
            target = by_camera_frame[(camera, frame + 1)]
            flow_name = f"{Path(source.file_stem).name}.npz"
            flow_path = flow_root / flow_name
            expected_names.add(flow_name)
            if not flow_path.is_file():
                raise ProvenanceError(f"P02 missing expected train flow sidecar: {flow_path}")
            record = _build_flow_record(
                scene=scene,
                flow_path=flow_path,
                source_record=source,
                target_record=target,
                direction=direction,
                generator_revision=generator_revision,
            )
            records.append(record)
            valid_fractions.append(float(record["valid_pixel_fraction"]))

    expected_count = len(train_cameras) * (len(train_frames) - 1)
    if len(records) != expected_count:
        raise ProvenanceError("P02 emitted the wrong number of flow records")
    unused_flow_files = sorted(raw_flow_names - expected_names)
    manifest_path = _expected_path(entry, "phase9-flow-manifest-v1")
    manifest = {
        "schema_version": "phase9-flow-manifest-v1",
        "run_id": entry["run_id"],
        "scene": scene,
        "flow_record_schema_version": "depth-visibility-flow-schema-v1",
        "flow_root": str(flow_root),
        "source_split": "train",
        "target_camera": str(config["data"]["test_camera"]),
        "cam00_rgb_opened": False,
        "direction": direction,
        "generator": {
            "name": "SEA-RAFT",
            "script_path": str(generator_path),
            "script_sha256": generator_revision,
            "provenance": "generate_dataset_flow.py saves model(img_t, img_t_plus_1) to the source-frame NPZ.",
        },
        "camera_ids": train_cameras,
        "frame_range": [int(train_frames[0]), int(train_frames[-1])],
        "temporal_pair_count_per_camera": len(train_frames) - 1,
        "expected_record_count": expected_count,
        "record_count": len(records),
        "raw_flow_file_count": len(raw_flow_paths),
        "unused_flow_file_count": len(unused_flow_files),
        "unused_flow_file_examples": unused_flow_files[:10],
        "valid_fraction_minimum": min(valid_fractions),
        "valid_fraction_mean": float(sum(valid_fractions) / len(valid_fractions)),
        "valid_fraction_maximum": max(valid_fractions),
        "split_binding": split_binding,
        "records": records,
        "label_dependent_gate_a": "not_evaluable",
    }
    write_json_atomic(manifest_path, manifest)
    reference = _scientific_file_ref(
        manifest_path, "phase9-flow-manifest-v1", entry["run_id"]
    )
    return [reference], {
        "scene": scene,
        "direction": direction,
        "record_count": len(records),
        "expected_record_count": expected_count,
        "raw_flow_file_count": len(raw_flow_paths),
        "unused_flow_file_count": len(unused_flow_files),
        "cam00_rgb_opened": False,
        "label_dependent_gate_a": "not_evaluable",
    }




def action_build_csvl(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a sealed full-cut label-free CSVL bin-layer evidence ledger."""

    config = _json(DEFAULT_CONFIG)
    scene = str(entry.get("scene") or "")
    expected_scene = str(config["data"]["development_scene"])
    target_camera = str(config["data"]["test_camera"])
    if scene != expected_scene or args.scene != scene or scene != "cut_roasted_beef":
        raise ProvenanceError("P03 v7 is admitted only for cut_roasted_beef")

    bindings = _bound_p03_inputs(execution)
    da3_manifest = bindings["da3_manifest"]
    da3_arrays = bindings["da3_arrays"]
    flow_manifest = bindings["flow_manifest"]
    if (
        da3_manifest.get("schema_version") != "phase9-da3-sidecar-v1"
        or da3_manifest.get("run_id") != P01_DA3_SIDECAR_RUN_ID
        or da3_manifest.get("scene") != scene
        or da3_manifest.get("target_camera") != target_camera
        or da3_manifest.get("cam00_rgb_opened") is not False
        or da3_manifest.get("label_dependent_gate_a") != "not_evaluable"
    ):
        raise ProvenanceError("P03 DA3 manifest is incompatible with the cut CSVL ledger")
    if (
        da3_arrays.get("schema_version") != "phase9-da3-array-inventory-v1"
        or da3_arrays.get("run_id") != P01_DA3_SIDECAR_RUN_ID
        or sha256_file(bindings["da3_arrays_path"]) != da3_manifest.get("array_inventory_sha256")
    ):
        raise ProvenanceError("P03 DA3 array inventory does not match the P01 manifest")
    if (
        flow_manifest.get("schema_version") != "phase9-flow-manifest-v1"
        or flow_manifest.get("run_id") != P02_FLOW_SIDECAR_RUN_ID
        or flow_manifest.get("scene") != scene
        or flow_manifest.get("target_camera") != target_camera
        or flow_manifest.get("cam00_rgb_opened") is not False
        or flow_manifest.get("direction") != "forward_t_to_t_plus_1"
        or int(flow_manifest.get("record_count", -1)) != int(flow_manifest.get("expected_record_count", -2))
    ):
        raise ProvenanceError("P03 flow manifest is incompatible with the cut CSVL ledger")

    project_root = _expand("$WORK/proj_adags")
    scene_root = project_root / "data/n3v" / scene
    index = load_scene_index(
        scene_root,
        scene=scene,
        expose_test_images=False,
        hash_train_images=False,
        timestamp_tolerance_seconds=float(config["camera"]["timestamp_tolerance_seconds"]),
    )
    split_manifest = _json(REPO_ROOT / config["data"]["split_manifest"])
    split_binding = {
        key: dict(value)
        for key, value in validate_split_binding(index, split_manifest).items()
    }
    test_records = index.by_camera_frame("test")
    if any(record.image_path is not None for record in test_records.values()):
        raise ProvenanceError("P03 attempted to expose held-out cam00 RGB")

    groups_by_frame: dict[int, list[Mapping[str, Any]]] = {}
    for group in da3_manifest.get("groups", []):
        if group.get("scene") != scene or group.get("target_camera") != target_camera:
            raise ProvenanceError("P03 DA3 sidecar group has incompatible scene or target camera")
        frame = int(group["frame"])
        groups_by_frame.setdefault(frame, []).append(group)
    frames = [int(item["frame"]) for item in da3_manifest.get("frames", [])]
    frames = sorted(frames)
    if (
        len(frames) != int(da3_manifest.get("frame_count", -1))
        or sum(len(value) for value in groups_by_frame.values()) != int(da3_manifest.get("group_count", -1))
        or set(frames) != set(groups_by_frame)
    ):
        raise ProvenanceError("P03 DA3 frame/group counts do not match the sidecar manifest")
    if list(flow_manifest.get("frame_range", [])) != [frames[0], frames[-1]]:
        raise ProvenanceError("P03 flow frame range does not cover the DA3 frame range")
    flow_cameras = [str(value) for value in flow_manifest.get("camera_ids", [])]
    if target_camera in flow_cameras:
        raise ProvenanceError("P03 flow manifest includes the held-out target camera")

    sidecar_root = bindings["da3_manifest_path"].parent
    frame_reports: list[dict[str, Any]] = []
    frame_bin_sets: list[tuple[int, set[tuple[int, int]]]] = []
    ordered_bin_rows: list[list[int]] = []
    ordered_layer_rows: list[list[float]] = []
    relative_limit = float(config["fusion"]["duplicate_relative_mad_maximum"])
    stride = int(config["sampling"]["grid_stride_pixels"])
    for frame in frames:
        target = test_records.get((target_camera, frame))
        if target is None or target.image_path is not None:
            raise ProvenanceError("P03 requires calibration-only cam00 target access")
        anchor_camera_id, selected_groups = _select_sidecar_anchor_groups(
            groups_by_frame[frame]
        )
        predictions = []
        group_inputs = []
        for group in selected_groups:
            prediction, group_input = _load_p01_group_prediction(sidecar_root, group)
            predictions.append(prediction)
            group_inputs.append(group_input)
        input_check = dict(
            _geometry_input_check(
                predictions,
                group_inputs,
                anchor_camera_id=anchor_camera_id,
            )
        )
        anchor_indices = [
            group["member_camera_ids"].index(anchor_camera_id)
            for group in group_inputs
        ]
        first_index, second_index = anchor_indices
        first_prediction, second_prediction = predictions
        extrinsic_max_abs = float(
            np.max(
                np.abs(
                    np.asarray(first_prediction["extrinsics"])[first_index]
                    - np.asarray(second_prediction["extrinsics"])[second_index]
                )
            )
        )
        input_check["anchor_extrinsic_maximum_absolute_difference"] = extrinsic_max_abs
        coordinate_admitted = (
            input_check["processed_k_corner_error_maximum_pixels"] <= 0.5
            and extrinsic_max_abs <= float(config["da3"]["conformance_repeat_atol"])
        )
        input_check["coordinate_admitted"] = bool(coordinate_admitted)
        group_ids = [str(group["group_id"]) for group in selected_groups]
        group_members = [list(group["member_camera_ids"]) for group in selected_groups]
        if not coordinate_admitted:
            frame_payload = {
                "scene": scene,
                "frame": int(frame),
                "target_camera": target_camera,
                "anchor_camera_id": anchor_camera_id,
                "source_da3_group_ids": group_ids,
                "geometry_executed": False,
                "geometry_input_check": input_check,
                "blocked_reason": "coordinate_not_admitted",
            }
            frame_reports.append(
                {
                    **frame_payload,
                    "frame_ledger_id": domain_id("phase9-p03-csvl-frame-ledger-v1", frame_payload),
                    "ordered_multilayer_bins": [],
                }
            )
            continue

        duplicate_report, aggregate_depth, retained_mask = anchor_duplicate_diagnostic(
            np.asarray(first_prediction["depth"])[first_index],
            np.asarray(second_prediction["depth"])[second_index],
            np.asarray(first_prediction["confidence"])[first_index],
            np.asarray(second_prediction["confidence"])[second_index],
            relative_limit=relative_limit,
            calibration_stride=stride,
        )
        geometry, supported_bins, layer_records = evaluate_frame_geometry(
            predictions,
            group_members,
            target_K=target.K,
            target_w2c=target.w2c_opencv,
            target_width=target.width,
            target_height=target.height,
            stride=stride,
            minimum_cameras=int(config["fusion"]["minimum_physical_cameras"]),
            maximum_depth_sigma=float(config["fusion"]["maximum_depth_sigma"]),
            target_bin_pixels=stride,
            camera_depth_overrides={anchor_camera_id: aggregate_depth},
            camera_valid_masks={anchor_camera_id: retained_mask},
            include_layer_records=True,
        )
        sealed_bins = []
        for raw_record in layer_records:
            payload = {
                "scene": scene,
                "frame": int(frame),
                "target_camera": target_camera,
                "hypothesis_type": "target_bin_ordered_multilayer",
                "visibility_event_state": "unknown_without_temporal_identity_or_human_reference",
                "physical_ancestry": sorted(str(value) for value in raw_record["source_cameras"]),
                "source_da3_group_ids": group_ids,
                "anchor_camera_id": anchor_camera_id,
                "target_bin": list(raw_record["target_bin"]),
                "target_bin_pixels": int(raw_record["target_bin_pixels"]),
                "physical_camera_count": int(raw_record["physical_camera_count"]),
                "sample_count": int(raw_record["sample_count"]),
                "source_cameras": list(raw_record["source_cameras"]),
                "layers": [dict(layer) for layer in raw_record["layers"]],
                "order_pairs": [dict(pair) for pair in raw_record["order_pairs"]],
            }
            sealed = {
                "csvl_hypothesis_id": domain_id(
                    "phase9-p03-csvl-bin-layer-hypothesis-v1", payload
                ),
                **payload,
            }
            sealed_bins.append(sealed)
            bin_x, bin_y = (int(value) for value in sealed["target_bin"])
            ordered_bin_rows.append(
                [
                    int(frame),
                    bin_x,
                    bin_y,
                    int(len(sealed["layers"])),
                    int(sealed["sample_count"]),
                    int(sealed["physical_camera_count"]),
                ]
            )
            for layer in sealed["layers"]:
                ordered_layer_rows.append(
                    [
                        float(frame),
                        float(bin_x),
                        float(bin_y),
                        float(layer["layer_ordinal"]),
                        float(layer["median_optical_z"]),
                        float(layer["sample_count"]),
                        float(layer["physical_camera_count"]),
                        float(layer["median_risk"]),
                    ]
                )
        frame_payload = {
            "scene": scene,
            "frame": int(frame),
            "target_camera": target_camera,
            "anchor_camera_id": anchor_camera_id,
            "source_da3_group_ids": group_ids,
            "source_group_members": group_members,
            "geometry_executed": True,
            "geometry_input_check": input_check,
            "anchor_duplicate": duplicate_report,
            "geometry": geometry,
            "ordered_multilayer_bins": sealed_bins,
        }
        frame_reports.append(
            {
                **frame_payload,
                "frame_ledger_id": domain_id("phase9-p03-csvl-frame-ledger-v1", frame_payload),
            }
        )
        frame_bin_sets.append((int(frame), supported_bins))

    csvl_root = _expected_path(entry, "phase9-csvl-ledger-v1").parent
    arrays_root = csvl_root / "arrays"
    bin_table = (
        np.asarray(ordered_bin_rows, dtype=np.int64).reshape((-1, 6))
        if ordered_bin_rows else np.empty((0, 6), dtype=np.int64)
    )
    layer_table = (
        np.asarray(ordered_layer_rows, dtype=np.float64).reshape((-1, 8))
        if ordered_layer_rows else np.empty((0, 8), dtype=np.float64)
    )
    array_refs = {
        "ordered_multilayer_bins": write_canonical_array(
            arrays_root / "ordered_multilayer_bins.npy",
            bin_table,
            "phase9-p03-csvl-ordered-multilayer-bins",
            relative_to=csvl_root,
        ),
        "ordered_layers": write_canonical_array(
            arrays_root / "ordered_layers.npy",
            layer_table,
            "phase9-p03-csvl-ordered-layers",
            relative_to=csvl_root,
        ),
    }
    inventory = build_inventory(
        arrays_root,
        paths=["ordered_multilayer_bins.npy", "ordered_layers.npy"],
    )
    arrays_payload_no_id = {
        "schema_version": "phase9-csvl-array-inventory-v1",
        "run_id": entry["run_id"],
        "method_id": str(config["method_id"]),
        "scene": scene,
        "array_root": "arrays",
        "file_count": len(inventory),
        "total_file_bytes": sum(int(item["bytes"]) for item in inventory),
        "files": inventory,
        "arrays": array_refs,
        "input_array_inventories": [
            {
                "path": str(bindings["da3_arrays_path"]),
                "schema": "phase9-da3-array-inventory-v1",
                "producer_run_id": P01_DA3_SIDECAR_RUN_ID,
                "sha256": bindings["da3_arrays_sha256"],
            }
        ],
    }
    arrays_payload = {
        **arrays_payload_no_id,
        "artifact_id": domain_id("phase9-p03-csvl-array-inventory-v1", arrays_payload_no_id),
    }
    arrays_path = _expected_path(entry, "phase9-csvl-array-inventory-v1")
    validate_payload("phase9-csvl-array-inventory-v1", arrays_payload)
    write_json_atomic(arrays_path, arrays_payload)

    aggregate = dict(_aggregate_layer_opportunities(frame_reports))
    aggregate["emitted_ordered_multilayer_bin_hypothesis_count"] = int(len(ordered_bin_rows))
    aggregate["emitted_layer_hypothesis_count"] = int(len(ordered_layer_rows))
    aggregate["interpretation"] = (
        "full-cut label-free target-bin layer evidence; not a human-reference Gate A score"
    )
    flow_summary = {
        "direction": str(flow_manifest["direction"]),
        "record_count": int(flow_manifest["record_count"]),
        "expected_record_count": int(flow_manifest["expected_record_count"]),
        "camera_count": len(flow_cameras),
        "frame_range": [int(value) for value in flow_manifest["frame_range"]],
        "valid_fraction_minimum": float(flow_manifest["valid_fraction_minimum"]),
        "valid_fraction_mean": float(flow_manifest["valid_fraction_mean"]),
        "valid_fraction_maximum": float(flow_manifest["valid_fraction_maximum"]),
        "consumption_status": "sealed_available_not_used_for_temporal_identity_propagation_v1",
    }
    ledger_payload_no_id = {
        "schema_version": "phase9-csvl-ledger-v1",
        "run_id": entry["run_id"],
        "method_id": str(config["method_id"]),
        "scene": scene,
        "target_camera": target_camera,
        "cam00_rgb_opened": False,
        "label_dependent_gate_a": "not_evaluable",
        "methodology_status": "full_cut_label_free_bin_layer_ledger_no_temporal_identity_propagation",
        "evidence_boundary": {
            "held_out_target_rgb": "not_opened",
            "human_labels": "not_consumed",
            "temporal_identity_status": "not_propagated_in_p03_v7",
            "transition_label_status": "unknown_without_human_reference",
            "capacity_admission_status": "not_admitted_by_p03_alone",
        },
        "input_bindings": {
            "p01_terminal": {
                "path": str(bindings["p01_terminal_path"]),
                "producer_run_id": P01_DA3_SIDECAR_RUN_ID,
                "schema": "phase9-terminal-manifest-v1",
                "sha256": bindings["p01_terminal_sha256"],
            },
            "p01_da3_sidecar": {
                "path": str(bindings["da3_manifest_path"]),
                "producer_run_id": P01_DA3_SIDECAR_RUN_ID,
                "schema": "phase9-da3-sidecar-v1",
                "sha256": bindings["da3_manifest_sha256"],
            },
            "p01_da3_array_inventory": {
                "path": str(bindings["da3_arrays_path"]),
                "producer_run_id": P01_DA3_SIDECAR_RUN_ID,
                "schema": "phase9-da3-array-inventory-v1",
                "sha256": bindings["da3_arrays_sha256"],
            },
            "p02_terminal": {
                "path": str(bindings["p02_terminal_path"]),
                "producer_run_id": P02_FLOW_SIDECAR_RUN_ID,
                "schema": "phase9-terminal-manifest-v1",
                "sha256": bindings["p02_terminal_sha256"],
            },
            "p02_flow_manifest": {
                "path": str(bindings["flow_manifest_path"]),
                "producer_run_id": P02_FLOW_SIDECAR_RUN_ID,
                "schema": "phase9-flow-manifest-v1",
                "sha256": bindings["flow_manifest_sha256"],
            },
        },
        "frame_count": len(frame_reports),
        "geometry_frame_count": int(sum(1 for item in frame_reports if item.get("geometry_executed"))),
        "frames": frame_reports,
        "aggregate_layer_opportunity": aggregate,
        "temporal_bin_transitions": temporal_bin_transitions(frame_bin_sets),
        "temporal_interpretation": "target-bin occupancy proxy only; not a surface-track reveal/hide label",
        "flow_summary": flow_summary,
        "duplicate_semantics": {
            "x02_rule_reused": "per-pixel two-sample relative half-difference >0.05 abstains; retained pixels use confidence-weighted aggregate",
            "threshold_changed": False,
            "heldout_scale_diagnostic_applied_to_geometry": False,
        },
        "threshold_authority": {
            "grid_stride_pixels": stride,
            "minimum_physical_cameras": int(config["fusion"]["minimum_physical_cameras"]),
            "maximum_depth_sigma": float(config["fusion"]["maximum_depth_sigma"]),
            "duplicate_relative_mad_maximum": relative_limit,
            "target_bin_pixels": stride,
        },
        "split_binding": split_binding,
        "array_inventory_sha256": sha256_file(arrays_path),
    }
    ledger_payload = {
        **ledger_payload_no_id,
        "artifact_id": domain_id("phase9-p03-csvl-ledger-v1", ledger_payload_no_id),
    }
    ledger_path = _expected_path(entry, "phase9-csvl-ledger-v1")
    validate_payload("phase9-csvl-ledger-v1", ledger_payload)
    write_json_atomic(ledger_path, ledger_payload)
    refs = [
        _scientific_file_ref(ledger_path, "phase9-csvl-ledger-v1", entry["run_id"]),
        _scientific_file_ref(arrays_path, "phase9-csvl-array-inventory-v1", entry["run_id"]),
    ]
    return refs, {
        "scene": scene,
        "frame_count": len(frame_reports),
        "geometry_frame_count": ledger_payload_no_id["geometry_frame_count"],
        "frames_with_ordered_layers": aggregate["frames_with_ordered_layers"],
        "total_ordered_multilayer_bins": aggregate["total_ordered_multilayer_bins"],
        "emitted_ordered_multilayer_bin_hypothesis_count": len(ordered_bin_rows),
        "emitted_layer_hypothesis_count": len(ordered_layer_rows),
        "csvl_ledger_sha256": sha256_file(ledger_path),
        "csvl_array_inventory_sha256": sha256_file(arrays_path),
        "cam00_rgb_opened": False,
        "label_dependent_gate_a": "not_evaluable",
        "temporal_identity_status": "not_propagated_in_p03_v7",
    }

def action_fast_visibility_pilot(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Quantify A03 drift and immediately exercise small real visibility geometry."""

    config = _json(DEFAULT_CONFIG)
    expected_scene = config["data"]["development_scene"]
    if args.scene != expected_scene or expected_scene != "cut_roasted_beef":
        raise ProvenanceError(f"fast pilot scene must be {expected_scene}")
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
    split_binding = {
        key: dict(value)
        for key, value in validate_split_binding(index, split_manifest).items()
    }
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

    frames = (125, 126, 127)
    test_records = index.by_camera_frame("test")
    frame_reports = []
    frame_bin_sets: list[tuple[int, set[tuple[int, int]]]] = []
    geometry_input_checks = []
    repeatability = None
    geometry_admitted = True
    r_scene = compute_r_scene(index.split("train"))
    duplicate_limit = float(config["fusion"]["duplicate_relative_mad_maximum"])
    for frame in frames:
        frame_records = [
            record for record in index.split("train") if record.frame == frame
        ]
        anchor_camera_id, groups = _select_conformance_groups(
            frame_records, r_scene, config
        )
        group_inputs = _build_real_group_inputs(frame_records, groups)
        first_prediction = run_group(
            model,
            group_inputs[0]["images"],
            group_inputs[0]["extrinsics_w2c"],
            group_inputs[0]["intrinsics"],
        )
        if frame == frames[0]:
            second_prediction = run_group(
                model,
                group_inputs[0]["images"],
                group_inputs[0]["extrinsics_w2c"],
                group_inputs[0]["intrinsics"],
            )
            repeatability = dict(
                repetition_delta_report(
                    first_prediction,
                    second_prediction,
                    expected_processed_intrinsics=group_inputs[0][
                        "expected_processed_intrinsics"
                    ],
                    repeat_atol=float(config["da3"]["conformance_repeat_atol"]),
                    repeat_rtol=float(config["da3"]["conformance_repeat_rtol"]),
                )
            )
        second_group_prediction = run_group(
            model,
            group_inputs[1]["images"],
            group_inputs[1]["extrinsics_w2c"],
            group_inputs[1]["intrinsics"],
        )
        predictions = [first_prediction, second_group_prediction]
        input_check = dict(
            _geometry_input_check(
                predictions,
                group_inputs,
                anchor_camera_id=anchor_camera_id,
            )
        )
        input_check["frame"] = frame
        geometry_input_checks.append(input_check)
        frame_admitted = (
            input_check["anchor_cross_group_relative_mad_maximum"]
            <= duplicate_limit
            and input_check["processed_k_corner_error_maximum_pixels"] <= 0.5
            and (
                frame != frames[0]
                or (
                    repeatability is not None
                    and repeatability["duplicate_relative_mad_maximum"]
                    <= duplicate_limit
                )
            )
        )
        input_check["admitted"] = frame_admitted
        if not frame_admitted:
            geometry_admitted = False
            break

        target = test_records.get((config["data"]["test_camera"], frame))
        if target is None or target.image_path is not None:
            raise ProvenanceError("fast pilot requires calibration-only cam00 target access")
        geometry, supported_bins = evaluate_frame_geometry(
            predictions,
            [item["member_camera_ids"] for item in group_inputs],
            target_K=target.K,
            target_w2c=target.w2c_opencv,
            target_width=target.width,
            target_height=target.height,
            stride=int(config["sampling"]["grid_stride_pixels"]),
            minimum_cameras=int(config["fusion"]["minimum_physical_cameras"]),
            maximum_depth_sigma=float(config["fusion"]["maximum_depth_sigma"]),
            target_bin_pixels=int(config["sampling"]["grid_stride_pixels"]),
        )
        frame_reports.append(
            {
                "frame": frame,
                "anchor_camera_id": anchor_camera_id,
                "groups": [list(group) for group in groups],
                "geometry_input_check": input_check,
                "geometry": geometry,
            }
        )
        frame_bin_sets.append((frame, supported_bins))

    if repeatability is None:
        raise ContractError("fast pilot did not execute its repeatability diagnostic")
    report_path = _expected_path(entry, "phase9-fast-visibility-pilot-v1")
    report = {
        "schema_version": "phase9-fast-visibility-pilot-v1",
        "run_id": entry["run_id"],
        "scene": expected_scene,
        "frames": list(frames),
        "methodology_status": "exploratory_fail_fast_geometry_diagnostic",
        "a03_registered_verdict": "failed_1e-5_repeatability_preserved",
        "repeatability": repeatability,
        "geometry_admission": {
            "admitted": geometry_admitted,
            "rule": "same-group and shared-anchor cross-group depth relative MAD <= frozen 0.05; every fused group processed-K corner error <= 0.5 pixels",
            "does_not_override_a03": True,
        },
        "geometry_input_checks": geometry_input_checks,
        "frame_reports": frame_reports,
        "temporal_bin_transitions": temporal_bin_transitions(frame_bin_sets),
        "temporal_interpretation": "calibration-only target-bin occupancy proxy; not surface-track reveal/hide evidence",
        "label_dependent_gate_a": "not_evaluable",
        "cam00_rgb_opened": False,
        "split_binding": split_binding,
        "a02_authority_path": str(authority_path),
        "a02_authority_sha256": sha256_file(authority_path),
        "a02_terminal_sha256": a02_terminal_sha,
        "model_authority": model_authority,
    }
    write_json_atomic(report_path, report)
    reference = _scientific_file_ref(
        report_path, "phase9-fast-visibility-pilot-v1", entry["run_id"]
    )
    payload = {
        "geometry_admitted": geometry_admitted,
        "frame125_same_group_strict_repeatability_pass": repeatability["strict_repeatability_pass"],
        "duplicate_relative_mad_maximum": repeatability[
            "duplicate_relative_mad_maximum"
        ],
        "frame_report_count": len(frame_reports),
        "supported_bin_counts": [
            item["geometry"]["target_supported_bin_count"] for item in frame_reports
        ],
        "ordered_multilayer_bin_counts": [
            item["geometry"]["target_ordered_multilayer_bin_count"]
            for item in frame_reports
        ],
        "label_dependent_gate_a": "not_evaluable",
    }
    return [reference], payload


def _require_completed_x01(execution: dict[str, Any] | None) -> Mapping[str, Any]:
    if execution is None:
        raise ProvenanceError("X02 requires a resolved execution manifest")
    matches = [
        item
        for item in execution.get("input_artifacts", [])
        if item.get("producer_run_id")
        == "P9-V2-X01-CUT-FAST-VISIBILITY-S20260717"
        and item.get("schema") == "phase9-terminal-manifest-v1"
        and str(item.get("status", "")).startswith("resolved_exact")
    ]
    if len(matches) != 1:
        raise ProvenanceError("X02 lacks one exact successful X01 terminal input")
    terminal = _json(_expand(str(matches[0]["path"])))
    if (
        terminal.get("schema_version") != "phase9-terminal-manifest-v1"
        or terminal.get("run_id")
        != "P9-V2-X01-CUT-FAST-VISIBILITY-S20260717"
        or terminal.get("action") != "fast-visibility-pilot"
        or terminal.get("status") != "completed"
        or terminal.get("exit_code") != 0
        or terminal.get("scientific_payload", {}).get("geometry_admitted") is not False
    ):
        raise ProvenanceError("X01 terminal does not bind the expected negative result")
    return terminal


def action_anchor_abstention_pilot(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen per-pixel duplicate rejection and run real geometry."""

    config = _json(DEFAULT_CONFIG)
    expected_scene = config["data"]["development_scene"]
    if args.scene != expected_scene or expected_scene != "cut_roasted_beef":
        raise ProvenanceError(f"anchor abstention pilot scene must be {expected_scene}")
    x01_terminal = _require_completed_x01(execution)
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
    split_binding = {
        key: dict(value)
        for key, value in validate_split_binding(index, split_manifest).items()
    }
    authority_path, authority, a02_terminal_sha = _bound_a02_authority(
        execution, Path(args.matrix).resolve()
    )
    model_authority = verify_model_authority(
        model_dir,
        expected_weight_sha256=authority["weight_sha256"],
        hash_weights=True,
    )
    _seed_conformance(int(entry["seeds"]["training"]))
    model = load_da3(da3_checkout, model_dir, device="cuda")

    frames = (125, 126, 127)
    test_records = index.by_camera_frame("test")
    r_scene = compute_r_scene(index.split("train"))
    frame_reports = []
    frame_bin_sets: list[tuple[int, set[tuple[int, int]]]] = []
    coordinate_admitted = True
    relative_limit = float(config["fusion"]["duplicate_relative_mad_maximum"])
    for frame in frames:
        frame_records = [
            record for record in index.split("train") if record.frame == frame
        ]
        anchor_camera_id, groups = _select_conformance_groups(
            frame_records, r_scene, config
        )
        group_inputs = _build_real_group_inputs(frame_records, groups)
        predictions = [
            run_group(
                model,
                group["images"],
                group["extrinsics_w2c"],
                group["intrinsics"],
            )
            for group in group_inputs
        ]
        input_check = dict(
            _geometry_input_check(
                predictions,
                group_inputs,
                anchor_camera_id=anchor_camera_id,
            )
        )
        anchor_indices = [
            group["member_camera_ids"].index(anchor_camera_id)
            for group in group_inputs
        ]
        first_index, second_index = anchor_indices
        first_prediction, second_prediction = predictions
        extrinsic_max_abs = float(
            np.max(
                np.abs(
                    np.asarray(first_prediction["extrinsics"][first_index])
                    - np.asarray(second_prediction["extrinsics"][second_index])
                )
            )
        )
        input_check["anchor_extrinsic_maximum_absolute_difference"] = extrinsic_max_abs
        input_check["coordinate_admitted"] = (
            input_check["processed_k_corner_error_maximum_pixels"] <= 0.5
            and extrinsic_max_abs
            <= float(config["da3"]["conformance_repeat_atol"])
        )
        if not input_check["coordinate_admitted"]:
            coordinate_admitted = False
            frame_reports.append(
                {
                    "frame": frame,
                    "anchor_camera_id": anchor_camera_id,
                    "groups": [list(group) for group in groups],
                    "geometry_input_check": input_check,
                    "geometry_executed": False,
                }
            )
            break

        duplicate_report, aggregate_depth, retained_mask = (
            anchor_duplicate_diagnostic(
                first_prediction["depth"][first_index],
                second_prediction["depth"][second_index],
                first_prediction["confidence"][first_index],
                second_prediction["confidence"][second_index],
                relative_limit=relative_limit,
                calibration_stride=int(config["sampling"]["grid_stride_pixels"]),
            )
        )
        target = test_records.get((config["data"]["test_camera"], frame))
        if target is None or target.image_path is not None:
            raise ProvenanceError("X02 requires calibration-only cam00 target access")
        geometry, supported_bins = evaluate_frame_geometry(
            predictions,
            [item["member_camera_ids"] for item in group_inputs],
            target_K=target.K,
            target_w2c=target.w2c_opencv,
            target_width=target.width,
            target_height=target.height,
            stride=int(config["sampling"]["grid_stride_pixels"]),
            minimum_cameras=int(config["fusion"]["minimum_physical_cameras"]),
            maximum_depth_sigma=float(config["fusion"]["maximum_depth_sigma"]),
            target_bin_pixels=int(config["sampling"]["grid_stride_pixels"]),
            camera_depth_overrides={anchor_camera_id: aggregate_depth},
            camera_valid_masks={anchor_camera_id: retained_mask},
        )
        frame_reports.append(
            {
                "frame": frame,
                "anchor_camera_id": anchor_camera_id,
                "groups": [list(group) for group in groups],
                "geometry_input_check": input_check,
                "anchor_duplicate": duplicate_report,
                "geometry_executed": True,
                "geometry": geometry,
            }
        )
        frame_bin_sets.append((frame, supported_bins))

    report_path = _expected_path(entry, "phase9-anchor-abstention-pilot-v1")
    report = {
        "schema_version": "phase9-anchor-abstention-pilot-v1",
        "run_id": entry["run_id"],
        "scene": expected_scene,
        "frames": list(frames),
        "methodology_status": "exploratory_per_pixel_duplicate_abstention",
        "x01_terminal_sha256": sha256_file(
            _expand(
                next(
                    item["path"]
                    for item in execution["input_artifacts"]
                    if item.get("producer_run_id")
                    == "P9-V2-X01-CUT-FAST-VISIBILITY-S20260717"
                )
            )
        ),
        "x01_geometry_admitted": x01_terminal["scientific_payload"][
            "geometry_admitted"
        ],
        "coordinate_admitted": coordinate_admitted,
        "frame_reports": frame_reports,
        "temporal_bin_transitions": temporal_bin_transitions(frame_bin_sets),
        "temporal_interpretation": "target-bin occupancy proxy, not surface-track reveal/hide evidence",
        "duplicate_semantics": {
            "global_x01_max_rule_preserved_as_negative_conformance": True,
            "x02_rule": "per-pixel two-sample relative half-difference >0.05 abstains; retained pixels use confidence-weighted aggregate",
            "threshold_changed": False,
            "heldout_scale_diagnostic_applied_to_geometry": False,
        },
        "label_dependent_gate_a": "not_evaluable",
        "cam00_rgb_opened": False,
        "split_binding": split_binding,
        "a02_authority_path": str(authority_path),
        "a02_authority_sha256": sha256_file(authority_path),
        "a02_terminal_sha256": a02_terminal_sha,
        "model_authority": model_authority,
    }
    write_json_atomic(report_path, report)
    reference = _scientific_file_ref(
        report_path, "phase9-anchor-abstention-pilot-v1", entry["run_id"]
    )
    payload = {
        "coordinate_admitted": coordinate_admitted,
        "frame_report_count": len(
            [item for item in frame_reports if item.get("geometry_executed")]
        ),
        "anchor_retained_fractions": [
            item["anchor_duplicate"]["retained_fraction"]
            for item in frame_reports
            if item.get("geometry_executed")
        ],
        "supported_bin_counts": [
            item["geometry"]["target_supported_bin_count"]
            for item in frame_reports
            if item.get("geometry_executed")
        ],
        "ordered_multilayer_bin_counts": [
            item["geometry"]["target_ordered_multilayer_bin_count"]
            for item in frame_reports
            if item.get("geometry_executed")
        ],
        "label_dependent_gate_a": "not_evaluable",
    }
    return [reference], payload


def _require_completed_x02(execution: dict[str, Any] | None) -> Mapping[str, Any]:
    if execution is None:
        raise ProvenanceError("opportunity mining requires a resolved execution manifest")
    matches = [
        item
        for item in execution.get("input_artifacts", [])
        if item.get("producer_run_id")
        == "P9-V3-X02-CUT-ANCHOR-ABSTENTION-S20260717"
        and item.get("schema") == "phase9-terminal-manifest-v1"
        and str(item.get("status", "")).startswith("resolved_exact")
    ]
    if len(matches) != 1:
        raise ProvenanceError("opportunity mining lacks one exact successful X02 terminal input")
    terminal = _json(_expand(str(matches[0]["path"])))
    payload = terminal.get("scientific_payload", {})
    if (
        terminal.get("schema_version") != "phase9-terminal-manifest-v1"
        or terminal.get("run_id")
        != "P9-V3-X02-CUT-ANCHOR-ABSTENTION-S20260717"
        or terminal.get("action") != "anchor-abstention-pilot"
        or terminal.get("status") != "completed"
        or terminal.get("exit_code") != 0
        or payload.get("coordinate_admitted") is not True
        or payload.get("ordered_multilayer_bin_counts") != [0, 0, 0]
    ):
        raise ProvenanceError("X02 terminal does not bind the expected zero-layer result")
    return terminal


def _aggregate_layer_opportunities(
    frame_reports: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    stage_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    bin_camera_histogram: Counter[str] = Counter()
    raw_layer_histogram: Counter[str] = Counter()
    accepted_layer_histogram: Counter[str] = Counter()
    ordered_counts = []
    supported_counts = []
    geometry_frames = 0
    coordinate_failed_frames = 0
    for report in frame_reports:
        if not report.get("geometry_executed"):
            coordinate_failed_frames += 1
            continue
        geometry_frames += 1
        geometry = report["geometry"]
        ordered_counts.append(int(geometry["target_ordered_multilayer_bin_count"]))
        supported_counts.append(int(geometry["target_supported_bin_count"]))
        opportunity = geometry.get("target_layer_opportunity", {})
        for item in opportunity.get("candidate_count_waterfall", []):
            stage_counts[str(item["stage"])] += int(item["count"])
        for key, value in opportunity.get("ordered_layer_rejection_counts", {}).items():
            rejection_counts[str(key)] += int(value)
        for key, value in opportunity.get("bin_camera_count_histogram", {}).items():
            bin_camera_histogram[str(key)] += int(value)
        for key, value in opportunity.get("raw_depth_layer_count_histogram", {}).items():
            raw_layer_histogram[str(key)] += int(value)
        for key, value in opportunity.get("accepted_depth_layer_count_histogram", {}).items():
            accepted_layer_histogram[str(key)] += int(value)
    return {
        "frame_count": len(frame_reports),
        "geometry_frame_count": geometry_frames,
        "coordinate_failed_frame_count": coordinate_failed_frames,
        "frames_with_ordered_layers": int(sum(1 for value in ordered_counts if value > 0)),
        "frames_with_supported_bins": int(sum(1 for value in supported_counts if value > 0)),
        "total_supported_bins": int(sum(supported_counts)),
        "total_ordered_multilayer_bins": int(sum(ordered_counts)),
        "maximum_supported_bins_per_frame": int(max(supported_counts) if supported_counts else 0),
        "maximum_ordered_multilayer_bins_per_frame": int(max(ordered_counts) if ordered_counts else 0),
        "candidate_count_waterfall": [
            {"stage": key, "count": int(stage_counts[key])}
            for key in (
                "projected_target_bins",
                "minimum_camera_bins",
                "two_raw_depth_cluster_bins",
                "two_min_camera_layer_bins",
                "ordered_multilayer_bins",
            )
        ],
        "ordered_layer_rejection_counts": {
            key: int(value) for key, value in sorted(rejection_counts.items())
        },
        "bin_camera_count_histogram": {
            key: int(value) for key, value in sorted(bin_camera_histogram.items())
        },
        "raw_depth_layer_count_histogram": {
            key: int(value) for key, value in sorted(raw_layer_histogram.items())
        },
        "accepted_depth_layer_count_histogram": {
            key: int(value) for key, value in sorted(accepted_layer_histogram.items())
        },
        "interpretation": "full-cut aggregate opportunity mining; labels and cam00 RGB are not consumed",
    }


def action_cut_opportunity_mining(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the X02 abstention rule across all cut frames and mine layer yield."""

    config = _json(DEFAULT_CONFIG)
    expected_scene = config["data"]["development_scene"]
    if args.scene != expected_scene or expected_scene != "cut_roasted_beef":
        raise ProvenanceError(f"cut opportunity mining scene must be {expected_scene}")
    x02_terminal = _require_completed_x02(execution)
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
    split_binding = {
        key: dict(value)
        for key, value in validate_split_binding(index, split_manifest).items()
    }
    authority_path, authority, a02_terminal_sha = _bound_a02_authority(
        execution, Path(args.matrix).resolve()
    )
    model_authority = verify_model_authority(
        model_dir,
        expected_weight_sha256=authority["weight_sha256"],
        hash_weights=True,
    )
    _seed_conformance(int(entry["seeds"]["training"]))
    model = load_da3(da3_checkout, model_dir, device="cuda")

    test_records = index.by_camera_frame("test")
    train_records = index.split("train")
    frames = sorted({record.frame for record in train_records})
    r_scene = compute_r_scene(train_records)
    frame_reports: list[dict[str, Any]] = []
    frame_bin_sets: list[tuple[int, set[tuple[int, int]]]] = []
    relative_limit = float(config["fusion"]["duplicate_relative_mad_maximum"])
    for frame in frames:
        frame_records = [record for record in train_records if record.frame == frame]
        anchor_camera_id, groups = _select_conformance_groups(
            frame_records, r_scene, config
        )
        group_inputs = _build_real_group_inputs(frame_records, groups)
        predictions = [
            run_group(
                model,
                group["images"],
                group["extrinsics_w2c"],
                group["intrinsics"],
            )
            for group in group_inputs
        ]
        input_check = dict(
            _geometry_input_check(
                predictions,
                group_inputs,
                anchor_camera_id=anchor_camera_id,
            )
        )
        anchor_indices = [
            group["member_camera_ids"].index(anchor_camera_id)
            for group in group_inputs
        ]
        first_index, second_index = anchor_indices
        first_prediction, second_prediction = predictions
        extrinsic_max_abs = float(
            np.max(
                np.abs(
                    np.asarray(first_prediction["extrinsics"][first_index])
                    - np.asarray(second_prediction["extrinsics"][second_index])
                )
            )
        )
        input_check["anchor_extrinsic_maximum_absolute_difference"] = extrinsic_max_abs
        coordinate_admitted = (
            input_check["processed_k_corner_error_maximum_pixels"] <= 0.5
            and extrinsic_max_abs
            <= float(config["da3"]["conformance_repeat_atol"])
        )
        input_check["coordinate_admitted"] = coordinate_admitted
        if not coordinate_admitted:
            frame_reports.append(
                {
                    "frame": frame,
                    "anchor_camera_id": anchor_camera_id,
                    "groups": [list(group) for group in groups],
                    "geometry_input_check": input_check,
                    "geometry_executed": False,
                    "blocked_reason": "coordinate_not_admitted",
                }
            )
            continue

        duplicate_report, aggregate_depth, retained_mask = anchor_duplicate_diagnostic(
            first_prediction["depth"][first_index],
            second_prediction["depth"][second_index],
            first_prediction["confidence"][first_index],
            second_prediction["confidence"][second_index],
            relative_limit=relative_limit,
            calibration_stride=int(config["sampling"]["grid_stride_pixels"]),
        )
        target = test_records.get((config["data"]["test_camera"], frame))
        if target is None or target.image_path is not None:
            raise ProvenanceError("opportunity mining requires calibration-only cam00 target access")
        geometry, supported_bins = evaluate_frame_geometry(
            predictions,
            [item["member_camera_ids"] for item in group_inputs],
            target_K=target.K,
            target_w2c=target.w2c_opencv,
            target_width=target.width,
            target_height=target.height,
            stride=int(config["sampling"]["grid_stride_pixels"]),
            minimum_cameras=int(config["fusion"]["minimum_physical_cameras"]),
            maximum_depth_sigma=float(config["fusion"]["maximum_depth_sigma"]),
            target_bin_pixels=int(config["sampling"]["grid_stride_pixels"]),
            camera_depth_overrides={anchor_camera_id: aggregate_depth},
            camera_valid_masks={anchor_camera_id: retained_mask},
        )
        frame_reports.append(
            {
                "frame": frame,
                "anchor_camera_id": anchor_camera_id,
                "groups": [list(group) for group in groups],
                "geometry_input_check": input_check,
                "anchor_duplicate": duplicate_report,
                "geometry_executed": True,
                "geometry": geometry,
            }
        )
        frame_bin_sets.append((frame, supported_bins))

    aggregate = _aggregate_layer_opportunities(frame_reports)
    report_path = _expected_path(entry, "phase9-cut-opportunity-mining-v1")
    report = {
        "schema_version": "phase9-cut-opportunity-mining-v1",
        "run_id": entry["run_id"],
        "scene": expected_scene,
        "frames": frames,
        "methodology_status": "full_cut_layer_opportunity_mining_after_x02",
        "x02_terminal_sha256": sha256_file(
            _expand(
                next(
                    item["path"]
                    for item in execution["input_artifacts"]
                    if item.get("producer_run_id")
                    == "P9-V3-X02-CUT-ANCHOR-ABSTENTION-S20260717"
                )
            )
        ),
        "x02_scientific_payload": x02_terminal["scientific_payload"],
        "frame_reports": frame_reports,
        "aggregate_layer_opportunity": aggregate,
        "temporal_bin_transitions": temporal_bin_transitions(frame_bin_sets),
        "temporal_interpretation": "target-bin occupancy proxy, not surface-track reveal/hide evidence",
        "duplicate_semantics": {
            "x02_rule_reused": "per-pixel two-sample relative half-difference >0.05 abstains; retained pixels use confidence-weighted aggregate",
            "threshold_changed": False,
            "heldout_scale_diagnostic_applied_to_geometry": False,
        },
        "label_dependent_gate_a": "not_evaluable",
        "cam00_rgb_opened": False,
        "split_binding": split_binding,
        "a02_authority_path": str(authority_path),
        "a02_authority_sha256": sha256_file(authority_path),
        "a02_terminal_sha256": a02_terminal_sha,
        "model_authority": model_authority,
    }
    write_json_atomic(report_path, report)
    reference = _scientific_file_ref(
        report_path, "phase9-cut-opportunity-mining-v1", entry["run_id"]
    )
    payload = {
        "frame_count": aggregate["frame_count"],
        "geometry_frame_count": aggregate["geometry_frame_count"],
        "frames_with_ordered_layers": aggregate["frames_with_ordered_layers"],
        "total_supported_bins": aggregate["total_supported_bins"],
        "total_ordered_multilayer_bins": aggregate["total_ordered_multilayer_bins"],
        "candidate_count_waterfall": aggregate["candidate_count_waterfall"],
        "ordered_layer_rejection_counts": aggregate["ordered_layer_rejection_counts"],
        "label_dependent_gate_a": "not_evaluable",
    }
    return [reference], payload



def _completed_terminal_inputs(execution: dict[str, Any] | None) -> list[dict[str, Any]]:
    if execution is None:
        raise ProvenanceError("action requires a resolved execution manifest")
    terminals = []
    for reference in execution.get("input_artifacts", []):
        if reference.get("schema") != "phase9-terminal-manifest-v1":
            continue
        if not str(reference.get("status", "")).startswith("resolved_exact"):
            continue
        path = _expand(str(reference.get("path", ""))).resolve()
        expected_sha = str(reference.get("sha256") or "")
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ProvenanceError(f"terminal input bytes do not match resolved binding: {path}")
        terminal = _json(path)
        if (
            terminal.get("schema_version") != "phase9-terminal-manifest-v1"
            or terminal.get("status") != "completed"
            or terminal.get("exit_code") != 0
        ):
            raise ProvenanceError(f"terminal input is not a successful completed run: {path}")
        terminals.append({"path": path, "sha256": expected_sha, "terminal": terminal})
    return terminals


def _bound_input_json_artifact(
    execution: dict[str, Any] | None,
    *,
    schema: str,
    producer_run_id: str | None = None,
    required: bool = True,
) -> tuple[Path, dict[str, Any], str, dict[str, Any]] | None:
    matches = []
    for terminal_record in _completed_terminal_inputs(execution):
        terminal = terminal_record["terminal"]
        for artifact in terminal.get("produced_artifacts", []):
            if artifact.get("schema") != schema:
                continue
            if producer_run_id is not None and artifact.get("producer_run_id") != producer_run_id:
                continue
            path = _expand(str(artifact.get("path", ""))).resolve()
            if not path.is_file():
                raise ProvenanceError(f"bound input artifact is missing: {path}")
            actual_sha = sha256_file(path)
            if artifact.get("sha256") != actual_sha:
                raise ProvenanceError(f"bound input artifact bytes changed: {path}")
            payload = _json(path)
            if payload.get("schema_version") != schema:
                raise ProvenanceError(f"bound input artifact has wrong schema: {path}")
            matches.append((path, payload, actual_sha, terminal_record))
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    raise ProvenanceError(f"expected exactly one bound input artifact for {schema}, got {len(matches)}")


def _maybe_expected_path(entry: dict[str, Any], schema: str) -> Path | None:
    matches = [item for item in entry["expected_outputs"] if item["schema"] == schema]
    if not matches:
        return None
    if len(matches) != 1:
        raise ContractError(f"{entry['run_id']} expects {len(matches)} outputs with schema {schema}")
    return _expand(matches[0]["path"])


def _external_json_input(
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
    *,
    role: str,
) -> tuple[Path, dict[str, Any], str]:
    candidates: list[dict[str, Any]] = []
    if args.input:
        candidates.append({"path": args.input, "sha256": None, "role": role})
    if execution is not None:
        for item in execution.get("external_inputs", []):
            if item.get("role") == role and item.get("path"):
                candidates.append(dict(item))
    if len(candidates) != 1:
        raise ProvenanceError(f"{role} requires exactly one external JSON input, got {len(candidates)}")
    reference = candidates[0]
    path = _expand(str(reference["path"])).resolve()
    if not path.is_file():
        raise ProvenanceError(f"external JSON input is missing: {path}")
    actual_sha = sha256_file(path)
    expected_sha = reference.get("sha256")
    if expected_sha and actual_sha != expected_sha:
        raise ProvenanceError(f"external JSON input hash mismatch: {path}")
    return path, _json(path), actual_sha


def _write_bound_copy(
    destination: Path,
    payload: dict[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    run_id: str,
) -> dict[str, Any]:
    sealed = {
        **payload,
        "freeze_run_id": run_id,
        "source_artifact_path": str(source_path),
        "source_artifact_sha256": source_sha256,
    }
    write_json_atomic(destination, sealed)
    return sealed


def action_freeze_train_sidecars(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ledger_path, ledger, ledger_sha, terminal_record = _bound_input_json_artifact(
        execution,
        schema="phase9-csvl-ledger-v1",
    )
    scene = str(entry.get("scene") or args.scene or ledger.get("scene") or "")
    if scene != ledger.get("scene"):
        raise ProvenanceError("train sidecar scene does not match the CSVL ledger")
    if ledger.get("cam00_rgb_opened") is not False:
        raise ProvenanceError("freeze-train-sidecars rejects any source that opened cam00 RGB")
    if ledger.get("label_dependent_gate_a") != "not_evaluable":
        raise ProvenanceError("freeze-train-sidecars rejects label-dependent Gate A fields")
    evidence = ledger.get("evidence_boundary", {})
    if evidence.get("human_labels") != "not_consumed":
        raise ProvenanceError("freeze-train-sidecars rejects human label inputs")
    if "r009" in json.dumps(ledger, sort_keys=True).lower():
        raise ProvenanceError("freeze-train-sidecars rejects R009-derived fields")
    aggregate = ledger.get("aggregate_layer_opportunity", {})
    destination = _expected_path(entry, "phase9-train-sidecars-v1")
    payload = {
        "schema_version": "phase9-train-sidecars-v1",
        "run_id": entry["run_id"],
        "scene": scene,
        "sidecar_type": "generic_capacity_only_event_blind_v1",
        "admitted_training_modes": ["capacity-only", "null-reset"],
        "unsupported_training_modes": ["oracle-capacity", "visibility-only", "full", "shuffled"],
        "cam00_rgb_opened": False,
        "label_dependent_gate_a": "not_evaluable",
        "read_boundary": {
            "csvl_ledger": "label_free_p03_v9_only",
            "cam00_rgb": "forbidden",
            "human_labels": "forbidden",
            "evaluator_masks": "forbidden",
            "r009_fields": "forbidden",
        },
        "capacity_policy": {
            "donor_selection": "event_blind_low_opacity_redundant_old_slot_v1",
            "target_construction": "event_blind_existing_dynamic_row_clone_v1",
            "point_neutral": True,
            "point_ceiling": int(_json(DEFAULT_CONFIG)["representation"]["point_ceiling"]),
        },
        "csvl_binding": {
            "path": str(ledger_path),
            "schema": "phase9-csvl-ledger-v1",
            "producer_run_id": terminal_record["terminal"]["run_id"],
            "sha256": ledger_sha,
        },
        "csvl_summary": {
            "frame_count": int(ledger.get("frame_count", 0)),
            "geometry_frame_count": int(ledger.get("geometry_frame_count", 0)),
            "frames_with_ordered_layers": int(aggregate.get("frames_with_ordered_layers", 0)),
            "total_ordered_multilayer_bins": int(aggregate.get("total_ordered_multilayer_bins", 0)),
            "emitted_ordered_multilayer_bin_hypothesis_count": int(aggregate.get("emitted_ordered_multilayer_bin_hypothesis_count", 0)),
        },
    }
    write_json_atomic(destination, payload)
    return [_scientific_file_ref(destination, "phase9-train-sidecars-v1", entry["run_id"])], payload


def action_freeze_human_labels(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path, artifact, source_sha = _external_json_input(args, execution, role="human_annotation_return")
    window_ref = artifact.get("source_window_manifest", {})
    window_path = _expand(str(window_ref.get("path", DEFAULT_WINDOWS))).resolve()
    if not window_path.is_file():
        raise ProvenanceError("completed annotation-window manifest is missing")
    expected_window_sha = window_ref.get("sha256")
    if expected_window_sha and sha256_file(window_path) != expected_window_sha:
        raise ProvenanceError("completed annotation-window manifest hash mismatch")
    windows_manifest = load_json(window_path)
    audit = validate_human_label_freeze(artifact, windows_manifest)
    annotators = artifact.get("annotator_records")
    adjudication = artifact.get("adjudication_record")
    if not isinstance(annotators, list) or len(annotators) != 2:
        raise ContractError("freeze-human-labels requires exactly two annotator records")
    annotator_ids = [str(item.get("annotator_id")) for item in annotators]
    if len(set(annotator_ids)) != 2 or any(value in {"", "None"} for value in annotator_ids):
        raise ContractError("freeze-human-labels requires distinct annotator IDs")
    if not isinstance(adjudication, Mapping) or adjudication.get("status") != "completed":
        raise ContractError("freeze-human-labels requires completed adjudication")
    scene = str(entry.get("scene") or args.scene or artifact.get("scene") or "")
    if scene and artifact.get("scene") not in {None, scene}:
        raise ProvenanceError("human label scene mismatch")
    destination = _expected_path(entry, "phase9-human-label-freeze-v1")
    sealed = _write_bound_copy(destination, dict(artifact), source_path=source_path, source_sha256=source_sha, run_id=entry["run_id"])
    refs = [_scientific_file_ref(destination, "phase9-human-label-freeze-v1", entry["run_id"])]
    return refs, {"scene": scene, "row_counts": audit["row_counts"], "evidence_type": sealed["evidence_type"], "labels_complete": True}


def action_run_monocular_baselines(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_baseline_registry(_json(DEFAULT_BASELINES))
    source_path, artifact, source_sha = _external_json_input(args, execution, role="monocular_baseline_predictions")
    if artifact.get("schema_version") != "phase9-r031-family-predictions-v1":
        raise SchemaError("wrong monocular baseline prediction schema")
    scene = str(entry.get("scene") or args.scene or "")
    if artifact.get("scene") != scene:
        raise ProvenanceError("monocular baseline scene mismatch")
    if artifact.get("labels_consumed") is not False:
        raise ProvenanceError("monocular baselines must be frozen before label consumption")
    destination = _expected_path(entry, "phase9-r031-family-predictions-v1")
    sealed = _write_bound_copy(destination, dict(artifact), source_path=source_path, source_sha256=source_sha, run_id=entry["run_id"])
    return [_scientific_file_ref(destination, "phase9-r031-family-predictions-v1", entry["run_id"])], {"scene": scene, "baseline_family_count": len(sealed.get("families", [])), "labels_consumed": False}


def action_score_gate_a(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ledger_ref = _bound_input_json_artifact(execution, schema="phase9-csvl-ledger-v1")
    label_ref = _bound_input_json_artifact(execution, schema="phase9-human-label-freeze-v1")
    baseline_ref = _bound_input_json_artifact(execution, schema="phase9-r031-family-predictions-v1")
    source_path, score, source_sha = _external_json_input(args, execution, role="gate_a_score_units")
    if score.get("schema_version") != "phase9-gate-a-score-v1":
        raise SchemaError("wrong Gate A score schema")
    scene = str(entry.get("scene") or args.scene or "")
    if score.get("scene") != scene:
        raise ProvenanceError("Gate A score scene mismatch")
    if score.get("evidence_type") != "human_reference" or score.get("labels_complete") is not True:
        raise ProvenanceError("Gate A scoring requires complete human_reference labels")
    metrics = score.get("metrics", {})
    required = {
        "ordering_accuracy", "ordering_auroc", "ordering_coverage", "event_f1",
        "event_recall", "boundary_f1_delta", "region_iou_delta",
        "cross_view_inconsistency_relative_reduction",
        "temporal_inconsistency_relative_reduction", "ordering_ece", "transition_ece",
        "evaluable_track_fraction",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ContractError(f"Gate A score is missing required engineering metrics: {missing}")
    config = _json(DEFAULT_CONFIG)
    engineering = decide_gate_a(score, config, tier="engineering")
    claim_grade = decide_gate_a(score, config, tier="claim_grade")
    destination = _expected_path(entry, "phase9-gate-a-score-v1")
    sealed = {
        **dict(score),
        "input_bindings": {
            "csvl_ledger": {"path": str(ledger_ref[0]), "sha256": ledger_ref[2]},
            "human_labels": {"path": str(label_ref[0]), "sha256": label_ref[2]},
            "monocular_baselines": {"path": str(baseline_ref[0]), "sha256": baseline_ref[2]},
            "score_units": {"path": str(source_path), "sha256": source_sha},
        },
        "decisions": {"engineering": engineering, "claim_grade": claim_grade},
    }
    write_json_atomic(destination, sealed)
    refs = [_scientific_file_ref(destination, "phase9-gate-a-score-v1", entry["run_id"])]
    calibrator_path = _maybe_expected_path(entry, "phase9-gate-a-calibrator-v1")
    if calibrator_path is not None:
        calibrator = {
            "schema_version": "phase9-gate-a-calibrator-v1",
            "run_id": entry["run_id"],
            "scene": scene,
            "source_score_sha256": sha256_file(destination),
            "calibration_status": "frozen_from_external_score_units",
            "engineering_decision_status": engineering["status"],
        }
        write_json_atomic(calibrator_path, calibrator)
        refs.append(_scientific_file_ref(calibrator_path, "phase9-gate-a-calibrator-v1", entry["run_id"]))
    return refs, {"scene": scene, "engineering_status": engineering["status"], "claim_grade_status": claim_grade["status"], "metrics": metrics}


def action_freeze_evaluator(
    entry: dict[str, Any],
    args: argparse.Namespace,
    execution: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flow_ref = _bound_input_json_artifact(execution, schema="phase9-flow-manifest-v1")
    scene = str(entry.get("scene") or args.scene or flow_ref[1].get("scene") or "")
    if flow_ref[1].get("scene") != scene:
        raise ProvenanceError("evaluator flow scene mismatch")
    label_ref = _bound_input_json_artifact(execution, schema="phase9-human-label-freeze-v1", required=False)
    if scene in {"cut_roasted_beef", "flame_steak", "sear_steak"} and label_ref is None:
        raise ProvenanceError("annotation scenes require frozen human labels before evaluator freeze")
    destination = _expected_path(entry, "phase9-evaluator-freeze-v1")
    payload = {
        "schema_version": "phase9-evaluator-freeze-v1",
        "run_id": entry["run_id"],
        "scene": scene,
        "cam00_rgb_opened": False,
        "outcome_renders_consumed": False,
        "formulas": {
            "event_psnr": "pooled_masked_psnr_v1",
            "static_admission": "static_scene_admission_v1",
            "flow_relative_flicker": "flow_relative_flicker_v1",
            "reveal_ghost": "reveal_ghost_v1",
        },
        "input_bindings": {
            "flow_manifest": {"path": str(flow_ref[0]), "sha256": flow_ref[2]},
            "human_labels": None if label_ref is None else {"path": str(label_ref[0]), "sha256": label_ref[2]},
        },
        "label_status": "human_reference_frozen" if label_ref is not None else "label_free_static_only",
    }
    write_json_atomic(destination, payload)
    return [_scientific_file_ref(destination, "phase9-evaluator-freeze-v1", entry["run_id"])], {"scene": scene, "label_status": payload["label_status"], "outcome_renders_consumed": False}


def action_operator_gpu_smoke(entry: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        raise ContractError("operator-gpu-smoke requires CUDA and must run under a GPU Slurm allocation")
    device = torch.device("cuda")
    n = 16
    parameters = {
        "_xyz": nn.Parameter(torch.arange(n * 3, dtype=torch.float32, device=device).reshape(n, 3) / 100.0),
        "_scaling": nn.Parameter(torch.log(torch.full((n, 3), 0.10, dtype=torch.float32, device=device))),
        "_opacity": nn.Parameter(torch.linspace(-5.0, 2.0, n, device=device).reshape(n, 1)),
    }
    accumulators = {"denom": torch.ones(n, 1, device=device), "xyz_gradient_accum": torch.ones(n, 1, device=device)}
    bank = CapacityBank(
        parameters=parameters,
        accumulators=accumulators,
        stable_ids=torch.arange(1000, 1000 + n, dtype=torch.long, device=device),
        generation=torch.zeros(n, dtype=torch.long, device=device),
        last_reassigned=torch.zeros(n, dtype=torch.long, device=device),
    )
    optimizer = torch.optim.Adam(list(parameters.values()), lr=0.01, amsgrad=True)
    loss = sum(parameter.square().sum() for parameter in parameters.values())
    loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    selected = select_event_blind_donors(
        xyz=parameters["_xyz"].detach(),
        scaling_log=parameters["_scaling"].detach(),
        opacity_logit=parameters["_opacity"].detach(),
        denom=accumulators["denom"].detach(),
        generation=bank.generation.detach(),
        stable_ids=bank.stable_ids.detach(),
        current_iteration=5001,
        k=1,
    )
    if selected.get("abstained"):
        raise ContractError(f"GPU donor selection abstained: {selected}")
    donors = torch.as_tensor(selected["selected_indices"], dtype=torch.long, device=device)
    targets, target_meta = build_event_blind_capacity_targets(bank, donors, seed=0, iteration=5001)
    transaction = apply_point_neutral_transaction(bank, optimizer, donors, targets, iteration=5001, mode="reassign")
    payload = {"device": str(device), "donor_selection": selected, "target_metadata": target_meta, "transaction": transaction, "finite": True}
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
    matrix_path = Path(args.matrix).resolve()
    matrix_sha = sha256_file(matrix_path)
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

    matrix = _json(matrix_path)
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


def action_operator_static(entry: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import unittest

    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "tests"),
        pattern="test_depth_visibility_capacity.py",
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise ContractError(
            f"operator static suite failed: failures={len(result.failures)} errors={len(result.errors)}"
        )
    payload = {
        "status": "pass",
        "test_count": int(result.testsRun),
        "covered_invariants": [
            "dynamic_plus_hard_static_budget_accounting",
            "optimizer_parameter_identity_preserved",
            "donor_rows_rewritten_in_place",
            "survivor_rows_bitwise_preserved",
            "adam_moment_rows_zeroed",
            "null_reset_value_noop_moment_surgery",
            "event_blind_low_opacity_redundant_donor_selection",
        ],
        "slice_b_scope": "B00 CPU static operator fixture only; no trainer mutation or Gate B claim",
    }
    return [], payload



def _phase9_train_base_config() -> Path:
    return REPO_ROOT / "configs/n3v/fixed_budget_lora_route0_filemask_residual_600k.yaml"


def _json_file_ref_payload(path: Path, schema: str, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    write_json_atomic(path, payload)
    return _scientific_file_ref(path, schema, run_id)


def action_train(entry: dict[str, Any], args: argparse.Namespace, execution: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from omegaconf import OmegaConf

    training = entry.get("training")
    if not isinstance(training, dict):
        raise ContractError("train action requires a registered training block")
    scene = args.scene or entry.get("scene")
    if not scene:
        raise ContractError("train action requires a scene")
    if entry.get("scene") and scene != entry["scene"]:
        raise ProvenanceError(f"scene mismatch: matrix={entry['scene']} argv={scene}")

    base_config = _phase9_train_base_config()
    if not base_config.is_file():
        raise ProvenanceError(f"missing Slice B base training config: {base_config}")
    output_root = _output_root(entry)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _expected_path(entry, "adags-checkpoint-v1")
    model_path = checkpoint_path.parent
    model_path.mkdir(parents=True, exist_ok=True)
    metrics_path = _expected_path(entry, "phase9-training-metrics-v1")
    capacity_path = _expected_path(entry, "phase9-capacity-ledger-v1")
    provenance_path = _expected_path(entry, "phase9-provenance-v1")
    renders_path = _expected_path(entry, "phase9-render-inventory-v1")

    cfg = OmegaConf.load(str(base_config))
    end_iteration = int(training["end_iteration"])
    topology_cutoff = int(training.get("topology_cutoff_iteration", end_iteration))
    point_ceiling = int(training.get("point_ceiling", 600000))
    mode = str(training.get("mode", "common"))
    cfg.OptimizationParams.iterations = end_iteration
    cfg.OptimizationParams.position_lr_max_steps = end_iteration
    cfg.OptimizationParams.densify_until_iter = min(topology_cutoff, end_iteration)
    cfg.OptimizationParams.densify_until_num_points = point_ceiling
    cfg.OptimizationParams.enable_hard_static_conversion = False
    cfg.OptimizationParams.slice_b_capacity_mode = "disabled" if mode == "common" else mode
    cfg.OptimizationParams.slice_b_capacity_iteration = int(training.get("start_iteration") or 5001)
    cfg.OptimizationParams.slice_b_capacity_k = int(training.get("requested_k") or 0)
    cfg.OptimizationParams.slice_b_capacity_seed = int(entry.get("seeds", {}).get("capacity") or 0)
    sidecar_ref = _bound_input_json_artifact(execution, schema="phase9-train-sidecars-v1", required=False)
    if sidecar_ref is not None:
        cfg.OptimizationParams.slice_b_capacity_sidecar = str(sidecar_ref[0])
    elif mode not in {"common", "route0"}:
        raise ProvenanceError(f"training mode {mode} requires a frozen train sidecar")

    derived_config = output_root / "train-config.yaml"
    OmegaConf.save(cfg, str(derived_config))
    dataset_root = Path(os.environ.get("ADAGS_PROJECT_ROOT", os.environ.get("WORK", "") + "/proj_adags")) / "data/n3v"
    dataset_path = dataset_root / scene
    if not dataset_path.is_dir():
        raise ProvenanceError(f"missing N3V scene directory: {dataset_path}")
    command = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--config",
        str(derived_config),
        "--model_path",
        str(model_path),
        "--source_path",
        str(dataset_path),
        "--seed",
        str(int(entry.get("seeds", {}).get("training") or 0)),
        "--test_iterations",
        str(end_iteration),
        "--save_iterations",
        str(end_iteration),
        "--wandb_mode",
        "disabled",
        "--experiment_name",
        "phase9_csvl_isr_v1",
        "--method_family",
        f"slice_b_{mode}",
        "--budget_label",
        f"phase9_{point_ceiling}",
    ]
    input_checkpoint = training.get("input_checkpoint") or {}
    if input_checkpoint.get("path"):
        start_checkpoint = _expand(input_checkpoint["path"])
        if not start_checkpoint.is_file():
            raise ProvenanceError(f"missing input checkpoint: {start_checkpoint}")
        command.extend(["--start_checkpoint", str(start_checkpoint)])

    completed = subprocess.run(command, cwd=str(REPO_ROOT), text=True)
    if completed.returncode != 0:
        raise ContractError(f"main.py training failed with exit code {completed.returncode}")
    if not checkpoint_path.is_file():
        raise ContractError(f"expected checkpoint was not produced: {checkpoint_path}")
    summary_path = model_path / "summary.json"
    if not summary_path.is_file():
        raise ContractError(f"expected local summary was not produced: {summary_path}")
    model_capacity_path = model_path / "capacity-ledger.json"
    if not model_capacity_path.is_file():
        raise ContractError(f"expected capacity ledger was not produced: {model_capacity_path}")

    summary = _json(summary_path)
    capacity_payload = _json(model_capacity_path)
    capacity_payload.update({"run_id": entry["run_id"], "source_path": str(model_capacity_path)})
    metrics_payload = {
        "schema_version": "phase9-training-metrics-v1",
        "run_id": entry["run_id"],
        "scene": scene,
        "training": training,
        "summary_path": str(summary_path),
        "summary": summary.get("summary", {}),
        "checkpoint_path": str(checkpoint_path),
    }
    render_files = []
    for suffix in ("*.png", "*.jpg", "*.jpeg"):
        for path in sorted(model_path.rglob(suffix)):
            render_files.append({
                "path": str(path),
                "relative_path": str(path.relative_to(model_path)),
                "sha256": sha256_file(path),
            })
    renders_payload = {
        "schema_version": "phase9-render-inventory-v1",
        "run_id": entry["run_id"],
        "scene": scene,
        "model_path": str(model_path),
        "render_count": len(render_files),
        "renders": render_files,
    }
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--short"], cwd=str(REPO_ROOT), text=True).splitlines()
    provenance_payload = {
        "schema_version": "phase9-provenance-v1",
        "run_id": entry["run_id"],
        "scene": scene,
        "action": "train",
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "base_config_path": str(base_config),
        "base_config_sha256": sha256_file(base_config),
        "derived_config_path": str(derived_config),
        "derived_config_sha256": sha256_file(derived_config),
        "command": command,
        "execution_manifest_sha256": sha256_file(_expand(args.execution_manifest)) if args.execution_manifest else None,
    }
    produced = [
        _scientific_file_ref(checkpoint_path, "adags-checkpoint-v1", entry["run_id"]),
        _json_file_ref_payload(metrics_path, "phase9-training-metrics-v1", entry["run_id"], metrics_payload),
        _json_file_ref_payload(capacity_path, "phase9-capacity-ledger-v1", entry["run_id"], capacity_payload),
        _json_file_ref_payload(provenance_path, "phase9-provenance-v1", entry["run_id"], provenance_payload),
        _json_file_ref_payload(renders_path, "phase9-render-inventory-v1", entry["run_id"], renders_payload),
    ]
    payload = {
        "schema_version": "phase9-train-terminal-payload-v1",
        "run_id": entry["run_id"],
        "scene": scene,
        "mode": mode,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "metrics_path": str(metrics_path),
        "capacity_ledger_path": str(capacity_path),
        "render_count": len(render_files),
        "summary": summary.get("summary", {}),
    }
    return produced, payload


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
        "produce-da3": lambda: action_produce_da3(entry, args, execution),
        "adapt-flow": lambda: action_adapt_flow(entry, args, execution),
        "build-csvl": lambda: action_build_csvl(entry, args, execution),
        "fast-visibility-pilot": lambda: action_fast_visibility_pilot(entry, args, execution),
        "anchor-abstention-pilot": lambda: action_anchor_abstention_pilot(entry, args, execution),
        "cut-opportunity-mining": lambda: action_cut_opportunity_mining(entry, args, execution),
        "build-annotation-packet": lambda: action_annotation_packet(entry, args),
        "freeze-human-labels": lambda: action_freeze_human_labels(entry, args, execution),
        "run-monocular-baselines": lambda: action_run_monocular_baselines(entry, args, execution),
        "score-gate-a": lambda: action_score_gate_a(entry, args, execution),
        "freeze-train-sidecars": lambda: action_freeze_train_sidecars(entry, args, execution),
        "freeze-evaluator": lambda: action_freeze_evaluator(entry, args, execution),
        "operator-static": lambda: action_operator_static(entry, args),
        "operator-gpu-smoke": lambda: action_operator_gpu_smoke(entry, args),
        "freeze-implementation": lambda: action_freeze_implementation(entry, args),
        "train": lambda: action_train(entry, args, execution),
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
