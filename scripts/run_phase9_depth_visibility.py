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
)
from depth_visibility.artifacts import build_inventory, write_canonical_array
from depth_visibility.baselines import validate_baseline_registry
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
        "fast-visibility-pilot": lambda: action_fast_visibility_pilot(entry, args, execution),
        "anchor-abstention-pilot": lambda: action_anchor_abstention_pilot(entry, args, execution),
        "cut-opportunity-mining": lambda: action_cut_opportunity_mining(entry, args, execution),
        "build-annotation-packet": lambda: action_annotation_packet(entry, args),
        "operator-static": lambda: action_operator_static(entry, args),
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
