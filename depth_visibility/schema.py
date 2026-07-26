"""Dependency-free, fail-closed validators for Phase 9 JSON payloads."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import CONFIG_SCHEMA_VERSION, METHOD_ID
from .canonical import sha256_file
from .errors import NonFiniteError, SchemaError


CONFIG_KEYS = {
    "annotation", "camera", "da3", "data", "evaluation", "frozen_before_phase9_outcomes",
    "fusion", "gate_a", "gate_b", "grouping", "method_id", "representation", "risk",
    "sampling", "schema_version", "scientific_authority", "surfaces", "temporal",
}
CONFIG_SECTION_KEYS = {
    "annotation": {
        "agreement_polygon_iou_minimum", "agreement_transition_tolerance_frames",
        "all_windows_double_assigned", "discovery_match_source_iou_minimum",
        "discovery_match_target_iou_minimum", "discovery_roles",
        "event_family_vocabulary", "human_fields_initially_empty", "one_sided_value",
        "two_stage_discovery_union_roster",
    },
    "camera": {
        "distortion_policy", "loader_adapter", "matrix_convention", "pixel_centers",
        "resize_sampling", "timestamp_tolerance_seconds",
    },
    "da3": {
        "align_to_input_ext_scale", "checkout_commit", "conformance_repeat_atol",
        "conformance_repeat_rtol", "depth_semantics", "first_observed_sha_required",
        "infer_gs", "model_id", "per_frame_normalization", "process_res",
        "process_res_method", "production_sidecar_immutable", "ref_view_strategy",
        "scale_shift_fit", "use_ray_pose", "weight_expected_bytes",
    },
    "data": {
        "annotation_windows", "development_scene", "final_scenes", "r009_use",
        "split_manifest", "test_camera", "test_source", "train_source", "transfer_scenes",
    },
    "evaluation": {
        "bootstrap_replicates", "bootstrap_seed", "boundary_primary_tolerance_pixels",
        "boundary_sensitivities_pixels", "color_space", "cycle_validity_pixels",
        "empty_frame_mask", "event_horizon_offsets", "event_macro_order",
        "exact_zero_mse_reported", "flow_direction", "lpips_backbone", "lpips_package",
        "lpips_version", "minimum_event_valid_frames", "psnr_data_range", "psnr_mse_floor",
        "reveal_ghost_requires_all_offsets", "sampling", "static_complement_erosion_pixels",
        "static_flicker_minimum_valid_pairs", "static_frame_population",
        "static_ghost_name_retired", "static_minimum_pixels_per_frame",
        "static_minimum_valid_frames", "static_motion_dilation_pixels",
        "static_primary_metrics", "zero_denominator_error_improvement",
        "zero_denominator_error_regression",
    },
    "fusion": {
        "covariance_inverse_clip", "duplicate_relative_mad_maximum", "maximum_depth_sigma",
        "maximum_pair_cost", "maximum_projection_residual_pixels", "merge_radius_rscene",
        "minimum_patch_ncc", "minimum_physical_cameras", "robust_mahalanobis_maximum",
    },
    "gate_a": {"claim_grade", "engineering"},
    "gate_b": {
        "event_lpips_relative_improvement_minimum", "event_psnr_delta_db_minimum",
        "event_strict_majority_both", "flicker_relative_regression_maximum",
        "full_vs_single_event_fraction_minimum", "full_vs_single_lpips_relative_improvement_minimum",
        "full_vs_single_psnr_delta_db_minimum", "reveal_ghost_relative_regression_maximum",
        "shuffle_gain_fraction_maximum", "static_lpips_relative_regression_maximum",
        "static_psnr_delta_db_minimum", "static_reconstruction_l1_relative_regression_maximum",
    },
    "grouping": {
        "maximum_cameras", "maximum_optical_axis_angle_degrees",
        "minimum_center_distance_rscene", "minimum_second_singular_value_rscene",
        "minimum_valid_groups_per_camera_time", "target_transitive_exclusion",
    },
    "representation": {
        "association_protection_radius_frames", "bottom_opacity_fraction",
        "capacity_seed_authority", "common_iteration", "comparable_iteration", "comparable_k",
        "minimum_slot_age_iterations", "opacity_initial", "optimizer_policy_source",
        "pilot_iteration", "pilot_k", "pixel_confidence", "point_budget_relative_tolerance",
        "point_ceiling", "rot_4d_required", "route_logit_source", "scaffold_required",
        "shuffle_algorithm", "shuffle_domain_separator_hex", "shuffle_map_domain_utf8",
        "shuffle_seed_offset", "shuffle_target_domain_utf8", "shuffled_weighting",
        "staticness_score_initial", "transaction_iteration", "visibility_weight_scale",
    },
    "risk": {
        "accepted_maximum", "bootstrap_replicates", "bootstrap_seed", "calibration_bins",
        "region_quantile", "report_thresholds",
    },
    "sampling": {
        "grid_stride_pixels", "minimum_sample_separation_pixels", "patch_radius_pixels",
        "salient_fraction_of_grid", "sobel_border",
    },
    "surfaces": {
        "connection_distance_rscene", "lambda1_minimum_rscene_squared",
        "lambda1_over_lambda2_minimum", "lambda_gap_over_lambda2_minimum", "linear_rgb_l2",
        "minimum_patch_points", "minimum_region_area_pixels", "minimum_shared_cameras",
        "normal_angle_degrees", "pca_minimum_neighbors", "pca_neighbor_count",
        "pca_radius_rscene", "raster_axis_max_pixels", "raster_axis_min_pixels",
        "voxel_size_rscene",
    },
    "temporal": {
        "centroid_distance_rscene", "dormant_maximum_frames", "forward_backward_cycle_pixels",
        "minimum_cameras", "minimum_node_matches_per_camera", "nearest_node_pixels",
        "reid_endpoint_pixels", "reid_ncc_minimum", "reid_normal_angle_degrees",
        "reid_rgb_l2", "rgb_l2", "search_cost_maximum", "second_best_ratio_minimum",
    },
}
GATE_A_TIER_KEYS = {
    "boundary_f1_delta_minimum", "cross_view_inconsistency_relative_reduction_minimum",
    "ece_maximum", "event_f1_minimum", "event_recall_minimum", "ordering_accuracy_minimum",
    "ordering_auroc_minimum", "ordering_coverage_minimum", "region_iou_delta_minimum",
    "temporal_inconsistency_relative_reduction_minimum",
}

EXECUTION_KEYS = {
    "schema_version", "run_id", "action", "method_id", "code_commit",
    "scientific_config", "command", "inputs", "expected_outputs",
}
RUNTIME_ENVELOPE_KEYS = {
    "runtime", "scheduler", "host", "environment", "timestamps", "attempt",
}
SCHEMA_RULES: dict[str, tuple[set[str], set[str]]] = {
    "phase9-execution-v1": (EXECUTION_KEYS, RUNTIME_ENVELOPE_KEYS),
    "phase9-terminal-v1": (
        {"schema_version", "run_id", "status", "outputs"},
        {"error", "runtime", "scheduler", "command", "inventory"},
    ),
    "n3v-split-v1": ({"schema_version", "dataset_root_env", "scenes"}, set()),
    "depth-visibility-flow-schema-v1": (
        {
            "schema_version", "source_camera", "target_camera", "source_image",
            "target_image", "source_frame", "target_frame", "direction", "dt",
            "height", "width", "units", "pixel_centers", "sampling",
            "generator_revision", "flow", "validity", "occlusion",
        },
        {"runtime"},
    ),
    "phase9-flow-manifest-v1": (
        {
            "schema_version", "run_id", "scene", "flow_record_schema_version",
            "flow_root", "source_split", "target_camera", "cam00_rgb_opened",
            "direction", "generator", "camera_ids", "frame_range",
            "temporal_pair_count_per_camera", "expected_record_count",
            "record_count", "raw_flow_file_count", "unused_flow_file_count",
            "valid_fraction_minimum", "valid_fraction_mean", "valid_fraction_maximum",
            "split_binding", "records", "label_dependent_gate_a",
        },
        {"unused_flow_file_examples"},
    ),
    "phase9-csvl-array-inventory-v1": (
        {
            "schema_version", "run_id", "method_id", "scene", "array_root",
            "file_count", "total_file_bytes", "files", "arrays",
            "input_array_inventories", "artifact_id",
        },
        set(),
    ),
    "phase9-csvl-ledger-v1": (
        {
            "schema_version", "run_id", "method_id", "scene", "target_camera",
            "cam00_rgb_opened", "label_dependent_gate_a", "methodology_status",
            "evidence_boundary", "input_bindings", "frame_count", "geometry_frame_count",
            "frames", "aggregate_layer_opportunity", "temporal_bin_transitions",
            "temporal_interpretation", "flow_summary", "duplicate_semantics",
            "threshold_authority", "split_binding", "array_inventory_sha256", "artifact_id",
        },
        set(),
    ),
    "phase9-csvl-vpl-stage1-ledger-v1": (
        {
            "schema_version", "method_id", "identity_semantics",
            "scientific_content_hash", "scientific_hash_contract",
            "scientific_payload", "runtime_metadata", "artifact_id",
        },
        set(),
    ),
    "phase9-csvl-vpl-stage1-diagnostics-v1": (
        {
            "schema_version", "method_id", "ledger_scientific_content_hash",
            "diagnostics", "cpu_only_evidence", "artifact_id",
        },
        set(),
    ),
    "phase9-csvl-vpl-stage1b-association-audit-v1": (
        {
            "schema_version", "method_id", "stage1_commit",
            "scientific_content_hash", "scientific_hash_contract",
            "scientific_payload", "runtime_metadata", "artifact_id",
        },
        set(),
    ),
    "phase9-csvl-vpl-stage1b-diagnostics-v1": (
        {
            "schema_version", "method_id", "stage1_commit",
            "audit_scientific_content_hash", "diagnostics",
            "cpu_only_evidence", "artifact_id",
        },
        set(),
    ),
}


def assert_finite_tree(value: Any, *, path: str = "$") -> None:
    """Reject any nonfinite scalar or ndarray recursively."""

    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise NonFiniteError(f"nonfinite number at {path}")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise NonFiniteError(f"nonfinite array at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError(f"non-string mapping key at {path}")
            assert_finite_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for index, item in enumerate(value):
            assert_finite_tree(item, path=f"{path}[{index}]")


def _strict_object(
    payload: Any,
    required: set[str],
    optional: set[str] | None = None,
    *,
    what: str,
) -> Mapping[str, Any]:
    optional = set() if optional is None else optional
    if not isinstance(payload, Mapping):
        raise SchemaError(f"{what} must be a JSON object")
    keys = set(payload)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise SchemaError(f"{what} missing keys: {sorted(missing)}")
    if unknown:
        raise SchemaError(f"{what} has unknown keys: {sorted(unknown)}")
    return payload


def _positive_number(value: Any, *, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise SchemaError(f"{what} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise SchemaError(f"{what} must be finite and positive")
    return result


def validate_config(payload: Any) -> Mapping[str, Any]:
    config = _strict_object(payload, CONFIG_KEYS, what="CSVL config")
    assert_finite_tree(config)
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise SchemaError("unexpected CSVL config schema_version")
    if config["method_id"] != METHOD_ID:
        raise SchemaError("unexpected CSVL method_id")
    for section, keys in CONFIG_SECTION_KEYS.items():
        _strict_object(config[section], keys, what=f"CSVL config section {section}")
    _strict_object(
        config["gate_a"]["engineering"],
        GATE_A_TIER_KEYS | {"evaluable_track_fraction_minimum"},
        what="CSVL config Gate A engineering tier",
    )
    _strict_object(
        config["gate_a"]["claim_grade"],
        GATE_A_TIER_KEYS | {"no_represented_event_family_missed"},
        what="CSVL config Gate A claim-grade tier",
    )

    camera = config["camera"]
    expected_camera = {
        "distortion_policy": "nonzero_fails_closed",
        "loader_adapter": "flip_camera_to_world_yz_columns_then_invert_once",
        "matrix_convention": "column_vector_opencv_world_to_camera_positive_z",
        "pixel_centers": "native_integer",
        "resize_sampling": "bilinear_align_corners_false",
    }
    for key, expected in expected_camera.items():
        if camera[key] != expected:
            raise SchemaError(f"unsafe camera override: {key}")

    da3 = config["da3"]
    fixed_da3 = {
        "align_to_input_ext_scale": True,
        "infer_gs": False,
        "use_ray_pose": False,
        "per_frame_normalization": False,
        "scale_shift_fit": False,
        "process_res": 504,
        "process_res_method": "upper_bound_resize",
        "ref_view_strategy": "saddle_balanced",
        "depth_semantics": "optical_axis_z",
    }
    for key, expected in fixed_da3.items():
        if da3[key] != expected:
            raise SchemaError(f"unsafe DA3 override: {key}")
    _positive_number(da3["weight_expected_bytes"], what="DA3 weight_expected_bytes")

    grouping = config["grouping"]
    if grouping["maximum_cameras"] != 6 or grouping["minimum_valid_groups_per_camera_time"] != 2:
        raise SchemaError("group cardinality contract changed")
    if grouping["target_transitive_exclusion"] is not True:
        raise SchemaError("target transitive exclusion must remain enabled")
    return config


def validate_execution(payload: Any) -> Mapping[str, Any]:
    execution = _strict_object(
        payload, EXECUTION_KEYS, RUNTIME_ENVELOPE_KEYS, what="execution manifest"
    )
    assert_finite_tree(execution)
    if execution["schema_version"] != "phase9-execution-v1":
        raise SchemaError("unexpected execution schema_version")
    if execution["method_id"] != METHOD_ID:
        raise SchemaError("execution method_id mismatch")
    command = execution["command"]
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise SchemaError("execution command must be an argv array")
    if not command or not all(isinstance(item, str) and item for item in command):
        raise SchemaError("execution command contains an invalid argv item")
    return execution


def validate_payload(
    schema_name: str,
    payload: Any,
    *,
    required_keys: Sequence[str] | None = None,
    optional_keys: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Validate a registered schema, or an explicit exact key contract."""

    if schema_name == CONFIG_SCHEMA_VERSION:
        return validate_config(payload)
    if schema_name == "phase9-execution-v1":
        return validate_execution(payload)
    if required_keys is None:
        try:
            required, optional = SCHEMA_RULES[schema_name]
        except KeyError as exc:
            raise SchemaError(f"unregistered schema: {schema_name}") from exc
    else:
        required, optional = set(required_keys), set(optional_keys)
    result = _strict_object(payload, set(required), set(optional), what=schema_name)
    assert_finite_tree(result)
    if result.get("schema_version") != schema_name:
        raise SchemaError("payload schema_version mismatch")
    return result


def _reject_json_constant(token: str) -> None:
    raise NonFiniteError(f"non-standard JSON numeric constant: {token}")


def load_json_object(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot read strict JSON: {source}") from exc
    if not isinstance(value, Mapping):
        raise SchemaError(f"JSON root must be an object: {source}")
    assert_finite_tree(value)
    return value


def load_config(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    source = Path(path)
    if expected_sha256 is not None and sha256_file(source) != expected_sha256:
        raise SchemaError("scientific config SHA-256 mismatch")
    return validate_config(load_json_object(source))


__all__ = [
    "SCHEMA_RULES",
    "assert_finite_tree",
    "load_config",
    "load_json_object",
    "validate_config",
    "validate_execution",
    "validate_payload",
]
