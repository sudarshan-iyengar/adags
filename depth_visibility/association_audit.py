"""Deterministic CSVL-VPL Stage-1B association-discrimination forensics.

This module does not define a new association mechanism.  It replays the
frozen Stage-1 candidate universe, exposes every term used by the existing
score, constructs explicitly documented negative controls, and runs score
decompositions against the same P03 observations and sealed P02 arrays.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import math
from typing import Any

import numpy as np

from .camera import camera_center, project_world, unproject_optical_z
from .canonical import domain_id
from .errors import FlowSemanticsError, ProvenanceError, SchemaError
from .flow import bilinear_flow
from .schema import assert_finite_tree, validate_payload
from .surface_tracks import (
    P02FlowStore,
    associate_observations,
    summarize_association,
)


METHOD_ID = "csvl-vpl-stage1b-association-discrimination-audit-v1"
CONFIG_SCHEMA = "csvl-vpl-stage1b-config-v1"
AUDIT_SCHEMA = "phase9-csvl-vpl-stage1b-association-audit-v1"
DIAGNOSTICS_SCHEMA = "phase9-csvl-vpl-stage1b-diagnostics-v1"
SCIENTIFIC_HASH_DOMAIN = "csvl-vpl-stage1b-v1/scientific-content"
STAGE1_COMMIT = "70a9a678df290a2ae9510f313fdb704cae2632f4"

CURRENT_CONTROLS = (
    "valid",
    "reversed_flow",
    "camera_swap",
    "temporal_offset",
    "corrupted_flow",
)
MATCHED_CONTROLS = (
    "direction_rotated_matched",
    "camera_swap_matched",
    "temporal_offset_matched",
)
ABLATIONS = (
    "geometry_p03_only",
    "flow_only",
    "camera_reprojection_only",
    "geometry_plus_camera_without_flow",
    "geometry_plus_flow_without_camera_specific_consistency",
    "full_current_score",
)


def validate_stage1b_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "method_id", "stage1_commit", "stage1_config_sha256",
        "candidate_universe", "controls", "ablations", "matched_controls",
        "canonical_hash", "prohibited_reads",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise SchemaError("Stage-1B config has missing or unknown keys")
    if payload["schema_version"] != CONFIG_SCHEMA or payload["method_id"] != METHOD_ID:
        raise SchemaError("Stage-1B config identity mismatch")
    if payload["stage1_commit"] != STAGE1_COMMIT:
        raise SchemaError("Stage-1B Stage-1 commit authority mismatch")
    if payload["candidate_universe"] != "exact_stage1_valid_candidate_evidence":
        raise SchemaError("Stage-1B must use the exact Stage-1 candidate universe")
    if tuple(payload["controls"]) != CURRENT_CONTROLS:
        raise SchemaError("Stage-1B current-control roster changed")
    if tuple(payload["ablations"]) != ABLATIONS:
        raise SchemaError("Stage-1B ablation roster changed")
    matched = payload["matched_controls"]
    if not isinstance(matched, Mapping) or set(matched) != {
        "camera_nearest_baseline_candidates", "temporal_offsets_frames",
        "direction_rotation_degrees",
    }:
        raise SchemaError("Stage-1B matched-control contract changed")
    if int(matched["camera_nearest_baseline_candidates"]) != 2:
        raise SchemaError("Stage-1B camera matching must remain frozen at two candidates")
    if list(matched["temporal_offsets_frames"]) != [-2, -1, 1, 2]:
        raise SchemaError("Stage-1B temporal matching offsets changed")
    if int(matched["direction_rotation_degrees"]) != 90:
        raise SchemaError("Stage-1B direction-matched rotation changed")
    hash_contract = payload["canonical_hash"]
    if not isinstance(hash_contract, Mapping) or set(hash_contract) != {
        "algorithm", "domain", "excluded_runtime_fields"
    }:
        raise SchemaError("Stage-1B canonical hash contract changed")
    if hash_contract["algorithm"] != "sha256_domain_separated_csvl_cjson_v1" or hash_contract["domain"] != SCIENTIFIC_HASH_DOMAIN:
        raise SchemaError("Stage-1B canonical hash identity changed")
    if set(hash_contract["excluded_runtime_fields"]) != {
        "timestamp_utc", "slurm_job_id", "absolute_output_root"
    }:
        raise SchemaError("Stage-1B runtime exclusions changed")
    prohibited = {str(value) for value in payload["prohibited_reads"]}
    if not {"cam00_rgb", "wandb", "model_weights", "new_depth", "new_flow"}.issubset(prohibited):
        raise SchemaError("Stage-1B prohibited-read contract is incomplete")
    assert_finite_tree(payload)
    return dict(payload)


def canonical_scientific_hash(payload: Mapping[str, Any]) -> str:
    assert_finite_tree(payload)
    return domain_id(SCIENTIFIC_HASH_DOMAIN, payload)


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "minimum": None, "mean": None, "quantiles": {}, "maximum": None}
    if not np.isfinite(array).all():
        raise SchemaError("Stage-1B diagnostic distribution contains nonfinite values")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "quantiles": {
            str(q): float(np.quantile(array, q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "maximum": float(np.max(array)),
    }


def edge_key(source_observation_id: str, destination_observation_id: str) -> str:
    return f"{source_observation_id}->{destination_observation_id}"


def control_definitions() -> dict[str, Any]:
    return {
        "valid": {
            "sample_camera": "original P03 common physical camera",
            "sample_time": "association source frame through destination frame minus one",
            "vector_transform": "identity",
            "calibration_for_projection": "original camera at source and destination times",
        },
        "reversed_flow": {
            "sample_camera": "original camera",
            "sample_time": "original forward P02 records",
            "vector_transform": "negate each sampled forward vector while continuing along the negated trajectory",
            "semantic_disclosure": "sign-inverted forward field; not a sealed backward t+1-to-t array",
        },
        "camera_swap": {
            "sample_camera": "next camera in lexicographically sorted P02 roster",
            "sample_time": "original association time",
            "vector_transform": "identity",
            "fixed_problem": "P03 endpoints, original-camera pixels, original calibration, and thresholds remain fixed; only flow array and mask ancestry change",
        },
        "temporal_offset": {
            "sample_camera": "original camera",
            "sample_time": "each P02 record is shifted by plus one frame",
            "vector_transform": "identity",
            "fixed_problem": "P03 endpoints and original source/destination times remain fixed",
        },
        "corrupted_flow": {
            "sample_camera": "original camera",
            "sample_time": "original association time",
            "vector_transform": "(dx,dy) becomes (dy+16,dx-16) pixels per step",
            "matched": False,
        },
        "direction_rotated_matched": {
            "source": "valid chain",
            "vector_transform": "rotate the valid accumulated displacement by 90 degrees",
            "preserves": ["flow magnitude", "valid-mask ancestry", "temporal relation", "candidate support"],
        },
        "camera_swap_matched": {
            "sample_camera": "one of the two nearest-baseline alternative cameras",
            "selection": "minimum sum of relative chain-magnitude difference and valid-mask-fraction difference from valid; then baseline and camera-id tie breaks",
            "fixed_problem": "original P03 endpoints, original-camera pixels, calibration, time, and thresholds remain fixed",
        },
        "temporal_offset_matched": {
            "sample_camera": "original camera",
            "candidate_offsets_frames": [-2, -1, 1, 2],
            "selection": "minimum sum of relative chain-magnitude difference and valid-mask-fraction difference from valid; then absolute-offset and signed-offset tie breaks",
            "fixed_problem": "original P03 endpoints, camera, calibration, and thresholds remain fixed",
        },
    }


def metric_definitions() -> dict[str, Any]:
    return {
        "stage1_reported_epe": {
            "formula": "median_camera ||(project(source_xyz, camera_t) + supplied_flow_chain) - project(destination_xyz, camera_t_plus_gap)||_2",
            "coordinate_frame": "native original-camera integer-center image pixels",
            "units": "pixels at 1352x1014 source resolution",
            "masking": "every sampled P02 step must pass its sealed forward-backward validity mask",
            "aggregation": "median across common cameras; track diagnostics aggregate selected-edge medians",
            "reference": "fixed P03-derived destination geometry projected with original camera calibration",
            "independence": "common across controls but not external ground truth; it is a P03/calibration consistency proxy",
            "self_evaluation": False,
        },
        "normalized_endpoint_error": {
            "formula": "EPE / (2 pixels + projected source P03 half-bin radius + projected destination P03 half-bin radius)",
            "current_score_weight": 0.8,
        },
        "geometry_cost": {
            "formula": "||source_world_xyz-destination_world_xyz|| / (0.05 * R_scene * frame_gap)",
            "candidate_prefilter": "candidate is absent when geometry_cost > 1",
            "current_score_weight": 0.2,
        },
        "current_cost": {
            "formula": "0.8*min(2, normalized_endpoint_error) + 0.2*min(2, geometry_cost)",
            "admission": "cost <= 1",
        },
        "current_risk": {
            "formula": "max(min(1, normalized_endpoint_error), 1-valid_camera_count/common_camera_count)",
            "geometry_or_p03_risk_used": False,
        },
        "depth_order_contribution": {"weight": 0.0, "gate": False},
        "p03_observation_uncertainty_contribution": {"weight": 0.0, "gate": False},
        "forward_backward_contribution": {
            "numeric_score": False,
            "use": "binary P02 validity-mask filter only",
            "generator_threshold_pixels": 1.0,
        },
        "valid_flow_reference_disagreement": {
            "formula": "median_camera ||control_predicted_endpoint-valid_predicted_endpoint||_2",
            "purpose": "fixed valid-flow reference diagnostic, not candidate correctness ground truth",
        },
    }


def _projected_bin_radius(observation: Mapping[str, Any], target: Any, camera: Any) -> float:
    size = int(observation["target_bin_pixels"])
    center = np.asarray(observation["target_pixel_center"], dtype=np.float64)
    half = 0.5 * size
    corners = np.asarray(
        [[center[0]-half, center[1]-half], [center[0]+half, center[1]-half],
         [center[0]-half, center[1]+half], [center[0]+half, center[1]+half]],
        dtype=np.float64,
    )
    world = unproject_optical_z(
        target.K, target.w2c_opencv, corners,
        np.full(4, float(observation["median_optical_z"]), dtype=np.float64),
    )
    pixels, _ = project_world(camera.K, camera.w2c_opencv, world)
    projected_center, _ = project_world(
        camera.K, camera.w2c_opencv,
        np.asarray(observation["world_xyz"], dtype=np.float64),
    )
    return float(np.max(np.linalg.norm(pixels - projected_center[None, :], axis=1)))


def _calibration_id(record: Any) -> str:
    return domain_id(
        "csvl-vpl-stage1b-v1/calibration",
        {
            "camera_id": str(record.camera_id), "frame": int(record.frame),
            "time": float(record.time), "width": int(record.width), "height": int(record.height),
            "K": np.asarray(record.K, dtype=np.float64).tolist(),
            "w2c_opencv": np.asarray(record.w2c_opencv, dtype=np.float64).tolist(),
        },
    )


def _sample_chain(
    store: P02FlowStore,
    *,
    sample_camera: str,
    source_frame: int,
    target_frame: int,
    source_xy: np.ndarray,
    frame_offset: int = 0,
    vector_transform: str = "identity",
) -> dict[str, Any]:
    position = np.asarray(source_xy, dtype=np.float64)
    steps = []
    for logical_frame in range(int(source_frame), int(target_frame)):
        record_frame = logical_frame + int(frame_offset)
        try:
            flow, valid, reference, record = store.load_bound_step_for_audit(sample_camera, record_frame)
        except FlowSemanticsError:
            return {"valid": False, "reason": "missing_flow_record", "steps": steps}
        vector = bilinear_flow(flow, float(position[0]), float(position[1]), valid)
        if vector is None:
            return {"valid": False, "reason": "invalid_mask_or_boundary_sample", "steps": steps}
        vector = np.asarray(vector, dtype=np.float64)
        sampled_vector = vector.copy()
        if vector_transform == "negate":
            vector = -vector
        elif vector_transform == "stage1_corrupt":
            vector = np.asarray([vector[1] + 16.0, vector[0] - 16.0], dtype=np.float64)
        elif vector_transform != "identity":
            raise SchemaError(f"unknown Stage-1B vector transform: {vector_transform}")
        width, height = int(record["source_width"]), int(record["source_height"])
        boundary = min(float(position[0]), float(position[1]), width-1-float(position[0]), height-1-float(position[1]))
        steps.append(
            {
                "logical_source_frame": logical_frame,
                "sample_source_frame": int(record["source_frame"]),
                "sample_target_frame": int(record["target_frame"]),
                "sample_camera": str(record["source_camera"]),
                "sample_target_camera": str(record["target_camera"]),
                "flow_direction": str(record["direction"]),
                "coordinate_convention": str(record["coordinate_convention"]),
                "units": str(record["units"]),
                "sampling": str(record["sampling"]),
                "source_resolution": [width, height],
                "scale_or_resize": "none_native_resolution",
                "dt_seconds": float(record["dt_seconds"]),
                "sample_position_xy": position.tolist(),
                "sampled_forward_vector_xy": sampled_vector.tolist(),
                "used_vector_xy": vector.tolist(),
                "used_vector_transform": vector_transform,
                "valid_mask_semantics": str(record["validity_semantics"]),
                "occlusion_semantics": str(record["occlusion_semantics"]),
                "manifest_valid_pixel_fraction": float(record["valid_pixel_fraction"]),
                "boundary_distance_pixels": boundary,
                "p02_record_id": str(reference["record_id"]),
                "p02_path": str(reference["path"]),
                "p02_npz_sha256": str(reference["sha256"]),
                "p02_flow_content_sha256": str(reference["flow_contiguous_sha256"]),
                "p02_valid_mask_sha256": str(reference["validity_contiguous_sha256"]),
                "source_image_hashes": list(record["source_hashes"]),
            }
        )
        position = position + vector
    displacement = position - np.asarray(source_xy, dtype=np.float64)
    return {
        "valid": True,
        "reason": None,
        "destination_xy": position.tolist(),
        "chain_displacement_xy": displacement.tolist(),
        "chain_magnitude_pixels": float(np.linalg.norm(displacement)),
        "steps": steps,
        "record_ids": [str(value["p02_record_id"]) for value in steps],
    }


def _rotate_valid_chain(valid_chain: Mapping[str, Any], source_xy: np.ndarray) -> dict[str, Any]:
    if not valid_chain.get("valid"):
        return dict(valid_chain)
    vector = np.asarray(valid_chain["chain_displacement_xy"], dtype=np.float64)
    rotated = np.asarray([-vector[1], vector[0]], dtype=np.float64)
    destination = np.asarray(source_xy, dtype=np.float64) + rotated
    steps = []
    for step in valid_chain["steps"]:
        copied = dict(step)
        original = np.asarray(step["used_vector_xy"], dtype=np.float64)
        copied["used_vector_xy"] = [-float(original[1]), float(original[0])]
        copied["used_vector_transform"] = "rotate_valid_displacement_90_degrees"
        steps.append(copied)
    return {
        "valid": True, "reason": None, "destination_xy": destination.tolist(),
        "chain_displacement_xy": rotated.tolist(),
        "chain_magnitude_pixels": float(np.linalg.norm(rotated)),
        "steps": steps, "record_ids": list(valid_chain["record_ids"]),
    }


def _nearest_alternative_cameras(
    camera: str,
    frame: int,
    train_records: Mapping[tuple[str, int], Any],
    count: int,
) -> list[tuple[str, float]]:
    original = train_records[(camera, frame)]
    center = camera_center(original.w2c_opencv)
    rows = []
    for (candidate_camera, candidate_frame), record in train_records.items():
        if candidate_frame != frame or candidate_camera == camera:
            continue
        baseline = float(np.linalg.norm(camera_center(record.w2c_opencv) - center))
        rows.append((str(candidate_camera), baseline))
    return sorted(rows, key=lambda value: (value[1], value[0]))[:count]


def _chain_score(
    chain: Mapping[str, Any],
    *,
    destination_xy: np.ndarray,
    tolerance: float,
    required_vector: np.ndarray,
    valid_chain: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not chain.get("valid"):
        return None
    predicted = np.asarray(chain["destination_xy"], dtype=np.float64)
    flow_vector = np.asarray(chain["chain_displacement_xy"], dtype=np.float64)
    endpoint_error = float(np.linalg.norm(predicted - destination_xy))
    flow_magnitude = float(np.linalg.norm(flow_vector))
    required_magnitude = float(np.linalg.norm(required_vector))
    magnitude_error = abs(flow_magnitude - required_magnitude)
    denominator = flow_magnitude * required_magnitude
    cosine = None if denominator == 0 else float(np.dot(flow_vector, required_vector) / denominator)
    valid_disagreement = None
    if valid_chain.get("valid"):
        valid_disagreement = float(
            np.linalg.norm(predicted - np.asarray(valid_chain["destination_xy"], dtype=np.float64))
        )
    return {
        "endpoint_error_pixels": endpoint_error,
        "normalized_endpoint_error": endpoint_error / tolerance,
        "flow_magnitude_pixels": flow_magnitude,
        "required_projected_displacement_pixels": required_magnitude,
        "flow_magnitude_absolute_error_pixels": magnitude_error,
        "flow_magnitude_normalized_error": magnitude_error / tolerance,
        "flow_direction_cosine": cosine,
        "valid_flow_reference_disagreement_pixels": valid_disagreement,
        "predicted_destination_xy": predicted.tolist(),
        "flow_chain": dict(chain),
    }


def _aggregate_control(
    camera_rows: Iterable[Mapping[str, Any]],
    *,
    control: str,
    geometry_cost: float,
    common_camera_count: int,
    minimum_cameras: int,
    endpoint_weight: float,
    geometry_weight: float,
    admission_threshold: float,
) -> dict[str, Any] | None:
    rows = [dict(value["controls"][control]) for value in camera_rows if value["controls"].get(control) is not None]
    if len(rows) < minimum_cameras:
        return None
    endpoint = float(np.median([value["normalized_endpoint_error"] for value in rows]))
    epe = float(np.median([value["endpoint_error_pixels"] for value in rows]))
    magnitude_error = float(np.median([value["flow_magnitude_normalized_error"] for value in rows]))
    magnitude_error_pixels = float(np.median([value["flow_magnitude_absolute_error_pixels"] for value in rows]))
    missing_risk = 1.0 - len(rows) / common_camera_count
    risk = max(min(1.0, endpoint), missing_risk)
    cost = endpoint_weight * min(2.0, endpoint) + geometry_weight * min(2.0, geometry_cost)
    record_ids = sorted({record for value in rows for record in value["flow_chain"]["record_ids"]})
    return {
        "control": control,
        "valid_camera_count": len(rows),
        "common_camera_count": common_camera_count,
        "endpoint_error_pixels_median": epe,
        "normalized_endpoint_error_median": endpoint,
        "flow_magnitude_absolute_error_pixels_median": magnitude_error_pixels,
        "flow_magnitude_normalized_error_median": magnitude_error,
        "flow_direction_cosine_median": (
            float(np.median([value["flow_direction_cosine"] for value in rows if value["flow_direction_cosine"] is not None]))
            if any(value["flow_direction_cosine"] is not None for value in rows) else None
        ),
        "valid_flow_reference_disagreement_pixels_median": (
            float(np.median([value["valid_flow_reference_disagreement_pixels"] for value in rows if value["valid_flow_reference_disagreement_pixels"] is not None]))
            if any(value["valid_flow_reference_disagreement_pixels"] is not None for value in rows) else None
        ),
        "cost": cost,
        "association_risk": risk,
        "association_confidence": 1.0 - risk,
        "admitted": bool(cost <= admission_threshold),
        "flow_record_ids": record_ids,
    }


def build_candidate_audit(
    observations: Iterable[Mapping[str, Any]],
    valid_candidates: Iterable[Mapping[str, Any]],
    *,
    train_records: Mapping[tuple[str, int], Any],
    target_records: Mapping[tuple[str, int], Any],
    flow_store: P02FlowStore,
    r_scene: float,
    stage1_config: Mapping[str, Any],
    stage1b_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expose exact bindings and score terms for every frozen Stage-1 candidate."""

    validate_stage1b_config(stage1b_config)
    by_id = {str(value["observation_id"]): dict(value) for value in observations}
    association = stage1_config["association"]
    endpoint_weight = float(association["endpoint_cost_weight"])
    geometry_weight = float(association["geometry_cost_weight"])
    minimum_cameras = int(association["minimum_cameras"])
    threshold = float(association["search_cost_maximum"])
    nearest_count = int(stage1b_config["matched_controls"]["camera_nearest_baseline_candidates"])
    temporal_offsets = [int(value) for value in stage1b_config["matched_controls"]["temporal_offsets_frames"]]
    output = []
    for base in sorted(valid_candidates, key=lambda value: (str(value["source_observation_id"]), str(value["destination_observation_id"]))):
        source = by_id[str(base["source_observation_id"])]
        destination = by_id[str(base["destination_observation_id"])]
        source_frame, destination_frame = int(source["frame"]), int(destination["frame"])
        gap = destination_frame - source_frame
        maximum_displacement = float(association["maximum_world_displacement_rscene_per_frame"]) * r_scene * gap
        geometry_cost = float(base["world_displacement"]) / maximum_displacement
        common_cameras = sorted(str(value["camera_id"]) for value in base["camera_evidence"])
        camera_rows = []
        source_target = target_records[(str(source["target_camera"]), source_frame)]
        destination_target = target_records[(str(destination["target_camera"]), destination_frame)]
        for camera in common_cameras:
            source_camera = train_records[(camera, source_frame)]
            destination_camera = train_records[(camera, destination_frame)]
            source_xy, _ = project_world(source_camera.K, source_camera.w2c_opencv, np.asarray(source["world_xyz"], dtype=np.float64))
            destination_xy, _ = project_world(destination_camera.K, destination_camera.w2c_opencv, np.asarray(destination["world_xyz"], dtype=np.float64))
            required_vector = np.asarray(destination_xy, dtype=np.float64) - np.asarray(source_xy, dtype=np.float64)
            tolerance = (
                float(association["base_endpoint_tolerance_pixels"])
                + _projected_bin_radius(source, source_target, source_camera)
                + _projected_bin_radius(destination, destination_target, destination_camera)
            )
            valid_chain = _sample_chain(
                flow_store, sample_camera=camera, source_frame=source_frame,
                target_frame=destination_frame, source_xy=np.asarray(source_xy),
            )
            chains = {
                "valid": valid_chain,
                "reversed_flow": _sample_chain(
                    flow_store, sample_camera=camera, source_frame=source_frame,
                    target_frame=destination_frame, source_xy=np.asarray(source_xy),
                    vector_transform="negate",
                ),
                "camera_swap": _sample_chain(
                    flow_store, sample_camera=flow_store.swapped_camera(camera),
                    source_frame=source_frame, target_frame=destination_frame,
                    source_xy=np.asarray(source_xy),
                ),
                "temporal_offset": _sample_chain(
                    flow_store, sample_camera=camera, source_frame=source_frame,
                    target_frame=destination_frame, source_xy=np.asarray(source_xy),
                    frame_offset=1,
                ),
                "corrupted_flow": _sample_chain(
                    flow_store, sample_camera=camera, source_frame=source_frame,
                    target_frame=destination_frame, source_xy=np.asarray(source_xy),
                    vector_transform="stage1_corrupt",
                ),
                "direction_rotated_matched": _rotate_valid_chain(valid_chain, np.asarray(source_xy)),
            }
            nearest = _nearest_alternative_cameras(camera, source_frame, train_records, nearest_count)
            camera_options = []
            for sample_camera, baseline in nearest:
                chain = _sample_chain(
                    flow_store, sample_camera=sample_camera, source_frame=source_frame,
                    target_frame=destination_frame, source_xy=np.asarray(source_xy),
                )
                if chain.get("valid") and valid_chain.get("valid"):
                    difference = abs(float(chain["chain_magnitude_pixels"]) - float(valid_chain["chain_magnitude_pixels"]))
                    relative_difference = difference / max(1.0, float(valid_chain["chain_magnitude_pixels"]))
                    valid_mask = float(np.mean([step["manifest_valid_pixel_fraction"] for step in valid_chain["steps"]]))
                    control_mask = float(np.mean([step["manifest_valid_pixel_fraction"] for step in chain["steps"]]))
                    mask_difference = abs(control_mask - valid_mask)
                    camera_options.append((relative_difference + mask_difference, baseline, sample_camera, chain, relative_difference, mask_difference))
            if camera_options:
                _, baseline, sample_camera, selected_chain, magnitude_difference, mask_difference = min(camera_options, key=lambda value: (value[0], value[1], value[2]))
                selected_chain = dict(selected_chain)
                selected_chain["matched_sample_camera"] = sample_camera
                selected_chain["camera_baseline_world"] = float(baseline)
                selected_chain["relative_magnitude_match_error"] = float(magnitude_difference)
                selected_chain["valid_mask_fraction_match_error"] = float(mask_difference)
                chains["camera_swap_matched"] = selected_chain
            else:
                chains["camera_swap_matched"] = {"valid": False, "reason": "no_matched_alternative_camera", "steps": []}
            temporal_options = []
            for offset in temporal_offsets:
                chain = _sample_chain(
                    flow_store, sample_camera=camera, source_frame=source_frame,
                    target_frame=destination_frame, source_xy=np.asarray(source_xy), frame_offset=offset,
                )
                if chain.get("valid") and valid_chain.get("valid"):
                    difference = abs(float(chain["chain_magnitude_pixels"]) - float(valid_chain["chain_magnitude_pixels"]))
                    relative_difference = difference / max(1.0, float(valid_chain["chain_magnitude_pixels"]))
                    valid_mask = float(np.mean([step["manifest_valid_pixel_fraction"] for step in valid_chain["steps"]]))
                    control_mask = float(np.mean([step["manifest_valid_pixel_fraction"] for step in chain["steps"]]))
                    mask_difference = abs(control_mask - valid_mask)
                    temporal_options.append((relative_difference + mask_difference, abs(offset), offset, chain, relative_difference, mask_difference))
            if temporal_options:
                _, _, offset, selected_chain, magnitude_difference, mask_difference = min(temporal_options, key=lambda value: (value[0], value[1], value[2]))
                selected_chain = dict(selected_chain)
                selected_chain["matched_temporal_offset_frames"] = int(offset)
                selected_chain["relative_magnitude_match_error"] = float(magnitude_difference)
                selected_chain["valid_mask_fraction_match_error"] = float(mask_difference)
                chains["temporal_offset_matched"] = selected_chain
            else:
                chains["temporal_offset_matched"] = {"valid": False, "reason": "no_matched_temporal_offset", "steps": []}
            controls = {
                name: _chain_score(
                    chain, destination_xy=np.asarray(destination_xy), tolerance=tolerance,
                    required_vector=required_vector, valid_chain=valid_chain,
                )
                for name, chain in chains.items()
            }
            camera_rows.append(
                {
                    "camera_id": camera,
                    "source_frame": source_frame, "destination_frame": destination_frame,
                    "source_time": float(source_camera.time), "destination_time": float(destination_camera.time),
                    "source_xy": np.asarray(source_xy).tolist(), "destination_xy": np.asarray(destination_xy).tolist(),
                    "required_projected_displacement_xy": required_vector.tolist(),
                    "quantization_aware_tolerance_pixels": tolerance,
                    "source_calibration_id": _calibration_id(source_camera),
                    "destination_calibration_id": _calibration_id(destination_camera),
                    "projection_camera_unchanged_across_controls": True,
                    "controls": controls,
                }
            )
        control_scores = {
            name: _aggregate_control(
                camera_rows, control=name, geometry_cost=geometry_cost,
                common_camera_count=len(common_cameras), minimum_cameras=minimum_cameras,
                endpoint_weight=endpoint_weight, geometry_weight=geometry_weight,
                admission_threshold=threshold,
            )
            for name in (*CURRENT_CONTROLS, *MATCHED_CONTROLS)
        }
        zero_errors = [
            float(np.linalg.norm(np.asarray(value["destination_xy"]) - np.asarray(value["source_xy"])))
            / float(value["quantization_aware_tolerance_pixels"])
            for value in camera_rows
        ]
        zero_error_pixels = [
            float(np.linalg.norm(np.asarray(value["destination_xy"]) - np.asarray(value["source_xy"])))
            for value in camera_rows
        ]
        valid_score = control_scores["valid"]
        if valid_score is None:
            raise ProvenanceError("exact Stage-1 valid candidate lost its minimum camera support")
        row = {
            "edge_key": edge_key(str(source["observation_id"]), str(destination["observation_id"])),
            "source_observation_id": str(source["observation_id"]),
            "destination_observation_id": str(destination["observation_id"]),
            "source_p03_hypothesis_id": str(source["source_observations"]["p03_hypothesis_id"]),
            "destination_p03_hypothesis_id": str(destination["source_observations"]["p03_hypothesis_id"]),
            "source_frame": source_frame, "destination_frame": destination_frame, "frame_gap": gap,
            "source_depth_order": str(source["depth_order"]),
            "destination_depth_order": str(destination["depth_order"]),
            "order_transition": f"{source['depth_order']}->{destination['depth_order']}",
            "depth_order_current_score_weight": 0.0,
            "source_p03_risk": float(source["observation_risk"]),
            "destination_p03_risk": float(destination["observation_risk"]),
            "p03_uncertainty_current_score_weight": 0.0,
            "world_displacement": float(base["world_displacement"]),
            "world_displacement_rscene_per_frame": float(base["world_displacement_rscene_per_frame"]),
            "geometry_cost": geometry_cost,
            "geometry_prefilter_passed": True,
            "common_cameras": common_cameras,
            "camera_evidence": camera_rows,
            "control_scores": control_scores,
            "zero_flow_reprojection_error_pixels_median": float(np.median(zero_error_pixels)),
            "zero_flow_normalized_error_median": float(np.median(zero_errors)),
            "stage1_base_candidate_id": str(base["candidate_id"]),
            "stage1_base_cost": float(base["cost"]),
            "stage1_base_admitted": bool(base["admitted"]),
        }
        if not math.isclose(float(valid_score["cost"]), float(base["cost"]), rel_tol=0.0, abs_tol=1e-12):
            raise ProvenanceError("Stage-1B valid-score replay differs from sealed Stage-1 candidate")
        output.append(row)
    assert_finite_tree(output)
    return output


def build_ablation_candidate_sets(
    audit_rows: Iterable[Mapping[str, Any]],
    valid_candidates: Iterable[Mapping[str, Any]],
    *,
    stage1_config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    base_by_edge = {
        edge_key(str(value["source_observation_id"]), str(value["destination_observation_id"])): dict(value)
        for value in valid_candidates
    }
    threshold = float(stage1_config["association"]["search_cost_maximum"])
    endpoint_weight = float(stage1_config["association"]["endpoint_cost_weight"])
    geometry_weight = float(stage1_config["association"]["geometry_cost_weight"])
    output = {name: [] for name in ABLATIONS}
    for row in audit_rows:
        base = base_by_edge[str(row["edge_key"])]
        geometry = float(row["geometry_cost"])
        valid = row["control_scores"]["valid"]
        assert valid is not None
        endpoint = float(valid["normalized_endpoint_error_median"])
        zero = float(row["zero_flow_normalized_error_median"])
        magnitude = float(valid["flow_magnitude_normalized_error_median"])
        definitions = {
            "geometry_p03_only": (geometry, min(1.0, geometry), []),
            "flow_only": (endpoint, float(valid["association_risk"]), list(valid["flow_record_ids"])),
            "camera_reprojection_only": (zero, min(1.0, zero), []),
            "geometry_plus_camera_without_flow": (
                endpoint_weight*min(2.0, zero) + geometry_weight*min(2.0, geometry),
                min(1.0, max(zero, geometry)), [],
            ),
            "geometry_plus_flow_without_camera_specific_consistency": (
                endpoint_weight*min(2.0, magnitude) + geometry_weight*min(2.0, geometry),
                min(1.0, max(magnitude, geometry)), list(valid["flow_record_ids"]),
            ),
            "full_current_score": (float(base["cost"]), float(base["association_risk"]), list(base["flow_record_ids"])),
        }
        for variant, (cost, risk, flow_ids) in definitions.items():
            candidate = dict(base)
            if variant != "full_current_score":
                candidate["candidate_id"] = domain_id(
                    "csvl-vpl-stage1b-v1/ablation-edge",
                    {"variant": variant, "edge_key": str(row["edge_key"])},
                )
            candidate.update(
                {
                    "mode": variant, "cost": float(cost),
                    "association_risk": float(risk),
                    "association_confidence": 1.0-float(risk),
                    "admitted": bool(cost <= threshold),
                    "flow_record_ids": flow_ids,
                    "audit_score_basis": variant,
                }
            )
            if variant in {"camera_reprojection_only", "geometry_plus_camera_without_flow"}:
                candidate["normalized_endpoint_error_median"] = zero
                candidate["endpoint_error_pixels_median"] = float(row["zero_flow_reprojection_error_pixels_median"])
                candidate["endpoint_error_pixels_maximum"] = float(row["zero_flow_reprojection_error_pixels_median"])
            elif variant == "geometry_plus_flow_without_camera_specific_consistency":
                candidate["normalized_endpoint_error_median"] = magnitude
                candidate["endpoint_error_pixels_median"] = float(valid["flow_magnitude_absolute_error_pixels_median"])
                candidate["endpoint_error_pixels_maximum"] = float(valid["flow_magnitude_absolute_error_pixels_median"])
            output[variant].append(candidate)
    return {key: sorted(value, key=lambda row: str(row["candidate_id"])) for key, value in output.items()}


def _selected_edges(result: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]]) -> list[str]:
    by_id = {
        str(value["candidate_id"]): edge_key(str(value["source_observation_id"]), str(value["destination_observation_id"]))
        for value in candidates
    }
    selected = []
    for track in result["tracks"]:
        for record in track["records"]:
            candidate_id = record["association"].get("candidate_id")
            if candidate_id is not None:
                selected.append(by_id[str(candidate_id)])
    return sorted(selected)


def replay_ablations(
    observations: Iterable[Mapping[str, Any]],
    candidate_sets: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    stage1_config: Mapping[str, Any],
    frame_range: tuple[int, int],
    sealed_stage1_result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    results = {}
    summaries = {}
    full_accepted: set[str] = set()
    full_selected: set[str] = set()
    for variant in ABLATIONS:
        candidates = list(candidate_sets[variant])
        result = associate_observations(
            observations, candidates, config=stage1_config, frame_range=frame_range,
        )
        if variant == "full_current_score" and result != sealed_stage1_result:
            raise ProvenanceError("Stage-1B full replay differs from sealed Stage-1 association result")
        accepted = sorted(
            edge_key(str(value["source_observation_id"]), str(value["destination_observation_id"]))
            for value in candidates if value["admitted"]
        )
        selected = _selected_edges(result, candidates)
        summary = summarize_association(result, candidates)
        results[variant] = result
        summaries[variant] = {
            "accepted_edge_count": len(accepted), "selected_edge_count": len(selected),
            "accepted_edges": accepted, "selected_edges": selected,
            "track_count": summary["track_count"],
            "multi_frame_track_count": summary["multi_frame_track_count"],
            "multi_frame_rear_track_count": summary["multi_frame_rear_track_count"],
            "track_duration_frames": summary["track_duration_frames"],
            "observed_frames_per_track": summary["observed_frames_per_track"],
            "state_counts": summary["state_counts"],
            "visibility_event_counts": summary["visibility_event_counts"],
            "association_confidence": summary["association_confidence"],
            "association_risk": summary["association_risk"],
            "coverage": summary["coverage"],
            "split_merge_ambiguity_counts": summary["split_merge_ambiguity_counts"],
        }
        if variant == "full_current_score":
            full_accepted, full_selected = set(accepted), set(selected)
    for variant in ABLATIONS:
        accepted = set(summaries[variant]["accepted_edges"])
        selected = set(summaries[variant]["selected_edges"])
        summaries[variant]["changed_accepted_edges_vs_full"] = len(accepted ^ full_accepted)
        summaries[variant]["changed_selected_edges_vs_full"] = len(selected ^ full_selected)
        summaries[variant]["selected_edge_overlap_fraction_vs_full"] = (
            len(selected & full_selected) / max(1, len(selected | full_selected))
        )
    return summaries, results


def paired_discrimination(audit_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(audit_rows)
    comparisons = {}
    for control in (*CURRENT_CONTROLS[1:], *MATCHED_CONTROLS):
        deltas = []
        epe_deltas = []
        admission_changes = 0
        for row in rows:
            valid = row["control_scores"]["valid"]
            other = row["control_scores"].get(control)
            if valid is None or other is None:
                continue
            deltas.append(float(other["cost"]) - float(valid["cost"]))
            epe_deltas.append(float(other["endpoint_error_pixels_median"]) - float(valid["endpoint_error_pixels_median"]))
            admission_changes += bool(other["admitted"]) != bool(valid["admitted"])
        comparisons[control] = {
            "paired_candidate_count": len(deltas),
            "control_minus_valid_cost": distribution(deltas),
            "control_minus_valid_epe_pixels": distribution(epe_deltas),
            "valid_lower_cost_fraction": sum(value > 0 for value in deltas) / max(1, len(deltas)),
            "control_lower_cost_fraction": sum(value < 0 for value in deltas) / max(1, len(deltas)),
            "equal_cost_fraction": sum(value == 0 for value in deltas) / max(1, len(deltas)),
            "admission_changed_count": int(admission_changes),
        }
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_observation_id"])].append(row)
    ranking = {}
    for condition in CURRENT_CONTROLS:
        ranks = []
        for candidates in by_source.values():
            available = [value for value in candidates if value["control_scores"].get(condition) is not None]
            if len(available) < 2:
                continue
            geometry_best = min(available, key=lambda value: (float(value["geometry_cost"]), str(value["destination_observation_id"])))
            ordered = sorted(
                available,
                key=lambda value: (float(value["control_scores"][condition]["cost"]), str(value["destination_observation_id"])),
            )
            ranks.append(ordered.index(geometry_best) + 1)
        ranking[condition] = {
            "multi_candidate_source_count": len(ranks),
            "geometry_consistent_proxy_top1_fraction": sum(value == 1 for value in ranks) / max(1, len(ranks)),
            "geometry_consistent_proxy_rank": distribution(ranks),
            "correctness_boundary": "lowest geometry cost is only a P03 proxy, not physical identity ground truth",
        }
    return {"paired_valid_vs_control": comparisons, "candidate_ranking": ranking}


def matched_control_diagnostics(audit_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for control in (*CURRENT_CONTROLS, *MATCHED_CONTROLS):
        flow_magnitudes = []
        mask_fractions = []
        boundaries = []
        baselines = []
        chain_offsets = []
        step_offsets = []
        camera_samples = Counter()
        camera_row_count = 0
        for candidate in audit_rows:
            for camera_row in candidate["camera_evidence"]:
                evidence = camera_row["controls"].get(control)
                if evidence is None:
                    continue
                camera_row_count += 1
                chain = evidence["flow_chain"]
                flow_magnitudes.append(float(chain["chain_magnitude_pixels"]))
                if chain.get("camera_baseline_world") is not None:
                    baselines.append(float(chain["camera_baseline_world"]))
                if chain.get("matched_temporal_offset_frames") is not None:
                    chain_offsets.append(float(chain["matched_temporal_offset_frames"]))
                for step in chain["steps"]:
                    mask_fractions.append(float(step["manifest_valid_pixel_fraction"]))
                    boundaries.append(float(step["boundary_distance_pixels"]))
                    camera_samples[str(step["sample_camera"])] += 1
                    step_offsets.append(float(step["sample_source_frame"] - step["logical_source_frame"]))
        result[control] = {
            "candidate_availability_count": sum(row["control_scores"].get(control) is not None for row in audit_rows),
            "camera_evidence_row_count": camera_row_count,
            "flow_chain_magnitude_pixels": distribution(flow_magnitudes),
            "manifest_valid_mask_fraction": distribution(mask_fractions),
            "sample_boundary_distance_pixels": distribution(boundaries),
            "camera_baseline_world": distribution(baselines),
            "selected_chain_time_offset_frames": distribution(chain_offsets),
            "record_step_time_offset_frames": distribution(step_offsets),
            "sample_camera_counts": dict(sorted(camera_samples.items())),
        }
    return result


def interval_diagnostics(
    observations: Iterable[Mapping[str, Any]],
    audit_rows: Iterable[Mapping[str, Any]],
    full_result: Mapping[str, Any],
) -> dict[str, Any]:
    observation_rows = list(observations)
    candidates = list(audit_rows)
    frames = sorted({int(value["frame"]) for value in observation_rows})
    by_source = Counter(str(value["source_observation_id"]) for value in candidates)
    flow_magnitudes = []
    projected_magnitudes = []
    camera_spreads = []
    for candidate in candidates:
        camera_magnitudes = []
        for camera_row in candidate["camera_evidence"]:
            projected_magnitudes.append(float(np.linalg.norm(np.asarray(camera_row["required_projected_displacement_xy"]))))
            valid = camera_row["controls"]["valid"]
            if valid is not None:
                magnitude = float(valid["flow_magnitude_pixels"])
                flow_magnitudes.append(magnitude)
                camera_magnitudes.append(magnitude)
        if len(camera_magnitudes) > 1:
            camera_spreads.append(float(np.std(camera_magnitudes)))
    cross_order = [value for value in candidates if value["source_depth_order"] != value["destination_depth_order"]]
    admitted_cross = [value for value in cross_order if value["control_scores"]["valid"] and value["control_scores"]["valid"]["admitted"]]
    event_counts = Counter(
        event for track in full_result["tracks"] for record in track["records"] for event in record["visibility_events"]
    )
    gap_counts = Counter(int(value["frame_gap"]) for value in candidates)
    return {
        "frame_range": [min(frames), max(frames)],
        "p03_observed_frame_count": len(frames),
        "p03_unobserved_frame_count": max(frames)-min(frames)+1-len(frames),
        "observation_count": len(observation_rows),
        "candidate_count": len(candidates),
        "sources_with_multiple_candidates": sum(value > 1 for value in by_source.values()),
        "maximum_candidates_per_source": max(by_source.values(), default=0),
        "frame_gap_counts": {str(key): value for key, value in sorted(gap_counts.items())},
        "valid_flow_chain_magnitude_pixels": distribution(flow_magnitudes),
        "required_projected_displacement_pixels": distribution(projected_magnitudes),
        "within_candidate_camera_flow_magnitude_std_pixels": distribution(camera_spreads),
        "cross_order_candidate_count": len(cross_order),
        "admitted_cross_order_candidate_count": len(admitted_cross),
        "selected_reveal_event_count": int(event_counts.get("revealed", 0)),
        "selected_reappearance_event_count": int(event_counts.get("reappeared", 0)),
        "p03_temporal_surface_identity_available": False,
        "real_reveal_evaluable_from_p03_alone": False,
        "reveal_absence_interpretation": "P03 emits ordered layers per frame but no temporal surface identity; candidate cross-order opportunities are proxies and the selected association emits no rear-to-front transition",
        "human_annotations_consumed": False,
    }


def build_audit_artifact(
    scientific_payload: Mapping[str, Any],
    *,
    runtime_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_stage1b_config(config)
    content_hash = canonical_scientific_hash(scientific_payload)
    payload = {
        "schema_version": AUDIT_SCHEMA, "method_id": METHOD_ID,
        "stage1_commit": STAGE1_COMMIT,
        "scientific_content_hash": content_hash,
        "scientific_hash_contract": {
            "algorithm": validated["canonical_hash"]["algorithm"],
            "domain": SCIENTIFIC_HASH_DOMAIN, "included": "scientific_payload",
            "excluded": "runtime_metadata",
            "excluded_runtime_fields": list(validated["canonical_hash"]["excluded_runtime_fields"]),
        },
        "scientific_payload": dict(scientific_payload),
        "runtime_metadata": dict(runtime_metadata),
    }
    artifact = {
        **payload,
        "artifact_id": domain_id(
            "csvl-vpl-stage1b-v1/audit-artifact",
            {"schema_version": AUDIT_SCHEMA, "scientific_content_hash": content_hash},
        ),
    }
    validate_payload(AUDIT_SCHEMA, artifact)
    return artifact


def build_diagnostics_artifact(
    *,
    audit_scientific_content_hash: str,
    diagnostics: Mapping[str, Any],
    cpu_only_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": DIAGNOSTICS_SCHEMA, "method_id": METHOD_ID,
        "stage1_commit": STAGE1_COMMIT,
        "audit_scientific_content_hash": str(audit_scientific_content_hash),
        "diagnostics": dict(diagnostics), "cpu_only_evidence": dict(cpu_only_evidence),
    }
    artifact = {
        **payload,
        "artifact_id": domain_id("csvl-vpl-stage1b-v1/diagnostics-artifact", payload),
    }
    validate_payload(DIAGNOSTICS_SCHEMA, artifact)
    return artifact


__all__ = [
    "ABLATIONS", "AUDIT_SCHEMA", "CONFIG_SCHEMA", "CURRENT_CONTROLS",
    "DIAGNOSTICS_SCHEMA", "MATCHED_CONTROLS", "METHOD_ID", "STAGE1_COMMIT",
    "build_ablation_candidate_sets", "build_audit_artifact", "build_candidate_audit",
    "build_diagnostics_artifact", "canonical_scientific_hash", "control_definitions",
    "interval_diagnostics", "matched_control_diagnostics", "metric_definitions",
    "paired_discrimination", "replay_ablations", "validate_stage1b_config",
]
