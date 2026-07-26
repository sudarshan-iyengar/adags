"""Label-free informative-interval admission for CSVL-VPL Stage 1C Gate C0."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .association_audit import distribution
from .canonical import domain_id, sha256_file
from .errors import ProvenanceError, SchemaError


CONFIG_SCHEMA = "csvl-vpl-stage1c-c0-config-v1"
ARTIFACT_SCHEMA = "phase9-csvl-vpl-stage1c-interval-selection-v1"
DIAGNOSTICS_SCHEMA = "phase9-csvl-vpl-stage1c-c0-diagnostics-v1"
METHOD_ID = "csvl-vpl-stage1c-label-free-interval-admission-v1"
STAGE1B_FREEZE_COMMIT = "d68b25db613ae245bdd83a7b9bfcfe9f6ff608cb"
SCIENTIFIC_DOMAIN = "csvl-vpl-stage1c-c0-v1/scientific-content"


FROZEN_SELECTION_RULE: dict[str, Any] = {
    "candidate_universe": (
        "sealed Stage-1 P03 geometry-prefiltered candidates with at least two permitted "
        "cameras and valid sealed P02 mask support, before any association score or admission"
    ),
    "window_length_frames": 30,
    "window_stride_frames": 15,
    "window_endpoint_rule": "both candidate endpoints must lie in the closed window",
    "tail_rule": "include one end-aligned window when the regular stride omits it",
    "maximum_candidate_gap_frames": 5,
    "minimum_permitted_cameras_per_candidate": 2,
    "projected_support_base_tolerance_pixels": 2.0,
    "hard_requirements": {
        "front_rear_cross_order_candidate_count_minimum": 1,
        "multi_candidate_source_count_minimum": 1,
        "complete_endpoint_and_camera_provenance_required": True,
    },
    "ranking_order": [
        "cross_order_candidate_count_descending",
        "cross_order_gap_gt_one_count_descending",
        "multi_candidate_source_count_descending",
        "gap_without_intermediate_compatible_edge_count_descending",
        "camera_direction_disagreement_mean_descending",
        "flow_directional_diversity_descending",
        "candidate_count_descending",
        "start_frame_ascending",
    ],
    "development_interval_count": 1,
    "secondary_interval_count": 0,
    "score_or_admission_used": False,
    "annotations_or_evaluator_masks_used": False,
    "reconstruction_outcomes_used": False,
}


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "method_id", "stage1b_freeze_commit", "selection_rule",
        "identity_semantics", "read_boundary", "stage_scope",
    }
    if set(config) != required:
        raise SchemaError("Stage-1C C0 config has missing or unknown keys")
    if config["schema_version"] != CONFIG_SCHEMA or config["method_id"] != METHOD_ID:
        raise SchemaError("Stage-1C C0 config identity mismatch")
    if config["stage1b_freeze_commit"] != STAGE1B_FREEZE_COMMIT:
        raise SchemaError("Stage-1C C0 Stage-1B commit binding mismatch")
    if config["selection_rule"] != FROZEN_SELECTION_RULE:
        raise SchemaError("Stage-1C C0 selection rule differs from the frozen rule")
    if config["identity_semantics"] != "candidate and track identities are algorithmic hypotheses, not proven physical surfaces":
        raise SchemaError("Stage-1C C0 identity semantics changed")
    if config["stage_scope"] != "Gate_C0_only_no_association_redesign_no_GateA_authority":
        raise SchemaError("Stage-1C C0 scope changed")
    boundary = config["read_boundary"]
    if boundary != {
        "sealed_stage1_ledger": True,
        "sealed_stage1b_audit": True,
        "annotations": False,
        "evaluation_masks": False,
        "rgb": False,
        "reconstruction_outputs": False,
        "wandb": False,
    }:
        raise SchemaError("Stage-1C C0 read boundary changed")
    return copy.deepcopy(dict(config))


def assert_label_free_read_path(path: str | Path) -> None:
    lowered = str(path).lower()
    prohibited = (
        "annotation", "eval_mask", "evaluation-mask", "ground_truth", "gt_mask",
        "images/", "cam00", "wandb", "chkpnt", "render",
    )
    if any(token in lowered for token in prohibited):
        raise ProvenanceError(f"Stage-1C C0 prohibited read path: {path}")


def _window_starts(start: int, end: int, *, length: int, stride: int) -> list[int]:
    if end < start or length <= 0 or stride <= 0:
        raise SchemaError("invalid Stage-1C C0 frame/window bounds")
    if end - start + 1 <= length:
        return [start]
    latest = end - length + 1
    starts = list(range(start, latest + 1, stride))
    if starts[-1] != latest:
        starts.append(latest)
    return starts


def _direction_summary(vectors: Iterable[Iterable[float]]) -> dict[str, Any]:
    angles = []
    octants = set()
    for value in vectors:
        vector = np.asarray(value, dtype=np.float64)
        magnitude = float(np.linalg.norm(vector))
        if magnitude <= 0:
            continue
        angle = math.atan2(float(vector[1]), float(vector[0]))
        angles.append(angle)
        octants.add(int(math.floor(((angle + math.pi) % (2 * math.pi)) / (math.pi / 4))))
    if not angles:
        return {
            "nonzero_vector_count": 0,
            "occupied_octant_count": 0,
            "circular_resultant_length": None,
            "directional_diversity": None,
        }
    cosine = float(np.mean(np.cos(np.asarray(angles))))
    sine = float(np.mean(np.sin(np.asarray(angles))))
    resultant = float(math.hypot(cosine, sine))
    return {
        "nonzero_vector_count": len(angles),
        "occupied_octant_count": len(octants),
        "circular_resultant_length": resultant,
        "directional_diversity": 1.0 - resultant,
    }


def _camera_direction_disagreement(vectors: list[np.ndarray]) -> float | None:
    unit = []
    for vector in vectors:
        magnitude = float(np.linalg.norm(vector))
        if magnitude > 0:
            unit.append(vector / magnitude)
    if len(unit) < 2:
        return None
    mean = np.mean(np.stack(unit), axis=0)
    return 1.0 - float(np.linalg.norm(mean))


def _candidate_complete(row: Mapping[str, Any]) -> bool:
    if not row.get("source_p03_hypothesis_id") or not row.get("destination_p03_hypothesis_id"):
        return False
    if not row.get("source_observation_id") or not row.get("destination_observation_id"):
        return False
    cameras = row.get("camera_evidence", [])
    if len(cameras) < FROZEN_SELECTION_RULE["minimum_permitted_cameras_per_candidate"]:
        return False
    for camera in cameras:
        valid = camera.get("controls", {}).get("valid")
        if not camera.get("source_calibration_id") or not camera.get("destination_calibration_id"):
            return False
        if valid is None or not valid.get("flow_chain", {}).get("record_ids"):
            return False
    return True


def validate_candidate_universe(
    observations: Iterable[Mapping[str, Any]],
    stage1_candidates: Iterable[Mapping[str, Any]],
    audit_rows: Iterable[Mapping[str, Any]],
) -> None:
    observation_ids = [str(value["observation_id"]) for value in observations]
    if len(observation_ids) != len(set(observation_ids)):
        raise ProvenanceError("Stage-1C C0 observation IDs are not unique")
    known = set(observation_ids)
    stage1_edges = {
        (str(value["source_observation_id"]), str(value["destination_observation_id"]))
        for value in stage1_candidates
    }
    audit = list(audit_rows)
    audit_edges = {
        (str(value["source_observation_id"]), str(value["destination_observation_id"]))
        for value in audit
    }
    if len(audit_edges) != len(audit) or stage1_edges != audit_edges:
        raise ProvenanceError("Stage-1C C0 candidate universe differs from sealed Stage 1")
    if any(source not in known or destination not in known for source, destination in audit_edges):
        raise ProvenanceError("Stage-1C C0 candidate endpoint is absent from sealed observations")
    if any(int(value["frame_gap"]) > FROZEN_SELECTION_RULE["maximum_candidate_gap_frames"] for value in audit):
        raise ProvenanceError("Stage-1C C0 candidate exceeds the frozen temporal gap")


def _window_metrics(
    observations: list[Mapping[str, Any]],
    candidates: list[Mapping[str, Any]],
    *,
    start: int,
    end: int,
) -> dict[str, Any]:
    window_observations = [value for value in observations if start <= int(value["frame"]) <= end]
    window_candidates = [
        value for value in candidates
        if start <= int(value["source_frame"]) <= int(value["destination_frame"]) <= end
    ]
    source_counts = Counter(str(value["source_observation_id"]) for value in window_candidates)
    destination_frames = defaultdict(set)
    for value in window_candidates:
        destination_frames[str(value["source_observation_id"])].add(int(value["destination_frame"]))
    gaps_without_intermediate = 0
    flow_magnitudes = []
    flow_vectors: list[list[float]] = []
    mask_coverage = []
    boundary_support = []
    camera_magnitude_disagreement = []
    camera_direction_disagreement = []
    projected_overlap = []
    common_cameras = []
    complete = []
    for candidate in window_candidates:
        gap = int(candidate["frame_gap"])
        if gap > 1:
            source = str(candidate["source_observation_id"])
            source_frame = int(candidate["source_frame"])
            destination_frame = int(candidate["destination_frame"])
            if not any(source_frame < frame < destination_frame for frame in destination_frames[source]):
                gaps_without_intermediate += 1
        common_cameras.append(len(candidate["camera_evidence"]))
        complete.append(_candidate_complete(candidate))
        candidate_magnitudes = []
        candidate_vectors = []
        overlap_rows = []
        for camera in candidate["camera_evidence"]:
            valid = camera["controls"]["valid"]
            chain = valid["flow_chain"]
            vector = np.asarray(chain["chain_displacement_xy"], dtype=np.float64)
            magnitude = float(np.linalg.norm(vector))
            flow_magnitudes.append(magnitude)
            flow_vectors.append(vector.tolist())
            candidate_magnitudes.append(magnitude)
            candidate_vectors.append(vector)
            for step in chain["steps"]:
                mask_coverage.append(float(step["manifest_valid_pixel_fraction"]))
                boundary_support.append(float(step["boundary_distance_pixels"]))
            required = float(np.linalg.norm(np.asarray(camera["required_projected_displacement_xy"], dtype=np.float64)))
            radius_sum = max(
                0.0,
                float(camera["quantization_aware_tolerance_pixels"])
                - FROZEN_SELECTION_RULE["projected_support_base_tolerance_pixels"],
            )
            overlap_rows.append(required <= radius_sum)
        if len(candidate_magnitudes) > 1:
            camera_magnitude_disagreement.append(float(np.std(candidate_magnitudes)))
        direction_disagreement = _camera_direction_disagreement(candidate_vectors)
        if direction_disagreement is not None:
            camera_direction_disagreement.append(direction_disagreement)
        if overlap_rows:
            projected_overlap.append(float(np.mean(overlap_rows)))
    cross_order = [
        value for value in window_candidates
        if str(value["source_depth_order"]) != str(value["destination_depth_order"])
    ]
    cross_transitions = Counter(str(value["order_transition"]) for value in cross_order)
    observation_orders = Counter(str(value["depth_order"]) for value in window_observations)
    frame_gaps = [int(value["frame_gap"]) for value in window_candidates]
    direction = _direction_summary(flow_vectors)
    completeness = float(np.mean(complete)) if complete else 0.0
    requirements = FROZEN_SELECTION_RULE["hard_requirements"]
    rejection_reasons = []
    if len(cross_order) < requirements["front_rear_cross_order_candidate_count_minimum"]:
        rejection_reasons.append("zero_front_rear_cross_order_candidates")
    if sum(count > 1 for count in source_counts.values()) < requirements["multi_candidate_source_count_minimum"]:
        rejection_reasons.append("no_source_with_multiple_plausible_candidates")
    if requirements["complete_endpoint_and_camera_provenance_required"] and completeness != 1.0:
        rejection_reasons.append("incomplete_endpoint_or_camera_provenance")
    return {
        "start_frame": start,
        "end_frame": end,
        "observation_count": len(window_observations),
        "observation_frame_count": len({int(value["frame"]) for value in window_observations}),
        "observation_depth_order_counts": dict(sorted(observation_orders.items())),
        "candidate_count": len(window_candidates),
        "candidate_source_count": len(source_counts),
        "multi_candidate_source_count": sum(count > 1 for count in source_counts.values()),
        "candidates_per_source": distribution(source_counts.values()),
        "cross_order_candidate_count": len(cross_order),
        "cross_order_gap_gt_one_count": sum(int(value["frame_gap"]) > 1 for value in cross_order),
        "cross_order_transition_counts": dict(sorted(cross_transitions.items())),
        "gap_without_intermediate_compatible_edge_count": gaps_without_intermediate,
        "temporal_gap_frames": distribution(frame_gaps),
        "permitted_camera_support": distribution(common_cameras),
        "p02_manifest_valid_mask_fraction": distribution(mask_coverage),
        "p02_sample_boundary_distance_pixels": distribution(boundary_support),
        "flow_chain_magnitude_pixels": distribution(flow_magnitudes),
        "flow_directional_diversity": direction,
        "camera_magnitude_disagreement_pixels": distribution(camera_magnitude_disagreement),
        "camera_direction_disagreement": distribution(camera_direction_disagreement),
        "projected_quantization_support_overlap_possible_fraction": distribution(projected_overlap),
        "complete_candidate_provenance_fraction": completeness,
        "admissible": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "interpretation_boundary": {
            "gap_metric": "a compatible later edge with no earlier compatible destination from that source; not proven physical disappearance",
            "projected_overlap_metric": "disk-overlap possibility from projected P03 quantization radii; not observed mask IoU",
            "candidate_identity": "P03 geometry candidate only; no temporal identity or association score used",
        },
    }


def scan_intervals(
    observations: Iterable[Mapping[str, Any]],
    audit_rows: Iterable[Mapping[str, Any]],
    *,
    frame_range: tuple[int, int],
) -> dict[str, Any]:
    items = sorted((dict(value) for value in observations), key=lambda value: (int(value["frame"]), str(value["observation_id"])))
    candidates = sorted((dict(value) for value in audit_rows), key=lambda value: str(value["edge_key"]))
    start, end = (int(value) for value in frame_range)
    starts = _window_starts(
        start,
        end,
        length=FROZEN_SELECTION_RULE["window_length_frames"],
        stride=FROZEN_SELECTION_RULE["window_stride_frames"],
    )
    windows = [
        _window_metrics(
            items,
            candidates,
            start=value,
            end=min(end, value + FROZEN_SELECTION_RULE["window_length_frames"] - 1),
        )
        for value in starts
    ]
    admissible = [value for value in windows if value["admissible"]]
    def rank(value: Mapping[str, Any]) -> tuple[Any, ...]:
        direction = value["camera_direction_disagreement"]["mean"]
        diversity = value["flow_directional_diversity"]["directional_diversity"]
        return (
            -int(value["cross_order_candidate_count"]),
            -int(value["cross_order_gap_gt_one_count"]),
            -int(value["multi_candidate_source_count"]),
            -int(value["gap_without_intermediate_compatible_edge_count"]),
            -float(direction if direction is not None else -1.0),
            -float(diversity if diversity is not None else -1.0),
            -int(value["candidate_count"]),
            int(value["start_frame"]),
        )
    ordered = sorted(admissible, key=rank)
    selected = ordered[: FROZEN_SELECTION_RULE["development_interval_count"]]
    selected_keys = {(value["start_frame"], value["end_frame"]) for value in selected}
    for value in windows:
        value["selected_as_development_interval"] = (value["start_frame"], value["end_frame"]) in selected_keys
    return {
        "selection_rule": copy.deepcopy(FROZEN_SELECTION_RULE),
        "window_count": len(windows),
        "admissible_window_count": len(admissible),
        "selected_intervals": selected,
        "rejected_intervals": [value for value in windows if not value["admissible"]],
        "all_windows": windows,
        "gate_c0_admitted": bool(selected),
        "gate_c0_reason": (
            "label_free_cross_order_and_ambiguity_requirements_satisfied"
            if selected else
            "no_window_contains_a_front_rear_cross_order_candidate_with_required_ambiguity_and_complete_provenance"
        ),
    }


def build_scientific_payload(
    *,
    stage1_ledger: Mapping[str, Any],
    stage1_ledger_path: str | Path,
    stage1b_audit: Mapping[str, Any],
    stage1b_audit_path: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_config(config)
    if stage1_ledger.get("schema_version") != "phase9-csvl-vpl-stage1-ledger-v1":
        raise ProvenanceError("Stage-1C C0 Stage-1 ledger schema mismatch")
    if stage1b_audit.get("schema_version") != "phase9-csvl-vpl-stage1b-association-audit-v1":
        raise ProvenanceError("Stage-1C C0 Stage-1B audit schema mismatch")
    stage1_payload = stage1_ledger.get("scientific_payload", {})
    stage1b_payload = stage1b_audit.get("scientific_payload", {})
    if stage1_payload.get("scene") != stage1b_payload.get("scene"):
        raise ProvenanceError("Stage-1C C0 input scene mismatch")
    bound_stage1 = stage1b_payload.get("input_bindings", {}).get("stage1_ledger", {})
    stage1_sha = sha256_file(stage1_ledger_path)
    stage1b_sha = sha256_file(stage1b_audit_path)
    if bound_stage1.get("sha256") != stage1_sha:
        raise ProvenanceError("Stage-1C C0 Stage-1B audit does not bind the supplied Stage-1 ledger")
    observations = stage1_payload.get("observations", [])
    stage1_candidates = stage1_payload.get("valid_candidate_evidence", [])
    audit_rows = stage1b_payload.get("candidate_score_audit", [])
    validate_candidate_universe(observations, stage1_candidates, audit_rows)
    frame_range = tuple(int(value) for value in stage1_payload.get("frame_range", []))
    if len(frame_range) != 2:
        raise ProvenanceError("Stage-1C C0 sealed frame range is malformed")
    selection = scan_intervals(observations, audit_rows, frame_range=frame_range)
    p02_records = stage1b_payload.get("input_bindings", {}).get("consumed_p02_flow_records", [])
    transitive = {
        key: copy.deepcopy(value)
        for key, value in stage1b_payload.get("input_bindings", {}).items()
        if key not in {"consumed_p02_flow_records"}
    }
    return {
        "method_id": METHOD_ID,
        "stage_scope": validated["stage_scope"],
        "stage1b_freeze_commit": STAGE1B_FREEZE_COMMIT,
        "scene": stage1_payload["scene"],
        "identity_semantics": validated["identity_semantics"],
        "selection": selection,
        "candidate_universe": {
            "observation_count": len(observations),
            "candidate_count": len(audit_rows),
            "source": FROZEN_SELECTION_RULE["candidate_universe"],
            "association_scores_read_for_selection": False,
            "association_admission_flags_read_for_selection": False,
        },
        "input_bindings": {
            "stage1_ledger": {"path": str(Path(stage1_ledger_path).resolve()), "schema": stage1_ledger["schema_version"], "sha256": stage1_sha, "scientific_content_hash": stage1_ledger["scientific_content_hash"]},
            "stage1b_audit": {"path": str(Path(stage1b_audit_path).resolve()), "schema": stage1b_audit["schema_version"], "sha256": stage1b_sha, "scientific_content_hash": stage1b_audit["scientific_content_hash"]},
            "transitive_sealed_bindings": transitive,
            "consumed_p02_record_count": len(p02_records),
            "consumed_p02_record_binding_content_hash": domain_id("csvl-vpl-stage1c-c0-v1/p02-bindings", p02_records),
        },
        "read_boundary": {
            **validated["read_boundary"],
            "scientific_data_files_opened": [str(Path(stage1_ledger_path).resolve()), str(Path(stage1b_audit_path).resolve())],
            "annotations_inspected_after_freeze": False,
        },
        "interpretation_boundary": {
            "gate_c0_only": True,
            "association_redesign_implemented": False,
            "tracking_run": False,
            "physical_surface_identity_claim": False,
            "gate_a_authority": False,
        },
    }


def canonical_scientific_hash(scientific_payload: Mapping[str, Any]) -> str:
    return domain_id(SCIENTIFIC_DOMAIN, scientific_payload)


def build_interval_artifact(
    scientific_payload: Mapping[str, Any],
    *,
    runtime_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validate_config(config)
    scientific = copy.deepcopy(dict(scientific_payload))
    content_hash = canonical_scientific_hash(scientific)
    artifact_id = domain_id("csvl-vpl-stage1c-c0-v1/artifact", {"scientific_content_hash": content_hash})
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "method_id": METHOD_ID,
        "artifact_id": artifact_id,
        "stage1b_freeze_commit": STAGE1B_FREEZE_COMMIT,
        "scientific_hash_contract": {
            "domain": SCIENTIFIC_DOMAIN,
            "included": "scientific_payload",
            "excluded": ["runtime_metadata.timestamp_utc", "runtime_metadata.slurm_job_id", "runtime_metadata.absolute_output_root"],
        },
        "scientific_content_hash": content_hash,
        "scientific_payload": scientific,
        "runtime_metadata": copy.deepcopy(dict(runtime_metadata)),
    }


__all__ = [
    "ARTIFACT_SCHEMA", "CONFIG_SCHEMA", "DIAGNOSTICS_SCHEMA", "FROZEN_SELECTION_RULE",
    "METHOD_ID", "STAGE1B_FREEZE_COMMIT", "assert_label_free_read_path",
    "build_interval_artifact", "build_scientific_payload", "canonical_scientific_hash",
    "scan_intervals", "validate_candidate_universe", "validate_config",
]
