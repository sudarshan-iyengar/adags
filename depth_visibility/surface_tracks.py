"""CSVL-VPL Stage-1 surface-hypothesis association and immutable ledger.

Track identifiers produced here are deterministic algorithmic association
hypotheses.  They are never physical-surface labels.  Current geometry and
depth order are emitted only for a P03 observation; dormancy retains an
identity descriptor and provenance pointer, never hidden xyz or hidden order.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .camera import project_world, unproject_optical_z
from .canonical import domain_id, sha256_file
from .errors import ArtifactError, FlowSemanticsError, ProvenanceError, SchemaError
from .flow import bilinear_flow, validate_p02_flow_record
from .schema import assert_finite_tree, validate_payload


METHOD_ID = "csvl-vpl-stage1-surface-hypothesis-ledger-v1"
CONFIG_SCHEMA = "csvl-vpl-stage1-config-v1"
LEDGER_SCHEMA = "phase9-csvl-vpl-stage1-ledger-v1"
DIAGNOSTICS_SCHEMA = "phase9-csvl-vpl-stage1-diagnostics-v1"
SCIENTIFIC_HASH_DOMAIN = "csvl-vpl-stage1-v1/scientific-content"
TRACK_DOMAIN = "csvl-vpl-stage1-v1/surface-hypothesis-track"
OBSERVATION_DOMAIN = "csvl-vpl-stage1-v1/p03-layer-observation"
EDGE_DOMAIN = "csvl-vpl-stage1-v1/association-edge"

CONTROL_MODES = (
    "valid",
    "corrupted_flow",
    "reversed_flow",
    "camera_swap",
    "temporal_offset",
)


def validate_stage1_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "method_id", "identity_semantics", "association",
        "dormancy", "controls", "canonical_hash", "prohibited_reads",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise SchemaError(
            f"Stage-1 config keys mismatch: missing={sorted(required - set(payload)) if isinstance(payload, Mapping) else sorted(required)} "
            f"unknown={sorted(set(payload) - required) if isinstance(payload, Mapping) else []}"
        )
    if payload["schema_version"] != CONFIG_SCHEMA or payload["method_id"] != METHOD_ID:
        raise SchemaError("Stage-1 config identity mismatch")
    if payload["identity_semantics"] != "algorithmic_association_hypothesis_not_physical_identity":
        raise SchemaError("Stage-1 track identity semantics cannot be weakened")
    association_required = {
        "minimum_cameras", "maximum_world_displacement_rscene_per_frame",
        "base_endpoint_tolerance_pixels", "search_cost_maximum",
        "second_best_ratio_minimum", "endpoint_cost_weight",
        "geometry_cost_weight", "p03_target_bin_pixels",
    }
    association = payload["association"]
    if not isinstance(association, Mapping) or set(association) != association_required:
        raise SchemaError("Stage-1 association config has missing or unknown keys")
    if int(association["minimum_cameras"]) < 2:
        raise SchemaError("Stage-1 association requires at least two cameras")
    positive = (
        "maximum_world_displacement_rscene_per_frame",
        "base_endpoint_tolerance_pixels",
        "search_cost_maximum",
        "second_best_ratio_minimum",
    )
    if any(not math.isfinite(float(association[key])) or float(association[key]) <= 0 for key in positive):
        raise SchemaError("Stage-1 association thresholds must be finite and positive")
    weights = float(association["endpoint_cost_weight"]), float(association["geometry_cost_weight"])
    if any(not math.isfinite(value) or value < 0 for value in weights) or not math.isclose(sum(weights), 1.0):
        raise SchemaError("Stage-1 association cost weights must be nonnegative and sum to one")
    if int(association["p03_target_bin_pixels"]) <= 0:
        raise SchemaError("Stage-1 P03 bin size must be positive")
    dormancy = payload["dormancy"]
    if not isinstance(dormancy, Mapping) or set(dormancy) != {"maximum_frames"}:
        raise SchemaError("Stage-1 dormancy config mismatch")
    if int(dormancy["maximum_frames"]) < 0:
        raise SchemaError("Stage-1 dormancy bound cannot be negative")
    if list(payload["controls"]) != list(CONTROL_MODES[1:]):
        raise SchemaError("Stage-1 invalid-control roster changed")
    hash_contract = payload["canonical_hash"]
    if not isinstance(hash_contract, Mapping) or set(hash_contract) != {
        "algorithm", "domain", "excluded_runtime_fields"
    }:
        raise SchemaError("Stage-1 canonical hash contract mismatch")
    if hash_contract["algorithm"] != "sha256_domain_separated_csvl_cjson_v1" or hash_contract["domain"] != SCIENTIFIC_HASH_DOMAIN:
        raise SchemaError("Stage-1 canonical hash algorithm/domain mismatch")
    required_exclusions = {"timestamp_utc", "slurm_job_id", "absolute_output_root"}
    if set(hash_contract["excluded_runtime_fields"]) != required_exclusions:
        raise SchemaError("Stage-1 runtime metadata exclusions changed")
    prohibited = {str(value) for value in payload["prohibited_reads"]}
    if not {"cam00_rgb", "annotations", "evaluation_masks", "wandb"}.issubset(prohibited):
        raise SchemaError("Stage-1 prohibited-read contract is incomplete")
    assert_finite_tree(payload)
    return dict(payload)


def canonical_scientific_hash(scientific_payload: Mapping[str, Any]) -> str:
    """Hash deterministic scientific content only, excluding runtime metadata."""

    assert_finite_tree(scientific_payload)
    return domain_id(SCIENTIFIC_HASH_DOMAIN, scientific_payload)


def _sha256_contiguous(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


class P02FlowStore:
    """Lazy hash-verifying reader for immutable P02 NPZ flow records."""

    def __init__(self, manifest: Mapping[str, Any], *, manifest_path: Path, manifest_sha256: str):
        if (
            manifest.get("schema_version") != "phase9-flow-manifest-v1"
            or manifest.get("direction") != "forward_t_to_t_plus_1"
            or manifest.get("cam00_rgb_opened") is not False
            or manifest.get("label_dependent_gate_a") != "not_evaluable"
        ):
            raise FlowSemanticsError("sealed P02 manifest is incompatible with Stage 1")
        if sha256_file(manifest_path) != manifest_sha256:
            raise ArtifactError("P02 manifest file hash mismatch")
        records: dict[tuple[str, int], dict[str, Any]] = {}
        for raw in manifest.get("records", []):
            record = validate_p02_flow_record(raw)
            key = str(record["source_camera"]), int(record["source_frame"])
            if key in records:
                raise FlowSemanticsError(f"duplicate P02 flow record: {key}")
            records[key] = record
        if len(records) != int(manifest.get("record_count", -1)) or len(records) != int(manifest.get("expected_record_count", -2)):
            raise FlowSemanticsError("P02 manifest record counts are incomplete")
        cameras = tuple(sorted(str(value) for value in manifest.get("camera_ids", [])))
        if not cameras or "cam00" in cameras or set(cameras) != {key[0] for key in records}:
            raise FlowSemanticsError("P02 manifest camera roster is incomplete or prohibited")
        self.manifest = dict(manifest)
        self.manifest_path = manifest_path.resolve()
        self.manifest_sha256 = manifest_sha256
        self.records = records
        self.cameras = cameras
        self._cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
        self._consumed: dict[str, dict[str, Any]] = {}

    def swapped_camera(self, camera: str) -> str:
        try:
            index = self.cameras.index(camera)
        except ValueError as exc:
            raise FlowSemanticsError(f"camera absent from P02 roster: {camera}") from exc
        return self.cameras[(index + 1) % len(self.cameras)]

    def _load(self, camera: str, frame: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        key = str(camera), int(frame)
        if key in self._cache:
            return self._cache[key]
        try:
            record = self.records[key]
        except KeyError as exc:
            raise FlowSemanticsError(f"missing sealed P02 flow record: {key}") from exc
        path = Path(str(record["flow_npz_path"])).resolve()
        hashes = record["array_hashes"]
        if not path.is_file() or sha256_file(path) != hashes["npz_sha256"]:
            raise ArtifactError(f"P02 flow NPZ missing or hash-mismatched: {path}")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if record["flow_key"] not in archive or record["valid_key"] not in archive:
                    raise ArtifactError(f"P02 flow NPZ lacks required arrays: {path}")
                flow = np.asarray(archive[record["flow_key"]])
                valid = np.asarray(archive[record["valid_key"]])
        except (OSError, ValueError) as exc:
            raise ArtifactError(f"P02 flow NPZ cannot be decoded safely: {path}") from exc
        expected_shape = (int(record["source_height"]), int(record["source_width"]))
        if flow.shape != expected_shape + (2,) or flow.dtype != np.dtype("float32") or not np.isfinite(flow).all():
            raise ArtifactError("P02 flow array shape/dtype/finiteness mismatch")
        if valid.shape != expected_shape or valid.dtype != np.dtype("bool"):
            raise ArtifactError("P02 flow validity array shape/dtype mismatch")
        if _sha256_contiguous(flow) != hashes["flow_contiguous_sha256"] or _sha256_contiguous(valid) != hashes["mask_contiguous_sha256"]:
            raise ArtifactError("P02 flow array content hash mismatch")
        record_id = domain_id(
            "csvl-vpl-stage1-v1/p02-flow-record",
            {
                "camera": camera,
                "source_frame": int(frame),
                "target_frame": int(record["target_frame"]),
                "npz_sha256": hashes["npz_sha256"],
            },
        )
        reference = {
            "path": str(path),
            "schema": str(record["schema_version"]),
            "record_id": record_id,
            "source_camera": camera,
            "source_frame": int(frame),
            "target_frame": int(record["target_frame"]),
            "sha256": hashes["npz_sha256"],
            "flow_contiguous_sha256": hashes["flow_contiguous_sha256"],
            "validity_contiguous_sha256": hashes["mask_contiguous_sha256"],
            "generator_revision": str(record["generator_revision"]),
        }
        self._consumed[str(path)] = reference
        self._cache[key] = flow, valid, reference
        return self._cache[key]

    def sample_chain(
        self,
        camera: str,
        source_frame: int,
        target_frame: int,
        source_xy: np.ndarray,
        *,
        mode: str = "valid",
    ) -> dict[str, Any]:
        if mode not in CONTROL_MODES:
            raise FlowSemanticsError(f"unknown flow control mode: {mode}")
        if target_frame <= source_frame:
            raise FlowSemanticsError("flow chain must advance in time")
        sample_camera = self.swapped_camera(camera) if mode == "camera_swap" else camera
        position = np.asarray(source_xy, dtype=np.float64)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise FlowSemanticsError("flow-chain source pixel must be a finite 2-vector")
        references = []
        step_vectors = []
        for frame in range(int(source_frame), int(target_frame)):
            record_frame = frame + 1 if mode == "temporal_offset" else frame
            try:
                flow, valid, reference = self._load(sample_camera, record_frame)
            except FlowSemanticsError:
                return {"valid": False, "reason": "missing_flow_record", "record_refs": references}
            vector = bilinear_flow(flow, float(position[0]), float(position[1]), valid)
            if vector is None:
                return {"valid": False, "reason": "invalid_flow_sample", "record_refs": references}
            vector = np.asarray(vector, dtype=np.float64)
            if mode == "reversed_flow":
                vector = -vector
            elif mode == "corrupted_flow":
                vector = np.array([vector[1] + 16.0, vector[0] - 16.0], dtype=np.float64)
            position = position + vector
            step_vectors.append(vector.tolist())
            references.append(reference["record_id"])
        return {
            "valid": True,
            "destination_xy": position.tolist(),
            "step_vectors": step_vectors,
            "record_refs": references,
            "sample_camera": sample_camera,
        }

    def consumed_artifacts(self) -> list[dict[str, Any]]:
        return [self._consumed[key] for key in sorted(self._consumed)]

    def load_bound_step_for_audit(
        self, camera: str, frame: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
        """Expose one already sealed P02 step to read-only forensic instrumentation."""

        flow, valid, reference = self._load(str(camera), int(frame))
        return flow, valid, dict(reference), dict(self.records[(str(camera), int(frame))])


def _bin_center(target_bin: Iterable[int], size: int) -> np.ndarray:
    x, y = (int(value) for value in target_bin)
    if x < 0 or y < 0 or size <= 0:
        raise ProvenanceError("P03 target bin is invalid")
    return np.array([x * size + 0.5 * (size - 1), y * size + 0.5 * (size - 1)], dtype=np.float64)


def extract_p03_observations(
    p03: Mapping[str, Any],
    *,
    target_records: Mapping[tuple[str, int], Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Convert immutable P03 layers into explicit observed hypotheses."""

    validate_stage1_config(config)
    if (
        p03.get("schema_version") != "phase9-csvl-ledger-v1"
        or p03.get("cam00_rgb_opened") is not False
        or p03.get("label_dependent_gate_a") != "not_evaluable"
        or p03.get("evidence_boundary", {}).get("human_labels") != "not_consumed"
        or p03.get("evidence_boundary", {}).get("temporal_identity_status") != "not_propagated_in_p03_v7"
    ):
        raise ProvenanceError("P03 evidence boundary is incompatible with Stage 1")
    target_camera = str(p03["target_camera"])
    if target_camera != "cam00":
        raise ProvenanceError("Stage 1 expects held-out cam00 calibration only")
    bin_size = int(config["association"]["p03_target_bin_pixels"])
    observations = []
    for frame_record in sorted(p03.get("frames", []), key=lambda item: int(item["frame"])):
        frame = int(frame_record["frame"])
        target = target_records.get((target_camera, frame))
        if target is None or target.image_path is not None:
            raise ProvenanceError("Stage 1 target camera must expose calibration but no RGB path")
        for hypothesis in sorted(frame_record.get("ordered_multilayer_bins", []), key=lambda item: str(item["csvl_hypothesis_id"])):
            if int(hypothesis["target_bin_pixels"]) != bin_size:
                raise ProvenanceError("P03 target-bin size differs from Stage-1 authority")
            physical_ancestry = sorted(str(value) for value in hypothesis.get("physical_ancestry", []))
            if not physical_ancestry or target_camera in physical_ancestry:
                raise ProvenanceError("P03 physical ancestry is incomplete or contains cam00")
            center = _bin_center(hypothesis["target_bin"], bin_size)
            layers = sorted(hypothesis.get("layers", []), key=lambda item: int(item["layer_ordinal"]))
            if len(layers) < 2:
                raise ProvenanceError("P03 ordered hypothesis no longer contains multiple layers")
            ordinals = [int(layer["layer_ordinal"]) for layer in layers]
            depths = [float(layer["median_optical_z"]) for layer in layers]
            if ordinals != list(range(len(layers))) or any(
                layer.get("depth_order") != "front_to_rear" for layer in layers
            ):
                raise ProvenanceError("P03 layer ordinals/depth-order semantics are inverted or incomplete")
            if any(not math.isfinite(value) or value <= 0 for value in depths) or any(
                rear <= front for front, rear in zip(depths, depths[1:])
            ):
                raise ProvenanceError("P03 front-to-rear optical-z order is inverted")
            for layer in layers:
                ordinal = int(layer["layer_ordinal"])
                depth = float(layer["median_optical_z"])
                risk = float(layer["median_risk"])
                if not math.isfinite(depth) or depth <= 0 or not math.isfinite(risk) or not 0 <= risk <= 1:
                    raise ProvenanceError("P03 layer depth/risk violates its domain")
                cameras = sorted(str(value) for value in layer.get("physical_cameras", []))
                if len(cameras) < int(config["association"]["minimum_cameras"]) or not set(cameras).issubset(physical_ancestry) or target_camera in cameras:
                    raise ProvenanceError("P03 layer camera ancestry is incomplete or prohibited")
                world = unproject_optical_z(target.K, target.w2c_opencv, center, depth)
                payload = {
                    "p03_hypothesis_id": str(hypothesis["csvl_hypothesis_id"]),
                    "layer_ordinal": ordinal,
                    "frame": frame,
                    "target_camera": target_camera,
                }
                observations.append(
                    {
                        "observation_id": domain_id(OBSERVATION_DOMAIN, payload),
                        "scene": str(p03["scene"]),
                        "frame": frame,
                        "time": float(target.time),
                        "target_camera": target_camera,
                        "target_bin": [int(value) for value in hypothesis["target_bin"]],
                        "target_bin_pixels": bin_size,
                        "target_pixel_center": center.tolist(),
                        "layer_ordinal": ordinal,
                        "depth_order": "front" if ordinal == 0 else "rear",
                        "target_visibility_state": "visible" if ordinal == 0 else "occluded",
                        "median_optical_z": depth,
                        "world_xyz": np.asarray(world, dtype=np.float64).tolist(),
                        "geometry_status": "directly_observed_p03_evidence_unprojected_at_bin_center",
                        "observation_risk": risk,
                        "uncertainty": {
                            "p03_median_risk": risk,
                            "target_bin_quantization_half_width_pixels": 0.5 * bin_size,
                        },
                        "physical_camera_ancestry": cameras,
                        "camera_time_ancestry": [
                            {"camera_id": camera, "frame": frame, "source": "P03_observed_geometry"}
                            for camera in cameras
                        ],
                        "source_observations": {
                            "p03_hypothesis_id": str(hypothesis["csvl_hypothesis_id"]),
                            "p03_frame_ledger_id": str(frame_record["frame_ledger_id"]),
                            "source_da3_group_ids": sorted(str(value) for value in hypothesis["source_da3_group_ids"]),
                            "sample_count": int(layer["sample_count"]),
                            "physical_camera_count": int(layer["physical_camera_count"]),
                        },
                        "order_evidence": {
                            "layer_ordinal": ordinal,
                            "depth_order": "front_to_rear",
                            "order_pairs": [dict(value) for value in hypothesis["order_pairs"]],
                            "provenance": "P03_ordered_multilayer_bin",
                        },
                    }
                )
    observations.sort(key=lambda item: (int(item["frame"]), str(item["observation_id"])))
    if len({str(item["observation_id"]) for item in observations}) != len(observations):
        raise ProvenanceError("P03 observation IDs are not unique")
    return observations


def _projected_bin_radius(observation: Mapping[str, Any], target_record: Any, camera_record: Any) -> float:
    size = int(observation["target_bin_pixels"])
    center = np.asarray(observation["target_pixel_center"], dtype=np.float64)
    half = 0.5 * size
    corners = np.array(
        [[center[0] - half, center[1] - half], [center[0] + half, center[1] - half],
         [center[0] - half, center[1] + half], [center[0] + half, center[1] + half]],
        dtype=np.float64,
    )
    world = unproject_optical_z(
        target_record.K,
        target_record.w2c_opencv,
        corners,
        np.full(4, float(observation["median_optical_z"]), dtype=np.float64),
    )
    pixels, _ = project_world(camera_record.K, camera_record.w2c_opencv, world)
    projected_center, _ = project_world(
        camera_record.K, camera_record.w2c_opencv, np.asarray(observation["world_xyz"], dtype=np.float64)
    )
    return float(np.max(np.linalg.norm(pixels - projected_center[None, :], axis=1)))


def candidate_evidence(
    source: Mapping[str, Any],
    destination: Mapping[str, Any],
    *,
    train_records: Mapping[tuple[str, int], Any],
    target_records: Mapping[tuple[str, int], Any],
    flow_store: P02FlowStore,
    r_scene: float,
    config: Mapping[str, Any],
    mode: str = "valid",
) -> dict[str, Any] | None:
    """Measure one association candidate without assigning identity."""

    association = config["association"]
    source_frame = int(source["frame"])
    target_frame = int(destination["frame"])
    gap = target_frame - source_frame
    if gap <= 0 or gap > int(config["dormancy"]["maximum_frames"]):
        return None
    common_cameras = sorted(
        set(source["physical_camera_ancestry"]) & set(destination["physical_camera_ancestry"])
    )
    if len(common_cameras) < int(association["minimum_cameras"]):
        return None
    displacement = float(
        np.linalg.norm(
            np.asarray(source["world_xyz"], dtype=np.float64)
            - np.asarray(destination["world_xyz"], dtype=np.float64)
        )
    )
    maximum_displacement = float(association["maximum_world_displacement_rscene_per_frame"]) * r_scene * gap
    if displacement > maximum_displacement:
        return None
    endpoint_rows = []
    flow_record_ids: set[str] = set()
    source_target = target_records[(str(source["target_camera"]), source_frame)]
    destination_target = target_records[(str(destination["target_camera"]), target_frame)]
    for camera in common_cameras:
        source_camera = train_records.get((camera, source_frame))
        destination_camera = train_records.get((camera, target_frame))
        if source_camera is None or destination_camera is None:
            continue
        try:
            source_xy, _ = project_world(
                source_camera.K,
                source_camera.w2c_opencv,
                np.asarray(source["world_xyz"], dtype=np.float64),
            )
            destination_xy, _ = project_world(
                destination_camera.K,
                destination_camera.w2c_opencv,
                np.asarray(destination["world_xyz"], dtype=np.float64),
            )
            if not (
                0 <= source_xy[0] <= source_camera.width - 1
                and 0 <= source_xy[1] <= source_camera.height - 1
                and 0 <= destination_xy[0] <= destination_camera.width - 1
                and 0 <= destination_xy[1] <= destination_camera.height - 1
            ):
                continue
            chain = flow_store.sample_chain(camera, source_frame, target_frame, source_xy, mode=mode)
            if not chain["valid"]:
                continue
            predicted = np.asarray(chain["destination_xy"], dtype=np.float64)
            error = float(np.linalg.norm(predicted - destination_xy))
            source_radius = _projected_bin_radius(source, source_target, source_camera)
            destination_radius = _projected_bin_radius(destination, destination_target, destination_camera)
            tolerance = float(association["base_endpoint_tolerance_pixels"]) + source_radius + destination_radius
            normalized = error / tolerance
        except (ValueError, FlowSemanticsError):
            continue
        endpoint_rows.append(
            {
                "camera_id": camera,
                "sample_camera_id": str(chain["sample_camera"]),
                "endpoint_error_pixels": error,
                "quantization_aware_tolerance_pixels": tolerance,
                "normalized_endpoint_error": normalized,
                "flow_record_ids": list(chain["record_refs"]),
            }
        )
        flow_record_ids.update(str(value) for value in chain["record_refs"])
    minimum = int(association["minimum_cameras"])
    if len(endpoint_rows) < minimum:
        return None
    normalized_errors = np.asarray(
        [float(value["normalized_endpoint_error"]) for value in endpoint_rows], dtype=np.float64
    )
    endpoint_errors = np.asarray(
        [float(value["endpoint_error_pixels"]) for value in endpoint_rows], dtype=np.float64
    )
    endpoint_cost = float(np.median(normalized_errors))
    geometry_cost = displacement / maximum_displacement
    cost = (
        float(association["endpoint_cost_weight"]) * min(2.0, endpoint_cost)
        + float(association["geometry_cost_weight"]) * min(2.0, geometry_cost)
    )
    missing_camera_risk = 1.0 - len(endpoint_rows) / len(common_cameras)
    association_risk = max(min(1.0, endpoint_cost), missing_camera_risk)
    payload = {
        "source_observation_id": str(source["observation_id"]),
        "destination_observation_id": str(destination["observation_id"]),
        "mode": mode,
        "flow_record_ids": sorted(flow_record_ids),
    }
    return {
        **payload,
        "candidate_id": domain_id(EDGE_DOMAIN, payload),
        "source_frame": source_frame,
        "destination_frame": target_frame,
        "frame_gap": gap,
        "common_camera_count": len(common_cameras),
        "valid_camera_count": len(endpoint_rows),
        "camera_evidence": endpoint_rows,
        "endpoint_error_pixels_median": float(np.median(endpoint_errors)),
        "endpoint_error_pixels_maximum": float(np.max(endpoint_errors)),
        "normalized_endpoint_error_median": endpoint_cost,
        "world_displacement": displacement,
        "world_displacement_rscene_per_frame": displacement / (r_scene * gap),
        "cost": cost,
        "association_risk": association_risk,
        "association_confidence": 1.0 - association_risk,
        "admitted": bool(cost <= float(association["search_cost_maximum"])),
    }


def build_candidate_evidence(
    observations: Iterable[Mapping[str, Any]],
    *,
    train_records: Mapping[tuple[str, int], Any],
    target_records: Mapping[tuple[str, int], Any],
    flow_store: P02FlowStore,
    r_scene: float,
    config: Mapping[str, Any],
    mode: str = "valid",
) -> list[dict[str, Any]]:
    items = sorted(observations, key=lambda item: (int(item["frame"]), str(item["observation_id"])))
    output = []
    maximum_gap = int(config["dormancy"]["maximum_frames"])
    by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        by_frame[int(item["frame"])].append(item)
    for source in items:
        for frame in range(int(source["frame"]) + 1, int(source["frame"]) + maximum_gap + 1):
            for destination in by_frame.get(frame, []):
                evidence = candidate_evidence(
                    source,
                    destination,
                    train_records=train_records,
                    target_records=target_records,
                    flow_store=flow_store,
                    r_scene=r_scene,
                    config=config,
                    mode=mode,
                )
                if evidence is not None:
                    output.append(evidence)
    return sorted(output, key=lambda item: str(item["candidate_id"]))


def _camera_states(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    states = [
        {
            "camera_id": str(observation["target_camera"]),
            "frame": int(observation["frame"]),
            "visibility_state": str(observation["target_visibility_state"]),
            "depth_order": str(observation["depth_order"]),
            "evidence": "P03_target_projection_order",
            "rgb_opened": False,
        }
    ]
    for camera in sorted(str(value) for value in observation["physical_camera_ancestry"]):
        states.append(
            {
                "camera_id": camera,
                "frame": int(observation["frame"]),
                "visibility_state": "observed_geometry_visibility_not_emitted_by_p03",
                "depth_order": "unknown_not_emitted_by_p03",
                "evidence": "P03_physical_camera_support",
                "rgb_opened": False,
            }
        )
    return states


def associate_observations(
    observations: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    frame_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Assign deterministic hypothesis tracks with conservative abstention."""

    validate_stage1_config(config)
    items = sorted((dict(value) for value in observations), key=lambda item: (int(item["frame"]), str(item["observation_id"])))
    by_id = {str(item["observation_id"]): item for item in items}
    if len(by_id) != len(items):
        raise ProvenanceError("surface observation IDs must be unique")
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_frame[int(item["frame"])].append(item)
    if frame_range is None:
        if not items:
            raise ProvenanceError("Stage-1 association requires observations")
        frame_range = min(by_frame), max(by_frame)
    start_frame, end_frame = (int(value) for value in frame_range)
    if end_frame < start_frame:
        raise ProvenanceError("Stage-1 frame range is reversed")
    candidates = [dict(value) for value in candidate_rows]
    for candidate in candidates:
        if candidate["source_observation_id"] not in by_id or candidate["destination_observation_id"] not in by_id:
            raise ProvenanceError("candidate endpoint is absent from observations")
    admitted_by_destination: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if bool(candidate.get("admitted", False)):
            admitted_by_destination[str(candidate["destination_observation_id"])].append(candidate)

    tracks: dict[str, dict[str, Any]] = {}
    observation_track: dict[str, str] = {}
    active: dict[str, str] = {}
    abstentions = []
    maximum_dormancy = int(config["dormancy"]["maximum_frames"])
    ratio_minimum = float(config["association"]["second_best_ratio_minimum"])

    for frame in range(start_frame, end_frame + 1):
        current = by_frame.get(frame, [])
        eligible = []
        for destination in current:
            for candidate in admitted_by_destination.get(str(destination["observation_id"]), []):
                source_id = str(candidate["source_observation_id"])
                source_track = observation_track.get(source_id)
                if source_track is None or active.get(source_track) != source_id:
                    continue
                age = frame - int(by_id[source_id]["frame"])
                if 0 < age <= maximum_dormancy:
                    eligible.append(candidate)
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_destination: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in eligible:
            by_source[str(candidate["source_observation_id"])].append(candidate)
            by_destination[str(candidate["destination_observation_id"])].append(candidate)

        ambiguous_destinations: dict[str, dict[str, Any]] = {}
        for destination_id, rows in by_destination.items():
            sources = {str(value["source_observation_id"]) for value in rows}
            split = any(len(by_source[source]) > 1 for source in sources)
            merge = len(sources) > 1
            ordered_costs = sorted(float(value["cost"]) for value in rows)
            ratio = math.inf if len(ordered_costs) == 1 else (
                math.inf if ordered_costs[0] == 0 < ordered_costs[1]
                else 1.0 if ordered_costs[0] == ordered_costs[1] == 0
                else ordered_costs[1] / ordered_costs[0]
            )
            competitive = ratio < ratio_minimum
            if split or merge or competitive:
                ambiguous_destinations[destination_id] = {
                    "split": split,
                    "merge": merge,
                    "competitive_second_best": competitive,
                    "second_best_ratio": None if math.isinf(ratio) else ratio,
                    "candidate_ids": sorted(str(value["candidate_id"]) for value in rows),
                }

        linked_tracks: set[str] = set()
        for observation in current:
            observation_id = str(observation["observation_id"])
            if observation_id in ambiguous_destinations:
                ambiguity = ambiguous_destinations[observation_id]
                reasons = [key for key in ("split", "merge", "competitive_second_best") if ambiguity[key]]
                abstentions.append(
                    {
                        "observation_id": observation_id,
                        "frame": frame,
                        "track_id": None,
                        "state": "abstained",
                        "abstention": True,
                        "abstention_reason": "+".join(reasons),
                        "ambiguity": ambiguity,
                        "geometry": {
                            "status": observation["geometry_status"],
                            "world_xyz": observation["world_xyz"],
                            "source_observation_id": observation_id,
                        },
                        "depth_order": observation["depth_order"],
                    }
                )
                continue

            rows = sorted(by_destination.get(observation_id, []), key=lambda value: (float(value["cost"]), str(value["source_observation_id"])))
            selected = rows[0] if rows else None
            if selected is None:
                track_payload = {
                    "method_id": METHOD_ID,
                    "initial_observation_id": observation_id,
                    "scene": str(observation["scene"]),
                }
                track_id = domain_id(TRACK_DOMAIN, track_payload)
                association = {
                    "decision": "new_hypothesis_no_admitted_predecessor",
                    "association_confidence": None,
                    "association_risk": None,
                    "candidate_id": None,
                    "abstention": False,
                    "abstention_reason": None,
                }
                transitions = []
            else:
                source_id = str(selected["source_observation_id"])
                track_id = observation_track[source_id]
                previous = by_id[source_id]
                transitions = []
                if int(selected["frame_gap"]) > 1:
                    transitions.append("reappeared")
                if previous["depth_order"] == "rear" and observation["depth_order"] == "front":
                    transitions.append("revealed")
                association = {
                    "decision": "propagated_algorithmic_hypothesis",
                    "association_confidence": float(selected["association_confidence"]),
                    "association_risk": float(selected["association_risk"]),
                    "candidate_id": str(selected["candidate_id"]),
                    "flow_record_ids": list(selected["flow_record_ids"]),
                    "endpoint_error_pixels_median": float(selected["endpoint_error_pixels_median"]),
                    "endpoint_error_pixels_maximum": float(selected["endpoint_error_pixels_maximum"]),
                    "normalized_endpoint_error_median": float(selected["normalized_endpoint_error_median"]),
                    "world_displacement_rscene_per_frame": float(selected["world_displacement_rscene_per_frame"]),
                    "abstention": False,
                    "abstention_reason": None,
                }
            observation_track[observation_id] = track_id
            active[track_id] = observation_id
            linked_tracks.add(track_id)
            record = {
                "frame": frame,
                "track_id": track_id,
                "observation_id": observation_id,
                "state": str(observation["target_visibility_state"]),
                "visibility_events": transitions,
                "depth_order": str(observation["depth_order"]),
                "geometry": {
                    "status": str(observation["geometry_status"]),
                    "world_xyz": list(observation["world_xyz"]),
                    "median_optical_z": float(observation["median_optical_z"]),
                    "source_observation_id": observation_id,
                },
                "camera_states": _camera_states(observation),
                "physical_camera_ancestry": list(observation["physical_camera_ancestry"]),
                "camera_time_ancestry": list(observation["camera_time_ancestry"]),
                "source_observations": dict(observation["source_observations"]),
                "order_evidence": dict(observation["order_evidence"]),
                "observation_risk": float(observation["observation_risk"]),
                "uncertainty": dict(observation["uncertainty"]),
                "association": association,
            }
            if track_id not in tracks:
                tracks[track_id] = {
                    "track_id": track_id,
                    "identity_semantics": "algorithmic_association_hypothesis_not_physical_identity",
                    "records": [],
                }
            tracks[track_id]["records"].append(record)

        for track_id, last_observation_id in sorted(list(active.items())):
            if track_id in linked_tracks:
                continue
            last_observation = by_id[last_observation_id]
            age = frame - int(last_observation["frame"])
            if age <= 0:
                continue
            if age > maximum_dormancy:
                del active[track_id]
                continue
            tracks[track_id]["records"].append(
                {
                    "frame": frame,
                    "track_id": track_id,
                    "observation_id": None,
                    "state": "dormant",
                    "visibility_events": [],
                    "depth_order": "unknown_not_observed",
                    "geometry": {
                        "status": "retained_identity_only_no_current_geometry",
                        "world_xyz": None,
                        "median_optical_z": None,
                        "source_observation_id": last_observation_id,
                    },
                    "camera_states": [],
                    "physical_camera_ancestry": list(last_observation["physical_camera_ancestry"]),
                    "camera_time_ancestry": list(last_observation["camera_time_ancestry"]),
                    "source_observations": {"last_observed_observation_id": last_observation_id},
                    "order_evidence": {"status": "not_observed_no_order_retained"},
                    "observation_risk": None,
                    "uncertainty": {"dormancy_age": age, "maximum_dormancy_age": maximum_dormancy},
                    "association": {
                        "decision": "bounded_dormancy_identity_descriptor_only",
                        "association_confidence": None,
                        "association_risk": None,
                        "candidate_id": None,
                        "abstention": False,
                        "abstention_reason": None,
                    },
                }
            )

    track_rows = []
    for track_id in sorted(tracks):
        track = tracks[track_id]
        records = sorted(track["records"], key=lambda value: (int(value["frame"]), str(value.get("observation_id") or "")))
        observed = [value for value in records if value["observation_id"] is not None]
        state_counts = Counter(str(value["state"]) for value in records)
        event_counts = Counter(event for value in records for event in value["visibility_events"])
        cameras = sorted({camera for value in observed for camera in value["physical_camera_ancestry"]})
        track_rows.append(
            {
                **{key: value for key, value in track.items() if key != "records"},
                "support_interval": {
                    "first_observed_frame": min(int(value["frame"]) for value in observed),
                    "last_observed_frame": max(int(value["frame"]) for value in observed),
                    "observed_frames": sorted({int(value["frame"]) for value in observed}),
                    "observed_frame_count": len({int(value["frame"]) for value in observed}),
                    "duration_frames_inclusive": max(int(value["frame"]) for value in observed) - min(int(value["frame"]) for value in observed) + 1,
                    "maximum_dormancy_age": max((int(value["uncertainty"]["dormancy_age"]) for value in records if value["state"] == "dormant"), default=0),
                },
                "camera_support": cameras,
                "state_counts": dict(sorted(state_counts.items())),
                "event_counts": dict(sorted(event_counts.items())),
                "front_observation_count": sum(value["depth_order"] == "front" for value in observed),
                "rear_observation_count": sum(value["depth_order"] == "rear" for value in observed),
                "records": records,
            }
        )
    return {
        "tracks": track_rows,
        "abstentions": sorted(abstentions, key=lambda value: (int(value["frame"]), str(value["observation_id"]))),
        "observation_count": len(items),
        "associated_observation_count": len(observation_track),
        "abstained_observation_count": len(abstentions),
        "candidate_count": len(candidates),
        "admitted_candidate_count": sum(bool(value.get("admitted", False)) for value in candidates),
    }


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "minimum": None, "mean": None, "quantiles": {}, "maximum": None}
    if not np.isfinite(array).all():
        raise SchemaError("diagnostic distribution contains nonfinite values")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "quantiles": {str(q): float(np.quantile(array, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
        "maximum": float(np.max(array)),
    }


def summarize_association(result: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    tracks = list(result["tracks"])
    candidate_rows = list(candidates)
    observed_records = [record for track in tracks for record in track["records"] if record["observation_id"] is not None]
    all_records = [record for track in tracks for record in track["records"]]
    propagated = [record for record in observed_records if record["association"]["candidate_id"] is not None]
    durations = [int(track["support_interval"]["duration_frames_inclusive"]) for track in tracks]
    observed_durations = [int(track["support_interval"]["observed_frame_count"]) for track in tracks]
    camera_support = [len(track["camera_support"]) for track in tracks]
    confidences = [float(record["association"]["association_confidence"]) for record in propagated]
    association_risks = [float(record["association"]["association_risk"]) for record in propagated]
    observation_risks = [float(record["observation_risk"]) for record in observed_records]
    endpoint_errors = [float(record["association"]["endpoint_error_pixels_median"]) for record in propagated]
    temporal_errors = [float(record["association"]["normalized_endpoint_error_median"]) for record in propagated]
    state_counts = Counter(str(record["state"]) for record in all_records)
    state_counts["abstained"] += int(result["abstained_observation_count"])
    event_counts = Counter(event for record in all_records for event in record["visibility_events"])
    order_counts = Counter(str(record["depth_order"]) for record in observed_records)
    frame_counts = Counter(int(record["frame"]) for record in observed_records)
    bin_counts = Counter(
        tuple(record["source_observations"].get("p03_hypothesis_id", "") for _ in (0,))
        for record in observed_records
    )
    multi_frame = [track for track in tracks if int(track["support_interval"]["observed_frame_count"]) > 1]
    rear_multi_frame = [track for track in multi_frame if int(track["rear_observation_count"]) > 0]
    ambiguity = Counter(str(value["abstention_reason"]) for value in result["abstentions"])
    provenance_complete = all(
        record["physical_camera_ancestry"]
        and record["camera_time_ancestry"]
        and record["source_observations"].get("p03_hypothesis_id")
        and record["geometry"]["status"].startswith("directly_observed")
        for record in observed_records
    ) and all(
        record["association"]["flow_record_ids"]
        for record in propagated
    ) and all(
        record["geometry"]["world_xyz"] is None
        and record["depth_order"] == "unknown_not_observed"
        for record in all_records if record["state"] == "dormant"
    )
    return {
        "track_count": len(tracks),
        "multi_frame_track_count": len(multi_frame),
        "multi_frame_rear_track_count": len(rear_multi_frame),
        "track_duration_frames": _distribution(durations),
        "observed_frames_per_track": _distribution(observed_durations),
        "camera_support_per_track": _distribution(camera_support),
        "state_counts": dict(sorted(state_counts.items())),
        "visibility_event_counts": dict(sorted(event_counts.items())),
        "depth_order_counts": dict(sorted(order_counts.items())),
        "association_confidence": _distribution(confidences),
        "association_risk": _distribution(association_risks),
        "observation_risk": _distribution(observation_risks),
        "reprojection_endpoint_error_pixels": _distribution(endpoint_errors),
        "temporal_consistency_normalized_error": _distribution(temporal_errors),
        "candidate_cost": _distribution(float(value["cost"]) for value in candidate_rows),
        "coverage": {
            "observation_count": int(result["observation_count"]),
            "associated_observation_count": int(result["associated_observation_count"]),
            "abstained_observation_count": int(result["abstained_observation_count"]),
            "propagated_observation_count": len(propagated),
            "propagated_observation_fraction": (
                len(propagated) / int(result["observation_count"])
                if int(result["observation_count"]) else 0.0
            ),
            "association_coverage_fraction": (
                int(result["associated_observation_count"]) / int(result["observation_count"])
                if int(result["observation_count"]) else 0.0
            ),
            "abstention_fraction": (
                int(result["abstained_observation_count"]) / int(result["observation_count"])
                if int(result["observation_count"]) else 0.0
            ),
        },
        "split_merge_ambiguity_counts": dict(sorted(ambiguity.items())),
        "provenance_complete": provenance_complete,
        "concentration": {
            "maximum_observations_in_one_frame": max(frame_counts.values(), default=0),
            "top_frame_fraction": max(frame_counts.values(), default=0) / max(1, len(observed_records)),
            "unique_p03_regions": len(bin_counts),
            "top_region_fraction": max(bin_counts.values(), default=0) / max(1, len(observed_records)),
        },
    }


def build_stage1_ledger(
    scientific_payload: Mapping[str, Any],
    *,
    runtime_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_stage1_config(config)
    scientific_hash = canonical_scientific_hash(scientific_payload)
    payload = {
        "schema_version": LEDGER_SCHEMA,
        "method_id": METHOD_ID,
        "identity_semantics": "track_id_is_algorithmic_association_hypothesis_not_proven_physical_surface_identity",
        "scientific_content_hash": scientific_hash,
        "scientific_hash_contract": {
            "algorithm": validated["canonical_hash"]["algorithm"],
            "domain": SCIENTIFIC_HASH_DOMAIN,
            "included": "scientific_payload",
            "excluded": "runtime_metadata",
            "excluded_runtime_fields": list(validated["canonical_hash"]["excluded_runtime_fields"]),
        },
        "scientific_payload": dict(scientific_payload),
        "runtime_metadata": dict(runtime_metadata),
    }
    ledger = {
        **payload,
        "artifact_id": domain_id(
            "csvl-vpl-stage1-v1/ledger-artifact",
            {"schema_version": LEDGER_SCHEMA, "scientific_content_hash": scientific_hash},
        ),
    }
    validate_payload(LEDGER_SCHEMA, ledger)
    return ledger


def build_diagnostics_artifact(
    *,
    ledger_scientific_content_hash: str,
    diagnostics: Mapping[str, Any],
    cpu_only_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": DIAGNOSTICS_SCHEMA,
        "method_id": METHOD_ID,
        "ledger_scientific_content_hash": str(ledger_scientific_content_hash),
        "diagnostics": dict(diagnostics),
        "cpu_only_evidence": dict(cpu_only_evidence),
    }
    artifact = {
        **payload,
        "artifact_id": domain_id("csvl-vpl-stage1-v1/diagnostics-artifact", payload),
    }
    validate_payload(DIAGNOSTICS_SCHEMA, artifact)
    return artifact


def validate_no_fabricated_hidden_geometry(result: Mapping[str, Any]) -> None:
    for track in result["tracks"]:
        for record in track["records"]:
            if record["state"] == "dormant":
                if record["geometry"]["world_xyz"] is not None or record["geometry"]["median_optical_z"] is not None:
                    raise ProvenanceError("dormant record fabricated current hidden geometry")
                if record["depth_order"] != "unknown_not_observed":
                    raise ProvenanceError("dormant record fabricated current hidden depth order")
            elif not record["geometry"]["status"].startswith("directly_observed"):
                raise ProvenanceError("observed track record lacks direct P03 geometry provenance")


__all__ = [
    "CONFIG_SCHEMA",
    "CONTROL_MODES",
    "DIAGNOSTICS_SCHEMA",
    "LEDGER_SCHEMA",
    "METHOD_ID",
    "P02FlowStore",
    "associate_observations",
    "build_candidate_evidence",
    "build_diagnostics_artifact",
    "build_stage1_ledger",
    "candidate_evidence",
    "canonical_scientific_hash",
    "extract_p03_observations",
    "summarize_association",
    "validate_no_fabricated_hidden_geometry",
    "validate_stage1_config",
]
