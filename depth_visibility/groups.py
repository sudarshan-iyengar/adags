"""Deterministic calibrated multiview grouping and target exclusion."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .camera import camera_center
from .errors import ProvenanceError, SchemaError
from .n3v import CameraRecord


def _camera_id(camera: Any) -> str:
    value = camera.camera_id if hasattr(camera, "camera_id") else camera.get("camera_id")
    if not isinstance(value, str) or not value:
        raise SchemaError("camera record lacks camera_id")
    return value


def _w2c(camera: Any) -> np.ndarray:
    value = camera.w2c_opencv if hasattr(camera, "w2c_opencv") else camera.get("w2c")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise SchemaError("camera record lacks finite 4x4 w2c")
    return result


def _camera_sort_key(camera_id: str) -> tuple[int, str]:
    suffix = camera_id[3:] if camera_id.startswith("cam") else ""
    return (int(suffix), camera_id) if suffix.isdigit() else (2**31 - 1, camera_id)


def _optical_axis_world(camera: Any) -> np.ndarray:
    rotation_cw = _w2c(camera)[:3, :3]
    axis = rotation_cw.T[:, 2]
    return axis / np.linalg.norm(axis)


def _angle_degrees(first: Any, second: Any) -> float:
    cosine = float(np.clip(np.dot(_optical_axis_world(first), _optical_axis_world(second)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _camera_map(cameras: Iterable[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for camera in cameras:
        camera_id = _camera_id(camera)
        if camera_id in result:
            raise ProvenanceError(f"duplicate camera geometry: {camera_id}")
        result[camera_id] = camera
    return result


def _second_singular_value(group: Sequence[str], cameras: Mapping[str, Any]) -> float:
    centers = np.stack([camera_center(_w2c(cameras[camera_id])) for camera_id in group])
    centered = centers - np.mean(centers, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    return float(singular[1]) if singular.size >= 2 else 0.0


def validate_group(
    group: Sequence[str],
    cameras: Mapping[str, Any] | Iterable[Any],
    r_scene: float,
    *,
    maximum_cameras: int = 6,
    maximum_optical_axis_angle_degrees: float = 75.0,
    minimum_center_distance_rscene: float = 0.02,
    minimum_second_singular_value_rscene: float = 0.01,
) -> tuple[str, ...]:
    """Validate a complete ordered anchor group and return it as a tuple."""

    camera_map = dict(cameras) if isinstance(cameras, Mapping) else _camera_map(cameras)
    ordered = tuple(group)
    if not math.isfinite(float(r_scene)) or r_scene <= 0:
        raise SchemaError("R_scene must be finite and positive")
    if len(ordered) != maximum_cameras or len(set(ordered)) != len(ordered):
        raise ProvenanceError(f"group must contain exactly {maximum_cameras} distinct cameras")
    if any(camera_id not in camera_map for camera_id in ordered):
        raise ProvenanceError("group names unknown camera geometry")
    anchor = camera_map[ordered[0]]
    anchor_center = camera_center(_w2c(anchor))
    for camera_id in ordered[1:]:
        candidate = camera_map[camera_id]
        distance = float(np.linalg.norm(camera_center(_w2c(candidate)) - anchor_center))
        if distance < minimum_center_distance_rscene * r_scene:
            raise ProvenanceError("group member violates center-distance threshold")
        if _angle_degrees(anchor, candidate) > maximum_optical_axis_angle_degrees:
            raise ProvenanceError("group member violates optical-axis threshold")
    if _second_singular_value(ordered, camera_map) < minimum_second_singular_value_rscene * r_scene:
        raise ProvenanceError("group camera centers are insufficiently diverse")
    return ordered


def select_anchor_group(
    anchor_camera_id: str,
    cameras: Mapping[str, Any] | Iterable[Any],
    r_scene: float,
    *,
    maximum_cameras: int = 6,
    maximum_optical_axis_angle_degrees: float = 75.0,
    minimum_center_distance_rscene: float = 0.02,
    minimum_second_singular_value_rscene: float = 0.01,
) -> tuple[str, ...]:
    """Select one six-camera group by deterministic farthest-point sampling."""

    camera_map = dict(cameras) if isinstance(cameras, Mapping) else _camera_map(cameras)
    if anchor_camera_id not in camera_map:
        raise ProvenanceError(f"unknown anchor camera: {anchor_camera_id}")
    if not math.isfinite(float(r_scene)) or r_scene <= 0:
        raise SchemaError("R_scene must be finite and positive")
    anchor = camera_map[anchor_camera_id]
    anchor_center = camera_center(_w2c(anchor))
    candidates: list[str] = []
    for camera_id, candidate in camera_map.items():
        if camera_id == anchor_camera_id:
            continue
        distance = float(np.linalg.norm(camera_center(_w2c(candidate)) - anchor_center))
        if distance < minimum_center_distance_rscene * r_scene:
            continue
        if _angle_degrees(anchor, candidate) > maximum_optical_axis_angle_degrees:
            continue
        candidates.append(camera_id)
    selected = [anchor_camera_id]
    remaining = set(candidates)
    while remaining and len(selected) < maximum_cameras:
        scores: list[tuple[float, tuple[int, str], str]] = []
        selected_centers = [camera_center(_w2c(camera_map[item])) for item in selected]
        for camera_id in remaining:
            center = camera_center(_w2c(camera_map[camera_id]))
            score = min(float(np.linalg.norm(center - other)) for other in selected_centers) / r_scene
            scores.append((-score, _camera_sort_key(camera_id), camera_id))
        _, _, chosen = min(scores)
        selected.append(chosen)
        remaining.remove(chosen)
    return validate_group(
        selected, camera_map, r_scene,
        maximum_cameras=maximum_cameras,
        maximum_optical_axis_angle_degrees=maximum_optical_axis_angle_degrees,
        minimum_center_distance_rscene=minimum_center_distance_rscene,
        minimum_second_singular_value_rscene=minimum_second_singular_value_rscene,
    )


def enumerate_anchor_groups(
    cameras: Mapping[str, Any] | Iterable[Any],
    r_scene: float,
    **grouping: Any,
) -> tuple[tuple[str, ...], ...]:
    """Generate one complete group for every anchor in camera-ID order."""

    camera_map = dict(cameras) if isinstance(cameras, Mapping) else _camera_map(cameras)
    return tuple(
        select_anchor_group(camera_id, camera_map, r_scene, **grouping)
        for camera_id in sorted(camera_map, key=_camera_sort_key)
    )


def filter_groups_for_target(
    groups: Iterable[Sequence[str]],
    target_camera_id: str,
    cameras: Mapping[str, Any] | Iterable[Any],
    r_scene: float,
    *,
    minimum_valid_groups_per_camera_time: int = 2,
    require_all_sources: bool = False,
    **grouping: Any,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Drop target-containing groups, revalidate, and recompute source support."""

    camera_map = dict(cameras) if isinstance(cameras, Mapping) else _camera_map(cameras)
    valid: list[tuple[str, ...]] = []
    for group in groups:
        ordered = tuple(group)
        if target_camera_id in ordered:
            continue
        valid.append(validate_group(ordered, camera_map, r_scene, **grouping))
    by_source: dict[str, list[tuple[str, ...]]] = {
        camera_id: [] for camera_id in camera_map if camera_id != target_camera_id
    }
    for group in valid:
        for source in group:
            if source != target_camera_id:
                by_source[source].append(group)
    result = {
        source: tuple(sorted(source_groups))
        for source, source_groups in by_source.items()
        if len(source_groups) >= minimum_valid_groups_per_camera_time
    }
    if require_all_sources:
        missing = sorted(set(by_source) - set(result), key=_camera_sort_key)
        if missing:
            raise ProvenanceError(f"target-filtered sources lack two valid groups: {missing}")
    return result


def physical_ancestry(payload: Any) -> frozenset[str]:
    """Return declared physical-camera ancestry without using scored-target labels."""

    if isinstance(payload, CameraRecord):
        return frozenset({payload.camera_id})
    if isinstance(payload, str):
        return frozenset({payload}) if payload.startswith("cam") else frozenset()
    if isinstance(payload, Mapping):
        for key in ("physical_camera_ancestry", "physical_camera_dependencies"):
            if key in payload:
                value = payload[key]
                if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                    raise ProvenanceError(f"{key} must be an array")
                declared = frozenset(str(item) for item in value)
                if not declared or any(not item.startswith("cam") for item in declared):
                    raise ProvenanceError("physical-camera ancestry is empty or malformed")
                nested = frozenset()
                for dependency_key in ("parents", "inputs", "source_nodes", "members"):
                    if dependency_key in payload:
                        nested |= physical_ancestry(payload[dependency_key])
                if nested and not nested.issubset(declared):
                    raise ProvenanceError("declared ancestry omits a transitive dependency")
                return declared
        if "camera_id" in payload:
            camera_id = str(payload["camera_id"])
            if camera_id.startswith("cam"):
                return frozenset({camera_id})
        ancestry = frozenset()
        for key in ("parents", "inputs", "source_nodes", "members"):
            if key in payload:
                ancestry |= physical_ancestry(payload[key])
        return ancestry
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        ancestry = frozenset()
        for item in payload:
            ancestry |= physical_ancestry(item)
        return ancestry
    return frozenset()


def _dependency_image_hashes(payload: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = key.lower()
            if "image" in lowered and ("sha" in lowered or "hash" in lowered):
                if isinstance(value, str):
                    result.add(value)
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    result.update(str(item) for item in value)
            elif key not in {"scored_target_camera", "target_camera_id"}:
                result.update(_dependency_image_hashes(value))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for item in payload:
            result.update(_dependency_image_hashes(item))
    return result


def assert_transitive_target_exclusion(
    payloads: Any,
    target_camera_id: str,
    *,
    target_image_sha256: str | None = None,
) -> None:
    """Reject target contamination or missing physical-camera ancestry."""

    ancestry = physical_ancestry(payloads)
    if not ancestry:
        raise ProvenanceError("prediction provenance has no physical-camera ancestry")
    if target_camera_id in ancestry:
        raise ProvenanceError(f"scored target contaminates transitive ancestry: {target_camera_id}")
    if target_image_sha256 is not None and target_image_sha256 in _dependency_image_hashes(payloads):
        raise ProvenanceError("scored target image hash contaminates prediction provenance")


__all__ = [
    "assert_transitive_target_exclusion",
    "enumerate_anchor_groups",
    "filter_groups_for_target",
    "physical_ancestry",
    "select_anchor_group",
    "validate_group",
]
