"""Reciprocal multiview matching and deterministic robust fusion."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .canonical import array_semantic_ref, domain_id
from .source_nodes import lower_median, weighted_median_stable


def _runtime_array(record: Mapping[str, Any], key: str, fallback: str) -> np.ndarray:
    value = record.get(key, record.get(fallback))
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{key} must be finite")
    return array


def _camera_id(node: Mapping[str, Any]) -> str:
    if "source_camera" in node:
        return str(node["source_camera"])
    if "camera_id" in node:
        return str(node["camera_id"])
    raise ValueError("fusion node is missing its physical source camera")


def _coordinate_lower_median(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError("coordinate median needs a nonempty finite matrix")
    return np.sort(array, axis=0)[(array.shape[0] - 1) // 2]


def project_node(node: Mapping[str, Any], camera: Mapping[str, Any]) -> tuple[float, float, float]:
    point = _runtime_array(node, "world_point_array", "world_point")
    w2c = np.asarray(camera["w2c"], dtype=np.float64)
    k = np.asarray(camera["K"], dtype=np.float64)
    if point.shape != (3,) or w2c.shape != (4, 4) or k.shape != (3, 3):
        raise ValueError("invalid projection shapes")
    if not np.all(np.isfinite(w2c)) or not np.all(np.isfinite(k)):
        raise ValueError("projection calibration must be finite")
    camera_point = w2c[:3, :3] @ point + w2c[:3, 3]
    if not np.all(np.isfinite(camera_point)) or camera_point[2] <= 0:
        raise ValueError("projected point has nonpositive/nonfinite z")
    uvw = k @ camera_point
    x, y = float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])
    if "width" in camera and "height" in camera:
        if not (0 <= x <= int(camera["width"]) - 1 and 0 <= y <= int(camera["height"]) - 1):
            raise ValueError("projected point is outside the declared camera aperture")
    return x, y, float(camera_point[2])


def patch_ncc(first: np.ndarray, second: np.ndarray) -> float:
    original_a = np.asarray(first, dtype=np.float64).reshape(-1)
    original_b = np.asarray(second, dtype=np.float64).reshape(-1)
    if (
        original_a.shape != original_b.shape
        or original_a.size == 0
        or not np.all(np.isfinite(original_a))
        or not np.all(np.isfinite(original_b))
    ):
        raise ValueError("NCC patches must be finite with equal nonzero size")
    a = original_a - original_a.mean()
    b = original_b - original_b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 1.0 if np.array_equal(original_a, original_b) else 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def proposal_terms(
    projection_residual: float,
    depth_residual: float,
    sigma_pair: float,
    ncc: float,
) -> dict[str, float]:
    values = tuple(float(value) for value in (projection_residual, depth_residual, sigma_pair, ncc))
    if (
        not all(math.isfinite(value) for value in values)
        or projection_residual < 0
        or depth_residual < 0
        or sigma_pair <= 0
        or not -1.0 <= ncc <= 1.0
    ):
        raise ValueError("proposal inputs violate the finite residual/NCC domain")
    projection_term = float(projection_residual) / 2.0
    depth_term = float(depth_residual) / (2.5 * float(sigma_pair))
    appearance_term = (1.0 - float(ncc)) / 0.40
    clipped = [float(np.clip(value, 0.0, 2.0)) for value in (projection_term, depth_term, appearance_term)]
    return {
        "projection": projection_term,
        "depth": depth_term,
        "appearance": appearance_term,
        "cost": sum(clipped) / 3.0,
        "risk": max(float(np.clip(value, 0.0, 1.0)) for value in (projection_term, depth_term, appearance_term)),
    }


def select_pair_match(candidates: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    checked = []
    for candidate in candidates:
        cost = float(candidate["cost"])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("pair candidate cost must be finite and nonnegative")
        if cost <= 1.0:
            checked.append(candidate)
    if not checked:
        return None
    return min(
        checked,
        key=lambda item: (
            float(item["cost"]),
            str(item["camera_id"]),
            int(item["y"]),
            int(item["x"]),
            str(item["node_id"]),
        ),
    )


def reciprocal_candidates(
    forward: Mapping[str, list[Mapping[str, Any]]],
    reverse: Mapping[str, list[Mapping[str, Any]]],
) -> list[tuple[str, str, float]]:
    """Return pairs whose independently selected identities are reciprocal."""

    forward_choice = {node: select_pair_match(items) for node, items in forward.items()}
    reverse_choice = {node: select_pair_match(items) for node, items in reverse.items()}
    pairs: list[tuple[str, str, float]] = []
    for source_id in sorted(forward_choice):
        choice = forward_choice[source_id]
        if choice is None:
            continue
        target_id = str(choice["node_id"])
        back = reverse_choice.get(target_id)
        if back is not None and str(back["node_id"]) == source_id:
            pairs.append((source_id, target_id, float(choice["cost"])))
    return pairs


def _node_covariance(node: Mapping[str, Any]) -> np.ndarray:
    covariance = _runtime_array(node, "covariance_array", "covariance")
    if covariance.shape != (3, 3):
        raise ValueError("fused node covariance must be 3x3")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
        raise ValueError("fused node covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-12:
        raise ValueError("fused node covariance must be positive semidefinite")
    return covariance


def _fusion_statistics(nodes: list[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.stack([_runtime_array(node, "world_point_array", "world_point") for node in nodes])
    if xyz.shape[1:] != (3,):
        raise ValueError("fused node world points must be 3-vectors")
    center = _coordinate_lower_median(xyz)
    mad = _coordinate_lower_median(np.abs(xyz - center))
    covariance = np.diag((1.4826 * mad) ** 2) + np.mean(
        [_node_covariance(node) for node in nodes], axis=0
    )
    diagonal = np.maximum(np.diag(covariance), 1e-12)
    distances = np.sqrt(np.sum((xyz - center) ** 2 / diagonal, axis=1))
    return xyz, center, covariance, distances


def robust_fuse(
    nodes: Iterable[Mapping[str, Any]],
    *,
    r_scene: float,
    anchor_id: str | None = None,
) -> dict[str, Any]:
    node_list = sorted(nodes, key=lambda item: str(item["node_id"]))
    if not math.isfinite(r_scene) or r_scene <= 0:
        raise ValueError("fusion requires positive finite R_scene")
    if len(node_list) < 3:
        raise ValueError("fusion requires at least three source nodes")
    source_id_set = {str(node["node_id"]) for node in node_list}
    anchor = min(source_id_set) if anchor_id is None else str(anchor_id)
    if anchor not in source_id_set:
        raise ValueError("fusion anchor must be one of the proposal source nodes")
    cameras = [_camera_id(node) for node in node_list]
    if len(cameras) != len(set(cameras)) or len(set(cameras)) < 3:
        raise ValueError("fusion requires exactly one node from each of at least three cameras")
    identity_keys = ("scene", "frame", "scored_target")
    for key in identity_keys:
        if key not in node_list[0] or any(node.get(key) != node_list[0][key] for node in node_list):
            raise ValueError(f"fusion cannot mix {key}")
    target = str(node_list[0]["scored_target"])
    for node in node_list:
        ancestry = {str(value) for value in node["physical_ancestry"]}
        if target in ancestry:
            raise ValueError("scored target leaked into a fusion node")
        confidence = float(node["confidence"])
        if not math.isfinite(confidence) or confidence < 0:
            raise ValueError("fusion confidence must be finite and nonnegative")

    retained = list(node_list)
    while True:
        xyz, center, covariance, distances = _fusion_statistics(retained)
        offenders = [index for index, distance in enumerate(distances) if float(distance) > 2.5]
        if not offenders:
            break
        worst = max(
            offenders,
            key=lambda index: (float(distances[index]), str(retained[index]["node_id"])),
        )
        retained.pop(worst)
        if len(retained) < 3:
            raise ValueError("robust pruning left fewer than three cameras")

    xyz, center, covariance, distances = _fusion_statistics(retained)
    stable_ids = [str(node["node_id"]) for node in retained]
    confidences = [float(node["confidence"]) for node in retained]
    colors = np.stack([_runtime_array(node, "linear_rgb_array", "linear_rgb") for node in retained])
    if colors.shape[1:] != (3,):
        raise ValueError("source-node center colors must be RGB vectors")
    fused_color = np.array(
        [
            weighted_median_stable(colors[:, channel], confidences, stable_ids)
            for channel in range(3)
        ],
        dtype=np.float64,
    )
    source_ids = sorted(str(node["node_id"]) for node in node_list)
    retained_ids = sorted(stable_ids)
    payload = {
        "scene": retained[0]["scene"],
        "frame": retained[0]["frame"],
        "scored_target": retained[0]["scored_target"],
        "source_node_ids": source_ids,
        "retained_source_node_ids": retained_ids,
        "world_point": array_semantic_ref(center, semantic_key="fused_world_point"),
        "covariance": array_semantic_ref(covariance, semantic_key="fused_covariance"),
    }
    fused_id = domain_id("csvl-v1/fused", payload)
    pair_risk = max(float(node.get("pair_risk", 0.0)) for node in retained)
    robust_risk = min(1.0, float(np.max(distances)) / 2.5)
    camera_count = len(retained)
    pair_costs = [float(node.get("pair_cost", 0.0)) for node in retained]
    if any(not math.isfinite(value) or value < 0 for value in pair_costs):
        raise ValueError("pair costs must be finite and nonnegative")
    return {
        "fused_id": fused_id,
        **payload,
        "world_point_array": center,
        "covariance_array": covariance,
        "linear_rgb": fused_color,
        "linear_rgb_ref": array_semantic_ref(fused_color, semantic_key="fused_linear_rgb"),
        "patch_color_weight": sum(confidences),
        "camera_count": camera_count,
        "median_pair_cost": lower_median(pair_costs),
        "anchor_id": anchor,
        "physical_ancestry": sorted(
            {str(camera) for node in node_list for camera in node["physical_ancestry"]}
        ),
        "risk": fused_risk(
            max(float(node.get("duplicate_risk", 0.0)) for node in retained),
            pair_risk,
            robust_risk,
            camera_count,
        ),
        "robust_clipping_floor": 1e-12,
    }


def enforce_source_exclusivity(proposals: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ordered = sorted(
        proposals,
        key=lambda item: (
            -int(item["camera_count"]),
            float(item["median_pair_cost"]),
            str(item["anchor_id"]),
            tuple(sorted(str(value) for value in item["source_node_ids"])),
        ),
    )
    owned: set[str] = set()
    accepted: list[Mapping[str, Any]] = []
    for proposal in ordered:
        ids = {str(value) for value in proposal["source_node_ids"]}
        if not ids:
            raise ValueError("fusion proposal cannot have zero source nodes")
        if ids.isdisjoint(owned):
            accepted.append(proposal)
            owned.update(ids)
    return accepted


def deduplicate_fused(proposals: Iterable[Mapping[str, Any]], *, r_scene: float) -> list[Mapping[str, Any]]:
    if not math.isfinite(r_scene) or r_scene <= 0:
        raise ValueError("R_scene must be positive and finite")
    ordered = sorted(
        proposals,
        key=lambda item: (
            -int(item["camera_count"]),
            float(item["median_pair_cost"]),
            str(item["anchor_id"]),
            str(item["fused_id"]),
        ),
    )
    retained: list[Mapping[str, Any]] = []
    radius = 0.002 * r_scene
    for proposal in ordered:
        point = _runtime_array(proposal, "world_point_array", "world_point")
        if point.shape != (3,):
            raise ValueError("fused proposal point must be a 3-vector")
        if all(
            np.linalg.norm(point - _runtime_array(other, "world_point_array", "world_point")) > radius
            for other in retained
        ):
            retained.append(proposal)
    return retained


def fused_risk(duplicate: float, pair: float, robust_distance: float, camera_count: int) -> float:
    if camera_count < 0:
        raise ValueError("camera count cannot be negative")
    values = (float(duplicate), float(pair), float(robust_distance))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("fusion risk terms must be finite")
    support = float(np.clip((4 - camera_count) / 2.0, 0.0, 1.0))
    return max(float(np.clip(value, 0.0, 1.0)) for value in (*values, support))


__all__ = [
    "deduplicate_fused",
    "enforce_source_exclusivity",
    "fused_risk",
    "patch_ncc",
    "project_node",
    "proposal_terms",
    "reciprocal_candidates",
    "robust_fuse",
    "select_pair_match",
]
