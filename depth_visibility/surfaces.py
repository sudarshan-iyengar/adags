"""Deterministic micro-surfaces, rasterization, and depth ordering."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .canonical import domain_id
from .source_nodes import weighted_median_stable


def _lower_coordinate_median(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError("coordinate median requires a nonempty finite matrix")
    return np.sort(array, axis=0)[(len(array) - 1) // 2]


def _point(record: Mapping[str, Any]) -> np.ndarray:
    value = record.get("world_point_array", record.get("world_point"))
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("fused world point must be a finite 3-vector")
    return point


def orient_normal(normal: np.ndarray, point: np.ndarray, support_center: np.ndarray) -> np.ndarray:
    vector = np.asarray(normal, dtype=np.float64).copy()
    point = np.asarray(point, dtype=np.float64)
    support_center = np.asarray(support_center, dtype=np.float64)
    if (
        vector.shape != (3,)
        or point.shape != (3,)
        or support_center.shape != (3,)
        or not np.all(np.isfinite(vector))
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(support_center))
    ):
        raise ValueError("normal orientation needs finite 3-vectors")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("normal orientation needs a nonzero vector")
    vector /= norm
    dot = float(np.dot(vector, support_center - point))
    if dot < 0:
        vector *= -1
    elif dot == 0:
        first = next((value for value in vector if value != 0), 0.0)
        if first < 0:
            vector *= -1
    return vector


def estimate_normal(
    point_index: int,
    points: np.ndarray,
    stable_ids: Iterable[str],
    support_centers: Iterable[np.ndarray],
    *,
    r_scene: float,
) -> np.ndarray | None:
    """Estimate the admitted PCA normal, returning None when uncertain."""

    xyz = np.asarray(points, dtype=np.float64)
    ids = [str(value) for value in stable_ids]
    centers = [np.asarray(value, dtype=np.float64) for value in support_centers]
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or len(ids) != len(xyz)
        or len(centers) != len(xyz)
        or not 0 <= point_index < len(xyz)
    ):
        raise ValueError("normal inputs have inconsistent shapes")
    if not math.isfinite(r_scene) or r_scene <= 0 or not np.all(np.isfinite(xyz)):
        raise ValueError("R_scene and points must be valid")
    if any(center.shape != (3,) or not np.all(np.isfinite(center)) for center in centers):
        raise ValueError("support centers must be finite 3-vectors")
    delta = xyz - xyz[point_index]
    squared = np.sum(delta * delta, axis=1)
    eligible = [
        index
        for index in range(len(xyz))
        if index != point_index and squared[index] <= (0.02 * r_scene) ** 2
    ]
    eligible.sort(key=lambda index: (float(squared[index]), ids[index]))
    neighbors = eligible[:8]
    if len(neighbors) < 6:
        return None
    local = xyz[neighbors]
    center = _lower_coordinate_median(local)
    covariance = (local - center).T @ (local - center) / len(local)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    lambda0, lambda1, lambda2 = (float(value) for value in eigenvalues)
    floor = 1e-12 * r_scene**2
    if lambda1 < 1e-6 * r_scene**2:
        return None
    if lambda1 / max(lambda2, floor) < 0.05:
        return None
    if (lambda1 - lambda0) / max(lambda2, floor) < 0.05:
        return None
    return orient_normal(eigenvectors[:, 0], xyz[point_index], centers[point_index])


def voxel_index(point: np.ndarray, scene_center: np.ndarray, *, r_scene: float) -> tuple[int, int, int]:
    if not math.isfinite(r_scene) or r_scene <= 0:
        raise ValueError("R_scene must be positive and finite")
    point_array = np.asarray(point, dtype=np.float64)
    center_array = np.asarray(scene_center, dtype=np.float64)
    if point_array.shape != (3,) or center_array.shape != (3,) or not np.all(np.isfinite(point_array)) or not np.all(np.isfinite(center_array)):
        raise ValueError("voxel inputs must be finite 3-vectors")
    values = np.floor((point_array - center_array) / (0.01 * r_scene))
    return tuple(int(value) for value in values)


def _normal_angle(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if first.shape != (3,) or second.shape != (3,) or denominator == 0:
        raise ValueError("normal angle needs nonzero 3-vectors")
    cosine = float(np.clip(np.dot(first, second) / denominator, -1, 1))
    return math.degrees(math.acos(cosine))


def _weighted_color(members: list[Mapping[str, Any]]) -> np.ndarray:
    ids = [str(member["fused_id"]) for member in members]
    weights = [float(member.get("patch_color_weight", 0.0)) for member in members]
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("patch color weights must be finite and nonnegative")
    colors = np.stack([np.asarray(member["linear_rgb"], dtype=np.float64) for member in members])
    if colors.shape[1:] != (3,) or not np.all(np.isfinite(colors)):
        raise ValueError("fused colors must be finite RGB vectors")
    return np.array(
        [weighted_median_stable(colors[:, channel], weights, ids) for channel in range(3)],
        dtype=np.float64,
    )


def build_micro_surfaces(
    fused_points: Iterable[Mapping[str, Any]],
    *,
    scene_center: np.ndarray,
    r_scene: float,
) -> list[dict[str, Any]]:
    points = sorted(fused_points, key=lambda item: str(item["fused_id"]))
    if points:
        identity = ("scene", "frame", "scored_target")
        for key in identity:
            if any(point.get(key) != points[0].get(key) for point in points):
                raise ValueError(f"micro-surface construction cannot mix {key}")
    by_voxel: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, point in enumerate(points):
        if point.get("normal") is None:
            continue
        by_voxel[voxel_index(_point(point), scene_center, r_scene=r_scene)].append(index)
    components: list[tuple[tuple[int, int, int], list[int]]] = []
    for voxel in sorted(by_voxel):
        indices = by_voxel[voxel]
        adjacency = {index: [] for index in indices}
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                left_point, right_point = points[left], points[right]
                connected = (
                    np.linalg.norm(_point(left_point) - _point(right_point)) <= 0.01 * r_scene
                    and _normal_angle(np.asarray(left_point["normal"]), np.asarray(right_point["normal"])) <= 30.0
                    and np.linalg.norm(np.asarray(left_point["linear_rgb"]) - np.asarray(right_point["linear_rgb"])) <= 0.15
                    and len(set(left_point["physical_ancestry"]) & set(right_point["physical_ancestry"])) >= 2
                )
                if connected:
                    adjacency[left].append(right)
                    adjacency[right].append(left)
        unseen = set(indices)
        while unseen:
            start = min(unseen, key=lambda index: str(points[index]["fused_id"]))
            queue: deque[int] = deque([start])
            unseen.remove(start)
            component: list[int] = []
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in sorted(adjacency[current], key=lambda index: str(points[index]["fused_id"])):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
            if len(component) >= 3:
                components.append((voxel, component))
    patches: list[dict[str, Any]] = []
    for voxel, component in components:
        members = [points[index] for index in component]
        fused_ids = sorted(str(member["fused_id"]) for member in members)
        payload = {
            "scene": members[0]["scene"],
            "frame": members[0]["frame"],
            "scored_target": members[0]["scored_target"],
            "voxel": list(voxel),
            "fused_hypothesis_ids": fused_ids,
        }
        normal_sum = np.sum([np.asarray(member["normal"], dtype=np.float64) for member in members], axis=0)
        normal_norm = float(np.linalg.norm(normal_sum))
        risks = sorted(float(member.get("risk", 0.0)) for member in members)
        if any(not math.isfinite(risk) or not 0 <= risk <= 1 for risk in risks):
            raise ValueError("fused point risks must be finite in [0,1]")
        risk_index = max(0, math.ceil(0.9 * len(risks)) - 1)
        patches.append(
            {
                "patch_id": domain_id("csvl-v1/patch", payload),
                **payload,
                "members": members,
                "centroid": _lower_coordinate_median(np.stack([_point(member) for member in members])),
                "linear_rgb": _weighted_color(members),
                "normal": normal_sum / normal_norm if normal_norm >= 1e-8 else None,
                "risk": risks[risk_index],
                "uncertain": normal_norm < 1e-8,
                "physical_ancestry": sorted(
                    {str(camera) for member in members for camera in member["physical_ancestry"]}
                ),
            }
        )
    return sorted(patches, key=lambda item: str(item["patch_id"]))


def _stable_eigendecomposition(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("pixel covariance must be finite symmetric 2x2")
    values, vectors = np.linalg.eigh(matrix)
    if float(np.min(values)) < -1e-12:
        raise ValueError("pixel covariance must be positive semidefinite")
    order = np.argsort(-values, kind="stable")
    values, vectors = values[order], vectors[:, order]
    for column in range(vectors.shape[1]):
        first = next((value for value in vectors[:, column] if value != 0), 0.0)
        if first < 0:
            vectors[:, column] *= -1
    return values, vectors


def rasterize_patch(patch: Mapping[str, Any], height: int, width: int) -> dict[tuple[int, int], dict[str, Any]]:
    """Rasterize explicitly projected constituent points into deterministic ellipses."""

    if height <= 0 or width <= 0:
        raise ValueError("raster dimensions must be positive")
    if patch.get("track_id") in (None, ""):
        raise ValueError("rasterization requires an assigned temporal track ID")
    pixels: dict[tuple[int, int], dict[str, Any]] = {}
    members = sorted(patch["members"], key=lambda item: str(item["fused_id"]))
    centers = [np.asarray(member["uv"], dtype=np.float64) for member in members]
    if any(center.shape != (2,) or not np.all(np.isfinite(center)) for center in centers):
        raise ValueError("projected constituent centers must be finite 2-vectors")
    patch_risk = float(patch.get("risk", 0.0))
    if not math.isfinite(patch_risk) or not 0 <= patch_risk <= 1:
        raise ValueError("patch risk must be finite in [0,1]")
    for index, member in enumerate(members):
        center = centers[index]
        values, vectors = _stable_eigendecomposition(np.asarray(member["pixel_covariance"], dtype=np.float64))
        covariance_axes = np.ceil(2.5 * np.sqrt(np.maximum(values, 0.0))).astype(int)
        distances = [np.linalg.norm(center - other) for j, other in enumerate(centers) if j != index]
        spacing_radius = math.ceil(0.5 * min(distances)) if distances else 4
        axes = np.clip(np.maximum(covariance_axes, spacing_radius), 2, 8).astype(float)
        x0, y0 = float(center[0]), float(center[1])
        z = float(member["z"])
        sigma_z = float(member["sigma_z"])
        member_risk = float(member.get("risk", 0.0))
        if not all(math.isfinite(value) for value in (z, sigma_z, member_risk)) or z <= 0 or sigma_z < 0 or not 0 <= member_risk <= 1:
            raise ValueError("raster member z/sigma/risk violates its domain")
        ancestry = tuple(sorted(str(camera) for camera in member["physical_ancestry"]))
        if not ancestry:
            raise ValueError("raster member physical ancestry is empty")
        for y in range(max(0, math.floor(y0 - 8)), min(height, math.ceil(y0 + 8) + 1)):
            for x in range(max(0, math.floor(x0 - 8)), min(width, math.ceil(x0 + 8) + 1)):
                delta = np.array([x - x0, y - y0], dtype=np.float64)
                rotated = vectors.T @ delta
                if float(np.sum((rotated / axes) ** 2)) <= 1.0:
                    candidate = {
                        "z": z,
                        "sigma_z": sigma_z,
                        "risk": max(member_risk, patch_risk),
                        "patch_id": str(patch["patch_id"]),
                        "track_id": str(patch["track_id"]),
                        "physical_ancestry": ancestry,
                        "forced_uncertain": bool(patch.get("uncertain", False)),
                    }
                    old = pixels.get((y, x))
                    if old is None or (candidate["z"], candidate["risk"], candidate["patch_id"]) < (
                        old["z"],
                        old["risk"],
                        old["patch_id"],
                    ):
                        pixels[(y, x)] = candidate
    return pixels


def dense_depth_order(
    track_pixels: Mapping[str, Mapping[tuple[int, int], Mapping[str, Any]]],
    visible_witnesses: Mapping[str, set[str]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Assign visible/occluded/tied states without filling unsupported pixels."""

    by_pixel: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for track_id in sorted(track_pixels):
        if not track_id:
            raise ValueError("track ID cannot be empty")
        for pixel, record in track_pixels[track_id].items():
            y, x = pixel
            if int(y) != y or int(x) != x:
                raise ValueError("ordered pixels must use integer coordinates")
            z, sigma, risk = (float(record[key]) for key in ("z", "sigma_z", "risk"))
            if (
                not all(math.isfinite(value) for value in (z, sigma, risk))
                or z <= 0
                or sigma < 0
                or not 0 <= risk <= 1
            ):
                raise ValueError("depth-order input violates z/sigma/risk domain")
            ancestry = tuple(sorted(str(value) for value in record["physical_ancestry"]))
            if not ancestry:
                raise ValueError("depth-order ancestry cannot be empty")
            by_pixel[(int(y), int(x))].append(
                {"track_id": str(track_id), **record, "physical_ancestry": ancestry}
            )
    ordered: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for pixel in sorted(by_pixel):
        candidates = sorted(
            by_pixel[pixel],
            key=lambda item: (float(item["z"]), float(item["risk"]), str(item["track_id"])),
        )
        if any(bool(candidate.get("forced_uncertain", False)) for candidate in candidates):
            ordered[pixel] = [{**candidate, "state": "uncertain"} for candidate in candidates]
            continue
        first = candidates[0]
        if len(candidates) == 1:
            ordered[pixel] = [{**first, "state": "visible"}]
            continue
        ties = []
        for candidate in candidates[1:]:
            margin = max(
                0.01 * float(first["z"]),
                2.5 * (float(first["sigma_z"]) + float(candidate["sigma_z"])),
            )
            if float(candidate["z"]) - float(first["z"]) <= margin:
                ties.append(str(candidate["track_id"]))
        first_margin = max(
            0.01 * float(first["z"]),
            2.5 * (float(first["sigma_z"]) + float(candidates[1]["sigma_z"])),
        )
        first_gap = float(candidates[1]["z"]) - float(first["z"])
        first_order_risk = min(1.0, first_margin / max(first_gap, np.finfo(np.float64).eps))
        states = [
            {
                **first,
                "state": "uncertain" if ties else "visible",
                "order_risk": first_order_risk,
                "risk": max(float(first["risk"]), first_order_risk),
            }
        ]
        for candidate in candidates[1:]:
            margin = max(
                0.01 * float(first["z"]),
                2.5 * (float(first["sigma_z"]) + float(candidate["sigma_z"])),
            )
            gap = float(candidate["z"]) - float(first["z"])
            order_risk = min(1.0, margin / max(gap, np.finfo(np.float64).eps))
            disjoint = set(candidate["physical_ancestry"]).isdisjoint(first["physical_ancestry"])
            has_witness = bool(visible_witnesses.get(str(candidate["track_id"]), set()))
            if str(candidate["track_id"]) in ties:
                state = "uncertain"
            else:
                state = "occluded" if gap > margin and has_witness and disjoint else "uncertain"
            states.append(
                {
                    **candidate,
                    "state": state,
                    "order_risk": order_risk,
                    "risk": max(float(candidate["risk"]), order_risk),
                }
            )
        ordered[pixel] = states
    return ordered


def connected_regions(
    ordered_pixels: Mapping[tuple[int, int], Iterable[Mapping[str, Any]]], *, minimum_area: int = 16
) -> list[dict[str, Any]]:
    if minimum_area <= 0:
        raise ValueError("minimum region area must be positive")
    units: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    records: dict[tuple[str, str, tuple[int, int]], Mapping[str, Any]] = {}
    for pixel, layers in ordered_pixels.items():
        for layer in layers:
            key = (str(layer["track_id"]), str(layer["state"]))
            units[key].add(pixel)
            records[(key[0], key[1], pixel)] = layer
    output: list[dict[str, Any]] = []
    for key in sorted(units):
        unseen = set(units[key])
        while unseen:
            seed = min(unseen)
            queue: deque[tuple[int, int]] = deque([seed])
            unseen.remove(seed)
            component: list[tuple[int, int]] = []
            while queue:
                pixel = queue.popleft()
                component.append(pixel)
                y, x = pixel
                for neighbor in (
                    (yy, xx)
                    for yy in range(y - 1, y + 2)
                    for xx in range(x - 1, x + 2)
                    if (yy, xx) != pixel
                ):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
            state = key[1] if len(component) >= minimum_area else "uncertain"
            risks = sorted(float(records[(key[0], key[1], pixel)]["risk"]) for pixel in component)
            if any(not math.isfinite(risk) or not 0 <= risk <= 1 for risk in risks):
                raise ValueError("region risk inputs must be finite in [0,1]")
            output.append(
                {
                    "track_id": key[0],
                    "state": state,
                    "pixels": sorted(component),
                    "area": len(component),
                    "risk": risks[max(0, math.ceil(0.9 * len(risks)) - 1)],
                }
            )
    return output


__all__ = [
    "build_micro_surfaces",
    "connected_regions",
    "dense_depth_order",
    "estimate_normal",
    "orient_normal",
    "rasterize_patch",
    "voxel_index",
]
