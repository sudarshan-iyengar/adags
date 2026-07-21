"""Bounded real-data geometry diagnostics for the Phase 9 fast cycle."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .errors import ContractError


_QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)


def _distribution(values: Sequence[float] | np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0, "mean": None, "maximum": None, "quantiles": {}}
    if not np.isfinite(array).all():
        raise ContractError("pilot distribution contains nonfinite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
        "quantiles": {str(q): float(np.quantile(array, q)) for q in _QUANTILES},
    }


def _sample_world(
    prediction: Mapping[str, Any],
    view_index: int,
    *,
    stride: int,
    depth_override: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(
        prediction["depth"][view_index] if depth_override is None else depth_override,
        dtype=np.float64,
    )
    K = np.asarray(prediction["intrinsics"][view_index], dtype=np.float64)
    w2c = np.asarray(prediction["extrinsics"][view_index], dtype=np.float64)
    if depth.ndim != 2 or K.shape != (3, 3) or w2c.shape != (4, 4):
        raise ContractError("pilot source prediction has invalid geometry")
    if stride <= 0:
        raise ContractError("pilot stride must be positive")
    ys = np.arange(stride // 2, depth.shape[0], stride, dtype=np.int64)
    xs = np.arange(stride // 2, depth.shape[1], stride, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pixels = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1))).astype(np.float64)
    optical_z = depth[grid_y, grid_x].reshape(-1)
    valid = np.isfinite(optical_z) & (optical_z > 0.0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ContractError("pilot source valid mask has wrong shape")
        valid &= mask[grid_y, grid_x].reshape(-1)
    pixels = pixels[valid]
    optical_z = optical_z[valid]
    homogeneous = np.column_stack((pixels, np.ones(len(pixels), dtype=np.float64)))
    rays = homogeneous @ np.linalg.inv(K).T
    camera = rays * optical_z[:, None]
    world = (camera - w2c[:3, 3][None, :]) @ w2c[:3, :3]
    if not np.isfinite(world).all():
        raise ContractError("pilot unprojection produced nonfinite world points")
    return world, pixels


def _project_world(
    world: np.ndarray,
    *,
    K: np.ndarray,
    w2c: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(world, dtype=np.float64)
    camera = points @ np.asarray(w2c, dtype=np.float64)[:3, :3].T
    camera += np.asarray(w2c, dtype=np.float64)[:3, 3][None, :]
    uvw = camera @ np.asarray(K, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = uvw[:, :2] / uvw[:, 2:3]
    valid = (
        np.isfinite(pixels).all(axis=1)
        & np.isfinite(camera[:, 2])
        & (camera[:, 2] > 0.0)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= height - 1)
    )
    return pixels, camera[:, 2], valid


def _unique_camera_predictions(
    group_predictions: Sequence[Mapping[str, Any]],
    group_members: Sequence[Sequence[str]],
) -> Mapping[str, tuple[Mapping[str, Any], int]]:
    if len(group_predictions) != len(group_members) or not group_predictions:
        raise ContractError("pilot groups and predictions are inconsistent")
    result: dict[str, tuple[Mapping[str, Any], int]] = {}
    for prediction, members in zip(group_predictions, group_members, strict=True):
        if len(members) != int(np.asarray(prediction["depth"]).shape[0]):
            raise ContractError("pilot group member count differs from prediction")
        for index, camera_id in enumerate(members):
            result.setdefault(str(camera_id), (prediction, index))
    if len(result) < 3:
        raise ContractError("pilot requires at least three physical cameras")
    return result


def _raw_depth_layer_candidates(
    samples: Sequence[tuple[float, str, float]],
) -> list[Mapping[str, Any]]:
    ordered = sorted(samples, key=lambda item: (item[0], item[1]))
    clusters: list[list[tuple[float, str, float]]] = []
    for sample in ordered:
        if not clusters:
            clusters.append([sample])
            continue
        previous = float(np.median([item[0] for item in clusters[-1]]))
        margin = max(
            0.01 * abs(previous),
            2.5 * (0.01 * abs(previous) + 0.01 * abs(sample[0])),
        )
        if sample[0] - previous <= margin:
            clusters[-1].append(sample)
        else:
            clusters.append([sample])

    layers = []
    for cluster in clusters:
        cameras = sorted({item[1] for item in cluster})
        layers.append(
            {
                "median_optical_z": float(np.median([item[0] for item in cluster])),
                "physical_camera_count": len(cameras),
                "physical_cameras": cameras,
                "sample_count": len(cluster),
                "median_risk": float(np.median([item[2] for item in cluster])),
            }
        )
    return layers


def _cluster_depth_layers(
    samples: Sequence[tuple[float, str, float]],
    *,
    minimum_cameras: int,
) -> list[Mapping[str, Any]]:
    return [
        layer
        for layer in _raw_depth_layer_candidates(samples)
        if layer["physical_camera_count"] >= minimum_cameras
    ]


def anchor_duplicate_diagnostic(
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    first_confidence: np.ndarray,
    second_confidence: np.ndarray,
    *,
    relative_limit: float = 0.05,
    calibration_stride: int = 8,
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray]:
    """Aggregate a repeated physical camera and abstain on inconsistent pixels."""

    first = np.asarray(first_depth, dtype=np.float64)
    second = np.asarray(second_depth, dtype=np.float64)
    confidence_first = np.asarray(first_confidence, dtype=np.float64)
    confidence_second = np.asarray(second_confidence, dtype=np.float64)
    if (
        first.shape != second.shape
        or first.shape != confidence_first.shape
        or first.shape != confidence_second.shape
        or first.ndim != 2
    ):
        raise ContractError("anchor duplicate arrays must be aligned 2D fields")
    if (
        not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or not np.isfinite(confidence_first).all()
        or not np.isfinite(confidence_second).all()
        or np.any(first <= 0.0)
        or np.any(second <= 0.0)
        or np.any(confidence_first < 0.0)
        or np.any(confidence_second < 0.0)
    ):
        raise ContractError("anchor duplicate arrays violate finite positive depth/confidence")
    if not math.isfinite(relative_limit) or relative_limit <= 0.0:
        raise ContractError("anchor duplicate limit must be finite and positive")
    if calibration_stride <= 1:
        raise ContractError("anchor calibration stride must exceed one")

    denominator = np.maximum(first + second, np.finfo(np.float64).tiny)
    relative_half_difference = np.abs(first - second) / denominator
    retained = relative_half_difference <= relative_limit

    lower_is_first = first <= second
    lower_depth = np.where(lower_is_first, first, second)
    upper_depth = np.where(lower_is_first, second, first)
    lower_weight = np.where(lower_is_first, confidence_first, confidence_second)
    upper_weight = np.where(lower_is_first, confidence_second, confidence_first)
    total_weight = lower_weight + upper_weight
    aggregate = np.where(
        (total_weight == 0.0) | (lower_weight >= total_weight / 2.0),
        lower_depth,
        upper_depth,
    )

    yy, xx = np.indices(first.shape)
    calibration = ((yy // calibration_stride + xx // calibration_stride) % 2) == 0
    held_out = ~calibration
    scale = float(np.median(first[calibration] / second[calibration]))
    scaled_second = second * scale
    raw_heldout = np.abs(first[held_out] - second[held_out]) / np.maximum(
        first[held_out] + second[held_out], np.finfo(np.float64).tiny
    )
    scaled_heldout = np.abs(first[held_out] - scaled_second[held_out]) / np.maximum(
        first[held_out] + scaled_second[held_out], np.finfo(np.float64).tiny
    )

    cell_y = yy // calibration_stride
    cell_x = xx // calibration_stride
    rejected = ~retained
    total_cells = int((cell_y.max() + 1) * (cell_x.max() + 1))
    rejected_cells = {
        (int(y), int(x)) for y, x in zip(cell_y[rejected], cell_x[rejected], strict=True)
    }
    report = {
        "relative_estimator": "two-sample half-difference over positive-depth sum",
        "relative_limit": relative_limit,
        "relative_disagreement": _distribution(relative_half_difference),
        "retained_pixel_count": int(np.count_nonzero(retained)),
        "rejected_pixel_count": int(np.count_nonzero(rejected)),
        "retained_fraction": float(np.mean(retained)),
        "rejected_cell_count": len(rejected_cells),
        "rejected_cell_fraction": len(rejected_cells) / total_cells,
        "aggregate_rule": "confidence-weighted lower depth crossing half total; all-zero uses lower depth",
        "heldout_scale_diagnostic": {
            "calibration_partition": "alternating 8x8 processed-image cells",
            "scale_second_to_first": scale,
            "raw_relative_disagreement": _distribution(raw_heldout),
            "scaled_relative_disagreement": _distribution(scaled_heldout),
            "raw_within_limit_fraction": float(np.mean(raw_heldout <= relative_limit)),
            "scaled_within_limit_fraction": float(np.mean(scaled_heldout <= relative_limit)),
            "applied_to_geometry": False,
        },
    }
    return report, aggregate, retained


def evaluate_frame_geometry(
    group_predictions: Sequence[Mapping[str, Any]],
    group_members: Sequence[Sequence[str]],
    *,
    target_K: np.ndarray,
    target_w2c: np.ndarray,
    target_width: int,
    target_height: int,
    stride: int = 8,
    minimum_cameras: int = 3,
    maximum_depth_sigma: float = 2.5,
    target_bin_pixels: int = 8,
    camera_depth_overrides: Mapping[str, np.ndarray] | None = None,
    camera_valid_masks: Mapping[str, np.ndarray] | None = None,
    include_layer_records: bool = False,
) -> tuple[Mapping[str, Any], set[tuple[int, int]]] | tuple[
    Mapping[str, Any], set[tuple[int, int]], list[Mapping[str, Any]]
]:
    """Measure calibrated cross-view support and target-projected depth ordering."""

    cameras = _unique_camera_predictions(group_predictions, group_members)
    depth_overrides = dict(camera_depth_overrides or {})
    valid_masks = dict(camera_valid_masks or {})
    if not set(depth_overrides).issubset(cameras) or not set(valid_masks).issubset(cameras):
        raise ContractError("pilot depth/mask override names an unknown camera")
    pair_normalized_residuals: list[float] = []
    pair_pixel_residuals: list[float] = []
    total_pair_opportunities = 0
    valid_pair_projections = 0
    agreeing_pairs = 0
    support_histogram: Counter[int] = Counter()
    supported_world: list[np.ndarray] = []
    supported_source: list[str] = []
    supported_risk: list[float] = []

    for source_camera, (source_prediction, source_index) in sorted(cameras.items()):
        world, _ = _sample_world(
            source_prediction,
            source_index,
            stride=stride,
            depth_override=depth_overrides.get(source_camera),
            valid_mask=valid_masks.get(source_camera),
        )
        total_pair_opportunities += len(world) * (len(cameras) - 1)
        support = np.ones(len(world), dtype=np.int64)
        residuals_by_point: list[list[float]] = [[] for _ in range(len(world))]
        for target_camera, (target_prediction, target_index) in sorted(cameras.items()):
            if target_camera == source_camera:
                continue
            target_depth = np.asarray(
                depth_overrides.get(
                    target_camera, target_prediction["depth"][target_index]
                ),
                dtype=np.float64,
            )
            pixels, projected_z, valid = _project_world(
                world,
                K=np.asarray(target_prediction["intrinsics"][target_index]),
                w2c=np.asarray(target_prediction["extrinsics"][target_index]),
                width=target_depth.shape[1],
                height=target_depth.shape[0],
            )
            valid_indices = np.flatnonzero(valid)
            valid_pair_projections += len(valid_indices)
            if len(valid_indices) == 0:
                continue
            rounded_x = np.rint(pixels[valid_indices, 0]).astype(np.int64)
            rounded_y = np.rint(pixels[valid_indices, 1]).astype(np.int64)
            observed_z = target_depth[rounded_y, rounded_x]
            finite_positive = np.isfinite(observed_z) & (observed_z > 0.0)
            target_mask = valid_masks.get(target_camera)
            if target_mask is not None:
                mask = np.asarray(target_mask, dtype=bool)
                if mask.shape != target_depth.shape:
                    raise ContractError("pilot target valid mask has wrong shape")
                finite_positive &= mask[rounded_y, rounded_x]
            selected = valid_indices[finite_positive]
            if len(selected) == 0:
                continue
            observed_z = observed_z[finite_positive]
            projected = projected_z[selected]
            sigma_pair = np.sqrt((0.01 * np.abs(projected)) ** 2 + (0.01 * np.abs(observed_z)) ** 2)
            normalized = np.abs(projected - observed_z) / sigma_pair
            pixel_residual = np.linalg.norm(
                pixels[selected] - np.rint(pixels[selected]), axis=1
            )
            pair_normalized_residuals.extend(normalized.tolist())
            pair_pixel_residuals.extend(pixel_residual.tolist())
            accepted = normalized <= maximum_depth_sigma
            agreeing_pairs += int(np.count_nonzero(accepted))
            accepted_indices = selected[accepted]
            support[accepted_indices] += 1
            for point_index, value in zip(selected, normalized, strict=True):
                residuals_by_point[int(point_index)].append(float(value))
        for point_index, count in enumerate(support):
            support_histogram[int(count)] += 1
            if count < minimum_cameras:
                continue
            local = residuals_by_point[point_index]
            residual_risk = (
                min(1.0, float(np.median(local)) / maximum_depth_sigma)
                if local else 1.0
            )
            missing_risk = 1.0 - float(count) / len(cameras)
            supported_world.append(world[point_index])
            supported_source.append(source_camera)
            supported_risk.append(max(residual_risk, missing_risk))

    supported_points = (
        np.stack(supported_world)
        if supported_world
        else np.empty((0, 3), dtype=np.float64)
    )
    target_pixels, target_depths, target_valid = _project_world(
        supported_points,
        K=np.asarray(target_K),
        w2c=np.asarray(target_w2c),
        width=target_width,
        height=target_height,
    )
    bins: dict[tuple[int, int], list[tuple[float, str, float]]] = defaultdict(list)
    for index in np.flatnonzero(target_valid):
        key = (
            int(target_pixels[index, 0] // target_bin_pixels),
            int(target_pixels[index, 1] // target_bin_pixels),
        )
        bins[key].append(
            (
                float(target_depths[index]),
                supported_source[index],
                supported_risk[index],
            )
        )

    supported_bins: set[tuple[int, int]] = set()
    multilayer_bins = 0
    depth_gaps = []
    bin_camera_histogram: Counter[int] = Counter()
    raw_layer_histogram: Counter[int] = Counter()
    accepted_layer_histogram: Counter[int] = Counter()
    rejection_counts: Counter[str] = Counter()
    supported_bin_depth_spans = []
    two_raw_layer_bins = 0
    two_accepted_layer_bins = 0
    ordered_layer_records: list[Mapping[str, Any]] = []
    for key, samples in bins.items():
        camera_count = len({item[1] for item in samples})
        bin_camera_histogram[camera_count] += 1
        if camera_count < minimum_cameras:
            rejection_counts["insufficient_bin_cameras"] += 1
            continue

        supported_bins.add(key)
        raw_layers = _raw_depth_layer_candidates(samples)
        layers = [
            layer
            for layer in raw_layers
            if layer["physical_camera_count"] >= minimum_cameras
        ]
        raw_layer_histogram[len(raw_layers)] += 1
        accepted_layer_histogram[len(layers)] += 1
        if len(raw_layers) >= 2:
            two_raw_layer_bins += 1
        if len(layers) >= 2:
            two_accepted_layer_bins += 1

        depths = [item[0] for item in samples]
        supported_bin_depth_spans.append(
            (max(depths) - min(depths))
            / max(abs(min(depths)), np.finfo(np.float64).tiny)
        )
        if len(layers) >= 2:
            multilayer_bins += 1
            layer_records = []
            for ordinal, layer in enumerate(layers):
                layer_records.append(
                    {
                        "layer_ordinal": int(ordinal),
                        "depth_order": "front_to_rear",
                        "median_optical_z": float(layer["median_optical_z"]),
                        "physical_camera_count": int(layer["physical_camera_count"]),
                        "physical_cameras": list(layer["physical_cameras"]),
                        "sample_count": int(layer["sample_count"]),
                        "median_risk": float(layer["median_risk"]),
                    }
                )
            order_pairs = []
            for front, rear in zip(layers, layers[1:]):
                relative_gap = (
                    (rear["median_optical_z"] - front["median_optical_z"])
                    / max(abs(front["median_optical_z"]), np.finfo(np.float64).tiny)
                )
                depth_gaps.append(relative_gap)
                order_pairs.append(
                    {
                        "front_layer_ordinal": int(len(order_pairs)),
                        "rear_layer_ordinal": int(len(order_pairs) + 1),
                        "relative_depth_gap": float(relative_gap),
                    }
                )
            if include_layer_records:
                ordered_layer_records.append(
                    {
                        "target_bin": [int(key[0]), int(key[1])],
                        "target_bin_pixels": int(target_bin_pixels),
                        "physical_camera_count": int(camera_count),
                        "sample_count": int(len(samples)),
                        "source_cameras": sorted({item[1] for item in samples}),
                        "layers": layer_records,
                        "order_pairs": order_pairs,
                    }
                )
        elif len(raw_layers) < 2:
            rejection_counts["single_depth_cluster"] += 1
        else:
            rejection_counts["depth_clusters_insufficient_camera_support"] += 1

    if supported_bins:
        xs = [item[0] for item in supported_bins]
        ys = [item[1] for item in supported_bins]
        bounding_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        compactness = len(supported_bins) / bounding_area
    else:
        compactness = 0.0
    total_target_bins = math.ceil(target_width / target_bin_pixels) * math.ceil(
        target_height / target_bin_pixels
    )
    risk_array = np.asarray(supported_risk, dtype=np.float64)
    risk_coverage = {}
    for threshold in (0.25, 0.5, 0.75):
        risk_coverage[str(threshold)] = (
            float(np.mean(risk_array <= threshold)) if risk_array.size else 0.0
        )
    report = {
        "physical_camera_count": len(cameras),
        "source_grid_point_count": int(sum(support_histogram.values())),
        "pair_opportunity_count": int(total_pair_opportunities),
        "valid_projection_count": int(valid_pair_projections),
        "valid_projection_fraction": (
            valid_pair_projections / total_pair_opportunities
            if total_pair_opportunities else 0.0
        ),
        "cross_view_agreeing_pair_count": int(agreeing_pairs),
        "cross_view_agreement_fraction_of_valid": (
            agreeing_pairs / valid_pair_projections if valid_pair_projections else 0.0
        ),
        "normalized_depth_residual": _distribution(pair_normalized_residuals),
        "nearest_pixel_projection_residual": _distribution(pair_pixel_residuals),
        "physical_support_histogram": {
            str(key): int(value) for key, value in sorted(support_histogram.items())
        },
        "supported_source_point_count": len(supported_world),
        "target_projected_point_count": int(np.count_nonzero(target_valid)),
        "target_projection_fraction_of_supported": (
            float(np.mean(target_valid)) if len(target_valid) else 0.0
        ),
        "target_supported_bin_count": len(supported_bins),
        "target_supported_bin_fraction": len(supported_bins) / total_target_bins,
        "target_support_compactness_in_bounding_box": compactness,
        "target_ordered_multilayer_bin_count": multilayer_bins,
        "target_ordered_multilayer_bin_fraction": (
            multilayer_bins / len(supported_bins) if supported_bins else 0.0
        ),
        "ordered_layer_relative_depth_gap": _distribution(depth_gaps),
        "target_layer_opportunity": {
            "candidate_count_waterfall": [
                {"stage": "projected_target_bins", "count": len(bins)},
                {"stage": "minimum_camera_bins", "count": len(supported_bins)},
                {"stage": "two_raw_depth_cluster_bins", "count": two_raw_layer_bins},
                {"stage": "two_min_camera_layer_bins", "count": two_accepted_layer_bins},
                {"stage": "ordered_multilayer_bins", "count": multilayer_bins},
            ],
            "bin_camera_count_histogram": {
                str(key): int(value) for key, value in sorted(bin_camera_histogram.items())
            },
            "raw_depth_layer_count_histogram": {
                str(key): int(value) for key, value in sorted(raw_layer_histogram.items())
            },
            "accepted_depth_layer_count_histogram": {
                str(key): int(value) for key, value in sorted(accepted_layer_histogram.items())
            },
            "ordered_layer_rejection_counts": {
                key: int(value) for key, value in sorted(rejection_counts.items())
            },
            "supported_bin_relative_depth_span": _distribution(supported_bin_depth_spans),
            "interpretation": "aggregate target-bin opportunity diagnostics; not a surface-track or label-dependent Gate A score",
        },
        "supported_point_risk": _distribution(risk_array),
        "risk_coverage": risk_coverage,
        "threshold_authority": {
            "grid_stride_pixels": stride,
            "minimum_physical_cameras": minimum_cameras,
            "maximum_depth_sigma": maximum_depth_sigma,
            "target_bin_pixels": target_bin_pixels,
            "minimum_sigma_z_relative": 0.01,
        },
    }
    if include_layer_records:
        return report, supported_bins, ordered_layer_records
    return report, supported_bins


def temporal_bin_transitions(
    frame_bins: Sequence[tuple[int, set[tuple[int, int]]]],
) -> list[Mapping[str, Any]]:
    """Report a bounded occupancy proxy; this is not surface-track evidence."""

    transitions = []
    for (previous_frame, previous), (current_frame, current) in zip(
        frame_bins, frame_bins[1:], strict=False
    ):
        union = previous | current
        transitions.append(
            {
                "from_frame": int(previous_frame),
                "to_frame": int(current_frame),
                "retained_bin_count": len(previous & current),
                "newly_supported_bin_count": len(current - previous),
                "newly_hidden_bin_count": len(previous - current),
                "jaccard": len(previous & current) / len(union) if union else 1.0,
                "interpretation": "target-bin occupancy proxy, not surface-track reveal/hide labels",
            }
        )
    return transitions


__all__ = ["anchor_duplicate_diagnostic", "evaluate_frame_geometry", "temporal_bin_transitions"]
