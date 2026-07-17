"""Deterministic Gate A and checkpoint-backed evaluation primitives."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import math
import random
from typing import Any

import numpy as np

from .errors import ContractError, NonFiniteError


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise NonFiniteError(f"{name} must be finite")
    return result


def division(numerator: float, denominator: float, *, zero: float = 0.0) -> float:
    if denominator < 0:
        raise ContractError("metric denominator cannot be negative")
    return zero if denominator == 0 else float(numerator / denominator)


def binary_counts(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray | None = None) -> dict[str, int]:
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if pred.shape != ref.shape:
        raise ContractError("prediction/reference mask shapes differ")
    mask = np.ones(pred.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if mask.shape != pred.shape:
        raise ContractError("valid mask shape differs")
    return {
        "tp": int(np.count_nonzero(mask & pred & ref)),
        "fp": int(np.count_nonzero(mask & pred & ~ref)),
        "fn": int(np.count_nonzero(mask & ~pred & ref)),
        "tn": int(np.count_nonzero(mask & ~pred & ~ref)),
    }


def prf_iou(tp: int, fp: int, fn: int) -> dict[str, float]:
    if min(tp, fp, fn) < 0:
        raise ContractError("counts cannot be negative")
    precision = division(tp, tp + fp)
    recall = division(tp, tp + fn)
    f1 = division(2 * precision * recall, precision + recall)
    iou = division(tp, tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def _shift(mask: np.ndarray, dy: int, dx: int, fill: bool = False) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    output = np.full(source.shape, fill, dtype=bool)
    y0 = max(0, dy)
    y1 = source.shape[0] + min(0, dy)
    x0 = max(0, dx)
    x1 = source.shape[1] + min(0, dx)
    sy0 = max(0, -dy)
    sy1 = source.shape[0] - max(0, dy)
    sx0 = max(0, -dx)
    sx1 = source.shape[1] - max(0, dx)
    if y0 < y1 and x0 < x1:
        output[y0:y1, x0:x1] = source[sy0:sy1, sx0:sx1]
    return output


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ContractError("dilation radius cannot be negative")
    source = np.asarray(mask, dtype=bool)
    result = np.zeros(source.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                result |= _shift(source, dy, dx)
    return result


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ContractError("erosion radius cannot be negative")
    source = np.asarray(mask, dtype=bool)
    result = np.ones(source.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                result &= _shift(source, dy, dx, fill=False)
    return result


def boundary(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    return dilate(source, 1) & ~erode(source, 1)


def boundary_counts(prediction: np.ndarray, reference: np.ndarray, tolerance: int, valid: np.ndarray | None = None) -> dict[str, int]:
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if pred.shape != ref.shape:
        raise ContractError("prediction/reference mask shapes differ")
    permitted = np.ones(pred.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if permitted.shape != pred.shape:
        raise ContractError("boundary valid mask shape differs")
    pred_b = boundary(pred) & permitted
    ref_b = boundary(ref) & permitted
    return {
        "precision_hits": int(np.count_nonzero(pred_b & dilate(ref_b, tolerance))),
        "precision_denominator": int(np.count_nonzero(pred_b)),
        "recall_hits": int(np.count_nonzero(ref_b & dilate(pred_b, tolerance))),
        "recall_denominator": int(np.count_nonzero(ref_b)),
    }


def boundary_prf(counts: Mapping[str, int]) -> dict[str, float]:
    precision = division(int(counts["precision_hits"]), int(counts["precision_denominator"]))
    recall = division(int(counts["recall_hits"]), int(counts["recall_denominator"]))
    return {"precision": precision, "recall": recall, "f1": division(2 * precision * recall, precision + recall)}


def spatial_frame_counts(
    prediction_union: np.ndarray,
    reference_union: np.ndarray,
    *,
    spatial_complete: bool,
    tolerance: int = 4,
) -> dict[str, Any]:
    """Score full semantic unions; unmatched predictions remain false positives."""

    if spatial_complete is not True:
        return {"evaluable": False, "reason": "spatial_review_incomplete"}
    region = binary_counts(prediction_union, reference_union)
    boundaries = boundary_counts(prediction_union, reference_union, tolerance)
    contributing = bool(region["tp"] + region["fp"] + region["fn"])
    return {
        "evaluable": True,
        "contributing": contributing,
        "true_negative": not contributing,
        "region_counts": region,
        "boundary_counts": boundaries,
    }


def aggregate_spatial_frames(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluable = [item for item in frames if item.get("evaluable") is True]
    if not evaluable:
        return {"evaluable": False, "reason": "no_spatial_complete_frames"}
    region = {key: sum(int(item["region_counts"][key]) for item in evaluable) for key in ("tp", "fp", "fn", "tn")}
    boundaries = {
        key: sum(int(item["boundary_counts"][key]) for item in evaluable)
        for key in ("precision_hits", "precision_denominator", "recall_hits", "recall_denominator")
    }
    contributing = bool(region["tp"] + region["fp"] + region["fn"])
    return {
        "evaluable": True,
        "contributing": contributing,
        "true_negative": not contributing,
        "region_counts": region,
        "region": prf_iou(region["tp"], region["fp"], region["fn"]),
        "boundary_counts": boundaries,
        "boundary": boundary_prf(boundaries),
        "frame_count": len(evaluable),
    }


def macro_mean(values: Iterable[float | None]) -> float | None:
    parsed = [_finite(value, "macro value") for value in values if value is not None]
    return None if not parsed else float(sum(parsed) / len(parsed))


def aggregate_window_scene(
    window_results: Sequence[Mapping[str, Any]],
    *,
    metric_path: Sequence[str],
) -> dict[str, Any]:
    """Unweighted contributing-window -> scene -> overall macro."""

    by_scene: dict[str, list[float]] = defaultdict(list)
    for item in window_results:
        if item.get("evaluable") is not True or item.get("contributing") is not True:
            continue
        value: Any = item
        for key in metric_path:
            value = value[key]
        by_scene[str(item["scene"])].append(_finite(value, ".".join(metric_path)))
    scene_values = {scene: macro_mean(values) for scene, values in sorted(by_scene.items())}
    return {"scene": scene_values, "overall": macro_mean(scene_values.values())}


def match_transitions(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    tolerance_frames: int = 1,
) -> dict[str, Any]:
    """Frozen greedy one-to-one match by type and stable sorted candidate key."""

    candidate_rows = []
    for prediction in predictions:
        ptype = str(prediction["type"])
        if ptype not in {"reveal", "hide"}:
            continue
        for reference in references:
            if str(reference["type"]) != ptype:
                continue
            offset = abs(int(prediction["frame"]) - int(reference["frame"]))
            if offset <= tolerance_frames:
                candidate_rows.append(
                    (
                        offset,
                        int(prediction["frame"]),
                        int(reference["frame"]),
                        str(prediction["event_id"]),
                        str(reference["event_id"]),
                    )
                )
    used_prediction: set[str] = set()
    used_reference: set[str] = set()
    matches = []
    for offset, pframe, rframe, pid, rid in sorted(candidate_rows):
        if pid in used_prediction or rid in used_reference:
            continue
        used_prediction.add(pid)
        used_reference.add(rid)
        matches.append({"predicted_event_id": pid, "reference_event_id": rid, "absolute_offset": offset, "predicted_frame": pframe, "reference_frame": rframe})
    valid_predictions = [item for item in predictions if str(item["type"]) in {"reveal", "hide"}]
    valid_references = [item for item in references if str(item["type"]) in {"reveal", "hide"}]
    tp = len(matches)
    fp = len(valid_predictions) - tp
    fn = len(valid_references) - tp
    return {"matches": matches, "tp": tp, "fp": fp, "fn": fn, **prf_iou(tp, fp, fn)}


def event_window_metrics(predictions: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = match_transitions(predictions, references)
    contributing = bool(result["tp"] + result["fp"] + result["fn"])
    return {**result, "evaluable": True, "contributing": contributing, "true_negative": not contributing, "specificity_unit": 1 if not contributing else 0}


def _binary_auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float | None:
    positives = [_finite(value, "positive score") for value in positive_scores]
    negatives = [_finite(value, "negative score") for value in negative_scores]
    if not positives or not negatives:
        return None
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            favorable += 1.0 if positive > negative else (0.5 if positive == negative else 0.0)
    return favorable / (len(positives) * len(negatives))


def ordering_metrics(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score directed front/rear units; abstention stays in coverage denominator."""

    if not units:
        return {"evaluable": False, "reason": "no_ordering_units"}
    accepted = []
    for unit in units:
        score = unit.get("score")
        if score is None or unit.get("abstained") is True:
            continue
        accepted.append(_finite(score, "ordering score"))
    coverage = len(accepted) / len(units)
    if not accepted:
        return {"evaluable": False, "coverage": coverage, "reason": "all_ordering_units_abstained"}
    accuracy = sum(score > 0.0 for score in accepted) / len(accepted)
    # Each directed unit is duplicated with reversed orientation and sign.
    auroc = _binary_auroc(accepted, [-score for score in accepted])
    return {"evaluable": auroc is not None, "accuracy": accuracy, "auroc": auroc, "coverage": coverage, "accepted_count": len(accepted), "unit_count": len(units)}


def relative_error_reduction(baseline_error: float, method_error: float) -> float:
    baseline = _finite(baseline_error, "baseline error")
    method = _finite(method_error, "method error")
    if baseline < 0 or method < 0:
        raise ContractError("errors must be nonnegative")
    if baseline == 0:
        return 0.0 if method == 0 else -math.inf
    return (baseline - method) / baseline


def relative_error_regression(reference_error: float, candidate_error: float) -> float:
    reference = _finite(reference_error, "reference error")
    candidate = _finite(candidate_error, "candidate error")
    if reference < 0 or candidate < 0:
        raise ContractError("errors must be nonnegative")
    if reference == 0:
        return 0.0 if candidate == 0 else math.inf
    return (candidate - reference) / reference


def isotonic_pav(
    risks: Sequence[float],
    errors: Sequence[int | bool],
    *,
    weights: Sequence[float] | None = None,
    stable_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit increasing risk-to-error probability by stable weighted PAV."""

    if len(risks) != len(errors) or not risks:
        raise ContractError("PAV risks/errors must be nonempty and aligned")
    if weights is None:
        weights = [1.0] * len(risks)
    if stable_ids is None:
        stable_ids = [f"{index:012d}" for index in range(len(risks))]
    if len(weights) != len(risks) or len(stable_ids) != len(risks):
        raise ContractError("PAV auxiliary arrays are not aligned")
    rows = []
    for risk, error, weight, stable_id in zip(risks, errors, weights, stable_ids, strict=True):
        r = _finite(risk, "risk")
        w = _finite(weight, "PAV weight")
        if not 0.0 <= r <= 1.0 or int(error) not in {0, 1} or w <= 0:
            raise ContractError("PAV requires risk in [0,1], binary errors, and positive weights")
        rows.append((r, str(stable_id), int(error), w))
    rows.sort(key=lambda item: (item[0], item[1]))
    blocks = []
    for index, (risk, stable_id, error, weight) in enumerate(rows):
        blocks.append({"start": index, "end": index, "weight": weight, "error_weight": error * weight})
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left["error_weight"] / left["weight"] <= right["error_weight"] / right["weight"]:
                break
            blocks[-2:] = [
                {
                    "start": left["start"],
                    "end": right["end"],
                    "weight": left["weight"] + right["weight"],
                    "error_weight": left["error_weight"] + right["error_weight"],
                }
            ]
    fitted_sorted = [0.0] * len(rows)
    for block in blocks:
        value = block["error_weight"] / block["weight"]
        for index in range(block["start"], block["end"] + 1):
            fitted_sorted[index] = value
    mapping = {stable_id: fitted for (_, stable_id, _, _), fitted in zip(rows, fitted_sorted, strict=True)}
    fitted = [mapping[str(stable_id)] for stable_id in stable_ids]
    return {"fitted_probability": fitted, "blocks": blocks, "sorted_stable_ids": [item[1] for item in rows]}


def equal_mass_ece(probability: Sequence[float], errors: Sequence[int | bool], stable_ids: Sequence[str], bins: int = 15) -> dict[str, Any]:
    if not (len(probability) == len(errors) == len(stable_ids)) or not probability:
        raise ContractError("ECE arrays must be nonempty and aligned")
    rows = []
    for p, error, stable_id in zip(probability, errors, stable_ids, strict=True):
        value = _finite(p, "calibrated probability")
        if not 0.0 <= value <= 1.0 or int(error) not in {0, 1}:
            raise ContractError("ECE needs probabilities in [0,1] and binary errors")
        rows.append((value, str(stable_id), int(error)))
    rows.sort(key=lambda item: (item[0], item[1]))
    bin_count = min(int(bins), len(rows))
    if bin_count <= 0:
        raise ContractError("ECE bin count must be positive")
    chunks = np.array_split(np.arange(len(rows)), bin_count)
    summaries = []
    ece = 0.0
    for indices in chunks:
        values = [rows[int(index)] for index in indices]
        confidence = sum(item[0] for item in values) / len(values)
        error_rate = sum(item[2] for item in values) / len(values)
        weight = len(values) / len(rows)
        ece += weight * abs(confidence - error_rate)
        summaries.append({"count": len(values), "mean_probability": confidence, "error_rate": error_rate})
    brier = sum((item[0] - item[2]) ** 2 for item in rows) / len(rows)
    return {"ece": ece, "brier": brier, "bins": summaries, "binning": "stable_equal_mass"}


def risk_coverage(units: Sequence[Mapping[str, Any]], thresholds: Sequence[float] = (0.25, 0.5, 0.75)) -> list[dict[str, Any]]:
    if not units:
        return []
    output = []
    for threshold in thresholds:
        limit = _finite(threshold, "risk threshold")
        accepted = [item for item in units if _finite(item["risk"], "unit risk") <= limit]
        errors = [int(item["error"]) for item in accepted]
        if any(value not in {0, 1} for value in errors):
            raise ContractError("risk-coverage errors must be binary")
        output.append(
            {
                "risk_threshold": limit,
                "coverage": len(accepted) / len(units),
                "accepted_count": len(accepted),
                "error_rate": None if not accepted else sum(errors) / len(errors),
            }
        )
    return output


def track_cluster_bootstrap(
    track_rows: Mapping[str, Sequence[float]],
    statistic,
    *,
    replicates: int = 10000,
    seed: int = 20260715,
) -> dict[str, Any]:
    """Deterministic track-clustered percentile bootstrap."""

    track_ids = sorted(track_rows)
    if not track_ids:
        return {"evaluable": False, "reason": "no_tracks"}
    rng = random.Random(seed)
    values = []
    for _ in range(int(replicates)):
        sampled = [track_ids[rng.randrange(len(track_ids))] for _ in track_ids]
        flattened = [value for track_id in sampled for value in track_rows[track_id]]
        values.append(_finite(statistic(flattened), "bootstrap statistic"))
    ordered = sorted(values)
    lower = ordered[max(0, math.ceil(0.025 * len(ordered)) - 1)]
    upper = ordered[max(0, math.ceil(0.975 * len(ordered)) - 1)]
    return {"evaluable": True, "replicates": len(values), "seed": seed, "percentile_2_5": lower, "percentile_97_5": upper}



def clamp_srgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if not np.isfinite(array).all():
        raise NonFiniteError("image contains nonfinite values")
    return np.clip(array, 0.0, 1.0)


def pooled_masked_psnr(
    render_frames: Sequence[np.ndarray],
    ground_truth_frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    mse_floor: float = 1e-12,
) -> dict[str, Any]:
    """Pool channel squared error over every valid event pixel/frame."""

    if not (len(render_frames) == len(ground_truth_frames) == len(masks)):
        raise ContractError("PSNR frame arrays are not aligned")
    squared_sum = 0.0
    channel_count = 0
    valid_frames = 0
    for render, truth, mask in zip(render_frames, ground_truth_frames, masks, strict=True):
        predicted = clamp_srgb(render)
        target = clamp_srgb(truth)
        region = np.asarray(mask, dtype=bool)
        if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[2] != 3 or region.shape != predicted.shape[:2]:
            raise ContractError("PSNR image/mask shapes differ")
        if not region.any():
            continue
        difference = predicted[region] - target[region]
        squared_sum += float(np.sum(difference * difference))
        channel_count += int(difference.size)
        valid_frames += 1
    if channel_count == 0:
        return {"evaluable": False, "reason": "no_valid_event_pixels", "excluded_frame_count": len(masks)}
    mse = squared_sum / channel_count
    return {
        "evaluable": True,
        "mse": mse,
        "psnr_db": 10.0 * math.log10(1.0 / max(mse, mse_floor)),
        "exact_zero_mse": mse == 0.0,
        "valid_frame_count": valid_frames,
        "valid_channel_count": channel_count,
    }


def masked_l1(render: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray) -> float | None:
    predicted = clamp_srgb(render)
    target = clamp_srgb(ground_truth)
    region = np.asarray(mask, dtype=bool)
    if predicted.shape != target.shape or predicted.ndim != 3 or region.shape != predicted.shape[:2]:
        raise ContractError("masked L1 image/mask shapes differ")
    if not region.any():
        return None
    return float(np.mean(np.abs(predicted[region] - target[region])))


def _bilinear_sample(array: np.ndarray, coordinates_x: np.ndarray, coordinates_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(array, dtype=np.float64)
    height, width = source.shape[:2]
    if coordinates_x.shape != coordinates_y.shape:
        raise ContractError("sampling coordinate shapes differ")
    valid = np.isfinite(coordinates_x) & np.isfinite(coordinates_y) & (coordinates_x >= 0.0) & (coordinates_x <= width - 1) & (coordinates_y >= 0.0) & (coordinates_y <= height - 1)
    x0 = np.floor(np.clip(coordinates_x, 0, width - 1)).astype(np.int64)
    y0 = np.floor(np.clip(coordinates_y, 0, height - 1)).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = coordinates_x - x0
    wy = coordinates_y - y0
    if source.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    sampled = source[y0, x0] * (1 - wx) * (1 - wy) + source[y0, x1] * wx * (1 - wy) + source[y1, x0] * (1 - wx) * wy + source[y1, x1] * wx * wy
    return sampled, valid


def backward_warp(previous: np.ndarray, backward_flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flow = np.asarray(backward_flow, dtype=np.float64)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ContractError("backward flow must be HxWx2")
    yy, xx = np.indices(flow.shape[:2], dtype=np.float64)
    return _bilinear_sample(previous, xx + flow[..., 0], yy + flow[..., 1])


def flow_relative_flicker(
    current_render: np.ndarray,
    previous_render: np.ndarray,
    current_truth: np.ndarray,
    previous_truth: np.ndarray,
    backward_flow: np.ndarray,
    current_mask: np.ndarray,
    previous_mask: np.ndarray,
    flow_valid: np.ndarray,
    cycle_error: np.ndarray,
    *,
    cycle_maximum: float = 1.5,
) -> dict[str, Any]:
    current_r = clamp_srgb(current_render)
    current_g = clamp_srgb(current_truth)
    warped_r, inside = backward_warp(clamp_srgb(previous_render), backward_flow)
    warped_g, _ = backward_warp(clamp_srgb(previous_truth), backward_flow)
    warped_mask, _ = backward_warp(np.asarray(previous_mask, dtype=np.float64), backward_flow)
    valid = np.asarray(current_mask, dtype=bool) & (warped_mask > 0.5) & np.asarray(flow_valid, dtype=bool) & (np.asarray(cycle_error, dtype=np.float64) <= cycle_maximum) & inside
    if not valid.any():
        return {"evaluable": False, "reason": "no_valid_flow_relative_pixels"}
    residual = (current_r - warped_r) - (current_g - warped_g)
    return {"evaluable": True, "l1": float(np.mean(np.abs(residual[valid]))), "valid_pixel_count": int(np.count_nonzero(valid))}


def reveal_ghost(
    render: np.ndarray,
    truth: np.ndarray,
    pre_reveal_truth: np.ndarray,
    backward_flow_from_reveal: np.ndarray,
    reveal_mask: np.ndarray,
    flow_valid: np.ndarray,
) -> dict[str, Any]:
    predicted = clamp_srgb(render)
    target = clamp_srgb(truth)
    warped_foreground, inside = backward_warp(clamp_srgb(pre_reveal_truth), backward_flow_from_reveal)
    valid = np.asarray(reveal_mask, dtype=bool) & np.asarray(flow_valid, dtype=bool) & inside
    if not valid.any():
        return {"evaluable": False, "reason": "no_valid_reveal_pixels"}
    reconstruction = np.mean(np.abs(predicted - target), axis=2)
    foreground_similarity = np.mean(np.abs(predicted - warped_foreground), axis=2)
    trail = np.maximum(0.0, reconstruction - foreground_similarity)
    return {"evaluable": True, "trail_l1": float(np.mean(trail[valid])), "valid_pixel_count": int(np.count_nonzero(valid))}


def static_scene_admission(
    frame_metrics: Sequence[Mapping[str, Any]],
    flicker_metrics: Sequence[Mapping[str, Any]],
    *,
    minimum_frames: int = 270,
    minimum_pairs: int = 269,
) -> dict[str, Any]:
    frames = [item for item in frame_metrics if item.get("evaluable") is True]
    pairs = [item for item in flicker_metrics if item.get("evaluable") is True]
    if len(frames) < minimum_frames or len(pairs) < minimum_pairs:
        return {"evaluable": False, "reason": "under_minimum_static_population", "valid_frame_count": len(frames), "valid_pair_count": len(pairs)}
    result = {name: macro_mean(item.get(name) for item in frames) for name in ("psnr_db", "lpips", "reconstruction_l1")}
    result["flicker"] = macro_mean(item.get("l1") for item in pairs)
    if any(value is None for value in result.values()):
        return {"evaluable": False, "reason": "missing_static_metric", "valid_frame_count": len(frames), "valid_pair_count": len(pairs)}
    return {"evaluable": True, **result, "valid_frame_count": len(frames), "valid_pair_count": len(pairs)}

__all__ = [
    "backward_warp",
    "clamp_srgb",
    "flow_relative_flicker",
    "masked_l1",
    "pooled_masked_psnr",
    "reveal_ghost",
    "static_scene_admission",
    "aggregate_spatial_frames",
    "aggregate_window_scene",
    "binary_counts",
    "boundary",
    "boundary_counts",
    "boundary_prf",
    "dilate",
    "equal_mass_ece",
    "event_window_metrics",
    "isotonic_pav",
    "macro_mean",
    "match_transitions",
    "ordering_metrics",
    "prf_iou",
    "relative_error_reduction",
    "relative_error_regression",
    "risk_coverage",
    "spatial_frame_counts",
    "track_cluster_bootstrap",
]
