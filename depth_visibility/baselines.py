"""Frozen R031/R032/R033 and R031-MT baseline calibration utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import domain_id, sha256_file
from .errors import ContractError, NonFiniteError, ProvenanceError, SchemaError


BASELINE_SCHEMA = "phase9-r031-baselines-v1"
_BASELINE_ORDER = {"R031": 0, "R032": 1, "R033": 2}


def threshold_candidates(scores: Iterable[float]) -> list[float]:
    """Return unique finite scores plus both infinities in descending order."""

    values = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in values):
        raise NonFiniteError("baseline threshold population contains nonfinite scores")
    return [math.inf, *sorted(set(values), reverse=True), -math.inf]


def selected_fraction(scores: Sequence[float] | np.ndarray, threshold: float) -> float:
    array = np.asarray(scores, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ContractError("selected fraction needs a nonempty finite population")
    if math.isnan(float(threshold)):
        raise NonFiniteError("threshold cannot be NaN")
    return float(np.count_nonzero(array >= float(threshold)) / array.size)


def match_selected_fraction(
    scores: Sequence[float] | np.ndarray,
    target_fraction: float,
    *,
    baseline_id: str | None = None,
) -> dict[str, Any]:
    """Select the nearest accepted fraction; higher threshold wins exact ties."""

    target = float(target_fraction)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ContractError("target fraction must lie in [0,1]")
    array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ContractError("threshold matching needs a nonempty finite population")
    rows = []
    for threshold_value in threshold_candidates(array.tolist()):
        fraction = selected_fraction(array, threshold_value)
        threshold = (
            "positive_infinity"
            if threshold_value == math.inf
            else ("negative_infinity" if threshold_value == -math.inf else threshold_value)
        )
        rows.append(
            {
                "threshold": threshold,
                "threshold_sort_value": threshold_value,
                "selected_fraction": fraction,
                "absolute_fraction_error": abs(fraction - target),
            }
        )
    best = min(rows, key=lambda row: (row["absolute_fraction_error"], -row["threshold_sort_value"]))
    rows_for_payload = [{key: value for key, value in row.items() if key != "threshold_sort_value"} for row in rows]
    payload = {
        "baseline_id": baseline_id,
        "population_size": int(array.size),
        "target_fraction": target,
        "threshold": best["threshold"],
        "selected_fraction": best["selected_fraction"],
        "absolute_fraction_error": best["absolute_fraction_error"],
        "selection_rule": "minimum_absolute_fraction_error_then_higher_threshold",
        "threshold_candidates": rows_for_payload,
    }
    return {**payload, "calibration_id": domain_id("csvl-v1/baseline-threshold", payload)}


def choose_spatial_baseline(calibration_scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose strongest spatial baseline by boundary F1, IoU, then R031<R032<R033."""

    if {str(item["baseline_id"]) for item in calibration_scores} != set(_BASELINE_ORDER):
        raise ContractError("spatial baseline selection requires exactly R031, R032, and R033")
    parsed = []
    for item in calibration_scores:
        baseline_id = str(item["baseline_id"])
        boundary = float(item["boundary_f1_4px"])
        region = float(item["region_iou"])
        if not (math.isfinite(boundary) and math.isfinite(region)):
            raise NonFiniteError("baseline selection metric is nonfinite")
        parsed.append({**dict(item), "baseline_id": baseline_id, "boundary_f1_4px": boundary, "region_iou": region})
    winner = max(
        parsed,
        key=lambda item: (
            item["boundary_f1_4px"],
            item["region_iou"],
            -_BASELINE_ORDER[item["baseline_id"]],
        ),
    )
    payload = {
        "winner": winner["baseline_id"],
        "winner_threshold": winner.get("threshold"),
        "ranking": sorted(
            parsed,
            key=lambda item: (
                -item["boundary_f1_4px"],
                -item["region_iou"],
                _BASELINE_ORDER[item["baseline_id"]],
            ),
        ),
        "selection_rule": "maximum_boundary_f1_4px_then_region_iou_then_R031_R032_R033",
    }
    return {**payload, "selection_id": domain_id("csvl-v1/spatial-baseline-selection", payload)}


def calibrate_r031_mt(
    aperture_score_arrays: Sequence[np.ndarray],
    csvl_accepted_arrays: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Freeze the global R031-MT threshold on identical valid aperture pixels."""

    if len(aperture_score_arrays) != len(csvl_accepted_arrays) or not aperture_score_arrays:
        raise ContractError("R031-MT calibration populations must be nonempty and aligned")
    scores: list[np.ndarray] = []
    accepted = 0
    valid = 0
    for index, (raw_score, raw_csvl) in enumerate(zip(aperture_score_arrays, csvl_accepted_arrays, strict=True)):
        score = np.asarray(raw_score, dtype=np.float64)
        csvl = np.asarray(raw_csvl)
        if score.shape != csvl.shape or score.size == 0:
            raise ContractError(f"R031-MT aperture {index} shape/population mismatch")
        mask = np.isfinite(score)
        if not mask.any():
            raise ContractError(f"R031-MT aperture {index} has no valid score pixels")
        if csvl.dtype != np.bool_ and not np.isin(csvl, [0, 1]).all():
            raise ContractError("CSVL accepted aperture must be boolean")
        scores.append(score[mask])
        accepted += int(np.count_nonzero(csvl[mask]))
        valid += int(np.count_nonzero(mask))
    target_fraction = accepted / valid
    result = match_selected_fraction(np.concatenate(scores), target_fraction, baseline_id="R031-MT-support-v1")
    return {
        **result,
        "aperture_count": len(scores),
        "valid_pixel_count": valid,
        "csvl_accepted_pixel_count": accepted,
        "state_rule": "visible_if_selected_fraction_greater_or_equal_0.5_else_occluded",
    }


def r031_mt_state(score: np.ndarray, aperture: np.ndarray, threshold: float | str) -> str:
    if threshold == "positive_infinity":
        threshold = math.inf
    elif threshold == "negative_infinity":
        threshold = -math.inf
    score_array = np.asarray(score, dtype=np.float64)
    aperture_array = np.asarray(aperture, dtype=bool)
    if score_array.shape != aperture_array.shape:
        raise ContractError("R031-MT score/aperture shape mismatch")
    valid = aperture_array & np.isfinite(score_array)
    if not valid.any():
        return "uncertain"
    fraction = float(np.count_nonzero(score_array[valid] >= threshold) / np.count_nonzero(valid))
    return "visible" if fraction >= 0.5 else "occluded"


def validate_baseline_registry(
    registry: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    verify_present_files: bool = False,
) -> dict[str, Any]:
    """Validate immutable historical provenance, optionally hashing present files."""

    if registry.get("schema_version") != BASELINE_SCHEMA:
        raise SchemaError("wrong R031 baseline registry schema")
    required = {
        "repository_commit",
        "source_code",
        "commands",
        "historical_manifests",
        "baselines",
        "threshold_policy",
        "r031_mt",
    }
    missing = required - set(registry)
    if missing:
        raise SchemaError(f"baseline registry missing keys: {sorted(missing)}")
    if registry["repository_commit"] != "94cd67df53cfc487989c71dc16a60fe853f53550":
        raise ProvenanceError("baseline repository commit changed")
    if registry["source_code"].get("sha256") != "3e8e19bb7fadafef25df1b6df0f75fc3d5d2fd8d245c9f6616488137466db933":
        raise ProvenanceError("baseline source code hash changed")
    if [item.get("baseline_id") for item in registry["baselines"]] != ["R031", "R032", "R033"]:
        raise ContractError("baseline IDs/order changed")
    expected_commands = {
        "prepare": "62c3e0b449a10430582c4920ef1d3c4a0bb22c56d2b92875afc0079cdb7425d4",
        "infer": "3f82fee7875ac623570b85545783413f01b9b11af5bfc82f2d46944bc7c64a2f",
        "R031": "a06d1c00c2c90a7bcab4fedf8827c31664b6f2593a065c3175963e6c9d622ed3",
        "R032": "fcd502c23a73c0280a5b52c19ba505e4e920fdde74d5dec07f6ed5d2fb237195",
        "R033": "d556d84dbe3c94e6d5f20119c0c788782b6f704f1f420bfa02a49f34a0681643",
    }
    if {key: value.get("sha256") for key, value in registry["commands"].items()} != expected_commands:
        raise ProvenanceError("baseline command hashes changed")
    expected_manifests = {
        "frame": "f4522dcf5ec18ea4212c1f19c08ec51f3e3e4cd881384f77a8481a006ab4d19e",
        "depth": "5a3437a5e78538f996e855c77fa1ffcf538fc2b7588022770aa7671bd06ccc63",
        "R031": "613db3f49650eb88b329858ad2442cb1f0b0192893b4be7113b8827cb07f24b3",
        "R032": "9f4dedbf0d1a83ed98417fec871b6afac39053f831aaa433119244f9c04fbdec",
        "R033": "bc7f33d5e396d0591d3ade9d2992912df88c6714821c51a886eb39000898c41e",
    }
    if {key: value.get("sha256") for key, value in registry["historical_manifests"].items()} != expected_manifests:
        raise ProvenanceError("historical baseline manifest hashes changed")
    verified = {}
    if verify_present_files:
        if repo_root is None:
            raise ContractError("repo_root is required to verify present files")
        root = Path(repo_root)
        source_path = root / registry["source_code"]["path"]
        if not source_path.is_file() or sha256_file(source_path) != registry["source_code"]["sha256"]:
            raise ProvenanceError(f"baseline source code hash mismatch: {source_path}")
        verified["source_code"] = registry["source_code"]["sha256"]
        for section in ("commands", "historical_manifests"):
            for key, value in registry[section].items():
                path = root / value["path"]
                if not path.is_file():
                    raise ProvenanceError(f"missing historical baseline file: {path}")
                actual = sha256_file(path)
                if actual != value["sha256"]:
                    raise ProvenanceError(f"historical baseline hash mismatch: {path}")
                verified[f"{section}.{key}"] = actual
    return {"valid": True, "verified_present_files": verified}


__all__ = [
    "BASELINE_SCHEMA",
    "calibrate_r031_mt",
    "choose_spatial_baseline",
    "match_selected_fraction",
    "r031_mt_state",
    "selected_fraction",
    "threshold_candidates",
    "validate_baseline_registry",
]
