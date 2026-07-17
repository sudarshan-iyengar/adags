"""Stable duplicate aggregation and source-node construction."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .canonical import array_semantic_ref, domain_id


def lower_median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not all(math.isfinite(value) for value in ordered):
        raise ValueError("median requires at least one finite value")
    return ordered[(len(ordered) - 1) // 2]


def weighted_median_stable(
    values: Iterable[float], weights: Iterable[float], stable_ids: Iterable[str]
) -> float:
    triples = [(float(v), str(i), float(w)) for v, w, i in zip(values, weights, stable_ids, strict=True)]
    if not triples:
        raise ValueError("weighted median requires at least one sample")
    if any(not math.isfinite(v) or not math.isfinite(w) or w < 0 for v, _, w in triples):
        raise ValueError("weighted median values/weights must be finite and weights nonnegative")
    triples.sort(key=lambda item: (item[0], item[1]))
    total = sum(item[2] for item in triples)
    if total == 0:
        return triples[(len(triples) - 1) // 2][0]
    threshold = total / 2.0
    cumulative = 0.0
    for value, _, weight in triples:
        cumulative += weight
        if cumulative >= threshold:
            return value
    raise AssertionError("unreachable weighted median state")


def duplicate_relative_mad(values: Iterable[float], center: float | None = None) -> float:
    vals = [float(value) for value in values]
    if not vals or not all(math.isfinite(value) for value in vals):
        raise ValueError("duplicate MAD requires finite samples")
    med = lower_median(vals) if center is None else float(center)
    if not math.isfinite(med):
        raise ValueError("duplicate MAD center must be finite")
    mad = lower_median(abs(value - med) for value in vals)
    denominator = abs(med)
    if denominator == 0.0:
        return 0.0 if mad == 0.0 else math.inf
    return mad / denominator


def node_geometry(
    *,
    K: np.ndarray,
    w2c: np.ndarray,
    x: float,
    y: float,
    optical_z: float,
    duplicate_depths: Iterable[float],
) -> dict[str, Any]:
    """Unproject optical z and transport the pinned (u,v,z) covariance."""

    intrinsic = np.asarray(K, dtype=np.float64)
    extrinsic = np.asarray(w2c, dtype=np.float64)
    z = float(optical_z)
    if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError("node geometry requires 3x3 K and 4x4 w2c")
    if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(extrinsic)):
        raise ValueError("node geometry calibration must be finite")
    if not all(math.isfinite(value) for value in (float(x), float(y), z)) or z <= 0:
        raise ValueError("node geometry requires finite coordinates and positive optical z")
    if not np.allclose(extrinsic[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1e-12):
        raise ValueError("w2c must be an affine homogeneous transform")
    try:
        k_inverse = np.linalg.inv(intrinsic)
    except np.linalg.LinAlgError as exc:
        raise ValueError("K must be invertible") from exc
    rotation_cw = extrinsic[:3, :3]
    if not np.allclose(rotation_cw @ rotation_cw.T, np.eye(3), rtol=0.0, atol=1e-9):
        raise ValueError("w2c rotation must be orthonormal")

    ray = k_inverse @ np.array([float(x), float(y), 1.0], dtype=np.float64)
    camera_point = z * ray
    rotation_wc = rotation_cw.T
    world_point = rotation_wc @ (camera_point - extrinsic[:3, 3])

    duplicate_values = [float(value) for value in duplicate_depths]
    if not duplicate_values or not all(math.isfinite(value) for value in duplicate_values):
        raise ValueError("duplicate depths must be a nonempty finite sequence")
    depth_mad = lower_median(abs(value - z) for value in duplicate_values)
    sigma_z = max(0.01 * abs(z), 1.4826 * depth_mad)
    jacobian_camera = np.column_stack((z * k_inverse[:, 0], z * k_inverse[:, 1], ray))
    covariance_uvwz = np.diag([0.5**2, 0.5**2, sigma_z**2])
    covariance_camera = jacobian_camera @ covariance_uvwz @ jacobian_camera.T
    covariance_world = rotation_wc @ covariance_camera @ rotation_wc.T
    return {
        "world_point": world_point,
        "covariance": covariance_world,
        "sigma_z": sigma_z,
        "camera_point": camera_point,
    }


def _covariance_ref(covariance: np.ndarray) -> Mapping[str, Any]:
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
        raise ValueError("source-node covariance must be finite 3x3")
    if not np.allclose(cov, cov.T, rtol=0.0, atol=1e-12):
        raise ValueError("source-node covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(cov))) < -1e-12:
        raise ValueError("source-node covariance must be positive semidefinite")
    return array_semantic_ref(cov, semantic_key="source_node_covariance")


def _base_payload(samples: list[Mapping[str, Any]], aggregate: Mapping[str, Any]) -> dict[str, Any]:
    if not samples:
        raise ValueError("source node requires at least one group sample")
    required = ("scene", "frame", "time", "source_camera", "scored_target", "y", "x")
    missing = [key for key in required if key not in aggregate]
    if missing:
        raise ValueError(f"missing aggregate identity fields: {missing}")
    sample_ids = sorted(str(sample["sample_id"]) for sample in samples)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("group-sample IDs must be unique")
    if any(not isinstance(sample.get("physical_ancestry"), (list, tuple, set)) or not sample["physical_ancestry"] for sample in samples):
        raise ValueError("every group sample needs explicit nonempty physical ancestry")
    ancestry = sorted({str(camera) for sample in samples for camera in sample["physical_ancestry"]})
    source_camera = str(aggregate["source_camera"])
    scored_target = str(aggregate["scored_target"])
    if any(source_camera not in {str(value) for value in sample["physical_ancestry"]} for sample in samples):
        raise ValueError("every group sample must include its source camera in physical ancestry")
    if scored_target in ancestry:
        raise ValueError("scored target leaked into source-node ancestry")
    depths = [float(sample["optical_z"]) for sample in samples]
    confidences = [float(sample["confidence"]) for sample in samples]
    if any(not math.isfinite(value) or value < 0 for value in confidences):
        raise ValueError("confidence must be finite and nonnegative")
    center = weighted_median_stable(depths, confidences, sample_ids)
    confidence = sum(confidences) / len(confidences)
    covariance_ref = _covariance_ref(np.asarray(aggregate["covariance"], dtype=np.float64))
    return {
        **{key: aggregate[key] for key in required},
        "contributing_group_sample_ids": sample_ids,
        "physical_ancestry": ancestry,
        "optical_z": center,
        "confidence": confidence,
        "covariance": covariance_ref,
    }


def build_source_node(samples: Iterable[Mapping[str, Any]], aggregate: Mapping[str, Any]) -> dict[str, Any]:
    sample_list = list(samples)
    payload = _base_payload(sample_list, aggregate)
    node_id = domain_id("csvl-v1/source-node", payload)
    return {"node_id": node_id, **payload, "retained": True}


def build_rejected_source_node(
    samples: Iterable[Mapping[str, Any]], aggregate: Mapping[str, Any], *, reason: str
) -> dict[str, Any]:
    sample_list = list(samples)
    payload = _base_payload(sample_list, aggregate)
    identity = {**payload, "retained": False, "reason": str(reason)}
    node_id = domain_id("csvl-v1/source-node-rejected", identity)
    return {"node_id": node_id, **identity}


def aggregate_group_samples(
    samples: Iterable[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    *,
    relative_mad_maximum: float = 0.05,
) -> dict[str, Any]:
    """Aggregate same-camera/pixel group samples or emit a rejected record."""

    sample_list = list(samples)
    if not sample_list:
        raise ValueError("cannot aggregate zero samples")
    if not math.isfinite(relative_mad_maximum) or relative_mad_maximum <= 0:
        raise ValueError("relative MAD maximum must be finite and positive")
    identity = ("scene", "frame", "time", "source_camera", "scored_target", "y", "x")
    for key in identity:
        if key not in aggregate:
            raise ValueError(f"aggregate is missing identity field {key}")
        expected = aggregate[key]
        if any(sample.get(key) != expected for sample in sample_list):
            raise ValueError(f"group sample identity mismatch for {key}")
    depths = [float(sample["optical_z"]) for sample in sample_list]
    center = weighted_median_stable(
        depths,
        [float(sample["confidence"]) for sample in sample_list],
        [str(sample["sample_id"]) for sample in sample_list],
    )
    rel_mad = duplicate_relative_mad(depths, center)
    record = (
        build_rejected_source_node(sample_list, aggregate, reason="duplicate_relative_mad")
        if rel_mad > relative_mad_maximum
        else build_source_node(sample_list, aggregate)
    )
    record["duplicate_relative_mad"] = rel_mad
    record["duplicate_risk"] = (
        min(1.0, rel_mad / relative_mad_maximum) if math.isfinite(rel_mad) else 1.0
    )
    return record


__all__ = [
    "aggregate_group_samples",
    "build_rejected_source_node",
    "build_source_node",
    "duplicate_relative_mad",
    "lower_median",
    "node_geometry",
    "weighted_median_stable",
]
