"""Read-only adapter for the pinned calibrated Depth Anything 3 API."""

from __future__ import annotations

import importlib
import math
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_write_json, write_canonical_array
from .camera import project_world, unproject_optical_z
from .canonical import array_semantic_ref, sha256_file
from .errors import ContractError, NonFiniteError, ProvenanceError, SchemaError


DA3_CHECKOUT_COMMIT = "41736238f5bced4debf3f2a12375d2466874866d"
DA3_MODEL_ID = "DA3NESTED-GIANT-LARGE-1.1"
DA3_CODE_PINS = {
    "src/depth_anything_3/api.py": "042ae8f6bfd1e5610100a585ebb00f8cbc3320f9aa9e03a07530fd4deda8d2d8",
    "src/depth_anything_3/utils/io/input_processor.py": "cdebb6021f5f1001d3aa56261c224a80b9d8431efedfb040537c58a5509410c1",
    "src/depth_anything_3/utils/geometry.py": "3d67391f2eeed57ee5d07387947dea7a5d99fc0ce0a63c6890485183c851c29b",
}
DA3_CONFIG_SHA256 = "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468"
DA3_WEIGHT_EXPECTED_BYTES = 6_759_558_100
INFERENCE_ARGUMENTS = {
    "align_to_input_ext_scale": True,
    "infer_gs": False,
    "use_ray_pose": False,
    "ref_view_strategy": "saddle_balanced",
    "process_res": 504,
    "process_res_method": "upper_bound_resize",
}


def verify_da3_checkout(checkout: str | Path) -> Mapping[str, Any]:
    """Verify the read-only checkout commit and all lightweight code pins."""

    root = Path(checkout).resolve()
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProvenanceError(f"cannot inspect DA3 checkout: {root}") from exc
    if head != DA3_CHECKOUT_COMMIT:
        raise ProvenanceError(f"DA3 checkout mismatch: {head}")
    hashes: dict[str, str] = {}
    for relative, expected in DA3_CODE_PINS.items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ProvenanceError(f"DA3 code pin mismatch: {relative}")
        hashes[relative] = actual
    return {"checkout": str(root), "commit": head, "file_sha256": hashes}


def verify_model_authority(
    model_dir: str | Path,
    *,
    expected_weight_sha256: str | None = None,
    hash_weights: bool = False,
) -> Mapping[str, Any]:
    """Verify model config and size; hash the 6.76 GB weights only when admitted."""

    root = Path(model_dir).resolve()
    config_path = root / "config.json"
    weight_path = root / "model.safetensors"
    if sha256_file(config_path) != DA3_CONFIG_SHA256:
        raise ProvenanceError("DA3 model config SHA-256 mismatch")
    if not weight_path.is_file() or weight_path.stat().st_size != DA3_WEIGHT_EXPECTED_BYTES:
        raise ProvenanceError("DA3 weight file size mismatch")
    report: dict[str, Any] = {
        "model_dir": str(root),
        "model_id": DA3_MODEL_ID,
        "config_sha256": DA3_CONFIG_SHA256,
        "weight_bytes": weight_path.stat().st_size,
        "weight_sha256": None,
    }
    if hash_weights or expected_weight_sha256 is not None:
        actual = sha256_file(weight_path)
        if expected_weight_sha256 is not None and actual != expected_weight_sha256:
            raise ProvenanceError("DA3 weight SHA-256 mismatch")
        report["weight_sha256"] = actual
    return report


def load_da3(
    checkout: str | Path,
    model_dir: str | Path,
    *,
    device: str = "cuda",
) -> Any:
    """Load the pinned model locally after authority verification."""

    verify_da3_checkout(checkout)
    verify_model_authority(model_dir)
    source = str((Path(checkout).resolve() / "src"))
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module("depth_anything_3.api")
    model = module.DepthAnything3.from_pretrained(str(Path(model_dir).resolve()))
    import torch

    model = model.to(device=torch.device(device))
    model.eval()
    return model


def _prediction_field(prediction: Any, field: str) -> Any:
    if isinstance(prediction, Mapping):
        return prediction.get(field)
    return getattr(prediction, field, None)


def validate_prediction(prediction: Any, expected_views: int) -> Mapping[str, Any]:
    """Normalize and validate DA3 prediction arrays without changing scale."""

    depth = np.asarray(_prediction_field(prediction, "depth"))
    if depth.ndim != 3 or depth.shape[0] != expected_views:
        raise ContractError("DA3 depth has unexpected shape or view count")
    if not np.isfinite(depth).all():
        raise NonFiniteError("DA3 depth contains NaN or infinity")
    confidence_raw = _prediction_field(prediction, "conf")
    if confidence_raw is None:
        raise ContractError("DA3 did not return confidence required by CSVL")
    confidence = np.asarray(confidence_raw)
    if confidence.shape != depth.shape or not np.isfinite(confidence).all():
        raise ContractError("DA3 confidence has invalid shape or values")
    if np.any(confidence < 0):
        raise ContractError("DA3 confidence must be nonnegative")
    intrinsics_raw = _prediction_field(prediction, "intrinsics")
    intrinsics = None if intrinsics_raw is None else np.asarray(intrinsics_raw)
    if intrinsics is None or intrinsics.shape != (expected_views, 3, 3) or not np.isfinite(intrinsics).all():
        raise ContractError("DA3 did not return finite processed intrinsics")
    extrinsics_raw = _prediction_field(prediction, "extrinsics")
    if extrinsics_raw is None:
        raise ContractError("DA3 did not return aligned extrinsics")
    extrinsics = np.asarray(extrinsics_raw)
    if extrinsics.shape == (expected_views, 3, 4):
        homogeneous = np.repeat(np.eye(4, dtype=extrinsics.dtype)[None, ...], expected_views, axis=0)
        homogeneous[:, :3, :] = extrinsics
        extrinsics = homogeneous
    if extrinsics.shape != (expected_views, 4, 4) or not np.isfinite(extrinsics).all():
        raise ContractError("DA3 returned invalid extrinsics")
    expected_last_row = np.repeat(
        np.array([[[0.0, 0.0, 0.0, 1.0]]]), expected_views, axis=0
    )
    if not np.array_equal(extrinsics[:, 3:4, :], expected_last_row):
        raise ContractError("DA3 extrinsics are not homogeneous w2c matrices")
    processed_raw = _prediction_field(prediction, "processed_images")
    if processed_raw is None:
        raise ContractError("DA3 did not return processed images required for provenance")
    processed = np.asarray(processed_raw)
    if (
        processed.ndim != 4 or processed.shape[0] != expected_views
        or processed.shape[1:3] != depth.shape[1:3] or not np.isfinite(processed).all()
    ):
        raise ContractError("DA3 returned invalid processed images")
    scale = _prediction_field(prediction, "scale_factor")
    if scale is not None and (not math.isfinite(float(scale)) or float(scale) <= 0):
        raise ContractError("DA3 returned invalid scale_factor")
    return {
        "depth": depth,
        "confidence": confidence,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "processed_images": processed,
        "is_metric": _prediction_field(prediction, "is_metric"),
        "scale_factor": scale,
    }


def run_group(
    model: Any,
    images: Sequence[Any],
    extrinsics_w2c: Any,
    intrinsics: Any,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Run one complete calibrated group with every scientific flag explicit."""

    if overrides:
        unknown = set(overrides) - set(INFERENCE_ARGUMENTS)
        changed = {key for key, value in overrides.items() if INFERENCE_ARGUMENTS.get(key) != value}
        if unknown or changed:
            raise ContractError(f"runtime DA3 override rejected: {sorted(unknown | changed)}")
    if len(images) != 6:
        raise ContractError("DA3 group must contain exactly six ordered images")
    extrinsics = np.asarray(extrinsics_w2c, dtype=np.float64)
    intrinsic_array = np.asarray(intrinsics, dtype=np.float64)
    if extrinsics.shape != (6, 4, 4) or intrinsic_array.shape != (6, 3, 3):
        raise ContractError("DA3 calibrated group matrices have invalid shape")
    if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsic_array).all():
        raise NonFiniteError("DA3 calibrated group matrices are nonfinite")
    prediction = model.inference(
        list(images),
        extrinsics=extrinsics,
        intrinsics=intrinsic_array,
        align_to_input_ext_scale=True,
        infer_gs=False,
        use_ray_pose=False,
        ref_view_strategy="saddle_balanced",
        process_res=504,
        process_res_method="upper_bound_resize",
    )
    return validate_prediction(prediction, 6)


def run_analytic_conformance() -> Mapping[str, float]:
    """Run the pinned camera-z and translated/yawed round-trip fixture."""

    K = np.array([[50.0, 0.0, 31.5], [0.0, 52.0, 23.5], [0.0, 0.0, 1.0]])
    first = np.eye(4)
    pixels = np.array([[31.5, 23.5], [0.0, 0.0], [63.0, 47.0]])
    world = unproject_optical_z(K, first, pixels, np.full(3, 2.0))
    recovered, recovered_z = project_world(K, first, world)
    pixel_error = float(np.max(np.abs(recovered - pixels)))
    z_error = float(np.max(np.abs(recovered_z - 2.0)))
    angle = math.radians(23.0)
    rotation = np.array(
        [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0],
         [-math.sin(angle), 0.0, math.cos(angle)]]
    )
    second = np.eye(4)
    second[:3, :3] = rotation
    second[:3, 3] = [0.4, -0.2, 0.1]
    points = unproject_optical_z(K, second, pixels, np.full(3, 2.0))
    second_pixels, second_z = project_world(K, second, points)
    second_error = max(
        float(np.max(np.abs(second_pixels - pixels))),
        float(np.max(np.abs(second_z - 2.0))),
    )
    if pixel_error > 1e-10 or z_error > 1e-12 or second_error > 1e-9:
        raise ContractError("analytic DA3 camera conformance failed")
    return {
        "center_corner_pixel_error_maximum": pixel_error,
        "optical_z_error_maximum": z_error,
        "translated_yawed_roundtrip_error_maximum": second_error,
    }


def _relative_mad(depths: np.ndarray) -> float:
    center = np.median(depths, axis=0)
    mad = np.median(np.abs(depths - center[None, ...]), axis=0)
    denominator = np.abs(center)
    ratio = np.full_like(mad, np.inf, dtype=np.float64)
    np.divide(mad, denominator, out=ratio, where=denominator != 0.0)
    ratio[(denominator == 0.0) & (mad == 0.0)] = 0.0
    return float(np.max(ratio))


def _delta_statistics(first: np.ndarray, second: np.ndarray, *, atol: float, rtol: float) -> Mapping[str, Any]:
    """Summarize deterministic-repeat deltas without deciding admission."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ContractError("repeatability arrays must have equal nonempty shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ContractError("repeatability arrays must be finite")
    absolute = np.abs(left - right)
    denominator = np.maximum(
        (np.abs(left) + np.abs(right)) / 2.0, np.finfo(np.float64).tiny
    )
    relative = absolute / denominator
    quantiles = (0.5, 0.9, 0.95, 0.99, 0.999)
    tolerance = atol + rtol * np.abs(right)
    return {
        "absolute_mean": float(np.mean(absolute)),
        "absolute_rmse": float(np.sqrt(np.mean(absolute * absolute))),
        "absolute_maximum": float(np.max(absolute)),
        "absolute_quantiles": {
            str(value): float(np.quantile(absolute, value)) for value in quantiles
        },
        "symmetric_relative_mean": float(np.mean(relative)),
        "symmetric_relative_maximum": float(np.max(relative)),
        "symmetric_relative_quantiles": {
            str(value): float(np.quantile(relative, value)) for value in quantiles
        },
        "within_registered_tolerance_fraction": float(np.mean(absolute <= tolerance)),
        "allclose_pass": bool(np.allclose(left, right, atol=atol, rtol=rtol)),
    }


def relative_mad_maximum(depths: np.ndarray) -> float:
    """Expose the frozen relative-MAD statistic for bounded diagnostics."""

    array = np.asarray(depths, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2 or not np.isfinite(array).all():
        raise ContractError("relative MAD requires at least two finite aligned arrays")
    return _relative_mad(array)


def _processed_k_corner_error(
    expected: np.ndarray,
    returned: np.ndarray,
    height: int,
    width: int,
) -> float:
    if expected.shape != returned.shape or expected.ndim != 3 or expected.shape[1:] != (3, 3):
        raise ContractError("independent processed-K authority has invalid shape")
    corners = np.array(
        [[0.0, 0.0, 1.0], [width - 1.0, 0.0, 1.0],
         [0.0, height - 1.0, 1.0], [width - 1.0, height - 1.0, 1.0]]
    )
    maximum = 0.0
    for expected_k, returned_k in zip(expected, returned):
        normalized = (np.linalg.inv(expected_k) @ corners.T).T
        projected = (returned_k @ normalized.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        maximum = max(maximum, float(np.max(np.linalg.norm(projected - corners[:, :2], axis=1))))
    return maximum


def processed_k_corner_error(
    expected: np.ndarray,
    returned: np.ndarray,
    height: int,
    width: int,
) -> float:
    """Expose the independent processed-K corner check for each fused group."""

    return _processed_k_corner_error(expected, returned, height, width)


def repetition_delta_report(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    expected_processed_intrinsics: Any,
    repeat_atol: float,
    repeat_rtol: float,
) -> Mapping[str, Any]:
    """Report real-repeat differences while preserving the registered gate."""

    if not math.isfinite(repeat_atol) or not math.isfinite(repeat_rtol):
        raise ContractError("repeatability tolerances must be finite")
    if repeat_atol < 0.0 or repeat_rtol < 0.0:
        raise ContractError("repeatability tolerances must be nonnegative")
    reports = []
    for index, prediction in enumerate((first, second)):
        depth = np.asarray(prediction["depth"])
        confidence = np.asarray(prediction["confidence"])
        finite_fraction = float(np.mean(np.isfinite(depth)))
        positive_fraction = float(np.mean(depth > 0.0))
        confidence_finite_fraction = float(np.mean(np.isfinite(confidence)))
        reports.append(
            {
                "repetition": index,
                "depth": array_semantic_ref(depth, f"da3-conformance-depth-r{index}"),
                "confidence": array_semantic_ref(
                    confidence, f"da3-conformance-confidence-r{index}"
                ),
                "processed_intrinsics": array_semantic_ref(
                    prediction["intrinsics"], f"da3-conformance-K-r{index}"
                ),
                "aligned_extrinsics": array_semantic_ref(
                    prediction["extrinsics"], f"da3-conformance-w2c-r{index}"
                ),
                "finite_depth_fraction": finite_fraction,
                "finite_confidence_fraction": confidence_finite_fraction,
                "positive_depth_fraction": positive_fraction,
            }
        )
        if finite_fraction < 0.99 or confidence_finite_fraction < 0.99 or positive_fraction < 0.95:
            raise ContractError("real DA3 conformance finite/positive rate failed")
    depth_delta = _delta_statistics(
        first["depth"], second["depth"], atol=repeat_atol, rtol=repeat_rtol
    )
    confidence_delta = _delta_statistics(
        first["confidence"], second["confidence"], atol=repeat_atol, rtol=repeat_rtol
    )
    duplicate_mad = _relative_mad(np.stack([first["depth"], second["depth"]]))
    expected_k = np.asarray(expected_processed_intrinsics, dtype=np.float64)
    corner_errors = [
        _processed_k_corner_error(
            expected_k,
            prediction["intrinsics"],
            prediction["depth"].shape[1],
            prediction["depth"].shape[2],
        )
        for prediction in (first, second)
    ]
    strict_pass = (
        depth_delta["allclose_pass"]
        and confidence_delta["allclose_pass"]
        and duplicate_mad <= 0.05
        and max(corner_errors) <= 0.5
    )
    return {
        "repetitions": reports,
        "repeat_atol": repeat_atol,
        "repeat_rtol": repeat_rtol,
        "depth_delta": depth_delta,
        "confidence_delta": confidence_delta,
        "duplicate_relative_mad_maximum": duplicate_mad,
        "processed_k_corner_error_maximum_pixels": max(corner_errors),
        "strict_repeatability_pass": bool(strict_pass),
        "hash_equality_required": False,
    }


def _evaluate_real_repetitions(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    expected_processed_intrinsics: Any,
    repeat_atol: float,
    repeat_rtol: float,
) -> Mapping[str, Any]:
    """Enforce A03 exactly while retaining the reusable diagnostic report."""

    report = dict(
        repetition_delta_report(
            first,
            second,
            expected_processed_intrinsics=expected_processed_intrinsics,
            repeat_atol=repeat_atol,
            repeat_rtol=repeat_rtol,
        )
    )
    if not report["depth_delta"]["allclose_pass"]:
        raise ContractError("real DA3 depth repetition exceeds numeric tolerance")
    if not report["confidence_delta"]["allclose_pass"]:
        raise ContractError("real DA3 confidence repetition exceeds numeric tolerance")
    if report["duplicate_relative_mad_maximum"] > 0.05:
        raise ContractError("real DA3 duplicate relative MAD exceeds 0.05")
    if report["processed_k_corner_error_maximum_pixels"] > 0.5:
        raise ContractError("real DA3 processed-K corner error exceeds 0.5 pixels")
    report["numerically_repeatable"] = True
    return report


def run_real_conformance(
    model: Any,
    images: Sequence[Any],
    extrinsics_w2c: Any,
    intrinsics: Any,
    *,
    expected_processed_intrinsics: Any,
    repeat_atol: float = 1e-5,
    repeat_rtol: float = 1e-5,
) -> Mapping[str, Any]:
    """Run one registered six-view group twice and report hashes plus numerics."""

    first = run_group(model, images, extrinsics_w2c, intrinsics)
    second = run_group(model, images, extrinsics_w2c, intrinsics)
    return _evaluate_real_repetitions(
        first,
        second,
        expected_processed_intrinsics=expected_processed_intrinsics,
        repeat_atol=repeat_atol,
        repeat_rtol=repeat_rtol,
    )


def run_two_group_conformance(
    model: Any,
    group_inputs: Sequence[Mapping[str, Any]],
    *,
    anchor_camera_id: str,
    repeat_atol: float = 1e-5,
    repeat_rtol: float = 1e-5,
) -> Mapping[str, Any]:
    """Run the admitted two-group fixture and compare the repeated anchor."""

    if len(group_inputs) != 2:
        raise ContractError("DA3 conformance requires exactly two groups")
    reports: list[Mapping[str, Any]] = []
    anchor_depths: list[np.ndarray] = []
    for ordinal, group in enumerate(group_inputs):
        members = tuple(group["member_camera_ids"])
        if len(members) != 6 or len(set(members)) != 6 or anchor_camera_id not in members:
            raise ContractError("invalid DA3 conformance group or missing anchor")
        first = run_group(model, group["images"], group["extrinsics_w2c"], group["intrinsics"])
        second = run_group(model, group["images"], group["extrinsics_w2c"], group["intrinsics"])
        report = _evaluate_real_repetitions(
            first,
            second,
            expected_processed_intrinsics=group["expected_processed_intrinsics"],
            repeat_atol=repeat_atol,
            repeat_rtol=repeat_rtol,
        )
        reports.append({"group_ordinal": ordinal, "member_camera_ids": list(members), **report})
        anchor_depths.append(np.asarray(first["depth"])[members.index(anchor_camera_id)])
    cross_group_mad = _relative_mad(np.stack(anchor_depths))
    if cross_group_mad > 0.05:
        raise ContractError("anchor duplicate relative MAD across groups exceeds 0.05")
    return {
        "anchor_camera_id": anchor_camera_id,
        "groups": reports,
        "anchor_cross_group_relative_mad_maximum": cross_group_mad,
        "numerically_repeatable": True,
    }

def produce_scene_sidecars(
    model: Any,
    group_requests: Iterable[Mapping[str, Any]],
    output_root: str | Path,
    *,
    authority: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Produce immutable per-group raw arrays; callers must run this on Slurm."""

    required_authority = {
        "checkout_commit", "model_id", "config_sha256", "weight_sha256",
        "code_sha256", "environment_sha256",
    }
    if set(authority) != required_authority:
        raise ProvenanceError("DA3 production authority has missing or unknown keys")
    if (
        authority["checkout_commit"] != DA3_CHECKOUT_COMMIT
        or authority["model_id"] != DA3_MODEL_ID
        or authority["config_sha256"] != DA3_CONFIG_SHA256
    ):
        raise ProvenanceError("DA3 production authority does not match frozen pins")
    for key in ("weight_sha256", "code_sha256", "environment_sha256"):
        value = authority[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ProvenanceError(f"invalid DA3 production authority hash: {key}")
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise ProvenanceError(f"production sidecar root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifests: list[Mapping[str, Any]] = []
    for ordinal, request in enumerate(group_requests):
        members = tuple(request["member_camera_ids"])
        if len(members) != 6 or len(set(members)) != 6:
            raise ContractError("production request is not a complete six-camera group")
        image_hashes = tuple(request.get("native_image_sha256", ()))
        native_sizes = tuple(request.get("native_image_sizes", ()))
        if len(image_hashes) != 6 or len(native_sizes) != 6:
            raise ProvenanceError("production request lacks six native image hashes/sizes")
        if any(
            not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in image_hashes
        ):
            raise ProvenanceError("production request has malformed native image SHA-256")
        input_w2c = np.asarray(request["extrinsics_w2c"], dtype=np.float64)
        input_k = np.asarray(request["intrinsics"], dtype=np.float64)
        prediction = run_group(model, request["images"], input_w2c, input_k)
        group_id = str(request["group_id"])
        shard = root / f"{ordinal:06d}-{group_id}"
        shard.mkdir(parents=False, exist_ok=False)
        depth_ref = write_canonical_array(
            shard / "depth.npy", prediction["depth"], f"{group_id}/depth", relative_to=root
        )
        confidence_ref = None
        if prediction["confidence"] is not None:
            confidence_ref = write_canonical_array(
                shard / "confidence.npy", prediction["confidence"],
                f"{group_id}/confidence", relative_to=root,
            )
        k_ref = write_canonical_array(
            shard / "processed_intrinsics.npy", prediction["intrinsics"],
            f"{group_id}/processed-intrinsics", relative_to=root,
        )
        processed_image_ref = write_canonical_array(
            shard / "processed_images.npy", prediction["processed_images"],
            f"{group_id}/processed-images", relative_to=root,
        )
        output_w2c_ref = write_canonical_array(
            shard / "aligned_extrinsics.npy", prediction["extrinsics"],
            f"{group_id}/aligned-extrinsics", relative_to=root,
        )
        manifest = {
            "schema_version": "phase9-da3-group-sidecar-v1",
            "group_id": group_id,
            "ordered_member_camera_ids": list(members),
            "physical_camera_ancestry": sorted(members),
            "frame": int(request["frame"]),
            "time": float(request["time"]),
            "native_image_sha256": list(image_hashes),
            "native_image_sizes": [list(size) for size in native_sizes],
            "input_intrinsics": array_semantic_ref(input_k, f"{group_id}/input-intrinsics"),
            "input_extrinsics_w2c": array_semantic_ref(input_w2c, f"{group_id}/input-extrinsics"),
            "inference_arguments": dict(INFERENCE_ARGUMENTS),
            "authority": dict(authority),
            "depth": depth_ref,
            "confidence": confidence_ref,
            "processed_intrinsics": k_ref,
            "processed_images": processed_image_ref,
            "aligned_extrinsics": output_w2c_ref,
            "is_metric": prediction["is_metric"],
            "scale_factor": prediction["scale_factor"],
        }
        atomic_write_json(shard / "manifest.json", manifest)
        manifests.append(manifest)
    atomic_write_json(root / "inventory.json", {"groups": manifests})
    return manifests


__all__ = [
    "DA3_CHECKOUT_COMMIT",
    "DA3_CODE_PINS",
    "DA3_CONFIG_SHA256",
    "DA3_MODEL_ID",
    "DA3_WEIGHT_EXPECTED_BYTES",
    "INFERENCE_ARGUMENTS",
    "load_da3",
    "processed_k_corner_error",
    "produce_scene_sidecars",
    "relative_mad_maximum",
    "repetition_delta_report",
    "run_analytic_conformance",
    "run_group",
    "run_real_conformance",
    "run_two_group_conformance",
    "validate_prediction",
    "verify_da3_checkout",
    "verify_model_authority",
]
