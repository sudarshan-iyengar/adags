"""Float64 camera geometry for the CSVL OpenCV column-vector convention."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .errors import CameraConventionError, NonFiniteError


_YZ_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def _finite_array(value: Any, shape: tuple[int, ...], *, what: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise CameraConventionError(f"{what} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise NonFiniteError(f"{what} contains NaN or infinity")
    return result


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a finite positive-focal-length 3x3 intrinsic matrix."""

    values = np.asarray([fx, fy, cx, cy], dtype=np.float64)
    if not np.isfinite(values).all() or fx <= 0 or fy <= 0:
        raise CameraConventionError("intrinsics require finite positive focal lengths")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def validate_calibration(
    K: Any,
    w2c: Any,
    width: int,
    height: int,
    *,
    distortion: Any = None,
    rolling_shutter: bool = False,
    timestamps: Sequence[float] | None = None,
    timestamp_tolerance_seconds: float = 1e-6,
) -> None:
    """Fail closed on unsupported or malformed calibrated-camera metadata."""

    intrinsic = _finite_array(K, (3, 3), what="K")
    extrinsic = _finite_array(w2c, (4, 4), what="w2c")
    if (
        isinstance(width, bool) or isinstance(height, bool)
        or not isinstance(width, (int, np.integer))
        or not isinstance(height, (int, np.integer))
        or width <= 0 or height <= 0
    ):
        raise CameraConventionError("image dimensions must be positive integers")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise CameraConventionError("focal lengths must be positive")
    if not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise CameraConventionError("K has an unsupported projective last row")
    if not np.allclose(extrinsic[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise CameraConventionError("w2c has an invalid homogeneous last row")
    rotation = extrinsic[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8, rtol=0.0):
        raise CameraConventionError("w2c rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
        raise CameraConventionError("w2c rotation is not right handed")
    if distortion is not None:
        coefficients = np.asarray(distortion, dtype=np.float64)
        if not np.isfinite(coefficients).all():
            raise NonFiniteError("distortion contains NaN or infinity")
        if np.any(coefficients != 0.0):
            raise CameraConventionError("nonzero distortion is unsupported")
    if rolling_shutter is not False:
        raise CameraConventionError("rolling shutter is unsupported")
    if not math.isfinite(float(timestamp_tolerance_seconds)) or timestamp_tolerance_seconds < 0:
        raise CameraConventionError("timestamp tolerance must be finite and nonnegative")
    if timestamps is not None:
        times = np.asarray(tuple(timestamps), dtype=np.float64)
        if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all():
            raise CameraConventionError("timestamps must be a nonempty finite vector")
        if float(np.max(times) - np.min(times)) > timestamp_tolerance_seconds:
            raise CameraConventionError("camera timestamps are not synchronized")


def opengl_c2w_to_opencv_w2c(c2w_opengl: Any) -> np.ndarray:
    """Flip OpenGL camera Y/Z columns and invert exactly once."""

    c2w = _finite_array(c2w_opengl, (4, 4), what="OpenGL c2w")
    if not np.allclose(c2w[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise CameraConventionError("OpenGL c2w has an invalid homogeneous last row")
    rotation = c2w[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8, rtol=0.0):
        raise CameraConventionError("OpenGL c2w rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
        raise CameraConventionError("OpenGL c2w rotation is not right handed")
    c2w_opencv = c2w @ _YZ_FLIP
    try:
        w2c = np.linalg.inv(c2w_opencv)
    except np.linalg.LinAlgError as exc:
        raise CameraConventionError("camera transform is singular") from exc
    return np.asarray(w2c, dtype=np.float64)


def camera_center(w2c: Any) -> np.ndarray:
    """Return the world-space camera center."""

    extrinsic = _finite_array(w2c, (4, 4), what="w2c")
    rotation = extrinsic[:3, :3]
    return -(rotation.T @ extrinsic[:3, 3])


def project_world(K: Any, w2c: Any, points_world: Any) -> tuple[np.ndarray, np.ndarray | float]:
    """Project world points, rejecting nonpositive camera-z observations."""

    intrinsic = _finite_array(K, (3, 3), what="K")
    extrinsic = _finite_array(w2c, (4, 4), what="w2c")
    points = np.asarray(points_world, dtype=np.float64)
    single = points.shape == (3,)
    if single:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise CameraConventionError("world points must be finite with shape (3,) or (N,3)")
    camera_points = (extrinsic[:3, :3] @ points.T).T + extrinsic[:3, 3]
    z = camera_points[:, 2]
    if np.any(z <= 0.0):
        raise CameraConventionError("projection encountered nonpositive camera z")
    homogeneous = (intrinsic @ camera_points.T).T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    if single:
        return pixels[0], float(z[0])
    return pixels, z


def unproject_optical_z(K: Any, w2c: Any, pixels: Any, optical_z: Any) -> np.ndarray:
    """Unproject native integer-center pixels using optical-axis camera z."""

    intrinsic = _finite_array(K, (3, 3), what="K")
    extrinsic = _finite_array(w2c, (4, 4), what="w2c")
    uv = np.asarray(pixels, dtype=np.float64)
    single = uv.shape == (2,)
    if single:
        uv = uv[None, :]
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise CameraConventionError("pixels must be finite with shape (2,) or (N,2)")
    z = np.asarray(optical_z, dtype=np.float64)
    if z.ndim == 0:
        z = np.full(uv.shape[0], float(z))
    if z.shape != (uv.shape[0],) or not np.isfinite(z).all() or np.any(z <= 0.0):
        raise CameraConventionError("optical z must be finite, positive, and pixel-aligned")
    try:
        inverse_k = np.linalg.inv(intrinsic)
    except np.linalg.LinAlgError as exc:
        raise CameraConventionError("K is singular") from exc
    rays = (inverse_k @ np.column_stack([uv, np.ones(len(uv))]).T).T
    camera_points = rays * z[:, None]
    world_points = (extrinsic[:3, :3].T @ (camera_points - extrinsic[:3, 3]).T).T
    return world_points[0] if single else world_points


def processed_image_transform(
    native_width: int,
    native_height: int,
    processed_width: int,
    processed_height: int,
) -> np.ndarray:
    """Return native-to-processed pixel transform for align_corners=False."""

    sizes = (native_width, native_height, processed_width, processed_height)
    if any(isinstance(value, bool) or int(value) != value or value <= 0 for value in sizes):
        raise CameraConventionError("native and processed image sizes must be positive integers")
    sx = float(processed_width) / float(native_width)
    sy = float(processed_height) / float(native_height)
    return np.array(
        [[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def transport_pixel_depth_covariance(
    K: Any,
    w2c: Any,
    pixel: Any,
    optical_z: float,
    sigma_z: float,
    *,
    sigma_u: float = 0.5,
    sigma_v: float = 0.5,
) -> np.ndarray:
    """Transport diag(pixel, depth) uncertainty to world coordinates."""

    intrinsic = _finite_array(K, (3, 3), what="K")
    extrinsic = _finite_array(w2c, (4, 4), what="w2c")
    uv = np.asarray(pixel, dtype=np.float64)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise CameraConventionError("pixel must be a finite length-two vector")
    sigmas = np.asarray([sigma_u, sigma_v, sigma_z], dtype=np.float64)
    if not np.isfinite(sigmas).all() or np.any(sigmas < 0.0) or optical_z <= 0:
        raise CameraConventionError("uncertainty and optical z are invalid")
    inv_k = np.linalg.inv(intrinsic)
    ray = inv_k @ np.array([uv[0], uv[1], 1.0])
    camera_jacobian = np.column_stack(
        [optical_z * inv_k[:, 0], optical_z * inv_k[:, 1], ray]
    )
    world_jacobian = extrinsic[:3, :3].T @ camera_jacobian
    covariance = world_jacobian @ np.diag(sigmas * sigmas) @ world_jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    if not np.isfinite(covariance).all():
        raise NonFiniteError("transported covariance is nonfinite")
    return covariance


__all__ = [
    "camera_center",
    "intrinsics_matrix",
    "opengl_c2w_to_opencv_w2c",
    "processed_image_transform",
    "project_world",
    "transport_pixel_depth_covariance",
    "unproject_optical_z",
    "validate_calibration",
]
