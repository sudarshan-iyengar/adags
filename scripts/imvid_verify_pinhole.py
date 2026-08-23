#!/usr/bin/env python3
"""Numeric reprojection gate for the ImViD OPENCV -> PINHOLE conversion.

The decisive, zero-GPU-computation check that the undistorted PINHOLE
raster the trainer would consume is geometrically consistent with the
sealed fixed-pose COLMAP reconstructions. It consumes ONLY artifacts that
already exist: no decode, no triangulation, no new data.

--------------------------------------------------------------------
WHAT IS COMPARED, AND WHY IT IS DECISIVE
--------------------------------------------------------------------

For every (3D point `X`, observing image `i`) pair in a sealed model:

  A   the ANALYTIC pinhole projection of `X`,
      `p_proj = pi(K_new @ (R_i @ X + t_i))`, with NO distortion term;

  B   the observed native-resolution 2D feature carried into the SAME
      raster, `cv2.undistortPoints(p_dist_obs, K, distCoeffs, P=K_new)`.

  residual = ||A - B||_2, in undistorted-PINHOLE-raster pixels.

`p_dist_obs` is read from the sealed `images.bin` POINTS2D block. The
sibling parser at `scripts/imvid_sparse_init.py:164-191` SKIPS those
24-byte records (line 187, `offset += n_points2d * 24`); this file
carries the same layout and READS them.

This should reproduce COLMAP's own reported residual through a
completely independent code path. That is what makes it decisive: a
pose-convention error (`R` vs `R^T`, world-to-camera vs
camera-to-world), a mis-ordered `distCoeffs`, or distortion dropped
entirely all produce residuals in the TENS to HUNDREDS of pixels, not a
mild degradation. `--mode self-test` MEASURES each of those corruptions
on a synthetic fixture and refuses to run if the instrument fails to
detect one.

--------------------------------------------------------------------
STATE THE RASTER, EVERY TIME
--------------------------------------------------------------------

The record's gate is **2 px at NATIVE 5312x2988**
(`imvid-baseline-freeze.md:322-323`, and the same 2 px bar at
`dataset-admission-matrix-2026-08-18.md:145`; met at 1.1953 / 1.1361 /
1.1808 px, `imvid-baseline-freeze.md:336-341`). Residuals scale with the
raster: at `scale = 0.5` the same geometry reads HALF.

**A 0.5-scale residual compared against a native 2 px gate passes
trivially and measures nothing.** So every statistic here is emitted in
BOTH rasters with its raster named in the string, the gate is applied to
the NATIVE-EQUIVALENT number only, and `_gate_decision` re-derives the
native number from the scaled one and refuses if the two disagree.

The relation is EXACT, not approximate. The residual is
`f * s * |X/Z - x_undistorted|`, so the principal point of `K_new`
cancels between A and B and both focal lengths carry the same factor
`s`. Hence `residual_native = residual_scaled / s` to machine precision.

**The consequence is a real limitation and it is measured rather than
asserted:** this gate is BLIND to the principal point of `K_new`. The
quarter-pixel `(c + 0.5) * s - 0.5` versus naive `c * s` convention
(`imvid-baseline-freeze.md:239-242`) changes the residual by EXACTLY
zero. Check `k_new_matches_frozen_record` -- equality against the
recorded experiment-156 camera to 1e-9 -- is the instrument that catches
that, and it is why it exists as a separate assertion.

--------------------------------------------------------------------
TWO IMPLEMENTATIONS, EACH WITH A STATED ROLE
--------------------------------------------------------------------

The undistortion inverse exists twice, deliberately:

  cv2     the PRODUCTION path, exactly as the record specifies. Used by
          `--mode verify`. `cv2` is not installed on the workstation, so
          this path cannot be exercised locally.
  numpy   a fixed-point inverse (cv2's own algorithm) that makes
          `--mode self-test` runnable on Windows with NO data files and
          NO cv2, so the arithmetic is verifiable before a cell is
          submitted.

When `cv2` IS importable the self-test additionally asserts the two
agree to `CV2_AGREEMENT_TOL_PX`, which certifies the stand-in and the
runtime's cv2 at once. `--mode verify` ALWAYS runs the self-test first,
so the gate number and the instrument's self-certification land in the
same manifest from the same cell.

--------------------------------------------------------------------
FAIL-CLOSED
--------------------------------------------------------------------

Raised immediately: a missing or unparseable model file; an unhandled
camera model; a camera id referenced by an image but absent from
`cameras.bin`; an observation naming a 3D point that is not in
`points3D.bin`; a failed self-test check; an output path inside the git
repository.

Collected, WRITTEN TO THE MANIFEST, and then raised so the evidence
survives the refusal: the gate itself; any image with zero usable pairs;
any observation whose point falls behind its camera (a pose-convention
signature); a non-finite residual.

Usage:
  python3 scripts/imvid_verify_pinhole.py --mode self-test

  python3 scripts/imvid_verify_pinhole.py --mode verify \
      --model /apollo/users/sri/proj_adags/data/imvid/sparse35/frame_000000/model_out \
      --scale 0.5 \
      --ply   /apollo/users/sri/proj_adags/data/imvid/init35/points3d_colmap_union.ply \
      --ply-first-n 5140 \
      --out   /apollo/users/sri/proj_adags/runs/elgs/<run>/MANIFEST.imvid_verify_pinhole.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.artifacts import atomic_write_bytes  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402

SCHEMA = "imvid-verify-pinhole-v1"

#: THE GATE, and it is stated WITH ITS RASTER. 2.0 px at native
#: 5312x2988 (`imvid-baseline-freeze.md:322-323`, restated as a
#: cross-dataset bar at `dataset-admission-matrix-2026-08-18.md:145`). Never
#: compared against a scaled-raster statistic -- see the module
#: docstring.
GATE_MEAN_REPROJ_PX_NATIVE = 2.0

#: The supplied ImViD Opera camera, read from the data and recorded at
#: [[operations/imvid-sample-ingestion]]:54-70. Used ONLY by the
#: self-test and as an advisory comparison; `--mode verify` always
#: derives from the model's own `cameras.bin`.
RECORDED_OPENCV_CAMERA = {
    "model": "OPENCV",
    "width": 5312,
    "height": 2988,
    "params": [
        2603.3326864600399, 2602.2436600602796, 2656.0, 1494.0,
        -0.024546867645992888, 0.0035148158874614976,
        -0.00045079985723632071, -0.00023832152424359775,
    ],
}

#: The derived PINHOLE camera experiment 156 produced at scale 0.5
#: ([[operations/imvid-baseline-freeze]]:234-237). A DIFFERENT script
#: computed these, so reproducing them is a cross-implementation check
#: of this file's arithmetic, not a tautology.
RECORDED_PINHOLE_AT_SCALE_HALF = {
    "width": 2656, "height": 1494,
    "fx": 1301.66634323002, "fy": 1301.1218300301398,
    "cx": 1327.75, "cy": 746.75,
}

#: Agreement required against the frozen experiment-156 camera. Not a
#: tuning tolerance: the two derivations are the same closed formula on
#: the same decimal inputs and agree to the last bit in practice.
K_NEW_TOL = 1e-9

#: cv2-vs-numpy undistortion agreement. `cv2.undistortPoints` runs 5
#: fixed-point iterations; this file's numpy inverse runs
#: `UNDISTORT_ITERS`. The measured round-trip non-convergence of the
#: 5-iteration solver on THIS camera is 1.7e-05 px at native
#: ([[operations/imvid-baseline-freeze]]:252-253), so 1e-3 px is ~60x
#: above the expected disagreement and still ~2000x below the gate.
CV2_AGREEMENT_TOL_PX = 1e-3

#: The correct synthetic pipeline must close to this. It is a
#: round-trip through an analytic forward model and its own inverse, so
#: the residual is convergence noise, not geometry.
SYNTHETIC_EXACT_TOL_PX = 1e-6

UNDISTORT_ITERS = 30

#: COLMAP camera model ids -> (name, n_params). Same table as
#: `scripts/imvid_sparse_init.py:130-131`.
_CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
                  2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8)}

#: Only these two carry an unambiguous (fx, fy, cx, cy[, k1, k2, p1, p2])
#: reading. Opera is OPENCV; Meeting and Playing ship PINHOLE with no
#: distortion at all (`dataset-admission-matrix-2026-08-18.md:424-437`),
#: and the gate is nearly free on them, so both are supported.
_SUPPORTED_MODELS = ("OPENCV", "PINHOLE")

_POINT2D_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("pid", "<i8")])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ====================================================================
# COLMAP binary model -- parsers take BYTES so the self-test can drive
# them with no files on disk.
# ====================================================================


def parse_cameras_bin(data: bytes) -> dict[int, dict]:
    """`cameras.bin` -> {camera_id: {model, width, height, params}}.

    THE TEXT EXPORT CANNOT SUBSTITUTE. COLMAP 3.6's text writer emits
    about six significant figures, so `2602.2436600602796` comes back as
    `2602.24` -- a 3.66e-03 error on a focal length, which would enter
    every projection here. The binary model stores the raw doubles.
    """
    offset = 0
    (n_cameras,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    cameras: dict[int, dict] = {}
    for _ in range(n_cameras):
        camera_id, model_id, width, height = struct.unpack_from("<iiQQ", data, offset)
        offset += 24
        if model_id not in _CAMERA_MODELS:
            raise ContractError(f"unhandled COLMAP camera model id {model_id}")
        name, n_params = _CAMERA_MODELS[model_id]
        params = struct.unpack_from("<" + "d" * n_params, data, offset)
        offset += 8 * n_params
        cameras[int(camera_id)] = {"model": name, "width": int(width),
                                   "height": int(height), "params": list(params)}
    return cameras


def parse_images_bin(data: bytes) -> dict[str, dict]:
    """`images.bin` -> {name: {image_id, camera_id, qvec, tvec, xy, pids}}.

    Same record layout as `scripts/imvid_sparse_init.py:164-191`, with
    ONE difference: that parser advances past the POINTS2D block
    (`offset += n_points2d * 24`, line 187) and this one decodes it.
    Each record is `(x: float64, y: float64, point3D_id: int64)`;
    `point3D_id` is COLMAP's invalid sentinel (max uint64, i.e. -1 read
    as int64) for a feature that was never triangulated.
    """
    if _POINT2D_DTYPE.itemsize != 24:
        raise ContractError(
            f"POINTS2D record must be 24 bytes, numpy reports "
            f"{_POINT2D_DTYPE.itemsize}"
        )
    offset = 0
    (n_images,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    images: dict[str, dict] = {}
    for _ in range(n_images):
        image_id = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        qvec = struct.unpack_from("<dddd", data, offset)
        offset += 32
        tvec = struct.unpack_from("<ddd", data, offset)
        offset += 24
        camera_id = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        end = data.index(b"\x00", offset)
        name = data[offset:end].decode("utf-8")
        offset = end + 1
        (n_points2d,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        if n_points2d:
            block = np.frombuffer(data, dtype=_POINT2D_DTYPE,
                                  count=int(n_points2d), offset=offset)
            xy = np.stack([block["x"], block["y"]], axis=1).astype(np.float64)
            pids = block["pid"].astype(np.int64)
        else:
            xy = np.zeros((0, 2), dtype=np.float64)
            pids = np.zeros((0,), dtype=np.int64)
        offset += int(n_points2d) * 24
        if name in images:
            raise ContractError(f"duplicate image name in images.bin: {name!r}")
        images[name] = {"image_id": int(image_id), "camera_id": int(camera_id),
                        "qvec": list(qvec), "tvec": list(tvec),
                        "xy": xy, "pids": pids}
    return images


def parse_points3d_bin(data: bytes) -> dict[int, list[float]]:
    """`points3D.bin` -> {point3D_id: [x, y, z]} (raw doubles)."""
    offset = 0
    (n_points,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    points: dict[int, list[float]] = {}
    for _ in range(n_points):
        point_id, x, y, z = struct.unpack_from("<qddd", data, offset)
        offset += 32
        offset += 3 + 8  # rgb (3 x uint8) + reprojection error (double)
        (track_len,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        offset += int(track_len) * 8  # (image_id: int32, point2D_idx: int32)
        points[int(point_id)] = [x, y, z]
    return points


def _read_model(model_dir: Path) -> tuple[dict, dict, dict, dict]:
    required = ("cameras.bin", "images.bin", "points3D.bin")
    missing = [n for n in required if not (model_dir / n).is_file()]
    if missing:
        raise ContractError(
            f"{model_dir} is not a COLMAP binary model: missing {missing}. "
            "Point --model at the `model_out/` directory a "
            "scripts/imvid_sparse_init.py cell produced, not at its parent."
        )
    cameras = parse_cameras_bin((model_dir / "cameras.bin").read_bytes())
    images = parse_images_bin((model_dir / "images.bin").read_bytes())
    points = parse_points3d_bin((model_dir / "points3D.bin").read_bytes())
    hashes = {n: _sha256(model_dir / n) for n in required}
    return cameras, images, points, hashes


# ====================================================================
# Minimal PLY reader -- the union is written by
# `scripts/diva360_to_blender.py::write_points3d_ply` (ASCII, the
# x,y,z,nx,ny,nz,red,green,blue schema). `binary_little_endian` is also
# accepted because the file on Apollo cannot be inspected from here and
# a wasted cell is more expensive than 20 lines.
# ====================================================================

_PLY_TYPES = {
    "char": "<i1", "int8": "<i1", "uchar": "<u1", "uint8": "<u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def parse_ply_xyz(data: bytes) -> np.ndarray:
    """First `vertex` element's (x, y, z) as an (N, 3) float64 array."""
    marker = b"end_header"
    cut = data.find(marker)
    if cut < 0:
        raise ContractError("PLY has no end_header; it is not a PLY file")
    line_end = data.find(b"\n", cut)
    header = data[:cut].decode("ascii", errors="replace").splitlines()
    body = data[line_end + 1:]

    fmt = None
    element = None
    count = 0
    properties: list[tuple[str, str]] = []
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            if element is not None:
                break  # only the first element is read
            element, count = parts[1], int(parts[2])
        elif parts[0] == "property" and element is not None:
            if parts[1] == "list":
                raise ContractError(
                    f"PLY element {element!r} carries a list property; this "
                    "reader handles fixed-width vertex records only"
                )
            properties.append((parts[1], parts[2]))
    if element != "vertex":
        raise ContractError(f"first PLY element is {element!r}, expected 'vertex'")
    names = [n for _, n in properties]
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise ContractError(f"PLY vertex element has no {axis!r} property")
    index = [names.index(a) for a in ("x", "y", "z")]

    if fmt == "ascii":
        rows = body.split(b"\n")
        values = np.empty((count, 3), dtype=np.float64)
        taken = 0
        for row in rows:
            if taken == count:
                break
            fields = row.split()
            if not fields:
                continue
            values[taken] = [float(fields[i]) for i in index]
            taken += 1
        if taken != count:
            raise ContractError(
                f"PLY header declares {count} vertices, body holds {taken}"
            )
        return values
    if fmt == "binary_little_endian":
        try:
            dtype = np.dtype([(n, _PLY_TYPES[t]) for t, n in properties])
        except KeyError as exc:
            raise ContractError(f"unhandled PLY property type {exc}") from exc
        need = count * dtype.itemsize
        if len(body) < need:
            raise ContractError(
                f"PLY body is {len(body)} bytes, {need} needed for {count} vertices"
            )
        block = np.frombuffer(body, dtype=dtype, count=count)
        return np.stack([block["x"], block["y"], block["z"]],
                        axis=1).astype(np.float64)
    raise ContractError(
        f"unhandled PLY format {fmt!r}; expected 'ascii' or 'binary_little_endian'"
    )


# ====================================================================
# Geometry
# ====================================================================


def qvec2rotmat(qvec) -> np.ndarray:
    """COLMAP world-to-camera rotation from (QW, QX, QY, QZ)."""
    w, x, y, z = (float(v) for v in qvec)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        raise ContractError("degenerate zero quaternion in images.bin")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def camera_matrices(camera: dict) -> tuple[np.ndarray, np.ndarray]:
    """(K, distCoeffs) for a supported COLMAP camera.

    `distCoeffs` is length 4, which OpenCV reads as `(k1, k2, p1, p2)` --
    COLMAP's OPENCV parameter order exactly. A PINHOLE camera gets
    zeros, so the same code path serves the undistorted scenes.
    """
    model = camera["model"]
    if model not in _SUPPORTED_MODELS:
        raise ContractError(
            f"camera model {model!r} is not handled; this gate supports "
            f"{_SUPPORTED_MODELS}. Refusing rather than guessing a parameter "
            "order."
        )
    fx, fy, cx, cy = (float(v) for v in camera["params"][:4])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if model == "OPENCV":
        dist = np.array([float(v) for v in camera["params"][4:8]], dtype=np.float64)
    else:
        dist = np.zeros(4, dtype=np.float64)
    return K, dist


def derive_pinhole(K: np.ndarray, width: int, height: int, scale: float) -> dict:
    """The PINHOLE camera the undistorted images are expressed in.

    `f * scale` for the focals and `(c + 0.5) * scale - 0.5` for the
    principal point -- the frozen convention
    ([[operations/imvid-baseline-freeze]]:239-242, implemented at
    `scripts/imvid_pilot_prepare.py:231-238`). The naive `c * scale`
    differs by a quarter pixel in both axes on every camera, and this
    gate CANNOT see that difference (module docstring), which is why
    the self-test compares this derivation against the frozen record.
    """
    if scale <= 0.0:
        raise ContractError("--scale must be positive")
    K_new = np.array([
        [K[0, 0] * scale, 0.0, (K[0, 2] + 0.5) * scale - 0.5],
        [0.0, K[1, 1] * scale, (K[1, 2] + 0.5) * scale - 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return {"K_new": K_new,
            "width": int(round(width * scale)),
            "height": int(round(height * scale))}


def project_pinhole(xyz: np.ndarray, R: np.ndarray, t: np.ndarray,
                    K_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Limb A: analytic pinhole projection, NO distortion term."""
    cam = xyz @ R.T + t
    depth = cam[:, 2]
    safe = np.where(np.abs(depth) < 1e-12, 1e-12, depth)
    uv = np.stack([
        K_new[0, 0] * cam[:, 0] / safe + K_new[0, 2],
        K_new[1, 1] * cam[:, 1] / safe + K_new[1, 2],
    ], axis=1)
    return uv, depth


def project_distorted(xyz: np.ndarray, R: np.ndarray, t: np.ndarray,
                      K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Native-raster projection through the FULL camera model.

    The cross-check limb: this is the quantity COLMAP itself minimizes,
    recomputed here in the distorted space. Deliberately NOT
    `cv2.projectPoints` -- `distort_normalized` is certified against a
    known-answer grid by the self-test and needs no cv2, so the cv2
    dependency stays confined to the one call the record specifies.
    """
    cam = xyz @ R.T + t
    depth = cam[:, 2]
    safe = np.where(np.abs(depth) < 1e-12, 1e-12, depth)
    xd, yd = distort_normalized(cam[:, 0] / safe, cam[:, 1] / safe, dist)
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=1)


def distort_normalized(x: np.ndarray, y: np.ndarray,
                       dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV / COLMAP `OPENCV` forward distortion, (k1, k2, p1, p2)."""
    k1, k2, p1, p2 = (float(v) for v in dist[:4])
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def undistort_normalized(xd: np.ndarray, yd: np.ndarray,
                         dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cv2's fixed-point inverse of `distort_normalized`, in numpy."""
    k1, k2, p1, p2 = (float(v) for v in dist[:4])
    x, y = np.array(xd, dtype=np.float64), np.array(yd, dtype=np.float64)
    for _ in range(UNDISTORT_ITERS):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


def undistort_pixels(uv: np.ndarray, K: np.ndarray, dist: np.ndarray,
                     K_new: np.ndarray, *, backend: str) -> np.ndarray:
    """Limb B: a distorted native observation carried into the K_new raster.

    `backend='cv2'` is the production path the record specifies;
    `backend='numpy'` exists so `--mode self-test` runs where cv2 is
    absent. The self-test asserts they agree whenever cv2 is present.

    Note that `(u - cx) / fx` is invariant to the half-pixel difference
    between COLMAP's corner-origin convention and OpenCV's
    centre-origin one, because `u` and `cx` shift together. The gate is
    therefore unaffected by that convention.
    """
    if backend == "cv2":
        import cv2

        out = cv2.undistortPoints(
            np.ascontiguousarray(uv.reshape(-1, 1, 2), dtype=np.float64),
            K, dist, P=K_new,
        )
        return np.asarray(out, dtype=np.float64).reshape(-1, 2)
    if backend == "numpy":
        xd = (uv[:, 0] - K[0, 2]) / K[0, 0]
        yd = (uv[:, 1] - K[1, 2]) / K[1, 1]
        x, y = undistort_normalized(xd, yd, dist)
        return np.stack([K_new[0, 0] * x + K_new[0, 2],
                         K_new[1, 1] * y + K_new[1, 2]], axis=1)
    raise ContractError(f"unknown undistortion backend {backend!r}")


# ====================================================================
# Statistics, raster bookkeeping, and the gate
# ====================================================================


def _stats_px(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "min": float(values.min()),
    }


def _dual_raster_stats(scaled: np.ndarray, scale: float, out_w: int, out_h: int,
                       native_w: int, native_h: int) -> dict:
    """The SAME residual in both rasters, each carrying its own label.

    Nothing downstream may read a bare number: the raster is part of the
    payload.
    """
    return {
        "scaled": {"raster": f"PINHOLE {out_w}x{out_h} (scale {scale:.6f})",
                   **_stats_px(scaled)},
        "native_equivalent": {
            "raster": f"NATIVE {native_w}x{native_h} (= scaled / {scale:.6f})",
            **_stats_px(scaled / scale)},
    }


def _gate_decision(stats: dict, scale: float) -> dict:
    """Apply the 2 px NATIVE gate, and refuse inconsistent bookkeeping."""
    mean_scaled = stats["scaled"]["mean"]
    mean_native = stats["native_equivalent"]["mean"]
    if abs(mean_native - mean_scaled / scale) > 1e-9 * max(1.0, abs(mean_native)):
        raise ContractError(
            f"raster bookkeeping is inconsistent: scaled mean {mean_scaled} at "
            f"scale {scale} implies {mean_scaled / scale} native, but the "
            f"native-equivalent statistic reads {mean_native}"
        )
    return {
        "statistic": "mean residual",
        "raster": stats["native_equivalent"]["raster"],
        "threshold_px": GATE_MEAN_REPROJ_PX_NATIVE,
        "value_px": mean_native,
        "passed": bool(mean_native <= GATE_MEAN_REPROJ_PX_NATIVE),
        "scaled_raster_value_px": mean_scaled,
        "scaled_raster": stats["scaled"]["raster"],
        "trap": (
            "the scaled-raster value MUST NOT be compared against this "
            "threshold; at scale 0.5 it reads half and would pass trivially"
        ),
    }


# ====================================================================
# Limb 1 -- the reprojection gate
# ====================================================================


def _residual_limb(images: dict, points: dict, derived: dict, scale: float,
                   *, backend: str) -> dict:
    """Per-(point, image) residuals in the K_new raster, plus the
    distorted-space cross-check at native resolution."""
    refusals: list[str] = []
    per_camera: dict[str, dict] = {}
    all_scaled: list[np.ndarray] = []
    all_native_dist: list[np.ndarray] = []
    n_behind_total = 0
    behind_examples: list[str] = []
    empty_images: list[str] = []

    for name in sorted(images):
        image = images[name]
        camera_id = image["camera_id"]
        if camera_id not in derived:
            raise ContractError(
                f"image {name!r} references camera id {camera_id}, which is "
                "absent from cameras.bin"
            )
        entry = derived[camera_id]
        pids = image["pids"]
        keep = pids >= 0
        n_pairs = int(keep.sum())
        if n_pairs == 0:
            empty_images.append(name)
            per_camera[name] = {"pairs": 0, "camera_id": camera_id}
            continue
        sel = pids[keep]
        try:
            xyz = np.array([points[int(i)] for i in sel], dtype=np.float64)
        except KeyError as exc:
            raise ContractError(
                f"image {name!r} observes 3D point id {exc} which is not in "
                "points3D.bin; the model is inconsistent"
            ) from exc

        R = qvec2rotmat(image["qvec"])
        t = np.asarray(image["tvec"], dtype=np.float64)
        uv_obs = image["xy"][keep]

        uv_proj, depth = project_pinhole(xyz, R, t, entry["K_new"])
        behind = depth <= 0.0
        if behind.any():
            n_behind_total += int(behind.sum())
            if len(behind_examples) < 5:
                behind_examples.append(f"{name}:{int(behind.sum())}")

        uv_undist = undistort_pixels(uv_obs, entry["K"], entry["dist"],
                                     entry["K_new"], backend=backend)
        residual = np.linalg.norm(uv_proj - uv_undist, axis=1)

        proj_dist = project_distorted(xyz, R, t, entry["K"], entry["dist"])
        residual_native_dist = np.linalg.norm(proj_dist - uv_obs, axis=1)

        if not np.isfinite(residual).all():
            refusals.append(f"non-finite residual on {name}")
        all_scaled.append(residual)
        all_native_dist.append(residual_native_dist)
        per_camera[name] = {
            "camera_id": camera_id,
            "pairs": n_pairs,
            "points_behind_camera": int(behind.sum()),
            "mean_px_scaled": float(residual.mean()),
            "mean_px_native_equivalent": float(residual.mean() / scale),
            "max_px_native_equivalent": float(residual.max() / scale),
            "mean_px_native_distorted_crosscheck": float(residual_native_dist.mean()),
        }

    if not all_scaled:
        raise ContractError(
            "no (point, image) pairs at all; the POINTS2D block is empty for "
            "every image, so this model carries no observations to check"
        )
    scaled = np.concatenate(all_scaled)
    native_dist = np.concatenate(all_native_dist)

    if empty_images:
        refusals.append(
            f"{len(empty_images)} image(s) contribute zero pairs: "
            f"{empty_images[:5]}"
        )
    if n_behind_total:
        refusals.append(
            f"{n_behind_total} observation(s) project BEHIND their camera "
            f"({behind_examples}). An observed point is in front of the camera "
            "by construction, so this is a POSE CONVENTION error: check "
            "world-to-camera vs camera-to-world and R vs R-transpose."
        )

    # The pooled statistics carry ONE raster label, so they are only
    # meaningful if every camera shares that raster. ImViD's 39 views
    # share CAMERA_ID 2 ([[operations/imvid-sample-ingestion]]:54-70);
    # a multi-raster model would silently mislabel the gate.
    rasters = {(e["width"], e["height"], e["native_width"], e["native_height"])
               for e in derived.values()}
    if len(rasters) != 1:
        raise ContractError(
            f"cameras disagree on the raster ({sorted(rasters)}); a pooled "
            "residual cannot carry one raster label, and the gate is stated "
            "per raster. Run this model one camera group at a time."
        )
    any_entry = next(iter(derived.values()))
    stats = _dual_raster_stats(scaled, scale, any_entry["width"],
                               any_entry["height"], any_entry["native_width"],
                               any_entry["native_height"])
    return {
        "definition": {
            "A": "pi(K_new @ (R_i @ X + t_i)) -- analytic pinhole, no distortion",
            "B": f"undistort(observed native POINTS2D, K, dist, P=K_new) "
                 f"[backend={backend}]",
            "residual": "||A - B||_2",
        },
        "pairs": int(scaled.size),
        "images_with_pairs": int(sum(1 for v in per_camera.values() if v["pairs"])),
        "images_total": len(images),
        "residual": stats,
        "gate": _gate_decision(stats, scale),
        "native_distorted_crosscheck": {
            "raster": f"NATIVE {any_entry['native_width']}x"
                      f"{any_entry['native_height']} (distorted space)",
            "definition": "||project_distorted(X, R, t, K, dist) - observed||_2",
            "note": (
                "this is COLMAP's OWN reprojection statistic recomputed here. "
                "It is NOT the gate: the gated residual lives in the "
                "undistorted raster and differs by the local Jacobian of the "
                "undistortion map, which exceeds 1 toward the periphery."
            ),
            **_stats_px(native_dist),
        },
        "per_camera": per_camera,
        "undistortion_backend": backend,
        "backend_is_the_production_path": backend == "cv2",
        "refusals": refusals,
    }


# ====================================================================
# Limb 2 -- PLY coverage
# ====================================================================


#: Stated in the output every time, because the fraction is otherwise
#: unreadable: a low value may be correct.
PLY_UNION_NOTE = (
    "The union PLY is a CONCATENATION of three per-frame clouds (frames 0, "
    "150, 299, ascending, no dedup -- scripts/imvid_build_initialization.py). "
    "A point triangulated at frame 150 NEED NOT be visible in a frame-0 "
    "camera, so a fraction below 1.0 is expected and is NOT a failure. What "
    "IS a hard failure is a FRAME-0 point failing to land in a FRAME-0 "
    "camera; use --ply-first-n with that frame's own point count (frame 0 = "
    "5140) to read that segment separately. No threshold is applied to "
    "either number -- this limb reports, it does not gate."
)


def _ply_coverage(xyz: np.ndarray, images: dict, derived: dict,
                  label: str) -> dict:
    per_camera: dict[str, dict] = {}
    fractions: list[float] = []
    for name in sorted(images):
        entry = derived[images[name]["camera_id"]]
        R = qvec2rotmat(images[name]["qvec"])
        t = np.asarray(images[name]["tvec"], dtype=np.float64)
        uv, depth = project_pinhole(xyz, R, t, entry["K_new"])
        in_front = depth > 0.0
        inside = (
            in_front
            & (uv[:, 0] >= -0.5) & (uv[:, 0] <= entry["width"] - 0.5)
            & (uv[:, 1] >= -0.5) & (uv[:, 1] <= entry["height"] - 0.5)
        )
        fraction = float(inside.sum()) / float(xyz.shape[0])
        fractions.append(fraction)
        per_camera[name] = {
            "in_front": int(in_front.sum()),
            "inside_raster": int(inside.sum()),
            "fraction_inside": fraction,
        }
    values = np.asarray(fractions, dtype=np.float64)
    return {
        "segment": label,
        "points": int(xyz.shape[0]),
        "cameras": len(images),
        "raster_bounds": "[-0.5, W-0.5] x [-0.5, H-0.5] in the K_new raster",
        "fraction_inside": {"min": float(values.min()), "mean": float(values.mean()),
                            "max": float(values.max())},
        "per_camera": per_camera,
    }


# ====================================================================
# Self-test -- runs with NO data files and NO cv2
# ====================================================================


def _synthetic_fixture() -> dict:
    """A synthetic scene built from the RECORDED ImViD camera.

    Native pixels on a grid are back-projected to true rays, given
    depths, and pushed into a world frame through a non-trivial pose.
    The forward path therefore has an exactly known answer, so the
    CORRECT pipeline must close to convergence noise and any corruption
    of it must not.
    """
    camera = RECORDED_OPENCV_CAMERA
    K, dist = camera_matrices(camera)
    pin = derive_pinhole(K, camera["width"], camera["height"], 0.5)

    axis = np.array([0.3, -0.5, 0.81], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(25.0)
    qvec = [np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)]
    R = qvec2rotmat(qvec)
    t = np.array([0.1, -0.2, 6.0], dtype=np.float64)

    us = np.linspace(0.0, camera["width"] - 1.0, 24)
    vs = np.linspace(0.0, camera["height"] - 1.0, 16)
    gu, gv = np.meshgrid(us, vs)
    uv_obs = np.stack([gu.ravel(), gv.ravel()], axis=1)

    xd = (uv_obs[:, 0] - K[0, 2]) / K[0, 0]
    yd = (uv_obs[:, 1] - K[1, 2]) / K[1, 1]
    x, y = undistort_normalized(xd, yd, dist)
    rng = np.random.default_rng(0)
    depth = 3.0 + 6.0 * rng.random(x.shape[0])
    cam = np.stack([x * depth, y * depth, depth], axis=1)
    xyz = (cam - t) @ R  # R.T @ (cam - t) for row vectors

    return {"camera": camera, "K": K, "dist": dist, "K_new": pin["K_new"],
            "scale": 0.5, "R": R, "t": t, "qvec": qvec, "xyz": xyz,
            "uv_obs": uv_obs}


def _synthetic_residual(fx: dict, *, R=None, t=None, K=None, dist=None,
                        K_new=None, camera_to_world=False,
                        backend: str = "numpy") -> tuple[float, int]:
    """(mean native-equivalent residual, points behind camera)."""
    R = fx["R"] if R is None else R
    t = fx["t"] if t is None else t
    K = fx["K"] if K is None else K
    dist = fx["dist"] if dist is None else dist
    K_new = fx["K_new"] if K_new is None else K_new
    if camera_to_world:
        cam = (fx["xyz"] - t) @ R
    else:
        cam = fx["xyz"] @ R.T + t
    depth = cam[:, 2]
    safe = np.where(np.abs(depth) < 1e-12, 1e-12, depth)
    uv_proj = np.stack([K_new[0, 0] * cam[:, 0] / safe + K_new[0, 2],
                        K_new[1, 1] * cam[:, 1] / safe + K_new[1, 2]], axis=1)
    uv_undist = undistort_pixels(fx["uv_obs"], K, dist, K_new, backend=backend)
    residual = np.linalg.norm(uv_proj - uv_undist, axis=1) / fx["scale"]
    return float(residual.mean()), int((depth <= 0.0).sum())


def _encode_cameras_bin(cameras: list[tuple[int, int, int, int, list[float]]]) -> bytes:
    out = struct.pack("<Q", len(cameras))
    for camera_id, model_id, width, height, params in cameras:
        out += struct.pack("<iiQQ", camera_id, model_id, width, height)
        out += struct.pack("<" + "d" * len(params), *params)
    return out


def _encode_images_bin(images: list[tuple]) -> bytes:
    out = struct.pack("<Q", len(images))
    for image_id, qvec, tvec, camera_id, name, xy, pids in images:
        out += struct.pack("<i", image_id)
        out += struct.pack("<dddd", *qvec)
        out += struct.pack("<ddd", *tvec)
        out += struct.pack("<i", camera_id)
        out += name.encode("utf-8") + b"\x00"
        out += struct.pack("<Q", len(pids))
        for (px, py), pid in zip(xy, pids):
            out += struct.pack("<ddq", px, py, pid)
    return out


def _encode_points3d_bin(points: list[tuple[int, tuple, int]]) -> bytes:
    out = struct.pack("<Q", len(points))
    for point_id, xyz, track_len in points:
        out += struct.pack("<qddd", point_id, *xyz)
        out += struct.pack("<BBB", 10, 20, 30)
        out += struct.pack("<d", 0.5)
        out += struct.pack("<Q", track_len)
        out += b"\x00" * (track_len * 8)
    return out


def _check(name: str, ok: bool, detail: dict) -> dict:
    if not ok:
        raise ContractError(f"SELF-TEST FAILED: {name} -- {detail}")
    return {"name": name, "status": "PASS", **detail}


def run_self_test() -> list[dict]:
    checks: list[dict] = []
    camera = RECORDED_OPENCV_CAMERA
    K, dist = camera_matrices(camera)

    # 1 -- the derivation reproduces a number a DIFFERENT script produced
    pin = derive_pinhole(K, camera["width"], camera["height"], 0.5)
    K_new = pin["K_new"]
    rec = RECORDED_PINHOLE_AT_SCALE_HALF
    deltas = {
        "fx": abs(K_new[0, 0] - rec["fx"]), "fy": abs(K_new[1, 1] - rec["fy"]),
        "cx": abs(K_new[0, 2] - rec["cx"]), "cy": abs(K_new[1, 2] - rec["cy"]),
    }
    checks.append(_check(
        "k_new_matches_frozen_record",
        max(deltas.values()) <= K_NEW_TOL
        and (pin["width"], pin["height"]) == (rec["width"], rec["height"]),
        {"tolerance": K_NEW_TOL, "max_abs_delta": float(max(deltas.values())),
         "per_parameter": {k: float(v) for k, v in deltas.items()},
         "derived": [float(K_new[0, 0]), float(K_new[1, 1]),
                     float(K_new[0, 2]), float(K_new[1, 2])],
         "width_height": [pin["width"], pin["height"]],
         "note": "reproduces experiment 156's PINHOLE line, computed by "
                 "scripts/imvid_pilot_prepare.py, through this file's formula"}))

    # 2 -- unit scale must be the identity
    unit = derive_pinhole(K, camera["width"], camera["height"], 1.0)
    checks.append(_check(
        "k_new_identity_at_unit_scale",
        bool(np.allclose(unit["K_new"], K, rtol=0.0, atol=0.0))
        and (unit["width"], unit["height"]) == (camera["width"], camera["height"]),
        {"note": "(c + 0.5) * 1 - 0.5 == c exactly"}))

    # 3 -- rotation algebra
    R = qvec2rotmat([np.cos(0.3), np.sin(0.3) * 0.6, np.sin(0.3) * 0.8, 0.0])
    checks.append(_check(
        "quaternion_to_rotation_is_special_orthogonal",
        bool(np.allclose(R @ R.T, np.eye(3), atol=1e-14))
        and abs(np.linalg.det(R) - 1.0) < 1e-14
        and bool(np.allclose(qvec2rotmat([1.0, 0.0, 0.0, 0.0]), np.eye(3))),
        {"orthonormality_max_abs_error": float(np.abs(R @ R.T - np.eye(3)).max()),
         "det": float(np.linalg.det(R))}))

    # 4 -- the binary parsers, INCLUDING the POINTS2D block the sibling
    #      parser skips. Two images with different observation counts, so
    #      a wrong 24-byte stride corrupts the second image's name.
    cam_bytes = _encode_cameras_bin([(2, 4, camera["width"], camera["height"],
                                      camera["params"])])
    img_a = (7, (1.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.3), 2, "cam01.png",
             [(11.5, 12.5), (13.5, 14.5), (15.5, 16.5)], [1, -1, 3])
    img_b = (9, (0.5, 0.5, 0.5, 0.5), (1.0, 2.0, 3.0), 2, "cam02.png",
             [(21.5, 22.5), (23.5, 24.5)], [3, 1])
    parsed_cams = parse_cameras_bin(cam_bytes)
    parsed_imgs = parse_images_bin(_encode_images_bin([img_a, img_b]))
    parsed_pts = parse_points3d_bin(_encode_points3d_bin(
        [(1, (1.0, 2.0, 3.0), 4), (3, (-1.5, 0.25, 9.0), 2)]))
    ok = (
        parsed_cams == {2: {"model": "OPENCV", "width": camera["width"],
                            "height": camera["height"],
                            "params": camera["params"]}}
        and sorted(parsed_imgs) == ["cam01.png", "cam02.png"]
        and parsed_imgs["cam02.png"]["image_id"] == 9
        and np.array_equal(parsed_imgs["cam01.png"]["pids"],
                           np.array([1, -1, 3], dtype=np.int64))
        and np.allclose(parsed_imgs["cam01.png"]["xy"], np.array(img_a[5]))
        and np.allclose(parsed_imgs["cam02.png"]["xy"], np.array(img_b[5]))
        and parsed_pts == {1: [1.0, 2.0, 3.0], 3: [-1.5, 0.25, 9.0]}
    )
    checks.append(_check(
        "binary_parsers_round_trip", ok,
        {"note": "cameras.bin / images.bin (POINTS2D READ, not skipped) / "
                 "points3D.bin, from in-memory bytes -- no files on disk",
         "images": sorted(parsed_imgs), "point_ids": sorted(parsed_pts)}))

    # 5 -- PLY reader, both encodings
    ply_xyz = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, 6.25]], dtype=np.float64)
    ascii_ply = (
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
        "1.000000 2.000000 3.000000 0 0 0 1 2 3\n"
        "-4.000000 5.500000 6.250000 0 0 0 4 5 6\n"
    ).encode("utf-8")
    bin_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                          ("red", "<u1"), ("green", "<u1"), ("blue", "<u1")])
    block = np.zeros(2, dtype=bin_dtype)
    block["x"], block["y"], block["z"] = ply_xyz[:, 0], ply_xyz[:, 1], ply_xyz[:, 2]
    binary_ply = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("utf-8") + block.tobytes()
    checks.append(_check(
        "ply_parser_round_trip",
        bool(np.allclose(parse_ply_xyz(ascii_ply), ply_xyz))
        and bool(np.allclose(parse_ply_xyz(binary_ply), ply_xyz)),
        {"formats": ["ascii", "binary_little_endian"]}))

    # 6 -- the CORRECT pipeline closes on a synthetic fixture
    fixture = _synthetic_fixture()
    correct, behind = _synthetic_residual(fixture)
    forward_uv = project_distorted(fixture["xyz"], fixture["R"], fixture["t"],
                                   fixture["K"], fixture["dist"])
    forward_err = float(np.abs(forward_uv - fixture["uv_obs"]).max())
    checks.append(_check(
        "synthetic_projection_is_exact",
        correct <= SYNTHETIC_EXACT_TOL_PX and behind == 0
        and forward_err <= SYNTHETIC_EXACT_TOL_PX,
        {"mean_residual_px_native_equivalent": correct,
         "tolerance_px": SYNTHETIC_EXACT_TOL_PX,
         "points_behind_camera": behind,
         "forward_model_max_abs_error_px_native": forward_err,
         "n_points": int(fixture["xyz"].shape[0])}))

    # 7 -- the raster trap, demonstrated numerically
    trap_scaled = np.full(4, 1.1)
    trap = _dual_raster_stats(trap_scaled, 0.5, 2656, 1494, 5312, 2988)
    decision = _gate_decision(trap, 0.5)
    pass_scaled = _gate_decision(
        _dual_raster_stats(np.full(4, 0.9), 0.5, 2656, 1494, 5312, 2988), 0.5)
    checks.append(_check(
        "raster_arithmetic_and_the_gate_trap",
        decision["passed"] is False and abs(decision["value_px"] - 2.2) < 1e-12
        and pass_scaled["passed"] is True
        and abs(pass_scaled["value_px"] - 1.8) < 1e-12,
        {"demonstration": (
            "a 1.1 px mean on the 2656x1494 raster is 2.2 px NATIVE and FAILS "
            "the 2 px gate, yet a naive comparison of the bare 1.1 against 2.0 "
            "would have PASSED it"),
         "failing_native_px": decision["value_px"],
         "passing_native_px": pass_scaled["value_px"]}))

    # 8-11 -- the instrument must DETECT each corruption
    corruptions = [
        ("detects_transposed_rotation",
         dict(R=fixture["R"].T),
         "R vs R-transpose"),
        ("detects_camera_to_world_pose",
         dict(camera_to_world=True),
         "world-to-camera vs camera-to-world"),
        ("detects_swapped_distortion_order",
         dict(dist=np.array([fixture["dist"][2], fixture["dist"][3],
                             fixture["dist"][0], fixture["dist"][1]])),
         "distCoeffs given as (p1, p2, k1, k2)"),
        ("detects_dropped_distortion",
         dict(dist=np.zeros(4)),
         "OPENCV camera treated as PINHOLE -- the silent-corruption path"),
    ]
    for name, kwargs, description in corruptions:
        mean_px, n_behind = _synthetic_residual(fixture, **kwargs)
        detected = bool(n_behind > 0 or mean_px > GATE_MEAN_REPROJ_PX_NATIVE)
        checks.append(_check(
            name, detected,
            {"corruption": description,
             "mean_residual_px_native_equivalent": mean_px,
             "points_behind_camera": n_behind,
             "gate_px_native": GATE_MEAN_REPROJ_PX_NATIVE,
             "correct_pipeline_px_native": correct}))

    # 12 -- the MEASURED blind spot. Not a failure; a limitation, and the
    #       reason check 1 exists.
    naive = np.array([[K_new[0, 0], 0.0, camera["params"][2] * 0.5],
                      [0.0, K_new[1, 1], camera["params"][3] * 0.5],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    naive_mean, _ = _synthetic_residual(fixture, K_new=naive)
    checks.append(_check(
        "blind_to_principal_point_by_construction",
        abs(naive_mean - correct) <= 1e-9,
        {"finding": (
            "replacing (c + 0.5) * s - 0.5 with the naive c * s moves the "
            "principal point a quarter pixel and changes this residual by "
            "ZERO, because the principal point cancels between A and B. THIS "
            "GATE CANNOT SEE THAT ERROR. Check "
            "'k_new_matches_frozen_record' is what catches it."),
         "residual_with_frozen_convention_px_native": correct,
         "residual_with_naive_convention_px_native": naive_mean,
         "difference_px": abs(naive_mean - correct)}))

    # 13 -- certify the numpy stand-in against the production backend
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        checks.append({
            "name": "cv2_cross_check", "status": "SKIPPED",
            "reason": f"cv2 is not importable ({exc})",
            "consequence": (
                "the production undistortion backend was NOT exercised. Every "
                "other check above ran on the numpy stand-in. Run --mode "
                "self-test inside the Apollo image to close this."),
        })
    else:
        cv2_backend, _ = _synthetic_residual(fixture, backend="cv2")
        uv_np = undistort_pixels(fixture["uv_obs"], fixture["K"], fixture["dist"],
                                 fixture["K_new"], backend="numpy")
        uv_cv = undistort_pixels(fixture["uv_obs"], fixture["K"], fixture["dist"],
                                 fixture["K_new"], backend="cv2")
        agreement = float(np.abs(uv_np - uv_cv).max())
        checks.append(_check(
            "cv2_cross_check",
            agreement <= CV2_AGREEMENT_TOL_PX
            and cv2_backend <= CV2_AGREEMENT_TOL_PX / fixture["scale"],
            {"max_abs_disagreement_px_scaled": agreement,
             "tolerance_px": CV2_AGREEMENT_TOL_PX,
             "cv2_backend_residual_px_native_equivalent": cv2_backend,
             "note": "cv2.undistortPoints runs 5 fixed-point iterations, this "
                     f"file's numpy inverse runs {UNDISTORT_ITERS}"}))
    return checks


# ====================================================================
# Commands
# ====================================================================


def cmd_self_test(args) -> dict:
    checks = run_self_test()
    for check in checks:
        print(f"[imvid] self-test {check['status']:7s} {check['name']}", flush=True)
        if check["status"] == "SKIPPED":
            print(f"[imvid]   REASON: {check['reason']}", flush=True)
            print(f"[imvid]   {check['consequence']}", flush=True)
    skipped = [c["name"] for c in checks if c["status"] == "SKIPPED"]
    print(f"[imvid] SELF-TEST PASSED: {len(checks) - len(skipped)} check(s) ran"
          + (f", {len(skipped)} SKIPPED {skipped}" if skipped else ""), flush=True)
    return {"mode": "self-test", "checks": checks,
            "checks_run": len(checks) - len(skipped), "checks_skipped": skipped}


def _prepare_cameras(cameras: dict, scale: float) -> dict:
    derived = {}
    for camera_id, camera in cameras.items():
        K, dist = camera_matrices(camera)
        pin = derive_pinhole(K, camera["width"], camera["height"], scale)
        recorded = RECORDED_PINHOLE_AT_SCALE_HALF
        matches = bool(
            scale == 0.5
            and camera["model"] == RECORDED_OPENCV_CAMERA["model"]
            and abs(pin["K_new"][0, 0] - recorded["fx"]) <= K_NEW_TOL
            and abs(pin["K_new"][1, 1] - recorded["fy"]) <= K_NEW_TOL
            and abs(pin["K_new"][0, 2] - recorded["cx"]) <= K_NEW_TOL
            and abs(pin["K_new"][1, 2] - recorded["cy"]) <= K_NEW_TOL
        )
        derived[camera_id] = {
            "K": K, "dist": dist, "K_new": pin["K_new"],
            "width": pin["width"], "height": pin["height"],
            "native_width": camera["width"], "native_height": camera["height"],
            "model": camera["model"],
            "matches_recorded_imvid_pinhole": matches,
        }
    return derived


def _describe_cameras(derived: dict, scale: float) -> dict:
    out = {}
    for camera_id, entry in derived.items():
        out[str(camera_id)] = {
            "source_model": entry["model"],
            "source_raster": f"{entry['native_width']}x{entry['native_height']}",
            "scale": scale,
            "pinhole_raster": f"{entry['width']}x{entry['height']}",
            "K_new": {"fx": float(entry["K_new"][0, 0]),
                      "fy": float(entry["K_new"][1, 1]),
                      "cx": float(entry["K_new"][0, 2]),
                      "cy": float(entry["K_new"][1, 2])},
            "dist_k1_k2_p1_p2": [float(v) for v in entry["dist"]],
            "matches_recorded_imvid_pinhole": entry["matches_recorded_imvid_pinhole"],
        }
    return out


def _print_cameras(described: dict) -> None:
    for camera_id, entry in sorted(described.items()):
        print(f"[imvid] camera {camera_id}: {entry['source_model']} "
              f"{entry['source_raster']} -> PINHOLE {entry['pinhole_raster']} "
              f"@ scale {entry['scale']:.6f}", flush=True)
        k = entry["K_new"]
        print(f"[imvid]   K_new fx={k['fx']!r} fy={k['fy']!r} "
              f"cx={k['cx']!r} cy={k['cy']!r}", flush=True)
        print("[imvid]   matches the recorded experiment-156 PINHOLE line: "
              f"{entry['matches_recorded_imvid_pinhole']}", flush=True)


def _print_ply_limb(limb: dict) -> None:
    frac = limb["fraction_inside"]
    print(f"[imvid] PLY coverage [{limb['segment']}] {limb['points']} point(s) "
          f"over {limb['cameras']} camera(s), bounds {limb['raster_bounds']}",
          flush=True)
    print(f"[imvid]   fraction inside: min {frac['min']:.4f}  "
          f"mean {frac['mean']:.4f}  max {frac['max']:.4f}", flush=True)
    for name, entry in sorted(limb["per_camera"].items()):
        print(f"[imvid]     {name:>14s}  inside {entry['inside_raster']:>7d}  "
              f"in-front {entry['in_front']:>7d}  "
              f"fraction {entry['fraction_inside']:.4f}", flush=True)


def _load_ply(args) -> tuple[np.ndarray, dict]:
    ply_path = Path(args.ply)
    if not ply_path.is_file():
        raise ContractError(f"--ply {ply_path} does not exist")
    xyz = parse_ply_xyz(ply_path.read_bytes())
    if not xyz.size:
        raise ContractError(f"--ply {ply_path} holds no vertices")
    return xyz, {"path": str(ply_path), "points": int(xyz.shape[0]),
                 "bytes": ply_path.stat().st_size, "sha256": _sha256(ply_path)}


def _ply_limbs(args, xyz: np.ndarray, images: dict, derived: dict) -> list[dict]:
    limbs = [_ply_coverage(xyz, images, derived, "full union")]
    if args.ply_first_n is not None:
        first_n = int(args.ply_first_n)
        if first_n <= 0 or first_n > xyz.shape[0]:
            raise ContractError(
                f"--ply-first-n {first_n} is outside 1..{xyz.shape[0]}"
            )
        limbs.append(_ply_coverage(xyz[:first_n], images, derived,
                                   f"leading {first_n} points"))
    return limbs


def cmd_verify(args) -> dict:
    # The instrument certifies itself in the SAME cell that produces the
    # number, so the manifest never carries a gate result whose backend
    # was never checked.
    self_test = cmd_self_test(args)

    model_dir = Path(args.model)
    cameras, images, points, hashes = _read_model(model_dir)
    scale = float(args.scale)
    derived = _prepare_cameras(cameras, scale)
    described = _describe_cameras(derived, scale)

    print(f"[imvid] model {model_dir}: {len(cameras)} camera(s), "
          f"{len(images)} image(s), {len(points)} point(s)", flush=True)
    _print_cameras(described)

    if args.backend != "cv2":
        print(f"[imvid] WARNING: undistortion backend is {args.backend!r}, NOT "
              "the cv2 path the record specifies. This is the plumbing "
              "rehearsal, not the record's gate; the manifest says so.",
              flush=True)
    limb = _residual_limb(images, points, derived, scale, backend=args.backend)

    payload = {
        "mode": "verify",
        "model_dir": str(model_dir),
        "model_sha256": hashes,
        "scale": scale,
        "cameras": described,
        "reprojection": limb,
        "self_test": self_test,
        "refusals": list(limb["refusals"]),
    }
    if not limb["gate"]["passed"]:
        payload["refusals"].append(
            f"GATE FAILED: mean residual {limb['gate']['value_px']:.6f} px on "
            f"{limb['gate']['raster']} exceeds "
            f"{GATE_MEAN_REPROJ_PX_NATIVE} px. This is the kill step: nothing "
            "downstream of the OPENCV -> PINHOLE conversion should run."
        )

    print(f"[imvid] pairs {limb['pairs']} over {limb['images_with_pairs']}/"
          f"{limb['images_total']} images", flush=True)
    print("[imvid] RESIDUAL  A = pi(K_new @ (R X + t))   "
          f"B = undistort(obs, K, dist, P=K_new) [backend={args.backend}]",
          flush=True)
    for key in ("scaled", "native_equivalent"):
        stats = limb["residual"][key]
        print(f"[imvid]   {stats['raster']}: mean {stats['mean']:.6f}  "
              f"median {stats['median']:.6f}  p99 {stats['p99']:.6f}  "
              f"max {stats['max']:.6f}", flush=True)
    cross = limb["native_distorted_crosscheck"]
    print(f"[imvid] CROSS-CHECK (COLMAP's own statistic, {cross['raster']}): "
          f"mean {cross['mean']:.6f}  median {cross['median']:.6f}  "
          f"max {cross['max']:.6f}", flush=True)
    print("[imvid] PER-CAMERA MEAN RESIDUAL "
          "(px @ NATIVE, = scaled / scale):", flush=True)
    for name, entry in sorted(limb["per_camera"].items()):
        if not entry["pairs"]:
            print(f"[imvid]     {name:>14s}  pairs 0  <-- NO OBSERVATIONS",
                  flush=True)
            continue
        print(f"[imvid]     {name:>14s}  pairs {entry['pairs']:>6d}  "
              f"mean {entry['mean_px_native_equivalent']:.6f}  "
              f"max {entry['max_px_native_equivalent']:.6f}", flush=True)
    gate = limb["gate"]
    print(f"[imvid] GATE  mean <= {gate['threshold_px']:.4f} px @ "
          f"{gate['raster']}  ->  "
          f"{'PASS' if gate['passed'] else 'FAIL'} "
          f"({gate['value_px']:.6f} px)", flush=True)
    print(f"[imvid]   the SAME statistic reads {gate['scaled_raster_value_px']:.6f} "
          f"px on {gate['scaled_raster']}; {gate['trap']}", flush=True)

    if args.ply is not None:
        xyz, ply_info = _load_ply(args)
        payload["ply"] = ply_info
        payload["ply_coverage"] = _ply_limbs(args, xyz, images, derived)
        payload["ply_note"] = PLY_UNION_NOTE
        print(f"[imvid] PLY {ply_info['path']} sha256 {ply_info['sha256'][:16]}...",
              flush=True)
        for entry in payload["ply_coverage"]:
            _print_ply_limb(entry)
        print(f"[imvid] NOTE: {PLY_UNION_NOTE}", flush=True)
    return payload


def cmd_project_ply(args) -> dict:
    if args.ply is None:
        raise ContractError("--ply is required in project-ply mode")
    model_dir = Path(args.model)
    cameras, images, points, hashes = _read_model(model_dir)
    scale = float(args.scale)
    derived = _prepare_cameras(cameras, scale)
    described = _describe_cameras(derived, scale)
    xyz, ply_info = _load_ply(args)

    print(f"[imvid] model {model_dir}: {len(cameras)} camera(s), "
          f"{len(images)} image(s)", flush=True)
    _print_cameras(described)
    print(f"[imvid] PLY {ply_info['path']} sha256 {ply_info['sha256'][:16]}...",
          flush=True)

    limbs = _ply_limbs(args, xyz, images, derived)
    for entry in limbs:
        _print_ply_limb(entry)
    print(f"[imvid] NOTE: {PLY_UNION_NOTE}", flush=True)
    return {"mode": "project-ply", "model_dir": str(model_dir),
            "model_sha256": hashes, "scale": scale, "cameras": described,
            "ply": ply_info, "ply_coverage": limbs, "ply_note": PLY_UNION_NOTE,
            "refusals": []}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True,
                        choices=("verify", "project-ply", "self-test"))
    parser.add_argument("--model", default=None,
                        help="a COLMAP BINARY model directory (model_out/), READ ONLY")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="declared output scale of the undistorted raster")
    parser.add_argument("--ply", default=None,
                        help="point cloud for the coverage limb")
    parser.add_argument("--ply-first-n", type=int, default=None,
                        help="also report coverage for the leading N points of "
                             "--ply (the union is concatenated in ascending "
                             "frame order, so these are the first frame's cloud)")
    parser.add_argument("--backend", default="cv2", choices=("cv2", "numpy"),
                        help="undistortion backend. 'cv2' is the production "
                             "path the record specifies and is the DEFAULT; "
                             "'numpy' exists only so the model plumbing can be "
                             "rehearsed where cv2 is absent, and the manifest "
                             "records which one produced the number")
    parser.add_argument("--out", default=None, help="manifest path; outside the repo")
    args = parser.parse_args(argv)

    if args.mode in ("verify", "project-ply"):
        if args.model is None:
            raise ContractError(f"--model is required in {args.mode} mode")
        if args.out is None:
            raise ContractError(f"--out is required in {args.mode} mode")
    if args.scale <= 0.0:
        raise ContractError("--scale must be positive")
    if args.out is not None:
        out = Path(args.out).resolve()
        if out == REPO_ROOT or REPO_ROOT in out.parents:
            raise ContractError(f"--out {out} is inside the repository")

    if args.mode == "self-test":
        payload = cmd_self_test(args)
    elif args.mode == "verify":
        payload = cmd_verify(args)
    else:
        payload = cmd_project_ply(args)

    payload["schema_version"] = SCHEMA
    payload["gate_declaration"] = {
        "statistic": "mean reprojection residual",
        "threshold_px": GATE_MEAN_REPROJ_PX_NATIVE,
        "raster": "NATIVE 5312x2988",
        "source": "imvid-baseline-freeze.md:322-323 (2 px, met at 1.1953 px "
                  "on frame 0); same bar at "
                  "dataset-admission-matrix-2026-08-18.md:145",
        "rule": (
            "a scaled-raster residual is NEVER compared against this "
            "threshold; divide by the scale first"
        ),
    }
    if args.out is not None:
        body = json.dumps(payload, allow_nan=False, sort_keys=True, indent=1)
        atomic_write_bytes(Path(args.out), body.encode("utf-8") + b"\n")
        print(f"[imvid] manifest -> {args.out}", flush=True)

    refusals = payload.get("refusals") or []
    if refusals:
        raise ContractError(
            "REFUSED after writing the manifest so the evidence survives: "
            + " | ".join(refusals)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
