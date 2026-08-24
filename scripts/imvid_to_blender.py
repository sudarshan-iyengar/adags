#!/usr/bin/env python3
"""ImViD -> ADAGS Blender-convention scene converter (undistort + transforms).

THE ONE JOB. `scene/dataset_readers.py::readCamerasFromTransforms` is the
route this project actually uses (N3V, DiVa-360) and the only one
structurally compatible with ImViD's four-camera split and shared
`CAMERA_ID`. It reads `fl_x / fl_y / cx / cy` straight out of JSON with
**no camera-model field, no distortion field and no check of any kind**
(`scene/dataset_readers.py:433-451`) -- so pinhole intrinsics paired with
still-distorted ImViD frames train and evaluate SILENTLY, wrong by a
median of 14.72 px and a maximum of 90.53 px
([[operations/imvid-baseline-freeze]] A3). Both halves of that mistake
already sit adjacent on Apollo: experiment 156 wrote a derived PINHOLE
`cameras.txt` while every decoded frame remains in the supplied OPENCV
frame.

**This script exists to make the two halves consistent and to make the
inconsistency impossible to produce by accident.** The pixels and the
intrinsics are produced by ONE `K_new` object in the SAME call: the
matrix handed to `cv2.initUndistortRectifyMap(newCameraMatrix=...)` is
the matrix whose floats are written into `transforms_*.json`. That
identity -- one matrix, two consumers -- is the whole correctness
argument, and `--self-test` asserts it.

--------------------------------------------------------------------
THE READER CONVENTION THIS FILE TARGETS, QUOTED
--------------------------------------------------------------------

`scene/dataset_readers.py:370-397` (pose + time + path):

    370        with open(os.path.join(path, transformsfile)) as json_file:
    371            contents = json.load(json_file)
    375        frames = contents["frames"]
    380            timestamp = frame.get('time', 0.0)
    387            cam_name = os.path.join(path, frame["file_path"] + extension)
    390            c2w = np.array(frame["transform_matrix"])
    392            c2w[:3, 1:3] *= -1
    395            w2c = np.linalg.inv(c2w)
    396            R = np.transpose(w2c[:3,:3])
    397            T = w2c[:3, 3]

`scene/dataset_readers.py:433-441` (the PER-FRAME intrinsics branch this
converter deliberately triggers):

    433            if 'fl_x' in frame and 'fl_y' in frame and 'cx' in frame and 'cy' in frame:
    434                FovX = FovY = -1.0
    435                fl_x = frame['fl_x']
    ...
    439                return CameraInfo(uid=idx, R=R, T=T, ... fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, far=far)

Consequences encoded here, each verified by `--self-test`:

* `file_path` carries NO extension; the reader appends `extension`
  (default `".png"`, `configs/n3v/ladder_b0_crb.yaml:31`).
* `transform_matrix` is CAMERA-TO-WORLD in the OpenGL/Blender convention
  (Y up, Z back). The reader negates columns 1 and 2 to recover COLMAP's
  camera-to-world, so this file writes COLMAP's `[R_cw^T | C]` with those
  same two columns already negated. Round-tripped in the self-test AND
  re-verified against every WRITTEN frame at conversion time.
* `time` is read verbatim (`:380`) -- this is the ONE rate-agnostic seam
  in the codebase. The COLMAP/Technicolor reader hard-codes `/30`
  (`scene/dataset_readers.py:700`) and the PanopticSports reader hard-codes
  `/30.0` (`:200`); for ImViD's measured `60000/1001` that is wrong by
  `(1/30)/(1001/60000) = 2000/1001 = 1.998001998...`
  ([[operations/imvid-baseline-freeze]] B12). This converter takes the
  rate as an EXACT RATIONAL on the command line and, when the decode
  manifest is present, refuses to proceed unless the declared rational
  equals the `r_frame_rate` ffprobe measured on EVERY camera.
* The point cloud must be named `points3d.ply` EXACTLY
  (`scene/dataset_readers.py:481`). A mis-named cloud is silently replaced
  by a uniform random fill in [-1.3, 1.3]^3 with NO error raised
  (`:481-491`), while ImViD Opera's content spans roughly x -35..34.
  The artifact is `points3d_colmap_union.ply`, so the rename is the
  whole hazard; this script copies it byte-identically and checks the
  copy's sha256 against the source.
* The output directory must NOT contain a `sparse/` subdirectory:
  `scene/__init__.py:50` dispatches on `sparse/` BEFORE
  `transforms_train.json` at `:56`, so a stray `sparse/` silently routes
  the scene into the structurally incompatible COLMAP path.

--------------------------------------------------------------------
UNDISTORTION -- WHAT IS DONE AND WHAT IS DECLARED
--------------------------------------------------------------------

`cv2.initUndistortRectifyMap` + `cv2.remap`, NOT `cv2.undistort`:
`cv2.undistort` cannot change raster size, so the declared 2x downscale
would become a SECOND resampling and a SECOND, separately-argued
intrinsic rescale -- exactly where the `(c + 0.5) * s - 0.5` convention
gets lost. It also rebuilds the map on every call, and it does not hand
back the maps, which is what makes the invalid-border measurement
possible at all.

`newCameraMatrix` is **the scaled original K**, never
`getOptimalNewCameraMatrix`. Stated explicitly because it changes the
derived intrinsics: the frozen experiment-156 camera is
`PINHOLE 2656 1494 fx 1301.66634323002 fy 1301.1218300301398
cx 1327.75 cy 746.75`, which is exactly `K` scaled by 0.5 under the
pixel-centre convention. `getOptimalNewCameraMatrix` produces a
different camera as a function of `alpha` and the OpenCV version's ROI
solver, silently contradicting a frozen, hashed record.
`--new-camera-matrix optimal` is accepted by argparse only so the
refusal names the reason.

`R = np.eye(3)`: no rectification rotation, so the world-to-camera
transform `(R_i, t_i)` is bit-identical before and after and the
existing 20,157-point sparse union is reusable UNCHANGED
([[operations/imvid-baseline-freeze]] B6, now backed by the passing
reprojection gate in B11/B11.1).

**The camera model is READ FROM DATA and branched on, never assumed.**
`OPENCV` -> undistort. A distortion-free `PINHOLE` (what Meeting and
Playing ship, `dataset-admission-matrix-2026-08-18.md:424-455`) -> the
same map with an all-zero `distCoeffs`, i.e. a pure scale, and
`--require-undistortion` REFUSES such a scene loudly. Applying Opera's
distortion to an already-rectified scene crashes nothing: it warps every
feature by ~14.7 px median while poses and intrinsics stay correct, and
the only symptom is a degraded number nobody can attribute.

--------------------------------------------------------------------
FAIL-CLOSED
--------------------------------------------------------------------

Refused immediately: an unhandled camera model; a camera id referenced
by an image but absent from `cameras.*`; a source image whose decoded
size is not the camera's declared native raster; a source image carrying
an ALPHA channel (the Blender reader would composite it,
`scene/dataset_readers.py:404-414`, against the frozen full-frame
no-alpha convention); a held-out camera missing from the model; a
held-out camera name appearing anywhere in the WRITTEN training
artifacts; an output PNG whose IHDR is not the declared output raster;
a `points3d.ply` copy whose sha256 differs from the source; an existing
destination `points3d.ply` with different content; a `sparse/`
subdirectory under the output root; an output path inside the git
repository; a declared frame rate that disagrees with the measured
`r_frame_rate`.

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------

  python3 scripts/imvid_to_blender.py --mode self-test

  python3 scripts/imvid_to_blender.py --mode convert \\
      --model       /apollo/users/sri/proj_adags/data/imvid/scene1_opera \\
      --frames-root /apollo/users/sri/proj_adags/data/imvid/frames \\
      --out         /apollo/users/sri/proj_adags/data/imvid/blender35_s050 \\
      --scale 0.5 --fps-rational 60000/1001 \\
      --ply /apollo/users/sri/proj_adags/data/imvid/init35/points3d_colmap_union.ply \\
      --manifest /apollo/users/sri/proj_adags/runs/elgs/<run>/MANIFEST.imvid_to_blender.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from depth_visibility.artifacts import atomic_write_bytes  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402

# The OPENCV -> PINHOLE conventions are NOT re-derived here. They are
# imported from the instrument that measured the reprojection gate
# (Determined 270/271/272, 1.215442 / 1.162289 / 1.213650 px at NATIVE
# against a 2.0 px NATIVE gate), so the camera this converter writes and
# the camera that gate validated are produced by the SAME code.
import imvid_verify_pinhole as vp  # noqa: E402

SCHEMA = "imvid-to-blender-v1"

#: The frozen split ([[operations/imvid-baseline-freeze]]:35-46), chosen
#: outcome-blind as `np.linspace(0, 38, 4)` over the sorted camera ids.
#: NOT exposed on the command line: it is frozen, and a CLI knob is how a
#: frozen split gets changed by accident.
HELD_OUT = ("cam00", "cam13", "cam25", "cam38")

#: `scene/dataset_readers.py:481` -- this exact basename or the reader
#: silently substitutes a random uniform cloud (`:481-491`).
POINTCLOUD_BASENAME = "points3d.ply"

TRAIN_JSON = "transforms_train.json"
TEST_JSON = "transforms_test.json"

#: `readNerfSyntheticInfo` reads `transforms_test.json` for every scene
#: whose path does not end in 'lego' (`scene/dataset_readers.py:473`).
DEFAULT_EXTENSION = ".png"

#: ImViD's MEASURED container rate, 39/39 videos
#: ([[operations/imvid-sample-ingestion]]:239-254, re-measured on the full
#: take at [[operations/imvid-baseline-freeze]] B10). Used ONLY by the
#: self-test and as the value the decode manifest is checked against --
#: `--fps-rational` is required on the command line so no run inherits a
#: constant from this file.
IMVID_MEASURED_FPS = Fraction(60000, 1001)

#: The N3V hard-coded period every config in this repo carries
#: (`motion_track_dt: 0.0333333333`), quoted so the self-test can state
#: the ratio rather than assert it from memory.
N3V_HARDCODED_FPS = Fraction(30, 1)

#: `image_name` convention. `scene/packet_birth_flow.py:114` parses
#: `^cam(\\d+)_(\\d+)$`, and this flat naming also makes every
#: `image_name` globally unique across (camera, frame), which the
#: `undist/camNN/<stem>` layout does not.
FRAME_INDEX_WIDTH = 6
IMAGES_SUBDIR = "images"

#: Equality required against the frozen experiment-156 PINHOLE line. Not a
#: tuning tolerance -- the two derivations are the same closed formula on
#: the same decimal inputs.
K_NEW_TOL = 1e-9

#: The written JSON must reproduce the model's own (R_cw^T, t) when run
#: through the reader's arithmetic. Pure float64 matrix algebra; a real
#: convention error lands in the tens of pixels, not at 1e-12.
POSE_ROUNDTRIP_TOL = 1e-12

_SUPPORTED_MODELS = ("OPENCV", "PINHOLE")


# ====================================================================
# Small utilities
# ====================================================================


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    """(width, height) straight from the PNG IHDR -- no image library.

    Same fail-closed shape as `scripts/imvid_decode_frames.py:120-130`.
    """
    with path.open("rb") as handle:
        header = handle.read(33)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ContractError(f"{path} is not a PNG")
    return (int.from_bytes(header[16:20], "big"),
            int.from_bytes(header[20:24], "big"))


def parse_rational(text: str) -> Fraction:
    """`"60000/1001"` -> Fraction. Refuses anything that is not NUM/DEN.

    A bare float is refused deliberately: the whole point of carrying the
    rate as a rational is that `59.94` and `60000/1001` are different
    numbers and the difference must not be introduced by a decimal
    literal.
    """
    parts = str(text).strip().split("/")
    if len(parts) != 2:
        raise ContractError(
            f"--fps-rational must be given as NUM/DEN (e.g. 60000/1001); got "
            f"{text!r}. A decimal is refused on purpose: the frame period is "
            "carried exactly, never as a rounded literal."
        )
    try:
        num, den = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ContractError(f"--fps-rational {text!r} is not NUM/DEN: {exc}") from exc
    if num <= 0 or den <= 0:
        raise ContractError(f"--fps-rational {text!r} must be positive")
    return Fraction(num, den)


def camera_token(image_name: str) -> str:
    """`cam07.png` / `cam07` -> `cam07`."""
    return Path(str(image_name)).stem


def frame_stem(camera: str, frame_index: int) -> str:
    return f"{camera}_{int(frame_index):0{FRAME_INDEX_WIDTH}d}"


# ====================================================================
# COLMAP model -- binary (reused, certified) or supplied text
# ====================================================================


def parse_cameras_txt(text: str) -> dict[int, dict]:
    """Supplied `cameras.txt` -> the same dict shape as `cameras.bin`.

    The SUPPLIED ImViD cameras.txt carries full double precision
    (`2603.3326864600399`). A COLMAP-EXPORTED text model does NOT --
    COLMAP 3.6's writer emits about six significant figures
    (`scripts/imvid_verify_pinhole.py:212-219`), which is a 3.66e-03 error
    on a focal length. This reader cannot tell the two apart, so
    `--model-format text` is recorded in the manifest as a declared
    reduction in provenance quality, and `--cross-check-model` exists to
    settle it against a binary model.
    """
    cameras: dict[int, dict] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise ContractError(f"malformed cameras.txt row: {stripped!r}")
        camera_id = int(parts[0])
        if camera_id in cameras:
            raise ContractError(f"duplicate CAMERA_ID {camera_id} in cameras.txt")
        cameras[camera_id] = {
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": [float(v) for v in parts[4:]],
        }
    if not cameras:
        raise ContractError("cameras.txt holds no camera rows")
    return cameras


def parse_images_txt(text: str) -> dict[str, dict]:
    """Supplied `images.txt` -> {name: {image_id, camera_id, qvec, tvec}}.

    Same header layout `scripts/imvid_pilot_prepare.py:75-89` parses:
    `IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`, followed by a
    POINTS2D line this converter does not need.
    """
    images: dict[str, dict] = {}
    body = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    index = 0
    while index < len(body):
        parts = body[index].split()
        if len(parts) >= 10:
            name = parts[9]
            if name in images:
                raise ContractError(f"duplicate image name in images.txt: {name!r}")
            images[name] = {
                "image_id": int(parts[0]),
                "qvec": [float(v) for v in parts[1:5]],
                "tvec": [float(v) for v in parts[5:8]],
                "camera_id": int(parts[8]),
            }
            index += 2  # skip the POINTS2D line
        else:
            index += 1
    if not images:
        raise ContractError("images.txt holds no image entries")
    return images


def read_model(model_dir: Path, model_format: str) -> tuple[dict, dict, dict]:
    """(cameras, images, provenance). `auto` prefers the binary model."""
    binary = [model_dir / n for n in ("cameras.bin", "images.bin")]
    text = [model_dir / n for n in ("cameras.txt", "images.txt")]
    have_binary = all(p.is_file() for p in binary)
    have_text = all(p.is_file() for p in text)

    if model_format == "binary" and not have_binary:
        raise ContractError(f"{model_dir}: --model-format binary but cameras.bin/images.bin are missing")
    if model_format == "text" and not have_text:
        raise ContractError(f"{model_dir}: --model-format text but cameras.txt/images.txt are missing")
    if model_format == "auto":
        if have_binary:
            model_format = "binary"
        elif have_text:
            model_format = "text"
        else:
            raise ContractError(
                f"{model_dir} is not a COLMAP model: no cameras.bin/images.bin and "
                "no cameras.txt/images.txt. Point --model at the directory that "
                "carries the SUPPLIED calibration (all cameras, both splits), not "
                "at a 35-camera training subset."
            )

    if model_format == "binary":
        cameras = vp.parse_cameras_bin(binary[0].read_bytes())
        raw_images = vp.parse_images_bin(binary[1].read_bytes())
        images = {name: {"image_id": entry["image_id"],
                         "camera_id": entry["camera_id"],
                         "qvec": entry["qvec"], "tvec": entry["tvec"]}
                  for name, entry in raw_images.items()}
        files = binary
    else:
        cameras = parse_cameras_txt(text[0].read_text(encoding="utf-8"))
        images = parse_images_txt(text[1].read_text(encoding="utf-8"))
        files = text

    provenance = {
        "model_dir": str(model_dir),
        "format": model_format,
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                  for p in files},
        "n_cameras": len(cameras),
        "n_images": len(images),
    }
    if model_format == "text":
        provenance["precision_note"] = (
            "TEXT model. The SUPPLIED ImViD cameras.txt carries full double "
            "precision, but a COLMAP-EXPORTED text model is truncated to about "
            "six significant figures and this reader cannot distinguish them. "
            "Use --cross-check-model against a binary model to settle it."
        )
    return cameras, images, provenance


# ====================================================================
# Geometry: the Blender-convention camera-to-world matrix
# ====================================================================


def blender_transform_matrix(qvec, tvec) -> np.ndarray:
    """COLMAP world-to-camera (qvec, tvec) -> the reader's `transform_matrix`.

    COLMAP stores world-to-camera: `x_cam = R_cw x_world + t`. So
    camera-to-world is `[[R_cw^T, -R_cw^T t], [0, 1]]`. The reader then
    applies `c2w[:3, 1:3] *= -1` (`scene/dataset_readers.py:392`) to get
    back to COLMAP axes, so what is WRITTEN is that same matrix with
    columns 1 and 2 already negated.
    """
    R_cw = vp.qvec2rotmat(qvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_cw.T
    c2w[:3, 3] = -R_cw.T @ t
    c2w[:3, 1:3] *= -1.0  # inverse of the reader's own flip
    return c2w


def reader_pose_arithmetic(transform_matrix) -> tuple[np.ndarray, np.ndarray]:
    """`scene/dataset_readers.py:390-397`, verbatim, for verification.

        390            c2w = np.array(frame["transform_matrix"])
        392            c2w[:3, 1:3] *= -1
        395            w2c = np.linalg.inv(c2w)
        396            R = np.transpose(w2c[:3,:3])
        397            T = w2c[:3, 3]
    """
    c2w = np.array(transform_matrix, dtype=np.float64)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]
    return R, T


# ====================================================================
# Cameras: one K_new, two consumers
# ====================================================================


def derive_output_camera(camera: dict, scale: float) -> dict:
    """K, distCoeffs and the SINGLE K_new that both the resampling map and
    the transforms JSON are built from.

    Delegates to `scripts/imvid_verify_pinhole.py::camera_matrices` and
    `::derive_pinhole` so the camera written here is produced by the same
    code that produced the camera the reprojection gate validated.
    """
    model = camera["model"]
    if model not in _SUPPORTED_MODELS:
        raise ContractError(
            f"camera model {model!r} is not handled; this converter supports "
            f"{_SUPPORTED_MODELS}. Refusing rather than guessing a parameter "
            "order -- a wrong distCoeffs order measures 56.77 px "
            "([[operations/imvid-baseline-freeze]] B11)."
        )
    K, dist = vp.camera_matrices(camera)
    pin = vp.derive_pinhole(K, camera["width"], camera["height"], scale)
    distorted = bool(np.any(dist != 0.0))
    return {
        "model": model,
        "K": K,
        "dist": dist,
        "is_distorted": distorted,
        "K_new": pin["K_new"],
        "width": pin["width"],
        "height": pin["height"],
        "native_width": int(camera["width"]),
        "native_height": int(camera["height"]),
    }


def describe_camera(entry: dict, scale: float) -> dict:
    """Every intrinsic BEFORE and AFTER, plus the frozen-record comparison."""
    K, K_new = entry["K"], entry["K_new"]
    recorded = vp.RECORDED_PINHOLE_AT_SCALE_HALF
    matches_frozen = bool(
        float(scale) == 0.5
        and entry["model"] == "OPENCV"
        and abs(K_new[0, 0] - recorded["fx"]) <= K_NEW_TOL
        and abs(K_new[1, 1] - recorded["fy"]) <= K_NEW_TOL
        and abs(K_new[0, 2] - recorded["cx"]) <= K_NEW_TOL
        and abs(K_new[1, 2] - recorded["cy"]) <= K_NEW_TOL
        and (entry["width"], entry["height"]) == (recorded["width"], recorded["height"])
    )
    return {
        "source": {
            "model": entry["model"],
            "raster": f"{entry['native_width']}x{entry['native_height']}",
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "dist_k1_k2_p1_p2": [float(v) for v in entry["dist"]],
            "is_distorted": entry["is_distorted"],
        },
        "derived_pinhole": {
            "model": "PINHOLE",
            "raster": f"{entry['width']}x{entry['height']}",
            "width": entry["width"], "height": entry["height"],
            "fx": float(K_new[0, 0]), "fy": float(K_new[1, 1]),
            "cx": float(K_new[0, 2]), "cy": float(K_new[1, 2]),
        },
        "scale": float(scale),
        "transform": {
            "focal": "f * scale",
            "principal_point": "(c + 0.5) * scale - 0.5",
            "raster": "round(native * scale)",
            "why": (
                "COLMAP puts pixel centres at integer coordinates. The naive "
                "c * scale is a quarter pixel off in both axes on every camera "
                "and survives every downstream check "
                "([[operations/imvid-baseline-freeze]]:239-242)."
            ),
        },
        "matches_frozen_experiment_156_pinhole": matches_frozen,
    }


# ====================================================================
# Undistortion maps, the invalid border, and the remap
# ====================================================================


def build_maps(entry: dict, new_camera_matrix: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """`cv2.initUndistortRectifyMap` with the arguments recorded verbatim.

    The `newCameraMatrix` choice is validated BEFORE cv2 is imported, so
    the refusal below holds on a workstation where cv2 is absent and
    `--self-test` can exercise it.
    """
    if new_camera_matrix == "optimal":
        raise ContractError(
            "--new-camera-matrix optimal is refused. getOptimalNewCameraMatrix "
            "produces a DIFFERENT camera as a function of alpha and the OpenCV "
            "version's ROI solver, so it would silently supersede the frozen, "
            "hashed experiment-156 PINHOLE line (fx 1301.66634323002, "
            "fy 1301.1218300301398, cx 1327.75, cy 746.75) that the passing "
            "reprojection gate was measured against. Changing it is a freeze "
            "amendment, not a flag."
        )
    if new_camera_matrix != "scaled_k":
        raise ContractError(f"unknown --new-camera-matrix {new_camera_matrix!r}")

    import cv2

    K = np.ascontiguousarray(entry["K"], dtype=np.float64)
    dist = np.ascontiguousarray(entry["dist"], dtype=np.float64)
    K_new = np.ascontiguousarray(entry["K_new"], dtype=np.float64)
    # LOAD-BEARING: identity rectification. The world-to-camera transform
    # (R_i, t_i) is then bit-identical before and after, which is exactly
    # why the existing sparse union is reusable unchanged.
    R_rect = np.eye(3, dtype=np.float64)
    size = (int(entry["width"]), int(entry["height"]))  # (w, h) of the OUTPUT

    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, R_rect, K_new, size, cv2.CV_32FC1
    )

    record = {
        "call": "cv2.initUndistortRectifyMap",
        "cv2_version": cv2.__version__,
        "arguments": {
            "cameraMatrix": [[float(v) for v in row] for row in K],
            "distCoeffs": [float(v) for v in dist],
            "distCoeffs_order": "(k1, k2, p1, p2) -- length 4, COLMAP OPENCV order exactly",
            "R": "np.eye(3)  # no rectification rotation; poses untouched",
            "newCameraMatrix": [[float(v) for v in row] for row in K_new],
            "newCameraMatrix_choice": "scaled_k",
            "newCameraMatrix_choice_meaning": (
                "the ORIGINAL K scaled by the declared factor under "
                "(c + 0.5) * s - 0.5; NOT a getOptimalNewCameraMatrix result"
            ),
            "size_w_h": list(size),
            "m1type": "cv2.CV_32FC1",
        },
        "newCameraMatrix_is_the_matrix_written_to_transforms_json": True,
    }
    return map1, map2, record


def _trim_to_all_valid(valid: np.ndarray) -> dict:
    """How much must come off each edge to leave an all-valid rectangle.

    Edge trimming, greedily removing whichever of the four boundary lines
    carries the most invalid pixels. Exact for a periphery-only invalid
    region, which is what barrel undistortion produces; if all four
    boundary lines are already clean while the interior is not, edge
    trimming cannot help and that is reported rather than papered over.

    A naive per-row/per-column "leading invalid depth" is WRONG here and
    was caught by the self-test: a column that is invalid end-to-end makes
    the top and bottom depths read the full height, so the reported crop
    came back with a negative dimension.
    """
    height, width = valid.shape
    top, bottom, left, right = 0, 0, 0, 0
    while True:
        y0, y1 = top, height - bottom
        x0, x1 = left, width - right
        if y1 <= y0 or x1 <= x0:
            return {"left": left, "right": right, "top": top, "bottom": bottom,
                    "width": max(0, x1 - x0), "height": max(0, y1 - y0),
                    "all_valid": False}
        window = valid[y0:y1, x0:x1]
        if bool(window.all()):
            return {"left": left, "right": right, "top": top, "bottom": bottom,
                    "width": int(x1 - x0), "height": int(y1 - y0),
                    "all_valid": True}
        counts = {
            "top": int((~window[0, :]).sum()),
            "bottom": int((~window[-1, :]).sum()),
            "left": int((~window[:, 0]).sum()),
            "right": int((~window[:, -1]).sum()),
        }
        worst = max(counts, key=lambda k: (counts[k], k))
        if counts[worst] == 0:
            # Clean boundary, dirty interior: not an edge-trim problem.
            return {"left": left, "right": right, "top": top, "bottom": bottom,
                    "width": int(x1 - x0), "height": int(y1 - y0),
                    "all_valid": False}
        if worst == "top":
            top += 1
        elif worst == "bottom":
            bottom += 1
        elif worst == "left":
            left += 1
        else:
            right += 1


def measure_invalid_border(map1: np.ndarray, map2: np.ndarray,
                           src_w: int, src_h: int) -> dict:
    """What fraction of the OUTPUT raster samples outside the source, and
    what fraction of the SOURCE the output actually covers.

    THE SIGN, WORKED OUT RATHER THAN ASSUMED. `initUndistortRectifyMap`
    builds an INVERSE map: an output (undistorted) pixel at normalized
    radius `r` is fetched from source radius `r * (1 + k1 r^2 + k2 r^4)`.
    Opera's `k1 = -0.0245...` is negative, so that factor is BELOW 1 and
    the map reaches INWARD. On this camera the expected invalid fraction
    is therefore ~0 and the real effect is the opposite one: peripheral
    SOURCE content is discarded, i.e. undistortion CROPS rather than
    letterboxes. The prose that says a negative `k1` "pushes the periphery
    outward and corner pixels sample outside the source" describes the
    FORWARD map, not the map cv2 builds. Both quantities are reported here
    and neither is assumed -- a non-zero invalid fraction on a `k1 < 0`
    camera would itself be a signal worth stopping for.

    Why either number matters: [[operations/imvid-baseline-freeze]]:75-78
    freezes "NO alpha compositing and NO black-background convention here.
    Full-frame metrics only". Any invalid pixel enters the training loss
    AND the held-out PSNR as an easy constant-black target the model
    learns perfectly, inflating PSNR by a term proportional to the invalid
    fraction -- identical for every arm (harmless for an A/B), NOT harmless
    for the "33.5 dB parity with STG" style of cross-paper comparison this
    lane exists to support. And a source-coverage well below 1.0 means the
    evaluated field of view is not the shipped one, which any comparison
    against ImViD's own numbers has to state.

    Reported, NEVER cropped. Cropping changes w, h, cx, cy and therefore
    supersedes the frozen experiment-156 camera; that is a freeze
    amendment and needs a decision, not a default.
    """
    valid = ((map1 >= 0.0) & (map1 <= float(src_w - 1))
             & (map2 >= 0.0) & (map2 <= float(src_h - 1)))
    # Fully outside the bilinear support: BOTH taps are borderValue, so
    # the output pixel is EXACTLY borderValue. This is the subset an
    # output image can be checked against without a false alarm on the
    # one-pixel blend band.
    fully_outside = ((map1 <= -1.0) | (map1 >= float(src_w))
                     | (map2 <= -1.0) | (map2 >= float(src_h)))

    height, width = valid.shape
    trim = _trim_to_all_valid(valid)

    # The complementary quantity: how much of the SOURCE raster the output
    # actually reaches. For an inward (barrel) map this, not the invalid
    # fraction, is where the loss of information shows up.
    x0, x1 = float(map1.min()), float(map1.max())
    y0, y1 = float(map2.min()), float(map2.max())
    covered_w = max(0.0, min(x1, src_w - 1.0) - max(x0, 0.0))
    covered_h = max(0.0, min(y1, src_h - 1.0) - max(y0, 0.0))

    return {
        "definition": "valid = (0 <= map1 <= W_src-1) and (0 <= map2 <= H_src-1)",
        "output_raster": f"{width}x{height}",
        "source_raster": f"{src_w}x{src_h}",
        "invalid_pixels": int((~valid).sum()),
        "total_pixels": int(valid.size),
        "invalid_fraction": float(1.0 - valid.mean()),
        "fully_outside_bilinear_support_pixels": int(fully_outside.sum()),
        "fully_outside_fraction": float(fully_outside.mean()),
        "per_edge_trim_to_all_valid_px": {
            "left": trim["left"], "right": trim["right"],
            "top": trim["top"], "bottom": trim["bottom"],
        },
        "largest_all_valid_axis_aligned_crop": {
            "width": trim["width"], "height": trim["height"],
            "offset_x": trim["left"], "offset_y": trim["top"],
            "verified_all_valid": trim["all_valid"],
            "APPLIED": False,
        },
        "source_coverage": {
            "map_extent_x": [x0, x1],
            "map_extent_y": [y0, y1],
            "source_bbox_fraction_reached": float(
                (covered_w / max(1.0, src_w - 1.0)) * (covered_h / max(1.0, src_h - 1.0))),
            "meaning": (
                "the fraction of the SOURCE raster's area the output map reaches. "
                "Below 1.0 means peripheral source content is DISCARDED -- the "
                "evaluated field of view is not the shipped one, which is the "
                "direction an inward (k1 < 0) map fails in"
            ),
        },
        "policy": (
            "MEASURED AND REPORTED, NOT CROPPED. Cropping changes w, h, cx, cy "
            "and supersedes the frozen experiment-156 camera; emitting a valid "
            "mask contradicts the frozen full-frame convention. Both are freeze "
            "amendments. Declare this fraction alongside every metric until one "
            "is taken."
        ),
    }


def remap_image(src_path: Path, dst_path: Path, map1, map2,
                entry: dict) -> dict:
    """One `cv2.remap`, fail-closed on size, channel count and alpha."""
    import cv2

    image = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ContractError(f"cv2.imread returned None for {src_path}")
    if image.ndim != 3:
        raise ContractError(
            f"{src_path}: expected a 3-channel image, got shape {image.shape}"
        )
    if image.shape[2] == 4:
        raise ContractError(
            f"{src_path} carries an ALPHA channel. The Blender reader composites "
            "alpha over the background (scene/dataset_readers.py:404-414), which "
            "contradicts the frozen full-frame, no-alpha ImViD convention "
            "([[operations/imvid-baseline-freeze]]:75-78). Refusing rather than "
            "silently compositing."
        )
    if image.shape[2] != 3:
        raise ContractError(
            f"{src_path}: expected 3 channels, got {image.shape[2]}"
        )
    if (image.shape[1], image.shape[0]) != (entry["native_width"], entry["native_height"]):
        raise ContractError(
            f"{src_path} decodes {image.shape[1]}x{image.shape[0]} but its camera "
            f"declares {entry['native_width']}x{entry['native_height']}. The "
            "supplied intrinsics describe the declared raster; a silent rescale "
            "would put the pixels in a different frame from the calibration."
        )

    out = cv2.remap(image, map1, map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst_path), out):
        raise ContractError(f"cv2.imwrite failed for {dst_path}")
    written = png_size(dst_path)
    if written != (entry["width"], entry["height"]):
        raise ContractError(
            f"{dst_path} IHDR reads {written[0]}x{written[1]}, expected "
            f"{entry['width']}x{entry['height']}"
        )
    return {"array": out, "size": written}


def remap_call_record() -> dict:
    import cv2

    return {
        "call": "cv2.remap",
        "cv2_version": cv2.__version__,
        "arguments": {
            "map1": "the x-map from initUndistortRectifyMap (CV_32FC1)",
            "map2": "the y-map from initUndistortRectifyMap (CV_32FC1)",
            "interpolation": "cv2.INTER_LINEAR",
            "borderMode": "cv2.BORDER_CONSTANT",
            "borderValue": 0,
        },
        "why_not_cv2_undistort": (
            "cv2.undistort cannot change raster size, so the declared downscale "
            "would become a SECOND resampling with a SECOND separately-argued "
            "intrinsic rescale; it also rebuilds the map on every call and does "
            "not return the maps, which is what the invalid-border measurement "
            "is computed from."
        ),
    }


# ====================================================================
# Frame rate -- declared as a rational, checked against the measurement
# ====================================================================


def frame_time(frame_index: int, fps: Fraction) -> float:
    """`time(i) = i / fps`, evaluated exactly then rounded once."""
    return float(Fraction(int(frame_index)) / fps)


def verify_fps_against_decode_manifest(frames_root: Path, fps: Fraction,
                                       allow_unverified: bool) -> dict:
    """The declared rational must equal the ffprobe `r_frame_rate` on EVERY
    camera. `scripts/imvid_decode_frames.py` records it per camera in
    `MANIFEST.imvid_frames.json` at the frames root."""
    manifest_path = frames_root / "MANIFEST.imvid_frames.json"
    if not manifest_path.is_file():
        if not allow_unverified:
            raise ContractError(
                f"{manifest_path} is absent, so the declared frame rate "
                f"{fps.numerator}/{fps.denominator} cannot be checked against the "
                "measured stream rate. Pass --allow-unverified-fps to proceed "
                "with an UNVERIFIED rate, and expect that to be recorded as such."
            )
        return {"verified": False, "reason": f"{manifest_path} absent",
                "declared": f"{fps.numerator}/{fps.denominator}",
                "consequence": (
                    "the frame period was NOT checked against any measurement; "
                    "every timestamp in this scene rests on the command line "
                    "alone")}
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    probe = report.get("probe") or {}
    if not probe:
        raise ContractError(f"{manifest_path} carries no 'probe' block")
    observed: dict[str, str] = {}
    disagree: list[str] = []
    for camera, entry in sorted(probe.items()):
        raw = entry.get("r_frame_rate")
        observed[camera] = raw
        if raw is None:
            disagree.append(f"{camera}: no r_frame_rate recorded")
            continue
        try:
            measured = parse_rational(raw)
        except ContractError as exc:
            disagree.append(f"{camera}: unparseable r_frame_rate {raw!r} ({exc})")
            continue
        if measured != fps:
            disagree.append(f"{camera}: measured {raw}, declared "
                            f"{fps.numerator}/{fps.denominator}")
    if disagree:
        raise ContractError(
            "declared frame rate disagrees with the MEASURED stream rate on "
            f"{len(disagree)} camera(s): {disagree[:5]}. The frame period must "
            "come from the stream, never from a constant "
            "([[operations/imvid-baseline-freeze]] B4/B12)."
        )
    return {
        "verified": True,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "cameras_checked": len(observed),
        "declared": f"{fps.numerator}/{fps.denominator}",
        "measured_distinct": sorted(set(v for v in observed.values() if v)),
    }


def fps_record(fps: Fraction) -> dict:
    period = Fraction(1) / fps
    return {
        "rate_rational": f"{fps.numerator}/{fps.denominator}",
        "rate_float": float(fps),
        "frame_period_rational": f"{period.numerator}/{period.denominator}",
        "frame_period_float": float(period),
        "time_formula": "time(i) = i * frame_period, evaluated as an exact Fraction",
        "recommended_config_motion_track_dt": float(period),
        "ratio_against_the_repo_hardcoded_30fps": float(
            (Fraction(1) / N3V_HARDCODED_FPS) / period),
        "hardcoded_sites_this_converter_bypasses": [
            "scene/dataset_readers.py:200  timestamp = frame_idx / 30.0  (PanopticSports)",
            "scene/dataset_readers.py:700  timestamp=(timestamp-startime)/30  (Colmap/Technicolor)",
            "73 configs carrying motion_track_dt: 0.0333333333",
        ],
        "seam_used_instead": (
            "scene/dataset_readers.py:380  timestamp = frame.get('time', 0.0) "
            "-- the Blender reader reads whatever the JSON says"
        ),
    }


# ====================================================================
# The transforms payloads
# ====================================================================


def build_frame_entry(camera: str, frame_index: int, matrix: np.ndarray,
                      out_camera: dict, fps: Fraction) -> dict:
    """One `frames[]` entry in exactly the shape the reader parses.

    `file_path` carries NO extension: `scene/dataset_readers.py:387` does
    `frame["file_path"] + extension`. `fl_x/fl_y/cx/cy` are emitted PER
    FRAME so the `:433` branch is taken rather than the `:443` top-level
    one or the `:453` FOV fallback -- and they are read straight off the
    SAME `K_new` object the resampling map was built from.
    """
    K_new = out_camera["K_new"]
    return {
        "file_path": f"{IMAGES_SUBDIR}/{frame_stem(camera, frame_index)}",
        "transform_matrix": [[float(v) for v in row] for row in matrix],
        "time": frame_time(frame_index, fps),
        "fl_x": float(K_new[0, 0]),
        "fl_y": float(K_new[1, 1]),
        "cx": float(K_new[0, 2]),
        "cy": float(K_new[1, 2]),
        "w": int(out_camera["width"]),
        "h": int(out_camera["height"]),
        # Informational; the reader ignores unknown keys.
        "camera": camera,
        "frame_index": int(frame_index),
    }


def build_transforms(frames: list[dict], out_cameras: dict) -> dict:
    """The payload. A top-level intrinsic block is emitted ONLY when every
    camera shares one `K_new`; otherwise it is omitted rather than
    silently naming one camera's intrinsic as the scene's."""
    payload: dict = {}
    unique = {(round(float(e["K_new"][0, 0]), 12), round(float(e["K_new"][1, 1]), 12),
               round(float(e["K_new"][0, 2]), 12), round(float(e["K_new"][1, 2]), 12),
               e["width"], e["height"])
              for e in out_cameras.values()}
    if len(unique) == 1:
        entry = next(iter(out_cameras.values()))
        K_new = entry["K_new"]
        payload.update({
            "w": int(entry["width"]), "h": int(entry["height"]),
            "fl_x": float(K_new[0, 0]), "fl_y": float(K_new[1, 1]),
            "cx": float(K_new[0, 2]), "cy": float(K_new[1, 2]),
        })
    payload["camera_model"] = "PINHOLE"
    payload["frames"] = frames
    return payload


def dump_transforms(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, allow_nan=False, sort_keys=False)
            + "\n").encode("utf-8")


# ====================================================================
# Reader simulation -- used to verify the WRITTEN artifacts
# ====================================================================


def simulate_reader_frame(contents: dict, frame: dict, extension: str) -> dict:
    """`readCamerasFromTransforms`'s own key accesses, replayed.

    Reproduces `scene/dataset_readers.py:375-457` for everything that does
    not need the image bytes, and reports WHICH intrinsics branch fires so
    a payload that silently falls through to the FOV fallback is caught.
    """
    if "frames" not in contents:
        raise ContractError("payload has no 'frames' key; the reader reads contents['frames'] at :375")
    timestamp = frame.get("time", 0.0)
    if "file_path" not in frame:
        raise ContractError("frame has no 'file_path'; the reader indexes it at :387")
    cam_name = str(frame["file_path"]) + extension
    if "transform_matrix" not in frame:
        raise ContractError("frame has no 'transform_matrix'; the reader indexes it at :390")
    R, T = reader_pose_arithmetic(frame["transform_matrix"])

    if all(k in frame for k in ("fl_x", "fl_y", "cx", "cy")):
        branch = "per_frame_intrinsics(:433)"
        fl_x, fl_y = frame["fl_x"], frame["fl_y"]
        cx, cy = frame["cx"], frame["cy"]
    elif all(k in contents for k in ("fl_x", "fl_y", "cx", "cy")):
        branch = "top_level_intrinsics(:443)"
        fl_x, fl_y = contents["fl_x"], contents["fl_y"]
        cx, cy = contents["cx"], contents["cy"]
    else:
        branch = "camera_angle_x_fallback(:453)"
        fl_x = fl_y = cx = cy = None
    return {"R": R, "T": T, "timestamp": timestamp, "relative_image_path": cam_name,
            "image_name": Path(cam_name).stem, "branch": branch,
            "fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy}


# ====================================================================
# Split enforcement -- against the WRITTEN artifacts
# ====================================================================


def assert_split_on_written(train_bytes: bytes, test_bytes: bytes,
                            expected_train_cameras: list[str],
                            frames_expected: int) -> dict:
    """The lesson from the 35-camera rebuild
    ([[operations/imvid-baseline-freeze]] A2): assert against what was
    WRITTEN, not against intent. An assertion about intent would not have
    caught a filter that silently matched nothing, so this re-parses the
    emitted bytes, checks the raw TEXT for held-out camera names, and
    requires the non-empty counts it expects."""
    train_text = train_bytes.decode("utf-8")
    leaks = [name for name in HELD_OUT if name in train_text]
    if leaks:
        raise ContractError(
            f"HELD-OUT LEAKAGE in {TRAIN_JSON}: the written bytes mention "
            f"{leaks}. The frozen split forbids any held-out camera "
            "influencing training ([[operations/imvid-baseline-freeze]]:47-50)."
        )
    train = json.loads(train_text)
    test = json.loads(test_bytes.decode("utf-8"))

    train_cams = sorted({str(f["camera"]) for f in train["frames"]})
    test_cams = sorted({str(f["camera"]) for f in test["frames"]})
    if train_cams != sorted(expected_train_cameras):
        raise ContractError(
            f"{TRAIN_JSON} carries cameras {train_cams}, expected "
            f"{sorted(expected_train_cameras)}"
        )
    if test_cams != sorted(HELD_OUT):
        raise ContractError(
            f"{TEST_JSON} carries cameras {test_cams}, expected {sorted(HELD_OUT)}. "
            "The held-out four go to the test file and NOWHERE else."
        )
    if not train["frames"] or not test["frames"]:
        raise ContractError(
            f"empty split: {len(train['frames'])} train / {len(test['frames'])} "
            "test frames. A filter that silently matched nothing looks exactly "
            "like this."
        )
    expect_train = len(expected_train_cameras) * frames_expected
    expect_test = len(HELD_OUT) * frames_expected
    if len(train["frames"]) != expect_train or len(test["frames"]) != expect_test:
        raise ContractError(
            f"split counts wrong: train {len(train['frames'])} (expected "
            f"{expect_train}), test {len(test['frames'])} (expected {expect_test})"
        )
    for f in train["frames"]:
        stem = Path(str(f["file_path"])).name
        if any(stem.startswith(name) for name in HELD_OUT):
            raise ContractError(f"HELD-OUT LEAKAGE: train file_path {f['file_path']!r}")
    return {
        "held_out": list(HELD_OUT),
        "train_cameras": train_cams,
        "test_cameras": test_cams,
        "train_frames": len(train["frames"]),
        "test_frames": len(test["frames"]),
        "checked_against": "the WRITTEN json bytes, not the in-memory payload",
    }


# ====================================================================
# Point cloud
# ====================================================================


def install_point_cloud(source: Path, out_root: Path, expect_points: int | None,
                        expect_sha256: str | None) -> dict:
    """Copy the union to `points3d.ply` -- THE EXACT NAME the reader needs.

    `scene/dataset_readers.py:481-491`: if `points3d.ply` is absent the
    reader generates a uniform random cloud in [-1.3, 1.3]^3 and prints
    only "Generating random point cloud". ImViD Opera's content spans
    roughly x -35..34, y -21..17, z -4..31, so a mis-named cloud is a
    catastrophically wrong initialization with no error at all.
    """
    if not source.is_file():
        raise ContractError(f"--ply {source} does not exist")
    source_sha = sha256_file(source)
    if expect_sha256 is not None and source_sha != expect_sha256:
        raise ContractError(
            f"--ply sha256 {source_sha} does not match the pinned "
            f"--expect-ply-sha256 {expect_sha256}"
        )
    destination = out_root / POINTCLOUD_BASENAME
    if destination.name != POINTCLOUD_BASENAME:  # defensive; constant-checked
        raise ContractError(f"destination must be named {POINTCLOUD_BASENAME}")
    if destination.exists():
        existing = sha256_file(destination)
        if existing != source_sha:
            raise ContractError(
                f"{destination} already exists with DIFFERENT content "
                f"(sha256 {existing} vs source {source_sha}). Refusing to "
                "overwrite an initialization that is not the one requested."
            )
    else:
        shutil.copyfile(source, destination)
    copy_sha = sha256_file(destination)
    if copy_sha != source_sha:
        raise ContractError(
            f"points3d.ply copy is not byte-identical: {copy_sha} vs {source_sha}"
        )

    xyz = vp.parse_ply_xyz(destination.read_bytes())
    if xyz.shape[0] == 0:
        raise ContractError(f"{destination} holds no vertices")
    if expect_points is not None and int(xyz.shape[0]) != int(expect_points):
        raise ContractError(
            f"{destination} holds {xyz.shape[0]} points, --expect-points said "
            f"{expect_points}"
        )
    return {
        "source": str(source),
        "source_sha256": source_sha,
        "destination": str(destination),
        "destination_sha256": copy_sha,
        "byte_identical": True,
        "points": int(xyz.shape[0]),
        "bbox_min": [float(v) for v in xyz.min(axis=0)],
        "bbox_max": [float(v) for v in xyz.max(axis=0)],
        "why_the_name_matters": (
            "scene/dataset_readers.py:481 looks for this exact basename; a "
            "mis-named cloud is silently replaced by a uniform random fill in "
            "[-1.3, 1.3]^3 with no error raised (:481-491)"
        ),
        "reused_unchanged": (
            "the sparse union is valid in the undistorted PINHOLE frame with NO "
            "modification: initUndistortRectifyMap ran with R = eye(3), so the "
            "world-to-camera transform is untouched and a world-frame point is "
            "invariant under a change to the projection model alone "
            "([[operations/imvid-baseline-freeze]] B6/B11)"
        ),
    }


# ====================================================================
# Frame / camera discovery
# ====================================================================


def discover_frames(frames_root: Path, wanted: list[int] | None) -> list[int]:
    if not frames_root.is_dir():
        raise ContractError(f"--frames-root {frames_root} is not a directory")
    found = []
    for child in sorted(frames_root.iterdir()):
        if child.is_dir() and child.name.startswith("frame_"):
            try:
                found.append(int(child.name.split("_", 1)[1]))
            except ValueError:
                continue
    if not found:
        raise ContractError(
            f"{frames_root} holds no frame_<NNNNNN>/ directories. The decode "
            "layout is <root>/frame_%06d/images/cam<NN>.png "
            "(scripts/imvid_decode_frames.py:209-212)."
        )
    if wanted is None:
        return sorted(found)
    missing = [i for i in wanted if i not in found]
    if missing:
        raise ContractError(f"--frames requested {missing} which are absent from {frames_root}")
    return sorted(wanted)


def source_image_path(frames_root: Path, frame_index: int, camera: str) -> Path:
    return frames_root / f"frame_{frame_index:06d}" / "images" / f"{camera}.png"


# ====================================================================
# convert
# ====================================================================


def cmd_convert(args) -> dict:
    scale = float(args.scale)
    if scale <= 0.0:
        raise ContractError("--scale must be positive")
    fps = parse_rational(args.fps_rational)

    out_root = Path(args.out).resolve()
    if out_root == REPO_ROOT or REPO_ROOT in out_root.parents:
        raise ContractError(f"--out {out_root} is inside the repository")
    if (out_root / "sparse").exists():
        raise ContractError(
            f"{out_root}/sparse exists. scene/__init__.py:50 dispatches on "
            "sparse/ BEFORE transforms_train.json at :56, so this scene would "
            "silently route into the structurally incompatible COLMAP path "
            "(hard-coded cam10 held-out at :574-575, uid = intr.id at :662, "
            "/30 timestamps at :700)."
        )

    frames_root = Path(args.frames_root)
    model_dir = Path(args.model)
    cameras, images, model_provenance = read_model(model_dir, args.model_format)

    # --- cameras: one K_new per camera id, both consumers fed from it ---
    out_cameras = {cid: derive_output_camera(cam, scale) for cid, cam in cameras.items()}
    described = {str(cid): describe_camera(e, scale) for cid, e in out_cameras.items()}
    distorted_models = {e["model"] for e in out_cameras.values() if e["is_distorted"]}
    undistorting = bool(distorted_models)

    if args.require_undistortion and not undistorting:
        raise ContractError(
            "--require-undistortion was given but this scene's camera(s) carry NO "
            f"distortion ({sorted({e['model'] for e in out_cameras.values()})}). "
            "ImViD Meeting and Playing ship a distortion-free PINHOLE camera "
            "(dataset-admission-matrix-2026-08-18.md:424-455); warping such a "
            "scene by Opera's distortion crashes nothing and displaces every "
            "feature by ~14.7 px median while poses and intrinsics stay correct. "
            "Refusing."
        )

    # --- split, computed from the model and then re-checked on disk ---
    all_names = sorted(camera_token(n) for n in images)
    if len(set(all_names)) != len(all_names):
        raise ContractError("duplicate camera names in the model")
    missing_held_out = [h for h in HELD_OUT if h not in all_names]
    if missing_held_out:
        raise ContractError(
            f"held-out camera(s) {missing_held_out} are absent from {model_dir}. "
            "Point --model at the SUPPLIED calibration that carries every camera, "
            "not at a 35-camera training subset -- the held-out four need poses "
            "so they can be scored once at the end."
        )
    train_cameras = [n for n in all_names if n not in HELD_OUT]
    test_cameras = [n for n in all_names if n in HELD_OUT]
    if len(test_cameras) != len(HELD_OUT):
        raise ContractError(f"expected {len(HELD_OUT)} held-out cameras, matched {test_cameras}")
    if not train_cameras:
        raise ContractError("the training split is empty; the held-out filter matched everything")

    name_to_image = {camera_token(n): (n, entry) for n, entry in images.items()}

    frame_indices = discover_frames(frames_root, args.frames)
    fps_check = verify_fps_against_decode_manifest(frames_root, fps, args.allow_unverified_fps)

    # --- source-image existence, checked BEFORE anything is written ---
    plan_images: list[tuple[str, int, Path, Path]] = []
    for frame_index in frame_indices:
        for camera in all_names:
            src = source_image_path(frames_root, frame_index, camera)
            if not src.is_file():
                raise ContractError(f"missing source image: {src}")
            dst = out_root / IMAGES_SUBDIR / f"{frame_stem(camera, frame_index)}.png"
            plan_images.append((camera, frame_index, src, dst))

    # --- maps + border, per distinct camera id (no images needed) ---
    map_records: dict[str, dict] = {}
    border_records: dict[str, dict] = {}
    maps: dict[int, tuple] = {}
    if not args.dry_run or args.measure_border:
        for cid, entry in out_cameras.items():
            map1, map2, record = build_maps(entry, args.new_camera_matrix)
            maps[cid] = (map1, map2)
            map_records[str(cid)] = record
            border_records[str(cid)] = measure_invalid_border(
                map1, map2, entry["native_width"], entry["native_height"])
            # The correctness argument, asserted rather than asserted-in-prose:
            # the floats about to be written are the floats handed to cv2.
            written = [float(entry["K_new"][0, 0]), float(entry["K_new"][1, 1]),
                       float(entry["K_new"][0, 2]), float(entry["K_new"][1, 2])]
            handed = record["arguments"]["newCameraMatrix"]
            if [handed[0][0], handed[1][1], handed[0][2], handed[1][2]] != written:
                raise ContractError(
                    "the newCameraMatrix handed to initUndistortRectifyMap is not "
                    "bit-identical to the intrinsic about to be written to JSON"
                )

    plan = {
        "mode": "convert",
        "schema_version": SCHEMA,
        "out": str(out_root),
        "model": model_provenance,
        "scale": scale,
        "new_camera_matrix": args.new_camera_matrix,
        "undistortion_applied": undistorting,
        "undistortion_reason": (
            "source camera model carries non-zero distortion" if undistorting
            else "source camera model is distortion-free; the map is a PURE SCALE "
                 "(all-zero distCoeffs) and no distortion correction is applied"),
        "cameras": described,
        "frame_indices": frame_indices,
        "frame_rate": {**fps_record(fps), "verification": fps_check},
        "split": {
            "held_out": list(HELD_OUT),
            "n_train_cameras": len(train_cameras),
            "train_cameras": train_cameras,
            "test_cameras": test_cameras,
        },
        "images_planned": len(plan_images),
        "cv2_calls": {"initUndistortRectifyMap": map_records},
        "invalid_border": border_records,
    }

    if args.dry_run:
        plan["dry_run"] = True
        plan["note"] = "no bytes were written"
        return plan

    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise ContractError(
            f"--out {out_root} is non-empty; pass --overwrite to write into it"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    plan["cv2_calls"]["remap"] = remap_call_record()

    # --- write the images ---
    image_hashes: dict[str, dict] = {}
    border_check: dict | None = None
    for position, (camera, frame_index, src, dst) in enumerate(plan_images):
        _, entry_image = name_to_image[camera]
        cam_entry = out_cameras[entry_image["camera_id"]]
        map1, map2 = maps[entry_image["camera_id"]]
        result = remap_image(src, dst, map1, map2, cam_entry)
        if border_check is None:
            # ONE image is checked against the analytic map: every pixel
            # that falls entirely outside the bilinear support must be
            # exactly borderValue. This proves the map that was MEASURED is
            # the map that was APPLIED. (Pixels in the one-pixel blend band
            # legitimately are not exactly 0, so they are excluded.)
            fully_outside = ((map1 <= -1.0) | (map1 >= float(cam_entry["native_width"]))
                             | (map2 <= -1.0) | (map2 >= float(cam_entry["native_height"])))
            array = result["array"]
            n_outside = int(fully_outside.sum())
            if n_outside and not bool((array[fully_outside] == 0).all()):
                raise ContractError(
                    f"{dst}: pixels outside the source raster are not borderValue; "
                    "the applied map is not the measured map"
                )
            border_check = {
                "image": str(dst),
                "pixels_fully_outside_source": n_outside,
                "all_exactly_borderValue": True if n_outside else None,
                "note": ("one-directional by design: an interior pixel may be "
                         "black for photographic reasons, so only the fully "
                         "outside set is asserted"),
            }
        if args.hash_images:
            image_hashes[dst.name] = {"bytes": dst.stat().st_size,
                                      "sha256": sha256_file(dst),
                                      "source": str(src),
                                      "source_sha256": sha256_file(src)}
        if (position + 1) % 25 == 0 or position + 1 == len(plan_images):
            print(f"[imvid] undistorted {position + 1}/{len(plan_images)} images",
                  flush=True)

    # --- build and write the transforms ---
    train_frames, test_frames = [], []
    pose_max_delta = 0.0
    for frame_index in frame_indices:
        for camera in all_names:
            name, entry_image = name_to_image[camera]
            cam_entry = out_cameras[entry_image["camera_id"]]
            matrix = blender_transform_matrix(entry_image["qvec"], entry_image["tvec"])
            record = build_frame_entry(camera, frame_index, matrix, cam_entry, fps)
            (test_frames if camera in HELD_OUT else train_frames).append(record)

    train_payload = build_transforms(train_frames, out_cameras)
    test_payload = build_transforms(test_frames, out_cameras)
    train_bytes = dump_transforms(train_payload)
    test_bytes = dump_transforms(test_payload)
    atomic_write_bytes(out_root / TRAIN_JSON, train_bytes)
    atomic_write_bytes(out_root / TEST_JSON, test_bytes)

    split_record = assert_split_on_written(train_bytes, test_bytes,
                                           train_cameras, len(frame_indices))

    # --- replay the reader over the WRITTEN artifacts ---
    reader_report = {}
    for label, path in ((TRAIN_JSON, out_root / TRAIN_JSON),
                        (TEST_JSON, out_root / TEST_JSON)):
        contents = json.loads(path.read_text(encoding="utf-8"))
        branches, missing_files, time_deltas = set(), [], []
        for frame in contents["frames"]:
            got = simulate_reader_frame(contents, frame, args.extension)
            branches.add(got["branch"])
            image_path = out_root / got["relative_image_path"]
            if not image_path.is_file():
                missing_files.append(str(image_path))
            camera = str(frame["camera"])
            _, entry_image = name_to_image[camera]
            R_expect = vp.qvec2rotmat(entry_image["qvec"]).T
            T_expect = np.asarray(entry_image["tvec"], dtype=np.float64)
            delta = max(float(np.abs(got["R"] - R_expect).max()),
                        float(np.abs(got["T"] - T_expect).max()))
            pose_max_delta = max(pose_max_delta, delta)
            time_deltas.append(abs(float(got["timestamp"])
                                   - frame_time(int(frame["frame_index"]), fps)))
        if missing_files:
            raise ContractError(
                f"{label} references {len(missing_files)} image(s) that do not "
                f"exist: {missing_files[:5]}"
            )
        if branches != {"per_frame_intrinsics(:433)"}:
            raise ContractError(
                f"{label}: the reader would take branch(es) {sorted(branches)}; "
                "this converter must trigger the per-frame branch at "
                "scene/dataset_readers.py:433 so fl_x/fl_y/cx/cy come from the "
                "same K_new the resampling map used"
            )
        reader_report[label] = {
            "frames": len(contents["frames"]),
            "intrinsics_branch": sorted(branches),
            "max_abs_time_delta": max(time_deltas) if time_deltas else 0.0,
            "every_referenced_image_exists": True,
        }
    if pose_max_delta > POSE_ROUNDTRIP_TOL:
        raise ContractError(
            f"POSE CONVENTION ERROR: replaying scene/dataset_readers.py:390-397 "
            f"over the WRITTEN transforms reproduces the model's (R_cw^T, t) only "
            f"to {pose_max_delta:.3e}, tolerance {POSE_ROUNDTRIP_TOL:.0e}"
        )

    ply_record = install_point_cloud(Path(args.ply), out_root, args.expect_points,
                                     args.expect_ply_sha256)

    if (out_root / "sparse").exists():
        raise ContractError(f"{out_root}/sparse appeared during conversion")

    max_time = max(frame_time(i, fps) for i in frame_indices)
    plan.update({
        "dry_run": False,
        "split_verified_on_written_artifacts": split_record,
        "reader_replay": reader_report,
        "pose_roundtrip_max_abs_delta": pose_max_delta,
        "pose_roundtrip_tolerance": POSE_ROUNDTRIP_TOL,
        "border_applied_check": border_check,
        "point_cloud": ply_record,
        "output_files": {
            TRAIN_JSON: {"bytes": len(train_bytes), "sha256": sha256_bytes(train_bytes)},
            TEST_JSON: {"bytes": len(test_bytes), "sha256": sha256_bytes(test_bytes)},
            POINTCLOUD_BASENAME: {"bytes": (out_root / POINTCLOUD_BASENAME).stat().st_size,
                                  "sha256": ply_record["destination_sha256"]},
        },
        "image_sha256": image_hashes if args.hash_images else
                        {"recorded": False, "reason": "--no-hash-images"},
        "required_config": {
            "eval": True,
            "why_eval": ("scene/dataset_readers.py:475-477 MERGES the test split "
                         "into training when eval is False; the frozen split "
                         "forbids that"),
            "resolution": 1,
            "why_resolution": ("utils/camera_utils.py:43-46 rescales cx/cy by a "
                               "naive cx / scale, NOT the frozen "
                               "(c + 0.5) * s - 0.5; undistorting offline to the "
                               "final raster and training at resolution 1 keeps "
                               "the two conventions from ever meeting"),
            "extension": args.extension,
            "frame_ratio": 1,
            "motion_track_dt": float(Fraction(1) / fps),
            "time_duration": [0.0, max_time],
            "num_pts_at_least": ply_record["points"],
            "why_num_pts": ("scene/dataset_readers.py:497-513 SUBSAMPLES the "
                            "cloud when it holds more points than num_pts"),
        },
        "not_established_by_this_script": [
            "no training has run and no ImViD metric exists",
            "the invalid-border fraction is MEASURED, not decided; no crop and "
            "no mask were applied",
            "presentations per unit is a schedule question this converter does "
            "not answer ([[operations/imvid-baseline-freeze]] B7/B10)",
        ],
    })
    return plan


# ====================================================================
# Self-test -- NO data files, NO cv2
# ====================================================================


def _check(name: str, ok: bool, detail: dict) -> dict:
    if not ok:
        raise ContractError(f"SELF-TEST FAILED: {name} -- {detail}")
    return {"name": name, "status": "PASS", **detail}


def run_self_test() -> list[dict]:
    checks: list[dict] = []
    camera = vp.RECORDED_OPENCV_CAMERA
    recorded = vp.RECORDED_PINHOLE_AT_SCALE_HALF

    # 1 -- the derived intrinsic reproduces the frozen experiment-156 line
    entry = derive_output_camera(camera, 0.5)
    K_new = entry["K_new"]
    deltas = {"fx": abs(K_new[0, 0] - recorded["fx"]), "fy": abs(K_new[1, 1] - recorded["fy"]),
              "cx": abs(K_new[0, 2] - recorded["cx"]), "cy": abs(K_new[1, 2] - recorded["cy"])}
    checks.append(_check(
        "intrinsics_match_frozen_experiment_156",
        max(deltas.values()) <= K_NEW_TOL
        and (entry["width"], entry["height"]) == (recorded["width"], recorded["height"]),
        {"tolerance": K_NEW_TOL,
         "max_abs_delta": float(max(deltas.values())),
         "derived": [float(K_new[0, 0]), float(K_new[1, 1]),
                     float(K_new[0, 2]), float(K_new[1, 2])],
         "frozen": [recorded["fx"], recorded["fy"], recorded["cx"], recorded["cy"]],
         "raster": [entry["width"], entry["height"]]}))

    # 2 -- the naive c * scale is a QUARTER PIXEL off, and by how much
    naive = [camera["params"][2] * 0.5, camera["params"][3] * 0.5]
    checks.append(_check(
        "naive_principal_point_is_a_quarter_pixel_off",
        abs((naive[0] - float(K_new[0, 2])) - 0.25) < 1e-12
        and abs((naive[1] - float(K_new[1, 2])) - 0.25) < 1e-12,
        {"frozen_convention": [float(K_new[0, 2]), float(K_new[1, 2])],
         "naive_c_times_s": naive,
         "difference_px": [naive[0] - float(K_new[0, 2]), naive[1] - float(K_new[1, 2])]}))

    # 3 -- unit scale is the identity
    unit = derive_output_camera(camera, 1.0)
    checks.append(_check(
        "identity_at_unit_scale",
        bool(np.allclose(unit["K_new"], unit["K"], rtol=0.0, atol=0.0))
        and (unit["width"], unit["height"]) == (camera["width"], camera["height"]),
        {"note": "(c + 0.5) * 1 - 0.5 == c exactly"}))

    # 4 -- the pose round trip THROUGH THE READER'S OWN ARITHMETIC
    axis = np.array([0.3, -0.5, 0.81]); axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(37.0)
    qvec = [float(np.cos(angle / 2)), *(np.sin(angle / 2) * axis)]
    tvec = [0.13, -0.27, 6.5]
    matrix = blender_transform_matrix(qvec, tvec)
    R_got, T_got = reader_pose_arithmetic(matrix)
    R_expect = vp.qvec2rotmat(qvec).T
    T_expect = np.asarray(tvec, dtype=np.float64)
    delta = max(float(np.abs(R_got - R_expect).max()), float(np.abs(T_got - T_expect).max()))
    # negative control: omit the Blender flip and the reader must NOT agree
    unflipped = np.array(matrix, dtype=np.float64)
    unflipped[:3, 1:3] *= -1.0
    R_bad, T_bad = reader_pose_arithmetic(unflipped)
    bad_delta = max(float(np.abs(R_bad - R_expect).max()), float(np.abs(T_bad - T_expect).max()))
    checks.append(_check(
        "pose_roundtrip_through_reader_arithmetic",
        delta <= POSE_ROUNDTRIP_TOL and bad_delta > 0.1,
        {"max_abs_delta": delta, "tolerance": POSE_ROUNDTRIP_TOL,
         "negative_control_without_the_c2w_flip": bad_delta,
         "lines_replayed": "scene/dataset_readers.py:390-397"}))

    # 5 -- the emitted frame entry takes the PER-FRAME intrinsics branch
    frame = build_frame_entry("cam07", 150, matrix, entry, IMVID_MEASURED_FPS)
    payload = build_transforms([frame], {2: entry})
    replay = simulate_reader_frame(payload, payload["frames"][0], DEFAULT_EXTENSION)
    checks.append(_check(
        "transforms_shape_matches_readCamerasFromTransforms",
        replay["branch"] == "per_frame_intrinsics(:433)"
        and replay["relative_image_path"] == "images/cam07_000150.png"
        and not str(frame["file_path"]).endswith(DEFAULT_EXTENSION)
        and np.shape(frame["transform_matrix"]) == (4, 4)
        and float(replay["fl_x"]) == float(K_new[0, 0])
        and float(replay["cx"]) == float(K_new[0, 2]),
        {"branch": replay["branch"],
         "file_path": frame["file_path"],
         "reader_builds": replay["relative_image_path"],
         "image_name": replay["image_name"],
         "keys": sorted(frame),
         "note": "file_path carries NO extension; the reader appends it at :387"}))

    # 6 -- image_name parses under the repo's camera-name convention
    import re
    checks.append(_check(
        "image_name_matches_repo_camera_convention",
        re.match(r"^cam(\d+)_(\d+)$", replay["image_name"]) is not None,
        {"image_name": replay["image_name"],
         "pattern": "scene/packet_birth_flow.py:114  ^cam(\\d+)_(\\d+)$"}))

    # 7 -- the K_new in the JSON is the K_new a map would be built from
    checks.append(_check(
        "one_K_new_two_consumers",
        float(frame["fl_x"]) == float(entry["K_new"][0, 0])
        and float(frame["fl_y"]) == float(entry["K_new"][1, 1])
        and float(frame["cx"]) == float(entry["K_new"][0, 2])
        and float(frame["cy"]) == float(entry["K_new"][1, 2]),
        {"note": ("the SAME K_new object is passed as newCameraMatrix and read "
                  "for the JSON; cmd_convert re-asserts this against the "
                  "recorded cv2 arguments before writing anything")}))

    # 8 -- the frame period is the measured rational, not 60 and not 30
    fps = IMVID_MEASURED_FPS
    t299 = frame_time(299, fps)
    checks.append(_check(
        "frame_period_is_the_measured_rational",
        abs(t299 - 299 * 1001 / 60000) < 1e-12
        and abs(t299 - 299 / 60) > 1e-4
        and abs(t299 - 299 / 30) > 1.0
        and abs(float(Fraction(1) / fps) - 0.016683333333333334) < 1e-15
        and abs(float((Fraction(1) / N3V_HARDCODED_FPS) / (Fraction(1) / fps))
                - 2000 / 1001) < 1e-12,
        {"time_299_s": t299,
         "if_60_fps_had_been_assumed": 299 / 60,
         "if_the_repo_30_fps_constant_had_been_inherited": 299 / 30,
         "frame_period_s": float(Fraction(1) / fps),
         "ratio_30fps_over_imvid": float(
             (Fraction(1) / N3V_HARDCODED_FPS) / (Fraction(1) / fps)),
         "note": "2000/1001 = 1.998001998..., corrected at "
                 "[[operations/imvid-baseline-freeze]] B12"}))

    # 9 -- a decimal frame rate is refused
    refused = False
    try:
        parse_rational("59.94")
    except ContractError:
        refused = True
    checks.append(_check(
        "decimal_frame_rate_is_refused", refused,
        {"note": "the rate is carried as NUM/DEN so 59.94 and 60000/1001 can "
                 "never be confused"}))

    # 10 -- the split, and the WRITTEN-artifact assertion's negative control
    names = [f"cam{i:02d}" for i in range(39)]
    train = [n for n in names if n not in HELD_OUT]
    test = [n for n in names if n in HELD_OUT]
    good_train = dump_transforms(build_transforms(
        [build_frame_entry(n, 0, matrix, entry, fps) for n in train], {2: entry}))
    good_test = dump_transforms(build_transforms(
        [build_frame_entry(n, 0, matrix, entry, fps) for n in test], {2: entry}))
    ok_split = assert_split_on_written(good_train, good_test, train, 1)
    leak_train = dump_transforms(build_transforms(
        [build_frame_entry(n, 0, matrix, entry, fps) for n in train + ["cam13"]],
        {2: entry}))
    caught_leak = False
    try:
        assert_split_on_written(leak_train, good_test, train, 1)
    except ContractError:
        caught_leak = True
    empty_caught = False
    try:
        assert_split_on_written(dump_transforms(build_transforms([], {2: entry})),
                                good_test, train, 1)
    except ContractError:
        empty_caught = True
    checks.append(_check(
        "split_logic_and_its_negative_controls",
        len(train) == 35 and sorted(test) == sorted(HELD_OUT)
        and ok_split["train_frames"] == 35 and ok_split["test_frames"] == 4
        and caught_leak and empty_caught,
        {"held_out": list(HELD_OUT), "n_train": len(train), "n_test": len(test),
         "leak_detected": caught_leak,
         "silently_empty_split_detected": empty_caught,
         "note": ("the assertion reads the WRITTEN bytes; an assertion about "
                  "intent would not catch a filter that silently matched "
                  "nothing ([[operations/imvid-baseline-freeze]] A2)")}))

    # 11 -- the point-cloud basename is the exact one the reader needs
    checks.append(_check(
        "point_cloud_basename_is_exact",
        POINTCLOUD_BASENAME == "points3d.ply"
        and POINTCLOUD_BASENAME != "points3d_colmap_union.ply",
        {"required": POINTCLOUD_BASENAME,
         "artifact_is_named": "points3d_colmap_union.ply",
         "consequence_if_missed": ("scene/dataset_readers.py:481-491 generates a "
                                   "uniform random cloud in [-1.3, 1.3]^3 with no "
                                   "error, against ImViD content spanning x -35..34")}))

    # 12 -- the invalid-border measurement, on synthetic maps with a known answer
    grid_h, grid_w = 6, 8
    map1 = np.tile(np.arange(grid_w, dtype=np.float32), (grid_h, 1))
    map2 = np.tile(np.arange(grid_h, dtype=np.float32).reshape(-1, 1), (1, grid_w))
    map1 = map1 - 2.0   # the two left-most columns now sample x < 0
    border = measure_invalid_border(map1, map2, grid_w, grid_h)
    # a corner-only case, which is the shape barrel undistortion produces
    corner_valid = np.ones((grid_h, grid_w), dtype=bool)
    corner_valid[0:2, 0:3] = False
    corner = _trim_to_all_valid(corner_valid)
    checks.append(_check(
        "invalid_border_measurement_known_answer",
        border["invalid_pixels"] == 2 * grid_h
        and abs(border["invalid_fraction"] - (2.0 / grid_w)) < 1e-12
        and border["per_edge_trim_to_all_valid_px"] == {"left": 2, "right": 0,
                                                        "top": 0, "bottom": 0}
        and border["largest_all_valid_axis_aligned_crop"]["width"] == grid_w - 2
        and border["largest_all_valid_axis_aligned_crop"]["height"] == grid_h
        and border["largest_all_valid_axis_aligned_crop"]["verified_all_valid"] is True
        and border["largest_all_valid_axis_aligned_crop"]["APPLIED"] is False
        and (corner["left"], corner["right"], corner["top"], corner["bottom"]) == (0, 0, 2, 0)
        and corner["all_valid"] is True,
        {"invalid_fraction": border["invalid_fraction"],
         "per_edge": border["per_edge_trim_to_all_valid_px"],
         "crop": border["largest_all_valid_axis_aligned_crop"],
         "corner_case_trim": corner,
         "note": ("a naive per-row leading-invalid depth reads the FULL height "
                  "whenever one column is invalid end-to-end and returns a "
                  "negative crop dimension; this check caught exactly that")}))

    # 13 -- getOptimalNewCameraMatrix is refused by NAME, without cv2
    optimal_refused = False
    try:
        build_maps(entry, "optimal")
    except ContractError as exc:
        optimal_refused = "getOptimalNewCameraMatrix" in str(exc)
    except ImportError:
        optimal_refused = False
    checks.append(_check(
        "optimal_new_camera_matrix_is_refused", bool(optimal_refused),
        {"note": ("the refusal is raised BEFORE cv2 is used, so it holds even "
                  "where cv2 is absent; it names the frozen experiment-156 "
                  "camera it would have superseded")}))

    # 14 -- an unsupported camera model is refused rather than guessed
    model_refused = False
    try:
        derive_output_camera({"model": "SIMPLE_RADIAL", "width": 10, "height": 10,
                              "params": [1.0, 5.0, 5.0, 0.0]}, 0.5)
    except ContractError:
        model_refused = True
    checks.append(_check(
        "unsupported_camera_model_is_refused", model_refused,
        {"supported": list(_SUPPORTED_MODELS),
         "note": "the model is READ FROM DATA and branched on, never assumed"}))

    # 15 -- a distortion-free PINHOLE scene is recognised as such
    pinhole_scene = derive_output_camera(
        {"model": "PINHOLE", "width": 5338, "height": 2991,
         "params": [2722.5516678678127, 2721.4363233225208, 2669.0, 1495.5]}, 0.5)
    checks.append(_check(
        "distortion_free_pinhole_is_recognised",
        pinhole_scene["is_distorted"] is False
        and bool(np.all(pinhole_scene["dist"] == 0.0))
        and entry["is_distorted"] is True,
        {"scene": "ImViD scene4_meeting, dataset-admission-matrix-2026-08-18.md:424-455",
         "meeting_is_distorted": pinhole_scene["is_distorted"],
         "opera_is_distorted": entry["is_distorted"],
         "consequence": ("--require-undistortion refuses such a scene; warping it "
                         "by Opera's distortion crashes nothing and displaces "
                         "every feature by ~14.7 px median")}))

    # 16 -- cv2, the production resampler
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        checks.append({
            "name": "cv2_present", "status": "SKIPPED",
            "reason": f"cv2 is not importable ({exc})",
            "consequence": (
                "NOT EXERCISED HERE: cv2.initUndistortRectifyMap, cv2.remap, "
                "cv2.imread/imwrite, the real invalid-border fraction, and the "
                "applied-vs-measured map check. Everything above is pure "
                "numpy/stdlib arithmetic. Run --mode self-test inside the "
                "Apollo image to close this."),
        })
    else:
        _, _, record = build_maps(entry, "scaled_k")
        handed = record["arguments"]["newCameraMatrix"]
        checks.append(_check(
            "cv2_present",
            handed[0][0] == float(entry["K_new"][0, 0])
            and handed[1][1] == float(entry["K_new"][1, 1])
            and handed[0][2] == float(entry["K_new"][0, 2])
            and handed[1][2] == float(entry["K_new"][1, 2]),
            {"cv2_version": record["cv2_version"],
             "note": "the map was built and its newCameraMatrix is bit-identical "
                     "to the intrinsic this converter writes"}))
    return checks


def cmd_self_test(_args) -> dict:
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


# ====================================================================
# CLI
# ====================================================================


def _parse_frames(text: str | None) -> list[int] | None:
    if text is None:
        return None
    out: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token.lstrip("-"):
            lo, hi = token.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(token))
    if not out:
        raise ContractError(f"--frames {text!r} selected nothing")
    return sorted(set(out))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=("convert", "self-test"))
    parser.add_argument("--model", default=None,
                        help="COLMAP model directory carrying poses for EVERY "
                             "camera in both splits (READ ONLY)")
    parser.add_argument("--model-format", default="auto",
                        choices=("auto", "binary", "text"))
    parser.add_argument("--frames-root", default=None,
                        help="decoded native frames: <root>/frame_%%06d/images/cam<NN>.png")
    parser.add_argument("--frames", default=None,
                        help="restrict to these frame indices, e.g. '0,150,299' "
                             "or '0-49'. Default: every frame_* directory found")
    parser.add_argument("--out", default=None, help="scene root; OUTSIDE the repository")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="declared output scale of the undistorted raster")
    parser.add_argument("--fps-rational", default=None,
                        help="MEASURED stream rate as NUM/DEN, e.g. 60000/1001. "
                             "Required: no run inherits a constant")
    parser.add_argument("--allow-unverified-fps", action="store_true",
                        help="proceed when the decode manifest is absent and the "
                             "declared rate therefore cannot be checked")
    parser.add_argument("--ply", default=None,
                        help="sparse union PLY; copied to points3d.ply verbatim")
    parser.add_argument("--expect-points", type=int, default=None,
                        help="refuse unless the PLY holds exactly this many points")
    parser.add_argument("--expect-ply-sha256", default=None,
                        help="pin the source PLY's sha256")
    parser.add_argument("--new-camera-matrix", default="scaled_k",
                        choices=("scaled_k", "optimal"),
                        help="scaled_k = the ORIGINAL K scaled under "
                             "(c + 0.5) * s - 0.5, which is the frozen "
                             "experiment-156 camera. 'optimal' names "
                             "getOptimalNewCameraMatrix and is REFUSED; the "
                             "choice exists so the refusal can state why")
    parser.add_argument("--require-undistortion", action="store_true",
                        help="refuse a scene whose camera carries no distortion")
    parser.add_argument("--extension", default=DEFAULT_EXTENSION,
                        help="the reader's image extension; must match the config")
    parser.add_argument("--no-hash-images", dest="hash_images", action="store_false",
                        help="skip per-image sha256 (large conversions only)")
    parser.add_argument("--measure-border", action="store_true",
                        help="dry-run only: build the maps and measure the "
                             "invalid border without writing anything")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default=None, help="manifest path; outside the repo")
    parser.set_defaults(hash_images=True)
    args = parser.parse_args(argv)

    if args.mode == "convert":
        for name in ("model", "frames_root", "out", "fps_rational", "ply"):
            if getattr(args, name) is None:
                raise ContractError(f"--{name.replace('_', '-')} is required in convert mode")
        if args.manifest is None and not args.dry_run:
            raise ContractError("--manifest is required in convert mode")
    if args.manifest is not None:
        manifest = Path(args.manifest).resolve()
        if manifest == REPO_ROOT or REPO_ROOT in manifest.parents:
            raise ContractError(f"--manifest {manifest} is inside the repository")
    args.frames = _parse_frames(args.frames)

    if args.mode == "self-test":
        payload = cmd_self_test(args)
    else:
        payload = cmd_self_test(args)  # certify the instrument in the same cell
        payload = {"self_test": payload, **cmd_convert(args)}

    payload["schema_version"] = SCHEMA
    payload["argv"] = list(sys.argv[1:])
    if args.manifest is not None:
        body = json.dumps(payload, allow_nan=False, sort_keys=True, indent=1,
                          default=str)
        atomic_write_bytes(Path(args.manifest), body.encode("utf-8") + b"\n")
        print(f"[imvid] manifest -> {args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
