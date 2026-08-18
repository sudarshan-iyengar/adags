"""ImViD Opera pilot: build the 35-TRAINING-CAMERA inputs, and validate the
OPENCV -> PINHOLE undistortion numerically.

WHY. The frozen sparse clouds under ``data/imvid/sparse/`` were triangulated
using ALL 39 cameras, so they carry observations from the four held-out views
``cam00, cam13, cam25, cam38`` ([[operations/imvid-baseline-freeze]]). That is
initialization-time leakage. The 39-camera artifacts are PRESERVED as the
calibration-validation evidence they were built for; this script produces the
35-camera inputs a baseline may legitimately initialize from.

Two independent jobs, selected by ``--mode``:

``subset``   write, per frozen frame, an images directory holding ONLY the 35
             training cameras and a COLMAP model directory whose
             ``images.txt`` holds only those 35 entries, with ``cameras.txt``
             byte-identical to the supplied one. ``scripts/imvid_sparse_init``
             then runs on that directory unchanged.

``undistort`` validate the OPENCV -> PINHOLE conversion at a declared scale
             WITHOUT re-triangulating: derive the PINHOLE intrinsics, measure
             the residual the undistortion map leaves on a dense grid, and
             report the maximum radial displacement. Writes the derived
             ``cameras.txt`` and a manifest; writes no images unless asked.

Nothing here modifies a raw or frozen artifact. Every output is a new file
under ``--workdir``, which must lie outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.artifacts import atomic_write_bytes  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402

# The frozen split ([[operations/imvid-baseline-freeze]]): four held-out
# cameras chosen outcome-blind as np.linspace(0, 38, 4) over the sorted ids.
HELD_OUT = ("cam00", "cam13", "cam25", "cam38")
N_CAMERAS_TOTAL = 39
SCHEMA = "imvid-pilot-prepare-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cameras(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or not line.strip():
            continue
        rows.append(line.split())
    return rows


def parse_images(path: Path) -> list[tuple[list[str], str]]:
    """COLMAP images.txt -> [(header_fields, points2d_line)]."""
    lines = path.read_text(encoding="utf-8").splitlines()
    body = [ln for ln in lines if not ln.strip().startswith("#")]
    out = []
    index = 0
    while index < len(body):
        parts = body[index].split()
        if len(parts) >= 10:
            points2d = body[index + 1] if index + 1 < len(body) else ""
            out.append((parts, points2d))
            index += 2
        else:
            index += 1
    return out


def camera_name(entry: list[str]) -> str:
    return Path(entry[9]).stem


def mode_subset(args) -> dict:
    model = Path(args.model)
    work = Path(args.workdir)
    cameras_txt = model / "cameras.txt"
    images_txt = model / "images.txt"
    for path in (cameras_txt, images_txt):
        if not path.is_file():
            raise ContractError(f"missing supplied model file: {path}")

    cam_rows = parse_cameras(cameras_txt)
    if len(cam_rows) != 1:
        raise ContractError(f"expected exactly one supplied camera, got {len(cam_rows)}")
    entries = parse_images(images_txt)
    if len(entries) != N_CAMERAS_TOTAL:
        raise ContractError(
            f"expected {N_CAMERAS_TOTAL} image entries, got {len(entries)}"
        )

    names = [camera_name(e) for e, _ in entries]
    if len(set(names)) != len(names):
        raise ContractError("duplicate camera names in images.txt")
    missing = [h for h in HELD_OUT if h not in names]
    if missing:
        raise ContractError(f"held-out cameras absent from the supplied model: {missing}")

    kept = [(e, p) for (e, p) in entries if camera_name(e) not in HELD_OUT]
    if len(kept) != N_CAMERAS_TOTAL - len(HELD_OUT):
        raise ContractError(f"expected 35 training entries, got {len(kept)}")

    out_model = work / "model_35"
    out_model.mkdir(parents=True, exist_ok=True)
    # cameras.txt is copied BYTE-IDENTICALLY: the supplied intrinsics are
    # fixed authority and the subset must not perturb them.
    shutil.copyfile(cameras_txt, out_model / "cameras.txt")
    if sha256_file(out_model / "cameras.txt") != sha256_file(cameras_txt):
        raise ContractError("cameras.txt copy is not byte-identical")

    header = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(kept)}, mean observations per image: 0",
    ]
    body = []
    for entry, points2d in kept:
        body.append(" ".join(entry))
        body.append(points2d)
    text = "\n".join(header + body) + "\n"
    atomic_write_bytes(out_model / "images.txt", text.encode("utf-8"))

    frames_written = {}
    for frame_dir in sorted(Path(args.frames_root).iterdir()):
        source = frame_dir / "images"
        if not source.is_dir():
            continue
        target = work / "frames_35" / frame_dir.name / "images"
        target.mkdir(parents=True, exist_ok=True)
        copied, skipped = [], []
        for image in sorted(source.iterdir()):
            if image.stem in HELD_OUT:
                skipped.append(image.name)
                continue
            destination = target / image.name
            if not destination.exists():
                shutil.copyfile(image, destination)
            copied.append(image.name)
        if len(copied) != 35 or len(skipped) != 4:
            raise ContractError(
                f"{frame_dir.name}: copied {len(copied)}, skipped {len(skipped)}; "
                "expected 35 and 4"
            )
        frames_written[frame_dir.name] = {
            "n_images": len(copied),
            "excluded": skipped,
        }

    # THE LEAKAGE ASSERTION, made against the written artifacts rather than
    # against intent: no held-out name may appear anywhere in the 35-camera
    # inputs.
    leaks = []
    written_images = sorted((work / "frames_35").rglob("*.png"))
    for path in written_images:
        if path.stem in HELD_OUT:
            leaks.append(str(path))
    model_text = (out_model / "images.txt").read_text(encoding="utf-8")
    for name in HELD_OUT:
        if name in model_text:
            leaks.append(f"images.txt mentions {name}")
    if leaks:
        raise ContractError(f"HELD-OUT LEAKAGE in the prepared inputs: {leaks[:5]}")

    return {
        "mode": "subset",
        "held_out": list(HELD_OUT),
        "n_training_cameras": len(kept),
        "training_cameras": sorted(camera_name(e) for e, _ in kept),
        "model_35": str(out_model),
        "cameras_txt_sha256": sha256_file(out_model / "cameras.txt"),
        "images_txt_sha256": sha256_bytes(text.encode("utf-8")),
        "frames": frames_written,
        "n_images_written": len(written_images),
        "leakage_check": "PASSED: no held-out camera name in any prepared input",
    }


def mode_undistort(args) -> dict:
    """Derive the PINHOLE camera at a declared scale and MEASURE what the
    undistortion actually does, instead of asserting that it is mild."""

    import cv2

    model = Path(args.model)
    work = Path(args.workdir)
    cam_rows = parse_cameras(model / "cameras.txt")
    if len(cam_rows) != 1:
        raise ContractError(f"expected exactly one supplied camera, got {len(cam_rows)}")
    row = cam_rows[0]
    model_name = row[1]
    if model_name != "OPENCV":
        raise ContractError(f"expected an OPENCV camera, got {model_name!r}")
    width, height = int(row[2]), int(row[3])
    fx, fy, cx, cy, k1, k2, p1, p2 = (float(v) for v in row[4:12])

    scale = float(args.scale)
    if scale <= 0:
        raise ContractError("--scale must be positive")
    out_w = int(round(width * scale))
    out_h = int(round(height * scale))

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2], dtype=np.float64)

    # The PINHOLE camera the undistorted images are expressed in. Pixel
    # centres are at integer coordinates in COLMAP's convention, so the
    # scaled principal point is (c + 0.5) * scale - 0.5.
    K_new = np.array(
        [
            [fx * scale, 0.0, (cx + 0.5) * scale - 0.5],
            [0.0, fy * scale, (cy + 0.5) * scale - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # MEASUREMENT 1: how far does the distortion model move a pixel? Project a
    # dense grid through the distorted model and compare to the pinhole
    # projection of the same rays.
    step = max(1, min(width, height) // 64)
    xs = np.arange(0, width, step, dtype=np.float64)
    ys = np.arange(0, height, step, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    pix = np.stack([gx.ravel(), gy.ravel()], axis=1)
    # normalized rays for those distorted pixels
    undist_norm = cv2.undistortPoints(
        pix.reshape(-1, 1, 2).astype(np.float64), K, dist
    ).reshape(-1, 2)
    # where the pinhole camera puts those rays
    pinhole_pix = np.stack(
        [
            undist_norm[:, 0] * K[0, 0] + K[0, 2],
            undist_norm[:, 1] * K[1, 1] + K[1, 2],
        ],
        axis=1,
    )
    displacement = np.linalg.norm(pinhole_pix - pix, axis=1)

    # MEASUREMENT 2: round-trip consistency. Re-distort the undistorted rays
    # and check we return to the original pixel. This is what actually
    # certifies the map, and it is measured rather than assumed.
    rays = np.concatenate(
        [undist_norm, np.zeros((undist_norm.shape[0], 1), dtype=np.float64)], axis=1
    )
    reproj, _ = cv2.projectPoints(
        rays.reshape(-1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K,
        dist,
    )
    round_trip = np.linalg.norm(reproj.reshape(-1, 2) - pix, axis=1)

    out_model = work / "model_pinhole"
    out_model.mkdir(parents=True, exist_ok=True)
    text = (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"{row[0]} PINHOLE {out_w} {out_h} "
        f"{K_new[0,0]!r} {K_new[1,1]!r} {K_new[0,2]!r} {K_new[1,2]!r}\n"
    )
    atomic_write_bytes(out_model / "cameras.txt", text.encode("utf-8"))

    return {
        "mode": "undistort",
        "source_camera": {
            "model": model_name,
            "width": width,
            "height": height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "k1": k1, "k2": k2, "p1": p1, "p2": p2,
        },
        "declared_scale": scale,
        "pinhole_camera": {
            "model": "PINHOLE",
            "width": out_w,
            "height": out_h,
            "fx": K_new[0, 0], "fy": K_new[1, 1],
            "cx": K_new[0, 2], "cy": K_new[1, 2],
        },
        "distortion_displacement_px_at_native_resolution": {
            "grid_step_px": step,
            "n_samples": int(pix.shape[0]),
            "min": float(displacement.min()),
            "median": float(np.median(displacement)),
            "p99": float(np.percentile(displacement, 99)),
            "max": float(displacement.max()),
        },
        "round_trip_error_px": {
            "median": float(np.median(round_trip)),
            "max": float(round_trip.max()),
        },
        "cameras_txt_sha256": sha256_bytes(text.encode("utf-8")),
        "note": (
            "the 35-camera sparse initialization is built in the SUPPLIED "
            "OPENCV frame; a PINHOLE-frame initialization would be a separate "
            "build and is not produced here"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=("subset", "undistort"))
    parser.add_argument("--model", required=True, help="supplied cameras.txt + images.txt (READ ONLY)")
    parser.add_argument("--workdir", required=True, help="all outputs land here; outside the repo")
    parser.add_argument("--frames-root", default=None, help="subset mode: the decoded frames root")
    parser.add_argument("--scale", type=float, default=0.5, help="undistort mode: declared output scale")
    parser.add_argument("--out", required=True, help="manifest path")
    args = parser.parse_args(argv)

    work = Path(args.workdir).resolve()
    if work == REPO_ROOT or REPO_ROOT in work.parents:
        raise ContractError(f"--workdir {work} is inside the repository")

    if args.mode == "subset":
        if args.frames_root is None:
            raise ContractError("--frames-root is required in subset mode")
        payload = mode_subset(args)
    else:
        payload = mode_undistort(args)

    payload["schema_version"] = SCHEMA
    payload["workdir"] = str(work)
    body = json.dumps(payload, allow_nan=False, sort_keys=True, indent=1)
    atomic_write_bytes(Path(args.out), body.encode("utf-8") + b"\n")
    print(json.dumps({"out": args.out, "mode": args.mode, "summary": {
        k: payload[k] for k in payload
        if k in ("n_training_cameras", "leakage_check", "n_images_written",
                 "pinhole_camera", "distortion_displacement_px_at_native_resolution",
                 "round_trip_error_px")
    }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
