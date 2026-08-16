#!/usr/bin/env python3
"""Fixed-pose sparse triangulation for one frozen ImViD frame (COLMAP 3.6).

Step 2 of the ImViD sparse-initialization lane. The sample ships
calibrated cameras and NO 3D points, so a cloud must be triangulated.
The supplied intrinsics and extrinsics are FIXED AUTHORITY: this script
exists to produce points UNDER them, never to re-estimate them.

--------------------------------------------------------------------
WHY THIS IS WRITTEN AGAINST 3.6 SPECIFICALLY
--------------------------------------------------------------------

The Determined runtime carries **COLMAP 3.6**
(`3.6+dev2+git20191105-1build1`), which predates the 3.9 changes that
made `point_triangulator` safe by default. Verified by probing the
installed binary rather than the documentation:

  --refine_intrinsics            ABSENT in 3.6 (passing it hard-errors)
  --Mapper.fix_existing_images   (=0)  LIVE, and defaults to NOT fixing poses
  --clear_points                 (=0)  no filename-transcription clause
  model_comparer                 ABSENT

An analysis of COLMAP 3.11.1 concludes that `point_triangulator` cannot
move poses "by construction, no flag needed". That is true of 3.11.1 and
FALSE here. Every guarantee this script relies on is therefore stated as
an explicit flag rather than inherited from a default:

    --Mapper.fix_existing_images 1        poses held
    --Mapper.ba_refine_focal_length 0     fx, fy held
    --Mapper.ba_refine_principal_point 0  cx, cy held
    --Mapper.ba_refine_extra_params 0     k1, k2, p1, p2 held

Whether the `ba_refine_*` flags are live or inert on 3.6's
`point_triangulator` is NOT established; passing 0 is correct under both
readings, which is why they are passed rather than reasoned about.

--------------------------------------------------------------------
THE IDENTIFIER REMAP, AND WHY IT TOUCHES NO GEOMETRY
--------------------------------------------------------------------

With `--clear_points` defaulting to 0 and no transcription clause in 3.6,
the supplied `images.txt` IDs cannot be assumed to be remapped onto the
database's by filename. So after feature extraction this script READS the
database's `image_id -> name` table and rewrites the input model's
identifier columns to match it, likewise for the single camera id.

**Only identifier columns change.** Every quaternion, every translation
and all eight OPENCV parameters are copied through as the exact decimal
strings the supplied files contain — they are never parsed to float and
re-formatted, so they cannot be perturbed by round-tripping. The final
numeric diff proves this rather than assuming it.

--------------------------------------------------------------------
FAIL-CLOSED
--------------------------------------------------------------------

A missing colmap; a COLMAP step returning non-zero; an image in the
model with no database row of the same name; more than one camera in the
supplied model; a non-empty output directory; an output inside the git
repository; and — the decisive one — ANY difference between the supplied
and produced intrinsics or extrinsics, which aborts before the result
can be used.

Feature extraction and matching run on CPU (`use_gpu 0`) so this can
occupy a zero-GPU-slot cell and never competes with training.
`SiftExtraction.max_image_size` is raised to the native width: its
default of 3200 would detect features on a downscaled raster while the
supplied intrinsics describe 5312x2988.

Usage:
  python3 scripts/imvid_sparse_init.py \
      --images  .../frames/frame_000000/images \
      --model   .../scene1_opera            (cameras.txt + images.txt) \
      --workdir .../sparse/frame_000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402

COLMAP = "colmap"
NATIVE_WIDTH = 5312


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(args: list[str], label: str) -> dict:
    started = time.perf_counter()
    out = subprocess.run(args, capture_output=True, text=True, timeout=36000)
    elapsed = time.perf_counter() - started
    print(f"[colmap] {label}: rc={out.returncode} in {elapsed:.1f}s", flush=True)
    if out.returncode != 0:
        raise ContractError(
            f"{label} failed (rc={out.returncode}): {out.stderr[-1500:]}"
        )
    return {"label": label, "argv": args, "seconds": elapsed,
            "stdout_tail": out.stdout[-4000:]}


def parse_cameras(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(line.split())
    return rows


def parse_images(path: Path) -> list[list[str]]:
    """COLMAP images.txt: two lines per image, the second is POINTS2D."""
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    body = [ln for ln in lines if not ln.strip().startswith("#")]
    index = 0
    while index < len(body):
        parts = body[index].split()
        if len(parts) >= 10:
            rows.append(parts)
            index += 2  # skip the POINTS2D line
        else:
            index += 1
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="one frame's images (READ ONLY)")
    parser.add_argument("--model", required=True, help="supplied cameras.txt + images.txt")
    parser.add_argument("--workdir", required=True, help="all outputs land here")
    parser.add_argument("--max-image-size", type=int, default=NATIVE_WIDTH)
    parser.add_argument("--use-gpu", type=int, default=0)
    args = parser.parse_args(argv)

    if shutil.which(COLMAP) is None:
        raise ContractError("colmap not on PATH")

    images = Path(args.images)
    supplied = Path(args.model)
    work = Path(args.workdir)
    if work.resolve() == REPO_ROOT or REPO_ROOT in work.resolve().parents:
        raise ContractError(f"--workdir {work} is inside the repository")
    if work.exists() and any(work.iterdir()):
        raise ContractError(f"workdir is not empty: {work}")

    cam_rows = parse_cameras(supplied / "cameras.txt")
    if len(cam_rows) != 1:
        raise ContractError(f"expected exactly one supplied camera, got {len(cam_rows)}")
    cam = cam_rows[0]                      # ID MODEL W H p1..pn
    cam_model, cam_w, cam_h = cam[1], cam[2], cam[3]
    cam_params = cam[4:]
    img_rows = parse_images(supplied / "images.txt")
    print(f"[imvid] supplied: camera {cam_model} {cam_w}x{cam_h} "
          f"({len(cam_params)} params), {len(img_rows)} images", flush=True)

    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    model_in = work / "model_in"
    model_out = work / "model_out"
    model_txt = work / "model_txt"
    for directory in (model_in, model_out, model_txt):
        directory.mkdir(parents=True, exist_ok=True)

    steps = []
    steps.append(_run([COLMAP, "database_creator", "--database_path", str(db)],
                      "database_creator"))
    steps.append(_run([
        COLMAP, "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(images),
        "--ImageReader.camera_model", cam_model,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params", ",".join(cam_params),
        "--SiftExtraction.max_image_size", str(args.max_image_size),
        "--SiftExtraction.use_gpu", str(args.use_gpu),
    ], "feature_extractor"))
    steps.append(_run([
        COLMAP, "exhaustive_matcher",
        "--database_path", str(db),
        "--SiftMatching.use_gpu", str(args.use_gpu),
    ], "exhaustive_matcher"))

    # --- identifier remap (values untouched) --------------------------
    with sqlite3.connect(str(db)) as conn:
        db_images = {name: int(iid) for iid, name in
                     conn.execute("SELECT image_id, name FROM images")}
        db_cameras = [int(r[0]) for r in conn.execute("SELECT camera_id FROM cameras")]
    if len(db_cameras) != 1:
        raise ContractError(f"database holds {len(db_cameras)} cameras, expected 1")
    db_camera_id = db_cameras[0]

    remapped = []
    for row in img_rows:
        name = row[9]
        if name not in db_images:
            raise ContractError(
                f"model image {name!r} has no database row; the decoded image "
                "filenames must match images.txt NAME exactly"
            )
        # row = ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        remapped.append([str(db_images[name])] + row[1:8]
                        + [str(db_camera_id), name])

    (model_in / "cameras.txt").write_text(
        "# Camera list\n"
        + " ".join([str(db_camera_id), cam_model, cam_w, cam_h] + cam_params) + "\n",
        encoding="utf-8",
    )
    with (model_in / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list\n")
        for row in remapped:
            handle.write(" ".join(row) + "\n\n")   # blank POINTS2D line
    (model_in / "points3D.txt").write_text("# 3D point list\n", encoding="utf-8")
    print(f"[imvid] remapped {len(remapped)} image ids and camera id -> "
          f"{db_camera_id}; no geometry value touched", flush=True)

    steps.append(_run([
        COLMAP, "point_triangulator",
        "--database_path", str(db),
        "--image_path", str(images),
        "--input_path", str(model_in),
        "--output_path", str(model_out),
        # every guarantee EXPLICIT -- 3.6 defaults are unsafe (see docstring)
        "--Mapper.fix_existing_images", "1",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
    ], "point_triangulator"))
    steps.append(_run([
        COLMAP, "model_converter",
        "--input_path", str(model_out),
        "--output_path", str(model_txt),
        "--output_type", "TXT",
    ], "model_converter"))
    analyzer = _run([COLMAP, "model_analyzer", "--path", str(model_out)],
                    "model_analyzer")
    steps.append(analyzer)

    # --- the decisive check: did anything move? -----------------------
    out_cams = parse_cameras(model_txt / "cameras.txt")
    out_imgs = parse_images(model_txt / "images.txt")
    if len(out_cams) != 1:
        raise ContractError(f"output holds {len(out_cams)} cameras, expected 1")
    cam_delta = max(
        abs(float(a) - float(b)) for a, b in zip(cam_params, out_cams[0][4:])
    )
    supplied_by_name = {row[9]: row for row in img_rows}
    pose_delta = 0.0
    for row in out_imgs:
        ref = supplied_by_name.get(row[9])
        if ref is None:
            raise ContractError(f"output image {row[9]!r} is not in the supplied model")
        pose_delta = max(
            pose_delta,
            max(abs(float(a) - float(b)) for a, b in zip(ref[1:8], row[1:8])),
        )
    print(f"[imvid] calibration drift: intrinsics {cam_delta:.3e}, "
          f"poses {pose_delta:.3e}", flush=True)
    if cam_delta != 0.0 or pose_delta != 0.0:
        raise ContractError(
            f"COLMAP ALTERED THE SUPPLIED CALIBRATION (intrinsics delta "
            f"{cam_delta:.6e}, pose delta {pose_delta:.6e}). The result is "
            "REFUSED: it is consistent with a different camera than the one "
            "the renderer will use."
        )

    # --- statistics ---------------------------------------------------
    points_txt = model_txt / "points3D.txt"
    xyz, track_lengths, per_camera = [], [], {}
    for line in points_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        track = parts[8:]
        track_lengths.append(len(track) // 2)
        for i in range(0, len(track), 2):
            per_camera[track[i]] = per_camera.get(track[i], 0) + 1

    report = {
        "schema": "imvid-sparse-init-v1",
        "images_dir": str(images),
        "supplied_model": str(supplied),
        "workdir": str(work),
        "colmap": {
            "binary": shutil.which(COLMAP),
            "version_note": "3.6 semantics; see the wiki reconciliation",
            "explicit_guarantees": [
                "Mapper.fix_existing_images=1", "Mapper.ba_refine_focal_length=0",
                "Mapper.ba_refine_principal_point=0", "Mapper.ba_refine_extra_params=0",
            ],
            "max_image_size": int(args.max_image_size),
            "use_gpu": int(args.use_gpu),
        },
        "calibration_preserved": {
            "intrinsics_max_abs_delta": cam_delta,
            "pose_max_abs_delta": pose_delta,
            "verified_by": "direct numeric diff (model_comparer absent in 3.6)",
        },
        "points": len(xyz),
        "mean_track_length": (sum(track_lengths) / len(track_lengths)) if track_lengths else 0.0,
        "cameras_with_observations": len(per_camera),
        "cameras_supplied": len(img_rows),
        "observations_per_camera": per_camera,
        "model_analyzer_stdout": analyzer["stdout_tail"],
        "steps": [{k: v for k, v in s.items() if k != "stdout_tail"} for s in steps],
    }
    if xyz:
        import statistics
        for axis, name in enumerate("xyz"):
            values = sorted(v[axis] for v in xyz)
            report.setdefault("spatial_support", {})[name] = {
                "min": values[0], "max": values[-1],
                "p01": values[len(values) // 100],
                "p50": statistics.median(values),
                "p99": values[min(len(values) - 1, 99 * len(values) // 100)],
            }

    report["artifact_hashes"] = {
        p.name: _sha256(p) for p in sorted(model_txt.glob("*.txt"))
    }
    manifest = work / "MANIFEST.imvid_sparse_init.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[imvid] points={report['points']} "
          f"cameras_with_obs={report['cameras_with_observations']}/"
          f"{report['cameras_supplied']} "
          f"mean_track={report['mean_track_length']:.2f}", flush=True)
    print(f"[imvid] manifest -> {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
