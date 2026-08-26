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

#: Pose agreement required between the supplied text model and COLMAP's
#: binary output. NOT a refinement tolerance — it forgives ONLY the
#: decimal-string-to-double parse, where two parsers may differ by one
#: ULP (measured worst case 1.11e-16 = 2^-53 on a quaternion component
#: near 0.87). Any bundle adjustment that actually moved a pose would
#: exceed this by many orders of magnitude. Intrinsics get no tolerance
#: at all and are required to match exactly, which they do.
POSE_REPRESENTATION_TOL = 1e-12


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


#: COLMAP camera model ids -> (name, n_params). Only what this lane uses.
_CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
                  2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8)}


def read_cameras_bin(path: Path) -> dict[int, dict]:
    """Exact doubles from `cameras.bin`.

    THE TEXT EXPORT CANNOT BE USED FOR THE EXACTNESS CHECK. COLMAP 3.6's
    text writer emits about SIX SIGNIFICANT FIGURES —
    `2602.2436600602796` comes back as `2602.24` — so a numeric diff
    against `model_txt/cameras.txt` reports ~4e-3 of apparent "drift" on
    a focal length that never moved. That is a property of the writer,
    not of the reconstruction. The binary model stores the raw doubles.
    """
    import struct

    data = path.read_bytes()
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
        cameras[camera_id] = {"model": name, "width": width, "height": height,
                              "params": list(params)}
    return cameras


def read_images_bin(path: Path) -> dict[str, dict]:
    """Exact quaternions/translations from `images.bin`, keyed by NAME."""
    import struct

    data = path.read_bytes()
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
        offset += n_points2d * 24  # (x, y, point3D_id) per observation
        images[name] = {"image_id": image_id, "camera_id": camera_id,
                        "qvec": list(qvec), "tvec": list(tvec),
                        "n_points2d": n_points2d}
    return images


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
    parser.add_argument("--max-num-features", type=int, default=8192,
                        help=(
                            "SiftExtraction.max_num_features. COLMAP's own default "
                            "is 8192, so the default here changes nothing. Exposed "
                            "because exhaustive matching costs O(F^2) per pair and "
                            "the feature CAP -- not the raster -- is what binds on "
                            "detailed scenes: halving F quarters the matching cost "
                            "while leaving keypoint LOCALISATION, and therefore the "
                            "reprojection residual, untouched. Reducing "
                            "--max-image-size instead would buy the same speed by "
                            "degrading precision, which the 2.0 px native gate does "
                            "not have room for."
                        ))
    parser.add_argument("--use-gpu", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true",
                        help="re-run the calibration check and statistics over an "
                             "EXISTING workdir; runs no COLMAP step. Exists so a "
                             "corrected check does not have to repay the "
                             "20-minute exhaustive match.")
    args = parser.parse_args(argv)

    if shutil.which(COLMAP) is None:
        raise ContractError("colmap not on PATH")

    images = Path(args.images)
    supplied = Path(args.model)
    work = Path(args.workdir)
    if work.resolve() == REPO_ROOT or REPO_ROOT in work.resolve().parents:
        raise ContractError(f"--workdir {work} is inside the repository")
    if not args.verify_only and work.exists() and any(work.iterdir()):
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
    if args.verify_only:
        print("[imvid] --verify-only: reusing the existing workdir artifacts",
              flush=True)
        analyzer = _run([COLMAP, "model_analyzer", "--path", str(model_out)],
                        "model_analyzer")
        steps.append(analyzer)
        return _finish(args, images, supplied, work, model_out, model_txt,
                       cam_params, img_rows, analyzer, steps)

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
        "--SiftExtraction.max_num_features", str(args.max_num_features),
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
    return _finish(args, images, supplied, work, model_out, model_txt,
                   cam_params, img_rows, analyzer, steps)


def _finish(args, images, supplied, work, model_out, model_txt,
            cam_params, img_rows, analyzer, steps) -> int:
    """Calibration check, statistics and manifest.

    Split out so `--verify-only` can re-run it over existing artifacts
    without repaying the exhaustive match.
    """
    # --- the decisive check: did anything move? -----------------------
    # Against the BINARY model, never the text export. See
    # `read_cameras_bin` -- 3.6's text writer keeps ~6 significant
    # figures, which manufactures ~4e-3 of phantom "drift" on a focal
    # length that never moved.
    out_cams = read_cameras_bin(model_out / "cameras.bin")
    out_imgs = read_images_bin(model_out / "images.bin")
    if len(out_cams) != 1:
        raise ContractError(f"output holds {len(out_cams)} cameras, expected 1")
    out_cam = next(iter(out_cams.values()))
    supplied_params = [float(v) for v in cam_params]
    if len(out_cam["params"]) != len(supplied_params):
        raise ContractError(
            f"output camera has {len(out_cam['params'])} params, supplied has "
            f"{len(supplied_params)}"
        )
    param_deltas = [abs(a - b) for a, b in zip(supplied_params, out_cam["params"])]
    cam_delta = max(param_deltas)

    supplied_by_name = {row[9]: row for row in img_rows}
    pose_delta = 0.0
    worst_image = None
    for name, entry in out_imgs.items():
        ref = supplied_by_name.get(name)
        if ref is None:
            raise ContractError(f"output image {name!r} is not in the supplied model")
        supplied_pose = [float(v) for v in ref[1:8]]
        produced_pose = entry["qvec"] + entry["tvec"]
        local = max(abs(a - b) for a, b in zip(supplied_pose, produced_pose))
        if local > pose_delta:
            pose_delta, worst_image = local, name

    print(f"[imvid] calibration drift (BINARY model): intrinsics "
          f"{cam_delta:.3e}, poses {pose_delta:.3e}"
          + (f" (worst {worst_image})" if worst_image else ""), flush=True)
    print(f"[imvid]   per-parameter: "
          + ", ".join(f"{d:.2e}" for d in param_deltas), flush=True)

    # INTRINSICS: exact equality, and it IS achieved -- COLMAP copies the
    # camera block through untouched when the refine flags are off, so
    # there is no representation step to forgive.
    #
    # POSES: exact equality is NOT achievable and demanding it would be
    # wrong. The supplied model is TEXT, so both COLMAP and this script
    # parse the same decimal strings into doubles, and the two parsers
    # may land one unit-in-the-last-place apart. The measured worst case
    # is 1.11e-16 on a quaternion component near 0.87 -- exactly 2^-53,
    # one ULP. `POSE_REPRESENTATION_TOL` forgives that and nothing more:
    # it is roughly four orders of magnitude above the observed ULP noise
    # and many orders BELOW any pose change a bundle adjustment could
    # produce, so a real refinement still fails this test loudly.
    if cam_delta != 0.0:
        raise ContractError(
            f"COLMAP ALTERED THE SUPPLIED INTRINSICS (delta {cam_delta:.6e}, "
            f"per-param {param_deltas}). REFUSED: the cloud would be "
            "consistent with a different camera than the renderer's. This "
            "check reads cameras.bin, so it is NOT a text-precision artifact."
        )
    if pose_delta > POSE_REPRESENTATION_TOL:
        raise ContractError(
            f"COLMAP ALTERED THE SUPPLIED POSES (delta {pose_delta:.6e} on "
            f"{worst_image}, tolerance {POSE_REPRESENTATION_TOL:.1e}). "
            "REFUSED. The tolerance forgives decimal-to-double parsing only "
            "(~1e-16); this is far larger and is real movement."
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
            "intrinsics_per_param_delta": param_deltas,
            "intrinsics_rule": "EXACT equality required, no tolerance",
            "pose_max_abs_delta": pose_delta,
            "pose_worst_image": worst_image,
            "pose_tolerance": POSE_REPRESENTATION_TOL,
            "pose_rule": (
                "decimal-to-double parse tolerance ONLY; the observed value is "
                "ULP-scale (2^-53), and any real pose refinement would exceed "
                "this by many orders of magnitude"
            ),
            "verified_by": (
                "cameras.bin / images.bin (raw doubles). The TEXT export is "
                "NOT usable for this: COLMAP 3.6 writes ~6 significant "
                "figures, which manufactures ~4e-3 of phantom drift. "
                "model_comparer does not exist in 3.6 and would only estimate "
                "a best-fit alignment anyway."
            ),
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
