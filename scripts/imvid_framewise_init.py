#!/usr/bin/env python3
"""Per-frame sparse triangulation across an ImViD window's training views.

Produces ONE fixed-pose COLMAP reconstruction per frame of a converted
window, using training cameras only, and writes each frame's points with
that frame's timestamp attached.  The union of these is the candidate
geometry both initialization arms are built from.

WHY PER-FRAME AND NOT A UNION.  The existing 20,157-point artifact is a
union over three frames (0 / 150 / 299), which was enough to prove the
loader and the calibration and is not a 300-frame initializer.  The paper's
initialization is per-frame sparse geometry, and the FG arm's whole
mechanism is deciding, per point, whether it is static (initialize once) or
dynamic (initialize at its own observation time).  A pre-unioned cloud has
already thrown away the frame identity that decision needs.

WHY IT RUNS ON THE UNDISTORTED PINHOLE RASTER.  Triangulating at native
5312x2988 costs ~21 minutes per frame, essentially all of it CPU SIFT
matching -- 300 frames x 2 scenes would be ~210 hours.  The undistorted 2656
x1494 raster has a quarter of the pixels, and exhaustive matching cost falls
roughly with the SQUARE of the feature count.  It is also the raster the
model actually trains on, so the geometry and the supervision share a
frame.  Poses are unchanged by undistortion (a pure image warp about the
same camera centre), so the supplied extrinsics stay authoritative.

THE CAMERA IS NOT RE-DERIVED HERE.  `derive_output_camera` is imported from
`imvid_to_blender` -- the same function that produced the undistorted
images.  Two implementations of that arithmetic would be two chances for the
COLMAP camera and the image content to disagree by a quarter pixel, and
nothing downstream could detect it.

Every COLMAP guarantee from `imvid_sparse_init.py` is inherited by invoking
that script per frame rather than reimplementing it: poses fixed
(`Mapper.fix_existing_images 1`, which on COLMAP 3.6 defaults to 0 and would
otherwise let a bundle adjustment move the supplied poses), intrinsics
fixed, and a numeric diff of the OUTPUT binary model against the supplied
calibration.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imvid_to_blender import (  # noqa: E402
    ContractError,
    camera_token,
    derive_output_camera,
    parse_cameras_txt,
    parse_images_txt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPARSE_INIT = REPO_ROOT / "scripts" / "imvid_sparse_init.py"


def pinhole_cameras_txt(cameras: dict[int, dict], scale: float) -> str:
    """Render the derived PINHOLE cameras.txt the undistorted images live in."""
    lines = []
    for cid in sorted(cameras):
        out = derive_output_camera(cameras[cid], scale)
        # K_new is the SAME 3x3 the resampling map and the transforms JSON are
        # built from, so the COLMAP camera cannot drift from the pixels it is
        # asked to explain.
        k = out["K_new"]
        lines.append(
            f"{cid} PINHOLE {out['width']} {out['height']} "
            f"{float(k[0, 0])!r} {float(k[1, 1])!r} "
            f"{float(k[0, 2])!r} {float(k[1, 2])!r}"
        )
    return "\n".join(lines) + "\n"


def images_txt_subset(images: dict[str, dict], keep: set[str]) -> str:
    """Re-render images.txt for a camera subset, preserving every pose field.

    COLMAP's text images.txt is two lines per image: the pose line and a
    POINTS2D line.  ImViD ships the second one empty, and it must stay
    present -- a missing blank line silently shifts the parse.
    """
    out = []
    for name, entry in images.items():
        if camera_token(name) not in keep:
            continue
        q = entry["qvec"]
        t = entry["tvec"]
        out.append(
            f"{entry['image_id']} {q[0]!r} {q[1]!r} {q[2]!r} {q[3]!r} "
            f"{t[0]!r} {t[1]!r} {t[2]!r} {entry['camera_id']} {name}"
        )
        out.append("")
    return "\n".join(out) + "\n"


def stage_frame(images_root: Path, frame: int, cameras: list[str], dest: Path) -> int:
    """Link/copy one frame's images under the bare names images.txt uses."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for cam in cameras:
        src = images_root / f"{cam}_{frame:06d}.png"
        if not src.is_file():
            raise ContractError(f"missing {src}")
        dst = dest / f"{cam}.png"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        n += 1
    return n


def run_one_frame(job: dict) -> dict:
    frame = job["frame"]
    work = Path(job["workdir"]) / f"frame_{frame:06d}"
    if work.exists():
        shutil.rmtree(work)
    images_dir = work / "images"
    model_dir = work / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    n_img = stage_frame(Path(job["images_root"]), frame, job["cameras"], images_dir)
    (model_dir / "cameras.txt").write_text(job["cameras_txt"], encoding="utf-8")
    (model_dir / "images.txt").write_text(job["images_txt"], encoding="utf-8")
    (model_dir / "points3D.txt").write_text("", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SPARSE_INIT),
         "--images", str(images_dir), "--model", str(model_dir),
         "--workdir", str(work / "out"),
         "--max-image-size", str(job["max_image_size"]), "--use-gpu", "0"],
        capture_output=True, text=True,
    )
    record = {
        "frame": frame,
        "cameras_staged": n_img,
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - started, 2),
    }
    if proc.returncode != 0:
        record["stderr_tail"] = proc.stderr[-2000:]
        record["stdout_tail"] = proc.stdout[-2000:]
        return record

    summary = work / "out" / "MANIFEST.imvid_sparse_init.json"
    if summary.is_file():
        record["sparse_manifest"] = json.loads(summary.read_text(encoding="utf-8"))

    ply = _collect_points(work / "out", frame, job["time_of_frame"], Path(job["out_root"]))
    record.update(ply)
    if job.get("cleanup", True):
        shutil.rmtree(images_dir, ignore_errors=True)
    return record


def _collect_points(out_dir: Path, frame: int, t: float, out_root: Path) -> dict:
    """Read the frame's triangulated points and store them with their time."""
    from plyfile import PlyData

    candidates = sorted(out_dir.rglob("points3D.ply")) + sorted(out_dir.rglob("*.ply"))
    if not candidates:
        # fall back to the text model COLMAP always writes
        txt = sorted(out_dir.rglob("points3D.txt"))
        if not txt:
            raise ContractError(f"no point output under {out_dir}")
        xyz, rgb = [], []
        for line in txt[0].read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
            rgb.append([int(p[4]), int(p[5]), int(p[6])])
        pts = np.asarray(xyz, dtype=np.float64)
        col = np.asarray(rgb, dtype=np.uint8)
    else:
        data = PlyData.read(str(candidates[0]))["vertex"]
        pts = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
        col = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)

    out_root.mkdir(parents=True, exist_ok=True)
    dst = out_root / f"frame_{frame:06d}.npz"
    np.savez_compressed(dst, xyz=pts.astype(np.float32), rgb=col,
                        frame=np.int32(frame), time=np.float64(t))
    return {"points": int(pts.shape[0]), "cloud": str(dst)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-root", help="converted scene images/ (undistorted, READ ONLY)")
    ap.add_argument("--model", help="SUPPLIED calibration dir (cameras.txt + images.txt)")
    ap.add_argument("--out-root", help="per-frame cloud destination")
    ap.add_argument("--workdir", help="scratch root for COLMAP")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--fps-rational", default="60000/1001")
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frame-count", type=int, default=300)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--exclude-cameras", default="cam00",
                    help="cameras that MUST NOT contribute an observation")
    ap.add_argument("--max-image-size", type=int, default=2656)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--keep-staged", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()
    for required in ("images_root", "model", "out_root", "workdir"):
        if not getattr(args, required):
            raise ContractError(f"--{required.replace('_', '-')} is required")

    model_dir = Path(args.model)
    cameras = parse_cameras_txt((model_dir / "cameras.txt").read_text(encoding="utf-8"))
    images = parse_images_txt((model_dir / "images.txt").read_text(encoding="utf-8"))

    exclude = tuple(c.strip() for c in args.exclude_cameras.split(",") if c.strip())
    all_names = sorted(camera_token(n) for n in images)
    missing = [c for c in exclude if c not in all_names]
    if missing:
        raise ContractError(
            f"excluded camera(s) {missing} are absent from the model; an "
            "exclusion that matches nothing protects nothing"
        )
    keep = [n for n in all_names if n not in exclude]
    if not keep:
        raise ContractError("every camera was excluded")

    num, den = (int(x) for x in args.fps_rational.split("/"))
    cams_txt = pinhole_cameras_txt(cameras, args.scale)
    imgs_txt = images_txt_subset(images, set(keep))

    frames = list(range(args.frame_start,
                        args.frame_start + args.frame_count,
                        args.frame_stride))
    jobs = [{
        "frame": f,
        "time_of_frame": f * den / num,
        "images_root": args.images_root,
        "out_root": args.out_root,
        "workdir": args.workdir,
        "cameras": keep,
        "cameras_txt": cams_txt,
        "images_txt": imgs_txt,
        "max_image_size": args.max_image_size,
        "cleanup": not args.keep_staged,
    } for f in frames]

    started = time.time()
    records: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rec in pool.map(run_one_frame, jobs):
            records.append(rec)
            status = "ok" if rec["returncode"] == 0 else "FAIL"
            print(f"[frame {rec['frame']:>4}] {status} "
                  f"points={rec.get('points', '-')} {rec['elapsed_s']}s", flush=True)

    failed = [r for r in records if r["returncode"] != 0]
    ok = [r for r in records if r["returncode"] == 0]
    manifest = {
        "schema": "imvid-framewise-init-v1",
        "images_root": args.images_root,
        "model": args.model,
        "scale": args.scale,
        "fps_rational": args.fps_rational,
        "frames_requested": len(frames),
        "frames_ok": len(ok),
        "frames_failed": [r["frame"] for r in failed],
        "cameras_used": keep,
        "excluded_cameras": list(exclude),
        "max_image_size": args.max_image_size,
        "total_points": int(sum(r.get("points", 0) for r in ok)),
        "points_per_frame": {str(r["frame"]): r.get("points") for r in ok},
        "elapsed_s": round(time.time() - started, 2),
        "records": records,
    }
    if args.manifest:
        mp = Path(args.manifest)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"},
                     indent=2, sort_keys=True))
    # A partial run is a failure, not a smaller success: a window missing
    # frames would silently become a sparser initializer.
    return 1 if failed else 0


def _check(name: str, ok: bool, detail) -> dict:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return {"name": name, "ok": bool(ok), "detail": detail}


def run_self_test() -> int:
    results = []
    cams = parse_cameras_txt(
        "2 OPENCV 5312 2988 2603.33268646004 2602.2436600602796 2656.0 1494.0 "
        "-0.024546867645992888 0.0035148158874614976 -0.0004507998572363207 "
        "-0.00023832152424359775\n"
    )
    txt = pinhole_cameras_txt(cams, 0.5)
    parts = txt.split()
    results.append(_check("pinhole_model_and_raster",
                          parts[1] == "PINHOLE" and parts[2] == "2656" and parts[3] == "1494",
                          " ".join(parts[:4])))
    cx = float(parts[6])
    results.append(_check("principal_point_uses_pixel_centre_convention",
                          abs(cx - ((2656.0 + 0.5) * 0.5 - 0.5)) < 1e-9,
                          f"cx={cx} vs (c+0.5)*s-0.5={(2656.0 + 0.5) * 0.5 - 0.5}"))
    results.append(_check("naive_divide_would_differ_by_a_quarter_pixel",
                          abs(cx - 2656.0 * 0.5) > 0.2, f"naive={2656.0 * 0.5}"))

    num, den = 60000, 1001
    t299 = 299 * den / num
    results.append(_check("frame_time_uses_measured_rational",
                          abs(t299 - 4.988316666666667) < 1e-12, t299))
    results.append(_check("thirty_fps_would_be_wrong_by_1_998",
                          abs((299 / 30.0) / t299 - 1.998001998) < 1e-6,
                          (299 / 30.0) / t299))

    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ContractError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        sys.exit(2)
