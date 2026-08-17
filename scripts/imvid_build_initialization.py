#!/usr/bin/env python3
"""Freeze the ImViD baseline initialization from the frozen-frame clouds.

Step 3 of the ImViD sparse-initialization lane. Takes the per-frame
fixed-pose COLMAP reconstructions produced by
`scripts/imvid_sparse_init.py` and emits the initialization artifact the
trainer would consume, PLUS a declared simple fallback to compare it
against.

--------------------------------------------------------------------
FROZEN RULES
--------------------------------------------------------------------

* UNION BY CONCATENATION over the frozen frames {0, 150, 299}, in
  ascending frame order. NO deduplication, NO subsampling, NO outlier
  trimming beyond what COLMAP already applied.

  Deduplication is deliberately NOT done: it needs a distance threshold,
  a threshold is a tuned parameter, and tuning it against anything would
  make the initialization a fitted object rather than a frozen one. The
  consequence is disclosed rather than hidden — static background is
  triangulated in all three frames and therefore appears about three
  times, so the union is denser on static content than on the performer
  at any one instant. For a Gaussian INITIALIZATION that is a density
  prior, not an error, and pruning is free to remove it.

  This follows the temporal-union principle already used for the
  DiVa-360 visual hull, for the same reason: the model must hold content
  wherever it appears over the window, not only at one instant.

* COLOURS come from COLMAP's own per-point RGB. Unlike the DiVa-360 hull
  — where colours were deliberately randomized to isolate initialization
  GEOMETRY in an A/B — there is no A/B here, so discarding real colour
  would throw away information for no experimental gain.

* THE DECLARED FALLBACK is uniform random sampling inside the COLMAP
  cloud's own p01-p99 axis-aligned box, at the SAME point count and seed
  0. It is deliberately NOT the camera-frustum-union box that performed
  badly on DiVa-360 (measured there at +/-6.5 world units against
  content at +/-1.2, and 84% of its seeds destroyed by iteration 990).
  The fallback is a fair simple baseline — a tight, content-derived box —
  precisely so that "COLMAP beats the fallback" cannot be won by
  comparing against a straw man.

Fail-closed on: a missing per-frame manifest; a frame whose calibration
check did not pass; an empty union; a non-empty output directory; an
output directory inside the git repository.

Usage:
  python3 scripts/imvid_build_initialization.py \
      --sparse-root /apollo/users/sri/proj_adags/data/imvid/sparse \
      --frames 0 150 299 \
      --output-dir /apollo/users/sri/proj_adags/data/imvid/init
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from scripts.diva360_to_blender import write_points3d_ply  # noqa: E402

FROZEN_FRAMES = (0, 150, 299)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_points3d(path: Path):
    """(xyz, rgb, error, track_len) from a COLMAP points3D.txt."""
    xyz, rgb, err, track = [], [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        rgb.append([int(p[4]), int(p[5]), int(p[6])])
        err.append(float(p[7]))
        track.append((len(p) - 8) // 2)
    return (np.asarray(xyz, dtype=np.float64),
            np.asarray(rgb, dtype=np.uint8),
            np.asarray(err, dtype=np.float64),
            np.asarray(track, dtype=np.int32))


def _stats(values: np.ndarray) -> dict:
    ordered = np.sort(values)
    n = len(ordered)
    return {
        "min": float(ordered[0]), "max": float(ordered[-1]),
        "p01": float(ordered[n // 100]),
        "p50": float(statistics.median(ordered)),
        "p99": float(ordered[min(n - 1, 99 * n // 100)]),
        "mean": float(ordered.mean()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-root", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=list(FROZEN_FRAMES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    root = Path(args.sparse_root)
    out = Path(args.output_dir)
    if out.resolve() == REPO_ROOT or REPO_ROOT in out.resolve().parents:
        raise ContractError(f"--output-dir {out} is inside the repository")
    if out.exists() and any(out.iterdir()):
        raise ContractError(f"output dir is not empty: {out}")

    parts, per_frame = [], []
    for frame in args.frames:
        frame_dir = root / f"frame_{frame:06d}"
        manifest = frame_dir / "MANIFEST.imvid_sparse_init.json"
        if not manifest.is_file():
            raise ContractError(f"missing per-frame manifest: {manifest}")
        record = json.loads(manifest.read_text(encoding="utf-8"))
        preserved = record["calibration_preserved"]
        if preserved["intrinsics_max_abs_delta"] != 0.0:
            raise ContractError(
                f"frame {frame} did not preserve intrinsics exactly "
                f"({preserved['intrinsics_max_abs_delta']}); refusing to build "
                "an initialization on it"
            )
        xyz, rgb, err, track = read_points3d(frame_dir / "model_txt" / "points3D.txt")
        if not len(xyz):
            raise ContractError(f"frame {frame} produced no points")
        parts.append((xyz, rgb))
        per_frame.append({
            "frame": int(frame),
            "points": int(len(xyz)),
            "mean_reproj_error": float(err.mean()),
            "mean_track_length": float(track.mean()),
            "sha256_points3D_txt": _sha256(frame_dir / "model_txt" / "points3D.txt"),
        })
        print(f"[imvid] frame {frame}: {len(xyz)} points, "
              f"mean reproj {err.mean():.4f} px", flush=True)

    xyz = np.concatenate([p[0] for p in parts], axis=0)
    rgb = np.concatenate([p[1] for p in parts], axis=0)
    if not len(xyz):
        raise ContractError("union is empty")

    out.mkdir(parents=True, exist_ok=True)
    colmap_ply = out / "points3d_colmap_union.ply"
    write_points3d_ply(colmap_ply, xyz, rgb)

    # --- the declared fallback: uniform in the union's OWN p01-p99 box
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    rng = np.random.default_rng(int(args.seed))
    fb_xyz = lo + rng.random((len(xyz), 3)) * (hi - lo)
    fb_rgb = rng.integers(0, 256, size=(len(xyz), 3), dtype=np.uint8)
    fallback_ply = out / "points3d_fallback_uniform_box.ply"
    write_points3d_ply(fallback_ply, fb_xyz, fb_rgb)

    report = {
        "schema": "imvid-initialization-v1",
        "sparse_root": str(root),
        "frozen_frames": [int(f) for f in args.frames],
        "frozen_rules": {
            "union": "concatenation in ascending frame order",
            "dedup": "NONE -- a distance threshold would be a tuned parameter",
            "subsampling": "NONE",
            "colours": "COLMAP per-point RGB",
            "fallback": (
                "uniform random inside the union's OWN p01-p99 box at the same "
                "point count, seed 0 -- deliberately NOT the frustum-union box "
                "that failed on DiVa-360"
            ),
            "disclosed_consequence": (
                "static content is triangulated in all three frames and so "
                "appears about three times; the union is denser on static "
                "content than on the performer at any one instant"
            ),
        },
        "per_frame": per_frame,
        "union_points": int(len(xyz)),
        "artifacts": {
            "colmap_union": {
                "path": colmap_ply.name, "points": int(len(xyz)),
                "bytes": colmap_ply.stat().st_size, "sha256": _sha256(colmap_ply),
            },
            "fallback_uniform_box": {
                "path": fallback_ply.name, "points": int(len(fb_xyz)),
                "bytes": fallback_ply.stat().st_size, "sha256": _sha256(fallback_ply),
            },
        },
        "spatial_support": {
            "colmap_union": {ax: _stats(xyz[:, i]) for i, ax in enumerate("xyz")},
            "fallback": {ax: _stats(fb_xyz[:, i]) for i, ax in enumerate("xyz")},
        },
        "distribution_note": (
            "reprojection error is defined for the COLMAP cloud only -- the "
            "fallback has no tracks and therefore no reprojection residual. "
            "The comparable axes are spatial support and point distribution."
        ),
    }
    # a crude but honest distribution comparison: occupancy of a 32^3 grid
    for label, pts in (("colmap_union", xyz), ("fallback", fb_xyz)):
        idx = np.floor((pts - lo) / np.maximum(hi - lo, 1e-9) * 32).clip(0, 31)
        occupied = len({tuple(v) for v in idx.astype(int)})
        report.setdefault("grid_occupancy_32", {})[label] = {
            "occupied_cells": int(occupied),
            "of_total": 32 ** 3,
            "fraction": float(occupied / 32 ** 3),
        }

    manifest = out / "MANIFEST.imvid_initialization.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[imvid] union {len(xyz)} points -> {colmap_ply.name}", flush=True)
    print(f"[imvid] fallback {len(fb_xyz)} points -> {fallback_ply.name}", flush=True)
    print(f"[imvid] occupancy(32^3): colmap "
          f"{report['grid_occupancy_32']['colmap_union']['occupied_cells']} vs "
          f"fallback {report['grid_occupancy_32']['fallback']['occupied_cells']}",
          flush=True)
    print(f"[imvid] manifest -> {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
