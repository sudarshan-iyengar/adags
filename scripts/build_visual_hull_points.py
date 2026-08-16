#!/usr/bin/env python3
"""TEMPORAL-UNION VISUAL-HULL initialization for a DiVa-360 window scene.

EVERY PARAMETER BELOW WAS FROZEN BEFORE THIS WAS RUN, and before any
outcome was examined. The freeze is recorded in
`research-wiki/operations/diva360-visual-hull-initialization.md`; this
docstring is the second copy, kept next to the code that implements it.

WHY. The converter synthesizes `points3d.ply` by sampling uniformly
inside the union of every camera frustum, and its own docstring calls
that "a coarse smoke-test volume, NOT a claim-grade initialization". Its
extent is about +/-6.5 world units against scissor content at about
+/-1.2. Experiment 84 pruned 20,000 seeds to 3,254 by iteration 990 --
roughly 84% of the cloud destroyed as empty space -- before
densification recovered. The frozen sweep then tested "more seeds in the
same wrong volume" (S1) and it LOST 3.976 dB, which sharpened rather than
weakened the diagnosis: the defect is the VOLUME, not the seed count.
This supplies a volume derived from the scene's own foreground masks.

NOT AN EXTENSION OF `scripts/reseed_diva360_points.py`. That script
resamples the SAME frustum volume at a different count and says so. This
one changes the volume and nothing else.

--------------------------------------------------------------------
THE FROZEN SPECIFICATION
--------------------------------------------------------------------

1. TEMPORAL KEYFRAMES. `--keyframes K` (frozen at 8) indices chosen
   uniformly over the scene's common frame-index list by POSITION, via
   `np.linspace(0, N-1, K)` rounded — never by content, never by motion,
   never adjusted after seeing a hull.

2. CAMERAS. The TRAINING cameras only, read from `transforms_train.json`.
   All 35 of them: this deliberately does NOT apply the tracker's extra
   `cid % 4 == 0` reservation that `build_elgs_tracks.load_temporal_scene`
   imposes. That reservation governs what the TRACKER may observe in the
   evidence lane; an initializer that feeds training may use every camera
   training already sees. The six OFFICIAL held-out cameras are a
   different matter and are excluded by construction: the script reads
   `transforms_test.json` and REFUSES if any camera appears in both
   splits, so held-out imagery can never reach the initialization.

3. MASK THRESHOLD. `> 127`, via `build_elgs_tracks.load_mask` unchanged.
   The converter already binarizes at that threshold when it derives the
   masks from the frame alpha channel, so this is a re-read of a binary
   image, not a second thresholding decision.

4. HULL-AGREEMENT RULE, per keyframe, reusing `TracksConfig`'s shipped
   constants rather than inventing new ones: a voxel is IN the hull when
   it is observed (projects in front of the camera and inside the image)
   by at least `hull_min_observers = 3` training cameras AND at least
   `ceil(hull_mask_agreement * observers) = ceil(0.9 * observers)` of
   those observations land on mask-positive pixels, with at least one
   positive.

5. TEMPORAL UNION. A voxel is in the union hull when it is in the hull at
   ONE OR MORE keyframes. This is the whole point of the lane: a single
   frame's hull covers where the object is at one instant, while the
   model must hold content wherever the object goes over the window.
   NO EROSION. `build_hull_seeds` keeps only the hull SURFACE because a
   tracker wants surface points to track; an initializer wants the
   volume, so the erosion step is deliberately not applied.

6. VOXEL GRID. Resolution 96^3 over a cube of half-extent
   `0.5 * camera-ring radius` centred on the camera-centre centroid --
   `TracksConfig.hull_resolution` and `hull_bounds_scale` unchanged.

7. SAMPLING. Exactly `--num-points` (frozen at 20,000, MATCHING the
   existing initialization so the comparison is capacity-neutral).
   Points are assigned to union voxels by the deterministic stride
   `(i * n_union) // num_points`, which spreads over the voxel list when
   there are more voxels than points and repeats each voxel evenly when
   there are fewer. Each point is then jittered uniformly inside its
   voxel cell, because an unjittered lattice would hand `distCUDA2` a
   degenerate constant nearest-neighbour distance and so a degenerate
   initial Gaussian scale.

8. SEED. 0 -- the converter's own.

9. COLOURS. Uniform random uint8 from the same RNG stream, which is the
   converter's colour rule. Content-aware colour (sampling the observing
   cameras' pixels) is deliberately NOT done: it would be a SECOND change
   and this comparison must isolate initialization GEOMETRY. Colours are
   not inert -- they seed the SH DC term via `RGB2SH` -- so leaving them
   identically distributed to the baseline's is what makes the geometry
   attributable.

10. FAILURE BEHAVIOUR, all fail-closed: a camera present in both splits;
    a missing mask file; an empty hull at ANY keyframe; an empty union;
    a degenerate rig; a requested keyframe outside the common index set;
    a non-empty output directory; an output directory inside the git
    repository; a non-positive point count.

Images, masks and transforms are RELATIVE SYMLINKS to the source, so the
source scene stays byte-identical and no image is re-extracted.

Usage:
  python3 scripts/build_visual_hull_points.py \
      --source-dir <scene> --output-dir <new scene> [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.camera import camera_center  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402
from elgs import diva360_schema as dschema  # noqa: E402

# Reused rather than reimplemented, and the module-private ones
# deliberately so: `_static_rig_matrix` / `_camera_intrinsics` carry the
# OpenGL-c2w-to-OpenCV-w2c convention that this repository has already
# had a frame-convention bug in once. Re-deriving a calibration
# convention in a second place is how the two silently disagree.
from scripts.build_elgs_tracks import (  # noqa: E402
    CameraModel,
    TracksConfig,
    _camera_intrinsics,
    _mask_positive,
    _project_batch,
    _static_rig_matrix,
    load_mask,
)
from scripts.diva360_to_blender import write_points3d_ply  # noqa: E402
from depth_visibility.camera import opengl_c2w_to_opencv_w2c  # noqa: E402

#: Symlinked through from the source scene, never copied.
_LINKED = ("undist", "masks", "transforms_train.json", "transforms_test.json",
           "transforms_val.json")

FROZEN_KEYFRAMES = 8
FROZEN_NUM_POINTS = 20_000
FROZEN_SEED = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_split_cameras(path: Path) -> dict[int, list]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError(f"{path} has no frames")
    per_camera: dict[int, list] = {}
    for frame in frames:
        camera_id = dschema.parse_camera_id(str(frame["file_path"]))
        per_camera.setdefault(camera_id, []).append(frame)
    return per_camera


def _ply_bounds(path: Path) -> dict | None:
    """min/max/extent of an ASCII points3d.ply, for the comparison."""
    if not path.is_file():
        return None
    points = []
    with path.open("r", encoding="utf-8") as handle:
        in_body = False
        for line in handle:
            if not in_body:
                if line.strip() == "end_header":
                    in_body = True
                continue
            parts = line.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not points:
        return None
    array = np.asarray(points)
    return {
        "n": int(len(array)),
        "min": [float(v) for v in array.min(axis=0)],
        "max": [float(v) for v in array.max(axis=0)],
        "extent": [float(v) for v in (array.max(axis=0) - array.min(axis=0))],
    }


def build_union_hull(
    source: Path,
    *,
    keyframes: int,
    cfg: TracksConfig,
) -> dict:
    """The frozen construction. Returns voxels, the union mask and stats."""

    train_path = source / "transforms_train.json"
    test_path = source / "transforms_test.json"
    if not train_path.is_file():
        raise ContractError(f"source scene has no transforms_train.json: {source}")
    train_cams = _load_split_cameras(train_path)

    # FAIL CLOSED on held-out leakage. The official split is whatever the
    # scene's own transforms_test.json declares -- read, not hardcoded --
    # and an id in both files would mean held-out imagery could reach the
    # initialization.
    held_out: list[int] = []
    if test_path.is_file():
        held_out = sorted(_load_split_cameras(test_path))
        overlap = sorted(set(held_out) & set(train_cams))
        if overlap:
            raise ContractError(
                f"cameras {overlap} appear in BOTH transforms_train.json and "
                "transforms_test.json; refusing to initialize from held-out views"
            )

    camera_ids = sorted(train_cams)
    if len(camera_ids) < cfg.hull_min_observers:
        raise ContractError(
            f"{len(camera_ids)} training cameras is below hull_min_observers="
            f"{cfg.hull_min_observers}"
        )

    cameras: dict[int, CameraModel] = {}
    names: dict[int, str] = {}
    index_sets: list[set[int]] = []
    for camera_id in camera_ids:
        entries = train_cams[camera_id]
        match = dschema._CAMERA_ID_RE.search(str(entries[0]["file_path"]))
        name = match.group(0) if match else f"cam{camera_id:02d}"
        names[camera_id] = name
        K, width, height = _camera_intrinsics(entries, name=name)
        cameras[camera_id] = CameraModel(
            camera_id=camera_id,
            name=name,
            K=K,
            w2c=opengl_c2w_to_opencv_w2c(_static_rig_matrix(entries, name=name)),
            width=width,
            height=height,
        )
        index_sets.append(
            {dschema.parse_frame_index(str(e["file_path"])) for e in entries}
        )

    common = sorted(set.intersection(*index_sets))
    if len(common) < keyframes:
        raise ContractError(
            f"{len(common)} common frame indices is fewer than the {keyframes} "
            "frozen keyframes"
        )
    positions = np.unique(np.rint(np.linspace(0, len(common) - 1, keyframes)).astype(int))
    selected = [common[p] for p in positions]

    centers = np.stack([camera_center(cameras[cid].w2c) for cid in camera_ids])
    centroid = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - centroid, axis=1).max())
    if radius <= 0.0:
        raise ContractError("degenerate rig: all camera centers coincide")
    half = radius * cfg.hull_bounds_scale

    res = cfg.hull_resolution
    axis = np.linspace(-half, half, res)
    spacing = float(axis[1] - axis[0]) if res > 1 else 0.0
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    voxels = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1) + centroid

    # Projection is per (camera, voxel) and the rig is STATIC, so it is
    # computed once per camera and reused for every keyframe. Only the
    # mask lookup varies with the frame.
    projections: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    observers = np.zeros(len(voxels), dtype=np.int32)
    for camera_id in camera_ids:
        uv, inside = _project_batch(cameras[camera_id], voxels)
        projections[camera_id] = (uv, inside)
        observers += inside

    agreement_floor = np.ceil(cfg.hull_mask_agreement * observers).astype(np.int32)
    observed_enough = observers >= cfg.hull_min_observers

    union = np.zeros(len(voxels), dtype=bool)
    per_keyframe: list[dict] = []
    for frame_index in selected:
        positives = np.zeros(len(voxels), dtype=np.int32)
        mask_positive_fraction: list[float] = []
        for camera_id in camera_ids:
            mask_path = source / "masks" / names[camera_id] / f"{frame_index:08d}.png"
            mask = load_mask(mask_path)
            mask_positive_fraction.append(float(mask.mean()))
            uv, inside = projections[camera_id]
            positives += _mask_positive(mask, uv, inside)
        in_hull = observed_enough & (positives >= agreement_floor) & (positives > 0)
        if not in_hull.any():
            raise ContractError(
                f"visual hull EMPTY at keyframe {frame_index}: masks and "
                "calibration produced no mutually consistent foreground volume"
            )
        union |= in_hull
        per_keyframe.append({
            "frame_index": int(frame_index),
            "hull_voxels": int(in_hull.sum()),
            "mean_mask_foreground_fraction": float(np.mean(mask_positive_fraction)),
        })

    if not union.any():
        raise ContractError("temporal union hull is empty")

    return {
        "voxels": voxels,
        "union": union,
        "spacing": spacing,
        "camera_ids": camera_ids,
        "held_out_camera_ids": held_out,
        "keyframe_indices": [int(v) for v in selected],
        "n_common_frame_indices": len(common),
        "grid": {
            "resolution": int(res),
            "half_extent": float(half),
            "camera_ring_radius": radius,
            "centroid": [float(v) for v in centroid],
            "voxel_spacing": spacing,
            "total_voxels": int(len(voxels)),
        },
        "per_keyframe": per_keyframe,
        "union_voxels": int(union.sum()),
        "union_occupancy": float(union.sum() / len(voxels)),
    }


def sample_union(hull: dict, num_points: int, seed: int):
    voxels: np.ndarray = hull["voxels"]
    union: np.ndarray = hull["union"]
    spacing: float = hull["spacing"]

    union_idx = np.flatnonzero(union)
    n_union = int(len(union_idx))
    chosen = union_idx[(np.arange(num_points) * n_union) // num_points]
    centers = voxels[chosen]

    rng = np.random.default_rng(seed)
    jitter = (rng.random((num_points, 3)) - 0.5) * spacing
    points = centers + jitter
    colors = rng.integers(0, 256, size=(num_points, 3), dtype=np.uint8)
    return points.astype(np.float64), colors, n_union


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="READ ONLY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-points", type=int, default=FROZEN_NUM_POINTS)
    parser.add_argument("--keyframes", type=int, default=FROZEN_KEYFRAMES)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="build the hull and report statistics; write nothing")
    args = parser.parse_args(argv)

    source = Path(args.source_dir)
    target = Path(args.output_dir)
    if args.num_points <= 0:
        raise ContractError("--num-points must be positive")
    if args.keyframes <= 0:
        raise ContractError("--keyframes must be positive")
    resolved = target.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ContractError(
            f"--output-dir {target} is inside the repository at {REPO_ROOT}; "
            "scenes are never stored in git"
        )

    cfg = TracksConfig()
    hull = build_union_hull(source, keyframes=int(args.keyframes), cfg=cfg)
    points, colors, n_union = sample_union(hull, int(args.num_points), int(args.seed))

    report = {
        "schema": "diva360-visual-hull-init-v1",
        "source_dir": str(source),
        "output_dir": str(target),
        "frozen_specification": {
            "keyframes": int(args.keyframes),
            "keyframe_rule": "uniform by POSITION over the common frame indices",
            "num_points": int(args.num_points),
            "seed": int(args.seed),
            "cameras": "all TRAINING cameras; the tracker's cid%4 reservation is NOT applied",
            "mask_threshold": "> 127 (build_elgs_tracks.load_mask, unchanged)",
            "hull_min_observers": int(cfg.hull_min_observers),
            "hull_mask_agreement": float(cfg.hull_mask_agreement),
            "hull_resolution": int(cfg.hull_resolution),
            "hull_bounds_scale": float(cfg.hull_bounds_scale),
            "temporal_rule": "UNION over keyframes; NO erosion (volume, not surface)",
            "colors": "uniform random uint8, the converter's rule -- geometry is the only change",
        },
        "n_training_cameras": len(hull["camera_ids"]),
        "training_camera_ids": hull["camera_ids"],
        "held_out_camera_ids_excluded": hull["held_out_camera_ids"],
        "keyframe_indices": hull["keyframe_indices"],
        "n_common_frame_indices": hull["n_common_frame_indices"],
        "grid": hull["grid"],
        "per_keyframe": hull["per_keyframe"],
        "union_voxels": hull["union_voxels"],
        "union_occupancy": hull["union_occupancy"],
        "points_per_union_voxel": float(args.num_points) / max(1, n_union),
        "sampled_bounds": {
            "min": [float(v) for v in points.min(axis=0)],
            "max": [float(v) for v in points.max(axis=0)],
            "extent": [float(v) for v in (points.max(axis=0) - points.min(axis=0))],
        },
        "baseline_points3d_ply_bounds": _ply_bounds(source / "points3d.ply"),
    }

    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if target.exists() and any(target.iterdir()):
        raise ContractError(f"output dir is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for name in _LINKED:
        src = source / name
        if src.exists():
            os.symlink(os.path.relpath(src, target), target / name)

    ply = target / "points3d.ply"
    write_points3d_ply(ply, points, colors)
    report["points3d_ply"] = {
        "path": "points3d.ply",
        "bytes": ply.stat().st_size,
        "sha256": _sha256(ply),
    }
    source_ply = source / "points3d.ply"
    if source_ply.is_file():
        report["source_points3d_ply_sha256"] = _sha256(source_ply)

    (target / "visual_hull_provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
