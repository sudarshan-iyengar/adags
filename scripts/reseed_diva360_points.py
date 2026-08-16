#!/usr/bin/env python3
"""Re-synthesize `points3d.ply` at a different seed count, in place of a
scene's existing one, WITHOUT re-extracting its images.

The DiVa-360 converter samples `points3d.ply` uniformly inside the union
of every camera frustum and discloses that volume as "a coarse
smoke-test volume, NOT a claim-grade initialization". Its measured
extent is about +/-6.5 world units against scissor content at about
+/-1.2, and experiment 84 pruned 20,000 seeds down to 3,254 by iteration
990 — roughly 84% of the cloud destroyed before densification recovered.

The frozen sweep matrix's initialization axis tests the direct
implication: more seeds in the SAME volume leave more survivors. This is
NOT a better prior and is not presented as one; a content-aware
initialization (visual hull from the per-frame masks the converter
already writes) is a different change and deliberately not this one.

Re-running the converter would re-extract several gigabytes of PNGs to
change one file. This instead emits a scene directory whose images,
masks and transforms are RELATIVE SYMLINKS to the source — so the source
stays byte-identical and immutable — and whose `points3d.ply` is
regenerated with the converter's OWN `frustum_union_bounding_box`,
`sample_random_points` and `write_points3d_ply`, so the volume, the RNG
and the file format are the converter's, not a reimplementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from scripts.diva360_to_blender import (  # noqa: E402
    frustum_union_bounding_box,
    sample_random_points,
    write_points3d_ply,
)

#: Files symlinked through from the source scene, never copied.
_LINKED = ("undist", "masks", "transforms_train.json", "transforms_test.json",
           "transforms_val.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="READ ONLY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-points", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0,
                        help="the converter's own sampling seed (default 0, as shipped)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_dir)
    target = Path(args.output_dir)
    train_json = source / "transforms_train.json"
    if not train_json.is_file():
        raise ContractError(f"source scene has no transforms_train.json: {source}")
    if args.num_points <= 0:
        raise ContractError("--num-points must be positive")

    # The bounding box is derived from EVERY frame the splits reference,
    # exactly as the converter does when it builds the scene.
    frames = []
    for name in ("transforms_train.json", "transforms_test.json"):
        path = source / name
        if path.is_file():
            frames.extend(json.loads(path.read_text(encoding="utf-8"))["frames"])
    bbox_min, bbox_max = frustum_union_bounding_box(frames)

    plan = {
        "schema": "diva360-reseed-points-v1",
        "source_dir": str(source),
        "output_dir": str(target),
        "num_points": int(args.num_points),
        "seed": int(args.seed),
        "bbox_min": [float(v) for v in bbox_min],
        "bbox_max": [float(v) for v in bbox_max],
        "bbox_extent": [float(b - a) for a, b in zip(bbox_min, bbox_max)],
        "n_frames_used_for_bbox": len(frames),
        "disclosure": (
            "same coarse frustum-union volume the converter uses; MORE SEEDS, "
            "not a better prior"
        ),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if target.exists() and any(target.iterdir()):
        raise ContractError(f"output dir is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for name in _LINKED:
        src = source / name
        if src.exists():
            os.symlink(os.path.relpath(src, target), target / name)

    points, colors = sample_random_points(
        bbox_min, bbox_max, int(args.num_points), seed=int(args.seed)
    )
    ply = target / "points3d.ply"
    write_points3d_ply(ply, points, colors)
    plan["points3d_ply"] = {
        "path": "points3d.ply",
        "bytes": ply.stat().st_size,
        "sha256": _sha256(ply),
    }
    source_ply = source / "points3d.ply"
    if source_ply.is_file():
        plan["source_points3d_ply_sha256"] = _sha256(source_ply)

    (target / "reseed_provenance.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
