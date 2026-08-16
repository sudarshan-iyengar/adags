#!/usr/bin/env python3
"""Slice a DiVa-360 window scene down to ONE frame index.

For the FROZEN per-frame Gaussian oracle. The oracle asks a single
question: how well can this renderer and this initialization fit a
scissor frame when the temporal representation is removed entirely? Each
selected frame becomes an independent STATIC multi-view reconstruction
problem — 35 training views of one instant, 6 official held-out views of
the same instant — so a strong per-frame result with a weak dynamic
result implicates the temporal representation or the motion
optimization, while a weak per-frame result implicates initialization,
rendering, training or evaluation BEFORE any temporal question.

FROZEN BEFORE TRAINING, and recorded in
`research-wiki/operations/diva360-visual-hull-initialization.md`'s
companion oracle section:

  * frame selection is `np.linspace(0, N-1, K)` rounded over the scene's
    sorted frame indices — uniform by POSITION, never by content, never
    revisited after seeing a result. This is deliberately the SAME rule
    the visual-hull keyframes use, so the oracle frames and the hull
    keyframes coincide and the two lanes cannot drift apart.
  * the camera split is untouched: whatever `transforms_train.json` and
    `transforms_test.json` already declare.
  * `points3d.ply` is symlinked from the source, so the oracle inherits
    whichever initialization the source scene carries. Pointing this at
    the visual-hull scene gives the oracle the hull initialization the
    directive specifies; pointing it at the stock scene gives the
    frustum one. That choice lives in the caller, not here.

The slice keeps `time` on every frame entry EXACTLY as the source wrote
it, and does not renormalize it. A single-frame scene has one timestamp,
so any monotone rescaling would be arbitrary; leaving it alone means the
slice differs from the source in exactly one way — which frames are
present.

Images and masks are RELATIVE SYMLINKS to the source directory, so the
source stays byte-identical and nothing is re-extracted. Only the three
transforms files are rewritten, and only by FILTERING their `frames`
list.

Usage:
  python3 scripts/make_diva360_frame_slice.py \
      --source-dir <scene> --output-root <dir> --frames 8
  python3 scripts/make_diva360_frame_slice.py \
      --source-dir <scene> --output-root <dir> --frame-indices 0 80 160
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from elgs import diva360_schema as dschema  # noqa: E402

_SPLITS = ("train", "test", "val")
#: Symlinked through; `points3d.ply` included so the slice inherits the
#: source scene's initialization rather than inventing one.
_LINKED = ("undist", "masks", "points3d.ply")


def _read_split(source: Path, split: str):
    path = source / f"transforms_{split}.json"
    if not path.is_file():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError(f"{path} has no frames")
    return payload, frames


def select_frame_indices(source: Path, count: int) -> list[int]:
    """Uniform by POSITION over the TRAIN split's sorted frame indices."""
    _, frames = _read_split(source, "train")
    if frames is None:
        raise ContractError(f"source scene has no transforms_train.json: {source}")
    indices = sorted({dschema.parse_frame_index(str(f["file_path"])) for f in frames})
    if count > len(indices):
        raise ContractError(
            f"asked for {count} frames but the scene has {len(indices)}"
        )
    positions = np.unique(np.rint(np.linspace(0, len(indices) - 1, count)).astype(int))
    return [indices[p] for p in positions]


def write_slice(source: Path, target: Path, frame_index: int) -> dict:
    if target.exists() and any(target.iterdir()):
        raise ContractError(f"output dir is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for name in _LINKED:
        src = source / name
        if src.exists():
            os.symlink(os.path.relpath(src, target), target / name)

    counts: dict[str, int] = {}
    cameras: dict[str, int] = {}
    for split in _SPLITS:
        payload, frames = _read_split(source, split)
        if payload is None:
            continue
        kept = [
            f for f in frames
            if dschema.parse_frame_index(str(f["file_path"])) == frame_index
        ]
        if not kept:
            raise ContractError(
                f"frame {frame_index} is absent from the {split} split"
            )
        payload["frames"] = kept
        (target / f"transforms_{split}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        counts[split] = len(kept)
        cameras[split] = len(
            {dschema.parse_camera_id(str(f["file_path"])) for f in kept}
        )
    return {"frame_index": int(frame_index), "units": counts, "cameras": cameras}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="READ ONLY")
    parser.add_argument("--output-root", required=True,
                        help="one subdirectory is created per selected frame")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frames", type=int, help="how many, uniform by position")
    group.add_argument("--frame-indices", type=int, nargs="+", help="explicit indices")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    source = Path(args.source_dir)
    root = Path(args.output_root)
    resolved = root.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ContractError(
            f"--output-root {root} is inside the repository at {REPO_ROOT}"
        )

    if args.frames is not None:
        selected = select_frame_indices(source, int(args.frames))
        rule = f"uniform by position, {args.frames} of the scene's frame indices"
    else:
        selected = sorted(set(int(v) for v in args.frame_indices))
        rule = "explicit --frame-indices"

    report = {
        "schema": "diva360-frame-slice-v1",
        "source_dir": str(source),
        "output_root": str(root),
        "selection_rule": rule,
        "frame_indices": [int(v) for v in selected],
        "slices": [],
    }

    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    for frame_index in selected:
        target = root / f"frame_{frame_index:08d}"
        entry = write_slice(source, target, frame_index)
        entry["dir"] = str(target)
        report["slices"].append(entry)
        print(f"[slice] {target}  {entry['units']}  cameras={entry['cameras']}",
              flush=True)

    root.mkdir(parents=True, exist_ok=True)
    (root / "frame_slice_provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
