#!/usr/bin/env python3
"""Carve a development split out of the 35 TRAINING cameras.

`transforms_val.json` ships byte-identical to `transforms_test.json` in
every DiVa-360 materialization this repository produces (verified: both
`sha256 f6c372c2…` at 120 FPS, both `sha256 599967a8…` at 30 FPS), so
there is NO development split. Ranking configurations without one means
ranking on the official held-out cameras, which is the single thing that
split exists to prevent.

This emits a new scene directory that:

* REUSES the source materialization's images and masks by RELATIVE
  SYMLINK — no re-extraction, no second copy of ~6 GB of PNGs, and the
  source stays byte-identical and immutable;
* splits the 35 training cameras into `train` (35 - k) and `test` (k),
  where the emitted `test` IS the development split;
* writes NO camera from the official held-out six into either file, so
  those six are unreachable from a run pointed at this directory. They
  cannot leak into selection even by accident.

Ranking happens here. The winner is then retrained on the full 35-camera
scene and scored ONCE on the official six with
`scripts/eval_diva360_heldout.py`. The retrain is deliberate: a model
selected while trained on 30 cameras is not the model the protocol
scores, and evaluating the 30-camera model on the official six would
understate it for a reason that has nothing to do with the axis under
test.

Camera choice is deterministic and outcome-blind: the k dev cameras are
evenly spaced through the sorted training-camera id list, so they span
the rig rather than clustering on one side, and the rule reads only
camera ids.
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


def _camera_of(frame: dict) -> str:
    """The camera directory a frame's file_path names."""
    parts = str(frame["file_path"]).replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("cam"):
            return part
    raise ContractError(f"cannot identify a camera in file_path {frame['file_path']!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True,
                        help="an existing DiVa-360 materialization (READ ONLY)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-cameras", type=int, default=5,
                        help="how many of the 35 training cameras become the dev split")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_dir)
    target = Path(args.output_dir)
    train_json = source / "transforms_train.json"
    test_json = source / "transforms_test.json"
    for path in (train_json, test_json):
        if not path.is_file():
            raise ContractError(f"source split is missing: {path}")

    train = json.loads(train_json.read_text(encoding="utf-8"))
    official_test = json.loads(test_json.read_text(encoding="utf-8"))

    train_cams = sorted({_camera_of(f) for f in train["frames"]})
    official_cams = sorted({_camera_of(f) for f in official_test["frames"]})
    overlap = sorted(set(train_cams) & set(official_cams))
    if overlap:
        raise ContractError(
            f"source train and official test share cameras {overlap}; refusing"
        )
    k = int(args.dev_cameras)
    if not (1 <= k < len(train_cams)):
        raise ContractError(
            f"--dev-cameras must be in [1, {len(train_cams) - 1}], got {k}"
        )

    # Evenly spaced through the sorted id list: spans the rig, reads only ids.
    dev_cams = [train_cams[(i * len(train_cams)) // k] for i in range(k)]
    dev_cams = sorted(dict.fromkeys(dev_cams))
    fit_cams = [c for c in train_cams if c not in set(dev_cams)]

    dev_frames = [f for f in train["frames"] if _camera_of(f) in set(dev_cams)]
    fit_frames = [f for f in train["frames"] if _camera_of(f) in set(fit_cams)]

    plan = {
        "schema": "diva360-dev-split-v1",
        "source_dir": str(source),
        "output_dir": str(target),
        "source_train_cameras": train_cams,
        "official_heldout_cameras_EXCLUDED": official_cams,
        "fit_cameras": fit_cams,
        "dev_cameras": dev_cams,
        "n_fit_frames": len(fit_frames),
        "n_dev_frames": len(dev_frames),
        "selection_rule": (
            "evenly spaced through the sorted training-camera id list; reads "
            "camera ids only, never a metric"
        ),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if target.exists() and any(target.iterdir()):
        raise ContractError(f"output dir is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)

    # Relative symlinks to the source payload; the source is never written.
    for name in ("undist", "masks", "points3d.ply"):
        src = source / name
        if not src.exists():
            continue
        os.symlink(os.path.relpath(src, target), target / name)

    def _emit(name: str, frames: list) -> dict:
        payload = {k2: v for k2, v in train.items() if k2 != "frames"}
        payload["frames"] = frames
        path = target / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)}

    outputs = {
        "train": _emit("transforms_train.json", fit_frames),
        "test": _emit("transforms_test.json", dev_frames),
        "val": _emit("transforms_val.json", dev_frames),
    }
    plan["output_files"] = outputs
    plan["source_transforms_sha256"] = {
        "train": _sha256(train_json), "test": _sha256(test_json),
    }
    provenance = target / "dev_split_provenance.json"
    provenance.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
