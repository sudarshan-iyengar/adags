#!/usr/bin/env python3
"""Assemble one training-arm scene root from a converted ImViD window.

A converted window costs ~11,700 undistorted remaps.  NF and FG differ ONLY
in `points3d.ply`, and the paper and development splits differ only in which
cameras the two transforms files list -- so converting once and assembling
per-arm roots around a SHARED `images/` directory is both far cheaper and
strictly safer than converting four times, because every arm is then reading
literally the same pixels rather than pixels that ought to be the same.

`images/` is a symlink.  `scene/dataset_readers.py` resolves each frame's
`file_path` relative to `source_path`, so a symlinked directory is
indistinguishable from a real one to the reader, and the shared bytes cannot
drift between arms.

THE SPLIT IS RE-DERIVED, NOT RE-TYPED.  The development transforms are built
by re-partitioning the converted scene's own camera list through
`imvid_to_blender.partition_cameras` -- the same function the converter used
-- and then asserted against the WRITTEN bytes with that file's own
`assert_split_on_written`.  Hand-filtering a JSON here would be a second
implementation of the split, and the whole point of the profile table is
that there is only one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imvid_to_blender import (  # noqa: E402
    TEST_JSON,
    TRAIN_JSON,
    ContractError,
    assert_split_on_written,
    dump_transforms,
    partition_cameras,
    split_profile,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-root", required=True,
                    help="the converted window (images/, transforms_*.json)")
    ap.add_argument("--arm-root", required=True, help="destination scene root for this arm")
    ap.add_argument("--ply", required=True, help="this arm's initial population")
    ap.add_argument("--expect-ply-sha256", default=None)
    ap.add_argument("--split-profile", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    scene_root = Path(args.scene_root)
    arm_root = Path(args.arm_root)
    ply = Path(args.ply)

    for required in (scene_root / "images", scene_root / TRAIN_JSON, scene_root / TEST_JSON):
        if not required.exists():
            raise ContractError(f"{required} is absent; --scene-root is not a converted window")
    if not ply.is_file():
        raise ContractError(f"--ply {ply} is absent")
    if (arm_root / "sparse").exists():
        raise ContractError(
            f"{arm_root}/sparse exists; scene/__init__.py:50 dispatches on sparse/ "
            "BEFORE transforms_train.json at :56 and this arm would silently route "
            "into the COLMAP path, which hard-codes cam10 as held out"
        )
    if arm_root.exists() and any(arm_root.iterdir()) and not args.overwrite:
        raise ContractError(f"{arm_root} is not empty; pass --overwrite to replace it")

    digest = sha256_file(ply)
    if args.expect_ply_sha256 and digest != args.expect_ply_sha256:
        raise ContractError(f"ply sha256 {digest} != expected {args.expect_ply_sha256}")

    # --- re-derive the split from the converted scene's own camera list ---
    src_train = json.loads((scene_root / TRAIN_JSON).read_text(encoding="utf-8"))
    src_test = json.loads((scene_root / TEST_JSON).read_text(encoding="utf-8"))
    all_frames = src_train["frames"] + src_test["frames"]
    all_cameras = sorted({str(f["camera"]) for f in all_frames})
    profile = split_profile(args.split_profile)
    part = partition_cameras(profile, all_cameras)

    train_payload = dict(src_train)
    test_payload = dict(src_test)
    train_payload["frames"] = [f for f in all_frames if str(f["camera"]) in set(part["train"])]
    test_payload["frames"] = [f for f in all_frames if str(f["camera"]) in set(part["test"])]
    train_payload["frames"].sort(key=lambda f: (str(f["camera"]), f.get("time", 0.0)))
    test_payload["frames"].sort(key=lambda f: (str(f["camera"]), f.get("time", 0.0)))
    train_bytes = dump_transforms(train_payload)
    test_bytes = dump_transforms(test_payload)

    frames_expected = len({f["file_path"] for f in src_train["frames"]}) // max(
        len({str(f["camera"]) for f in src_train["frames"]}), 1)
    split_record = assert_split_on_written(
        train_bytes, test_bytes, profile, part["train"], frames_expected, all_cameras)

    arm_root.mkdir(parents=True, exist_ok=True)
    link = arm_root / "images"
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            raise ContractError(f"{link} exists and is not a symlink; refusing to replace it")
    try:
        link.symlink_to(os.path.relpath(scene_root / "images", arm_root), target_is_directory=True)
        linked = "symlink"
    except OSError as exc:
        raise ContractError(
            f"could not symlink {link} -> {scene_root / 'images'} ({exc}). Copying "
            "~22 GiB per arm instead is possible but would mean the arms are no "
            "longer reading provably identical bytes; refusing to do it silently."
        ) from exc

    (arm_root / TRAIN_JSON).write_bytes(train_bytes)
    (arm_root / TEST_JSON).write_bytes(test_bytes)
    # `scene/dataset_readers.py:481` requires this exact basename. A mis-named
    # cloud is replaced by a uniform random fill with only a print to show for
    # it, so the copy is verified by hash rather than assumed.
    dst_ply = arm_root / "points3d.ply"
    shutil.copy2(ply, dst_ply)
    copied = sha256_file(dst_ply)
    if copied != digest:
        raise ContractError(f"points3d.ply copy hashed {copied}, source {digest}")

    manifest = {
        "schema": "imvid-arm-assembly-v1",
        "scene_root": str(scene_root),
        "arm_root": str(arm_root),
        "images": linked,
        "split_profile": args.split_profile,
        "held_out": list(profile["held_out"]),
        "excluded": list(profile["excluded"]),
        "train_cameras": part["train"],
        "test_cameras": part["test"],
        "excluded_cameras": part["excluded"],
        "train_frames": len(train_payload["frames"]),
        "test_frames": len(test_payload["frames"]),
        "split_assertion": split_record,
        "ply_source": str(ply),
        "ply_sha256": digest,
        "ply_bytes": dst_ply.stat().st_size,
    }
    if args.manifest:
        mp = Path(args.manifest)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ContractError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        sys.exit(2)
