#!/usr/bin/env python
"""Verify a derived counterfactual-absence scene against its original and manifest.

Exits non-zero and lists EVERY failure it found, not the first: a fixture with
three defects should be repaired in one pass, and a checker that stops at the
first one hides the other two.

The checks, and why each exists:

  1. every ``frames[].file_path`` resolves to a PNG of the declared raster in
     RGB with no alpha channel -- an alpha channel silently changes what the
     loader composites against;
  2. every frame OUTSIDE the authored window is the SAME INODE as the original
     (or, on a copy build, sha-equal) -- this is what makes "only the window
     changed" a fact about the filesystem rather than a claim in a README;
  3. every frame INSIDE the window differs from the original, the difference is
     CONFINED to the construction mask dilated by ``dilate + feather + 2`` px,
     and the mean absolute difference inside that mask is strictly positive --
     an edit that leaked outside its own mask would contaminate the control
     region, and an edit that changed nothing inside it would make the fixture
     vacuous while every file still "differs";
  4. construction masks are non-empty on every margin frame (and carry at least
     500 px on cam15, a camera that sees the object squarely) -- a mask that
     collapsed to a handful of pixels would let check 3 pass trivially;
  5. ``visible_object`` is all-zero inside the window -- the editing tool's own
     statement that the object is gone;
  6. the three evaluation ROIs exist for every margin frame and ``core`` is
     NON-EMPTY on every scored gap frame -- a scored region that is empty on
     some frames produces a pooled number over a varying support;
  7. ``time == frame/30`` on every transforms entry, and cam00 appears only in
     test -- the held-out camera must never enter training;
  8. ``points3d.ply`` matches the manifest sha and no ``flow/``,
     ``motion_priors/`` or ``seg/`` directory exists;
  9. the manifest's per-file shas are re-verified on a seeded random 5% sample
     PLUS every cam00 frame -- cam00 is the evaluation camera, so it is never
     sampled, always checked;
 10. the ORIGINAL tree's image-list signature still matches the manifest.

``--montage`` additionally writes a JPEG contact sheet of original vs derived at
``A-1, A, B, B+1`` for cam00 and two training cameras, with the core and ring
outlines drawn on cam00 and the construction-mask outline on the training
cameras (the ROIs are cam00-only by construction).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_absence_fixture_scene import (  # noqa: E402
    FORBIDDEN_DIRS,
    MASK_THRESHOLD,
    difference_confined,
    dilate,
    frame_token,
    frames_have_expected_time,
    image_list_signature,
    outline,
    parse_image_stem,
    read_mask,
    roi_filename,
    sha256_file,
)

CAM15_MIN_CONSTRUCTION_PX = 500
MONTAGE_TILE_WIDTH = 320


class Failures(object):
    """An accumulator, so one run reports every defect."""

    def __init__(self):
        self.items = []

    def check(self, ok, message):
        if not ok:
            self.items.append(message)
        return bool(ok)

    def add(self, message):
        self.items.append(message)

    def __len__(self):
        return len(self.items)


def load_rgb(path):
    with Image.open(path) as img:
        mode = img.mode
        size = img.size
        arr = np.asarray(img.convert("RGB"))
    return arr, mode, size


def _tile(arr, width):
    img = Image.fromarray(arr)
    if img.width != width:
        height = max(1, int(round(img.height * width / float(img.width))))
        img = img.resize((width, height), Image.BILINEAR)
    return img


def _draw_outline(arr, mask, colour):
    out = np.array(arr, copy=True)
    edge = outline(mask)
    out[edge] = np.asarray(colour, dtype=out.dtype)
    return out


def build_montage(orig_images, out_images, edit, cameras, frames, roi_root, path):
    """A contact sheet: rows are (camera, original/derived), columns are frames."""
    rows = []
    for cam in cameras:
        for which, root in (("orig", orig_images), ("derived", out_images)):
            tiles = []
            for frame in frames:
                name = "%s_%s.png" % (cam, frame_token(frame))
                src = Path(root) / name
                if not src.is_file():
                    tiles.append(None)
                    continue
                arr, _, _ = load_rgb(src)
                if cam == "cam00":
                    core_path = Path(roi_root) / "core" / roi_filename(frame)
                    ring_path = Path(roi_root) / "ring" / roi_filename(frame)
                    if core_path.is_file():
                        arr = _draw_outline(arr, read_mask(core_path), (0, 255, 0))
                    if ring_path.is_file():
                        arr = _draw_outline(arr, read_mask(ring_path), (255, 0, 0))
                else:
                    con = Path(edit) / "construction_masks" / name
                    if con.is_file():
                        arr = _draw_outline(arr, read_mask(con), (0, 128, 255))
                tiles.append(_tile(arr, MONTAGE_TILE_WIDTH))
            rows.append((cam, which, tiles))

    live = [t for _, _, tiles in rows for t in tiles if t is not None]
    if not live:
        return False
    tw, th = live[0].width, live[0].height
    sheet = Image.new("RGB", (tw * len(frames), th * len(rows)), (16, 16, 16))
    for r, (_cam, _which, tiles) in enumerate(rows):
        for c, tile in enumerate(tiles):
            if tile is not None:
                sheet.paste(tile, (c * tw, r * th))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=88)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--montage", default=None, help="write a JPEG contact sheet here")
    ap.add_argument("--sample_fraction", type=float, default=0.05,
                    help="fraction of the non-cam00 manifest entries to re-hash "
                         "(cam00 is ALWAYS re-hashed in full)")
    ap.add_argument("--sample_seed", type=int, default=0,
                    help="seed for the sha re-verification sample")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--cam15_min_px", type=int, default=CAM15_MIN_CONSTRUCTION_PX)
    args = ap.parse_args(argv)

    scene = Path(args.scene_dir)
    orig = Path(args.orig)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fail = Failures()

    orig_images = orig / "images"
    out_images = scene / "images"
    edit = Path(manifest.get("edit", ""))
    window = manifest["window"]
    a, b = int(window[0]), int(window[1])
    margin_lo, margin_hi = manifest["margin_range"]
    raster = manifest.get("raster", [1352, 1014])
    params = manifest.get("edit_params", {})
    # The renderer records its CLI under ``args`` (``dilate`` in px and
    # ``feather`` as a Gaussian sigma in px); a flat layout is accepted too.
    src = params.get("args", params) if isinstance(params, dict) else {}
    dilate_px = int(float(src.get("dilate", params.get("dilate", 0)) or 0))
    feather_sigma = float(src.get("feather", params.get("feather", 0)) or 0)
    slack = dilate_px + int(math.ceil(3.0 * feather_sigma)) + 2

    if not out_images.is_dir():
        fail.add("derived scene has no images/: %s" % out_images)
        print("\n".join(fail.items))
        return 1

    # --- 1 / 7: transforms ---------------------------------------------------
    declared = {}
    for tname, split in (("transforms_train.json", "train"),
                         ("transforms_test.json", "test")):
        path = scene / tname
        if not path.is_file():
            fail.add("missing %s" % tname)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("frames", []):
            fp = entry.get("file_path")
            parsed = parse_image_stem(fp)
            if parsed is None:
                fail.add("%s: unparseable file_path %r" % (tname, fp))
                continue
            cam, frame = parsed
            declared.setdefault((cam, frame), []).append(split)
            img_path = scene / (str(fp) + ".png")
            if not img_path.is_file():
                fail.add("%s: declared image missing: %s" % (tname, img_path))
                continue
            with Image.open(img_path) as img:
                mode, size = img.mode, img.size
            if list(size) != list(raster):
                fail.add("%s: %s is %dx%d, expected %dx%d"
                         % (tname, fp, size[0], size[1], raster[0], raster[1]))
            if mode != "RGB":
                fail.add("%s: %s has mode %s, expected RGB (no alpha)" % (tname, fp, mode))
        for fp, frame, time, ok in frames_have_expected_time(
                payload.get("frames", []), fps=args.fps):
            if not ok:
                fail.add("%s: time %r is not frame/%g for %s (frame %r)"
                         % (tname, time, args.fps, fp, frame))
        if split == "train":
            for entry in payload.get("frames", []):
                parsed = parse_image_stem(entry.get("file_path"))
                if parsed is not None and parsed[0] == "cam00":
                    fail.add("cam00 appears in transforms_train.json: %s"
                             % entry.get("file_path"))
    test_cams = {cam for (cam, _f), splits in declared.items() if "test" in splits}
    fail.check("cam00" in test_cams, "cam00 does not appear in transforms_test.json")

    # --- 2 / 3: per-frame identity vs edit ----------------------------------
    names = sorted(n for n in os.listdir(out_images) if n.lower().endswith(".png"))
    inside_checked = 0
    for name in names:
        parsed = parse_image_stem(name)
        if parsed is None:
            continue
        cam, frame = parsed
        src = orig_images / name
        dst = out_images / name
        if not src.is_file():
            fail.add("derived image has no original counterpart: %s" % name)
            continue
        if not (a <= frame <= b):
            s_src, s_dst = os.stat(src), os.stat(dst)
            shared = (s_src.st_ino == s_dst.st_ino and s_src.st_ino != 0
                      and s_dst.st_nlink >= 2)
            if not shared and sha256_file(src) != sha256_file(dst):
                fail.add("outside the window, %s is neither the same inode nor "
                         "sha-equal to the original" % name)
            continue
        inside_checked += 1
        con_path = edit / "construction_masks" / name
        if not con_path.is_file():
            fail.add("no construction mask for windowed frame %s: %s" % (name, con_path))
            continue
        src_arr, _, _ = load_rgb(src)
        dst_arr, _, _ = load_rgb(dst)
        if src_arr.shape != dst_arr.shape:
            fail.add("%s: derived shape %r != original %r"
                     % (name, dst_arr.shape, src_arr.shape))
            continue
        allowed = dilate(read_mask(con_path), slack)
        confined, outside_px, inside_mean = difference_confined(src_arr, dst_arr, allowed)
        if not confined:
            fail.add("%s: %d changed pixels lie OUTSIDE the construction mask "
                     "dilated by %d px" % (name, outside_px, slack))
        if not np.any(src_arr != dst_arr):
            fail.add("%s: inside the window but byte-identical to the original" % name)
        elif inside_mean <= 0.0:
            fail.add("%s: mean |diff| inside the construction mask is %r" % (name, inside_mean))

    fail.check(inside_checked > 0,
               "no derived frame lies inside the authored window [%d, %d]" % (a, b))

    # --- 4 / 5: the editing tool's own masks --------------------------------
    cameras = sorted({parse_image_stem(n)[0] for n in names if parse_image_stem(n)})
    for frame in range(margin_lo, margin_hi + 1):
        for cam in cameras:
            con_path = edit / "construction_masks" / ("%s_%s.png" % (cam, frame_token(frame)))
            if not con_path.is_file():
                fail.add("missing construction mask %s" % con_path)
                continue
            count = int(np.count_nonzero(read_mask(con_path)))
            if count == 0:
                fail.add("empty construction mask on margin frame: %s" % con_path)
            elif cam == "cam15" and count < args.cam15_min_px:
                fail.add("cam15 construction mask has %d px (< %d) on frame %d"
                         % (count, args.cam15_min_px, frame))
        if a <= frame <= b:
            for cam in cameras:
                vo = edit / "visible_object" / ("%s_%s.png" % (cam, frame_token(frame)))
                if not vo.is_file():
                    fail.add("missing visible_object mask %s" % vo)
                    continue
                with Image.open(vo) as img:
                    arr = np.asarray(img.convert("L"))
                if int(np.count_nonzero(arr)) != 0:
                    fail.add("visible_object is NOT all-zero inside the window: %s" % vo)

    # --- 6: evaluation ROIs --------------------------------------------------
    roi_root = scene / "absfix" / "evaluation_rois"
    for frame in range(margin_lo, margin_hi + 1):
        for kind in ("core", "ring", "object"):
            path = roi_root / kind / roi_filename(frame)
            if not path.is_file():
                fail.add("missing evaluation ROI %s" % path)
        core_path = roi_root / "core" / roi_filename(frame)
        if (a + 3) <= frame <= (b - 2) and core_path.is_file():
            if int(np.count_nonzero(read_mask(core_path))) == 0:
                fail.add("core ROI is EMPTY on scored gap frame %d" % frame)

    # --- 8: ply sha and forbidden directories -------------------------------
    ply = scene / "points3d.ply"
    if not ply.is_file():
        fail.add("missing points3d.ply")
    else:
        fail.check(sha256_file(ply) == manifest.get("points3d_ply_sha256"),
                   "points3d.ply sha does not match the manifest")
    for tname, sha in (manifest.get("transforms_sha256") or {}).items():
        path = scene / tname
        if path.is_file():
            fail.check(sha256_file(path) == sha, "%s sha does not match the manifest" % tname)
    for bad in FORBIDDEN_DIRS:
        fail.check(not (scene / bad).exists(),
                   "forbidden directory present in the derived scene: %s/" % bad)

    # --- 9: sha re-verification ---------------------------------------------
    recorded = manifest.get("file_sha256") or {}
    cam00_names = [n for n in recorded if str(n).startswith("cam00_")]
    others = sorted(n for n in recorded if not str(n).startswith("cam00_"))
    rng = random.Random(args.sample_seed)
    k = int(round(len(others) * float(args.sample_fraction)))
    sample = rng.sample(others, k) if k and others else []
    for name in sorted(set(cam00_names) | set(sample)):
        entry = recorded[name]
        src, dst = orig_images / name, out_images / name
        if src.is_file():
            fail.check(sha256_file(src) == entry.get("before"),
                       "manifest 'before' sha mismatch for %s" % name)
        else:
            fail.add("original file recorded in the manifest is missing: %s" % name)
        if dst.is_file():
            fail.check(sha256_file(dst) == entry.get("after"),
                       "manifest 'after' sha mismatch for %s" % name)
        else:
            fail.add("derived file recorded in the manifest is missing: %s" % name)

    # --- 10: the raw tree ----------------------------------------------------
    if orig_images.is_dir():
        now = image_list_signature(orig_images)
        recorded_after = (manifest.get("raw_tree_untouched") or {}).get("after")
        fail.check(now == recorded_after,
                   "the ORIGINAL images/ signature changed since the build: %r != %r"
                   % (now, recorded_after))
    else:
        fail.add("original images/ directory is missing: %s" % orig_images)

    # --- montage -------------------------------------------------------------
    if args.montage:
        train_cams = sorted({cam for (cam, _f), splits in declared.items()
                             if "train" in splits})[:2]
        montage_cams = ["cam00"] + train_cams
        montage_frames = [a - 1, a, b, b + 1]
        try:
            ok = build_montage(orig_images, out_images, edit, montage_cams,
                               montage_frames, roi_root, args.montage)
            fail.check(ok, "montage had no readable tiles")
        except Exception as exc:
            fail.add("montage failed: %s: %s" % (type(exc).__name__, exc))

    if len(fail):
        print("FAIL: %d check(s)" % len(fail))
        for item in fail.items:
            print("  - %s" % item)
        return 1
    print("OK: %s verifies against %s" % (scene, orig))
    print("  window [%d, %d], margin range [%d, %d], %d images"
          % (a, b, margin_lo, margin_hi, len(names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
