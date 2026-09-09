#!/usr/bin/env python
"""Per-frame PSNR profile inside event bounding boxes for a rendered held-out sequence.

Zero-training diagnostic: given the held-out renders and ground truth written by
`main.py --val` (``test/ours_<iter>/renders`` and ``.../gt``) and an event-mask
manifest (``configs/n3v/ladder_event_masks_crb0_299.json``), emit, for EVERY frame,
the whole-frame PSNR and the PSNR restricted to each event's bounding box, plus the
pooled PSNR over each event's own scored windows. The per-frame profile is what
the pooled number hides: whether the return frames dip relative to the frames
before the occlusion, i.e. whether there is any headroom for a presence gate.

Boxes are ``[x0, y0, x1, y1]`` in raster pixels, half-open on the far edge, in the
manifest camera's frame. PSNR uses the pooled MSE over the box and all channels,
values clamped to [0, 1] (the ``--val`` convention).

``--roi_dir`` adds a second, finer region source: per-frame binary ROI masks laid
out as ``<roi_dir>/<name>/<ABSFRAME:05d>.png``, uint8 0/255 at the render's
resolution, one subdirectory per region (e.g. ``core/``, ``ring/``, ``object/``).
Each subdirectory becomes an event named ``roi:<name>`` with
``"kind": "per_frame_mask"``, a ``per_frame_psnr`` list (``null`` on frames whose
mask is empty or absent) and the matching ``pixels_per_frame`` counts. Because a
per-frame mask does NOT contribute a constant pixel count, any pooling over such
an event must weight by ``pixels_per_frame``; the unweighted pooling used for the
fixed bounding boxes is invalid there. ``--masks`` and ``--roi_dir`` are
independent and at least one must be given; the bounding-box output is byte-for-
byte unchanged when ``--roi_dir`` is absent.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from PIL import Image


def _load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _psnr_from_mse(mse):
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def _frame_index(name):
    stem = os.path.splitext(name)[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits)


ROI_PREFIX = "roi:"
ROI_KIND = "per_frame_mask"


def _roi_names(roi_dir):
    """The ROI subdirectory names under `roi_dir`, sorted."""
    names = sorted(
        n for n in os.listdir(roi_dir) if os.path.isdir(os.path.join(roi_dir, n))
    )
    if not names:
        sys.exit(f"no ROI subdirectories under {roi_dir}")
    return names


def _load_roi_mask(path, shape):
    """A boolean (H, W) mask from a uint8 0/255 png, or None when absent."""
    if not os.path.isfile(path):
        return None
    m = np.asarray(Image.open(path).convert("L"))
    if m.shape != shape:
        sys.exit(f"ROI mask shape mismatch on {path}: {m.shape} vs render {shape}")
    return m > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--masks", help="event-mask manifest json (bounding-box events)")
    ap.add_argument(
        "--roi_dir",
        help="directory of per-frame ROI mask subdirs (<roi_dir>/<name>/<frame:05d>.png)",
    )
    ap.add_argument("--out", required=True, help="output json path (a .csv sibling is also written)")
    args = ap.parse_args()

    if not args.masks and not args.roi_dir:
        sys.exit("at least one of --masks and --roi_dir is required")

    if args.masks:
        with open(args.masks) as fh:
            manifest = json.load(fh)
        events = manifest["events"]
        raster = manifest.get("raster")
    else:
        events = []
        raster = None
    roi_names = _roi_names(args.roi_dir) if args.roi_dir else []

    names = sorted(f for f in os.listdir(args.renders) if f.lower().endswith(".png"))
    if not names:
        sys.exit("no renders found")
    frames = [_frame_index(n) for n in names]

    whole = []
    per_event = {e["name"]: [] for e in events}
    sq_sum = {e["name"]: 0.0 for e in events}
    n_sum = {e["name"]: 0 for e in events}
    per_roi = {nm: [] for nm in roi_names}
    roi_pixels = {nm: [] for nm in roi_names}
    shape_checked = False

    for name, fidx in zip(names, frames):
        r = np.clip(_load(os.path.join(args.renders, name)), 0.0, 1.0)
        g = np.clip(_load(os.path.join(args.gt, name)), 0.0, 1.0)
        if r.shape != g.shape:
            sys.exit(f"shape mismatch on {name}: {r.shape} vs {g.shape}")
        if not shape_checked:
            print(f"raster from images: height={r.shape[0]} width={r.shape[1]}; manifest raster={raster}")
            shape_checked = True
        d2 = (r - g) ** 2
        whole.append(_psnr_from_mse(float(d2.mean())))
        for e in events:
            x0, y0, x1, y1 = e["bbox"]
            box = d2[y0:y1, x0:x1, :]
            per_event[e["name"]].append(_psnr_from_mse(float(box.mean())))
            if any(a <= fidx <= b for a, b in e["frames"]):
                sq_sum[e["name"]] += float(box.sum())
                n_sum[e["name"]] += int(box.size)
        for nm in roi_names:
            mask_path = os.path.join(args.roi_dir, nm, "%05d.png" % fidx)
            mask = _load_roi_mask(mask_path, d2.shape[:2])
            npix = 0 if mask is None else int(mask.sum())
            roi_pixels[nm].append(npix)
            if npix == 0:
                per_roi[nm].append(None)
            else:
                per_roi[nm].append(_psnr_from_mse(float(d2[mask].mean())))

    out = {
        "renders": os.path.abspath(args.renders),
        "gt": os.path.abspath(args.gt),
        "masks": os.path.abspath(args.masks) if args.masks else None,
        "n_frames": len(frames),
        "frames": frames,
        "whole_frame_psnr": whole,
        "events": {},
    }
    for e in events:
        nm = e["name"]
        pooled = _psnr_from_mse(sq_sum[nm] / n_sum[nm]) if n_sum[nm] else None
        out["events"][nm] = {
            "bbox": e["bbox"],
            "frames": e["frames"],
            "class": e.get("class"),
            "per_frame_psnr": per_event[nm],
            "pooled_psnr_in_window": pooled,
            "pixel_times_in_window": n_sum[nm] // 3,
        }
    roi_keys = []
    for nm in roi_names:
        key = ROI_PREFIX + nm
        roi_keys.append(key)
        out["events"][key] = {
            "kind": ROI_KIND,
            "roi_name": nm,
            "dir": os.path.abspath(os.path.join(args.roi_dir, nm)),
            "per_frame_psnr": per_roi[nm],
            "pixels_per_frame": roi_pixels[nm],
            "n_frames_with_mask": int(sum(1 for p in roi_pixels[nm] if p > 0)),
        }
    if args.roi_dir:
        out["roi_dir"] = os.path.abspath(args.roi_dir)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "whole_frame"] + [e["name"] for e in events] + roi_keys)
        for i, fidx in enumerate(frames):
            w.writerow(
                [fidx, f"{whole[i]:.4f}"]
                + [f"{per_event[e['name']][i]:.4f}" for e in events]
                + [
                    "" if per_roi[nm][i] is None else f"{per_roi[nm][i]:.4f}"
                    for nm in roi_names
                ]
            )
    print(f"wrote {args.out} and {csv_path}")
    for e in events:
        nm = e["name"]
        print(f"{nm}: pooled_in_window={out['events'][nm]['pooled_psnr_in_window']}")
    for nm in roi_names:
        key = ROI_PREFIX + nm
        print(f"{key}: frames_with_mask={out['events'][key]['n_frames_with_mask']}/{len(frames)}")


if __name__ == "__main__":
    main()
