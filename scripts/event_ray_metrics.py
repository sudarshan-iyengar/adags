#!/usr/bin/env python3
"""Event-ray metrics over a --val pass's saved renders (CCR ladder gates).

Reads the renders/ and gt/ PNG directories a `main.py --val` pass writes
under `test/ours_<iter>/`, applies the FROZEN event-ray mask spec
(configs/n3v/ladder_event_masks_crb0_49.json, from
ccr-segment-selection-2026-08-20), and reports pooled PSNR per event
region and for the complement, so a global gain that skips the diagnosed
regions is visible. Pure post-processing: no model, no GPU required.

Frame convention: the --val pass writes one PNG per held-out view in
dataset order; for the N3V ladder that is cam00's frames 0..49 in
timestamp order, so PNG index == frame index. `--frame-offset` exists
for any future window that does not start at 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def pooled_psnr(sq_sum, n):
    if n == 0:
        return None
    mse = sq_sum / n
    if mse <= 0:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def load_pair(renders_dir, gt_dir, index):
    name = "%05d.png" % index
    r = Path(renders_dir) / name
    g = Path(gt_dir) / name
    if not r.is_file() or not g.is_file():
        raise FileNotFoundError(f"missing render/gt pair for frame {index}: {name}")
    render = np.asarray(Image.open(r), dtype=np.float64) / 255.0
    gt = np.asarray(Image.open(g), dtype=np.float64) / 255.0
    return render[..., :3], gt[..., :3]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--masks", required=True,
                        help="frozen event-ray mask spec JSON")
    parser.add_argument("--frame-offset", type=int, default=0,
                        help="PNG index of the mask spec's frame 0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    spec = json.loads(Path(args.masks).read_text())
    lo, hi = spec["window_frames"]

    acc = {e["name"]: [0.0, 0] for e in spec["events"]}
    acc["all_events_union"] = [0.0, 0]
    acc["complement"] = [0.0, 0]
    acc["whole_frame"] = [0.0, 0]

    for frame in range(lo, hi + 1):
        render, gt = load_pair(args.renders_dir, args.gt_dir,
                               frame + args.frame_offset)
        sq = (render - gt) ** 2
        h, w = sq.shape[:2]
        union = np.zeros((h, w), dtype=bool)
        for event in spec["events"]:
            active = any(a <= frame <= b for a, b in event["frames"])
            if not active:
                continue
            x0, y0, x1, y1 = event["bbox"]
            box = np.zeros((h, w), dtype=bool)
            box[y0:y1 + 1, x0:x1 + 1] = True
            union |= box
            acc[event["name"]][0] += float(sq[box].sum())
            acc[event["name"]][1] += int(box.sum()) * 3
        acc["all_events_union"][0] += float(sq[union].sum())
        acc["all_events_union"][1] += int(union.sum()) * 3
        acc["complement"][0] += float(sq[~union].sum())
        acc["complement"][1] += int((~union).sum()) * 3
        acc["whole_frame"][0] += float(sq.sum())
        acc["whole_frame"][1] += sq.size

    result = {
        "schema": "ccr-event-ray-metrics-v1",
        "masks": str(args.masks),
        "renders_dir": str(args.renders_dir),
        "regions": {
            name: {"psnr_pooled": pooled_psnr(s, n), "pixel_times": n // 3}
            for name, (s, n) in acc.items()
        },
    }
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
