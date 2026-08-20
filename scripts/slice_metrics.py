#!/usr/bin/env python3
"""Per-frame and slice metrics over a --val pass's saved renders (B0-C).

Post-processing companion to scripts/event_ray_metrics.py for the
300-frame canonical benchmark (b0c-canonical-300f-2026-08-20): computes
per-frame pooled PSNR from the renders/ and gt/ PNG directories, then
reports fixed 50-frame slice means (pooled over each slice's pixels),
per-frame minimum, median, worst-decile mean, and the frame indices of
the worst decile (late-sequence degradation diagnostic). 8-bit PNG
basis, disclosed; the primary endpoint remains the --val float metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def pooled_psnr(mse):
    if mse <= 0:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--slice_size", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    per_frame = []
    slice_acc = {}
    for frame in range(args.num_frames):
        name = "%05d.png" % frame
        r = np.asarray(Image.open(Path(args.renders_dir) / name),
                       dtype=np.float64)[..., :3] / 255.0
        g = np.asarray(Image.open(Path(args.gt_dir) / name),
                       dtype=np.float64)[..., :3] / 255.0
        sq = float(((r - g) ** 2).sum())
        n = r.size
        per_frame.append(pooled_psnr(sq / n))
        s = frame // args.slice_size
        acc = slice_acc.setdefault(s, [0.0, 0])
        acc[0] += sq
        acc[1] += n

    arr = np.array(per_frame)
    order = np.argsort(arr)
    decile = max(1, len(arr) // 10)
    worst_idx = sorted(int(i) for i in order[:decile])
    result = {
        "schema": "b0c-slice-metrics-v1",
        "renders_dir": str(args.renders_dir),
        "num_frames": len(arr),
        "slices": {
            "%d-%d" % (s * args.slice_size,
                       min((s + 1) * args.slice_size, args.num_frames) - 1):
            pooled_psnr(acc[0] / acc[1])
            for s, acc in sorted(slice_acc.items())
        },
        "per_frame": {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "worst_decile_mean": float(arr[order[:decile]].mean()),
            "worst_decile_frames": worst_idx,
            "max": float(arr.max()),
        },
    }
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
