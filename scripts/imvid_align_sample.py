#!/usr/bin/env python3
"""Locate the public 300-frame ImViD sample inside a full ImViD take.

WHY THIS EXISTS.  The ImViD paper states "For each scene, we select 300
frames for evaluation" and never says WHICH 300.  The public release ships
a 300-frame ``scene1_opera.zip`` preview.  If that preview is a contiguous
window of the full Opera take, its offset is a hard, checkable fact, and it
is the only empirical handle on window parity that exists without asking
the authors.  This script measures that offset; it does NOT decide whether
the preview is the benchmark clip, which remains unproven.

METHOD.  Both videos are decoded once, through ffmpeg, to a tiny grayscale
raster (default 32x18).  Every frame becomes a fixed-length byte signature.
The sample's signature block is then slid over the full take's and scored by
mean absolute difference.  The reported offset is the argmin.

WHAT MAKES THE ANSWER TRUSTWORTHY, rather than merely the best available:
  * UNIQUENESS is measured, not assumed.  The runner-up outside a guard
    band around the winner is reported, and the ratio between them is the
    separation.  A near-tie means the signal does not identify a window.
  * A SECOND, INDEPENDENT CAMERA is aligned in the same run.  Two cameras
    agreeing on one offset cannot happen by coincidence in a 15,000-frame
    search; disagreement is reported as a refusal, never averaged away.
  * The frame COUNT is checked against ffprobe, so a truncated decode is
    caught instead of silently shortening the search.

The signature raster is deliberately tiny.  The sample is a re-encode, so
pixels are NOT bit-identical to the full take; a coarse luma signature is
robust to that while still being far more specific than a per-frame mean.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def ffprobe_nb_frames(path: str) -> int:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path,
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return int(out)


def decode_signatures(path: str, width: int, height: int) -> np.ndarray:
    """Decode ``path`` to a (N, width*height) uint8 array of luma signatures."""
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", path,
            "-vf", f"scale={width}:{height}:flags=area",
            "-pix_fmt", "gray", "-f", "rawvideo", "-",
        ],
        check=True, capture_output=True,
    )
    frame_bytes = width * height
    buf = proc.stdout
    if len(buf) % frame_bytes != 0:
        raise SystemExit(
            f"REFUSE: {path} decoded to {len(buf)} bytes, not a multiple of "
            f"{frame_bytes}; the decode is truncated or the raster is wrong"
        )
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, frame_bytes)


def best_offset(sample: np.ndarray, full: np.ndarray, guard: int) -> dict:
    """Slide ``sample`` over ``full`` and score by mean absolute difference.

    Scored in float32 on centred signatures so a global exposure shift
    between the preview encode and the take encode cannot dominate.
    """
    n_s = sample.shape[0]
    n_f = full.shape[0]
    if n_f < n_s:
        raise SystemExit("REFUSE: full take is shorter than the sample")

    s = sample.astype(np.float32)
    s = s - s.mean(axis=1, keepdims=True)
    f = full.astype(np.float32)
    f = f - f.mean(axis=1, keepdims=True)

    n_off = n_f - n_s + 1
    scores = np.empty(n_off, dtype=np.float64)
    # Straightforward sliding MAD.  n_off is ~15k and n_s is 300, so this is
    # ~4.5M signature comparisons -- seconds, and far easier to audit than an
    # FFT correlation whose normalisation would need its own proof.
    for off in range(n_off):
        scores[off] = np.abs(f[off : off + n_s] - s).mean()

    win = int(np.argmin(scores))
    masked = scores.copy()
    lo = max(0, win - guard)
    hi = min(n_off, win + guard + 1)
    masked[lo:hi] = np.inf
    runner = int(np.argmin(masked))
    return {
        "offset": win,
        "score": float(scores[win]),
        "runner_up_offset": runner,
        "runner_up_score": float(masked[runner]),
        "separation_ratio": float(masked[runner] / scores[win]) if scores[win] > 0 else float("inf"),
        "median_score": float(np.median(scores)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-root", required=True, help="directory of the 300-frame sample mp4s")
    ap.add_argument("--full-root", required=True, help="directory of the full-take mp4s")
    ap.add_argument("--camera", default="cam00", help="primary camera to align")
    ap.add_argument("--confirm-camera", default="cam20", help="independent second camera")
    ap.add_argument("--sig-width", type=int, default=32)
    ap.add_argument("--sig-height", type=int, default=18)
    ap.add_argument("--guard", type=int, default=30, help="frames around the winner excluded from the runner-up search")
    ap.add_argument("--out", required=True, help="path to write the JSON result")
    args = ap.parse_args()

    result: dict = {
        "sample_root": args.sample_root,
        "full_root": args.full_root,
        "signature_raster": [args.sig_width, args.sig_height],
        "guard": args.guard,
        "cameras": {},
    }

    for role, cam in (("primary", args.camera), ("confirm", args.confirm_camera)):
        sample_path = str(Path(args.sample_root) / f"{cam}.mp4")
        full_path = str(Path(args.full_root) / f"{cam}.mp4")
        print(f"[{role}] {cam}: decoding signatures", flush=True)

        declared_sample = ffprobe_nb_frames(sample_path)
        declared_full = ffprobe_nb_frames(full_path)
        sig_s = decode_signatures(sample_path, args.sig_width, args.sig_height)
        sig_f = decode_signatures(full_path, args.sig_width, args.sig_height)
        if sig_s.shape[0] != declared_sample:
            raise SystemExit(
                f"REFUSE: {sample_path} declared {declared_sample} frames, decoded {sig_s.shape[0]}"
            )
        if sig_f.shape[0] != declared_full:
            raise SystemExit(
                f"REFUSE: {full_path} declared {declared_full} frames, decoded {sig_f.shape[0]}"
            )

        entry = best_offset(sig_s, sig_f, args.guard)
        entry.update(
            camera=cam,
            sample_frames=int(sig_s.shape[0]),
            full_frames=int(sig_f.shape[0]),
        )
        result["cameras"][role] = entry
        print(f"[{role}] {cam}: offset {entry['offset']} score {entry['score']:.4f} "
              f"separation {entry['separation_ratio']:.3f}x", flush=True)

    p = result["cameras"]["primary"]
    c = result["cameras"]["confirm"]
    result["cameras_agree"] = p["offset"] == c["offset"]
    result["offset_delta"] = abs(p["offset"] - c["offset"])
    result["verdict"] = (
        "ALIGNED" if result["cameras_agree"] else "DISAGREEMENT_NO_ALIGNMENT_CLAIMED"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
