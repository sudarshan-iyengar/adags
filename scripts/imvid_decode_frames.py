#!/usr/bin/env python3
"""Decode a FROZEN frame set from the ImViD sample, one image per camera.

Step 1 of the ImViD sparse-initialization lane. Produces the images that
COLMAP will later extract features from, and nothing else — no features,
no database, no reconstruction.

--------------------------------------------------------------------
FROZEN BEFORE THE FIRST DECODE
--------------------------------------------------------------------

* FRAME RULE: `np.linspace(0, n_frames - 1, K)` rounded, K frozen at
  **3**, giving **0, 150, 299** over the sample's 300 frames — first,
  middle, last. Uniform by POSITION, never by content, never revisited
  after seeing a reconstruction. This is the same family of rule the
  DiVa-360 hull keyframes and the per-frame oracle use, deliberately.

* PER-FRAME, NEVER ACROSS FRAMES. Each frozen frame is triangulated as
  its OWN static multi-view problem, using all 39 cameras at that one
  instant. Matching features ACROSS frames would assume a static scene,
  which an opera performance is not; the rig is static, the content is
  not. The three clouds are combined later by UNION, following the same
  temporal-union principle as the DiVa-360 visual hull, so no
  post-hoc "best frame" is ever chosen.

* NATIVE RESOLUTION ONLY. Frames are decoded at 5312x2988 and the
  decoded size is CHECKED against that, fail-closed. The supplied
  intrinsics (`2 OPENCV 5312 2988 ...`) describe that raster; any
  silent rescale would put the correspondences in a different frame from
  the calibration.

* Decoding is by **ffmpeg** (`/usr/bin/ffmpeg`, verified present in the
  Determined runtime), not `cv2.VideoCapture`. The MP4s are not
  `faststart` — `ffprobe` on a prefix returns `moov atom not found` —
  so the whole file must be local, which it is after extraction.

Fail-closed on: a missing camera video; an `ffprobe` frame count other
than the declared 300; a decoded image whose size is not 5312x2988; a
frame index outside the declared count; a non-empty output directory; an
output directory inside the git repository.

Usage:
  python3 scripts/imvid_decode_frames.py \
      --source-dir /apollo/users/sri/proj_adags/data/imvid/scene1_opera \
      --output-root /apollo/users/sri/proj_adags/data/imvid/frames
  python3 scripts/imvid_decode_frames.py ... --frames 1   # measure cost first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402

FROZEN_FRAME_COUNT = 3
DECLARED_FRAMES = 300
DECLARED_WIDTH = 5312
DECLARED_HEIGHT = 2988


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffprobe_streams(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate,codec_name,pix_fmt",
         "-of", "json", str(video)],
        capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0:
        raise ContractError(f"ffprobe failed on {video}: {out.stderr[:400]}")
    streams = json.loads(out.stdout).get("streams") or []
    if not streams:
        raise ContractError(f"no video stream in {video}")
    return streams[0]


def select_frame_indices(count: int, n_frames: int = DECLARED_FRAMES) -> list[int]:
    if count > n_frames:
        raise ContractError(f"asked for {count} frames of {n_frames}")
    positions = np.unique(np.rint(np.linspace(0, n_frames - 1, count)).astype(int))
    return [int(v) for v in positions]


def decode_frame(video: Path, frame_index: int, target: Path) -> None:
    """Exact frame by index, native resolution, PNG.

    `select='eq(n,IDX)'` with `-vsync 0` picks the frame by DECODE
    ORDER INDEX rather than by timestamp, so it does not depend on the
    container's frame-rate metadata being exact.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vf", f"select=eq(n\\,{frame_index})", "-vsync", "0",
         "-frames:v", "1", str(target)],
        capture_output=True, text=True, timeout=1800,
    )
    if out.returncode != 0 or not target.is_file():
        raise ContractError(
            f"ffmpeg failed for {video} frame {frame_index}: {out.stderr[:400]}"
        )


def _png_size(path: Path) -> tuple[int, int]:
    """Width/height from the PNG IHDR — no image library needed."""
    with path.open("rb") as handle:
        header = handle.read(33)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ContractError(f"{path} is not a PNG")
    return (
        int.from_bytes(header[16:20], "big"),
        int.from_bytes(header[20:24], "big"),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="READ ONLY (extracted mp4s)")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frames", type=int, default=FROZEN_FRAME_COUNT,
                        help="how many of the frozen set to decode (cost measurement)")
    parser.add_argument("--probe-only", action="store_true",
                        help="ffprobe every camera; decode nothing")
    args = parser.parse_args(argv)

    source = Path(args.source_dir)
    root = Path(args.output_root)
    resolved = root.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ContractError(f"--output-root {root} is inside the repository")

    videos = sorted(source.glob("cam*.mp4"))
    if not videos:
        raise ContractError(f"no cam*.mp4 under {source}")

    report = {
        "schema": "imvid-frame-decode-v1",
        "source_dir": str(source),
        "output_root": str(root),
        "frozen_frame_rule": (
            "np.linspace(0, 299, 3) rounded -> 0, 150, 299; uniform by "
            "POSITION, never by content"
        ),
        "frozen_frame_indices": select_frame_indices(FROZEN_FRAME_COUNT),
        "decoded_frame_indices": select_frame_indices(int(args.frames)),
        "n_cameras": len(videos),
        "declared": {
            "frames": DECLARED_FRAMES,
            "width": DECLARED_WIDTH,
            "height": DECLARED_HEIGHT,
        },
        "probe": {},
        "images": [],
    }

    mismatches = []
    for video in videos:
        stream = _ffprobe_streams(video)
        nb = stream.get("nb_frames")
        entry = {
            "width": stream.get("width"),
            "height": stream.get("height"),
            "nb_frames": int(nb) if nb not in (None, "N/A") else None,
            "codec": stream.get("codec_name"),
            "pix_fmt": stream.get("pix_fmt"),
            "r_frame_rate": stream.get("r_frame_rate"),
        }
        report["probe"][video.stem] = entry
        if entry["width"] != DECLARED_WIDTH or entry["height"] != DECLARED_HEIGHT:
            mismatches.append((video.stem, "size", entry["width"], entry["height"]))
        if entry["nb_frames"] is not None and entry["nb_frames"] != DECLARED_FRAMES:
            mismatches.append((video.stem, "frames", entry["nb_frames"], DECLARED_FRAMES))
    if mismatches:
        raise ContractError(
            f"declared-vs-observed mismatch on {len(mismatches)} stream(s): "
            f"{mismatches[:5]}"
        )
    print(f"[imvid] {len(videos)} cameras probed; all {DECLARED_WIDTH}x{DECLARED_HEIGHT}, "
          f"{DECLARED_FRAMES} frames", flush=True)

    if args.probe_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if root.exists() and any(root.iterdir()):
        raise ContractError(f"output root is not empty: {root}")

    import time

    for frame_index in report["decoded_frame_indices"]:
        frame_dir = root / f"frame_{frame_index:06d}" / "images"
        started = time.perf_counter()
        for video in videos:
            target = frame_dir / f"{video.stem}.png"
            decode_frame(video, frame_index, target)
            width, height = _png_size(target)
            if (width, height) != (DECLARED_WIDTH, DECLARED_HEIGHT):
                raise ContractError(
                    f"{target} decoded at {width}x{height}, expected "
                    f"{DECLARED_WIDTH}x{DECLARED_HEIGHT}"
                )
            report["images"].append({
                "frame": int(frame_index),
                "camera": video.stem,
                "path": str(target.relative_to(root)),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            })
        elapsed = time.perf_counter() - started
        print(f"[imvid] frame {frame_index}: {len(videos)} images in {elapsed:.1f}s "
              f"-> {frame_dir}", flush=True)

    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "MANIFEST.imvid_frames.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    total = sum(entry["bytes"] for entry in report["images"])
    print(f"[imvid] {len(report['images'])} images, {total / 2**30:.3f} GiB, "
          f"manifest -> {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
