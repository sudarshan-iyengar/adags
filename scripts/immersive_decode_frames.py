#!/usr/bin/env python3
"""Extract and decode Google Immersive (DeepView Video) frames, fail-closed.

Input is the publisher's zip, which is a container rather than a compressor
(ratio 1.0001): 46 `camera_00NN.mp4` plus one `models.json`. Output is
`<out>/<camera>/%06d.png` at NATIVE resolution, plus a manifest.

Three things this refuses to do, each because the alternative fails silently:

1. **It keys on `models.json`, never on the mp4 list.** The archive ships 46
   videos and 45 calibrations, and *which* camera lacks calibration varies by
   scene (`01_Welder` -> `camera_0036`, `04_Truck` -> `camera_0003`,
   `12_Cave` -> 45/45). Zipping the two lists positionally is the off-by-one
   that produces a silently mis-calibrated scene, so an uncalibrated video is
   dropped by name and the drop is recorded.

2. **It validates decoded frame size against `models.json`'s own `width` and
   `height`.** STG's shipped script never does this. A resolution disagreement
   means the calibration does not describe these pixels, which is
   unrecoverable downstream and invisible in the images.

3. **It decodes exactly the requested frame range.** The official
   `extractframes()` calls ffmpeg with no range and decodes all 300 frames of
   all 46 videos regardless of what was asked for -- roughly a 6x storage cost
   at a 50-frame protocol.

ffmpeg comes from `imageio_ffmpeg`'s bundled static binary because Leonardo has
no ffmpeg module.

Usage:
  python3 scripts/immersive_decode_frames.py --archive <scene>.zip \\
      --out <dir> --frames 0 50
  python3 scripts/immersive_decode_frames.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402

CAMERA_RE = re.compile(r"^camera_\d{4}$")
MODELS_MEMBER = "models.json"


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ContractError(
            "imageio_ffmpeg is required (Leonardo has no ffmpeg module). "
            "pip install imageio-ffmpeg"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def read_models(archive: Path) -> tuple[list[dict], str]:
    """Return (views, member_name) from the archive's models.json."""
    with zipfile.ZipFile(archive) as zf:
        member = None
        for name in zf.namelist():
            if Path(name).name == MODELS_MEMBER:
                member = name
                break
        if member is None:
            raise ContractError(f"{archive} contains no {MODELS_MEMBER}")
        views = json.loads(zf.read(member).decode("utf-8"))
    if not isinstance(views, list) or not views:
        raise ContractError(f"{member} is not a non-empty JSON list")
    for v in views:
        for key in ("name", "position", "orientation", "focal_length",
                    "principal_point", "width", "height", "radial_distortion",
                    "projection_type"):
            if key not in v:
                raise ContractError(f"view {v.get('name')!r} lacks {key!r}")
        if v["projection_type"] != "fisheye":
            raise ContractError(
                f"view {v['name']!r} has projection_type "
                f"{v['projection_type']!r}; this loader only handles fisheye"
            )
    return views, member


def plan(archive: Path) -> dict:
    """Reconcile the calibrated views against the videos actually shipped."""
    views, member = read_models(archive)
    calibrated = [v["name"] for v in views]
    if len(set(calibrated)) != len(calibrated):
        raise ContractError("models.json contains duplicate camera names")
    for name in calibrated:
        if not CAMERA_RE.match(name):
            raise ContractError(f"unexpected camera name {name!r}")

    with zipfile.ZipFile(archive) as zf:
        videos = {Path(n).stem: n for n in zf.namelist() if n.endswith(".mp4")}

    missing_video = sorted(set(calibrated) - set(videos))
    if missing_video:
        raise ContractError(
            f"calibrated cameras with no video: {missing_video}. The video set "
            "must be a SUPERSET of the calibrated set."
        )
    uncalibrated = sorted(set(videos) - set(calibrated))
    return {
        "models_member": member,
        "calibrated": calibrated,
        "videos": videos,
        "uncalibrated_dropped": uncalibrated,
        "width": int(views[0]["width"]),
        "height": int(views[0]["height"]),
    }


def extract_videos(archive: Path, work: Path, names: list[str],
                   videos: dict[str, str]) -> dict[str, Path]:
    work.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as zf:
        for cam in names:
            member = videos[cam]
            target = work / f"{cam}.mp4"
            declared = zf.getinfo(member).file_size
            if target.exists() and target.stat().st_size == declared:
                out[cam] = target
                continue
            tmp = target.with_suffix(".mp4.part")
            with zf.open(member) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
            got = tmp.stat().st_size
            if got != declared:
                tmp.unlink(missing_ok=True)
                raise ContractError(
                    f"{member}: extracted {got} bytes, archive declares {declared}"
                )
            tmp.rename(target)
            out[cam] = target
    return out


def decode(video: Path, dest: Path, start: int, end: int,
           width: int, height: int, ffmpeg: str) -> int:
    """Decode frames [start, end) to dest/%06d.png. Returns the count."""
    dest.mkdir(parents=True, exist_ok=True)
    n = end - start
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(video),
        # select by frame INDEX, not by timestamp: no rounding, and it does not
        # depend on the container's frame rate being what we think it is.
        "-vf", f"select='between(n\\,{start}\\,{end - 1})'",
        "-vsync", "0", "-start_number", str(start),
        "-frames:v", str(n),
        str(dest / "%06d.png"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ContractError(
            f"ffmpeg failed on {video.name} (rc={proc.returncode}): "
            f"{proc.stderr[-500:]}"
        )
    produced = sorted(dest.glob("*.png"))
    if len(produced) != n:
        raise ContractError(
            f"{video.name}: expected {n} frames in [{start},{end}), got "
            f"{len(produced)}. The video is shorter than the requested range."
        )
    # Fail closed on raster: models.json's calibration must describe THESE
    # pixels. STG's own script never checks this.
    from PIL import Image
    with Image.open(produced[0]) as im:
        got_w, got_h = im.size
    if (got_w, got_h) != (width, height):
        raise ContractError(
            f"{video.name}: decoded {got_w}x{got_h} but models.json declares "
            f"{width}x{height}; the calibration does not describe these pixels"
        )
    return len(produced)


def self_test() -> int:
    """Runs with no data and no ffmpeg: checks the reconciliation logic."""
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        arch = Path(td) / "fake.zip"
        views = [{"name": f"camera_{i:04d}", "position": [0, 0, 0],
                  "orientation": [0, 0, 0], "focal_length": 1000.0,
                  "principal_point": [1280.0, 960.0], "width": 2560.0,
                  "height": 1920.0, "radial_distortion": [0.1, -0.01, 0.0],
                  "projection_type": "fisheye"} for i in range(1, 46)]
        with zipfile.ZipFile(arch, "w") as zf:
            zf.writestr("S/models.json", json.dumps(views))
            for i in range(1, 47):          # 46 videos, 45 calibrations
                zf.writestr(f"S/camera_{i:04d}.mp4", b"x")
        p = plan(arch)
        if p["uncalibrated_dropped"] != ["camera_0046"]:
            print("FAIL: uncalibrated camera not identified"); ok = False
        if len(p["calibrated"]) != 45:
            print("FAIL: wrong calibrated count"); ok = False

        # Anti-vacuity: a calibrated camera with NO video must be refused, not
        # silently skipped -- that is the direction that mis-calibrates.
        arch2 = Path(td) / "fake2.zip"
        with zipfile.ZipFile(arch2, "w") as zf:
            zf.writestr("S/models.json", json.dumps(views))
            for i in range(1, 45):          # camera_0045 calibrated, no video
                zf.writestr(f"S/camera_{i:04d}.mp4", b"x")
        try:
            plan(arch2)
            print("FAIL: missing video was not refused"); ok = False
        except ContractError:
            pass

        # A non-fisheye view must be refused.
        arch3 = Path(td) / "fake3.zip"
        bad = [dict(views[0], projection_type="perspective")]
        with zipfile.ZipFile(arch3, "w") as zf:
            zf.writestr("S/models.json", json.dumps(bad))
            zf.writestr("S/camera_0001.mp4", b"x")
        try:
            plan(arch3)
            print("FAIL: non-fisheye view was not refused"); ok = False
        except ContractError:
            pass
    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--work", type=Path, default=None,
                    help="where extracted mp4s live (default <out>/_videos)")
    ap.add_argument("--frames", nargs=2, type=int, default=[0, 50],
                    metavar=("START", "END"), help="half-open [START, END)")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.archive or not args.out:
        ap.error("--archive and --out are required unless --self-test")

    start, end = args.frames
    if not (0 <= start < end):
        raise ContractError(f"bad frame range [{start}, {end})")

    p = plan(args.archive)
    print(json.dumps({"calibrated": len(p["calibrated"]),
                      "videos": len(p["videos"]),
                      "uncalibrated_dropped": p["uncalibrated_dropped"],
                      "native": [p["width"], p["height"]]}, sort_keys=True))

    work = args.work or (args.out / "_videos")
    paths = extract_videos(args.archive, work, p["calibrated"], p["videos"])
    ffmpeg = _ffmpeg_exe()

    counts = {}
    for i, cam in enumerate(p["calibrated"], 1):
        counts[cam] = decode(paths[cam], args.out / cam, start, end,
                             p["width"], p["height"], ffmpeg)
        print(f"  [{i:2d}/{len(p['calibrated'])}] {cam}: {counts[cam]} frames",
              flush=True)

    manifest = {
        "archive": str(args.archive),
        "out": str(args.out),
        "frames": [start, end],
        "native": [p["width"], p["height"]],
        "cameras": p["calibrated"],
        "uncalibrated_dropped": p["uncalibrated_dropped"],
        "frames_per_camera": counts,
        "models_member": p["models_member"],
        "ffmpeg": ffmpeg,
    }
    out = args.manifest or (args.out / "decode_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"manifest: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
