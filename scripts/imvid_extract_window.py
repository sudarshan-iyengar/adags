#!/usr/bin/env python3
r"""Decode a contiguous WINDOW of an ImViD full take, RENUMBERED FROM ZERO.

The ImViD sample lane decoded whole 300-frame takes
(`scripts/imvid_decode_frames.py`). The full takes are not 300 frames:
Opera is **15,215** and Puppy **5,936**, both 5312x2988 `yuv420p` H.264
at `r_frame_rate 60000/1001`, one MP4 per camera, `cam00.mp4` ..
`cam38.mp4`. The full Opera take is a **560x** exposure gap against
N3V-50f and cannot be trained at any authorized schedule
([[query_pack]] 2026-08-24 (4)), so a frozen event-selected TRANCHE is
mandatory. This script cuts that tranche and nothing else -- no
undistortion, no intrinsics, no conversion. `scripts/imvid_to_blender.py`
consumes its output unchanged.

--------------------------------------------------------------------
THE ONE DESIGN RULE: THE WINDOW IS RENUMBERED AT EXTRACTION
--------------------------------------------------------------------

Source frames `[start, start + count)` are written as output frame
indices `0 .. count-1`. The start offset lives in the manifest and
**never in a filename**.

The reason is that the offset is otherwise trainer-visible, silently.
`scripts/imvid_to_blender.py::frame_time` (:823-825) derives every
timestamp as `time(i) = i / fps` from the DIRECTORY INDEX, and the
trainer's `time_duration` is `[0, (count-1)/fps]`. Writing a window that
starts at source frame 9,000 under its source names would hand the
converter `frame_009000 .. frame_009299`, and every emitted timestamp
would be shifted by `start / fps` -- for Opera at 60000/1001 that is
**150.15 s** of phantom lead-in -- with no error raised anywhere,
because `:380` reads `time` verbatim and no consumer knows what the
first frame was supposed to be. Renumbering here makes the offset a
PREPROCESSING fact. Every downstream artifact is then byte-identical to
the 300-frame-sample case, and the only place the offset can be read is
the manifest, where it is recorded as data rather than as arithmetic.

--------------------------------------------------------------------
THE OUTPUT CONTRACT, READ OUT OF THE CONSUMERS
--------------------------------------------------------------------

* `scripts/imvid_to_blender.py::source_image_path` (:1173-1174):
  `<root>/frame_%06d/images/<camera>.png`. This script writes that
  path via ffmpeg's image2 `%06d` pattern, so the numbering is produced
  by the muxer and asserted against `source_image_path` in `--self-test`.
* `scripts/imvid_to_blender.py::discover_frames` (:1149-1170) enumerates
  `frame_<NNNNNN>/` directories off the FILESYSTEM, not the manifest.
  A stale directory from a previous, longer window would therefore be
  silently mixed into the scene, so this script refuses unless the final
  frame-directory set is EXACTLY `{0 .. count-1}`.
* `scripts/imvid_to_blender.py::verify_fps_against_decode_manifest`
  (:829-881) opens `<root>/MANIFEST.imvid_frames.json` -- that exact
  basename -- requires a non-empty `probe` block, and requires every
  `probe[<camera>]["r_frame_rate"]` to parse as `NUM/DEN` and to equal
  the declared rate. The manifest written here therefore carries the
  `probe` block keyed by camera stem with `r_frame_rate` verbatim from
  ffprobe. `--manifest` writes an ADDITIONAL copy for the run record;
  it does not and cannot replace the frames-root manifest, because the
  consumer looks only there.

`_ffprobe_streams`, `_png_size` and `_sha256` are IMPORTED from
`scripts/imvid_decode_frames.py` rather than reimplemented, so the
probe field list and the readback are the same code that produced the
sample manifests; agreement between the two manifests is then not a
coincidence of two independent reimplementations.

--------------------------------------------------------------------
WHY OUTPUT-SIDE SELECTION AND NOT AN INPUT-SIDE SEEK
--------------------------------------------------------------------

The argv is the existing decoder's argv (`imvid_decode_frames.py:110-115`)
with `select=eq(n,K)` generalized to `select=between(n,START,END)`:

    ffmpeg -v error -y -i <cam>.mp4
           -vf select=between(n\,START\,END) -vsync 0
           -frames:v COUNT
           -f image2 -start_number 0
           <root>/frame_%06d/images/<cam>.png

`n` is the **decode-order frame index**, so selection does not depend on
the container's timestamps being exact. An input-side `-ss` placed
before `-i` seeks by TIMESTAMP to a keyframe neighbourhood; on a
15,215-frame take with a `60000/1001` rate that is where an off-by-one
window comes from, and an off-by-one window is invisible downstream
because every frame still decodes and every timestamp still looks
plausible. Frame-exactness outranks speed here: the cost of decoding the
discarded prefix is CPU on a machine with 80 cores and 39 independent
files, and `-frames:v COUNT` stops the pipeline at the end of the
window so the tail is never decoded.

`-vsync 0` (passthrough) keeps the filter's dropped frames dropped
rather than resampled to a constant rate; without it the muxer would
duplicate frames to fill the gaps the `select` opened.

--------------------------------------------------------------------
FAIL-CLOSED
--------------------------------------------------------------------

Refusals, each guarding a failure that would otherwise be SILENT:

* Any disagreement between cameras on width, height, `r_frame_rate`,
  `pix_fmt` or `nb_frames` -- reported as the exact split, with the
  camera names on each side. A single odd camera means the take is not
  one rig recording and the window is not one window.
* `nb_frames` absent or `N/A` on any camera: the `start + count <=
  nb_frames` bound cannot be checked, so no bound is asserted.
* `start + count > nb_frames`: a short window would otherwise be a
  quietly truncated tranche.
* A `%` anywhere in the resolved output path other than the one `%06d`
  the muxer is meant to expand -- ffmpeg's image2 pattern would expand
  it too and scatter the frames.
* Fewer than `count` PNGs for any camera, or a total PNG count other
  than `count * n_cameras`, or any PNG whose IHDR raster is not the
  probed native raster. A short decode is an error, never a short
  window.
* A pre-existing frame directory outside `{0 .. count-1}`, before AND
  after decoding. This script NEVER deletes: it names the offending
  directories and asks for them to be removed.
* An output root that is inside the git repository, or inside the
  read-only source take.

Usage:
  python3 scripts/imvid_extract_window.py --self-test
  python3 scripts/imvid_extract_window.py \
      --source-dir  /apollo/users/sri/proj_adags/data/imvid/scene1_opera \
      --output-root /apollo/users/sri/proj_adags/data/imvid/window_opera_09000 \
      --start 9000 --count 300 \
      --manifest /apollo/users/sri/proj_adags/runs/elgs/<run>/MANIFEST.imvid_window.json
  python3 scripts/imvid_extract_window.py ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from depth_visibility.artifacts import atomic_write_bytes  # noqa: E402
from depth_visibility.errors import ContractError  # noqa: E402

# The probe field list, the PNG IHDR readback and the file hash are the
# SAME implementations that produced the 300-frame-sample manifests, so
# a field-by-field comparison between a sample manifest and a window
# manifest compares measurements, not two reimplementations.
from imvid_decode_frames import (  # noqa: E402
    _ffprobe_streams,
    _png_size,
    _sha256,
)

SCHEMA = "imvid-window-extract-v1"

#: `scripts/imvid_to_blender.py:834` opens EXACTLY this basename at the
#: frames root. Renaming it makes the frame rate unverifiable and the
#: converter refuses (or, with --allow-unverified-fps, proceeds on an
#: unchecked constant, which is worse).
DECODE_MANIFEST_NAME = "MANIFEST.imvid_frames.json"

#: `scripts/imvid_to_blender.py:1174`:
#: frames_root / f"frame_{i:06d}" / "images" / f"{camera}.png"
FRAME_DIR_PREFIX = "frame_"
FRAME_DIR_PATTERN = "frame_%06d"
IMAGES_SUBDIR = "images"
PNG_SUFFIX = ".png"

#: Every camera in a take must agree on all five. width/height because
#: the calibration describes one raster; r_frame_rate because every
#: downstream timestamp is derived from it; pix_fmt because a differing
#: one means a differing encode; nb_frames because the window bound is
#: asserted against it.
AGREEMENT_FIELDS = ("width", "height", "r_frame_rate", "pix_fmt", "nb_frames")

DEFAULT_COUNT = 300

#: Hashing `count * n_cameras` native PNGs (300 x 39 x ~15 MB ~ 175 GiB
#: for one Opera window) is not affordable at extraction time, so the
#: manifest hashes a DECLARED subset and says so in the manifest itself.
SHA256_SUBSET_RULE = "output frame 0 and output frame count-1 of every camera"


# ====================================================================
# Pure logic -- exercised by --self-test with no video and no ffmpeg
# ====================================================================


def parse_rate(text: str) -> Fraction:
    """`"60000/1001"` -> Fraction, NUM/DEN only.

    Deliberately as strict as the consumer:
    `scripts/imvid_to_blender.py::parse_rational` (:263-284) refuses a
    bare decimal, so a rate recorded here in any other form would make
    the converter refuse the manifest. `Fraction(text)` would happily
    accept `"59.94"`, which is a DIFFERENT number from `60000/1001`.
    """
    parts = str(text).strip().split("/")
    if len(parts) != 2:
        raise ContractError(
            f"r_frame_rate {text!r} is not NUM/DEN. The consumer "
            "(scripts/imvid_to_blender.py:271-277) refuses anything else, so "
            "a manifest carrying this value could not be verified."
        )
    try:
        num, den = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ContractError(f"r_frame_rate {text!r} is not NUM/DEN: {exc}") from exc
    if num <= 0 or den <= 0:
        raise ContractError(f"r_frame_rate {text!r} must be positive")
    return Fraction(num, den)


def output_index_for_source(source_index: int, start: int) -> int:
    """The renumbering, in one place so it can be asserted."""
    return int(source_index) - int(start)


def source_index_for_output(output_index: int, start: int) -> int:
    """The inverse. The manifest is the ONLY record of `start`."""
    return int(output_index) + int(start)


def check_window(start: int, count: int, nb_frames: int) -> dict:
    """Bound the window against the measured length. Fail closed."""
    start = int(start)
    count = int(count)
    if start < 0:
        raise ContractError(f"--start {start} is negative")
    if count < 1:
        raise ContractError(f"--count {count} must be at least 1")
    if nb_frames is None or int(nb_frames) < 1:
        raise ContractError(
            f"the take reports nb_frames={nb_frames!r}, so no window bound can "
            "be asserted"
        )
    nb_frames = int(nb_frames)
    end = start + count - 1
    if start + count > nb_frames:
        raise ContractError(
            f"window [{start}, {end}] needs {start + count} source frames but "
            f"the take holds {nb_frames}. A shorter window would be decoded "
            f"silently; reduce --count to {max(0, nb_frames - start)} or lower "
            f"--start to {max(0, nb_frames - count)}."
        )
    return {
        "start": start,
        "count": count,
        "end_inclusive": end,
        "source_frame_indices": [start, end],
        "output_frame_indices": [0, count - 1],
        "nb_frames": nb_frames,
    }


def check_stream_agreement(probe: dict) -> dict:
    """Every camera must agree on all of `AGREEMENT_FIELDS`.

    A rig where one camera differs is not one recording, and the window
    it produces would be one window only by name. The refusal names the
    exact split so the odd camera can be found without re-probing.
    """
    if not probe:
        raise ContractError(
            "no cameras were probed; refusing to extract a window of nothing"
        )
    cameras = sorted(probe)
    agreed: dict = {}
    disagreements: list[str] = []
    for field in AGREEMENT_FIELDS:
        by_value: dict = {}
        for camera in cameras:
            by_value.setdefault(repr(probe[camera].get(field)), []).append(camera)
        if len(by_value) != 1:
            split = "; ".join(
                f"{value} on {len(names)} camera(s) {names[:4]}"
                for value, names in sorted(by_value.items())
            )
            disagreements.append(f"{field}: {split}")
        else:
            agreed[field] = probe[cameras[0]].get(field)
    if disagreements:
        raise ContractError(
            f"the {len(cameras)} cameras of this take DISAGREE on "
            f"{len(disagreements)} stream field(s), so they are not one "
            f"recording: {disagreements}. Every field is load-bearing: raster "
            "for the calibration, r_frame_rate for every timestamp, pix_fmt "
            "for the encode, nb_frames for the window bound."
        )
    if agreed.get("nb_frames") is None:
        raise ContractError(
            "ffprobe reported no nb_frames for this take, so "
            "`start + count <= nb_frames` cannot be checked and a window "
            "running off the end would decode short and silently. Re-probe "
            "with `ffprobe -count_frames -show_entries stream=nb_read_frames` "
            "and record the counted value, or refuse the take."
        )
    for field in ("width", "height", "nb_frames"):
        if not isinstance(agreed[field], int) or agreed[field] < 1:
            raise ContractError(
                f"agreed {field}={agreed[field]!r} is not a positive integer"
            )
    agreed["fps"] = parse_rate(agreed["r_frame_rate"])
    return agreed


def frame_output_path(output_root: Path, output_index: int, camera: str) -> Path:
    """Must equal `imvid_to_blender.source_image_path` (:1173-1174)."""
    return (
        Path(output_root)
        / f"{FRAME_DIR_PREFIX}{int(output_index):06d}"
        / IMAGES_SUBDIR
        / f"{camera}{PNG_SUFFIX}"
    )


def output_pattern(output_root: Path, camera: str) -> str:
    """The image2 pattern whose `%06d` expansion IS `frame_output_path`.

    Guarded because ffmpeg expands EVERY `%` conversion in the pattern:
    an output root containing a stray `%` (a run directory named after a
    percentage, say) would scatter the window across invented
    directories instead of erroring.
    """
    pattern = os.fspath(
        Path(output_root) / FRAME_DIR_PATTERN / IMAGES_SUBDIR / f"{camera}{PNG_SUFFIX}"
    )
    if pattern.count("%") != 1 or FRAME_DIR_PATTERN not in pattern:
        raise ContractError(
            f"the image2 output pattern {pattern!r} does not hold exactly one "
            f"'%' conversion. ffmpeg expands every one of them, so the frames "
            "would be written to invented paths. Remove '%' from --output-root "
            "and from the camera names."
        )
    return pattern


def build_ffmpeg_argv(video: Path, camera: str, output_root: Path,
                      start: int, count: int) -> list[str]:
    """`imvid_decode_frames.py:110-115` with `eq(n,K)` -> `between(n,S,E)`.

    `select` runs POST-INPUT and matches on `n`, the decode-order index,
    so the window is frame-exact and independent of container
    timestamps. `-start_number 0` is what performs the RENUMBERING: the
    muxer numbers the frames it is HANDED, and `select` hands it the
    window, so the first surviving frame is written as `frame_000000`
    whatever its source index was.
    """
    end = int(start) + int(count) - 1
    return [
        "ffmpeg", "-v", "error", "-y",
        "-i", os.fspath(video),
        # The commas inside between() are escaped because the filtergraph
        # parser splits filters on unescaped commas.
        "-vf", f"select=between(n\\,{int(start)}\\,{end})",
        "-vsync", "0",
        "-frames:v", str(int(count)),
        "-f", "image2",
        "-start_number", "0",
        output_pattern(output_root, camera),
    ]


def window_time_record(fps: Fraction, count: int) -> dict:
    """The trainer-facing consequence of the renumbering, stated once."""
    period = Fraction(1) / fps
    span = Fraction(int(count) - 1) * period
    return {
        "fps_rational": f"{fps.numerator}/{fps.denominator}",
        "fps_float": float(fps),
        "frame_period_rational": f"{period.numerator}/{period.denominator}",
        "frame_period_float": float(period),
        "time_formula": "time(i) = output_index / fps  (imvid_to_blender.py:823-825)",
        "time_duration": [0.0, float(span)],
        "window_span_seconds": float(span),
        "suppressed_offset_seconds": None,
    }


def existing_frame_indices(root: Path) -> list[int]:
    """The same enumeration `discover_frames` performs (:1152-1159)."""
    if not Path(root).is_dir():
        return []
    found: list[int] = []
    for child in sorted(Path(root).iterdir()):
        if child.is_dir() and child.name.startswith(FRAME_DIR_PREFIX):
            try:
                found.append(int(child.name.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(found)


def check_frame_dir_set(root: Path, count: int, phase: str) -> list[int]:
    """No frame directory outside `{0 .. count-1}` may survive.

    `discover_frames` reads the FILESYSTEM, so a leftover
    `frame_000300` from a longer previous window would be pulled into
    the converted scene with no warning -- two windows spliced into one
    timeline. This script never deletes anything; it names them.
    """
    present = existing_frame_indices(root)
    stray = [i for i in present if i < 0 or i >= int(count)]
    if stray:
        raise ContractError(
            f"[{phase}] {len(stray)} frame directory/directories under {root} "
            f"lie outside this window's 0..{int(count) - 1}: {stray[:8]}. "
            "scripts/imvid_to_blender.py::discover_frames (:1149) enumerates "
            "the filesystem, so they would be spliced into the scene. Remove "
            "them (this script never deletes) or choose an empty "
            "--output-root."
        )
    return present


# ====================================================================
# Probe / plan
# ====================================================================


def discover_cameras(source: Path, wanted: list[str] | None) -> list[Path]:
    videos = sorted(Path(source).glob("cam*.mp4"))
    if not videos:
        raise ContractError(f"no cam*.mp4 under {source}")
    if wanted is None:
        return videos
    by_stem = {video.stem: video for video in videos}
    chosen: list[Path] = []
    missing: list[str] = []
    for name in wanted:
        stem = name[:-4] if name.endswith(".mp4") else name
        if stem in by_stem:
            chosen.append(by_stem[stem])
        else:
            missing.append(stem)
    if missing:
        raise ContractError(
            f"--cameras named {missing} which are absent from {source} "
            f"(present: {sorted(by_stem)[:8]}{'...' if len(by_stem) > 8 else ''})"
        )
    if not chosen:
        raise ContractError("--cameras selected nothing")
    return sorted(set(chosen))


def probe_cameras(videos: list[Path]) -> dict:
    """One ffprobe per camera, recording the fields the manifest carries.

    `probe` is keyed by camera STEM because that is the key
    `verify_fps_against_decode_manifest` iterates (:848).
    """
    probe: dict = {}
    for video in videos:
        stream = _ffprobe_streams(video)
        raw = stream.get("nb_frames")
        probe[video.stem] = {
            "width": stream.get("width"),
            "height": stream.get("height"),
            "nb_frames": int(raw) if raw not in (None, "N/A") else None,
            "codec": stream.get("codec_name"),
            "pix_fmt": stream.get("pix_fmt"),
            "r_frame_rate": stream.get("r_frame_rate"),
            "source_path": os.fspath(video),
            "source_bytes": video.stat().st_size,
        }
    return probe


def build_plan(videos: list[Path], probe: dict, agreed: dict, window: dict,
               source: Path, root: Path, workers: int) -> dict:
    fps: Fraction = agreed["fps"]
    times = window_time_record(fps, window["count"])
    times["suppressed_offset_seconds"] = float(
        Fraction(window["start"]) / fps
    )
    return {
        "schema": SCHEMA,
        "source_dir": os.fspath(source),
        "output_root": os.fspath(root),
        "n_cameras": len(videos),
        "cameras": [video.stem for video in videos],
        "workers": int(workers),
        "window": window,
        "renumbering": {
            "statement": (
                f"source frames {window['start']}..{window['end_inclusive']} "
                f"are written as output frame indices 0..{window['count'] - 1}; "
                "the start offset is recorded HERE and never in a filename"
            ),
            "output_index_for_source": "output = source - start",
            "source_index_for_output": "source = output + start",
            "why": (
                "imvid_to_blender.py:823-825 derives time(i) = i / fps from the "
                "DIRECTORY index and the trainer's time_duration is "
                f"[0, (count-1)/fps]. Under source-named directories every "
                f"timestamp would be shifted by start/fps = "
                f"{times['suppressed_offset_seconds']:.6f} s with no error "
                "raised. Renumbering makes the window offset a preprocessing "
                "fact and keeps every downstream artifact byte-identical to "
                "the 300-frame-sample case."
            ),
        },
        "agreed_stream": {
            "width": agreed["width"],
            "height": agreed["height"],
            "pix_fmt": agreed["pix_fmt"],
            "r_frame_rate": agreed["r_frame_rate"],
            "nb_frames": agreed["nb_frames"],
        },
        "fps": times,
        "layout": {
            "image_path": "<output_root>/frame_%06d/images/<camera>.png",
            "authority": "scripts/imvid_to_blender.py::source_image_path:1173-1174",
            "decode_manifest": DECODE_MANIFEST_NAME,
            "decode_manifest_authority": (
                "scripts/imvid_to_blender.py::verify_fps_against_decode_manifest"
                ":829-881 reads probe[<camera>]['r_frame_rate'] from exactly "
                "this basename at the frames root"
            ),
        },
        "ffmpeg_argv": {
            video.stem: build_ffmpeg_argv(
                video, video.stem, root, window["start"], window["count"]
            )
            for video in videos
        },
        "ffmpeg_argv_rationale": (
            "post-input `-vf select=between(n,S,E)` matches the DECODE-ORDER "
            "index, so the window is frame-exact and independent of container "
            "timestamps; an input-side `-ss` seeks by timestamp to a keyframe "
            "neighbourhood and is where an invisible off-by-one window comes "
            "from. `-frames:v COUNT` stops the pipeline at the end of the "
            "window so the tail is never decoded. `-start_number 0` performs "
            "the renumbering."
        ),
        "probe": probe,
    }


# ====================================================================
# Decode
# ====================================================================


def decode_camera(video: Path, camera: str, root: Path, start: int,
                  count: int) -> dict:
    argv = build_ffmpeg_argv(video, camera, root, start, count)
    began = time.perf_counter()
    out = subprocess.run(
        argv, capture_output=True, text=True, timeout=86400,
        stdin=subprocess.DEVNULL,
    )
    elapsed = time.perf_counter() - began
    if out.returncode != 0:
        raise ContractError(
            f"ffmpeg failed for {camera} on window [{start}, "
            f"{start + count - 1}] (rc={out.returncode}): "
            f"{out.stderr[:600]}"
        )
    return {"camera": camera, "seconds": elapsed, "argv": argv}


def verify_camera_outputs(root: Path, camera: str, count: int,
                          width: int, height: int) -> dict:
    """Readback, fail closed. A short decode is an error, not a short window."""
    missing: list[int] = []
    wrong: list[str] = []
    total_bytes = 0
    for index in range(int(count)):
        path = frame_output_path(root, index, camera)
        if not path.is_file():
            missing.append(index)
            continue
        total_bytes += path.stat().st_size
        got = _png_size(path)
        if got != (int(width), int(height)):
            wrong.append(f"frame_{index:06d}: {got[0]}x{got[1]}")
    if missing:
        raise ContractError(
            f"{camera} is SHORT: {len(missing)} of {count} window frames were "
            f"not written (first missing output indices {missing[:8]}). ffmpeg "
            "returned success, so this is a truncated decode, not a failed "
            "one; the window is unusable."
        )
    if wrong:
        raise ContractError(
            f"{camera} wrote {len(wrong)} frame(s) at a raster other than the "
            f"probed native {width}x{height}: {wrong[:5]}. The supplied "
            "intrinsics describe the native raster; a rescaled frame puts "
            "every correspondence in a different frame from the calibration."
        )
    return {"camera": camera, "files": int(count), "bytes": total_bytes}


def hash_subset(root: Path, cameras: list[str], count: int) -> dict:
    indices = sorted({0, int(count) - 1})
    entries = []
    for camera in cameras:
        for index in indices:
            path = frame_output_path(root, index, camera)
            entries.append({
                "camera": camera,
                "output_frame": index,
                "path": os.fspath(path.relative_to(Path(root))),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    return {
        "is_subset": True,
        "rule": SHA256_SUBSET_RULE,
        "output_frames_hashed": indices,
        "n_hashed": len(entries),
        "n_total_images": int(count) * len(cameras),
        "why_a_subset": (
            "hashing every native PNG in a 300x39 window is ~175 GiB of reads "
            "at extraction time. The subset is DECLARED here so no consumer "
            "can read this block as a whole-window digest."
        ),
        "entries": entries,
    }


# ====================================================================
# Self-test -- no video files, no ffmpeg
# ====================================================================


def _check(name: str, ok: bool, detail: dict) -> dict:
    if not ok:
        raise ContractError(f"SELF-TEST FAILED: {name} -- {detail}")
    return {"name": name, "status": "PASS", **detail}


def _refuses(fn) -> str | None:
    """Return the refusal message, or None if the call did NOT refuse."""
    try:
        fn()
    except ContractError as exc:
        return str(exc)
    return None


def _synthetic_probe(n_cameras: int = 4, nb_frames: int = 15215) -> dict:
    return {
        f"cam{i:02d}": {
            "width": 5312, "height": 2988, "nb_frames": nb_frames,
            "codec": "h264", "pix_fmt": "yuv420p",
            "r_frame_rate": "60000/1001",
            "source_path": f"/take/cam{i:02d}.mp4", "source_bytes": 1 << 30,
        }
        for i in range(n_cameras)
    }


def run_self_test() -> list[dict]:
    checks: list[dict] = []
    root = Path("/apollo/users/sri/proj_adags/data/imvid/window_opera_09000")
    start, count = 9000, 300

    # 1 -- the renumbering is a bijection onto 0..count-1
    outputs = [output_index_for_source(start + j, start) for j in range(count)]
    back = [source_index_for_output(o, start) for o in outputs]
    checks.append(_check(
        "renumbering_is_a_bijection_onto_zero_based_indices",
        outputs == list(range(count))
        and back == list(range(start, start + count))
        and source_index_for_output(0, start) == start
        and output_index_for_source(start + count - 1, start) == count - 1,
        {"start": start, "count": count,
         "source_span": [start, start + count - 1],
         "output_span": [outputs[0], outputs[-1]]}))

    # 2 -- the OUTPUT index set does not depend on where the window starts
    other = [output_index_for_source(14915 + j, 14915) for j in range(count)]
    checks.append(_check(
        "output_indices_are_independent_of_the_source_offset",
        other == outputs,
        {"start_a": start, "start_b": 14915,
         "note": "the offset survives only in the manifest"}))

    # 3 -- the hazard is real and is exactly start/fps, and renumbering kills it
    fps = Fraction(60000, 1001)
    shifted = Fraction(start) / fps
    renumbered = Fraction(outputs[0]) / fps
    checks.append(_check(
        "renumbering_removes_a_start_over_fps_timestamp_shift",
        shifted == Fraction(9000 * 1001, 60000) and renumbered == 0
        and abs(float(shifted) - 150.15) < 1e-9,
        {"suppressed_offset_seconds": float(shifted),
         "first_output_timestamp": float(renumbered),
         "consequence_if_unrenumbered": (
             "every frame's time would carry 150.15 s of phantom lead-in and "
             "no consumer could detect it (imvid_to_blender.py:380 reads "
             "`time` verbatim)")}))

    # 4 -- time_duration is [0, (count-1)/fps] exactly
    times = window_time_record(fps, count)
    expect_span = float(Fraction(count - 1) * (Fraction(1) / fps))
    checks.append(_check(
        "time_duration_is_zero_to_count_minus_one_over_fps",
        times["time_duration"] == [0.0, expect_span]
        and times["fps_rational"] == "60000/1001"
        and abs(times["fps_float"] - 60000.0 / 1001.0) < 1e-12,
        {"time_duration": times["time_duration"],
         "frame_period_rational": times["frame_period_rational"]}))

    # 5 -- the emitted path IS the consumer's source_image_path
    pattern = output_pattern(root, "cam07")
    expanded = [pattern % j for j in (0, 1, count - 1)]
    ours = [os.fspath(frame_output_path(root, j, "cam07")) for j in (0, 1, count - 1)]
    checks.append(_check(
        "image2_pattern_expands_to_the_contract_layout",
        expanded == ours
        and expanded[0].endswith(os.path.join("frame_000000", "images", "cam07.png"))
        and expanded[2].endswith(os.path.join("frame_000299", "images", "cam07.png")),
        {"pattern": pattern, "expanded": expanded[:2]}))

    # 6 -- a '%' in the output root is refused, not scattered
    refusal = _refuses(lambda: output_pattern(Path("/runs/50%_take"), "cam00"))
    checks.append(_check(
        "percent_in_the_output_root_is_refused",
        refusal is not None and "conversion" in refusal,
        {"refusal": (refusal or "")[:120]}))

    # 7 -- the argv is frame-exact and post-input, with a negative control
    argv = build_ffmpeg_argv(Path("/take/cam07.mp4"), "cam07", root, start, count)
    i_at = argv.index("-i")
    checks.append(_check(
        "ffmpeg_argv_selects_by_decode_order_after_the_input",
        argv[argv.index("-vf") + 1] == "select=between(n\\,9000\\,9299)"
        and argv[argv.index("-vsync") + 1] == "0"
        and argv[argv.index("-frames:v") + 1] == "300"
        and argv[argv.index("-start_number") + 1] == "0"
        and argv[argv.index("-f") + 1] == "image2"
        and argv.index("-vf") > i_at
        and "-ss" not in argv
        and argv[-1] == pattern,
        {"argv": argv,
         "negative_control": "'-ss' absent: no timestamp seek anywhere",
         "vf_is_post_input": True}))

    # 8 -- count=1 reduces to the existing decoder's single-frame selection
    one = build_ffmpeg_argv(Path("/take/cam00.mp4"), "cam00", root, 42, 1)
    checks.append(_check(
        "count_one_reduces_to_the_sample_decoders_selection",
        one[one.index("-vf") + 1] == "select=between(n\\,42\\,42)"
        and one[one.index("-frames:v") + 1] == "1",
        {"sample_decoder": "select=eq(n\\,42)  (imvid_decode_frames.py:112)",
         "this_script": one[one.index("-vf") + 1],
         "note": "between(n,K,K) and eq(n,K) select the same single frame"}))

    # 9 -- the agreement checker accepts a homogeneous take
    agreed = check_stream_agreement(_synthetic_probe())
    checks.append(_check(
        "agreement_accepts_a_homogeneous_take",
        agreed["width"] == 5312 and agreed["height"] == 2988
        and agreed["pix_fmt"] == "yuv420p"
        and agreed["nb_frames"] == 15215 and agreed["fps"] == fps,
        {"agreed": {k: str(v) for k, v in agreed.items()}}))

    # 10 -- every agreement field actually refuses when it disagrees
    refusals: dict = {}
    for field, bad in (("width", 4096), ("height", 2160),
                       ("r_frame_rate", "30000/1001"), ("pix_fmt", "yuv422p"),
                       ("nb_frames", 5936)):
        probe = _synthetic_probe()
        probe["cam02"][field] = bad
        refusals[field] = _refuses(lambda p=probe: check_stream_agreement(p))
    checks.append(_check(
        "every_agreement_field_refuses_and_names_the_split",
        all(msg is not None and field in msg and "cam02" in msg
            for field, msg in refusals.items()),
        {"fields_checked": sorted(refusals),
         "example": (refusals["pix_fmt"] or "")[:160]}))

    # 11 -- an absent nb_frames is refused (the window bound would be unassertable)
    no_count = _synthetic_probe()
    for entry in no_count.values():
        entry["nb_frames"] = None
    msg_none = _refuses(lambda: check_stream_agreement(no_count))
    msg_empty = _refuses(lambda: check_stream_agreement({}))
    checks.append(_check(
        "absent_nb_frames_and_an_empty_probe_are_both_refused",
        msg_none is not None and "nb_frames" in msg_none
        and msg_empty is not None,
        {"nb_frames_refusal": (msg_none or "")[:140],
         "empty_probe_refusal": (msg_empty or "")[:80]}))

    # 12 -- window bounds, in range and out of range
    ok_window = check_window(14915, 300, 15215)
    over = _refuses(lambda: check_window(15000, 300, 15215))
    neg = _refuses(lambda: check_window(-1, 300, 15215))
    zero = _refuses(lambda: check_window(0, 0, 15215))
    checks.append(_check(
        "window_bounds_accept_the_last_full_window_and_refuse_overruns",
        ok_window["end_inclusive"] == 15214
        and ok_window["output_frame_indices"] == [0, 299]
        and over is not None and "15215" in over
        and neg is not None and zero is not None,
        {"last_full_window": [ok_window["start"], ok_window["end_inclusive"]],
         "overrun_refusal": (over or "")[:160]}))

    # 13 -- a stray frame directory outside the window is refused
    with tempfile.TemporaryDirectory() as tmp:
        stray_root = Path(tmp) / "root"
        for index in (0, 1):
            (stray_root / f"frame_{index:06d}" / IMAGES_SUBDIR).mkdir(parents=True)
        (stray_root / "frame_000300").mkdir()
        stray = _refuses(lambda: check_frame_dir_set(stray_root, 300, "pre"))
        (stray_root / "frame_000300").rmdir()
        clean = check_frame_dir_set(stray_root, 300, "pre")
    checks.append(_check(
        "a_stray_frame_directory_outside_the_window_is_refused",
        stray is not None and "300" in stray and clean == [0, 1],
        {"refusal": (stray or "")[:160],
         "why": "discover_frames enumerates the filesystem (:1149-1159)"}))

    # 14 -- THE CONSUMERS THEMSELVES accept the layout and the manifest
    checks.append(_manifest_accepted_by_the_converter(fps))
    return checks


def _manifest_accepted_by_the_converter(fps: Fraction) -> dict:
    """Run the real consumer against a real manifest on a real temp tree.

    This is the only check that is not a restatement of this file's own
    beliefs: `discover_frames`, `source_image_path` and
    `verify_fps_against_decode_manifest` are imported from the converter
    and executed. If it cannot be imported the check is SKIPPED loudly
    rather than silently passing.
    """
    try:
        import imvid_to_blender as consumer
    except Exception as exc:  # noqa: BLE001 -- numpy/cv2 absence is the case
        return {
            "name": "converter_accepts_the_layout_and_the_manifest",
            "status": "SKIPPED",
            "reason": f"scripts/imvid_to_blender.py could not be imported: {exc}",
            "consequence": (
                "the output layout and the decode manifest were checked ONLY "
                "against this file's own restatement of the contract, never "
                "against the code that consumes them"),
        }
    count, start, cameras = 4, 9000, ["cam00", "cam07"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "window"
        for index in range(count):
            (root / f"frame_{index:06d}" / IMAGES_SUBDIR).mkdir(parents=True)
        probe = {c: dict(_synthetic_probe(1)["cam00"], source_path=f"/t/{c}.mp4")
                 for c in cameras}
        agreed = check_stream_agreement(probe)
        window = check_window(start, count, agreed["nb_frames"])
        plan = build_plan([Path(f"/t/{c}.mp4") for c in cameras], probe, agreed,
                          window, Path("/t"), root, 2)
        (root / DECODE_MANIFEST_NAME).write_text(
            json.dumps(plan, indent=1, sort_keys=True, default=str), encoding="utf-8")

        found = consumer.discover_frames(root, None)
        paths_agree = all(
            consumer.source_image_path(root, index, camera)
            == frame_output_path(root, index, camera)
            for index in range(count) for camera in cameras)
        verified = consumer.verify_fps_against_decode_manifest(root, fps, False)
        wrong_rate = _refuses(
            lambda: consumer.verify_fps_against_decode_manifest(
                root, Fraction(30, 1), False))
    return _check(
        "converter_accepts_the_layout_and_the_manifest",
        found == list(range(count)) and paths_agree
        and verified["verified"] is True
        and verified["cameras_checked"] == len(cameras)
        and wrong_rate is not None and "60000/1001" in wrong_rate,
        {"discover_frames": found,
         "source_image_path_agrees": paths_agree,
         "fps_verified": verified["measured_distinct"],
         "negative_control_declaring_30_1": (wrong_rate or "")[:120],
         "consumer_lines": "imvid_to_blender.py:829-881, :1149-1174"})


def cmd_self_test() -> dict:
    checks = run_self_test()
    for check in checks:
        print(f"[imvid] self-test {check['status']:7s} {check['name']}", flush=True)
        if check["status"] == "SKIPPED":
            print(f"[imvid]   REASON: {check['reason']}", flush=True)
            print(f"[imvid]   {check['consequence']}", flush=True)
    skipped = [c["name"] for c in checks if c["status"] == "SKIPPED"]
    print(f"[imvid] SELF-TEST PASSED: {len(checks) - len(skipped)} check(s) ran"
          + (f", {len(skipped)} SKIPPED {skipped}" if skipped else ""), flush=True)
    return {"mode": "self-test", "checks": checks,
            "checks_run": len(checks) - len(skipped), "checks_skipped": skipped}


# ====================================================================
# CLI
# ====================================================================


def _refuse_inside(label: str, path: Path, container: Path, why: str) -> None:
    resolved = Path(path).resolve()
    container = Path(container).resolve()
    if resolved == container or container in resolved.parents:
        raise ContractError(f"{label} {resolved} is inside {container}: {why}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true",
                        help="exercise the pure logic; no video, no ffmpeg")
    parser.add_argument("--source-dir", default=None,
                        help="READ ONLY: the take's cam*.mp4 directory")
    parser.add_argument("--output-root", default=None,
                        help="frames root; OUTSIDE the repository and the take")
    parser.add_argument("--start", type=int, default=None,
                        help="first SOURCE frame of the window (written as 0)")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help=f"window length in frames (default {DEFAULT_COUNT})")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="explicit camera list; default every cam*.mp4 present")
    parser.add_argument("--workers", type=int, default=None,
                        help="concurrent ffmpeg processes; default "
                             "min(cpu_count, n_cameras). ffmpeg is itself "
                             "multithreaded, so a value near the core count "
                             "oversubscribes rather than speeding up")
    parser.add_argument("--manifest", default=None,
                        help="ADDITIONAL copy of the manifest for the run "
                             f"record; the consumer reads only "
                             f"<output-root>/{DECODE_MANIFEST_NAME}")
    parser.add_argument("--overwrite", action="store_true",
                        help="permit a non-empty --output-root. Frame "
                             "directories outside 0..count-1 are still "
                             "refused; nothing is ever deleted")
    parser.add_argument("--dry-run", action="store_true",
                        help="probe, plan and print the exact ffmpeg argv; "
                             "write nothing")
    args = parser.parse_args(argv)

    if args.self_test:
        print(json.dumps(cmd_self_test(), indent=1, sort_keys=True, default=str))
        return 0

    for name in ("source_dir", "output_root", "start"):
        if getattr(args, name) is None:
            raise ContractError(f"--{name.replace('_', '-')} is required")

    source = Path(args.source_dir)
    root = Path(args.output_root)
    _refuse_inside("--output-root", root, REPO_ROOT,
                   "decoded frames are never committed (AGENTS.md)")
    _refuse_inside("--output-root", root, source,
                   "the take is READ ONLY and this script would write into it")
    if args.manifest is not None:
        _refuse_inside("--manifest", Path(args.manifest), REPO_ROOT,
                       "run records live outside the repository (AGENTS.md)")

    videos = discover_cameras(source, args.cameras)
    workers = args.workers if args.workers else min(
        os.cpu_count() or 1, len(videos))
    if workers < 1:
        raise ContractError(f"--workers {args.workers} must be at least 1")

    probe = probe_cameras(videos)
    agreed = check_stream_agreement(probe)
    window = check_window(args.start, args.count, agreed["nb_frames"])
    # The pattern is validated before anything is written, so a '%' in the
    # output root refuses at plan time rather than after 39 decodes.
    for video in videos:
        output_pattern(root, video.stem)
    plan = build_plan(videos, probe, agreed, window, source, root, workers)

    print(f"[imvid] {len(videos)} camera(s), {agreed['width']}x{agreed['height']} "
          f"{agreed['pix_fmt']} @ {agreed['r_frame_rate']}, take {agreed['nb_frames']} "
          f"frames", flush=True)
    print(f"[imvid] window source [{window['start']}, {window['end_inclusive']}] "
          f"-> output [0, {window['count'] - 1}]; suppressed offset "
          f"{plan['fps']['suppressed_offset_seconds']:.6f} s", flush=True)

    if args.dry_run:
        print(json.dumps(plan, indent=1, sort_keys=True, default=str))
        print("[imvid] DRY RUN: nothing was written", flush=True)
        return 0

    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise ContractError(
            f"output root is not empty: {root}. Pass --overwrite to write into "
            "it anyway (frame directories outside this window are still "
            "refused, and nothing is ever deleted)."
        )
    check_frame_dir_set(root, window["count"], "pre-decode")
    # Pre-created because the image2 muxer expands the %06d pattern but does
    # NOT create the directories it names; every worker writes its own
    # <camera>.png into the shared per-frame directories, so they are made
    # once, here, before any process is spawned.
    for index in range(window["count"]):
        (root / f"{FRAME_DIR_PREFIX}{index:06d}" / IMAGES_SUBDIR).mkdir(
            parents=True, exist_ok=True)

    began = time.perf_counter()
    timings: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # One ffmpeg PROCESS per camera; the Python worker only waits on it,
        # so the GIL is never contended and --workers bounds the processes.
        futures = [pool.submit(decode_camera, video, video.stem, root,
                               window["start"], window["count"])
                   for video in videos]
        for future in futures:
            record = future.result()
            timings.append({"camera": record["camera"],
                            "seconds": record["seconds"]})
            print(f"[imvid] {record['camera']}: {window['count']} frames in "
                  f"{record['seconds']:.1f}s", flush=True)
    decode_seconds = time.perf_counter() - began

    per_camera = [
        verify_camera_outputs(root, video.stem, window["count"],
                              agreed["width"], agreed["height"])
        for video in videos
    ]
    written = sum(1 for _ in root.rglob(f"*{PNG_SUFFIX}"))
    expected = window["count"] * len(videos)
    if written != expected:
        raise ContractError(
            f"{written} PNG(s) under {root} but this window is {expected} "
            f"({window['count']} frames x {len(videos)} cameras). A surplus "
            "means images from a previous run or a different --cameras list "
            "survive and would be converted as part of this window."
        )
    check_frame_dir_set(root, window["count"], "post-decode")

    plan["decode"] = {
        "seconds": decode_seconds,
        "per_camera_seconds": timings,
        "workers": workers,
    }
    plan["verification"] = {
        "per_camera": per_camera,
        "images_written": written,
        "images_expected": expected,
        "raster_checked_from": "PNG IHDR of every written file",
        "total_bytes": sum(entry["bytes"] for entry in per_camera),
    }
    plan["sha256_subset"] = hash_subset(
        root, [video.stem for video in videos], window["count"])
    plan["argv"] = list(sys.argv[1:])

    body = json.dumps(plan, indent=1, sort_keys=True, default=str).encode("utf-8")
    atomic_write_bytes(root / DECODE_MANIFEST_NAME, body + b"\n")
    print(f"[imvid] manifest -> {root / DECODE_MANIFEST_NAME}", flush=True)
    if args.manifest is not None:
        atomic_write_bytes(Path(args.manifest), body + b"\n")
        print(f"[imvid] manifest copy -> {args.manifest}", flush=True)
    print(f"[imvid] {written} images, "
          f"{plan['verification']['total_bytes'] / 2**30:.3f} GiB, "
          f"{decode_seconds:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
