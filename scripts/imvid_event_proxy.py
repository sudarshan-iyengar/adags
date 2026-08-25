#!/usr/bin/env python3
"""ImViD temporal event SCOUTING proxies: probe, build, and census.

Full PNG expansion of an ImViD scene is prohibitively large -- the measured
figure on the record is 300 frames x 35 cameras = 76.9 GiB at native
([[operations/imvid-baseline-freeze]] A5), and a FULL take is minutes rather
than the sample's 5 seconds (Opera full = 125.6 GB across 39 mp4s). So event
SCOUTING has to happen on low-resolution, low-frame-rate temporal proxies
BEFORE any training-resolution extraction is authorized. This script builds
those proxies and measures candidate temporal structure on them.

Three independent jobs, selected by ``--mode`` (plus ``self-test``):

``probe``     ffprobe every ``cam*.mp4`` in a scene folder and record codec,
              raster, ``nb_frames``, the EXACT rational ``r_frame_rate``,
              ``avg_frame_rate``, duration, ``pix_fmt`` and a streaming
              SHA-256. REFUSES the scene if the cameras disagree on
              resolution, frame count, or rate, naming exactly which ones.

``proxy``     one ffmpeg invocation per camera producing a small, low-rate
              PNG sequence named by TRUE SOURCE FRAME INDEX. Restartable:
              a camera whose per-camera manifest is complete and agrees with
              the requested parameters is skipped.

``census``    over the proxies, a per-camera temporal-difference and
              template-deviation signal, candidate changepoints, and -- the
              column that actually matters -- HOW MANY CAMERAS INDEPENDENTLY
              SUPPORT each candidate, with the per-camera timing spread in
              FRAMES and in MILLISECONDS at the measured rate.

--------------------------------------------------------------------
THIS IS A SCOUTING INSTRUMENT, NOT GROUND TRUTH
--------------------------------------------------------------------

The census output is a RANKED CANDIDATE LIST FOR HUMAN / GROUND-TRUTH
CURATION. It does not establish that any candidate is an event, and this
script never labels one as such. The precedent is explicit and expensive:
the N3V 300-frame curation
([[operations/crb300-event-mask-curation-2026-08-23]]) selected its masks
from GROUND TRUTH ONLY, deliberately excluded model renders from mask
selection, rejected several high-scoring automated candidates only at
frame-by-frame visual verification, and found that the previously FROZEN
0-49 dev masks were NOT confirmed by ground truth at all -- two of the three
appear to label the occluder-PRESENT window rather than the reveal. A
heuristic changepoint on a 480-px proxy is several rungs weaker than the
detector that produced that mistake.

Nothing here freezes or proposes an event DEFINITION. It measures candidates.

--------------------------------------------------------------------
THE FRAME-INDEX MAPPING IS THE LOAD-BEARING PART
--------------------------------------------------------------------

An event onset is only useful if it can be stated in SOURCE frame numbers,
so the proxy must carry that mapping exactly rather than approximately.

* Frames are selected by DECODE-ORDER INDEX -- ``select=not(mod(n-S0\\,S))``
  -- never by the ``fps`` filter's timestamp resampling, whose rounding mode
  and start-time handling would be an assumption about ffmpeg internals.
  This is the same reason ``scripts/imvid_decode_frames.py`` selects with
  ``eq(n,IDX)``: the choice does not depend on container metadata being
  exact.
* The stride is an INTEGER derived from the MEASURED rate:
  ``S = round(R / P)``, with ``R`` the exact ``r_frame_rate`` rational and
  ``P`` the requested proxy rate. The achievable proxy rate is therefore
  ``R/S``, which is recorded exactly alongside the requested one and its
  relative error. The requested rate is a request, not a result.
* Proxy output ``j`` maps to source frame ``n = S0 + j*S``, and the files
  are RENAMED to ``src_<n:06d>.png`` from ffmpeg's sequential ``%06d``
  output. The rename is where the mapping is applied, and the written count
  is checked against the closed-form expected count, fail-closed.
* Frame index -> seconds uses ``n * den/num`` from the measured rational.
  ``scene/dataset_readers.py`` hard-codes ``/30`` (``:200``, ``:700``) and 73
  configs carry ``motion_track_dt: 0.0333333333``; for ImViD's 60000/1001
  that is wrong by 2.002x ([[operations/imvid-baseline-freeze]] B4). No rate
  is ever assumed here -- it is read from the stream or the run refuses.

--------------------------------------------------------------------
FAIL-CLOSED
--------------------------------------------------------------------

A missing ffmpeg/ffprobe; no ``cam*.mp4`` under the source; a camera with no
video stream; cameras disagreeing on raster, frame count or rate; a frame
range outside the declared count; an output root inside the repository, at or
under the raw source directory, or under a path containing ``/raw/``; an
ffmpeg step returning non-zero; a written proxy count differing from the
closed-form expectation; a PNG this reader cannot decode exactly; a census
window with no readable proxy frames.

Dependencies: python stdlib + numpy + ``ffmpeg``/``ffprobe`` subprocesses.
No cv2, no torch, ZERO GPU. PNGs are decoded by a small exact reader here
rather than by an image library, following the same "no image library
needed" habit as ``imvid_decode_frames._png_size``.

--------------------------------------------------------------------
THE DEFAULT SIGNALS ARE WHOLE-FRAME MEANS AND CANNOT SEE A SMALL OBJECT
--------------------------------------------------------------------

All three signals in ``frame_signals`` are means over the WHOLE frame, so a
small object cannot move them: on the 960x540 census raster a 32x32 patch at
25 grey levels moves the frame mean by 0.049 grey levels, 40x below the
absolute floor. ``--tile-mode`` (OFF by default, additive, nothing existing
changes) adds a high-recall companion pass that computes the SAME
``template_dist`` quantity per 60 px tile and screens the MAXIMUM over tiles,
reporting the global signal beside every candidate so the sensitivity bought
is visible. Its frozen preconditions run offline under ``--tile-selftest``.
NO zero-event claim from this instrument is exhaustive without that pass.

--------------------------------------------------------------------
AND THE MAXIMUM-OVER-TILES REDUCTION IS ITSELF REFUTED
--------------------------------------------------------------------

``--per-tile-mode`` (OFF by default, additive, nothing existing changes)
implements [[operations/imvid-tile-scout-v2-per-tile-2026-08-25]], which
SUPERSEDES the detection reduction of ``--tile-mode`` -- and only that
reduction; every measurement the earlier page recorded remains correct about
what it measured. A ground-truth audit established that ``tile_max`` is
MONOPOLISED by the loudest region of the raster, which therefore sets both
the median AND the MAD of the screened scalar: a quiet tile can never move a
maximum it does not win. On `cam12` of `scene6_puppy` the resulting
``median + k_mad*MAD`` threshold EXCEEDS the signal's own maximum in every
window covering either of the two audited real events, so zero candidates was
a STRUCTURAL CERTAINTY rather than a measurement.

The v2 reduction runs the SAME two-gate changepoint detector independently on
each of the 144 per-tile signals, so tile ``(i,j)``'s relative gate is
``median_t(S_ij) + k_mad * 1.4826 * MAD_t(S_ij)`` from THAT TILE's own series
-- a quiet tile gets a quiet threshold. v1's inert 2.0 absolute floor is
replaced by ``F * median over all (camera, window, tile) of
1.4826*MAD_t(S_ij)`` with ``F = 3.0`` frozen, and spatial coherence (a
face-adjacent connected component of >= 3 tiles, fewer than 48 of 144 tiles
firing at that sample) carries the false-positive cost of screening 144
signals instead of 1. Frozen preconditions run offline under
``--per-tile-selftest``.

Usage:
  python3 scripts/imvid_event_proxy.py --self-test
  python3 scripts/imvid_event_proxy.py --tile-selftest
  python3 scripts/imvid_event_proxy.py --per-tile-selftest
  python3 scripts/imvid_event_proxy.py --mode probe \
      --source-dir /apollo/users/sri/proj_adags/data/imvid/raw/scene1_opera \
      --out        /apollo/users/sri/proj_adags/data/imvid/derived/proxy/scene1_opera.probe.json
  python3 scripts/imvid_event_proxy.py --mode proxy \
      --source-dir /apollo/users/sri/proj_adags/data/imvid/raw/scene1_opera \
      --long-edge 480 --proxy-fps 2
  python3 scripts/imvid_event_proxy.py --mode census \
      --proxy-root /apollo/users/sri/proj_adags/data/imvid/derived/proxy/scene1_opera \
      --out        /apollo/users/sri/proj_adags/data/imvid/derived/proxy/scene1_opera.census.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

#: Default derived root. Proxies NEVER land beside the read-only raw files.
DEFAULT_DERIVED_ROOT = "/apollo/users/sri/proj_adags/data/imvid/derived/proxy"

#: Long-edge target and requested proxy rate. Both CLI-settable; both are
#: recorded per camera so a census can never be read against the wrong one.
DEFAULT_LONG_EDGE = 480
DEFAULT_PROXY_FPS = "2"

#: Census window, in SOURCE frames. 300 matches the sample's whole length and
#: the N3V curation's reporting unit.
DEFAULT_WINDOW_FRAMES = 300

#: Robust changepoint threshold: median + K_MAD * (1.4826 * MAD).
DEFAULT_K_MAD = 3.0

#: ABSOLUTE floor, in 8-bit grey levels, that a candidate's template
#: deviation must exceed. A purely scale-free (MAD-relative) screen cannot
#: tell a large excursion from a tiny one on a flat signal -- the exact
#: defect the 2026-08-23 adversarial review recorded against the ratio
#: screen ("a scale-free ratio screen is insufficient ... needs an
#: absolute-magnitude floor"). Both gates must pass.
DEFAULT_MIN_AMPLITUDE = 2.0

#: TILE PASS -- OFF BY DEFAULT, additive to everything above.
#:
#: All three signals in ``frame_signals`` are WHOLE-FRAME MEANS, so the
#: instrument is BLIND TO SMALL OBJECTS: a small object cannot move a
#: whole-frame mean. Measured on the census raster, a 32x32 patch at 25 grey
#: levels moves the 960x540 frame mean by 25*1024/518400 = 0.049 grey levels
#: -- 40x BELOW ``DEFAULT_MIN_AMPLITUDE``, so the absolute gate rejects it and
#: the census would report a clean zero. The tile pass computes the SAME
#: ``template_dist`` quantity over square tiles and screens on the MAXIMUM
#: over tiles, so a localized change is measured against its own tile area
#: instead of against the whole raster. The same patch inside one 60 px tile
#: reads 25*1024/3600 = 7.11 grey levels, 144x larger.
#:
#: 60 px on the 960x540 census raster is a 16 x 9 = 144-tile grid exactly.
DEFAULT_TILE_SIZE = 60

#: ABSOLUTE floor for the tile pass, in 8-bit grey levels, measured on a TILE
#: mean. Same units and the same two-gate role as ``DEFAULT_MIN_AMPLITUDE``
#: (robust relative gate AND absolute floor, both must pass); a separate
#: constant because the two are means over different areas and must never be
#: silently interchanged.
DEFAULT_TILE_MIN_AMPLITUDE = 2.0

#: Contributing tiles reported per candidate for the human-review overlay.
DEFAULT_TILE_TOP_N = 8

#: PER-TILE PASS (v2) -- OFF BY DEFAULT, additive to everything above, and it
#: SUPERSEDES the ``tile_max`` reduction above (only the reduction; every
#: number the v1 page recorded stays correct about what it measured). See
#: [[operations/imvid-tile-scout-v2-per-tile-2026-08-25]].
#:
#: THE ABSOLUTE FLOOR IS DECLARED ON NOISE, NOT ON THE KNOWN EVENTS. v1's flat
#: 2.0 grey levels was measured INERT on real footage -- 119 to 144 of 144
#: tiles exceeded it in every frame, and 95.4% of v1's candidates would also
#: have cleared the WHOLE-FRAME floor, so the tile pass fired uniquely on 4.6%
#: of its own output. The replacement is scaled to the take's own per-tile
#: noise:
#:
#:     floor = F * median over all (camera, window, tile) of
#:                 [ 1.4826 * MAD_t( S_ij ) ]
#:
#: ``F = 3.0`` is a DECLARED JUDGMENT fixed in the frozen spec, chosen to match
#: the existing relative gate's ``k_mad = 3.0`` so the two gates express the
#: same strictness in different denominators. It is NOT derived from data and
#: every emitted record says so. The measured floor is recorded in the census
#: manifest.
DEFAULT_PER_TILE_FLOOR_F = 3.0

#: Supplementary sensitivity sweep over ``F``. THE PRIMARY READING IS F = 3.0
#: ONLY; every other point is labelled a sensitivity probe in the JSON and may
#: not be reported as the census.
PER_TILE_FLOOR_F_SWEEP = (1.5, 2.0, 3.0, 4.5, 6.0)

#: SPATIAL COHERENCE, frozen. Per-tile detection raises the false-positive
#: rate by construction (144 signals instead of 1). The discriminator is
#: measured: the audited real events fire 4-9 CONTIGUOUS tiles, illumination
#: artefacts fire 38-75 tiles at once. Both bounds are DECLARED JUDGMENTS
#: derived from observation of a known-positive and a known-negative, and that
#: dependence is disclosed rather than hidden.
#:
#: Lower bound: a face-adjacent (4-connected) component of at least this many
#: tiles. Admits the smallest observed real event (4 tiles) with one tile of
#: margin. DIAGONAL neighbours are NOT adjacent.
DEFAULT_PER_TILE_MIN_COMPONENT = 3

#: Upper bound, EXCLUSIVE: a sample with this many or more tiles firing is a
#: global change, not a localized one. 48 of 144 = 33%, below the smallest
#: observed artefact (38 tiles).
DEFAULT_PER_TILE_MAX_FIRING_TILES = 48

#: Wording carried in the census JSON and in the docstring of every function
#: that produces it. The N3V curation precedent
#: ([[operations/crb300-event-mask-curation-2026-08-23]]) is that automated
#: spatial support is routinely mistaken for object extent; it is not.
TILE_EXPLANATION_NOTE = (
    "EXPLANATION OF THE DETECTOR SIGNAL, NOT AN INSTANCE MASK. These tile "
    "values state WHERE ON THE PROXY RASTER the scalar that the detector "
    "thresholded came from. They do not segment an object, do not bound one, "
    "and do not assert that anything is present, absent, revealed or occluded "
    "in any tile. A tile is high because the pixels inside it departed from "
    "this window's own per-pixel temporal median, which a moving object, a "
    "moving occluder, a shadow, a lighting change or compression noise all "
    "produce equally."
)

#: ImViD's stated synchronization uncertainty, milliseconds. Reference only;
#: nothing here measures it. See the census resolution note.
IMVID_SYNC_UNCERTAINTY_MS = (10.0, 20.0)

DISCLAIMER = (
    "THIS IS A SCOUTING INSTRUMENT, NOT GROUND TRUTH. Every row below is a "
    "CANDIDATE for human / ground-truth curation. No candidate is claimed to "
    "be an event. Candidate polarity (rise/fall) is a direction of the "
    "measured signal, NOT an established disappearance or reappearance."
)


# ---------------------------------------------------------------------------
# Exact rational rates, strides, and the frame-index mapping
# ---------------------------------------------------------------------------

def parse_rational(text: str) -> Fraction:
    """``'60000/1001'`` -> ``Fraction(60000, 1001)``, EXACTLY.

    Never converted to float on the way in. ``60000/1001`` is 59.94005994...
    and the whole point of carrying the rational is that no step re-derives
    a rate from a rounded float.
    """
    if text is None:
        raise ContractError("no frame rate given")
    value = str(text).strip()
    if value in ("", "N/A", "0/0"):
        raise ContractError(f"unusable frame rate {text!r} (stream reports no rate)")
    if not re.fullmatch(r"\d+(/\d+)?|\d+\.\d+", value):
        raise ContractError(f"unparsable frame rate {text!r}")
    try:
        rate = Fraction(value)
    except (ZeroDivisionError, ValueError) as exc:
        raise ContractError(f"unparsable frame rate {text!r}: {exc}") from exc
    if rate <= 0:
        raise ContractError(f"non-positive frame rate {text!r}")
    return rate


def rational_str(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def derive_stride(source_rate: Fraction, proxy_rate: Fraction) -> int:
    """Integer decode-index stride ``S = round(R/P)``, clamped to ``>= 1``.

    Clamped because a requested proxy rate ABOVE the source rate cannot be
    honoured by subsampling; the honest result is every frame (S=1) plus a
    recorded relative error, not a fabricated interpolation.
    """
    if source_rate <= 0 or proxy_rate <= 0:
        raise ContractError("rates must be positive")
    ratio = source_rate / proxy_rate
    # round-half-to-even on an exact rational, no float anywhere
    floor = ratio.numerator // ratio.denominator
    remainder = ratio - floor
    if remainder > Fraction(1, 2):
        stride = floor + 1
    elif remainder < Fraction(1, 2):
        stride = floor
    else:
        stride = floor if floor % 2 == 0 else floor + 1
    return max(1, int(stride))


def proxy_index_to_source(j: int, start_frame: int, stride: int) -> int:
    if j < 0:
        raise ContractError(f"negative proxy index {j}")
    return start_frame + j * stride


def source_to_proxy_index(n: int, start_frame: int, stride: int) -> int:
    """Inverse of :func:`proxy_index_to_source`; refuses non-sampled ``n``."""
    if n < start_frame:
        raise ContractError(f"source frame {n} precedes start {start_frame}")
    offset = n - start_frame
    if offset % stride:
        raise ContractError(
            f"source frame {n} is not on the proxy lattice "
            f"(start {start_frame}, stride {stride})"
        )
    return offset // stride


def expected_proxy_count(start_frame: int, end_frame: int, stride: int) -> int:
    """How many frames ``select=between*not(mod)`` must emit. Closed form."""
    if stride < 1:
        raise ContractError(f"stride must be >= 1, got {stride}")
    if end_frame < start_frame:
        return 0
    return (end_frame - start_frame) // stride + 1


def frames_to_ms(n_frames: float, source_rate: Fraction) -> float:
    """Frames -> milliseconds at the MEASURED rate. Never at 30, never at 60."""
    return float(Fraction(1000) * source_rate.denominator
                 / source_rate.numerator) * float(n_frames)


def scaled_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Aspect-preserving resize to a long-edge target, computed HERE.

    Computed in Python rather than delegated to a ``scale=-1`` expression so
    the exact output raster is known before ffmpeg runs and can be written
    into the manifest and checked against the produced PNG.
    """
    if width < 1 or height < 1:
        raise ContractError(f"bad source raster {width}x{height}")
    if long_edge < 8:
        raise ContractError(f"--long-edge {long_edge} is too small to be useful")
    if width >= height:
        out_w = int(long_edge)
        out_h = max(1, int(round(height * long_edge / width)))
    else:
        out_h = int(long_edge)
        out_w = max(1, int(round(width * long_edge / height)))
    return out_w, out_h


# ---------------------------------------------------------------------------
# Small utilities in the house style
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run(args: list[str], label: str, timeout: int = 36000) -> dict:
    started = time.perf_counter()
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except OSError as exc:
        raise ContractError(f"{label} could not be launched: {exc}") from exc
    elapsed = time.perf_counter() - started
    if out.returncode != 0:
        raise ContractError(
            f"{label} failed (rc={out.returncode}): {out.stderr[-1500:]}"
        )
    return {"label": label, "argv": args, "seconds": elapsed,
            "stdout": out.stdout, "stderr_tail": out.stderr[-2000:]}


def _tool_version(binary: str) -> str:
    out = subprocess.run([binary, "-version"], capture_output=True, text=True,
                         timeout=120)
    if out.returncode != 0:
        raise ContractError(f"{binary} -version failed: {out.stderr[:400]}")
    return out.stdout.strip()


def _require_tools() -> None:
    for binary in (FFMPEG, FFPROBE):
        if shutil.which(binary) is None:
            raise ContractError(f"{binary} not on PATH")


def _check_output_root(root: Path, source: Path | None) -> None:
    """Refuse to write into the repository or anywhere near the raw data."""
    resolved = root.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ContractError(f"output root {root} is inside the repository")
    if source is not None:
        src = source.resolve()
        if resolved == src or src in resolved.parents:
            raise ContractError(
                f"output root {root} is at or inside the READ-ONLY source {source}"
            )
    if any(part == "raw" for part in resolved.parts):
        raise ContractError(
            f"output root {root} lies under a 'raw' path component; raw ImViD "
            "files are read-only (0444) and derived artifacts never go there"
        )


# ---------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------

def ffprobe_video(video: Path, count_frames: bool = False) -> dict:
    """Everything the mapping needs, plus the fields the manifest records."""
    entries = ("stream=codec_name,width,height,nb_frames,nb_read_frames,"
               "r_frame_rate,avg_frame_rate,duration,pix_fmt,time_base")
    argv = [FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", entries + ":format=duration,format_name",
            "-of", "json"]
    if count_frames:
        argv += ["-count_frames"]
    argv += [str(video)]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=36000)
    if out.returncode != 0:
        raise ContractError(f"ffprobe failed on {video}: {out.stderr[:400]}")
    payload = json.loads(out.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise ContractError(f"no video stream in {video}")
    stream = streams[0]
    fmt = payload.get("format") or {}

    def _int(value):
        return int(value) if value not in (None, "N/A", "") else None

    def _float(value):
        return float(value) if value not in (None, "N/A", "") else None

    return {
        "codec_name": stream.get("codec_name"),
        "width": _int(stream.get("width")),
        "height": _int(stream.get("height")),
        "nb_frames": _int(stream.get("nb_frames")),
        "nb_read_frames": _int(stream.get("nb_read_frames")),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "time_base": stream.get("time_base"),
        "pix_fmt": stream.get("pix_fmt"),
        "stream_duration_s": _float(stream.get("duration")),
        "format_duration_s": _float(fmt.get("duration")),
        "format_name": fmt.get("format_name"),
    }


def _agree_or_refuse(probes: dict[str, dict], field: str, label: str) -> None:
    """Refuse a heterogeneous scene, NAMING the dissenting cameras."""
    groups: dict[object, list[str]] = {}
    for camera, entry in probes.items():
        groups.setdefault(entry[field], []).append(camera)
    if len(groups) <= 1:
        return
    majority = max(groups, key=lambda key: len(groups[key]))
    detail = "; ".join(
        f"{value!r}: {len(cams)} camera(s) {sorted(cams)}"
        for value, cams in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    )
    dissenting = sorted(c for value, cams in groups.items()
                        if value != majority for c in cams)
    raise ContractError(
        f"cameras disagree on {label}: {detail}. Dissenting from the majority "
        f"({majority!r}): {dissenting}. REFUSED -- a heterogeneous scene "
        "breaks the shared frame-index/time mapping every downstream step uses."
    )


def probe_scene(source: Path, count_frames: bool = False,
                skip_hash: bool = False) -> dict:
    videos = sorted(source.glob("cam*.mp4"))
    if not videos:
        raise ContractError(f"no cam*.mp4 under {source}")

    probes: dict[str, dict] = {}
    for video in videos:
        entry = ffprobe_video(video, count_frames=count_frames)
        entry["path"] = str(video)
        entry["bytes"] = video.stat().st_size
        entry["sha256"] = None if skip_hash else _sha256(video)
        # exact rational, carried as a string so JSON never floats it
        rate = parse_rational(entry["r_frame_rate"])
        entry["r_frame_rate_exact"] = rational_str(rate)
        entry["r_frame_rate_float"] = float(rate)
        entry["frame_period_s_exact"] = rational_str(1 / rate)
        probes[video.stem] = entry
        print(f"[imvid-proxy] {video.stem}: {entry['codec_name']} "
              f"{entry['width']}x{entry['height']} nb_frames={entry['nb_frames']} "
              f"rate={entry['r_frame_rate_exact']} "
              f"({entry['r_frame_rate_float']:.5f} fps) "
              f"{entry['bytes'] / 2**30:.3f} GiB", flush=True)

    _agree_or_refuse(probes, "width", "raster width")
    _agree_or_refuse(probes, "height", "raster height")
    _agree_or_refuse(probes, "nb_frames", "frame count (nb_frames)")
    _agree_or_refuse(probes, "r_frame_rate_exact", "r_frame_rate")

    first = probes[sorted(probes)[0]]
    rate = parse_rational(first["r_frame_rate_exact"])
    n_frames = first["nb_frames"]
    return {
        "schema": "imvid-event-proxy-probe-v1",
        "source_dir": str(source),
        "n_cameras": len(videos),
        "agreed": {
            "width": first["width"],
            "height": first["height"],
            "nb_frames": n_frames,
            "r_frame_rate_exact": first["r_frame_rate_exact"],
            "r_frame_rate_float": float(rate),
            "frame_period_s_exact": rational_str(1 / rate),
            "frame_period_ms": frames_to_ms(1, rate),
            "codec_name": first["codec_name"],
            "pix_fmt": first["pix_fmt"],
        },
        "duration_s_from_rate": (float(n_frames / rate) if n_frames else None),
        "nb_frames_source": ("nb_read_frames (decoded)" if count_frames
                             else "container nb_frames (NOT decoded; may lie)"),
        "hashes_computed": not skip_hash,
        "ffprobe_version": _tool_version(FFPROBE),
        "cameras": probes,
    }


# ---------------------------------------------------------------------------
# An exact, dependency-free PNG reader
# ---------------------------------------------------------------------------

_PNG_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def _unfilter(raw: bytes, width: int, height: int, channels: int) -> np.ndarray:
    stride = width * channels
    if len(raw) != height * (stride + 1):
        raise ContractError(
            f"PNG scanline payload is {len(raw)} bytes, expected "
            f"{height * (stride + 1)} for {width}x{height}x{channels}"
        )
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride + 1)
    filters = buf[:, 0]
    body = buf[:, 1:]
    if not filters.any():
        # ffmpeg is invoked with `-pred none`, so this is the normal path.
        return body.reshape(height, width, channels).copy()

    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    bpp = channels
    for y in range(height):
        kind = int(filters[y])
        cur = body[y].copy()
        if kind == 0:
            rec = cur
        elif kind == 1:
            rec = cur
            for lane in range(bpp):
                rec[lane::bpp] = (np.cumsum(rec[lane::bpp].astype(np.int64))
                                  % 256).astype(np.uint8)
        elif kind == 2:
            rec = ((cur.astype(np.int64) + prev.astype(np.int64)) % 256
                   ).astype(np.uint8)
        elif kind in (3, 4):
            rec = np.zeros(stride, dtype=np.uint8)
            for i in range(stride):
                a = int(rec[i - bpp]) if i >= bpp else 0
                b = int(prev[i])
                if kind == 3:
                    rec[i] = (int(cur[i]) + ((a + b) >> 1)) & 0xFF
                else:
                    c = int(prev[i - bpp]) if i >= bpp else 0
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                    rec[i] = (int(cur[i]) + pred) & 0xFF
        else:
            raise ContractError(f"unknown PNG filter type {kind} on row {y}")
        out[y] = rec
        prev = rec
    return out.reshape(height, width, channels)


def read_png(path: Path) -> np.ndarray:
    """8-bit non-interlaced PNG -> ``(H, W, C)`` uint8. Exact, no library.

    Deliberately narrow: 8-bit, non-interlaced, non-palette. Those are the
    only PNGs this pipeline creates, and a reader that silently coped with
    more would be able to misread something it should refuse.
    """
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ContractError(f"{path} is not a PNG")
    pos, ihdr, idat = 8, None, []
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            ihdr = body
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if ihdr is None or len(ihdr) < 13:
        raise ContractError(f"{path} has no IHDR")
    width = int.from_bytes(ihdr[0:4], "big")
    height = int.from_bytes(ihdr[4:8], "big")
    bit_depth, colour_type = ihdr[8], ihdr[9]
    interlace = ihdr[12]
    if bit_depth != 8:
        raise ContractError(f"{path}: bit depth {bit_depth}, only 8 supported")
    if interlace != 0:
        raise ContractError(f"{path}: interlaced PNG not supported")
    if colour_type not in _PNG_CHANNELS:
        raise ContractError(f"{path}: colour type {colour_type} not supported")
    if not idat:
        raise ContractError(f"{path} has no IDAT")
    try:
        raw = zlib.decompress(b"".join(idat))
    except zlib.error as exc:
        raise ContractError(f"{path}: corrupt or truncated IDAT ({exc})") from exc
    return _unfilter(raw, width, height, _PNG_CHANNELS[colour_type])


#: Rec.601 luma. Recorded in the manifest so a later reader knows exactly
#: what the census signal was computed on.
LUMA_WEIGHTS = (0.299, 0.587, 0.114)


def png_to_gray(path: Path) -> np.ndarray:
    arr = read_png(path)
    channels = arr.shape[2]
    if channels in (1, 2):
        return arr[:, :, 0].astype(np.float32)
    weights = np.asarray(LUMA_WEIGHTS, dtype=np.float32)
    return (arr[:, :, :3].astype(np.float32) * weights).sum(axis=2)


# ---------------------------------------------------------------------------
# Mode: proxy
# ---------------------------------------------------------------------------

PROXY_MANIFEST_NAME = "MANIFEST.imvid_proxy.json"
PROXY_FRAME_GLOB = "src_*.png"
PROXY_FRAME_RE = re.compile(r"^src_(\d{6,})\.png$")


def _listing_hash(paths: list[Path], root: Path) -> str:
    """Order-independent hash over ``name:bytes`` of the produced listing."""
    rows = sorted(f"{p.relative_to(root).as_posix()}:{p.stat().st_size}"
                  for p in paths)
    return _sha256_text("\n".join(rows))


def _proxy_is_complete(camera_dir: Path, wanted: dict) -> bool:
    manifest = camera_dir / PROXY_MANIFEST_NAME
    if not manifest.is_file():
        return False
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("schema") != "imvid-event-proxy-camera-v1":
        return False
    for key, value in wanted.items():
        if record.get("mapping", {}).get(key) != value:
            return False
    frames = sorted((camera_dir / "frames").glob(PROXY_FRAME_GLOB))
    if len(frames) != record.get("n_output_frames"):
        return False
    return _listing_hash(frames, camera_dir) == record.get("listing_sha256")


def build_camera_proxy(video: Path, camera_dir: Path, *, source_rate: Fraction,
                       requested_rate: Fraction, stride: int, start_frame: int,
                       end_frame: int, out_w: int, out_h: int, src_w: int,
                       src_h: int, pix_fmt: str, sws_flags: str,
                       source_sha256: str | None, ffmpeg_version: str) -> dict:
    frames_dir = camera_dir / "frames"
    staging = camera_dir / "_staging"
    for directory in (frames_dir, staging):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    select = (f"select=between(n\\,{start_frame}\\,{end_frame})"
              f"*not(mod(n-{start_frame}\\,{stride}))")
    vfilter = f"{select},scale={out_w}:{out_h}"
    argv = [
        FFMPEG, "-v", "error", "-y", "-nostdin",
        "-i", str(video),
        "-an", "-sn", "-dn",
        "-vf", vfilter,
        "-vsync", "0",
        "-sws_flags", sws_flags,
        "-pix_fmt", pix_fmt,
        "-pred", "none",          # PNG filter type 0 on every row
        "-start_number", "0",
        str(staging / "_seq_%06d.png"),
    ]
    step = _run(argv, f"ffmpeg[{video.stem}]")

    produced = sorted(staging.glob("_seq_*.png"))
    expected = expected_proxy_count(start_frame, end_frame, stride)
    if len(produced) != expected:
        raise ContractError(
            f"{video.stem}: ffmpeg wrote {len(produced)} proxy frames, the "
            f"mapping expects {expected} (start {start_frame}, end "
            f"{end_frame}, stride {stride}). REFUSED -- the frame-index "
            "mapping and the produced files must agree exactly or every "
            "reported source frame number is wrong."
        )

    written = []
    for path in produced:
        j = int(path.stem.rsplit("_", 1)[1])
        source_index = proxy_index_to_source(j, start_frame, stride)
        target = frames_dir / f"src_{source_index:06d}.png"
        path.rename(target)
        written.append(target)
    shutil.rmtree(staging)

    # Verify the produced raster against the size computed BEFORE ffmpeg ran.
    probe_arr = read_png(written[0])
    if (probe_arr.shape[1], probe_arr.shape[0]) != (out_w, out_h):
        raise ContractError(
            f"{video.stem}: proxy raster is {probe_arr.shape[1]}x"
            f"{probe_arr.shape[0]}, expected {out_w}x{out_h}"
        )

    record = {
        "schema": "imvid-event-proxy-camera-v1",
        "camera": video.stem,
        "source_video": str(video),
        "source_sha256": source_sha256,
        "source_raster": [src_w, src_h],
        "proxy_raster": [out_w, out_h],
        "scale_filter": f"scale={out_w}:{out_h}",
        "sws_flags": sws_flags,
        "pix_fmt": pix_fmt,
        "mapping": {
            "source_rate_exact": rational_str(source_rate),
            "source_rate_float": float(source_rate),
            "requested_proxy_rate_exact": rational_str(requested_rate),
            "stride_frames": stride,
            "effective_proxy_rate_exact": rational_str(source_rate / stride),
            "effective_proxy_rate_float": float(source_rate / stride),
            "rate_relative_error": float(
                abs(source_rate / stride - requested_rate) / requested_rate),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "rule": ("source_frame = start_frame + proxy_index * stride; "
                     "selection is by DECODE-ORDER INDEX, never by timestamp"),
            "frame_period_s_exact": rational_str(1 / source_rate),
            "proxy_step_ms": frames_to_ms(stride, source_rate),
        },
        "source_frame_range_written": [
            proxy_index_to_source(0, start_frame, stride),
            proxy_index_to_source(len(written) - 1, start_frame, stride),
        ],
        "n_output_frames": len(written),
        "n_output_frames_expected": expected,
        "ffmpeg_argv": argv,
        "ffmpeg_command": " ".join(argv),
        "ffmpeg_version": ffmpeg_version,
        "ffmpeg_seconds": step["seconds"],
        "listing_sha256": _listing_hash(written, camera_dir),
        "bytes": sum(p.stat().st_size for p in written),
    }
    (camera_dir / PROXY_MANIFEST_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def mode_proxy(args) -> dict:
    _require_tools()
    source = Path(args.source_dir)
    scene = args.scene or source.name
    out_root = Path(args.out_root) if args.out_root else Path(args.derived_root) / scene
    _check_output_root(out_root, source)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.probe_manifest:
        probe = json.loads(Path(args.probe_manifest).read_text(encoding="utf-8"))
        if probe.get("schema") != "imvid-event-proxy-probe-v1":
            raise ContractError(
                f"--probe-manifest {args.probe_manifest} is not an "
                "imvid-event-proxy-probe-v1 manifest")
    else:
        probe = probe_scene(source, count_frames=False, skip_hash=args.skip_hash)

    agreed = probe["agreed"]
    source_rate = parse_rational(agreed["r_frame_rate_exact"])
    requested = parse_rational(args.proxy_fps)
    stride = derive_stride(source_rate, requested)
    src_w, src_h = int(agreed["width"]), int(agreed["height"])
    out_w, out_h = scaled_size(src_w, src_h, int(args.long_edge))

    n_frames = agreed["nb_frames"]
    if n_frames is None and args.end_frame is None:
        raise ContractError(
            "the container reports no nb_frames and no --end-frame was given; "
            "refusing to guess the frame count")
    start_frame = int(args.start_frame)
    end_frame = int(args.end_frame) if args.end_frame is not None else int(n_frames) - 1
    if start_frame < 0:
        raise ContractError(f"--start-frame {start_frame} is negative")
    if n_frames is not None and end_frame > int(n_frames) - 1:
        raise ContractError(
            f"--end-frame {end_frame} exceeds the declared count "
            f"({n_frames} frames -> last index {int(n_frames) - 1})")
    if end_frame < start_frame:
        raise ContractError(f"--end-frame {end_frame} precedes --start-frame {start_frame}")

    ffmpeg_version = _tool_version(FFMPEG)
    expected = expected_proxy_count(start_frame, end_frame, stride)
    print(f"[imvid-proxy] scene {scene}: source {src_w}x{src_h} @ "
          f"{rational_str(source_rate)} ({float(source_rate):.5f} fps)", flush=True)
    print(f"[imvid-proxy] requested {rational_str(requested)} fps -> stride "
          f"{stride} -> effective {rational_str(source_rate / stride)} "
          f"({float(source_rate / stride):.5f} fps, rel err "
          f"{float(abs(source_rate / stride - requested) / requested):.4%})",
          flush=True)
    print(f"[imvid-proxy] frames {start_frame}..{end_frame} -> {expected} proxy "
          f"frames/camera at {out_w}x{out_h}; one proxy step = {stride} source "
          f"frames = {frames_to_ms(stride, source_rate):.2f} ms", flush=True)

    cameras = sorted(probe["cameras"])
    if args.limit_cameras is not None:
        cameras = cameras[:int(args.limit_cameras)]

    wanted = {
        "source_rate_exact": rational_str(source_rate),
        "stride_frames": stride,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }

    records, skipped = {}, []
    for camera in cameras:
        camera_dir = out_root / camera
        if not args.force and _proxy_is_complete(camera_dir, wanted):
            record = json.loads((camera_dir / PROXY_MANIFEST_NAME)
                                .read_text(encoding="utf-8"))
            #  The mapping agrees (that is what "complete" checked), but the
            #  RENDERING parameters might not. Refuse rather than silently
            #  rebuild: a full-take proxy set is hours of decode and a
            #  fat-fingered --long-edge must not destroy one.
            existing = {
                "proxy_raster": list(record["proxy_raster"]),
                "pix_fmt": record.get("pix_fmt"),
                "sws_flags": record.get("sws_flags"),
            }
            requested_render = {
                "proxy_raster": [out_w, out_h],
                "pix_fmt": args.pix_fmt,
                "sws_flags": args.sws_flags,
            }
            if existing != requested_render:
                raise ContractError(
                    f"{camera}: an existing COMPLETE proxy was built with "
                    f"{existing} but {requested_render} was requested. REFUSED "
                    "-- pass --force to rebuild it, or a different --out-root "
                    "to keep both.")
            records[camera] = record
            skipped.append(camera)
            print(f"[imvid-proxy] {camera}: complete, skipped", flush=True)
            continue
        camera_dir.mkdir(parents=True, exist_ok=True)
        entry = probe["cameras"][camera]
        record = build_camera_proxy(
            Path(entry["path"]), camera_dir,
            source_rate=source_rate, requested_rate=requested, stride=stride,
            start_frame=start_frame, end_frame=end_frame,
            out_w=out_w, out_h=out_h, src_w=src_w, src_h=src_h,
            pix_fmt=args.pix_fmt, sws_flags=args.sws_flags,
            source_sha256=entry.get("sha256"), ffmpeg_version=ffmpeg_version,
        )
        records[camera] = record
        print(f"[imvid-proxy] {camera}: {record['n_output_frames']} frames, "
              f"{record['bytes'] / 2**20:.1f} MiB, "
              f"{record['ffmpeg_seconds']:.1f}s", flush=True)

    report = {
        "schema": "imvid-event-proxy-scene-v1",
        "scene": scene,
        "source_dir": str(source),
        "proxy_root": str(out_root),
        "n_cameras": len(records),
        "cameras_skipped_complete": skipped,
        "mapping": {
            "source_rate_exact": rational_str(source_rate),
            "source_rate_float": float(source_rate),
            "requested_proxy_rate_exact": rational_str(requested),
            "stride_frames": stride,
            "effective_proxy_rate_exact": rational_str(source_rate / stride),
            "effective_proxy_rate_float": float(source_rate / stride),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_proxy_frames_per_camera": expected,
            "proxy_step_ms": frames_to_ms(stride, source_rate),
            "rule": "source_frame = start_frame + proxy_index * stride",
        },
        "source_raster": [src_w, src_h],
        "proxy_raster": [out_w, out_h],
        "pix_fmt": args.pix_fmt,
        "sws_flags": args.sws_flags,
        "ffmpeg_version": ffmpeg_version,
        "ffprobe_version": probe.get("ffprobe_version"),
        "total_bytes": sum(r["bytes"] for r in records.values()),
        "cameras": records,
        "disclaimer": DISCLAIMER,
    }
    return report


# ---------------------------------------------------------------------------
# Mode: census
# ---------------------------------------------------------------------------

def frame_signals(stack: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame summaries of a ``(T, H, W)`` float32 proxy window.

    ``absdiff_mean``   mean |I_t - I_{t-1}|, a MOTION summary. First entry 0.
    ``template_dist``  mean |I_t - median_t(I)|, deviation from the window's
                       own per-pixel temporal median. This is the signal the
                       N3V curation used for occluder detection, and it is
                       the one changepoints are taken on: a covered surface
                       departs from the template and a revealed one returns
                       to it, whereas adjacent-frame difference only sees the
                       transition itself.
    ``changed_frac``   fraction of pixels whose |I_t - median| exceeds 8/255
                       of full scale; a coarse spatial-extent summary.
    """
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ContractError(
            f"need at least 2 proxy frames, got shape {tuple(stack.shape)}")
    template = np.median(stack, axis=0)
    deviation = np.abs(stack - template[None, :, :])
    absdiff = np.zeros(stack.shape[0], dtype=np.float64)
    absdiff[1:] = np.abs(np.diff(stack, axis=0)).mean(axis=(1, 2))
    return {
        "absdiff_mean": absdiff,
        "template_dist": deviation.mean(axis=(1, 2)).astype(np.float64),
        "changed_frac": (deviation > 8.0).mean(axis=(1, 2)).astype(np.float64),
    }


def tile_edges(extent: int, tile_size: int) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` spans covering ``0..extent`` EXACTLY ONCE.

    Partial edge tiles are INCLUDED and are neither padded, dropped, nor
    merged into their neighbour: the last span is short whenever ``tile_size``
    does not divide ``extent``. NOTHING IS WEIGHTED BY AREA -- each tile's
    statistic is a mean over its own pixels, so a short edge tile is a mean
    over fewer pixels and is still read in grey levels, directly comparable
    with a full tile.
    """
    if int(tile_size) < 1:
        raise ContractError(f"tile size must be >= 1, got {tile_size}")
    if int(extent) < 1:
        raise ContractError(f"raster extent must be >= 1, got {extent}")
    extent, tile_size = int(extent), int(tile_size)
    return [(s, min(s + tile_size, extent)) for s in range(0, extent, tile_size)]


def tile_pixel_box(row: int, col: int, row_spans: list[tuple[int, int]],
                   col_spans: list[tuple[int, int]]) -> list[int]:
    """Pixel box ``[x0, y0, x1, y1]`` (half-open) of tile ``(row, col)``.

    Coordinates are on the PROXY raster, not the source raster. A reader that
    wants source pixels must rescale by the recorded proxy raster; nothing
    here does that for them, because the proxy scale factor is a property of
    how the proxy was built and is recorded in the proxy manifest.
    """
    y0, y1 = row_spans[int(row)]
    x0, x1 = col_spans[int(col)]
    return [int(x0), int(y0), int(x1), int(y1)]


def tiled_template_signals(stack: np.ndarray,
                           tile_size: int = DEFAULT_TILE_SIZE) -> dict:
    """TILE-SENSITIVE companion to ``frame_signals``; changes nothing in it.

    Computes the SAME quantity as ``frame_signals``' ``template_dist`` --
    ``|I_t - median_t(I)|`` -- but averaged over each square tile instead of
    over the whole frame, and reduces the grid to a scalar per frame with a
    MAXIMUM over tiles. The maximum is the load-bearing choice: a mean over
    tiles is algebraically an area-weighted whole-frame mean again (exactly
    it, when the tiling is regular), i.e. the very blindness this pass exists
    to remove.

    Returns
    -------
    ``tile_template_dist``  ``(T, n_tile_y, n_tile_x)`` float64 grid.
    ``tile_max``            ``(T,)`` float64, max over tiles -- the scalar the
                            changepoint detector runs on in tile mode.
    ``tile_argmax``         ``(T, 2)`` int, ``(row, col)`` of that maximum.
    ``row_spans``/``col_spans``  the exact half-open pixel spans.
    ``n_tile_y``/``n_tile_x``/``tile_size``/``proxy_raster``.

    The grid is an EXPLANATION OF THE DETECTOR SIGNAL, NOT AN INSTANCE MASK
    (see ``TILE_EXPLANATION_NOTE``).

    MEMORY. The per-pixel temporal median is independent per pixel, so the
    median of a spatial slice equals that slice of the whole-frame median;
    this walks tile blocks and never materializes a second ``(T, H, W)``
    array. Peak above the caller's ``(T, H, W)`` float32 ``stack`` is one
    ``np.median`` partition copy plus one absolute-deviation buffer, both
    TILE-SIZED: for T=594, 60x60 tiles that is 2 x 594*3600*4 B = 17.1 MiB,
    plus the ``(T, 9, 16)`` float64 output grid (0.65 MiB). For the census
    raster the ``stack`` itself is 594*540*960*4 B = 1.147 GiB and dominates.
    A whole-frame ``(T, H, W)`` float64 deviation copy, which this avoids,
    would have been 2.29 GiB.

    MEASURED (Windows peak working set, T=120 at 540x960, stack 237.3 MiB):
    this function adds 3.6 MiB to the process peak; ``frame_signals`` on the
    same stack adds 707.1 MiB, because it materializes the median partition
    copy and a whole-frame deviation array. Scaling the measurement to
    T=594 gives ~18 MiB here against ~3.5 GiB there. The tile pass is
    therefore NOT the memory constraint on a long window; the pre-existing
    global path is, and it is left exactly as it was.
    """
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ContractError(
            f"need at least 2 proxy frames, got shape {tuple(stack.shape)}")
    n_frames, height, width = (int(v) for v in stack.shape)
    row_spans = tile_edges(height, tile_size)
    col_spans = tile_edges(width, tile_size)
    grid = np.zeros((n_frames, len(row_spans), len(col_spans)), dtype=np.float64)
    for i, (y0, y1) in enumerate(row_spans):
        for j, (x0, x1) in enumerate(col_spans):
            block = stack[:, y0:y1, x0:x1]
            template = np.median(block, axis=0)
            grid[:, i, j] = np.abs(block - template[None, :, :]).mean(
                axis=(1, 2), dtype=np.float64)
    flat = grid.reshape(n_frames, -1)
    arg = np.argmax(flat, axis=1)
    tile_max = flat[np.arange(n_frames), arg].astype(np.float64)
    tile_argmax = np.stack([arg // len(col_spans), arg % len(col_spans)],
                           axis=1).astype(np.int64)
    return {
        "tile_template_dist": grid,
        "tile_max": tile_max,
        "tile_argmax": tile_argmax,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "n_tile_y": len(row_spans),
        "n_tile_x": len(col_spans),
        "tile_size": int(tile_size),
        "proxy_raster": [width, height],
    }


def tile_candidate_explanation(tiled: dict, proxy_index: int, *,
                               source_frame: int, top_n: int,
                               global_template_dist: float,
                               global_template_dist_before: float,
                               global_min_amplitude: float,
                               tile_min_amplitude: float) -> dict:
    """Spatial explanation of ONE tile-mode candidate, for the review overlay.

    THIS IS AN EXPLANATION OF THE DETECTOR SIGNAL, NOT AN INSTANCE MASK. It
    carries the tile grid at the candidate frame (enough to draw a heatmap),
    the top contributing tiles with their pixel boxes on the proxy raster,
    and -- so the reader can see exactly how much sensitivity the tile pass
    bought and nothing is hidden -- the GLOBAL whole-frame ``template_dist``
    at the same frame, which is what the default census would have screened.
    """
    grid = tiled["tile_template_dist"]
    j = int(proxy_index)
    plane = np.asarray(grid[j], dtype=np.float64)
    order = np.argsort(plane, axis=None)[::-1][:max(int(top_n), 1)]
    top = []
    for flat_index in order:
        row = int(flat_index) // tiled["n_tile_x"]
        col = int(flat_index) % tiled["n_tile_x"]
        box = tile_pixel_box(row, col, tiled["row_spans"], tiled["col_spans"])
        top.append({
            "tile_row": row,
            "tile_col": col,
            "value": round(float(plane[row, col]), 4),
            "pixel_box_xyxy": box,
            "pixel_box_is_tile_extent_not_object_extent": True,
            "n_pixels": int((box[3] - box[1]) * (box[2] - box[0])),
        })
    argmax = [int(v) for v in tiled["tile_argmax"][j]]
    tile_value = float(tiled["tile_max"][j])
    gain = (tile_value / global_template_dist
            if global_template_dist > 0 else None)
    return {
        "what_this_is": TILE_EXPLANATION_NOTE,
        "is_instance_mask": False,
        "kind": "detector_signal_explanation",
        "source_frame": int(source_frame),
        "proxy_index_in_window": j,
        "tile_size_px": int(tiled["tile_size"]),
        "proxy_raster": list(tiled["proxy_raster"]),
        "grid_shape": [int(tiled["n_tile_y"]), int(tiled["n_tile_x"])],
        "tile_template_dist_grid": [[round(float(v), 4) for v in row]
                                    for row in plane],
        "tile_max": round(tile_value, 4),
        "tile_argmax_row_col": argmax,
        "tile_argmax_pixel_box_xyxy": tile_pixel_box(
            argmax[0], argmax[1], tiled["row_spans"], tiled["col_spans"]),
        "top_tiles": top,
        "global_template_dist_at_candidate": round(
            float(global_template_dist), 6),
        "global_template_dist_before_candidate": round(
            float(global_template_dist_before), 6),
        "tile_max_over_global_template_dist": (round(float(gain), 3)
                                               if gain is not None else None),
        "global_absolute_floor_grey_levels": float(global_min_amplitude),
        "tile_absolute_floor_grey_levels": float(tile_min_amplitude),
        "global_pass_would_clear_its_own_floor": bool(
            float(global_template_dist) >= float(global_min_amplitude)),
    }


def robust_threshold(signal: np.ndarray, k_mad: float) -> dict:
    med = float(np.median(signal))
    mad = float(np.median(np.abs(signal - med))) * 1.4826
    return {"median": med, "mad_scaled": mad, "threshold": med + k_mad * mad}


def detect_changepoints(signal: np.ndarray, source_frames: list[int], *,
                        k_mad: float, min_amplitude: float,
                        stride: int) -> list[dict]:
    """Threshold-crossing changepoints on one camera's window signal.

    TWO gates, and both must pass. A robust relative gate (median + k*MAD)
    and an ABSOLUTE amplitude floor in grey levels. The relative gate alone
    cannot distinguish a large excursion from a tiny one on a flat signal --
    recorded on this project as a method-level negative against scale-free
    screens -- and the absolute gate alone would be blind to a noisy camera.

    `rise` is a low->high crossing and `fall` a high->low one. Those are
    DIRECTIONS OF THE SIGNAL, not established disappearances or
    reappearances: a rise is consistent with an occluder arriving, with
    content leaving, or with a lighting change, and this instrument cannot
    separate them.
    """
    stats = robust_threshold(signal, k_mad)
    high = signal > stats["threshold"]
    events = []
    for j in range(1, len(signal)):
        if high[j] == high[j - 1]:
            continue
        amplitude = abs(float(signal[j]) - float(signal[j - 1]))
        excess = abs(float(signal[j]) - stats["median"])
        if max(amplitude, excess) < min_amplitude:
            continue
        events.append({
            "polarity": "rise" if high[j] else "fall",
            "source_frame": int(source_frames[j]),
            "bracket_source_frames": [int(source_frames[j - 1]),
                                      int(source_frames[j])],
            "localization_frames": int(stride),
            "proxy_index_in_window": int(j),
            "signal_before": float(signal[j - 1]),
            "signal_after": float(signal[j]),
            "amplitude": amplitude,
            "excess_over_median": excess,
        })
    return events


# ---------------------------------------------------------------------------
# PER-TILE DETECTION (v2) -- the reduction that REPLACES max-over-tiles
# ---------------------------------------------------------------------------

def per_tile_noise_scale(grid: np.ndarray) -> np.ndarray:
    """``1.4826 * MAD_t(S_ij)`` for every tile of one ``(T, ny, nx)`` grid.

    This is the per-tile NOISE SCALE that the v2 absolute floor is declared
    against. It is a property of each tile's OWN temporal series, measured
    before any candidate is read, and it is the quantity the frozen rule takes
    a median over.
    """
    array = np.asarray(grid, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] < 2:
        raise ContractError(
            f"need a (T>=2, ny, nx) tile grid, got shape {tuple(array.shape)}")
    med = np.median(array, axis=0)
    return 1.4826 * np.median(np.abs(array - med[None, :, :]), axis=0)


def per_tile_absolute_floor(scales, floor_f: float) -> dict:
    """The FROZEN v2 absolute floor, measured on the take's own per-tile noise.

    ::

        floor = F * median over all (camera, window, tile) of
                    [ 1.4826 * MAD_t( S_ij ) ]

    ``scales`` is any iterable of arrays of per-tile noise scales -- one array
    per (camera, window) pair -- and the median is taken over ALL of them
    pooled, exactly as the rule says. ``F`` is a DECLARED JUDGMENT, not a
    quantity derived from data, and the returned record says so in a field
    that survives into the manifest.

    A DEGENERATE ZERO floor (every tile series constant) is reported rather
    than refused: it makes the absolute gate inert, and a reader must be able
    to see that from the manifest instead of inferring it.
    """
    if float(floor_f) <= 0:
        raise ContractError(f"floor F must be positive, got {floor_f}")
    arrays = [np.asarray(s, dtype=np.float64).reshape(-1) for s in scales]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        raise ContractError(
            "no per-tile noise scales were supplied, so the absolute floor "
            "cannot be measured. REFUSED -- the v2 floor is defined as a "
            "multiple of the take's OWN noise and may not be guessed")
    flat = np.concatenate(arrays)
    if not np.all(np.isfinite(flat)):
        raise ContractError("per-tile noise scales contain non-finite values")
    median = float(np.median(flat))
    return {
        "F": float(floor_f),
        "F_is_a_declared_judgment_not_derived_from_data": True,
        "rule": ("floor = F * median over all (camera, window, tile) of "
                 "1.4826 * MAD_t(S_ij)"),
        "noise_scale_median_grey_levels": median,
        "floor_grey_levels": float(floor_f) * median,
        "n_tile_series": int(flat.size),
        "n_camera_window_grids": len(arrays),
        "noise_scale_min_grey_levels": float(np.min(flat)),
        "noise_scale_max_grey_levels": float(np.max(flat)),
        "n_zero_scale_tile_series": int(np.count_nonzero(flat == 0.0)),
        "floor_is_degenerate_zero": bool(median == 0.0),
    }


def grid_connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected (FACE-ADJACENT) components of a boolean tile grid.

    Face-adjacent means up/down/left/right ONLY. Two tiles touching at a
    corner are NOT adjacent and do not join a component -- the frozen spec
    says "face-adjacent in the 16x9 grid", and 8-connectivity would let a
    scatter of independent single-tile false positives masquerade as a
    contiguous region, which is the exact discrimination this gate exists to
    make.

    Components are returned in row-major order of their first tile, each
    component sorted, so the output is deterministic.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ContractError(
            f"need a 2-D tile mask, got shape {tuple(array.shape)}")
    n_y, n_x = array.shape
    seen = np.zeros((n_y, n_x), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for i in range(n_y):
        for j in range(n_x):
            if not array[i, j] or seen[i, j]:
                continue
            seen[i, j] = True
            pending = [(i, j)]
            component: list[tuple[int, int]] = []
            while pending:
                row, col = pending.pop()
                component.append((row, col))
                for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r, c = row + d_row, col + d_col
                    if (0 <= r < n_y and 0 <= c < n_x
                            and array[r, c] and not seen[r, c]):
                        seen[r, c] = True
                        pending.append((r, c))
            components.append(sorted(component))
    return components


def per_tile_camera_candidates(
        tiled: dict, source_frames: list[int], *, k_mad: float,
        min_amplitude: float, stride: int,
        min_component_tiles: int = DEFAULT_PER_TILE_MIN_COMPONENT,
        max_firing_tiles: int = DEFAULT_PER_TILE_MAX_FIRING_TILES) -> dict:
    """One camera-window's per-tile candidates. THE GATE FOLLOWS THE TILE.

    ``detect_changepoints`` -- the SAME two-gate detector, unmodified -- is run
    independently on each of the ``ny * nx`` per-tile signals, so the relative
    gate for tile ``(i,j)`` is ``median_t(S_ij) + k_mad*1.4826*MAD_t(S_ij)``
    computed from that tile's OWN temporal series. A quiet tile therefore gets
    a quiet threshold, which is precisely the property the ``tile_max``
    reduction lacked: a maximum over tiles is monopolised by the loudest
    region, which sets both its median and its MAD, and a quiet tile can never
    move a maximum it does not win.

    Tiles firing at the SAME proxy sample are then screened for SPATIAL
    COHERENCE: a face-adjacent connected component of at least
    ``min_component_tiles`` tiles, and strictly fewer than ``max_firing_tiles``
    tiles firing anywhere in the grid at that sample. Both bounds are declared
    judgments (see the module constants), and both are evaluated on the SAME
    firing set, as the frozen text states.

    Every emitted candidate is an EXPLANATION OF THE DETECTOR SIGNAL, NOT AN
    INSTANCE MASK (see ``TILE_EXPLANATION_NOTE``): the tile boxes bound the
    tiles whose scalars crossed, and bound no object.
    """
    grid = np.asarray(tiled["tile_template_dist"], dtype=np.float64)
    if grid.ndim != 3 or grid.shape[0] < 2:
        raise ContractError(
            f"need a (T>=2, ny, nx) tile grid, got shape {tuple(grid.shape)}")
    n_frames, n_y, n_x = (int(v) for v in grid.shape)
    if len(source_frames) != n_frames:
        raise ContractError(
            f"{len(source_frames)} source frames for {n_frames} proxy samples")
    if int(min_component_tiles) < 1:
        raise ContractError(
            f"min component tiles must be >= 1, got {min_component_tiles}")
    if int(max_firing_tiles) < 1:
        raise ContractError(
            f"max firing tiles must be >= 1, got {max_firing_tiles}")

    thresholds: dict[tuple[int, int], dict] = {}
    by_sample: dict[int, dict[tuple[int, int], dict]] = {}
    n_tiles_with_events = 0
    for i in range(n_y):
        for j in range(n_x):
            series = np.ascontiguousarray(grid[:, i, j])
            thresholds[(i, j)] = robust_threshold(series, k_mad)
            events = detect_changepoints(series, source_frames, k_mad=k_mad,
                                         min_amplitude=min_amplitude,
                                         stride=stride)
            if events:
                n_tiles_with_events += 1
            for event in events:
                by_sample.setdefault(
                    int(event["proxy_index_in_window"]), {})[(i, j)] = event

    row_spans, col_spans = tiled["row_spans"], tiled["col_spans"]
    candidates: list[dict] = []
    rejected: list[dict] = []
    for sample in sorted(by_sample):
        firing = by_sample[sample]
        firing_rc = sorted(firing)
        n_firing = len(firing_rc)
        if n_firing >= int(max_firing_tiles):
            #  A global change -- illumination, exposure, auto-white-balance.
            #  Recorded, never silently dropped.
            rejected.append({
                "proxy_index_in_window": int(sample),
                "source_frame": int(source_frames[sample]),
                "reason": "too_many_tiles_firing_global_change",
                "n_tiles_firing_at_sample": n_firing,
                "max_firing_tiles_exclusive": int(max_firing_tiles),
            })
            continue
        mask = np.zeros((n_y, n_x), dtype=bool)
        for row, col in firing_rc:
            mask[row, col] = True
        for component in grid_connected_components(mask):
            if len(component) < int(min_component_tiles):
                rejected.append({
                    "proxy_index_in_window": int(sample),
                    "source_frame": int(source_frames[sample]),
                    "reason": "component_smaller_than_min_component_tiles",
                    "component_size_tiles": len(component),
                    "min_component_tiles": int(min_component_tiles),
                    "component_tiles_row_col": [[int(r), int(c)]
                                                for r, c in component],
                })
                continue
            events = [firing[rc] for rc in component]
            amplitudes = [float(e["amplitude"]) for e in events]
            excesses = [float(e["excess_over_median"]) for e in events]
            polarities = [str(e["polarity"]) for e in events]
            n_rise = polarities.count("rise")
            n_fall = polarities.count("fall")
            #  A component's polarity is its MAJORITY. Polarity is retired as
            #  an ordering constraint by the frozen spec (it records whether
            #  the object is the temporal MAJORITY of its window, not whether
            #  it is present), so the tie-break is deterministic and both the
            #  counts and the tie flag are emitted rather than hidden.
            polarity = "rise" if n_rise >= n_fall else "fall"
            boxes = [tile_pixel_box(r, c, row_spans, col_spans)
                     for r, c in component]
            bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes)]
            candidates.append({
                "polarity": polarity,
                "polarity_is_tied": bool(n_rise == n_fall),
                "n_tiles_rise": n_rise,
                "n_tiles_fall": n_fall,
                "source_frame": int(source_frames[sample]),
                "bracket_source_frames": [int(source_frames[sample - 1]),
                                          int(source_frames[sample])],
                "localization_frames": int(stride),
                "proxy_index_in_window": int(sample),
                #  The scalar the census RANKS on: the mean over the
                #  component's own tiles, in grey levels.
                "amplitude": float(np.mean(amplitudes)),
                "max_tile_amplitude": float(np.max(amplitudes)),
                "excess_over_median": float(np.mean(excesses)),
                "component_size_tiles": len(component),
                "n_tiles_firing_at_sample": n_firing,
                "n_tiles_in_grid": n_y * n_x,
                "component_tiles_row_col": [[int(r), int(c)]
                                            for r, c in component],
                "firing_tiles_row_col": [[int(r), int(c)]
                                         for r, c in firing_rc],
                "per_tile_amplitude": [round(v, 4) for v in amplitudes],
                "per_tile_excess_over_median": [round(v, 4) for v in excesses],
                "per_tile_polarity": polarities,
                "per_tile_signal_before": [round(float(e["signal_before"]), 4)
                                           for e in events],
                "per_tile_signal_after": [round(float(e["signal_after"]), 4)
                                          for e in events],
                "per_tile_threshold": [
                    round(float(thresholds[rc]["threshold"]), 4)
                    for rc in component],
                "per_tile_median": [round(float(thresholds[rc]["median"]), 4)
                                    for rc in component],
                "per_tile_mad_scaled": [
                    round(float(thresholds[rc]["mad_scaled"]), 4)
                    for rc in component],
                "tile_pixel_boxes_xyxy": boxes,
                "component_pixel_bbox_xyxy": bbox,
                "pixel_boxes_are_tile_extent_not_object_extent": True,
                "is_instance_mask": False,
                "kind": "detector_signal_explanation",
                "what_this_is": TILE_EXPLANATION_NOTE,
            })
    return {
        "candidates": candidates,
        "rejected": rejected,
        "per_tile_threshold": thresholds,
        "n_tiles": n_y * n_x,
        "n_tiles_with_events": n_tiles_with_events,
        "n_samples_with_any_tile_firing": len(by_sample),
    }


def cluster_candidates(per_camera: dict[str, list[dict]], polarity: str,
                       tol_frames: int) -> list[dict]:
    """Agglomerate same-polarity candidates across cameras.

    One candidate per camera per cluster (the nearest); the support count is
    therefore a count of DISTINCT CAMERAS, which is the whole point -- a
    single-camera change is an occlusion or a lighting change in that view,
    not scene-level structure.
    """
    flat = []
    for camera, events in per_camera.items():
        for event in events:
            if event["polarity"] == polarity:
                flat.append((event["source_frame"], camera, event))
    flat.sort(key=lambda row: (row[0], row[1]))

    clusters: list[dict] = []
    for frame, camera, event in flat:
        placed = False
        for cluster in clusters:
            if abs(frame - cluster["_anchor"]) <= tol_frames:
                if camera in cluster["members"]:
                    existing = cluster["members"][camera]
                    if abs(frame - cluster["_anchor"]) < abs(
                            existing["source_frame"] - cluster["_anchor"]):
                        cluster["members"][camera] = event
                else:
                    cluster["members"][camera] = event
                placed = True
                break
        if not placed:
            clusters.append({"_anchor": frame, "polarity": polarity,
                             "members": {camera: event}})
    return clusters


def mode_census(args) -> dict:
    proxy_root = Path(args.proxy_root)
    if not proxy_root.is_dir():
        raise ContractError(f"--proxy-root {proxy_root} is not a directory")

    camera_dirs = sorted(p for p in proxy_root.iterdir()
                         if p.is_dir() and (p / PROXY_MANIFEST_NAME).is_file())
    if not camera_dirs:
        raise ContractError(
            f"no per-camera proxy manifests under {proxy_root}; run --mode proxy first")
    if args.limit_cameras is not None:
        camera_dirs = camera_dirs[:int(args.limit_cameras)]

    manifests = {}
    for camera_dir in camera_dirs:
        record = json.loads((camera_dir / PROXY_MANIFEST_NAME)
                            .read_text(encoding="utf-8"))
        manifests[record["camera"]] = record

    strides = {m["mapping"]["stride_frames"] for m in manifests.values()}
    rates = {m["mapping"]["source_rate_exact"] for m in manifests.values()}
    rasters = {tuple(m["proxy_raster"]) for m in manifests.values()}
    if len(strides) != 1 or len(rates) != 1 or len(rasters) != 1:
        raise ContractError(
            f"proxies are heterogeneous: strides {sorted(strides)}, rates "
            f"{sorted(rates)}, rasters {sorted(rasters)}. REFUSED -- a census "
            "across mixed proxies would compare incomparable signals")
    stride = int(next(iter(strides)))
    source_rate = parse_rational(next(iter(rates)))
    proxy_step_ms = frames_to_ms(stride, source_rate)

    tol_frames = (int(args.match_tol_frames)
                  if args.match_tol_frames is not None else stride)
    window = int(args.window_frames)
    if window < 2 * stride:
        raise ContractError(
            f"--window-frames {window} holds fewer than 2 proxy samples at "
            f"stride {stride}; nothing can be measured in it")

    # available source indices, per camera
    available: dict[str, list[int]] = {}
    for camera, record in manifests.items():
        frames_dir = proxy_root / camera / "frames"
        indices = []
        for path in frames_dir.glob(PROXY_FRAME_GLOB):
            match = PROXY_FRAME_RE.match(path.name)
            if match is None:
                raise ContractError(f"unexpected proxy filename {path}")
            indices.append(int(match.group(1)))
        available[camera] = sorted(indices)
        if not indices:
            raise ContractError(f"{camera}: no proxy frames under {frames_dir}")

    global_min = min(v[0] for v in available.values())
    global_max = max(v[-1] for v in available.values())
    window_starts = list(range(global_min, global_max + 1, window))

    tile_mode = bool(getattr(args, "tile_mode", False))
    tile_size = int(getattr(args, "tile_size", DEFAULT_TILE_SIZE))
    tile_floor = float(getattr(args, "tile_min_amplitude",
                               DEFAULT_TILE_MIN_AMPLITUDE))
    tile_top_n = int(getattr(args, "tile_top_n", DEFAULT_TILE_TOP_N))

    per_tile_mode = bool(getattr(args, "per_tile_mode", False))
    per_tile_floor_f = float(getattr(args, "per_tile_floor_f",
                                     DEFAULT_PER_TILE_FLOOR_F))
    per_tile_min_component = int(getattr(args, "per_tile_min_component",
                                         DEFAULT_PER_TILE_MIN_COMPONENT))
    per_tile_max_firing = int(getattr(args, "per_tile_max_firing_tiles",
                                      DEFAULT_PER_TILE_MAX_FIRING_TILES))
    if tile_mode and per_tile_mode:
        raise ContractError(
            "--tile-mode and --per-tile-mode are two DIFFERENT reductions of "
            "the same per-tile grid and may not be combined. --per-tile-mode "
            "supersedes the max-over-tiles reduction of --tile-mode "
            "([[operations/imvid-tile-scout-v2-per-tile-2026-08-25]]); pick "
            "one, and say which one produced any number you report.")

    #  PER-TILE PRE-PASS. The v2 absolute floor is defined over ALL
    #  (camera, window, tile) noise scales, so it cannot be known until every
    #  window's tile grid exists. Rather than decode the whole proxy set
    #  twice, the pre-pass caches the SMALL derived products -- the
    #  (T, ny, nx) tile grid and the per-frame global signals -- and drops the
    #  (T, H, W) stack immediately. For scene6_puppy (39 cameras, ~20 windows
    #  of 30 proxy samples, 144 tiles) that cache is ~28 MiB, against ~1.1 GiB
    #  for a SINGLE window's stack. This pre-pass runs ONLY in per-tile mode;
    #  every other path is exactly what it was.
    per_tile_cache: dict[tuple[str, int], dict] = {}
    per_tile_scales: list[np.ndarray] = []
    per_tile_floor_record: dict | None = None
    if per_tile_mode:
        for w_start in window_starts:
            w_end = min(w_start + window - 1, global_max)
            for camera in sorted(manifests):
                indices = [n for n in available[camera]
                           if w_start <= n <= w_end]
                if len(indices) < 2:
                    continue
                stack = np.stack([
                    png_to_gray(proxy_root / camera / "frames"
                                / f"src_{n:06d}.png") for n in indices])
                signals = frame_signals(stack)
                tiled = tiled_template_signals(stack, tile_size)
                del stack
                per_tile_cache[(camera, w_start)] = {
                    "signals": signals, "tiled": tiled, "indices": indices}
                per_tile_scales.append(
                    per_tile_noise_scale(tiled["tile_template_dist"]))
        per_tile_floor_record = per_tile_absolute_floor(per_tile_scales,
                                                        per_tile_floor_f)
        per_tile_floor_record["is_primary_reading"] = bool(
            float(per_tile_floor_f) == float(DEFAULT_PER_TILE_FLOOR_F))
        print(f"[census] PER-TILE MODE: absolute floor = F * median over "
              f"{per_tile_floor_record['n_tile_series']} (camera, window, "
              f"tile) noise scales = {per_tile_floor_f} * "
              f"{per_tile_floor_record['noise_scale_median_grey_levels']:.6f} "
              f"= {per_tile_floor_record['floor_grey_levels']:.6f} grey levels",
              flush=True)
        if not per_tile_floor_record["is_primary_reading"]:
            print(f"[census] WARNING: F = {per_tile_floor_f} is a SENSITIVITY "
                  f"PROBE, not the census. The primary reading is F = "
                  f"{DEFAULT_PER_TILE_FLOOR_F} only.", flush=True)
        if per_tile_floor_record["floor_is_degenerate_zero"]:
            print("[census] WARNING: the measured noise scale median is ZERO, "
                  "so the absolute gate is INERT on this input.", flush=True)

    windows_out = []
    for w_start in window_starts:
        w_end = min(w_start + window - 1, global_max)
        per_camera_events: dict[str, list[dict]] = {}
        per_camera_signals: dict[str, dict] = {}
        per_camera_tile_explanations: dict[str, list[dict]] = {}
        per_camera_per_tile: dict[str, dict] = {}
        for camera in sorted(manifests):
            indices = [n for n in available[camera] if w_start <= n <= w_end]
            if len(indices) < 2:
                continue
            cached = per_tile_cache.get((camera, w_start))
            if cached is not None:
                #  Identical values to the branch below, computed once in the
                #  pre-pass by the same two functions.
                signals, tiled = cached["signals"], cached["tiled"]
                events = per_tile_camera_candidates(
                    tiled, indices, k_mad=float(args.k_mad),
                    min_amplitude=per_tile_floor_record["floor_grey_levels"],
                    stride=stride,
                    min_component_tiles=per_tile_min_component,
                    max_firing_tiles=per_tile_max_firing)
                per_camera_per_tile[camera] = events
                per_camera_events[camera] = events["candidates"]
                stats = robust_threshold(signals["template_dist"],
                                         float(args.k_mad))
                per_camera_signals[camera] = {
                    "source_frames": indices,
                    "absdiff_mean": [round(float(v), 4)
                                     for v in signals["absdiff_mean"]],
                    "template_dist": [round(float(v), 4)
                                      for v in signals["template_dist"]],
                    "changed_frac": [round(float(v), 5)
                                     for v in signals["changed_frac"]],
                    "threshold": stats,
                    "n_candidates": len(events["candidates"]),
                    "signal_used_for_detection": "per_tile_template_dist",
                    "n_tiles": events["n_tiles"],
                    "n_tiles_with_events": events["n_tiles_with_events"],
                    "n_samples_with_any_tile_firing": events[
                        "n_samples_with_any_tile_firing"],
                    "n_rejected_by_spatial_coherence": len(events["rejected"]),
                    "per_tile_noise_scale_median": float(np.median(
                        per_tile_noise_scale(tiled["tile_template_dist"]))),
                }
                continue
            stack = np.stack([png_to_gray(proxy_root / camera / "frames"
                                          / f"src_{n:06d}.png") for n in indices])
            signals = frame_signals(stack)
            #  TILE MODE screens the per-tile MAXIMUM against the tile floor;
            #  the global signal is still computed above and is reported
            #  side by side so the sensitivity the tile pass bought is
            #  visible rather than asserted.
            tiled = (tiled_template_signals(stack, tile_size)
                     if tile_mode else None)
            detect_on = (tiled["tile_max"] if tile_mode
                         else signals["template_dist"])
            events = detect_changepoints(
                detect_on, indices, k_mad=float(args.k_mad),
                min_amplitude=(tile_floor if tile_mode
                               else float(args.min_amplitude)),
                stride=stride)
            per_camera_events[camera] = events
            stats = robust_threshold(signals["template_dist"], float(args.k_mad))
            per_camera_signals[camera] = {
                "source_frames": indices,
                "absdiff_mean": [round(float(v), 4) for v in signals["absdiff_mean"]],
                "template_dist": [round(float(v), 4) for v in signals["template_dist"]],
                "changed_frac": [round(float(v), 5) for v in signals["changed_frac"]],
                "threshold": stats,
                "n_candidates": len(events),
            }
            if tile_mode:
                explanations = []
                for event in events:
                    j = int(event["proxy_index_in_window"])
                    explanation = tile_candidate_explanation(
                        tiled, j, source_frame=event["source_frame"],
                        top_n=tile_top_n,
                        global_template_dist=float(signals["template_dist"][j]),
                        global_template_dist_before=float(
                            signals["template_dist"][max(j - 1, 0)]),
                        global_min_amplitude=float(args.min_amplitude),
                        tile_min_amplitude=tile_floor)
                    event["tile_argmax_row_col"] = explanation[
                        "tile_argmax_row_col"]
                    event["global_template_dist_at_candidate"] = explanation[
                        "global_template_dist_at_candidate"]
                    explanations.append(explanation)
                per_camera_tile_explanations[camera] = explanations
                per_camera_signals[camera].update({
                    "signal_used_for_detection": "tile_max",
                    "tile_max": [round(float(v), 4) for v in tiled["tile_max"]],
                    "tile_argmax_row_col": [[int(a), int(b)]
                                            for a, b in tiled["tile_argmax"]],
                    "tile_threshold": robust_threshold(tiled["tile_max"],
                                                       float(args.k_mad)),
                })
        if not per_camera_signals:
            continue

        clusters_out = []
        for polarity in ("rise", "fall"):
            for cluster in cluster_candidates(per_camera_events, polarity,
                                              tol_frames):
                members = cluster["members"]
                frames = sorted(e["source_frame"] for e in members.values())
                spread = frames[-1] - frames[0]
                amplitudes = [e["amplitude"] for e in members.values()]
                tile_extra = {}
                if tile_mode:
                    #  Additive, tile-mode only: where on the raster each
                    #  supporting camera's scalar came from, and what the
                    #  whole-frame signal read at the same frame.
                    tile_extra = {
                        "signal_used_for_detection": "tile_max",
                        "per_camera_tile_argmax_row_col": {
                            c: e.get("tile_argmax_row_col")
                            for c, e in sorted(members.items())},
                        "per_camera_global_template_dist": {
                            c: e.get("global_template_dist_at_candidate")
                            for c, e in sorted(members.items())},
                        "tile_explanation_note": TILE_EXPLANATION_NOTE,
                    }
                elif per_tile_mode:
                    #  Additive, per-tile-mode only: WHICH tiles carried each
                    #  supporting camera's crossing, how big the face-adjacent
                    #  component was, and how many tiles fired anywhere in the
                    #  grid at that sample. An EXPLANATION OF THE DETECTOR
                    #  SIGNAL, NOT AN INSTANCE MASK.
                    tile_extra = {
                        "signal_used_for_detection": "per_tile_template_dist",
                        "per_camera_component_size_tiles": {
                            c: e.get("component_size_tiles")
                            for c, e in sorted(members.items())},
                        "per_camera_n_tiles_firing_at_sample": {
                            c: e.get("n_tiles_firing_at_sample")
                            for c, e in sorted(members.items())},
                        "per_camera_component_tiles_row_col": {
                            c: e.get("component_tiles_row_col")
                            for c, e in sorted(members.items())},
                        "per_camera_firing_tiles_row_col": {
                            c: e.get("firing_tiles_row_col")
                            for c, e in sorted(members.items())},
                        "per_camera_per_tile_amplitude": {
                            c: e.get("per_tile_amplitude")
                            for c, e in sorted(members.items())},
                        "per_camera_tile_pixel_boxes_xyxy": {
                            c: e.get("tile_pixel_boxes_xyxy")
                            for c, e in sorted(members.items())},
                        "per_camera_component_pixel_bbox_xyxy": {
                            c: e.get("component_pixel_bbox_xyxy")
                            for c, e in sorted(members.items())},
                        "pixel_boxes_are_tile_extent_not_object_extent": True,
                        "is_instance_mask": False,
                        "tile_explanation_note": TILE_EXPLANATION_NOTE,
                    }
                clusters_out.append({
                    **tile_extra,
                    "polarity": polarity,
                    "n_cameras_supporting": len(members),
                    "cameras": sorted(members),
                    "per_camera_source_frame": {c: e["source_frame"]
                                                for c, e in sorted(members.items())},
                    "source_frame_median": int(np.median(frames)),
                    "source_frame_min": int(frames[0]),
                    "source_frame_max": int(frames[-1]),
                    "spread_frames": int(spread),
                    "spread_ms": frames_to_ms(spread, source_rate),
                    "spread_std_frames": float(np.std(frames)),
                    "mean_amplitude": float(np.mean(amplitudes)),
                    "max_amplitude": float(np.max(amplitudes)),
                    "localization_frames": stride,
                    "localization_ms": proxy_step_ms,
                })
        if per_tile_mode:
            #  RANKING, frozen by the v2 spec §8. Camera count is RETIRED as
            #  the primary order and C_min stays a corroboration floor, never
            #  a sort key: only ~15 of 39 cameras see the subject at all, so
            #  camera count measures how GLOBAL a change is, which is the
            #  opposite of the target property. Ranked by amplitude the two
            #  audited real events sit at 3/76 and 17/76; ranked by camera
            #  support the two promoted clusters were 54/76 and 60/76 -- the
            #  two WEAKEST-amplitude clusters in the set.
            clusters_out.sort(key=lambda c: (-c["mean_amplitude"],
                                             c["source_frame_median"]))
        else:
            clusters_out.sort(key=lambda c: (-c["n_cameras_supporting"],
                                             -c["mean_amplitude"],
                                             c["source_frame_median"]))
        window_out = {
            "window_source_frames": [int(w_start), int(w_end)],
            "n_cameras": len(per_camera_signals),
            "n_candidates_total": sum(len(v) for v in per_camera_events.values()),
            "n_clusters": len(clusters_out),
            "n_clusters_multi_camera": sum(
                1 for c in clusters_out
                if c["n_cameras_supporting"] >= int(args.min_cameras)),
            "candidate_clusters": clusters_out,
            "per_camera_signals": (per_camera_signals if args.emit_signals else None),
        }
        if tile_mode:
            #  The gallery overlay consumes this. It is an EXPLANATION OF THE
            #  DETECTOR SIGNAL, NOT AN INSTANCE MASK.
            window_out["tile_explanations"] = per_camera_tile_explanations
            window_out["tile_explanations_note"] = TILE_EXPLANATION_NOTE
        if per_tile_mode:
            #  Per-camera candidates in full, plus every crossing the spatial
            #  coherence gate REJECTED and why -- a rejection is a measurement
            #  and is never silently dropped.
            window_out["per_tile_candidates"] = {
                camera: entry["candidates"]
                for camera, entry in sorted(per_camera_per_tile.items())}
            window_out["per_tile_rejected_by_spatial_coherence"] = {
                camera: entry["rejected"]
                for camera, entry in sorted(per_camera_per_tile.items())}
            window_out["per_tile_n_tiles_with_events"] = {
                camera: entry["n_tiles_with_events"]
                for camera, entry in sorted(per_camera_per_tile.items())}
            window_out["tile_explanations_note"] = TILE_EXPLANATION_NOTE
        windows_out.append(window_out)

    resolution_note = (
        f"The instrument localizes a candidate to ONE PROXY STEP = {stride} "
        f"source frames = {proxy_step_ms:.2f} ms. ImViD states a "
        f"~{IMVID_SYNC_UNCERTAINTY_MS[0]:.0f}-{IMVID_SYNC_UNCERTAINTY_MS[1]:.0f} ms "
        f"synchronization uncertainty, i.e. "
        f"{IMVID_SYNC_UNCERTAINTY_MS[0] / frames_to_ms(1, source_rate):.2f}-"
        f"{IMVID_SYNC_UNCERTAINTY_MS[1] / frames_to_ms(1, source_rate):.2f} source "
        f"frames. The proxy step is "
        f"{proxy_step_ms / IMVID_SYNC_UNCERTAINTY_MS[1]:.1f}x coarser than the "
        "upper end of that uncertainty, so a cross-camera spread measured here "
        "is DOMINATED BY PROXY SAMPLING and CANNOT test synchronization. The "
        "spread_ms column is reported because it was asked for and because it "
        "bounds candidate agreement; it is not a sync measurement."
    )
    sync_resolvable = proxy_step_ms <= IMVID_SYNC_UNCERTAINTY_MS[1]

    report = {
        "schema": "imvid-event-proxy-census-v1",
        "instrument_status": "SCOUTING INSTRUMENT, NOT GROUND TRUTH",
        "disclaimer": DISCLAIMER,
        "proxy_root": str(proxy_root),
        "n_cameras": len(manifests),
        "cameras": sorted(manifests),
        "mapping": {
            "source_rate_exact": rational_str(source_rate),
            "source_rate_float": float(source_rate),
            "stride_frames": stride,
            "frame_period_ms": frames_to_ms(1, source_rate),
            "proxy_step_ms": proxy_step_ms,
            "proxy_raster": list(next(iter(rasters))),
        },
        "parameters": {
            "window_frames": window,
            "k_mad": float(args.k_mad),
            "min_amplitude_grey_levels": float(args.min_amplitude),
            "match_tol_frames": tol_frames,
            "match_tol_ms": frames_to_ms(tol_frames, source_rate),
            "min_cameras_for_multi": int(args.min_cameras),
            "signal": "template_dist = mean |I_t - per-pixel temporal median|",
            "luma_weights": list(LUMA_WEIGHTS),
        },
        "temporal_resolution_note": resolution_note,
        "sync_uncertainty_resolvable_at_this_proxy_rate": bool(sync_resolvable),
        "imvid_stated_sync_uncertainty_ms": list(IMVID_SYNC_UNCERTAINTY_MS),
        "source_frame_range": [int(global_min), int(global_max)],
        "windows": windows_out,
    }
    if tile_mode:
        #  Additive: present ONLY when --tile-mode is on, so an ordinary
        #  census manifest is byte-for-byte what it was before this pass
        #  existed.
        raster = list(next(iter(rasters)))
        report["tile_pass"] = {
            "enabled": True,
            "note": TILE_EXPLANATION_NOTE,
            "signal": ("tile_template_dist[t,i,j] = mean over tile (i,j) of "
                       "|I_t - per-pixel temporal median|; the detector runs "
                       "on tile_max[t] = max over (i,j)"),
            "why": ("the default signals are WHOLE-FRAME MEANS and are blind "
                    "to small objects; this pass is a high-recall companion, "
                    "not a replacement, and the global signal is reported "
                    "alongside every candidate"),
            "tile_size_px": tile_size,
            "tile_min_amplitude_grey_levels": tile_floor,
            "tile_grid": [
                -(-raster[0] // tile_size), -(-raster[1] // tile_size)],
            "tile_grid_order": "[n_tile_x, n_tile_y]",
            "global_min_amplitude_grey_levels": float(args.min_amplitude),
            "top_tiles_per_candidate": tile_top_n,
            "k_mad": float(args.k_mad),
        }
    if per_tile_mode:
        #  Additive: present ONLY when --per-tile-mode is on, so an ordinary
        #  census manifest is byte-for-byte what it was before this pass
        #  existed.
        raster = list(next(iter(rasters)))
        sweep = per_tile_floor_sweep(
            per_tile_cache, per_tile_scales,
            f_values=PER_TILE_FLOOR_F_SWEEP, k_mad=float(args.k_mad),
            stride=stride, tol_frames=tol_frames,
            min_cameras=int(args.min_cameras),
            min_component_tiles=per_tile_min_component,
            max_firing_tiles=per_tile_max_firing)
        report["per_tile_pass"] = {
            "enabled": True,
            "note": TILE_EXPLANATION_NOTE,
            "is_instance_mask": False,
            "spec": ("research-wiki/operations/"
                     "imvid-tile-scout-v2-per-tile-2026-08-25.md"),
            "supersedes": ("the max-over-tiles (tile_max) DETECTION REDUCTION "
                           "of imvid-tile-scout-freeze-2026-08-25, and only "
                           "that reduction; every measurement that page "
                           "recorded remains correct about what it measured"),
            "signal": ("tile_template_dist[t,i,j] = mean over tile (i,j) of "
                       "|I_t - per-pixel temporal median|; the SAME two-gate "
                       "detector runs INDEPENDENTLY on each of the n_tiles "
                       "per-tile signals"),
            "why": ("tile_max is monopolised by the loudest region, which "
                    "sets both its median AND its MAD, so a quiet tile can "
                    "never move a maximum it does not win; on cam12 the "
                    "resulting threshold EXCEEDED the signal's own maximum in "
                    "every window covering either audited real event, making "
                    "zero candidates a structural certainty"),
            "tile_size_px": tile_size,
            "tile_grid": [
                -(-raster[0] // tile_size), -(-raster[1] // tile_size)],
            "tile_grid_order": "[n_tile_x, n_tile_y]",
            "k_mad": float(args.k_mad),
            "relative_gate": ("per TILE: median_t(S_ij) + k_mad * 1.4826 * "
                              "MAD_t(S_ij), from that tile's own series"),
            "absolute_floor": per_tile_floor_record,
            "spatial_coherence": {
                "min_component_tiles": per_tile_min_component,
                "connectivity": ("4 (face-adjacent); DIAGONAL tiles are NOT "
                                 "adjacent"),
                "max_firing_tiles_exclusive": per_tile_max_firing,
                "both_bounds_are_declared_judgments": True,
                "bounds_derived_from_a_known_positive_and_known_negative": True,
                "disclosure": ("the lower bound admits the smallest observed "
                               "real event (4 tiles) with one tile of margin; "
                               "the upper bound sits below the smallest "
                               "observed artefact (38 tiles). They are NOT "
                               "independent of the ground-truth audit and "
                               "that dependence is disclosed here"),
            },
            "ranking": {
                "order": ("mean_amplitude DESCENDING, then "
                          "source_frame_median ascending"),
                "camera_count_is_retired_as_the_primary_order": True,
                "c_min_is_a_corroboration_floor_never_a_sort_key": True,
                "min_cameras_for_multi": int(args.min_cameras),
            },
            "primary_reading_F": float(DEFAULT_PER_TILE_FLOOR_F),
            "is_primary_reading": per_tile_floor_record["is_primary_reading"],
            "sensitivity_sweep": sweep,
            "sensitivity_sweep_is_supplementary": (
                "THE PRIMARY READING IS F = 3.0 ONLY. Every other point in "
                "this sweep is a sensitivity probe and may not be reported as "
                "the census."),
            "windowing_defect_not_repaired_here": (
                "Windows are non-overlapping and each is templated on its OWN "
                "temporal median. Both audited real absences EXCEED one "
                "window, so their endpoints land in different windows and are "
                "measured against different templates, making their "
                "amplitudes non-comparable; and content present through one "
                "window and absent through the next produces near-zero "
                "residual in BOTH and no changepoint at the boundary. This is "
                "a LIVE DEFECT, deliberately not repaired in v2. NO CLAIM OF "
                "EXHAUSTIVE RECALL MAY BE MADE WHILE IT STANDS."),
        }
    return report


def per_tile_floor_sweep(cache: dict, scales: list, *, f_values, k_mad: float,
                         stride: int, tol_frames: int, min_cameras: int,
                         min_component_tiles: int,
                         max_firing_tiles: int) -> list[dict]:
    """SUPPLEMENTARY sensitivity of the census to the declared floor ``F``.

    THE PRIMARY READING IS ``F = 3.0`` ONLY. Every other point returned here
    is labelled a sensitivity probe and may not be reported as the census.
    The sweep re-runs the whole per-tile detection and cross-camera clustering
    at each ``F`` from the cached tile grids -- no proxy frame is decoded
    twice -- and reports COUNTS only, never a candidate list, so a probe
    cannot be mistaken for a census by copying rows out of it.
    """
    by_window: dict[int, dict[str, dict]] = {}
    for (camera, w_start), entry in cache.items():
        by_window.setdefault(int(w_start), {})[camera] = entry
    out = []
    for floor_f in f_values:
        record = per_tile_absolute_floor(scales, float(floor_f))
        floor = record["floor_grey_levels"]
        n_candidates = n_clusters = n_multi = 0
        n_rejected = 0
        for w_start in sorted(by_window):
            per_camera_events: dict[str, list[dict]] = {}
            for camera, entry in sorted(by_window[w_start].items()):
                result = per_tile_camera_candidates(
                    entry["tiled"], entry["indices"], k_mad=k_mad,
                    min_amplitude=floor, stride=stride,
                    min_component_tiles=min_component_tiles,
                    max_firing_tiles=max_firing_tiles)
                per_camera_events[camera] = result["candidates"]
                n_candidates += len(result["candidates"])
                n_rejected += len(result["rejected"])
            clusters = []
            for polarity in ("rise", "fall"):
                clusters += cluster_candidates(per_camera_events, polarity,
                                               tol_frames)
            n_clusters += len(clusters)
            n_multi += sum(1 for c in clusters
                           if len(c["members"]) >= int(min_cameras))
        is_primary = float(floor_f) == float(DEFAULT_PER_TILE_FLOOR_F)
        out.append({
            "F": float(floor_f),
            "is_primary_reading": is_primary,
            "label": ("PRIMARY READING -- this row IS the census" if is_primary
                      else "SENSITIVITY PROBE -- NOT the census, may not be "
                           "reported as one"),
            "floor_grey_levels": floor,
            "n_per_camera_candidates": n_candidates,
            "n_rejected_by_spatial_coherence": n_rejected,
            "n_clusters": n_clusters,
            "n_clusters_at_min_cameras": n_multi,
            "min_cameras_for_multi": int(min_cameras),
        })
    return out


def print_census(report: dict, top: int) -> None:
    print("", flush=True)
    print("=" * 78, flush=True)
    for line in _wrap(report["disclaimer"], 78):
        print(line, flush=True)
    print("=" * 78, flush=True)
    print(f"[census] {report['n_cameras']} cameras, source frames "
          f"{report['source_frame_range'][0]}..{report['source_frame_range'][1]}, "
          f"proxy step {report['mapping']['stride_frames']} frames "
          f"({report['mapping']['proxy_step_ms']:.2f} ms)", flush=True)
    for line in _wrap(report["temporal_resolution_note"], 78):
        print(f"[census] {line}", flush=True)
    if report.get("tile_pass"):
        tile = report["tile_pass"]
        print(f"[census] TILE MODE: detection on tile_max over "
              f"{tile['tile_grid'][0]}x{tile['tile_grid'][1]} tiles of "
              f"{tile['tile_size_px']} px, absolute floor "
              f"{tile['tile_min_amplitude_grey_levels']} grey levels. The "
              f"global whole-frame signal is reported beside every candidate.",
              flush=True)
        for line in _wrap(tile["note"], 78):
            print(f"[census] {line}", flush=True)
    per_tile = report.get("per_tile_pass")
    if per_tile:
        floor = per_tile["absolute_floor"]
        print(f"[census] PER-TILE MODE: detection runs INDEPENDENTLY on each "
              f"of {per_tile['tile_grid'][0]}x{per_tile['tile_grid'][1]} "
              f"per-tile signals of {per_tile['tile_size_px']} px, each with "
              f"its OWN median+{per_tile['k_mad']}*MAD gate.", flush=True)
        print(f"[census] Absolute floor F={floor['F']} x median noise scale "
              f"{floor['noise_scale_median_grey_levels']:.6f} = "
              f"{floor['floor_grey_levels']:.6f} grey levels over "
              f"{floor['n_tile_series']} (camera, window, tile) series. "
              f"F IS A DECLARED JUDGMENT, not derived from data.", flush=True)
        coherence = per_tile["spatial_coherence"]
        print(f"[census] Spatial coherence: face-adjacent component >= "
              f"{coherence['min_component_tiles']} tiles (4-connectivity; "
              f"diagonal tiles are NOT adjacent) AND < "
              f"{coherence['max_firing_tiles_exclusive']} tiles firing at "
              f"that sample.", flush=True)
        print("[census] RANKED BY MEAN AMPLITUDE. Camera count is retired as "
              "the primary order; C_min is a corroboration floor, never a "
              "sort key.", flush=True)
        print("[census] Sweep over F (SUPPLEMENTARY; the census is F=3.0 "
              "only):", flush=True)
        print(f"[census]   {'F':>5}  {'floor':>10}  {'cands':>7}  "
              f"{'clusters':>8}  {'>=C_min':>7}  reading", flush=True)
        for row in per_tile["sensitivity_sweep"]:
            print(f"[census]   {row['F']:>5.1f}  "
                  f"{row['floor_grey_levels']:>10.6f}  "
                  f"{row['n_per_camera_candidates']:>7}  "
                  f"{row['n_clusters']:>8}  "
                  f"{row['n_clusters_at_min_cameras']:>7}  "
                  f"{'PRIMARY' if row['is_primary_reading'] else 'probe'}",
                  flush=True)
        for line in _wrap(per_tile["windowing_defect_not_repaired_here"], 78):
            print(f"[census] {line}", flush=True)
    for window in report["windows"]:
        lo, hi = window["window_source_frames"]
        print(f"\n[census] window {lo}-{hi}: {window['n_candidates_total']} "
              f"per-camera candidates -> {window['n_clusters']} clusters, "
              f"{window['n_clusters_multi_camera']} multi-camera", flush=True)
        print(f"[census]   {'pol':>4}  {'cams':>4}  {'src_frame':>9}  "
              f"{'spread_f':>8}  {'spread_ms':>9}  {'amp':>7}"
              + (f"  {'comp':>4}  {'fire':>4}" if per_tile else ""), flush=True)
        for cluster in window["candidate_clusters"][:top]:
            suffix = ""
            if per_tile:
                sizes = cluster.get("per_camera_component_size_tiles") or {}
                firing = cluster.get("per_camera_n_tiles_firing_at_sample") or {}
                suffix = (f"  {max(sizes.values()) if sizes else 0:>4}  "
                          f"{max(firing.values()) if firing else 0:>4}")
            print(f"[census]   {cluster['polarity']:>4}  "
                  f"{cluster['n_cameras_supporting']:>4}  "
                  f"{cluster['source_frame_median']:>9}  "
                  f"{cluster['spread_frames']:>8}  "
                  f"{cluster['spread_ms']:>9.1f}  "
                  f"{cluster['mean_amplitude']:>7.3f}{suffix}", flush=True)
    print("\n[census] Ranked CANDIDATES only. Curation against ground truth "
          "decides what, if anything, is an event.", flush=True)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Mode: self-test  (NO media files, NO ffmpeg, runs anywhere)
# ---------------------------------------------------------------------------

def _encode_png(array: np.ndarray, filter_type: int) -> bytes:
    """Minimal PNG encoder used ONLY by the self-test, to prove the reader."""
    height, width, channels = array.shape
    colour = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    bpp, rows = channels, []
    prev = np.zeros(width * channels, dtype=np.uint8)
    for y in range(height):
        raw = array[y].reshape(-1)
        out = np.zeros_like(raw)
        for i in range(len(raw)):
            a = int(raw[i - bpp]) if i >= bpp else 0
            b = int(prev[i])
            c = int(prev[i - bpp]) if i >= bpp else 0
            if filter_type == 0:
                pred = 0
            elif filter_type == 1:
                pred = a
            elif filter_type == 2:
                pred = b
            elif filter_type == 3:
                pred = (a + b) >> 1
            else:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            out[i] = (int(raw[i]) - pred) & 0xFF
        rows.append(bytes([filter_type]) + out.tobytes())
        prev = raw
    ihdr = (width.to_bytes(4, "big") + height.to_bytes(4, "big")
            + bytes([8, colour, 0, 0, 0]))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (len(body).to_bytes(4, "big") + kind + body
                + zlib.crc32(kind + body).to_bytes(4, "big"))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"".join(rows)))
            + chunk(b"IEND", b""))


class _Check:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def eq(self, label: str, got, want) -> None:
        if got == want:
            self.passed += 1
            print(f"  PASS  {label}: {got!r}", flush=True)
        else:
            self.failures.append(f"{label}: got {got!r}, want {want!r}")
            print(f"  FAIL  {label}: got {got!r}, want {want!r}", flush=True)

    def close(self, label: str, got: float, want: float, tol: float) -> None:
        if abs(got - want) <= tol:
            self.passed += 1
            print(f"  PASS  {label}: {got:.9f} (want {want:.9f} +/- {tol:g})",
                  flush=True)
        else:
            self.failures.append(f"{label}: got {got}, want {want} +/- {tol}")
            print(f"  FAIL  {label}: got {got}, want {want} +/- {tol}", flush=True)

    def raises(self, label: str, fn) -> None:
        try:
            fn()
        except ContractError as exc:
            self.passed += 1
            print(f"  PASS  {label}: refused ({str(exc)[:60]}...)", flush=True)
        else:
            self.failures.append(f"{label}: did NOT refuse")
            print(f"  FAIL  {label}: did NOT refuse", flush=True)


def self_test() -> int:
    print("[self-test] imvid_event_proxy -- arithmetic, mapping, PNG reader, "
          "detector. NO media files, NO ffmpeg.", flush=True)
    check = _Check()

    print("\n[1] exact rational rate handling", flush=True)
    imvid = parse_rational("60000/1001")
    check.eq("r_frame_rate 60000/1001 parsed exactly", imvid, Fraction(60000, 1001))
    check.eq("  and is NOT 60", imvid == Fraction(60), False)
    check.close("  float view", float(imvid), 59.94005994005994, 1e-12)
    check.eq("round-trip to string", rational_str(imvid), "60000/1001")
    check.eq("integer rate '30'", parse_rational("30"), Fraction(30))
    check.eq("NTSC 30000/1001", parse_rational("30000/1001"), Fraction(30000, 1001))
    check.raises("'N/A' refused", lambda: parse_rational("N/A"))
    check.raises("'0/0' refused", lambda: parse_rational("0/0"))
    check.raises("'0/1' refused", lambda: parse_rational("0/1"))
    check.raises("garbage refused", lambda: parse_rational("sixty"))

    print("\n[2] stride derivation from the MEASURED rate", flush=True)
    check.eq("59.94 fps -> 2 fps", derive_stride(imvid, Fraction(2)), 30)
    check.eq("59.94 fps -> 1 fps", derive_stride(imvid, Fraction(1)), 60)
    check.eq("59.94 fps -> 4 fps", derive_stride(imvid, Fraction(4)), 15)
    check.eq("59.94 fps -> 1/2 fps", derive_stride(imvid, Fraction(1, 2)), 120)
    check.eq("30 fps -> 2 fps", derive_stride(Fraction(30), Fraction(2)), 15)
    check.eq("requested ABOVE source clamps to 1",
             derive_stride(Fraction(30), Fraction(120)), 1)
    effective = imvid / 30
    check.eq("effective rate is exact", rational_str(effective), "2000/1001")
    check.close("  effective float", float(effective), 1.998001998001998, 1e-12)
    check.close("  relative error vs requested 2 fps",
                float(abs(effective - 2) / 2), 0.0009990009990009991, 1e-12)

    print("\n[3] frame-index mapping (the load-bearing arithmetic)", flush=True)
    check.eq("j=0 -> source 0", proxy_index_to_source(0, 0, 30), 0)
    check.eq("j=1 -> source 30", proxy_index_to_source(1, 0, 30), 30)
    check.eq("j=9 -> source 270", proxy_index_to_source(9, 0, 30), 270)
    check.eq("start offset 7, j=3 -> 97", proxy_index_to_source(3, 7, 30), 97)
    check.eq("inverse of 270", source_to_proxy_index(270, 0, 30), 9)
    check.eq("inverse with offset", source_to_proxy_index(97, 7, 30), 3)
    check.raises("off-lattice source refused",
                 lambda: source_to_proxy_index(271, 0, 30))
    check.raises("pre-start source refused",
                 lambda: source_to_proxy_index(5, 7, 30))
    round_trip = all(source_to_proxy_index(proxy_index_to_source(j, 7, 30), 7, 30) == j
                     for j in range(200))
    check.eq("round trip j -> n -> j for j in 0..199", round_trip, True)

    print("\n[4] expected output count (checked against ffmpeg's real output)",
          flush=True)
    check.eq("0..299 stride 30", expected_proxy_count(0, 299, 30), 10)
    check.eq("0..300 stride 30", expected_proxy_count(0, 300, 30), 11)
    check.eq("0..299 stride 1", expected_proxy_count(0, 299, 1), 300)
    check.eq("0..29 stride 30", expected_proxy_count(0, 29, 30), 1)
    check.eq("0..0 stride 30", expected_proxy_count(0, 0, 30), 1)
    check.eq("empty range", expected_proxy_count(10, 9, 30), 0)
    check.eq("100..299 stride 30", expected_proxy_count(100, 299, 30), 7)
    brute = len([n for n in range(100, 300) if (n - 100) % 30 == 0])
    check.eq("  brute force agrees", brute, 7)
    check.raises("stride 0 refused", lambda: expected_proxy_count(0, 10, 0))

    print("\n[5] frames -> milliseconds at the MEASURED rate", flush=True)
    check.close("1 frame @ 60000/1001", frames_to_ms(1, imvid), 16.68333333, 1e-6)
    check.close("30 frames @ 60000/1001", frames_to_ms(30, imvid), 500.5, 1e-9)
    check.close("300 frames @ 60000/1001", frames_to_ms(300, imvid), 5005.0, 1e-9)
    check.close("1 frame @ 30 (the repo's hard-coded rate)",
                frames_to_ms(1, Fraction(30)), 33.33333333, 1e-6)
    #  NOTE FOR THE RECORD: [[operations/imvid-baseline-freeze]] B4 states the
    #  hard-coded /30 is "wrong by a factor of 2.002". The exact factor is
    #  (1/30) / (1001/60000) = 2000/1001 = 1.998001998..., i.e. 2 x 0.999 and
    #  not 2 x 1.001. The DIRECTION and the ORDER OF MAGNITUDE of that record
    #  are right and nothing downstream changes; the digit is not. Asserted at
    #  the exact value here so this script cannot inherit the slip.
    ratio = frames_to_ms(1, Fraction(30)) / frames_to_ms(1, imvid)
    check.close("  /30 is 2000/1001 = 1.998x wrong for ImViD", ratio,
                2000 / 1001, 1e-12)
    check.close("ImViD 20 ms sync in frames", 20.0 / frames_to_ms(1, imvid),
                20.0 * 60000 / (1001 * 1000), 1e-12)

    print("\n[6] proxy raster from the long-edge target", flush=True)
    check.eq("5312x2988 @ 480", scaled_size(5312, 2988, 480), (480, 270))
    check.eq("5312x2988 @ 320", scaled_size(5312, 2988, 320), (320, 180))
    check.eq("portrait 2988x5312 @ 480", scaled_size(2988, 5312, 480), (270, 480))
    check.eq("square 1000x1000 @ 480", scaled_size(1000, 1000, 480), (480, 480))
    check.raises("absurd long edge refused", lambda: scaled_size(100, 100, 2))
    check.raises("bad raster refused", lambda: scaled_size(0, 100, 480))

    print("\n[7] PNG reader, all five filter types, no image library", flush=True)
    rng = np.random.default_rng(0)
    for channels, name in ((1, "gray"), (3, "rgb"), (4, "rgba")):
        source = rng.integers(0, 256, size=(9, 13, channels), dtype=np.uint8)
        for filter_type in range(5):
            blob = _encode_png(source, filter_type)
            tmp = Path(__file__).resolve().parent / f".selftest_{name}_{filter_type}.png"
            try:
                tmp.write_bytes(blob)
                back = read_png(tmp)
                check.eq(f"{name} filter {filter_type} round trip",
                         bool(np.array_equal(back, source)), True)
            finally:
                tmp.unlink(missing_ok=True)
    grey_source = np.zeros((4, 4, 3), dtype=np.uint8)
    grey_source[..., 0], grey_source[..., 1], grey_source[..., 2] = 10, 20, 30
    tmp = Path(__file__).resolve().parent / ".selftest_luma.png"
    try:
        tmp.write_bytes(_encode_png(grey_source, 0))
        luma = png_to_gray(tmp)
        check.close("Rec.601 luma of (10,20,30)", float(luma[0, 0]),
                    0.299 * 10 + 0.587 * 20 + 0.114 * 30, 1e-4)
    finally:
        tmp.unlink(missing_ok=True)

    print("\n[8] window signals and the changepoint detector", flush=True)
    #  a flat window with a raised plateau over proxy samples 3..6
    stack = np.full((10, 16, 16), 40.0, dtype=np.float32)
    stack[3:7] += 60.0
    signals = frame_signals(stack)
    check.eq("template_dist has one entry per frame",
             len(signals["template_dist"]), 10)
    check.close("template_dist is flat off-plateau",
                float(signals["template_dist"][0]), 0.0, 1e-6)
    check.close("template_dist is raised on-plateau",
                float(signals["template_dist"][4]), 60.0, 1e-6)
    source_frames = [n * 30 for n in range(10)]
    events = detect_changepoints(signals["template_dist"], source_frames,
                                 k_mad=3.0, min_amplitude=2.0, stride=30)
    check.eq("two candidates found", len(events), 2)
    check.eq("first is a rise at source frame 90",
             (events[0]["polarity"], events[0]["source_frame"]), ("rise", 90))
    check.eq("second is a fall at source frame 210",
             (events[1]["polarity"], events[1]["source_frame"]), ("fall", 210))
    check.eq("rise bracket names both source frames",
             events[0]["bracket_source_frames"], [60, 90])
    check.eq("localization is one proxy step", events[0]["localization_frames"], 30)
    tiny = np.full((10, 8, 8), 40.0, dtype=np.float32)
    tiny[3:7] += 0.5
    tiny_events = detect_changepoints(frame_signals(tiny)["template_dist"],
                                      source_frames, k_mad=3.0,
                                      min_amplitude=2.0, stride=30)
    check.eq("sub-floor excursion rejected by the ABSOLUTE gate",
             len(tiny_events), 0)
    check.raises("single-frame window refused",
                 lambda: frame_signals(np.zeros((1, 4, 4), dtype=np.float32)))

    print("\n[9] cross-camera support counting and spread", flush=True)
    jitter = {"cam00": 90, "cam01": 90, "cam02": 120, "cam03": 900}
    per_camera = {
        cam: [{"polarity": "rise", "source_frame": frame, "amplitude": 5.0}]
        for cam, frame in jitter.items()
    }
    clusters = cluster_candidates(per_camera, "rise", tol_frames=30)
    sizes = sorted(len(c["members"]) for c in clusters)
    check.eq("one 3-camera cluster and one singleton", sizes, [1, 3])
    big = max(clusters, key=lambda c: len(c["members"]))
    frames = sorted(e["source_frame"] for e in big["members"].values())
    check.eq("supporting cameras", sorted(big["members"]),
             ["cam00", "cam01", "cam02"])
    check.eq("spread in frames", frames[-1] - frames[0], 30)
    check.close("spread in ms at 60000/1001",
                frames_to_ms(frames[-1] - frames[0], imvid), 500.5, 1e-9)
    check.eq("the far camera is NOT absorbed", len(big["members"]) == 4, False)
    solo = cluster_candidates(
        {"cam00": [{"polarity": "rise", "source_frame": 90, "amplitude": 5.0}]},
        "rise", tol_frames=30)
    check.eq("a single-camera change stays a 1-camera cluster",
             len(solo[0]["members"]), 1)
    check.eq("polarity filter excludes falls",
             len(cluster_candidates(per_camera, "fall", 30)), 0)

    print("\n[10] the sync-uncertainty comparison is honest about itself",
          flush=True)
    step_ms = frames_to_ms(30, imvid)
    check.close("proxy step at 2 fps", step_ms, 500.5, 1e-9)
    check.eq("2 fps CANNOT resolve a 20 ms sync uncertainty",
             step_ms <= IMVID_SYNC_UNCERTAINTY_MS[1], False)
    native_step = frames_to_ms(derive_stride(imvid, imvid), imvid)
    check.eq("only a native-rate proxy could",
             native_step <= IMVID_SYNC_UNCERTAINTY_MS[1], True)

    print("\n[11] output-root refusals", flush=True)
    check.raises("repository root refused",
                 lambda: _check_output_root(REPO_ROOT / "derived", None))
    check.raises("a 'raw' path component refused",
                 lambda: _check_output_root(
                     Path("/apollo/users/sri/proj_adags/data/imvid/raw/x"), None))
    check.raises("inside the source dir refused",
                 lambda: _check_output_root(
                     Path("/data/scene1_opera/proxy"), Path("/data/scene1_opera")))

    print("\n[12] heterogeneous-scene refusal names the dissenters", flush=True)
    probes = {"cam00": {"width": 5312}, "cam01": {"width": 5312},
              "cam02": {"width": 1920}}
    try:
        _agree_or_refuse(probes, "width", "raster width")
    except ContractError as exc:
        message = str(exc)
        check.eq("names the dissenting camera", "cam02" in message, True)
        check.eq("names the majority value", "5312" in message, True)
    else:
        check.eq("heterogeneous width refused", False, True)
    _agree_or_refuse({"cam00": {"width": 5312}, "cam01": {"width": 5312}},
                     "width", "raster width")
    check.eq("homogeneous scene accepted", True, True)

    print("\n" + "=" * 78, flush=True)
    if check.failures:
        print(f"[self-test] FAILED: {len(check.failures)} of "
              f"{check.passed + len(check.failures)} checks", flush=True)
        for failure in check.failures:
            print(f"  - {failure}", flush=True)
        return 1
    print(f"[self-test] PASSED: {check.passed}/{check.passed} checks", flush=True)
    for line in _wrap(DISCLAIMER, 78):
        print(f"[self-test] {line}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Mode: tile-selftest  -- the FROZEN PRECONDITIONS of the tile pass
# ---------------------------------------------------------------------------

#: Declared detection scale for P1. Stated BEFORE any score is read, and
#: about the SETUP only: a square patch of this many proxy pixels at this
#: grey-level contrast is what the tile pass claims to be able to see and the
#: global pass is expected to miss.
TILE_P1_PATCH_PX = 32
TILE_P1_CONTRAST = 25.0
TILE_P1_RASTER = (540, 960)          # (height, width) -- the census raster
TILE_P1_FRAMES = 12
TILE_P1_PLATEAU = (4, 8)             # half-open run of frames carrying the patch
TILE_P1_TILE_ROW_COL = (2, 5)        # tile fully containing the patch


def _tile_p1_fixture() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Flat window with a small patch present over a run of frames.

    Deliberately built so the patch lies WHOLLY INSIDE one 60 px tile: a
    patch straddling a tile boundary is a different (harder) question and is
    not what P1 declares.
    """
    height, width = TILE_P1_RASTER
    stack = np.full((TILE_P1_FRAMES, height, width), 40.0, dtype=np.float32)
    row, col = TILE_P1_TILE_ROW_COL
    y0 = row * DEFAULT_TILE_SIZE + (DEFAULT_TILE_SIZE - TILE_P1_PATCH_PX) // 2
    x0 = col * DEFAULT_TILE_SIZE + (DEFAULT_TILE_SIZE - TILE_P1_PATCH_PX) // 2
    y1, x1 = y0 + TILE_P1_PATCH_PX, x0 + TILE_P1_PATCH_PX
    lo, hi = TILE_P1_PLATEAU
    stack[lo:hi, y0:y1, x0:x1] += TILE_P1_CONTRAST
    return stack, (y0, y1, x0, x1)


def tile_self_test() -> int:
    """FROZEN PRECONDITIONS for the tile pass. Every one is a statement about
    the SETUP -- what was injected, what the tiling covers, what is finite --
    and none of them reads a census score. The project's standing rule is
    that freezing a READING RULE is not enough: a frozen rule needs a frozen
    PRECONDITION asserting the mechanism it reads was actually exercised
    ([[operations/block-2026-08-24-handover]]). This is that precondition set
    for the tile pass, and it fails loudly.
    """
    print("[tile-selftest] imvid_event_proxy TILE PASS -- frozen "
          "preconditions P1-P5. Statements about the SETUP, never about a "
          "census score. NO media files, NO ffmpeg, NO GPU.", flush=True)
    check = _Check()
    source_frames = [n * 30 for n in range(TILE_P1_FRAMES)]

    print("\n[P1] detection at a DECLARED SCALE: a "
          f"{TILE_P1_PATCH_PX}x{TILE_P1_PATCH_PX} px patch at "
          f"{TILE_P1_CONTRAST:.0f} grey levels on a "
          f"{TILE_P1_RASTER[1]}x{TILE_P1_RASTER[0]} raster", flush=True)
    stack, (py0, py1, px0, px1) = _tile_p1_fixture()
    plateau_index = TILE_P1_PLATEAU[0] + 1
    global_signals = frame_signals(stack)
    tiled = tiled_template_signals(stack, DEFAULT_TILE_SIZE)
    global_value = float(global_signals["template_dist"][plateau_index])
    tile_value = float(tiled["tile_max"][plateau_index])
    patch_area = TILE_P1_PATCH_PX ** 2
    frame_area = TILE_P1_RASTER[0] * TILE_P1_RASTER[1]
    tile_area = DEFAULT_TILE_SIZE ** 2
    check.close("  GLOBAL template_dist on the patch frame (grey levels)",
                global_value, TILE_P1_CONTRAST * patch_area / frame_area, 1e-6)
    check.close("  TILE   tile_max     on the patch frame (grey levels)",
                tile_value, TILE_P1_CONTRAST * patch_area / tile_area, 1e-6)
    #  Tolerance 1e-3 on a ratio of 144: ``frame_signals`` accumulates its
    #  whole-frame mean in float32 over 518,400 pixels, which is where the
    #  ~1e-6 absolute departure comes from. The tile path accumulates in
    #  float64 and lands on the closed form exactly.
    check.close("  tile / global sensitivity ratio", tile_value / global_value,
                frame_area / tile_area, 1e-3)
    global_events = detect_changepoints(
        global_signals["template_dist"], source_frames, k_mad=DEFAULT_K_MAD,
        min_amplitude=DEFAULT_MIN_AMPLITUDE, stride=30)
    tile_events = detect_changepoints(
        tiled["tile_max"], source_frames, k_mad=DEFAULT_K_MAD,
        min_amplitude=DEFAULT_TILE_MIN_AMPLITUDE, stride=30)
    print(f"  ---- P1 NUMBERS: global {global_value:.4f} vs floor "
          f"{DEFAULT_MIN_AMPLITUDE} -> {len(global_events)} candidates; "
          f"tile {tile_value:.4f} vs floor {DEFAULT_TILE_MIN_AMPLITUDE} -> "
          f"{len(tile_events)} candidates "
          f"({tile_value / global_value:.1f}x more signal)", flush=True)
    check.eq("  the GLOBAL pass MISSES it (this is the blindness on record)",
             len(global_events), 0)
    check.eq("  the GLOBAL value is below its own absolute floor",
             global_value < DEFAULT_MIN_AMPLITUDE, True)
    check.eq("  the TILE pass DETECTS it (rise and fall)", len(tile_events), 2)
    check.eq("  rise then fall", [e["polarity"] for e in tile_events],
             ["rise", "fall"])
    check.eq("  tile argmax names the tile the patch was injected into",
             [int(v) for v in tiled["tile_argmax"][plateau_index]],
             list(TILE_P1_TILE_ROW_COL))
    box = tile_pixel_box(TILE_P1_TILE_ROW_COL[0], TILE_P1_TILE_ROW_COL[1],
                         tiled["row_spans"], tiled["col_spans"])
    check.eq("  its pixel box contains the injected patch",
             (box[0] <= px0 and box[1] <= py0
              and box[2] >= px1 and box[3] >= py1), True)
    check.eq("  the grid is 16x9 = 144 tiles at 960x540",
             (tiled["n_tile_x"], tiled["n_tile_y"],
              tiled["n_tile_x"] * tiled["n_tile_y"]), (16, 9, 144))

    print("\n[P2] flat control: nothing injected must yield NO candidates",
          flush=True)
    flat = np.full((TILE_P1_FRAMES, 120, 180), 40.0, dtype=np.float32)
    flat_tiled = tiled_template_signals(flat, DEFAULT_TILE_SIZE)
    flat_events = detect_changepoints(
        flat_tiled["tile_max"], source_frames, k_mad=DEFAULT_K_MAD,
        min_amplitude=DEFAULT_TILE_MIN_AMPLITUDE, stride=30)
    check.close("  a constant window has zero tile deviation everywhere",
                float(np.max(flat_tiled["tile_template_dist"])), 0.0, 1e-12)
    check.eq("  constant window -> ZERO accepted candidates",
             len(flat_events), 0)
    rng = np.random.default_rng(20260825)
    noisy = (np.full((TILE_P1_FRAMES, 120, 180), 40.0, dtype=np.float32)
             + rng.uniform(-0.5, 0.5,
                           size=(TILE_P1_FRAMES, 120, 180)).astype(np.float32))
    noisy_tiled = tiled_template_signals(noisy, DEFAULT_TILE_SIZE)
    noisy_events = detect_changepoints(
        noisy_tiled["tile_max"], source_frames, k_mad=DEFAULT_K_MAD,
        min_amplitude=DEFAULT_TILE_MIN_AMPLITUDE, stride=30)
    check.eq("  sub-floor noise stays below the ABSOLUTE floor",
             float(np.max(noisy_tiled["tile_max"])) < DEFAULT_TILE_MIN_AMPLITUDE,
             True)
    check.eq("  sub-floor noise -> ZERO accepted candidates",
             len(noisy_events), 0)

    print("\n[P3] every tile statistic is FINITE, and non-constant where "
          "change exists", flush=True)
    check.eq("  P1 grid all finite",
             bool(np.all(np.isfinite(tiled["tile_template_dist"]))), True)
    check.eq("  noise grid all finite",
             bool(np.all(np.isfinite(noisy_tiled["tile_template_dist"]))), True)
    check.eq("  P1 tile_max is NOT constant (the mechanism was exercised)",
             float(np.std(tiled["tile_max"])) > 0.0, True)
    check.eq("  P1 tile_max is zero on frames with no patch",
             float(tiled["tile_max"][0]), 0.0)
    check.eq("  argmax is a valid grid coordinate on every frame",
             bool(np.all(tiled["tile_argmax"][:, 0] < tiled["n_tile_y"])
                  and np.all(tiled["tile_argmax"][:, 1] < tiled["n_tile_x"])),
             True)

    print("\n[P4] the tiling covers every pixel EXACTLY ONCE, edge tiles "
          "included", flush=True)
    for height, width in ((540, 960), (100, 130), (61, 59), (60, 60), (1, 1)):
        rows = tile_edges(height, DEFAULT_TILE_SIZE)
        cols = tile_edges(width, DEFAULT_TILE_SIZE)
        cover = np.zeros((height, width), dtype=np.int32)
        for y0, y1 in rows:
            for x0, x1 in cols:
                cover[y0:y1, x0:x1] += 1
        check.eq(f"  {width}x{height}: {len(cols)}x{len(rows)} tiles, every "
                 "pixel covered exactly once",
                 (int(cover.min()), int(cover.max()),
                  sum(y1 - y0 for y0, y1 in rows),
                  sum(x1 - x0 for x0, x1 in cols)),
                 (1, 1, height, width))
    check.eq("  960 divides into 16 whole tiles",
             tile_edges(960, DEFAULT_TILE_SIZE)[-1], (900, 960))
    check.eq("  130 leaves a PARTIAL edge tile of 10 px, not dropped",
             tile_edges(130, DEFAULT_TILE_SIZE)[-1], (120, 130))
    check.raises("  tile size 0 refused", lambda: tile_edges(100, 0))
    check.raises("  empty raster refused", lambda: tile_edges(0, 60))

    print("\n[P5] NEUTER CHECK: the tile MAXIMUM is load-bearing", flush=True)
    grid = tiled["tile_template_dist"]
    neutered = grid.mean(axis=(1, 2))
    neutered_events = detect_changepoints(
        neutered, source_frames, k_mad=DEFAULT_K_MAD,
        min_amplitude=DEFAULT_TILE_MIN_AMPLITUDE, stride=30)
    check.eq("  tile_max IS the max over tiles, exactly",
             bool(np.array_equal(tiled["tile_max"], grid.max(axis=(1, 2)))),
             True)
    check.eq("  tile_max is NOT the mean over tiles",
             bool(np.allclose(tiled["tile_max"], neutered)), False)
    check.close("  a tile MEAN is the whole-frame mean again (the blindness)",
                float(neutered[plateau_index]), global_value, 1e-9)
    check.eq("  replacing max by mean DESTROYS the P1 detection",
             len(neutered_events), 0)
    check.eq("  ... while the real reduction keeps it", len(tile_events), 2)

    print("\n" + "=" * 78, flush=True)
    if check.failures:
        print(f"[tile-selftest] FAILED: {len(check.failures)} of "
              f"{check.passed + len(check.failures)} preconditions", flush=True)
        for failure in check.failures:
            print(f"  - {failure}", flush=True)
        return 1
    print(f"[tile-selftest] PASSED: {check.passed}/{check.passed} "
          "preconditions", flush=True)
    for line in _wrap(TILE_EXPLANATION_NOTE, 78):
        print(f"[tile-selftest] {line}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Mode: per-tile-selftest -- the FROZEN PRECONDITIONS of the v2 per-tile pass
# ---------------------------------------------------------------------------

#: P1-REL fixture. Declared here, BEFORE any score is read, and every value is
#: a statement about the SETUP.
#:
#: v1's P1 fixture was a CONSTANT background plus a patch, so its ``tile_max``
#: series had median 0 and MAD 0 and the relative threshold was EXACTLY
#: 0.0000. That precondition proved the max-over-tiles MECHANISM was
#: exercised; it never exercised THE GATE THAT ACTUALLY DECIDES ON REAL
#: FOOTAGE. This fixture is NON-CONSTANT by construction: every tile has a
#: strictly positive temporal MAD, one region is loud, and the injected target
#: is quiet.
PER_TILE_P1REL_RASTER = (540, 960)       # (height, width) -- the census raster
PER_TILE_P1REL_FRAMES = 24
PER_TILE_P1REL_BASE = 40.0
PER_TILE_P1REL_BG_NOISE = 1.5            # +/- grey levels, per pixel per frame
PER_TILE_P1REL_SEED = 20260825

#: THE LOUD DISTRACTOR -- a 2x2 block of tiles carrying a full-span excursion,
#: standing in for the audited "person's legs/torso at 40-83 grey levels
#: continuously" that monopolised ``tile_max`` on the real footage. The
#: excursion is a SYMMETRIC RAMP, which makes the failure structural rather
#: than a lucky draw: for a symmetric-uniform excursion the deviation series
#: has median A/2 and MAD A/4, so its own median + 3*1.4826*MAD = 1.61*A
#: EXCEEDS its own maximum A. That is exactly the audited cam12 signature
#: (threshold 73.99 against a signal maximum of 72.19).
PER_TILE_P1REL_DISTRACTOR_TILES = ((6, 1), (6, 2), (7, 1), (7, 2))
PER_TILE_P1REL_DISTRACTOR_SPAN = 43.0

#: THE QUIET TARGET -- three FACE-ADJACENT tiles, each carrying a 32x32 px
#: patch at 25 grey levels (the v1 declared detection scale). Three tiles so
#: the frozen spatial-coherence gate (component >= 3) can admit it; quiet so
#: it can never win a maximum against the distractor.
PER_TILE_P1REL_TARGET_TILES = ((2, 11), (2, 12), (2, 13))
PER_TILE_P1REL_TARGET_PATCH = 32
PER_TILE_P1REL_TARGET_CONTRAST = 25.0
PER_TILE_P1REL_TARGET_PLATEAU = (10, 14)

#: P3 sub-floor control. Most tiles are LOUD and a few are very QUIET, so the
#: pooled median noise scale -- and therefore the absolute floor -- is set by
#: the loud tiles. A tiny excursion is then injected into three adjacent QUIET
#: tiles: it clears their own (tiny) relative gate and it clears the coherence
#: gate, so the ABSOLUTE FLOOR is the only thing that can reject it. Without
#: this construction a "sub-floor control" would be rejected by the relative
#: gate first and would say nothing about the floor at all.
PER_TILE_P3_RASTER = (300, 480)
PER_TILE_P3_FRAMES = 20
PER_TILE_P3_LOUD_NOISE = 12.0
PER_TILE_P3_QUIET_NOISE = 0.2
PER_TILE_P3_QUIET_TILES = ((0, 0), (0, 1), (0, 2))
PER_TILE_P3_SUBFLOOR_CONTRAST = 0.15
PER_TILE_P3_PLATEAU = (8, 12)


def _per_tile_p1rel_fixture(with_distractor: bool = True) -> np.ndarray:
    """The P1-REL window. Identical seed both ways, so the two variants are
    BIT-IDENTICAL outside the distractor tiles -- which is what lets the test
    assert that the distractor does not move the target tile's threshold at
    all, rather than merely that it moves it a little.
    """
    height, width = PER_TILE_P1REL_RASTER
    n_frames = PER_TILE_P1REL_FRAMES
    rng = np.random.default_rng(PER_TILE_P1REL_SEED)
    noise = rng.random((n_frames, height, width), dtype=np.float32)
    stack = (np.float32(PER_TILE_P1REL_BASE)
             + (noise * np.float32(2.0) - np.float32(1.0))
             * np.float32(PER_TILE_P1REL_BG_NOISE))
    if with_distractor:
        for t in range(n_frames):
            offset = np.float32(PER_TILE_P1REL_DISTRACTOR_SPAN * t
                                / (n_frames - 1))
            for row, col in PER_TILE_P1REL_DISTRACTOR_TILES:
                y0, x0 = row * DEFAULT_TILE_SIZE, col * DEFAULT_TILE_SIZE
                stack[t, y0:y0 + DEFAULT_TILE_SIZE,
                      x0:x0 + DEFAULT_TILE_SIZE] += offset
    lo, hi = PER_TILE_P1REL_TARGET_PLATEAU
    pad = (DEFAULT_TILE_SIZE - PER_TILE_P1REL_TARGET_PATCH) // 2
    for row, col in PER_TILE_P1REL_TARGET_TILES:
        y0 = row * DEFAULT_TILE_SIZE + pad
        x0 = col * DEFAULT_TILE_SIZE + pad
        stack[lo:hi, y0:y0 + PER_TILE_P1REL_TARGET_PATCH,
              x0:x0 + PER_TILE_P1REL_TARGET_PATCH] += np.float32(
                  PER_TILE_P1REL_TARGET_CONTRAST)
    return stack


def _per_tile_p3_fixture(contrast: float) -> np.ndarray:
    """Loud background, three quiet tiles, a sub-floor excursion in them."""
    height, width = PER_TILE_P3_RASTER
    n_frames = PER_TILE_P3_FRAMES
    rng = np.random.default_rng(PER_TILE_P1REL_SEED + 1)
    noise = rng.random((n_frames, height, width), dtype=np.float32) * 2.0 - 1.0
    stack = np.float32(PER_TILE_P1REL_BASE) + noise * np.float32(
        PER_TILE_P3_LOUD_NOISE)
    for row, col in PER_TILE_P3_QUIET_TILES:
        y0, x0 = row * DEFAULT_TILE_SIZE, col * DEFAULT_TILE_SIZE
        block = slice(y0, y0 + DEFAULT_TILE_SIZE), slice(
            x0, x0 + DEFAULT_TILE_SIZE)
        stack[:, block[0], block[1]] = (
            np.float32(PER_TILE_P1REL_BASE)
            + noise[:, block[0], block[1]] * np.float32(PER_TILE_P3_QUIET_NOISE))
    lo, hi = PER_TILE_P3_PLATEAU
    for row, col in PER_TILE_P3_QUIET_TILES:
        y0, x0 = row * DEFAULT_TILE_SIZE, col * DEFAULT_TILE_SIZE
        stack[lo:hi, y0:y0 + DEFAULT_TILE_SIZE,
              x0:x0 + DEFAULT_TILE_SIZE] += np.float32(contrast)
    return stack


def per_tile_self_test() -> int:
    """FROZEN PRECONDITIONS for the v2 PER-TILE pass.

    P1-REL is the one v1 lacked and it is the reason this pass exists. The
    project's standing rule -- *"every frozen rule needs a frozen precondition
    asserting the mechanism it reads was actually exercised, stated about the
    setup and never about the score"* -- was applied by v1 to the wrong
    mechanism: its fixture had median 0 and MAD 0, so the RELATIVE GATE, which
    is what actually decides on real footage, was never exercised at all.
    """
    print("[per-tile-selftest] imvid_event_proxy PER-TILE PASS (v2) -- frozen "
          "preconditions P1-REL, P3, P4, P5. Statements about the SETUP, "
          "never about a census score. NO media files, NO ffmpeg, NO GPU.",
          flush=True)
    check = _Check()
    stride = 30
    source_frames = [n * stride for n in range(PER_TILE_P1REL_FRAMES)]

    print("\n[P1-REL] THE RELATIVE GATE FOLLOWS THE TILE, on a NON-CONSTANT "
          "fixture", flush=True)
    with_distractor = _per_tile_p1rel_fixture(True)
    without_distractor = _per_tile_p1rel_fixture(False)
    tiled = tiled_template_signals(with_distractor, DEFAULT_TILE_SIZE)
    tiled_clean = tiled_template_signals(without_distractor, DEFAULT_TILE_SIZE)
    scales = per_tile_noise_scale(tiled["tile_template_dist"])

    #  (0) the fixture is genuinely non-constant -- the defect v1 shipped.
    check.eq("  every tile has a STRICTLY POSITIVE temporal MAD "
             "(v1's fixture had MAD 0 everywhere)",
             bool(np.all(scales > 0.0)), True)
    check.eq("  the grid is 16x9 = 144 tiles at 960x540",
             (tiled["n_tile_x"], tiled["n_tile_y"],
              tiled["n_tile_x"] * tiled["n_tile_y"]), (16, 9, 144))

    #  (a) the distractor does NOT raise the target tile's own threshold.
    target_rc = PER_TILE_P1REL_TARGET_TILES[1]
    distractor_rc = PER_TILE_P1REL_DISTRACTOR_TILES[0]
    target_series = tiled["tile_template_dist"][:, target_rc[0], target_rc[1]]
    target_series_clean = tiled_clean["tile_template_dist"][
        :, target_rc[0], target_rc[1]]
    target_gate = robust_threshold(target_series, DEFAULT_K_MAD)
    target_gate_clean = robust_threshold(target_series_clean, DEFAULT_K_MAD)
    distractor_gate = robust_threshold(
        tiled["tile_template_dist"][:, distractor_rc[0], distractor_rc[1]],
        DEFAULT_K_MAD)
    max_gate = robust_threshold(tiled["tile_max"], DEFAULT_K_MAD)
    max_gate_clean = robust_threshold(tiled_clean["tile_max"], DEFAULT_K_MAD)
    check.close(f"  target tile {list(target_rc)} threshold is IDENTICAL with "
                "and without the loud distractor",
                target_gate["threshold"], target_gate_clean["threshold"], 1e-12)
    check.eq("  the loud distractor's OWN threshold is far higher than the "
             "target tile's",
             distractor_gate["threshold"] > 10 * target_gate["threshold"], True)
    check.eq("  the MAX-over-tiles threshold IS raised by the distractor",
             max_gate["threshold"] > 10 * max_gate_clean["threshold"], True)

    #  (b) the max reduction FAILS on this fixture; the per-tile one succeeds.
    floor_record = per_tile_absolute_floor([scales], DEFAULT_PER_TILE_FLOOR_F)
    floor = floor_record["floor_grey_levels"]
    max_events = detect_changepoints(
        tiled["tile_max"], source_frames, k_mad=DEFAULT_K_MAD,
        min_amplitude=floor, stride=stride)
    per_tile = per_tile_camera_candidates(
        tiled, source_frames, k_mad=DEFAULT_K_MAD, min_amplitude=floor,
        stride=stride,
        min_component_tiles=DEFAULT_PER_TILE_MIN_COMPONENT,
        max_firing_tiles=DEFAULT_PER_TILE_MAX_FIRING_TILES)
    candidates = per_tile["candidates"]
    target_set = {tuple(rc) for rc in PER_TILE_P1REL_TARGET_TILES}
    on_target = [c for c in candidates
                 if target_set.issubset({tuple(rc) for rc
                                         in c["component_tiles_row_col"]})]
    print(f"  ---- P1-REL NUMBERS ----", flush=True)
    print(f"       MAX-over-tiles reduction : threshold "
          f"{max_gate['threshold']:.4f} vs signal maximum "
          f"{float(np.max(tiled['tile_max'])):.4f} "
          f"-> {len(max_events)} candidates "
          f"(THRESHOLD EXCEEDS THE SIGNAL'S OWN MAXIMUM: "
          f"{max_gate['threshold'] > float(np.max(tiled['tile_max']))})",
          flush=True)
    print(f"       PER-TILE  reduction      : target tile {list(target_rc)} "
          f"threshold {target_gate['threshold']:.4f} vs its own plateau value "
          f"{float(np.max(target_series)):.4f} -> {len(candidates)} accepted "
          f"candidates, {len(on_target)} of them on the injected target",
          flush=True)
    print(f"       distractor tile {list(distractor_rc)} threshold "
          f"{distractor_gate['threshold']:.4f} "
          f"({distractor_gate['threshold'] / target_gate['threshold']:.1f}x "
          f"the target tile's) -- the loud region gets a loud gate, and it "
          f"only gets its own", flush=True)
    print(f"       absolute floor F={DEFAULT_PER_TILE_FLOOR_F} x median noise "
          f"scale {floor_record['noise_scale_median_grey_levels']:.6f} = "
          f"{floor:.6f} grey levels", flush=True)
    check.eq("  the MAX reduction's threshold EXCEEDS the signal's own "
             "maximum (the audited cam12 signature)",
             max_gate["threshold"] > float(np.max(tiled["tile_max"])), True)
    check.eq("  so the MAX reduction FINDS NOTHING on this fixture",
             len(max_events), 0)
    check.eq("  the PER-TILE reduction finds the target", len(on_target), 2)
    check.eq("  ... as a rise then a fall",
             [c["polarity"] for c in on_target], ["rise", "fall"])
    check.eq("  ... at the injected plateau edges",
             [c["source_frame"] for c in on_target],
             [PER_TILE_P1REL_TARGET_PLATEAU[0] * stride,
              PER_TILE_P1REL_TARGET_PLATEAU[1] * stride])
    check.eq("  ... on a face-adjacent component of exactly the 3 target tiles",
             [c["component_tiles_row_col"] for c in on_target],
             [[list(rc) for rc in PER_TILE_P1REL_TARGET_TILES]] * 2)
    check.eq("  the loud distractor tile NEVER fires (its own gate holds it)",
             any(list(distractor_rc) in c["firing_tiles_row_col"]
                 for c in candidates), False)
    check.eq("  every accepted candidate is under the 48-tile global-change "
             "cap",
             all(c["n_tiles_firing_at_sample"]
                 < DEFAULT_PER_TILE_MAX_FIRING_TILES for c in candidates), True)

    print("\n[P3] flat and SUB-FLOOR controls yield ZERO candidates", flush=True)
    flat = np.full((PER_TILE_P1REL_FRAMES, 120, 180), 40.0, dtype=np.float32)
    flat_tiled = tiled_template_signals(flat, DEFAULT_TILE_SIZE)
    flat_scales = per_tile_noise_scale(flat_tiled["tile_template_dist"])
    flat_floor = per_tile_absolute_floor([flat_scales],
                                         DEFAULT_PER_TILE_FLOOR_F)
    flat_result = per_tile_camera_candidates(
        flat_tiled, source_frames, k_mad=DEFAULT_K_MAD,
        min_amplitude=flat_floor["floor_grey_levels"], stride=stride)
    check.eq("  a constant window has zero tile deviation everywhere",
             float(np.max(flat_tiled["tile_template_dist"])), 0.0)
    check.eq("  its measured floor is DEGENERATE ZERO, and says so",
             flat_floor["floor_is_degenerate_zero"], True)
    check.eq("  constant window -> ZERO accepted candidates",
             len(flat_result["candidates"]), 0)

    p3_frames = [n * stride for n in range(PER_TILE_P3_FRAMES)]
    sub = tiled_template_signals(_per_tile_p3_fixture(
        PER_TILE_P3_SUBFLOOR_CONTRAST), DEFAULT_TILE_SIZE)
    sub_scales = per_tile_noise_scale(sub["tile_template_dist"])
    sub_floor = per_tile_absolute_floor([sub_scales],
                                        DEFAULT_PER_TILE_FLOOR_F)
    quiet_rc = PER_TILE_P3_QUIET_TILES[1]
    quiet_series = sub["tile_template_dist"][:, quiet_rc[0], quiet_rc[1]]
    quiet_gate = robust_threshold(quiet_series, DEFAULT_K_MAD)
    quiet_peak = float(np.max(quiet_series))
    quiet_excess = quiet_peak - quiet_gate["median"]
    sub_result = per_tile_camera_candidates(
        sub, p3_frames, k_mad=DEFAULT_K_MAD,
        min_amplitude=sub_floor["floor_grey_levels"], stride=stride)
    open_result = per_tile_camera_candidates(
        sub, p3_frames, k_mad=DEFAULT_K_MAD, min_amplitude=0.0, stride=stride)
    print(f"  ---- P3 SUB-FLOOR NUMBERS: quiet tile {list(quiet_rc)} peak "
          f"{quiet_peak:.6f} vs its own relative gate "
          f"{quiet_gate['threshold']:.6f} (PASSES), excess "
          f"{quiet_excess:.6f} vs the absolute floor "
          f"{sub_floor['floor_grey_levels']:.6f} (FAILS)", flush=True)
    check.eq("  the sub-floor excursion CLEARS its own tile's relative gate",
             quiet_peak > quiet_gate["threshold"], True)
    check.eq("  ... and is BELOW the pooled absolute floor",
             quiet_excess < sub_floor["floor_grey_levels"], True)
    check.eq("  so the ABSOLUTE FLOOR alone rejects it: ZERO candidates",
             len(sub_result["candidates"]), 0)
    check.eq("  ... and with the floor removed it WOULD have been accepted, "
             "so the floor is what did the work",
             len(open_result["candidates"]) > 0, True)

    print("\n[P4] the tiling covers every pixel EXACTLY ONCE, edge tiles "
          "included", flush=True)
    for height, width in ((540, 960), (100, 130), (61, 59), (60, 60), (1, 1)):
        rows = tile_edges(height, DEFAULT_TILE_SIZE)
        cols = tile_edges(width, DEFAULT_TILE_SIZE)
        cover = np.zeros((height, width), dtype=np.int32)
        for y0, y1 in rows:
            for x0, x1 in cols:
                cover[y0:y1, x0:x1] += 1
        check.eq(f"  {width}x{height}: {len(cols)}x{len(rows)} tiles, every "
                 "pixel covered exactly once",
                 (int(cover.min()), int(cover.max()),
                  sum(y1 - y0 for y0, y1 in rows),
                  sum(x1 - x0 for x0, x1 in cols)),
                 (1, 1, height, width))
    check.raises("  tile size 0 refused", lambda: tile_edges(100, 0))
    check.raises("  empty raster refused", lambda: tile_edges(0, 60))

    print("\n[P4b] SPATIAL COHERENCE is 4-connected: DIAGONAL tiles are NOT "
          "adjacent", flush=True)
    diagonal = np.zeros((4, 4), dtype=bool)
    diagonal[0, 0] = diagonal[1, 1] = diagonal[2, 2] = True
    check.eq("  three diagonal tiles are THREE components, not one",
             [len(c) for c in grid_connected_components(diagonal)], [1, 1, 1])
    straight = np.zeros((4, 4), dtype=bool)
    straight[1, 0] = straight[1, 1] = straight[1, 2] = True
    check.eq("  three face-adjacent tiles are ONE component of 3",
             [len(c) for c in grid_connected_components(straight)], [3])
    elbow = np.zeros((4, 4), dtype=bool)
    elbow[0, 0] = elbow[1, 0] = elbow[1, 1] = True
    check.eq("  an L of three tiles is ONE component of 3",
             [len(c) for c in grid_connected_components(elbow)], [3])
    check.eq("  an empty grid has no components",
             grid_connected_components(np.zeros((3, 3), dtype=bool)), [])

    print("\n[P5] the MEASURED floor and its sweep are recorded", flush=True)
    sweep = [per_tile_absolute_floor([scales], f)
             for f in PER_TILE_FLOOR_F_SWEEP]
    print(f"  ---- P5 SWEEP over F (SUPPLEMENTARY; the census is F=3.0 only)",
          flush=True)
    for record in sweep:
        marker = ("PRIMARY" if record["F"] == DEFAULT_PER_TILE_FLOOR_F
                  else "probe")
        print(f"       F={record['F']:>4} -> floor "
              f"{record['floor_grey_levels']:.6f} grey levels  [{marker}]",
              flush=True)
    check.eq("  the sweep has the five frozen points",
             [r["F"] for r in sweep], list(PER_TILE_FLOOR_F_SWEEP))
    check.eq("  the floor is monotone non-decreasing in F",
             [r["floor_grey_levels"] for r in sweep]
             == sorted(r["floor_grey_levels"] for r in sweep), True)
    check.eq("  F = 3.0 is the primary reading and is in the sweep",
             DEFAULT_PER_TILE_FLOOR_F in PER_TILE_FLOOR_F_SWEEP, True)
    check.eq("  every record says F is a DECLARED JUDGMENT, not derived",
             all(r["F_is_a_declared_judgment_not_derived_from_data"]
                 for r in sweep), True)
    check.close("  the recorded floor is exactly F x the median noise scale",
                floor_record["floor_grey_levels"],
                DEFAULT_PER_TILE_FLOOR_F
                * floor_record["noise_scale_median_grey_levels"], 1e-12)
    check.eq("  measured over all 144 tile series of this (camera, window)",
             floor_record["n_tile_series"], 144)
    check.raises("  an empty scale set is REFUSED, never guessed",
                 lambda: per_tile_absolute_floor([], 3.0))

    print("\n[P2-RECALL] NOT RUNNABLE OFFLINE -- it needs the real proxies",
          flush=True)
    print("  The frozen spec's P2-RECALL requires the detector to fire on "
          "both audited ground-truth events on cam11/cam12 within +/-2 proxy "
          "samples and to be SILENT through the absence. That is a "
          "measurement on real footage and cannot be asserted here. It is "
          "recorded as NOT RUN rather than omitted. Those two events are a "
          "RECALL FIXTURE, NOT A TUNING SET: no threshold in this file may be "
          "moved to make them pass.", flush=True)

    print("\n" + "=" * 78, flush=True)
    if check.failures:
        print(f"[per-tile-selftest] FAILED: {len(check.failures)} of "
              f"{check.passed + len(check.failures)} preconditions", flush=True)
        for failure in check.failures:
            print(f"  - {failure}", flush=True)
        return 1
    print(f"[per-tile-selftest] PASSED: {check.passed}/{check.passed} "
          "preconditions", flush=True)
    for line in _wrap(TILE_EXPLANATION_NOTE, 78):
        print(f"[per-tile-selftest] {line}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("probe", "proxy", "census", "self-test",
                                           "tile-selftest", "per-tile-selftest"),
                        default=None)
    parser.add_argument("--self-test", action="store_true",
                        help="run the offline arithmetic/mapping/reader checks "
                             "and exit; needs no media, no ffmpeg, no GPU")
    parser.add_argument("--tile-selftest", action="store_true",
                        help="run the TILE PASS frozen preconditions P1-P5 and "
                             "exit; needs no media, no ffmpeg, no GPU")
    parser.add_argument("--per-tile-selftest", action="store_true",
                        help="run the PER-TILE PASS (v2) frozen preconditions "
                             "P1-REL, P3, P4, P5 and exit; needs no media, no "
                             "ffmpeg, no GPU")
    # probe / proxy
    parser.add_argument("--source-dir", default=None,
                        help="scene folder of cam*.mp4 (READ ONLY)")
    parser.add_argument("--scene", default=None,
                        help="scene name for the derived path (default: source basename)")
    parser.add_argument("--derived-root", default=DEFAULT_DERIVED_ROOT,
                        help="derived proxy root; NEVER the raw directory")
    parser.add_argument("--out-root", default=None,
                        help="explicit proxy root (overrides --derived-root/--scene)")
    parser.add_argument("--out", default=None, help="manifest path (probe, census)")
    parser.add_argument("--count-frames", action="store_true",
                        help="probe: DECODE to count frames (nb_read_frames) "
                             "instead of trusting the container's nb_frames")
    parser.add_argument("--skip-hash", action="store_true",
                        help="probe/proxy: skip the source SHA-256 (fast "
                             "structural probe only; the manifest records that "
                             "hashes were not computed)")
    parser.add_argument("--probe-manifest", default=None,
                        help="proxy: reuse an existing probe manifest instead "
                             "of re-hashing multi-GB sources")
    parser.add_argument("--long-edge", type=int, default=DEFAULT_LONG_EDGE)
    parser.add_argument("--proxy-fps", default=DEFAULT_PROXY_FPS,
                        help="requested proxy rate; '2', '1/2', '5/2'. The "
                             "ACHIEVED rate is source_rate/round(source/req) "
                             "and both are recorded")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None,
                        help="inclusive; default = nb_frames - 1")
    parser.add_argument("--pix-fmt", default="rgb24", choices=("rgb24", "gray"))
    parser.add_argument("--sws-flags", default="bicubic")
    parser.add_argument("--limit-cameras", type=int, default=None,
                        help="bounded smoke run over the first N cameras")
    parser.add_argument("--force", action="store_true",
                        help="proxy: rebuild cameras that are already complete")
    # census
    parser.add_argument("--proxy-root", default=None)
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument("--k-mad", type=float, default=DEFAULT_K_MAD)
    parser.add_argument("--min-amplitude", type=float, default=DEFAULT_MIN_AMPLITUDE,
                        help="ABSOLUTE floor in 8-bit grey levels; a purely "
                             "scale-free screen is insufficient")
    parser.add_argument("--match-tol-frames", type=int, default=None,
                        help="cross-camera match tolerance; default = one proxy step")
    parser.add_argument("--min-cameras", type=int, default=2,
                        help="cameras needed before a cluster is counted as "
                             "multi-camera in the summary (does not filter output)")
    # census: tile pass (OFF by default; the census is unchanged without it)
    parser.add_argument("--tile-mode", action="store_true",
                        help="census: run changepoint detection on the per-tile "
                             "MAXIMUM of template_dist instead of the "
                             "whole-frame mean. The default signals are "
                             "whole-frame means and cannot see a small object; "
                             "this is the high-recall companion pass. The "
                             "global signal is still reported beside every "
                             "candidate. Adds spatial explanation data to the "
                             "manifest -- an EXPLANATION OF THE DETECTOR "
                             "SIGNAL, NOT an instance mask")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE,
                        help="census: square tile edge in PROXY pixels "
                             f"(default {DEFAULT_TILE_SIZE}; 16x9 = 144 tiles "
                             "at the 960x540 census raster). Partial edge "
                             "tiles are included and never area-weighted")
    parser.add_argument("--tile-min-amplitude", type=float,
                        default=DEFAULT_TILE_MIN_AMPLITUDE,
                        help="census: ABSOLUTE floor in 8-bit grey levels for "
                             "the tile pass, measured on a TILE mean "
                             f"(default {DEFAULT_TILE_MIN_AMPLITUDE}). Same "
                             "two-gate structure as --min-amplitude: the "
                             "robust relative gate AND this floor must pass")
    parser.add_argument("--tile-top-n", type=int, default=DEFAULT_TILE_TOP_N,
                        help="census: contributing tiles recorded per candidate "
                             "for the review overlay")
    # census: PER-TILE pass, v2 (OFF by default; supersedes --tile-mode's
    # reduction and may not be combined with it)
    parser.add_argument("--per-tile-mode", action="store_true",
                        help="census: run the changepoint detector "
                             "INDEPENDENTLY on each per-tile signal instead of "
                             "on the max over tiles. Each tile's relative gate "
                             "comes from its OWN temporal median and MAD, so a "
                             "quiet tile gets a quiet threshold -- the "
                             "property the max reduction lacked, because a "
                             "maximum is monopolised by the loudest region, "
                             "which sets both its median and its MAD. "
                             "SUPERSEDES --tile-mode's reduction; the two may "
                             "not be combined")
    parser.add_argument("--per-tile-floor-f", type=float,
                        default=DEFAULT_PER_TILE_FLOOR_F,
                        help="census: the FROZEN absolute-floor multiplier. "
                             "floor = F * median over all (camera, window, "
                             "tile) of 1.4826*MAD_t(S_ij). F = "
                             f"{DEFAULT_PER_TILE_FLOOR_F} is the PRIMARY "
                             "READING and is a declared judgment, not derived "
                             "from data; any other value is a SENSITIVITY "
                             "PROBE and may not be reported as the census")
    parser.add_argument("--per-tile-min-component", type=int,
                        default=DEFAULT_PER_TILE_MIN_COMPONENT,
                        help="census: FROZEN minimum face-adjacent "
                             "(4-connected) component of tiles firing at the "
                             f"same proxy sample (default "
                             f"{DEFAULT_PER_TILE_MIN_COMPONENT})")
    parser.add_argument("--per-tile-max-firing-tiles", type=int,
                        default=DEFAULT_PER_TILE_MAX_FIRING_TILES,
                        help="census: FROZEN EXCLUSIVE cap on how many tiles "
                             "may fire at one proxy sample before the sample "
                             "is read as a global change rather than a "
                             f"localized one (default "
                             f"{DEFAULT_PER_TILE_MAX_FIRING_TILES} of 144)")
    parser.add_argument("--emit-signals", action="store_true",
                        help="census: include the per-camera per-frame signals "
                             "in the manifest (large)")
    parser.add_argument("--top", type=int, default=20,
                        help="census: candidate rows printed per window")
    args = parser.parse_args(argv)

    mode = "self-test" if args.self_test else args.mode
    if args.tile_selftest and not args.self_test:
        mode = "tile-selftest"
    if args.per_tile_selftest and not args.self_test and not args.tile_selftest:
        mode = "per-tile-selftest"
    if mode is None:
        parser.error("one of --mode {probe,proxy,census,self-test,"
                     "tile-selftest,per-tile-selftest} or --self-test or "
                     "--tile-selftest or --per-tile-selftest")

    if mode == "self-test":
        return self_test()
    if mode == "tile-selftest":
        return tile_self_test()
    if mode == "per-tile-selftest":
        return per_tile_self_test()

    if mode == "probe":
        if not args.source_dir:
            raise ContractError("--mode probe needs --source-dir")
        _require_tools()
        report = probe_scene(Path(args.source_dir),
                             count_frames=args.count_frames,
                             skip_hash=args.skip_hash)
        agreed = report["agreed"]
        print(f"[imvid-proxy] {report['n_cameras']} cameras AGREE: "
              f"{agreed['width']}x{agreed['height']}, {agreed['nb_frames']} frames, "
              f"{agreed['r_frame_rate_exact']} fps "
              f"({agreed['frame_period_ms']:.5f} ms/frame)", flush=True)
    elif mode == "proxy":
        if not args.source_dir:
            raise ContractError("--mode proxy needs --source-dir")
        report = mode_proxy(args)
        print(f"[imvid-proxy] {report['n_cameras']} cameras, "
              f"{report['total_bytes'] / 2**20:.1f} MiB total, proxies -> "
              f"{report['proxy_root']}", flush=True)
        #  ALWAYS beside the proxies, whether or not --out was given: the
        #  scene manifest is the record of how those files were produced and
        #  must not be separable from them.
        scene_manifest = (Path(report["proxy_root"])
                          / "MANIFEST.imvid_event_proxy_scene.json")
        scene_manifest.write_text(json.dumps(report, indent=2, sort_keys=True),
                                  encoding="utf-8")
        print(f"[imvid-proxy] scene manifest -> {scene_manifest}", flush=True)
    else:
        if not args.proxy_root:
            raise ContractError("--mode census needs --proxy-root")
        report = mode_census(args)
        print_census(report, int(args.top))

    if args.out:
        out = Path(args.out)
        _check_output_root(out.parent,
                           Path(args.source_dir) if args.source_dir else None)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[imvid-proxy] manifest -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
