#!/usr/bin/env python3
"""Turn an ImViD event CENSUS into a self-contained HTML REVIEW GALLERY.

The census produced by ``scripts/imvid_event_proxy.py --mode census`` is a
JSON manifest, in one of THREE detection schemas -- ``--per-tile-mode``
(v2, the current one), ``--tile-mode`` (v1, max-over-tiles, superseded) and
plain whole-frame. This script DETECTS which one it was handed and RENDERS
THAT ONE; it refuses anything it does not recognize rather than falling back
(see ``detect_census_mode``). A human cannot curate events by reading
JSON: the N3V precedent on this project
([[operations/crb300-event-mask-curation-2026-08-23]]) is that automated
candidates were accepted or rejected only at FRAME-BY-FRAME VISUAL
verification, and that several high-scoring automated candidates died there.
This script builds the artefact that makes that verification possible --
pictures, with the exact source frame index and the exact measured timestamp
beside every one of them -- and a form for recording the human's provisional
class.

--------------------------------------------------------------------
THIS GALLERY SHOWS CANDIDATES. IT NEVER CLAIMS AN EVENT
--------------------------------------------------------------------

Every rendered page carries that wording. Three things it must never be read
as saying:

* a candidate is an event. It is a proposal for human / ground-truth
  classification, and nothing here has been compared against ground truth;
* a polarity is a semantic. ``rise``/``fall`` is a DIRECTION OF THE MEASURED
  SIGNAL. A rise is equally consistent with an occluder arriving, with
  content leaving, and with a lighting change;
* a failure to find an occluder in the pictures is evidence of absence. The
  proxy raster is a downscale of a downscale and the detector is blind by
  construction to anything below its declared scale.

--------------------------------------------------------------------
THE SPATIAL OVERLAY IS AN EXPLANATION, NOT A MASK
--------------------------------------------------------------------

The tile heatmap answers "WHERE ON THE PROXY RASTER did the scalar that the
detector thresholded come from". It does not segment an object, does not
bound one, and does not assert that anything is present, absent, revealed or
occluded in any tile. The census's own ``TILE_EXPLANATION_NOTE`` is carried
verbatim onto the page next to the picture, because a spatial overlay is
exactly the artefact a reader mistakes for an instance mask.

In PER-TILE mode the overlay is drawn from the FIRING CONNECTED COMPONENT --
every face-adjacent tile whose own two-gate detector crossed at that proxy
sample -- and never from a single argmax tile, because in that schema no
argmax exists: the detector ran independently on each tile and the component
IS the candidate.

--------------------------------------------------------------------
RANKING: MEAN AMPLITUDE, NEVER CAMERA SUPPORT (per-tile mode)
--------------------------------------------------------------------

[[operations/imvid-tile-scout-v2-per-tile-2026-08-25]] §8 measured the
inversion: ranked by amplitude the two audited real events sit at 3/76 and
17/76; ranked by ``n_cameras_supporting`` the two promoted clusters were
54/76 and 60/76 -- the two WEAKEST-amplitude clusters in the set. Only
~15 of 39 cameras see the subject at all, so camera count measures how
GLOBAL a change is, which is the opposite of the target property. ``C_min``
is a corroboration floor, never a sort key. The camera count is still shown
on every card because it is informative; it orders nothing.

--------------------------------------------------------------------
TIMESTAMPS COME FROM THE RATIONAL RATE, NEVER FROM AN ASSUMED FPS
--------------------------------------------------------------------

``t = source_frame * denominator / numerator`` on the exact ``Fraction``
recorded in the proxy manifests and echoed in the census
(``mapping.source_rate_exact``). ImViD is 60000/1001, so an assumed 60.0
drifts by one whole frame every 1001 frames; on a 15,215-frame take that is
15 frames of error at the end. Both a float and the exact rational string are
emitted for every candidate so the derivation is checkable from the output.

--------------------------------------------------------------------
NO SILENT CAPS
--------------------------------------------------------------------

Caps exist (bundle size is a real constraint: this is pulled over rclone to a
workstation). Every one of them, when it bites, is printed to stdout, written
into ``MANIFEST.gallery.json``, and rendered as a VISIBLE notice at the top of
the index page. A gallery that quietly showed 40 of 300 candidates would be a
worse instrument than no gallery.

Output bundle::

    <out>/index.html                     one file, opens over file://
    <out>/assets/*.jpg                   downscaled sheets
    <out>/MANIFEST.gallery.json          provenance

Self-contained: no CDN, no external stylesheet, no external script, no
network of any kind. Needs numpy and PIL only.

Examples::

  python3 scripts/imvid_build_gallery.py \\
      --census /apollo/users/sri/proj_adags/runs/.../census.json \\
      --proxy-root /apollo/users/sri/proj_adags/data/imvid/derived/proxy/<scene> \\
      --out /apollo/users/sri/proj_adags/runs/.../gallery
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import html
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402
from scripts import imvid_event_proxy as proxy  # noqa: E402


# ---------------------------------------------------------------------------
# Frozen wording. These strings are contractual: tests pin them, and the
# reviewer is expected to read them, so they are defined once and reused.
# ---------------------------------------------------------------------------

GALLERY_SCHEMA = "imvid-gallery-v1"
DECISIONS_SCHEMA = "imvid-gallery-decisions-v1"
GALLERY_MANIFEST_NAME = "MANIFEST.gallery.json"

#: The literal string the primary asked to appear on every candidate card.
#: The dash is U+2014. Do not "normalize" it.
SPREAD_CAVEAT = "proxy-resolution localization spread — not synchronization"

NOT_AN_EVENT = (
    "THESE ARE CANDIDATES, NOT EVENTS. Every row in this gallery is a "
    "PROPOSAL FOR HUMAN / GROUND-TRUTH CLASSIFICATION. Nothing here has been "
    "compared against ground truth and nothing here is claimed to be an "
    "event. Candidate polarity (rise / fall) is a DIRECTION OF THE MEASURED "
    "SIGNAL, not a semantic: a rise is equally consistent with an occluder "
    "arriving, with content leaving, and with a lighting change. FAILURE TO "
    "FIND AN OCCLUDER IN THESE PICTURES IS NOT EVIDENCE OF ABSENCE -- the "
    "proxy raster is a downscale and the detector is blind by construction "
    "below its declared scale."
)

OVERLAY_HEADING = (
    "SPATIAL EXPLANATION OF THE DETECTOR SIGNAL — NOT AN INSTANCE MASK, "
    "NOT PROOF OF IDENTITY"
)

OVERLAY_CAVEAT = (
    "This heatmap is not an instance mask. It shows WHERE ON THE PROXY RASTER "
    "the scalar the detector thresholded came from. The boxes are TILE "
    "EXTENTS, not object extents. A tile is bright because its pixels "
    "departed from this window's own per-pixel temporal median, which a "
    "moving object, a moving occluder, a shadow, a lighting change and "
    "compression noise all produce equally. Nothing in this picture "
    "identifies an object."
)

PER_TILE_OVERLAY_HEADING = (
    "SPATIAL EXPLANATION OF THE DETECTOR SIGNAL — THE FIRING COMPONENT, "
    "NOT AN INSTANCE MASK, NOT PROOF OF IDENTITY"
)

PER_TILE_OVERLAY_CAVEAT = (
    "This overlay is not an instance mask. It shows the FACE-ADJACENT "
    "CONNECTED COMPONENT of tiles whose OWN per-tile detector crossed its OWN "
    "median + k_mad*MAD threshold at this proxy sample, which is what the "
    "per-tile census screened. The boxes are TILE EXTENTS, not object "
    "extents, and the component bounding box bounds TILES, not an object. A "
    "tile fired because its pixels departed from this window's own per-pixel "
    "temporal median, which a moving object, a moving occluder, a shadow, a "
    "lighting change and compression noise all produce equally. Nothing in "
    "this picture identifies an object."
)

#: Census detection schemas this gallery understands. Anything else is
#: REFUSED: a gallery that silently rendered an unrecognized schema would
#: produce a ranking and an overlay that no spec authorizes.
MODE_PER_TILE = "per_tile"
MODE_TILE = "tile"
MODE_WHOLE_FRAME = "whole_frame"

#: ``signal_used_for_detection`` -> mode. The census WRITES this string on
#: every cluster it emits from a tile-derived reduction, so it is the
#: cheapest available cross-check that the pass block and the clusters agree.
SIGNAL_TO_MODE = {
    "per_tile_template_dist": MODE_PER_TILE,
    "tile_max": MODE_TILE,
}

#: The per-cluster fields a PER-TILE census emits. Every one is load-bearing
#: for the overlay or for the card, and every one is read WITHOUT a default.
PER_TILE_CLUSTER_FIELDS = (
    "per_camera_component_size_tiles",
    "per_camera_n_tiles_firing_at_sample",
    "per_camera_component_tiles_row_col",
    "per_camera_firing_tiles_row_col",
    "per_camera_per_tile_amplitude",
    "per_camera_tile_pixel_boxes_xyxy",
    "per_camera_component_pixel_bbox_xyxy",
)

#: The ``per_tile_pass`` sub-fields the index page reports as provenance.
PER_TILE_FLOOR_FIELDS = (
    "F",
    "floor_grey_levels",
    "noise_scale_median_grey_levels",
    "n_tile_series",
    "n_zero_scale_tile_series",
    "floor_is_degenerate_zero",
)

CLASS_CHOICES = ("A", "B", "C", "reject")

#: THE FROZEN CLASS DEFINITIONS. These are NOT "the reviewer's own tiers":
#: they are fixed by [[operations/imvid-event-definition-2026-08-24]] §3, and
#: only class A is an event.
CLASS_DEFINITIONS = (
    ("A", "genuine scene ABSENCE — the object has LEFT THE REPRESENTED "
          "VOLUME. Requires POSITIVE multi-view evidence that the volume the "
          "object occupied is observed FREE for the duration of the gap. "
          "FAILURE TO FIND AN OCCLUDER IS NOT EVIDENCE — that residual "
          "inference is exactly what produced 0-of-597 on DiVa-360."),
    ("B", "ordinary OCCLUSION — the object is still present but hidden from "
          "the applicable cameras. A class-B candidate is not worthless; it "
          "is recorded AS class B and may not be reported under a heading "
          "that implies absence."),
    ("C", "RIG- or CAMERA-INDUCED — the object left the frustum, the camera "
          "moved, exposure / white balance changed, or the applicable camera "
          "set itself changed."),
    ("reject", "not a real scene change."),
)

CLASS_FRAME_NOTE = (
    "A ball thrown out of ONE camera's frame is C for that camera; it is A "
    "only if it left the volume observed by the APPLICABLE CAMERA SET. "
    "Frozen in imvid-event-definition-2026-08-24 §3 — these definitions are "
    "not this tool's, and this tool neither ranks nor acts on them."
)

CLASS_LEGEND_TITLE = "THE A / B / C CLASSES ARE FROZEN, NOT THIS TOOL'S"

RANKING_NOTE_PER_TILE = (
    "RANKED BY MEAN AMPLITUDE, DESCENDING. Camera count is RETIRED as the "
    "primary order by imvid-tile-scout-v2-per-tile-2026-08-25 §8 and is shown "
    "on every card for information only: C_min is a corroboration floor, "
    "never a sort key. Ranked by camera support the two audited real events "
    "sat at 54/76 and 60/76 — the two WEAKEST-amplitude clusters in the set."
)

RANKING_NOTE_LEGACY = (
    "Ranked by supporting-camera count, then amplitude, then time — the "
    "order this census schema itself printed. This is the SUPERSEDED v1 "
    "ordering and it is used here only because this manifest is a v1 "
    "manifest."
)

#: Column offsets, in PROXY STEPS, sampled around the cluster anchor frame.
DEFAULT_CONTEXT_STEPS = 3
DEFAULT_MAX_CANDIDATES = 40
DEFAULT_MAX_CAMERAS_PER_CANDIDATE = 12
DEFAULT_MAX_MONTAGE_PAGES = 4
DEFAULT_MONTAGE_TILE_WIDTH = 180
DEFAULT_FOCUS_TILE_WIDTH = 420
DEFAULT_OVERLAY_WIDTH = 720
DEFAULT_JPEG_QUALITY = 80
DEFAULT_MAX_BYTES = 200 * 2 ** 20

#: Label band colours, keyed by the pre / gap / post phase of a tile.
PHASE_STYLE = {
    "pre": ((36, 62, 110), (150, 190, 245)),
    "gap": ((122, 74, 8), (255, 214, 128)),
    "post": ((22, 82, 52), (150, 235, 185)),
    "missing": ((70, 70, 70), (200, 200, 200)),
}

#: magma anchors; a perceptual ramp without importing matplotlib.
_HEAT_STOPS = (
    (0.00, (0, 0, 4)),
    (0.25, (81, 18, 124)),
    (0.50, (183, 55, 121)),
    (0.75, (252, 137, 97)),
    (1.00, (252, 253, 191)),
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int):
    """A legible font without shipping one.

    ``load_default(size=...)`` exists from Pillow 10.1; older Pillow returns a
    fixed ~11 px bitmap font, which is ugly but readable, and a missing font
    must never be the reason a gallery cannot be built.
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        pass
    for name in ("DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


#: Pillow's bundled default face has no em dash and no arrows, and a missing
#: glyph renders as a tofu box INSIDE THE PICTURE, where it cannot be
#: corrected by the reader. Burned text is transliterated; the HTML keeps the
#: real characters, which is where the frozen wording is pinned.
_BURN_SUBSTITUTIONS = {
    "—": "--", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "≥": ">=", "≤": "<=",
    "→": "->", "·": "-", "…": "...", "×": "x",
}


def _drawable(text: str) -> str:
    out = str(text)
    for source, target in _BURN_SUBSTITUTIONS.items():
        out = out.replace(source, target)
    return out


def _text_height(font) -> int:
    try:
        box = font.getbbox("Ag")
        return int(box[3] - box[1]) + 2
    except AttributeError:
        return 12


# ---------------------------------------------------------------------------
# The frame-index -> time mapping. The load-bearing arithmetic.
# ---------------------------------------------------------------------------

def frame_time_exact(source_frame: int, source_rate: Fraction) -> Fraction:
    """Seconds of ``source_frame``, EXACTLY, at the measured rational rate.

    ``t = n / rate``. Never ``n / 30``, never ``n / 60``, never
    ``n / round(rate)``. At 60000/1001 an assumed 60.0 is short by
    ``n * 1/60000 * 1`` seconds per 1001 frames -- one whole frame of error
    every 1001 frames, 15 frames over a full 15,215-frame ImViD take.
    """
    if source_rate <= 0:
        raise ContractError(f"non-positive source rate {source_rate}")
    return Fraction(int(source_frame)) * Fraction(source_rate.denominator,
                                                  source_rate.numerator)


def format_seconds(value: Fraction, places: int = 4) -> str:
    return f"{float(value):.{places}f}"


# ---------------------------------------------------------------------------
# Reading the inputs
# ---------------------------------------------------------------------------

def load_census(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "imvid-event-proxy-census-v1":
        raise ContractError(
            f"{path} is not an imvid-event-proxy-census-v1 manifest "
            f"(schema {report.get('schema')!r}). REFUSED -- a gallery built "
            "from an unknown schema would mislabel every frame index it "
            "renders.")
    return report


def _refuse(found: str, expected: str) -> ContractError:
    """One refusal shape: what was FOUND, and what was EXPECTED instead.

    Every load-bearing schema field in this module is read through a refusal
    like this rather than through ``.get(..., default)``. The gallery's
    original per-tile defect was not a crash: it read ``n_cameras_supporting``
    and ``tile_explanations`` off a per-tile manifest with ``.get()``, found
    neither, and DEGRADED SILENTLY into exactly the ranking §8 forbids with no
    overlay at all. A crash would have been better, so this is the crash.
    """
    return ContractError(
        f"REFUSED. FOUND: {found}. EXPECTED: {expected}. This gallery renders "
        "the schema it was handed and refuses the rest; it does not fall back, "
        "because a silent fallback is how a forbidden ranking gets published.")


def _declared_cluster_modes(census: dict) -> set:
    """The detection mode every cluster in the census DECLARES for itself."""
    modes = set()
    for window in census.get("windows") or []:
        for cluster in window.get("candidate_clusters") or []:
            if "signal_used_for_detection" not in cluster:
                modes.add(MODE_WHOLE_FRAME)
                continue
            signal = cluster["signal_used_for_detection"]
            if signal not in SIGNAL_TO_MODE:
                raise _refuse(
                    f"a cluster whose signal_used_for_detection is "
                    f"{signal!r}, which this gallery has never seen",
                    f"one of {sorted(SIGNAL_TO_MODE)} or the field to be "
                    f"absent (whole-frame census)")
            modes.add(SIGNAL_TO_MODE[signal])
    return modes


def detect_census_mode(census: dict) -> str:
    """Which detection schema this manifest is, or a REFUSAL.

    Three schemas exist and they are not interchangeable:

    ``per_tile``
        ``--per-tile-mode``. ``per_tile_pass`` is present; each window carries
        ``per_tile_candidates``; each cluster carries the firing COMPONENT per
        supporting camera. Ranked by MEAN AMPLITUDE
        ([[operations/imvid-tile-scout-v2-per-tile-2026-08-25]] §8).
    ``tile``
        ``--tile-mode``, the SUPERSEDED max-over-tiles reduction.
        ``tile_pass`` is present and each window carries ``tile_explanations``.
    ``whole_frame``
        neither pass ran; detection was on whole-frame means and there is no
        spatial explanation to draw.

    The declared pass block and the clusters' own
    ``signal_used_for_detection`` must AGREE. A manifest where they disagree
    is refused, not reconciled.
    """
    if not isinstance(census, dict):
        raise _refuse(f"a {type(census).__name__}", "a census manifest object")
    has_tile = bool((census.get("tile_pass") or {}).get("enabled"))
    has_per_tile = bool((census.get("per_tile_pass") or {}).get("enabled"))
    if has_tile and has_per_tile:
        raise _refuse(
            "BOTH tile_pass.enabled and per_tile_pass.enabled",
            "exactly one of them — they are two DIFFERENT reductions of the "
            "same per-tile grid and the census itself refuses to combine them")

    declared = _declared_cluster_modes(census)
    if has_per_tile:
        mode = MODE_PER_TILE
    elif has_tile:
        mode = MODE_TILE
    elif "tile_pass" in census or "per_tile_pass" in census:
        raise _refuse(
            "a tile_pass / per_tile_pass block that is present but not "
            "enabled",
            "either an enabled pass block or no pass block at all")
    else:
        mode = MODE_WHOLE_FRAME

    wrong = declared - {mode}
    if wrong:
        raise _refuse(
            f"a {mode!r} manifest whose clusters declare "
            f"signal_used_for_detection for mode(s) {sorted(wrong)}",
            f"every cluster to declare the {mode!r} signal, or a manifest "
            f"whose pass block matches its clusters")

    if mode == MODE_PER_TILE:
        _validate_per_tile(census)
    elif mode == MODE_TILE:
        _validate_tile(census)
    return mode


def _validate_per_tile(census: dict) -> None:
    """Every field the per-tile view reads, checked BEFORE anything renders."""
    block = census["per_tile_pass"]
    for key in ("absolute_floor", "sensitivity_sweep", "spatial_coherence",
                "ranking", "tile_size_px", "tile_grid", "k_mad",
                "primary_reading_F"):
        if key not in block:
            raise _refuse(
                f"a per_tile_pass block without {key!r} "
                f"(it holds {sorted(block)})",
                f"a per_tile_pass block carrying {key!r}, as "
                f"imvid_event_proxy.py --per-tile-mode emits it")
    floor = block["absolute_floor"]
    for key in PER_TILE_FLOOR_FIELDS:
        if key not in floor:
            raise _refuse(
                f"a per_tile_pass.absolute_floor without {key!r} "
                f"(it holds {sorted(floor)})",
                f"the measured floor record, carrying {key!r} — the floor "
                f"provenance is reported on the index page and may not be "
                f"guessed")
    for window in census.get("windows") or []:
        if "per_tile_candidates" not in window:
            raise _refuse(
                f"a per-tile census window {window.get('window_source_frames')} "
                f"without 'per_tile_candidates'",
                "each window to carry its per-camera per-tile candidate lists")
        for cluster in window.get("candidate_clusters") or []:
            for key in PER_TILE_CLUSTER_FIELDS:
                if key not in cluster:
                    raise _refuse(
                        f"a per-tile cluster at source frame "
                        f"{cluster.get('source_frame_median')} without {key!r}",
                        f"every per-tile cluster to carry {key!r}; without it "
                        f"the firing component cannot be drawn and the "
                        f"overlay would silently vanish")
            if "mean_amplitude" not in cluster:
                raise _refuse(
                    f"a per-tile cluster at source frame "
                    f"{cluster.get('source_frame_median')} without "
                    f"'mean_amplitude'",
                    "mean_amplitude on every cluster — it is the FROZEN SORT "
                    "KEY and the gallery will not substitute another")


def _validate_tile(census: dict) -> None:
    """The superseded v1 view still refuses a manifest it cannot render."""
    for window in census.get("windows") or []:
        if not (window.get("candidate_clusters") or []):
            continue
        if "tile_explanations" not in window:
            raise _refuse(
                f"a tile-mode census window "
                f"{window.get('window_source_frames')} with clusters but "
                f"without 'tile_explanations'",
                "each tile-mode window to carry its per-camera tile "
                "explanations, as --tile-mode emits them")


def load_proxy_tree(proxy_root: Path) -> dict:
    """Per-camera manifests plus the available source-frame lattice."""
    if not proxy_root.is_dir():
        raise ContractError(f"--proxy-root {proxy_root} is not a directory")
    cameras: dict[str, dict] = {}
    for camera_dir in sorted(p for p in proxy_root.iterdir() if p.is_dir()):
        manifest = camera_dir / proxy.PROXY_MANIFEST_NAME
        if not manifest.is_file():
            continue
        record = json.loads(manifest.read_text(encoding="utf-8"))
        frames = {}
        frames_dir = camera_dir / "frames"
        if frames_dir.is_dir():
            for item in frames_dir.glob(proxy.PROXY_FRAME_GLOB):
                match = proxy.PROXY_FRAME_RE.match(item.name)
                if match is None:
                    raise ContractError(f"unexpected proxy filename {item}")
                frames[int(match.group(1))] = item
        cameras[record.get("camera", camera_dir.name)] = {
            "manifest": record,
            "dir": camera_dir,
            "frames": frames,
        }
    if not cameras:
        raise ContractError(
            f"no per-camera proxy manifests ({proxy.PROXY_MANIFEST_NAME}) "
            f"under {proxy_root}")
    return cameras


def scene_manifest_sha256(proxy_root: Path) -> tuple[str | None, str | None]:
    path = proxy_root / "MANIFEST.imvid_event_proxy_scene.json"
    if path.is_file():
        return str(path), _sha256(path)
    return None, None


# ---------------------------------------------------------------------------
# Candidate flattening and ranking
# ---------------------------------------------------------------------------

def _per_tile_rank_key(row: dict) -> tuple:
    """THE FROZEN PER-TILE ORDER: mean amplitude DESCENDING, then time.

    ``n_cameras_supporting`` DOES NOT APPEAR HERE and may not be added.
    [[operations/imvid-tile-scout-v2-per-tile-2026-08-25]] §8 measured the
    inversion that retired it: only ~15 of 39 cameras see the subject at all,
    so camera count measures how GLOBAL a change is, which is the opposite of
    the target property. ``mean_amplitude`` is read WITHOUT a default: a
    cluster that lacks it was already refused by ``_validate_per_tile``, and
    a ``.get(..., 0.0)`` here would silently sort every such cluster last.
    """
    cluster = row["cluster"]
    return (-float(cluster["mean_amplitude"]),
            int(cluster.get("source_frame_median", 0)),
            (row["window_source_frames"] or [0])[0])


def _legacy_rank_key(row: dict) -> tuple:
    """The v1 / whole-frame order: support, then amplitude, then time.

    This repeats what those census schemas printed top-down. It is NOT
    reachable from a per-tile manifest.
    """
    cluster = row["cluster"]
    return (-int(cluster.get("n_cameras_supporting", 0)),
            -float(cluster.get("mean_amplitude", 0.0)),
            int(cluster.get("source_frame_median", 0)),
            (row["window_source_frames"] or [0])[0])


RANK_KEY_BY_MODE = {
    MODE_PER_TILE: _per_tile_rank_key,
    MODE_TILE: _legacy_rank_key,
    MODE_WHOLE_FRAME: _legacy_rank_key,
}


def flatten_candidates(census: dict, mode: str | None = None) -> list[dict]:
    """All clusters from all windows, ranked, tagged with their window.

    Ranking is decided by the DETECTED SCHEMA: a per-tile census ranks by
    MEAN AMPLITUDE (frozen, §8), every other schema repeats its own printed
    order. No threshold is applied here and nothing is filtered: ranking
    decides ORDER, the caps decide how much fits, and both are reported.
    """
    if mode is None:
        mode = detect_census_mode(census)
    if mode not in RANK_KEY_BY_MODE:
        raise _refuse(f"detection mode {mode!r}",
                      f"one of {sorted(RANK_KEY_BY_MODE)}")
    rows = []
    for window in census.get("windows", []):
        window_frames = window.get("window_source_frames")
        for cluster in window.get("candidate_clusters", []):
            rows.append({
                "cluster": cluster,
                "window_source_frames": window_frames,
                "window": window,
            })
    rows.sort(key=RANK_KEY_BY_MODE[mode])
    for i, row in enumerate(rows, start=1):
        row["candidate_id"] = f"c{i:04d}"
    return rows


def tile_explanation_for(window: dict, camera: str,
                         source_frame: int) -> dict | None:
    """The per-candidate tile explanation for ``camera`` at ``source_frame``.

    The census stores explanations per camera as a LIST parallel to that
    camera's event list; the cluster stores only the per-camera source frame.
    Matching on the source frame is exact because a camera emits at most one
    changepoint per proxy sample.
    """
    table = window.get("tile_explanations") or {}
    for entry in table.get(camera, []) or []:
        if int(entry.get("source_frame", -1)) == int(source_frame):
            return entry
    return None


def per_tile_explanation_for(window: dict, cluster: dict,
                             camera: str) -> dict | None:
    """The FIRING COMPONENT that carried ``camera``'s crossing in this cluster.

    Assembled from the cluster's own per-camera columns -- which the census
    emits precisely so the component survives clustering -- and enriched, when
    an exact match exists, with the full per-camera candidate record in
    ``window["per_tile_candidates"]``. The match is on BOTH the source frame
    and the component tile set, because a camera can emit several
    non-adjacent components at the same proxy sample and a frame-only match
    would silently pick the wrong one.

    Returns ``None`` only when this camera does not support the cluster.
    Every field it does read is read WITHOUT a default.
    """
    frames = cluster["per_camera_source_frame"]
    if camera not in frames:
        return None
    source_frame = int(frames[camera])
    component = [[int(r), int(c)]
                 for r, c in cluster["per_camera_component_tiles_row_col"][camera]]
    boxes = [[int(v) for v in box]
             for box in cluster["per_camera_tile_pixel_boxes_xyxy"][camera]]
    if len(boxes) != len(component):
        raise _refuse(
            f"{camera} src {source_frame}: {len(boxes)} tile pixel boxes for "
            f"{len(component)} component tiles",
            "one pixel box per component tile, in the same order")
    amplitudes = [float(v)
                  for v in cluster["per_camera_per_tile_amplitude"][camera]]
    firing = [[int(r), int(c)]
              for r, c in cluster["per_camera_firing_tiles_row_col"][camera]]
    explanation = {
        "kind": "per_tile_component_explanation",
        "is_instance_mask": False,
        "what_this_is": cluster.get("tile_explanation_note",
                                    proxy.TILE_EXPLANATION_NOTE),
        "camera": camera,
        "source_frame": source_frame,
        "component_tiles_row_col": component,
        "tile_pixel_boxes_xyxy": boxes,
        "component_pixel_bbox_xyxy": [
            int(v) for v in cluster["per_camera_component_pixel_bbox_xyxy"][camera]],
        "per_tile_amplitude": amplitudes,
        "firing_tiles_row_col": firing,
        "component_size_tiles": int(
            cluster["per_camera_component_size_tiles"][camera]),
        "n_tiles_firing_at_sample": int(
            cluster["per_camera_n_tiles_firing_at_sample"][camera]),
    }
    wanted = sorted(tuple(rc) for rc in component)
    for entry in (window["per_tile_candidates"] or {}).get(camera, []) or []:
        if int(entry.get("source_frame", -1)) != source_frame:
            continue
        got = sorted(tuple(int(v) for v in rc)
                     for rc in entry.get("component_tiles_row_col", []))
        if got != wanted:
            continue
        explanation["detail"] = entry
        for key in ("n_tiles_in_grid", "n_tiles_rise", "n_tiles_fall",
                    "polarity_is_tied", "max_tile_amplitude",
                    "excess_over_median", "per_tile_threshold",
                    "per_tile_median", "per_tile_mad_scaled",
                    "per_tile_signal_before", "per_tile_signal_after",
                    "bracket_source_frames", "proxy_index_in_window"):
            if key in entry:
                explanation[key] = entry[key]
        break
    return explanation


def choose_reference_camera(cluster: dict, window: dict,
                            mode: str = MODE_TILE) -> tuple[str, dict | None]:
    """The camera whose sheet is rendered at legible resolution.

    Chosen by the LARGEST detector signal among the supporting cameras --
    the mean over that camera's firing component in per-tile mode, the
    ``tile_max`` in tile mode -- because that is the camera on which the
    detector actually had something to look at; ties break to the camera
    closest to the cluster's median frame and then alphabetically, so the
    choice is deterministic and re-derivable. This is a presentation choice,
    not a scientific one: nothing downstream depends on which camera was
    picked, and in particular it is not a ranking.
    """
    per_camera = cluster.get("per_camera_source_frame", {})
    median = int(cluster.get("source_frame_median", 0))
    best, best_key, best_expl = None, None, None
    for camera in sorted(cluster.get("cameras", sorted(per_camera))):
        frame = int(per_camera.get(camera, median))
        if mode == MODE_PER_TILE:
            expl = per_tile_explanation_for(window, cluster, camera)
            amplitudes = expl["per_tile_amplitude"] if expl else []
            strength = (float(np.mean(amplitudes)) if amplitudes else -1.0)
        else:
            expl = tile_explanation_for(window, camera, frame)
            strength = float(expl.get("tile_max", 0.0)) if expl else -1.0
        key = (-strength, abs(frame - median), camera)
        if best_key is None or key < best_key:
            best, best_key, best_expl = camera, key, expl
    return best, best_expl


# ---------------------------------------------------------------------------
# Image construction
# ---------------------------------------------------------------------------

def _phase(column_frame: int, member_frame: int, stride: int) -> str:
    """``pre`` before the bracket, ``gap`` inside it, ``post`` after it.

    The bracket is ``[f - stride, f]``: the detector saw the signal on one
    side at ``f - stride`` and on the other at ``f`` and CANNOT say where
    between them the change happened. Calling those two columns ``gap``
    rather than picking one of them is the honest label.
    """
    if column_frame < member_frame - stride:
        return "pre"
    if column_frame > member_frame:
        return "post"
    return "gap"


def _load_frame(entry: dict, source_frame: int) -> Image.Image | None:
    path = entry["frames"].get(int(source_frame))
    if path is None:
        return None
    with Image.open(path) as handle:
        return handle.convert("RGB")


def _labelled_tile(image: Image.Image | None, width: int, height: int,
                   lines: list[str], phase: str, font, small_font) -> Image.Image:
    """One montage cell: the picture, a coloured border, and a label band."""
    band_line = _text_height(small_font)
    band = band_line * len(lines) + 6
    border, text_colour = PHASE_STYLE[phase]
    cell = Image.new("RGB", (width, height + band), (18, 18, 20))
    if image is None:
        placeholder = Image.new("RGB", (width, height), (46, 46, 50))
        draw = ImageDraw.Draw(placeholder)
        draw.text((6, height // 2 - 8), "no proxy frame", font=font,
                  fill=(190, 190, 190))
        cell.paste(placeholder, (0, 0))
    else:
        cell.paste(image.resize((width, height), Image.BILINEAR), (0, 0))
    draw = ImageDraw.Draw(cell)
    draw.rectangle([0, height, width - 1, height + band - 1], fill=border)
    for i, line in enumerate(lines):
        draw.text((4, height + 3 + i * band_line), _drawable(line),
                  font=small_font, fill=text_colour)
    draw.rectangle([0, 0, width - 1, height + band - 1], outline=border, width=2)
    return cell


def _measure(font, text: str) -> float:
    try:
        return float(font.getlength(text))
    except AttributeError:
        return 0.6 * len(text) * _text_height(font)


def _wrap_to_width(font, text: str, limit: int) -> list[str]:
    """Word-wrap for BURNED text. A clipped caveat is a deleted caveat."""
    words = _drawable(text).split()
    if not words:
        return [""]
    lines, current = [], words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if _measure(font, trial) <= limit:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _sheet_header(width: int, lines: list[tuple[str, tuple[int, int, int]]],
                  font, pad: int = 8) -> Image.Image:
    line_h = _text_height(font) + 2
    limit = max(40, width - 2 * pad)
    wrapped: list[tuple[str, tuple[int, int, int]]] = []
    for text, colour in lines:
        for part in _wrap_to_width(font, text, limit):
            wrapped.append((part, colour))
    header = Image.new("RGB", (width, line_h * len(wrapped) + 2 * pad),
                       (14, 14, 16))
    draw = ImageDraw.Draw(header)
    for i, (text, colour) in enumerate(wrapped):
        draw.text((pad, pad + i * line_h), text, font=font, fill=colour)
    return header


def build_montage(entry_by_camera: dict, cameras: list[str],
                  per_camera_frame: dict, anchor: int, stride: int,
                  source_rate: Fraction, *, context_steps: int,
                  tile_width: int, raster: tuple[int, int],
                  header_lines: list[tuple[str, tuple[int, int, int]]],
                  ) -> tuple[Image.Image, list[dict]]:
    """Rows = cameras, columns = a pre / gap / post context series.

    Columns are anchored on the SAME absolute source frames for every row, so
    the rows are directly comparable; the pre/gap/post label is computed per
    row from that row's OWN bracket, so a camera whose changepoint landed one
    proxy step away is labelled truthfully rather than being forced to agree
    with the cluster.
    """
    raster_w, raster_h = raster
    tile_h = max(1, int(round(tile_width * raster_h / raster_w)))
    font = _font(15)
    small = _font(13)
    gutter = 6
    offsets = list(range(-context_steps, context_steps + 1))
    columns = [anchor + k * stride for k in offsets]

    col_head_font = _font(14)
    col_head_h = _text_height(col_head_font) * 2 + 8
    cells: list[list[Image.Image]] = []
    legend: list[dict] = []
    for camera in cameras:
        member = int(per_camera_frame.get(camera, anchor))
        row_cells = []
        for column in columns:
            phase = _phase(column, member, stride)
            image = _load_frame(entry_by_camera[camera], column)
            seconds = frame_time_exact(column, source_rate)
            lines = [
                f"{camera}  src {column}",
                f"t={format_seconds(seconds)}s  {phase.upper()}",
            ]
            row_cells.append(_labelled_tile(
                image, tile_width, tile_h, lines,
                phase if image is not None else "missing", font, small))
            legend.append({
                "camera": camera,
                "source_frame": int(column),
                "t_seconds": round(float(seconds), 6),
                "t_seconds_exact": f"{seconds.numerator}/{seconds.denominator}",
                "phase": phase,
                "present": image is not None,
            })
        cells.append(row_cells)

    cell_w = tile_width + gutter
    cell_h = cells[0][0].height + gutter
    grid_w = cell_w * len(columns) + gutter
    header = _sheet_header(grid_w, header_lines, font)
    sheet = Image.new("RGB", (grid_w, header.height + col_head_h
                              + cell_h * len(cells) + gutter), (14, 14, 16))
    sheet.paste(header, (0, 0))
    draw = ImageDraw.Draw(sheet)
    for c, column in enumerate(columns):
        seconds = frame_time_exact(column, source_rate)
        x = gutter + c * cell_w
        draw.text((x + 2, header.height + 2),
                  f"src {column}  ({offsets[c]:+d} steps)",
                  font=col_head_font, fill=(225, 225, 230))
        draw.text((x + 2, header.height + 2 + _text_height(col_head_font)),
                  f"t={format_seconds(seconds)} s", font=col_head_font,
                  fill=(170, 175, 190))
    for r, row_cells in enumerate(cells):
        for c, cell in enumerate(row_cells):
            sheet.paste(cell, (gutter + c * cell_w,
                               header.height + col_head_h + gutter + r * cell_h))
    return sheet, legend


def build_focus(entry: dict, camera: str, member_frame: int, stride: int,
                source_rate: Fraction, *, tile_width: int,
                raster: tuple[int, int],
                header_lines: list[tuple[str, tuple[int, int, int]]],
                ) -> tuple[Image.Image, list[dict]]:
    """pre / candidate / post at legible resolution, one camera."""
    raster_w, raster_h = raster
    tile_h = max(1, int(round(tile_width * raster_h / raster_w)))
    font = _font(20)
    small = _font(17)
    gutter = 8
    columns = [member_frame - stride, member_frame, member_frame + stride]
    role = ["PRE  (bracket start)", "CANDIDATE  (bracket end)", "POST"]
    cells, legend = [], []
    for column, label in zip(columns, role):
        phase = _phase(column, member_frame, stride)
        image = _load_frame(entry, column)
        seconds = frame_time_exact(column, source_rate)
        lines = [
            f"{camera}   src {column}   t={format_seconds(seconds)} s",
            label,
        ]
        cells.append(_labelled_tile(
            image, tile_width, tile_h, lines,
            phase if image is not None else "missing", font, small))
        legend.append({
            "camera": camera,
            "source_frame": int(column),
            "t_seconds": round(float(seconds), 6),
            "t_seconds_exact": f"{seconds.numerator}/{seconds.denominator}",
            "phase": phase,
            "role": label,
            "present": image is not None,
        })
    cell_w = tile_width + gutter
    grid_w = cell_w * len(cells) + gutter
    header = _sheet_header(grid_w, header_lines, font)
    sheet = Image.new("RGB", (grid_w, header.height + cells[0].height + 2 * gutter),
                      (14, 14, 16))
    sheet.paste(header, (0, 0))
    for i, cell in enumerate(cells):
        sheet.paste(cell, (gutter + i * cell_w, header.height + gutter))
    return sheet, legend


def heat_rgb(normalized: np.ndarray) -> np.ndarray:
    """A perceptual ramp on ``[0, 1]`` without matplotlib."""
    values = np.clip(np.asarray(normalized, dtype=np.float64), 0.0, 1.0)
    out = np.zeros(values.shape + (3,), dtype=np.float64)
    for (lo, c_lo), (hi, c_hi) in zip(_HEAT_STOPS[:-1], _HEAT_STOPS[1:]):
        mask = (values >= lo) & (values <= hi)
        if not mask.any():
            continue
        span = hi - lo
        frac = (values[mask] - lo) / span if span > 0 else np.zeros(mask.sum())
        for ch in range(3):
            out[..., ch][mask] = c_lo[ch] + (c_hi[ch] - c_lo[ch]) * frac
    return out.astype(np.uint8)


def build_overlay(entry: dict, camera: str, explanation: dict,
                  source_rate: Fraction, *, width: int,
                  header_lines: list[tuple[str, tuple[int, int, int]]],
                  ) -> tuple[Image.Image, dict]:
    """Tile heatmap over the candidate frame, with the top-tile boxes drawn.

    EXPLANATION OF THE DETECTOR SIGNAL, NOT AN INSTANCE MASK. The boxes are
    TILE EXTENTS. Nothing here identifies or bounds an object; the caption is
    burned into the picture as well as written beside it, because a picture
    travels further than its caption.
    """
    source_frame = int(explanation["source_frame"])
    base = _load_frame(entry, source_frame)
    raster_w, raster_h = (int(v) for v in explanation["proxy_raster"])
    if base is None:
        base = Image.new("RGB", (raster_w, raster_h), (40, 40, 44))
    base = base.resize((raster_w, raster_h), Image.BILINEAR)

    grid = np.asarray(explanation["tile_template_dist_grid"], dtype=np.float64)
    tile_size = int(explanation["tile_size_px"])
    rows = proxy.tile_edges(raster_h, tile_size)
    cols = proxy.tile_edges(raster_w, tile_size)
    if grid.shape != (len(rows), len(cols)):
        raise ContractError(
            f"{camera} src {source_frame}: tile grid {grid.shape} does not "
            f"match a {tile_size} px tiling of {raster_w}x{raster_h} "
            f"({len(rows)}x{len(cols)}). REFUSED -- drawing it anyway would "
            "put the explanation over the wrong pixels.")
    lo, hi = float(grid.min()), float(grid.max())
    span = hi - lo
    full = np.zeros((raster_h, raster_w), dtype=np.float64)
    for i, (y0, y1) in enumerate(rows):
        for j, (x0, x1) in enumerate(cols):
            full[y0:y1, x0:x1] = (grid[i, j] - lo) / span if span > 0 else 0.0

    heat = Image.fromarray(heat_rgb(full), mode="RGB")
    grey = base.convert("L").convert("RGB")
    blended = Image.blend(grey, heat, 0.55)

    draw = ImageDraw.Draw(blended)
    for y0, _y1 in rows:
        draw.line([(0, y0), (raster_w, y0)], fill=(60, 60, 70), width=1)
    for x0, _x1 in cols:
        draw.line([(x0, 0), (x0, raster_h)], fill=(60, 60, 70), width=1)
    for tile in explanation.get("top_tiles", []):
        x0, y0, x1, y1 = (int(v) for v in tile["pixel_box_xyxy"])
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(120, 220, 255), width=2)
    ax0, ay0, ax1, ay1 = (int(v)
                          for v in explanation["tile_argmax_pixel_box_xyxy"])
    draw.rectangle([ax0, ay0, ax1 - 1, ay1 - 1], outline=(255, 90, 90), width=4)

    scale = width / raster_w
    scaled = blended.resize((width, max(1, int(round(raster_h * scale)))),
                            Image.BILINEAR)
    font = _font(16)
    small = _font(14)
    header = _sheet_header(width, header_lines, font)
    caption_lines = [
        (OVERLAY_HEADING, (255, 190, 190)),
        ("Boxes are TILE EXTENTS, not object extents. Red = argmax tile, "
         "cyan = top contributing tiles.", (215, 215, 225)),
        (f"heat range {lo:.4f} .. {hi:.4f} grey levels (per-tile mean "
         f"|I_t - temporal median|); whole-frame signal at this frame = "
         f"{explanation.get('global_template_dist_at_candidate')}",
         (185, 190, 205)),
    ]
    caption = _sheet_header(width, caption_lines, small)
    sheet = Image.new("RGB", (width, header.height + scaled.height
                              + caption.height), (14, 14, 16))
    sheet.paste(header, (0, 0))
    sheet.paste(scaled, (0, header.height))
    sheet.paste(caption, (0, header.height + scaled.height))
    meta = {
        "camera": camera,
        "source_frame": source_frame,
        "heat_min_grey_levels": round(lo, 4),
        "heat_max_grey_levels": round(hi, 4),
        "tile_size_px": tile_size,
        "grid_shape": [len(rows), len(cols)],
        "is_instance_mask": False,
    }
    return sheet, meta


def build_per_tile_overlay(entry: dict, camera: str, explanation: dict,
                           source_rate: Fraction, *, width: int,
                           tile_size: int, raster: tuple[int, int],
                           header_lines: list[tuple[str, tuple[int, int, int]]],
                           ) -> tuple[Image.Image, dict]:
    """The FIRING CONNECTED COMPONENT drawn over the candidate frame.

    EXPLANATION OF THE DETECTOR SIGNAL, NOT AN INSTANCE MASK. There is no
    argmax to draw in this schema: the detector ran independently on every
    tile and a candidate IS a face-adjacent component of tiles that each
    crossed their OWN threshold. Drawing one tile of it would misrepresent
    the instrument, so the whole component is drawn, its size is stated, and
    the other tiles firing at the same sample -- the ones the coherence gate
    did NOT join to it -- are drawn dimmer so the reader can see how global
    the change was.

    Every pixel box is re-derived from the census's own ``tile_size_px`` and
    proxy raster and CHECKED against the box the census recorded. A mismatch
    is refused rather than drawn: putting the explanation over the wrong
    pixels is the one failure this picture must not have.
    """
    source_frame = int(explanation["source_frame"])
    raster_w, raster_h = (int(v) for v in raster)
    base = _load_frame(entry, source_frame)
    if base is None:
        base = Image.new("RGB", (raster_w, raster_h), (40, 40, 44))
    base = base.resize((raster_w, raster_h), Image.BILINEAR)

    rows = proxy.tile_edges(raster_h, int(tile_size))
    cols = proxy.tile_edges(raster_w, int(tile_size))
    declared_tiles = explanation.get("n_tiles_in_grid")
    if declared_tiles is not None and int(declared_tiles) != len(rows) * len(cols):
        raise _refuse(
            f"{camera} src {source_frame}: the census recorded "
            f"{int(declared_tiles)} tiles in the grid, but a {tile_size} px "
            f"tiling of {raster_w}x{raster_h} has {len(rows)}x{len(cols)} = "
            f"{len(rows) * len(cols)}",
            "the recorded tile count to match the recorded tile size and "
            "proxy raster")

    component = [(int(r), int(c))
                 for r, c in explanation["component_tiles_row_col"]]
    boxes = [[int(v) for v in box]
             for box in explanation["tile_pixel_boxes_xyxy"]]
    amplitudes = [float(v) for v in explanation["per_tile_amplitude"]]
    if not component:
        raise _refuse(f"{camera} src {source_frame}: an EMPTY firing component",
                      "at least one component tile — the coherence gate never "
                      "emits an empty component")
    if len(amplitudes) != len(component):
        raise _refuse(
            f"{camera} src {source_frame}: {len(amplitudes)} per-tile "
            f"amplitudes for {len(component)} component tiles",
            "one amplitude per component tile, in the same order")
    for (row, col), box in zip(component, boxes):
        if not (0 <= row < len(rows) and 0 <= col < len(cols)):
            raise _refuse(
                f"{camera} src {source_frame}: component tile ({row}, {col}) "
                f"outside a {len(rows)}x{len(cols)} grid",
                f"every component tile inside the {tile_size} px tiling of "
                f"{raster_w}x{raster_h}")
        expected = proxy.tile_pixel_box(row, col, rows, cols)
        if list(box) != expected:
            raise _refuse(
                f"{camera} src {source_frame}: the census recorded pixel box "
                f"{box} for tile ({row}, {col}), which a {tile_size} px "
                f"tiling of {raster_w}x{raster_h} puts at {expected}",
                "the recorded boxes to match the recorded tiling — drawing "
                "them anyway would put the explanation over the wrong pixels")

    lo, hi = float(min(amplitudes)), float(max(amplitudes))
    span = hi - lo
    canvas = np.asarray(base.convert("L").convert("RGB"),
                        dtype=np.float64).copy()
    component_set = {(int(r), int(c))
                     for r, c in explanation["component_tiles_row_col"]}
    #  Tiles that fired at this sample but were NOT joined to the component.
    #  They are part of what the coherence gate saw and are never hidden.
    other_firing = []
    for row, col in explanation["firing_tiles_row_col"]:
        row, col = int(row), int(col)
        if (row, col) in component_set:
            continue
        if 0 <= row < len(rows) and 0 <= col < len(cols):
            other_firing.append((row, col))
    for row, col in other_firing:
        x0, y0, x1, y1 = proxy.tile_pixel_box(row, col, rows, cols)
        patch = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = 0.72 * patch + 0.28 * np.array(
            (46, 92, 150), dtype=np.float64)
    for index, (row, col) in enumerate(component):
        x0, y0, x1, y1 = boxes[index]
        #  Ramp over [0.30, 1.0], not [0, 1]: the ramp's own zero is nearly
        #  black, and the WEAKEST tile of a firing component is still a tile
        #  that crossed its threshold. Rendering it as black would read as
        #  "nothing here", which is the opposite of what it means.
        norm = (amplitudes[index] - lo) / span if span > 0 else 1.0
        colour = np.asarray(heat_rgb(np.array([0.30 + 0.70 * norm]))[0],
                            dtype=np.float64)
        patch = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = 0.45 * patch + 0.55 * colour

    blended = Image.fromarray(
        np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(blended)
    for y0, _y1 in rows:
        draw.line([(0, y0), (raster_w, y0)], fill=(60, 60, 70), width=1)
    for x0, _x1 in cols:
        draw.line([(x0, 0), (x0, raster_h)], fill=(60, 60, 70), width=1)
    for row, col in other_firing:
        x0, y0, x1, y1 = proxy.tile_pixel_box(row, col, rows, cols)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(110, 150, 200),
                       width=1)
    for box in boxes:
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(120, 220, 255),
                       width=2)
    bx0, by0, bx1, by1 = explanation["component_pixel_bbox_xyxy"]
    draw.rectangle([int(bx0), int(by0), int(bx1) - 1, int(by1) - 1],
                   outline=(255, 90, 90), width=3)

    scale = width / raster_w
    scaled = blended.resize((width, max(1, int(round(raster_h * scale)))),
                            Image.BILINEAR)
    font = _font(16)
    small = _font(14)
    header = _sheet_header(width, header_lines, font)
    n_component = int(explanation["component_size_tiles"])
    n_firing = int(explanation["n_tiles_firing_at_sample"])
    n_grid = int(declared_tiles) if declared_tiles is not None else len(rows) * len(cols)
    caption_lines = [
        (PER_TILE_OVERLAY_HEADING, (255, 190, 190)),
        ("Boxes are TILE EXTENTS, not object extents. Red = the FIRING "
         "COMPONENT's bounding box (it bounds TILES, not an object), cyan = "
         "the component's own tiles, dim blue = other tiles firing at this "
         "sample that the coherence gate did NOT join to it.",
         (215, 215, 225)),
        (f"firing component {n_component} face-adjacent tiles; "
         f"{n_firing} of {n_grid} tiles firing anywhere in the grid at this "
         f"sample; component tile amplitudes {lo:.4f} .. {hi:.4f} grey levels "
         f"(per-tile mean |I_t - temporal median|), mean "
         f"{float(np.mean(amplitudes)):.4f}", (185, 190, 205)),
    ]
    caption = _sheet_header(width, caption_lines, small)
    sheet = Image.new("RGB", (width, header.height + scaled.height
                              + caption.height), (14, 14, 16))
    sheet.paste(header, (0, 0))
    sheet.paste(scaled, (0, header.height))
    sheet.paste(caption, (0, header.height + scaled.height))
    meta = {
        "camera": camera,
        "source_frame": source_frame,
        "component_size_tiles": n_component,
        "n_tiles_firing_at_sample": n_firing,
        "n_tiles_in_grid": n_grid,
        "amplitude_min_grey_levels": round(lo, 4),
        "amplitude_max_grey_levels": round(hi, 4),
        "amplitude_mean_grey_levels": round(float(np.mean(amplitudes)), 4),
        "tile_size_px": int(tile_size),
        "grid_shape": [len(rows), len(cols)],
        "drawn_from": "firing_connected_component",
        "is_instance_mask": False,
    }
    return sheet, meta


def _save_jpeg(image: Image.Image, path: Path, quality: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=int(quality), optimize=True,
               progressive=False)
    return path.stat().st_size


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def _json_script(obj, element_id: str) -> str:
    """JSON inside a script element, with ``<`` escaped so it cannot close it."""
    payload = json.dumps(obj, indent=2, sort_keys=True).replace("<", "\\u003c")
    return (f'<script type="application/json" id="{element_id}">\n'
            f"{payload}\n</script>")


CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 0 0 6rem 0; background: #f4f4f6; color: #16161a;
  font: 15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
header.top { background: #16161a; color: #f4f4f6; padding: 1.2rem 1.5rem; }
header.top h1 { margin: 0 0 .3rem 0; font-size: 1.35rem; }
main { max-width: 1500px; margin: 0 auto; padding: 0 1.5rem; }
.banner { border-left: 6px solid #b3261e; background: #fff2f0; color: #4a1210;
  padding: .85rem 1rem; margin: 1rem 0; border-radius: 4px; }
.banner.warn { border-left-color: #a76b00; background: #fff8e8; color: #4a3300; }
.banner.info { border-left-color: #1b5e8a; background: #eef6fb; color: #10344c; }
.banner strong { display: block; margin-bottom: .25rem; }
section.cand { background: #fff; border: 1px solid #d7d7dd; border-radius: 6px;
  margin: 1.6rem 0; padding: 1rem 1.2rem; }
section.cand h2 { margin: 0 0 .2rem 0; font-size: 1.15rem; }
table.kv { border-collapse: collapse; font-size: 14px; margin: .6rem 0; }
table.kv td, table.kv th { border: 1px solid #dcdce2; padding: .3rem .6rem;
  text-align: left; vertical-align: top; }
table.kv th { background: #f0f0f4; font-weight: 600; white-space: nowrap; }
table.idx { border-collapse: collapse; font-size: 14px; width: 100%; }
table.idx td, table.idx th { border: 1px solid #dcdce2; padding: .3rem .55rem; }
table.idx th { background: #f0f0f4; text-align: left; }
img.sheet { max-width: 100%; height: auto; display: block; border: 1px solid #c9c9d1;
  border-radius: 3px; margin: .5rem 0; background: #16161a; }
.controls { background: #f7f7fa; border: 1px solid #d7d7dd; border-radius: 5px;
  padding: .8rem 1rem; margin-top: .9rem; }
.controls fieldset { border: 0; margin: 0 0 .6rem 0; padding: 0; }
.controls legend { font-weight: 600; padding: 0; margin-bottom: .3rem; }
.controls label.cls { display: inline-block; margin-right: 1rem; padding: .2rem .5rem;
  border: 1px solid #c2c2cc; border-radius: 4px; background: #fff; cursor: pointer; }
.controls input[type=text], .controls textarea { width: 100%; font: inherit;
  padding: .35rem .5rem; border: 1px solid #c2c2cc; border-radius: 4px; }
.controls .field { margin-bottom: .55rem; }
.controls .field span { display: block; font-size: 13px; color: #55555f;
  margin-bottom: .15rem; }
details { margin: .6rem 0; }
details > summary { cursor: pointer; font-weight: 600; }
pre.raw { background: #16161a; color: #e6e6ea; padding: .7rem; overflow: auto;
  max-height: 26rem; font-size: 12px; border-radius: 4px; }
.caveat { font-size: 13px; color: #4a1210; background: #fff2f0; padding: .5rem .7rem;
  border-left: 4px solid #b3261e; border-radius: 3px; margin: .4rem 0 .8rem 0; }
.export { position: sticky; bottom: 0; background: #16161a; color: #f4f4f6;
  padding: .9rem 1.5rem; border-top: 3px solid #b3261e; }
.export button { font: inherit; padding: .45rem .9rem; margin-right: .6rem;
  border-radius: 4px; border: 1px solid #55555f; background: #2a2a32;
  color: #f4f4f6; cursor: pointer; }
.export textarea { width: 100%; height: 8rem; font: 12px/1.4 ui-monospace,
  Consolas, "Courier New", monospace; margin-top: .5rem; }
code { background: #ececef; padding: .05rem .25rem; border-radius: 3px; }
.small { font-size: 13px; color: #55555f; }
ul.classdef { margin: .4rem 0 .4rem 0; padding-left: 1.2rem; }
ul.classdef li { margin-bottom: .25rem; }
"""


JS = """
(function () {
  var TEMPLATE = JSON.parse(document.getElementById('decisions-template').textContent);
  var KEY = 'imvid-gallery:' + TEMPLATE.gallery_id;

  function loadState() {
    try { return JSON.parse(window.localStorage.getItem(KEY)) || {}; }
    catch (e) { return {}; }
  }
  function saveState(state) {
    try { window.localStorage.setItem(KEY, JSON.stringify(state)); }
    catch (e) { /* private mode / file:// restrictions: the textarea still works */ }
  }
  function payload() {
    var state = loadState();
    var out = JSON.parse(JSON.stringify(TEMPLATE));
    var decided = 0;
    out.decisions.forEach(function (d) {
      var row = state[d.candidate_id] || {};
      d['class'] = row.cls || null;
      d.object_of_interest = row.obj || '';
      d.boundary_notes = row.notes || '';
      if (d['class']) { decided += 1; }
    });
    out.n_decided = decided;
    out.exported_utc = new Date().toISOString();
    return out;
  }
  function refresh() {
    var text = JSON.stringify(payload(), null, 2);
    document.getElementById('decisions-text').value = text;
    document.getElementById('decided-count').textContent =
      payload().n_decided + ' / ' + TEMPLATE.decisions.length + ' classified';
  }
  function record(id, field, value) {
    var state = loadState();
    state[id] = state[id] || {};
    state[id][field] = value;
    saveState(state);
    refresh();
  }

  document.addEventListener('change', function (ev) {
    var el = ev.target;
    if (!el.dataset || !el.dataset.cand) { return; }
    record(el.dataset.cand, el.dataset.field, el.value);
  });
  document.addEventListener('input', function (ev) {
    var el = ev.target;
    if (!el.dataset || !el.dataset.cand) { return; }
    if (el.dataset.field === 'cls') { return; }
    record(el.dataset.cand, el.dataset.field, el.value);
  });

  function restore() {
    var state = loadState();
    Object.keys(state).forEach(function (id) {
      var row = state[id];
      if (row.cls) {
        var radio = document.querySelector(
          'input[type=radio][data-cand="' + id + '"][value="' + row.cls + '"]');
        if (radio) { radio.checked = true; }
      }
      ['obj', 'notes'].forEach(function (field) {
        if (typeof row[field] !== 'string') { return; }
        var el = document.querySelector(
          '[data-cand="' + id + '"][data-field="' + field + '"]');
        if (el) { el.value = row[field]; }
      });
    });
    refresh();
  }

  document.getElementById('download-decisions').addEventListener('click', function () {
    var blob = new Blob([JSON.stringify(payload(), null, 2)],
                        { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = TEMPLATE.gallery_id + '.decisions.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
  });
  document.getElementById('copy-decisions').addEventListener('click', function () {
    var area = document.getElementById('decisions-text');
    area.focus();
    area.select();
    try { document.execCommand('copy'); } catch (e) { /* select-and-copy manually */ }
  });
  document.getElementById('clear-decisions').addEventListener('click', function () {
    if (!window.confirm('Erase every recorded class, object and note in this gallery?')) {
      return;
    }
    try { window.localStorage.removeItem(KEY); } catch (e) { /* nothing stored */ }
    document.querySelectorAll('input[type=radio][data-cand]').forEach(function (el) {
      el.checked = false;
    });
    document.querySelectorAll('[data-cand][data-field="obj"], [data-cand][data-field="notes"]')
      .forEach(function (el) { el.value = ''; });
    refresh();
  });

  restore();
}());
"""


def _kv_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><th>{_esc(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="kv">{body}</table>'


def _controls_html(candidate_id: str) -> str:
    radios = "".join(
        f'<label class="cls"><input type="radio" name="cls-{candidate_id}" '
        f'data-cand="{candidate_id}" data-field="cls" value="{choice}"> '
        f'{_esc(choice)}</label>'
        for choice in CLASS_CHOICES)
    definitions = "".join(
        f"<li><strong>{_esc(name)}</strong> — {_esc(text)}</li>"
        for name, text in CLASS_DEFINITIONS)
    return f"""<div class="controls">
<fieldset><legend>Provisional class for {_esc(candidate_id)}</legend>{radios}
<ul class="classdef small">{definitions}</ul>
<div class="small">{_esc(CLASS_FRAME_NOTE)}</div></fieldset>
<div class="field"><span>Object of interest (free text)</span>
<input type="text" data-cand="{candidate_id}" data-field="obj"
 placeholder="e.g. the tray the cook lifts"></div>
<div class="field"><span>Boundary notes (what happens, and where you think the
onset / offset really is in SOURCE frames)</span>
<textarea rows="3" data-cand="{candidate_id}" data-field="notes"
 placeholder="e.g. occluder enters around src 1440; fully clear by src 1500"></textarea></div>
</div>"""


def _banner(kind: str, title: str, body: str) -> str:
    return (f'<div class="banner {kind}"><strong>{_esc(title)}</strong>'
            f"{_esc(body)}</div>")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_gallery(*, census_path: Path, proxy_root: Path, out_dir: Path,
                  scene: str | None = None,
                  max_candidates: int = DEFAULT_MAX_CANDIDATES,
                  max_cameras_per_candidate: int = DEFAULT_MAX_CAMERAS_PER_CANDIDATE,
                  max_montage_pages: int = DEFAULT_MAX_MONTAGE_PAGES,
                  context_steps: int = DEFAULT_CONTEXT_STEPS,
                  montage_tile_width: int = DEFAULT_MONTAGE_TILE_WIDTH,
                  focus_tile_width: int = DEFAULT_FOCUS_TILE_WIDTH,
                  overlay_width: int = DEFAULT_OVERLAY_WIDTH,
                  jpeg_quality: int = DEFAULT_JPEG_QUALITY,
                  max_bytes: int = DEFAULT_MAX_BYTES,
                  clean: bool = False) -> dict:
    census = load_census(census_path)
    #  WHICH SCHEMA IS THIS? Decided once, loudly, before anything renders.
    mode = detect_census_mode(census)
    cameras_on_disk = load_proxy_tree(proxy_root)
    scene_manifest_path, scene_manifest_hash = scene_manifest_sha256(proxy_root)

    mapping = census.get("mapping", {})
    source_rate = proxy.parse_rational(mapping["source_rate_exact"])
    stride = int(mapping["stride_frames"])
    raster = tuple(int(v) for v in mapping.get("proxy_raster", [960, 540]))
    per_tile_tile_size = (int(census["per_tile_pass"]["tile_size_px"])
                          if mode == MODE_PER_TILE else None)

    if out_dir.exists() and clean:
        shutil.rmtree(out_dir)
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    ranked = flatten_candidates(census, mode)
    n_total = len(ranked)
    kept = ranked[:max(int(max_candidates), 0)]
    dropped_by_count = n_total - len(kept)

    census_hash = _sha256(census_path)
    gallery_id = f"{scene or Path(census.get('proxy_root', proxy_root)).name}-{census_hash[:12]}"

    drops = {
        "candidates_dropped_by_max_candidates": int(dropped_by_count),
        "candidates_dropped_by_size_budget": 0,
        "camera_rows_dropped_by_page_cap": 0,
        "candidates_with_dropped_cameras": [],
        "candidates_without_tile_explanation": [],
        "cameras_missing_from_proxy_tree": [],
    }

    rendered: list[dict] = []
    image_count = 0
    total_bytes = 0

    for row in kept:
        cluster = row["cluster"]
        window = row["window"]
        candidate_id = row["candidate_id"]
        per_camera_frame = {c: int(f) for c, f in
                            cluster.get("per_camera_source_frame", {}).items()}
        supporting = list(cluster.get("cameras", sorted(per_camera_frame)))
        median = int(cluster.get("source_frame_median", 0))

        present = [c for c in supporting if c in cameras_on_disk]
        missing = [c for c in supporting if c not in cameras_on_disk]
        if missing:
            drops["cameras_missing_from_proxy_tree"].append(
                {"candidate_id": candidate_id, "cameras": missing})
        if not present:
            drops["candidates_without_tile_explanation"].append(candidate_id)
            continue

        # Cameras nearest the cluster centre first, so page 1 is the most
        # representative page when the page cap bites.
        present.sort(key=lambda c: (abs(per_camera_frame.get(c, median) - median), c))
        anchor_camera = present[0]
        anchor = int(per_camera_frame.get(anchor_camera, median))

        per_page = max(int(max_cameras_per_candidate), 1)
        pages = [present[i:i + per_page] for i in range(0, len(present), per_page)]
        n_pages_possible = len(pages)
        if max_montage_pages > 0 and n_pages_possible > max_montage_pages:
            shown = pages[:max_montage_pages]
            dropped_rows = sum(len(p) for p in pages[max_montage_pages:])
            drops["camera_rows_dropped_by_page_cap"] += dropped_rows
            drops["candidates_with_dropped_cameras"].append({
                "candidate_id": candidate_id,
                "cameras_supporting": len(present),
                "cameras_shown": sum(len(p) for p in shown),
                "cameras_dropped": dropped_rows,
                "dropped_cameras": [c for p in pages[max_montage_pages:] for c in p],
            })
            pages = shown

        polarity = cluster.get("polarity")
        bracket = [anchor - stride, anchor]
        t_median = frame_time_exact(median, source_rate)
        t_bracket = [frame_time_exact(bracket[0], source_rate),
                     frame_time_exact(bracket[1], source_rate)]

        head_common = [
            (f"{candidate_id}  |  polarity {polarity}  |  cluster median src "
             f"{median}  (t={format_seconds(t_median)} s @ "
             f"{proxy.rational_str(source_rate)} fps)", (240, 240, 245)),
            (f"{cluster.get('n_cameras_supporting')} supporting cameras  |  "
             f"spread {cluster.get('spread_frames')} frames "
             f"({float(cluster.get('spread_ms', 0.0)):.1f} ms) — "
             f"{SPREAD_CAVEAT}", (200, 205, 220)),
            ("CANDIDATE, NOT AN EVENT. Polarity is a signal direction, not a "
             "semantic.", (255, 175, 175)),
        ]
        if mode == MODE_PER_TILE:
            head_common.insert(2, (
                f"mean amplitude {float(cluster['mean_amplitude']):.4f} grey "
                f"levels — RANKED BY THIS. The camera count above orders "
                f"nothing.", (200, 205, 220)))

        montage_files = []
        for page_index, page_cameras in enumerate(pages, start=1):
            head = list(head_common)
            head.insert(2, (f"montage page {page_index} of {len(pages)}  |  "
                            f"rows = cameras, columns = context at "
                            f"{stride}-frame proxy steps", (200, 205, 220)))
            sheet, legend = build_montage(
                cameras_on_disk, page_cameras, per_camera_frame, anchor,
                stride, source_rate, context_steps=context_steps,
                tile_width=montage_tile_width, raster=raster,
                header_lines=head)
            name = f"{candidate_id}_montage_p{page_index}.jpg"
            size = _save_jpeg(sheet, assets / name, jpeg_quality)
            total_bytes += size
            image_count += 1
            montage_files.append({"file": name, "page": page_index,
                                  "cameras": page_cameras, "bytes": size,
                                  "legend": legend})

        reference_camera, explanation = choose_reference_camera(
            cluster, window, mode)
        if reference_camera not in cameras_on_disk:
            reference_camera = anchor_camera
            explanation = (
                per_tile_explanation_for(window, cluster, reference_camera)
                if mode == MODE_PER_TILE else
                tile_explanation_for(
                    window, reference_camera,
                    per_camera_frame.get(reference_camera, median)))
        focus_frame = int(per_camera_frame.get(reference_camera, median))
        focus_head = list(head_common)
        focus_head.insert(2, (f"reference camera {reference_camera} "
                              f"(largest tile signal among supporting cameras)",
                              (200, 205, 220)))
        focus_sheet, focus_legend = build_focus(
            cameras_on_disk[reference_camera], reference_camera, focus_frame,
            stride, source_rate, tile_width=focus_tile_width, raster=raster,
            header_lines=focus_head)
        focus_name = f"{candidate_id}_focus.jpg"
        size = _save_jpeg(focus_sheet, assets / focus_name, jpeg_quality)
        total_bytes += size
        image_count += 1

        overlay_name, overlay_meta = None, None
        if explanation is not None:
            heading = (PER_TILE_OVERLAY_HEADING if mode == MODE_PER_TILE
                       else OVERLAY_HEADING)
            overlay_head = [
                (f"{candidate_id}  |  {reference_camera}  |  src "
                 f"{explanation['source_frame']}  t="
                 f"{format_seconds(frame_time_exact(int(explanation['source_frame']), source_rate))} s",
                 (240, 240, 245)),
                (heading, (255, 175, 175)),
            ]
            if mode == MODE_PER_TILE:
                overlay_sheet, overlay_meta = build_per_tile_overlay(
                    cameras_on_disk[reference_camera], reference_camera,
                    explanation, source_rate, width=overlay_width,
                    tile_size=per_tile_tile_size, raster=raster,
                    header_lines=overlay_head)
            else:
                overlay_sheet, overlay_meta = build_overlay(
                    cameras_on_disk[reference_camera], reference_camera,
                    explanation, source_rate, width=overlay_width,
                    header_lines=overlay_head)
            overlay_name = f"{candidate_id}_overlay.jpg"
            size = _save_jpeg(overlay_sheet, assets / overlay_name, jpeg_quality)
            total_bytes += size
            image_count += 1
        else:
            drops["candidates_without_tile_explanation"].append(candidate_id)

        rendered.append({
            "candidate_id": candidate_id,
            "window_source_frames": row["window_source_frames"],
            "cluster": cluster,
            "polarity": polarity,
            "median": median,
            "anchor": anchor,
            "bracket": bracket,
            "t_median": t_median,
            "t_bracket": t_bracket,
            "montages": montage_files,
            "focus": {"file": focus_name, "camera": reference_camera,
                      "source_frame": focus_frame, "legend": focus_legend},
            "overlay": ({"file": overlay_name, "meta": overlay_meta,
                         "explanation": explanation}
                        if overlay_name else None),
            "cameras_present": present,
            "cameras_missing": missing,
        })

        if max_bytes > 0 and total_bytes > max_bytes:
            remaining = len(kept) - len(rendered)
            if remaining > 0:
                drops["candidates_dropped_by_size_budget"] = int(remaining)
            break

    manifest = _write_outputs(
        out_dir=out_dir, census=census, census_path=census_path, mode=mode,
        census_hash=census_hash, proxy_root=proxy_root,
        scene_manifest_path=scene_manifest_path,
        scene_manifest_hash=scene_manifest_hash, scene=scene,
        gallery_id=gallery_id, rendered=rendered, drops=drops,
        n_total_candidates=n_total, image_count=image_count,
        total_bytes=total_bytes, source_rate=source_rate, stride=stride,
        raster=raster,
        settings={
            "max_candidates": int(max_candidates),
            "max_cameras_per_candidate": int(max_cameras_per_candidate),
            "max_montage_pages": int(max_montage_pages),
            "context_steps": int(context_steps),
            "montage_tile_width": int(montage_tile_width),
            "focus_tile_width": int(focus_tile_width),
            "overlay_width": int(overlay_width),
            "jpeg_quality": int(jpeg_quality),
            "max_bytes": int(max_bytes),
        })
    return manifest


def _declared_scale_rows(census: dict, raster: tuple[int, int],
                         mode: str = MODE_TILE) -> list[tuple[str, str]]:
    """What the census could and could not have seen. Recorded numbers only."""
    params = census.get("parameters", {})
    tile = census.get("tile_pass") or {}
    rows = [
        ("window_frames", _esc(params.get("window_frames"))),
        ("k_mad (robust relative gate)", _esc(params.get("k_mad"))),
        ("whole-frame absolute floor",
         f"{_esc(params.get('min_amplitude_grey_levels'))} grey levels"),
        ("signal", _esc(params.get("signal"))),
        ("cross-camera match tolerance",
         f"{_esc(params.get('match_tol_frames'))} frames "
         f"({float(params.get('match_tol_ms') or 0.0):.2f} ms)"),
    ]
    if mode == MODE_PER_TILE:
        block = census["per_tile_pass"]
        floor = block["absolute_floor"]
        coherence = block["spatial_coherence"]
        grid = block["tile_grid"]
        rows.append((
            "detection reduction",
            _esc(f"PER-TILE (v2). The SAME two-gate detector runs "
                 f"INDEPENDENTLY on each of {grid[0]}x{grid[1]} per-tile "
                 f"signals of {block['tile_size_px']} proxy px. This "
                 f"SUPERSEDES the max-over-tiles (tile_max) reduction: a "
                 f"maximum over tiles is monopolised by the loudest region, "
                 f"which sets both its median and its MAD, so a quiet tile "
                 f"can never move a maximum it does not win.")))
        rows.append((
            "relative gate (per tile)",
            _esc(f"median_t(S_ij) + {block['k_mad']} * 1.4826 * MAD_t(S_ij), "
                 f"from THAT TILE'S OWN temporal series")))
        rows.append((
            "per-tile absolute floor (MEASURED)",
            _esc(f"{floor['floor_grey_levels']} grey levels on a TILE mean = "
                 f"F={floor['F']} x the take's own median per-tile noise "
                 f"scale {floor['noise_scale_median_grey_levels']}")))
        rows.append((
            "spatial coherence",
            _esc(f"a face-adjacent connected component of >= "
                 f"{coherence['min_component_tiles']} tiles (4-connectivity; "
                 f"DIAGONAL tiles are NOT adjacent) AND strictly fewer than "
                 f"{coherence['max_firing_tiles_exclusive']} tiles firing "
                 f"anywhere in the grid at that sample. Both bounds are "
                 f"DECLARED JUDGMENTS derived from observation of a "
                 f"known-positive and a known-negative, and that dependence "
                 f"is disclosed.")))
        rows.append((
            "declared detection scale",
            _esc(f"a change confined to fewer than "
                 f"{coherence['min_component_tiles']} face-adjacent "
                 f"{block['tile_size_px']}x{block['tile_size_px']} px tiles, "
                 f"or one that moves no tile mean by "
                 f"{floor['floor_grey_levels']} grey levels above that "
                 f"tile's own median+MAD threshold, is INVISIBLE to this "
                 f"census by construction, and its absence from this gallery "
                 f"says nothing about the scene.")))
        rows.append((
            "ranking",
            _esc(f"{block['ranking']['order']}. "
                 f"camera_count_is_retired_as_the_primary_order="
                 f"{block['ranking']['camera_count_is_retired_as_the_primary_order']}, "
                 f"c_min_is_a_corroboration_floor_never_a_sort_key="
                 f"{block['ranking']['c_min_is_a_corroboration_floor_never_a_sort_key']}")))
        return rows
    if tile.get("enabled"):
        tile_px = int(tile.get("tile_size_px", 0))
        floor = float(tile.get("tile_min_amplitude_grey_levels", 0.0))
        rows.append(("tile pass", f"ON — {_esc(tile.get('tile_grid'))} "
                                  f"{_esc(tile.get('tile_grid_order'))} tiles of "
                                  f"{tile_px} proxy px"))
        rows.append(("tile absolute floor", f"{floor} grey levels on a TILE mean"))
        if tile_px > 0 and raster[0] > 0 and raster[1] > 0:
            equivalent = floor * (tile_px * tile_px) / (raster[0] * raster[1])
            rows.append((
                "declared detection scale",
                _esc(f"a change confined to one {tile_px}x{tile_px} px tile must move "
                     f"that tile's mean by >= {floor} grey levels to be screened. The "
                     f"same change moves the whole {raster[0]}x{raster[1]} frame mean by "
                     f"{equivalent:.4f} grey levels, which the whole-frame floor of "
                     f"{params.get('min_amplitude_grey_levels')} rejects. Anything "
                     f"smaller or fainter than that is INVISIBLE to this census and "
                     f"its absence from this gallery says nothing about the scene.")))
    else:
        rows.append(("tile pass", "OFF — detection ran on WHOLE-FRAME MEANS, "
                                  "which cannot see a small object"))
    return rows


def _ranking_order_string(mode: str) -> str:
    if mode == MODE_PER_TILE:
        return ("mean_amplitude DESCENDING, then source_frame_median "
                "ascending. n_cameras_supporting IS NOT A SORT KEY.")
    return ("n_cameras_supporting DESCENDING, then mean_amplitude "
            "descending, then source_frame_median ascending (the v1 / "
            "whole-frame census's own printed order)")


def _write_outputs(*, out_dir: Path, census: dict, census_path: Path,
                   mode: str, census_hash: str, proxy_root: Path,
                   scene_manifest_path: str | None,
                   scene_manifest_hash: str | None, scene: str | None,
                   gallery_id: str, rendered: list[dict], drops: dict,
                   n_total_candidates: int, image_count: int,
                   total_bytes: int, source_rate: Fraction, stride: int,
                   raster: tuple[int, int], settings: dict) -> dict:
    scene_name = scene or Path(census.get("proxy_root", str(proxy_root))).name

    decisions_template = {
        "schema": DECISIONS_SCHEMA,
        "gallery_id": gallery_id,
        "scene": scene_name,
        "census_path": str(census_path),
        "census_sha256": census_hash,
        "proxy_manifest_sha256": scene_manifest_hash,
        "source_rate_exact": proxy.rational_str(source_rate),
        "exported_utc": None,
        "n_candidates_in_gallery": len(rendered),
        "n_candidates_in_census": int(n_total_candidates),
        "n_decided": 0,
        "class_choices": list(CLASS_CHOICES),
        "class_definitions": {name: text for name, text in CLASS_DEFINITIONS},
        "class_definitions_source": (
            "research-wiki/operations/imvid-event-definition-2026-08-24.md §3"),
        "class_frame_note": CLASS_FRAME_NOTE,
        "census_detection_mode": mode,
        "ranked_by": _ranking_order_string(mode),
        "reading_rule": (
            "class / object_of_interest / boundary_notes are a HUMAN "
            "provisional judgement recorded against a CANDIDATE. They are not "
            "ground truth and this file establishes no event."),
        "decisions": [
            {
                "candidate_id": item["candidate_id"],
                "window_source_frames": item["window_source_frames"],
                "polarity": item["polarity"],
                "source_frame_median": int(item["median"]),
                "bracket_source_frames": [int(v) for v in item["bracket"]],
                "t_seconds": round(float(item["t_median"]), 6),
                "t_seconds_exact": (f"{item['t_median'].numerator}/"
                                    f"{item['t_median'].denominator}"),
                "n_cameras_supporting": int(
                    item["cluster"].get("n_cameras_supporting", 0)),
                "cameras": list(item["cluster"].get("cameras", [])),
                "reference_camera": item["focus"]["camera"],
                "mean_amplitude": item["cluster"].get("mean_amplitude"),
                "component_size_tiles": (
                    (item["overlay"]["meta"] or {}).get("component_size_tiles")
                    if item["overlay"] else None),
                "n_tiles_firing_at_sample": (
                    (item["overlay"]["meta"] or {}).get(
                        "n_tiles_firing_at_sample")
                    if item["overlay"] else None),
                "class": None,
                "object_of_interest": "",
                "boundary_notes": "",
            }
            for item in rendered
        ],
    }

    html_text = _render_html(
        census=census, census_path=census_path, mode=mode,
        census_hash=census_hash,
        proxy_root=proxy_root, scene_manifest_path=scene_manifest_path,
        scene_manifest_hash=scene_manifest_hash, scene_name=scene_name,
        gallery_id=gallery_id, rendered=rendered, drops=drops,
        n_total_candidates=n_total_candidates, source_rate=source_rate,
        stride=stride, raster=raster, settings=settings,
        decisions_template=decisions_template)
    index = out_dir / "index.html"
    index.write_text(html_text, encoding="utf-8")
    html_bytes = index.stat().st_size

    manifest = {
        "schema": GALLERY_SCHEMA,
        "gallery_id": gallery_id,
        "scene": scene_name,
        "census_detection_mode": mode,
        "ranking": {
            "order": _ranking_order_string(mode),
            "camera_count_is_a_sort_key": mode != MODE_PER_TILE,
            "note": (RANKING_NOTE_PER_TILE if mode == MODE_PER_TILE
                     else RANKING_NOTE_LEGACY),
        },
        "per_tile_floor_provenance": (
            census["per_tile_pass"]["absolute_floor"]
            if mode == MODE_PER_TILE else None),
        "class_definitions": {name: text for name, text in CLASS_DEFINITIONS},
        "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"),
        "census_path": str(census_path),
        "census_sha256": census_hash,
        "proxy_root": str(proxy_root),
        "proxy_scene_manifest_path": scene_manifest_path,
        "proxy_manifest_sha256": scene_manifest_hash,
        "source_rate_exact": proxy.rational_str(source_rate),
        "stride_frames": int(stride),
        "proxy_raster": list(raster),
        "n_candidates_in_census": int(n_total_candidates),
        "n_candidates_rendered": len(rendered),
        "dropped": drops,
        "n_images": int(image_count),
        "bytes_images": int(total_bytes),
        "bytes_index_html": int(html_bytes),
        "bytes_total": int(total_bytes + html_bytes),
        "settings": settings,
        "decisions_schema": DECISIONS_SCHEMA,
        "instrument_status": "SCOUTING INSTRUMENT, NOT GROUND TRUTH",
        "disclaimer": NOT_AN_EVENT,
        "overlay_status": OVERLAY_CAVEAT,
    }
    (out_dir / GALLERY_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _class_legend_html() -> str:
    """The FROZEN A / B / C definitions. Not the reviewer's own tiers."""
    items = "".join(
        f"<li><strong>{_esc(name)}</strong> — {_esc(text)}</li>"
        for name, text in CLASS_DEFINITIONS)
    return (f'<div class="banner info"><strong>{_esc(CLASS_LEGEND_TITLE)}'
            f"</strong>These four classes are fixed by "
            f"<code>imvid-event-definition-2026-08-24</code> §3 and only "
            f"class A is an event. This tool does not define them, does not "
            f"rank them, and does not act on them."
            f'<ul class="classdef">{items}</ul>'
            f"<div>{_esc(CLASS_FRAME_NOTE)}</div></div>")


def _per_tile_floor_html(census: dict) -> str:
    """The MEASURED floor, F=3.0 marked PRIMARY, and the sweep as probes."""
    block = census["per_tile_pass"]
    floor = block["absolute_floor"]
    parts: list[str] = []
    if floor["floor_is_degenerate_zero"]:
        parts.append(_banner(
            "", "THE MEASURED PER-TILE NOISE SCALE MEDIAN IS ZERO — THE "
                "ABSOLUTE GATE IS INERT",
            f"The v2 absolute floor is F x the take's OWN median per-tile "
            f"noise scale, and that median measured 0.0 over "
            f"{floor['n_tile_series']} (camera, window, tile) series "
            f"({floor['n_zero_scale_tile_series']} of them are themselves "
            f"exactly zero). The floor is therefore "
            f"{floor['floor_grey_levels']} and the absolute gate SCREENS "
            f"NOTHING: every candidate below was decided by the relative gate "
            f"and the spatial-coherence gate alone. This is the vacuity "
            f"failure the v2 spec was written to catch — read no candidate "
            f"here as having cleared an absolute floor."))
    if not block.get("is_primary_reading", True):
        parts.append(_banner(
            "", "THIS CENSUS WAS RUN AT A SENSITIVITY PROBE, NOT AT THE "
                "PRIMARY READING",
            f"The frozen primary reading is F = "
            f"{block['primary_reading_F']} ONLY. This manifest was produced "
            f"at F = {floor['F']}, which the spec labels a sensitivity probe "
            f"and forbids reporting as the census."))
    primary = "PRIMARY READING" if block.get("is_primary_reading") else "PROBE"
    parts.append("<h2>Per-tile absolute floor — provenance</h2>")
    parts.append('<p class="small">The floor is <code>F * median over all '
                 '(camera, window, tile) of 1.4826 * MAD_t(S_ij)</code>, '
                 'measured on the take&#39;s own noise BEFORE any candidate '
                 'was read. <strong>F is a DECLARED JUDGMENT, not a quantity '
                 'derived from data</strong>; it is fixed at 3.0 to match the '
                 'relative gate&#39;s k_mad = 3.0 so the two gates express '
                 'the same strictness in different denominators.</p>')
    parts.append(_kv_table([
        ("measured noise scale median",
         f"{_esc(floor['noise_scale_median_grey_levels'])} grey levels, over "
         f"{_esc(floor['n_tile_series'])} (camera, window, tile) series "
         f"from {_esc(floor.get('n_camera_window_grids', 'n/a'))} grids"),
        ("F (declared judgment)",
         f"<strong>{_esc(floor['F'])}</strong> — {_esc(primary)}"
         f" (the frozen primary reading is F = "
         f"{_esc(block['primary_reading_F'])})"),
        ("floor_grey_levels (MEASURED)",
         f"<strong>{_esc(floor['floor_grey_levels'])}</strong> grey levels on "
         f"a TILE mean"),
        ("noise scale range",
         f"{_esc(floor.get('noise_scale_min_grey_levels'))} .. "
         f"{_esc(floor.get('noise_scale_max_grey_levels'))} grey levels"),
        ("tile series with EXACTLY ZERO noise scale",
         _esc(floor["n_zero_scale_tile_series"])),
        ("floor_is_degenerate_zero",
         f"<strong>{_esc(floor['floor_is_degenerate_zero'])}</strong>"),
        ("rule", f"<code>{_esc(floor.get('rule', ''))}</code>"),
    ]))
    sweep = block["sensitivity_sweep"]
    head = ("<tr><th>F</th><th>floor (grey levels)</th>"
            "<th>per-camera candidates</th><th>rejected by coherence</th>"
            "<th>clusters</th><th>clusters at C_min</th><th>reading</th></tr>")
    rows = []
    for row in sweep:
        label = ("PRIMARY READING — this row IS the census"
                 if row.get("is_primary_reading")
                 else "SENSITIVITY PROBE — NOT the census")
        strong = "<strong>%s</strong>" if row.get("is_primary_reading") else "%s"
        rows.append(
            "<tr>"
            f"<td>{strong % _esc(row['F'])}</td>"
            f"<td>{_esc('%.6f' % float(row['floor_grey_levels']))}</td>"
            f"<td>{_esc(row.get('n_per_camera_candidates'))}</td>"
            f"<td>{_esc(row.get('n_rejected_by_spatial_coherence'))}</td>"
            f"<td>{_esc(row.get('n_clusters'))}</td>"
            f"<td>{_esc(row.get('n_clusters_at_min_cameras'))}</td>"
            f"<td>{strong % _esc(label)}</td></tr>")
    parts.append("<h3>Sensitivity sweep over F — SUPPLEMENTARY</h3>")
    parts.append(f'<div class="caveat">'
                 f'{_esc(block.get("sensitivity_sweep_is_supplementary", ""))}'
                 f"</div>")
    parts.append(f'<table class="idx">{head}{"".join(rows)}</table>')
    defect = block.get("windowing_defect_not_repaired_here")
    if defect:
        parts.append(_banner(
            "warn", "A LIVE WINDOWING DEFECT STANDS — NO CLAIM OF EXHAUSTIVE "
                    "RECALL MAY BE MADE", defect))
    return "".join(parts)


def _drop_banners(drops: dict, n_total: int, n_rendered: int,
                  mode: str = MODE_TILE) -> str:
    """No silent caps: anything dropped is stated at the top of the page."""
    parts = []
    by_count = int(drops["candidates_dropped_by_max_candidates"])
    by_size = int(drops["candidates_dropped_by_size_budget"])
    rows = int(drops["camera_rows_dropped_by_page_cap"])
    if by_count or by_size:
        parts.append(_banner(
            "warn", "CANDIDATES DROPPED — THIS GALLERY IS NOT THE WHOLE CENSUS",
            f"The census holds {n_total} candidate clusters; {n_rendered} are "
            f"rendered here. {by_count} were dropped by --max-candidates and "
            f"{by_size} by the bundle size budget. Dropped candidates were NOT "
            f"reviewed and NOT rejected; raise the cap and rebuild to see them."))
    if rows:
        detail = ", ".join(
            f"{item['candidate_id']} ({item['cameras_dropped']} of "
            f"{item['cameras_supporting']})"
            for item in drops["candidates_with_dropped_cameras"][:20])
        parts.append(_banner(
            "warn", "SUPPORTING CAMERAS DROPPED FROM SOME MONTAGES",
            f"{rows} camera rows were dropped by the montage page cap: "
            f"{detail}. Those cameras still support the candidate in the "
            f"census; only their pictures are missing here."))
    if drops["cameras_missing_from_proxy_tree"]:
        detail = ", ".join(
            f"{item['candidate_id']}: {', '.join(item['cameras'])}"
            for item in drops["cameras_missing_from_proxy_tree"][:20])
        parts.append(_banner(
            "warn", "SUPPORTING CAMERAS ABSENT FROM THE PROXY TREE",
            f"The census names cameras that this --proxy-root does not "
            f"contain: {detail}. Either the census was run against a different "
            f"proxy root, or the proxy set is incomplete."))
    missing_expl = [c for c in drops["candidates_without_tile_explanation"]]
    if missing_expl:
        cause = {
            MODE_PER_TILE: ("Every per-tile candidate carries its firing "
                            "component, so this is NOT the ordinary "
                            "no-tile-pass case: the supporting cameras are "
                            "absent from this --proxy-root."),
            MODE_TILE: ("The census names cameras this --proxy-root does not "
                        "contain, or the tile explanation is missing for "
                        "them."),
            MODE_WHOLE_FRAME: ("The census was run WITHOUT --per-tile-mode "
                               "and without --tile-mode, so no spatial "
                               "explanation exists to draw."),
        }[mode]
        parts.append(_banner(
            "info", "NO SPATIAL EXPLANATION FOR SOME CANDIDATES",
            f"{len(missing_expl)} candidate(s) carry no tile explanation "
            f"({', '.join(missing_expl[:20])}). {cause} The montage and focus "
            f"sheets are still complete."))
    if not parts:
        parts.append(_banner(
            "info", "NOTHING WAS DROPPED",
            f"All {n_total} candidate clusters in the census are rendered "
            f"here, with every supporting camera."))
    return "".join(parts)


def _render_html(*, census: dict, census_path: Path, mode: str,
                 census_hash: str,
                 proxy_root: Path, scene_manifest_path: str | None,
                 scene_manifest_hash: str | None, scene_name: str,
                 gallery_id: str, rendered: list[dict], drops: dict,
                 n_total_candidates: int, source_rate: Fraction, stride: int,
                 raster: tuple[int, int], settings: dict,
                 decisions_template: dict) -> str:
    proxy_step_ms = float(census.get("mapping", {}).get("proxy_step_ms", 0.0))
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, '
                 'initial-scale=1">')
    parts.append(f"<title>ImViD candidate review — {_esc(scene_name)}</title>")
    parts.append(f"<style>{CSS}</style>")
    parts.append("</head><body>")

    parts.append('<header class="top">')
    parts.append(f"<h1>ImViD event-census review gallery — "
                 f"{_esc(scene_name)}</h1>")
    parts.append(f'<div class="small" style="color:#c9c9d4">gallery '
                 f'<code>{_esc(gallery_id)}</code> · '
                 f'{len(rendered)} of {n_total_candidates} candidate clusters '
                 f'· source rate {_esc(proxy.rational_str(source_rate))} fps '
                 f'· proxy step {stride} frames ({proxy_step_ms:.2f} ms) '
                 f'· proxy raster {raster[0]}x{raster[1]} '
                 f'· census schema <code>{_esc(mode)}</code></div>')
    parts.append("</header><main>")

    parts.append(_banner("", "THESE ARE CANDIDATES, NOT EVENTS", NOT_AN_EVENT))
    parts.append(_banner(
        "info",
        ("HOW THIS GALLERY IS ORDERED — BY MEAN AMPLITUDE"
         if mode == MODE_PER_TILE
         else "HOW THIS GALLERY IS ORDERED"),
        (RANKING_NOTE_PER_TILE if mode == MODE_PER_TILE
         else RANKING_NOTE_LEGACY)))
    parts.append(_class_legend_html())
    parts.append(_banner(
        "info", "HOW TO READ THE TIMESTAMPS",
        f"Every timestamp on every picture is t = source_frame * "
        f"{source_rate.denominator} / {source_rate.numerator} seconds, from "
        f"the exact rational rate recorded in the proxy manifests. No frame "
        f"rate is assumed anywhere in this bundle."))
    parts.append(_drop_banners(drops, n_total_candidates, len(rendered), mode))

    parts.append("<h2>Census settings and declared detection scale</h2>")
    parts.append(_kv_table(_declared_scale_rows(census, raster, mode)))
    note = census.get("temporal_resolution_note")
    if note:
        parts.append(f'<div class="caveat">{_esc(note)}</div>')
    if mode == MODE_PER_TILE:
        parts.append(_per_tile_floor_html(census))
    parts.append(_kv_table([
        ("census file", f"<code>{_esc(census_path)}</code>"),
        ("census sha256", f"<code>{_esc(census_hash)}</code>"),
        ("proxy root", f"<code>{_esc(proxy_root)}</code>"),
        ("proxy scene manifest",
         f"<code>{_esc(scene_manifest_path or 'ABSENT')}</code>"),
        ("proxy manifest sha256",
         f"<code>{_esc(scene_manifest_hash or 'ABSENT')}</code>"),
        ("gallery settings",
         f"<code>{_esc(json.dumps(settings, sort_keys=True))}</code>"),
    ]))

    if not rendered:
        if int(n_total_candidates) == 0:
            parts.append(_banner(
                "warn", "THE CENSUS RETURNED ZERO CANDIDATES",
                "Zero is a RESULT, not an error and not a failure of this "
                "tool. The census ran and screened no changepoint that cleared "
                "both its relative gate and its absolute floor. That is a "
                "statement about THIS INSTRUMENT AT THIS SCALE ON THIS PROXY, "
                "and nothing more: it is NOT evidence that the scene contains "
                "no occlusion, no reveal and no disappearance. Anything below "
                "the declared detection scale above is invisible to it by "
                "construction."))
        else:
            parts.append(_banner(
                "warn", "NOTHING IS RENDERED, BUT THE CENSUS IS NOT EMPTY",
                f"The census holds {n_total_candidates} candidate clusters and "
                f"this gallery renders none of them. That is a consequence of "
                f"the caps or of missing proxy frames recorded above, NOT a "
                f"zero-candidate census result."))
        parts.append("<h2>Candidates</h2><p>None rendered. There is nothing to "
                     "classify here, so no decision form is rendered.</p>")
    else:
        parts.append("<h2>Index</h2>")
        parts.append(f'<p class="small">{_esc(_ranking_order_string(mode))}'
                     f"</p>")
        extra_head = ("<th>component tiles</th><th>tiles firing</th>"
                      if mode == MODE_PER_TILE else "")
        head = ("<tr><th>candidate</th>"
                "<th>mean amplitude (SORT KEY)</th>"
                if mode == MODE_PER_TILE else
                "<tr><th>candidate</th><th>mean amplitude</th>")
        head += ("<th>polarity</th><th>cameras "
                 + ("(NOT a sort key)" if mode == MODE_PER_TILE else "")
                 + "</th><th>median src frame</th><th>t (s)</th>"
                 "<th>spread</th>" + extra_head + "</tr>")
        rows = []
        for item in rendered:
            cluster = item["cluster"]
            meta = (item["overlay"]["meta"] or {}) if item["overlay"] else {}
            extra = ("" if mode != MODE_PER_TILE else
                     f'<td>{_esc(meta.get("component_size_tiles"))}</td>'
                     f'<td>{_esc(meta.get("n_tiles_firing_at_sample"))}</td>')
            rows.append(
                f'<tr><td><a href="#{_esc(item["candidate_id"])}">'
                f'{_esc(item["candidate_id"])}</a></td>'
                f'<td>{_esc(round(float(cluster.get("mean_amplitude", 0.0)), 4))}'
                f'</td>'
                f'<td>{_esc(item["polarity"])}</td>'
                f'<td>{_esc(cluster.get("n_cameras_supporting"))}</td>'
                f'<td>{_esc(item["median"])}</td>'
                f'<td>{_esc(format_seconds(item["t_median"]))}</td>'
                f'<td>{_esc(cluster.get("spread_frames"))} f / '
                f'{float(cluster.get("spread_ms", 0.0)):.1f} ms</td>'
                f'{extra}</tr>')
        parts.append(f'<table class="idx">{head}{"".join(rows)}</table>')
        for item in rendered:
            parts.append(_render_candidate(item, source_rate, stride,
                                           proxy_step_ms, mode))

    parts.append("</main>")
    parts.append(_json_script(decisions_template, "decisions-template"))
    parts.append('<div class="export"><strong>Decisions</strong> '
                 '<span id="decided-count" class="small"></span><br>'
                 '<button type="button" id="download-decisions">'
                 'Download decisions as JSON</button>'
                 '<button type="button" id="copy-decisions">Copy the JSON below'
                 '</button>'
                 '<button type="button" id="clear-decisions">Clear all</button>'
                 '<div class="small" style="color:#c9c9d4">Selections are kept '
                 'in this browser (localStorage). If the download button is '
                 'blocked by the viewer sandbox, select the text below and copy '
                 'it — both paths produce the same file.</div>'
                 '<textarea id="decisions-text" readonly '
                 'spellcheck="false"></textarea></div>')
    parts.append(f"<script>{JS}</script>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _render_candidate(item: dict, source_rate: Fraction, stride: int,
                      proxy_step_ms: float, mode: str = MODE_TILE) -> str:
    cluster = item["cluster"]
    cid = item["candidate_id"]
    per_camera = cluster.get("per_camera_source_frame", {})
    global_by_camera = cluster.get("per_camera_global_template_dist", {}) or {}
    parts = [f'<section class="cand" id="{_esc(cid)}">']
    if mode == MODE_PER_TILE:
        parts.append(
            f"<h2>{_esc(cid)} — mean amplitude "
            f"{_esc(round(float(cluster['mean_amplitude']), 4))} grey levels "
            f"— polarity {_esc(item['polarity'])} — "
            f"{_esc(cluster.get('n_cameras_supporting'))} supporting cameras"
            f"</h2>")
    else:
        parts.append(f"<h2>{_esc(cid)} — polarity {_esc(item['polarity'])} "
                     f"— {_esc(cluster.get('n_cameras_supporting'))} "
                     f"supporting cameras</h2>")
    parts.append('<div class="caveat">CANDIDATE, NOT AN EVENT. This is a '
                 'proposal for human / ground-truth classification. '
                 '&quot;rise&quot; / &quot;fall&quot; is the direction of the '
                 'measured signal, not a semantic. Not finding an occluder in '
                 'these pictures is NOT evidence of absence.</div>')

    global_values = [v for v in global_by_camera.values() if v is not None]
    tile_signal = (item["overlay"]["explanation"].get("tile_max")
                   if item["overlay"] else None)
    support_note = (" — CORROBORATION FLOOR ONLY, this count orders nothing"
                    if mode == MODE_PER_TILE else "")
    signal_rows = [
        ("multi-camera support",
         f"{_esc(cluster.get('n_cameras_supporting'))} distinct cameras "
         f"({_esc(', '.join(cluster.get('cameras', [])))})"
         f"{_esc(support_note)}"),
        ("amplitude / score",
         f"mean {_esc(round(float(cluster.get('mean_amplitude', 0.0)), 4))} grey "
         f"levels, max {_esc(round(float(cluster.get('max_amplitude', 0.0)), 4))}"),
        ("candidate bracket (source frames)",
         f"[{_esc(item['bracket'][0])}, {_esc(item['bracket'][1])}] &rarr; "
         f"t = [{_esc(format_seconds(item['t_bracket'][0]))}, "
         f"{_esc(format_seconds(item['t_bracket'][1]))}] s. The detector saw "
         f"one side at each end and CANNOT say where between them the change "
         f"happened."),
        ("cluster median source frame",
         f"{_esc(item['median'])} &rarr; t = "
         f"{_esc(format_seconds(item['t_median']))} s "
         f"(exact {_esc(item['t_median'].numerator)}/"
         f"{_esc(item['t_median'].denominator)} s)"),
        ("cross-camera spread",
         f"{_esc(cluster.get('spread_frames'))} frames / "
         f"{float(cluster.get('spread_ms', 0.0)):.1f} ms &mdash; "
         f"<strong>{_esc(SPREAD_CAVEAT)}</strong>. One proxy step is "
         f"{stride} source frames ({proxy_step_ms:.2f} ms), so a spread at or "
         f"below one step is the sampling grid, not a measured disagreement."),
    ]
    if mode == MODE_PER_TILE:
        signal_rows.append((
            "firing component per camera",
            f"component size (face-adjacent tiles) = "
            f"{_esc(json.dumps(cluster['per_camera_component_size_tiles'], sort_keys=True))}"
            f"; tiles firing anywhere in the grid at that sample = "
            f"{_esc(json.dumps(cluster['per_camera_n_tiles_firing_at_sample'], sort_keys=True))}"
            f". These are TILE counts, not object measurements."))
        signal_rows.append((
            "component tiles (row, col) per camera",
            f"{_esc(json.dumps(cluster['per_camera_component_tiles_row_col'], sort_keys=True))}"
            f" — TILE INDICES, not an object"))
        signal_rows.append((
            "per-tile amplitudes within the component",
            _esc(json.dumps(cluster["per_camera_per_tile_amplitude"],
                            sort_keys=True))))
    else:
        signal_rows.append((
            "global vs tile signal",
            f"tile signal at the candidate frame = {_esc(tile_signal)}; "
            f"whole-frame template_dist per camera = "
            f"{_esc(json.dumps(global_by_camera, sort_keys=True))}"
            + (f" (whole-frame values span "
               f"{_esc(round(min(global_values), 4))} .. "
               f"{_esc(round(max(global_values), 4))} grey levels)"
               if global_values else "")))
    signal_rows.append((
        "per-camera candidate frames",
        _esc(json.dumps({k: per_camera[k] for k in sorted(per_camera)}))))
    parts.append(_kv_table(signal_rows))

    parts.append("<h3>Multi-camera montage</h3>")
    parts.append('<p class="small">Rows are supporting cameras, columns are a '
                 'context series at proxy-step spacing anchored on the same '
                 'absolute source frames for every row. Each tile carries its '
                 'camera, its exact source frame index, its timestamp from the '
                 'rational rate, and a PRE / GAP / POST label computed from '
                 'that row&#39;s own bracket. GAP is the two-sample bracket the '
                 'change is known only to lie inside.</p>')
    n_pages = len(item["montages"])
    for montage in item["montages"]:
        parts.append(f'<p class="small">montage page {_esc(montage["page"])} '
                     f'of {_esc(n_pages)} &mdash; cameras '
                     f'{_esc(", ".join(montage["cameras"]))}</p>')
        parts.append(f'<img class="sheet" loading="lazy" '
                     f'src="assets/{_esc(montage["file"])}" '
                     f'alt="{_esc(cid)} montage page {_esc(montage["page"])} '
                     f'of {_esc(n_pages)}: '
                     f'{_esc(", ".join(montage["cameras"]))}">')
        parts.append(_legend_table(montage["legend"]))

    parts.append(f"<h3>Reference camera {_esc(item['focus']['camera'])} — "
                 f"pre / candidate / post</h3>")
    parts.append('<p class="small">The reference camera is the supporting '
                 'camera with the largest tile signal; ties break to the '
                 'camera nearest the cluster median and then alphabetically. '
                 'That is a presentation choice and nothing depends on it.</p>')
    parts.append(f'<img class="sheet" loading="lazy" '
                 f'src="assets/{_esc(item["focus"]["file"])}" '
                 f'alt="{_esc(cid)} pre/candidate/post on '
                 f'{_esc(item["focus"]["camera"])}">')
    parts.append(_legend_table(item["focus"]["legend"]))

    heading = (PER_TILE_OVERLAY_HEADING if mode == MODE_PER_TILE
               else OVERLAY_HEADING)
    if item["overlay"]:
        parts.append(f"<h3>{_esc(heading)}</h3>")
        parts.append(f'<div class="caveat">{_esc(OVERLAY_CAVEAT)}</div>')
        explanation = item["overlay"]["explanation"]
        meta = item["overlay"]["meta"] or {}
        if mode == MODE_PER_TILE:
            parts.append(f'<div class="caveat">'
                         f"{_esc(PER_TILE_OVERLAY_CAVEAT)}</div>")
        parts.append(f'<img class="sheet" loading="lazy" '
                     f'src="assets/{_esc(item["overlay"]["file"])}" '
                     f'alt="{_esc(cid)} detector-signal explanation heatmap '
                     f'(not an instance mask)">')
        parts.append(f'<p class="small">{_esc(explanation.get("what_this_is", ""))}'
                     f'</p>')
        if mode == MODE_PER_TILE:
            parts.append(_kv_table([
                ("drawn from",
                 f"{_esc(meta.get('drawn_from'))} &mdash; the FACE-ADJACENT "
                 f"CONNECTED COMPONENT of tiles that each crossed their OWN "
                 f"threshold, never a single argmax tile"),
                ("component size (face-adjacent tiles)",
                 f"<strong>{_esc(explanation.get('component_size_tiles'))}"
                 f"</strong> of "
                 f"{_esc(meta.get('n_tiles_in_grid'))} tiles in the grid"),
                ("tiles firing anywhere at this sample",
                 f"<strong>"
                 f"{_esc(explanation.get('n_tiles_firing_at_sample'))}"
                 f"</strong> &mdash; the coherence gate rejects a sample at "
                 f"or above its upper bound as a global change"),
                ("component tiles (row, col)",
                 _esc(explanation.get("component_tiles_row_col"))
                 + " &mdash; TILE indices, not an object"),
                ("component pixel bbox (proxy raster)",
                 _esc(explanation.get("component_pixel_bbox_xyxy"))
                 + " &mdash; it bounds TILES, not an object extent"),
                ("per-tile amplitudes in the component",
                 f"{_esc(explanation.get('per_tile_amplitude'))} grey levels "
                 f"(mean {_esc(meta.get('amplitude_mean_grey_levels'))})"),
                ("per-tile thresholds crossed",
                 _esc(explanation.get("per_tile_threshold", "not recorded"))),
                ("polarity of the component tiles",
                 f"{_esc(explanation.get('n_tiles_rise'))} rise / "
                 f"{_esc(explanation.get('n_tiles_fall'))} fall, tied="
                 f"{_esc(explanation.get('polarity_is_tied'))}"),
                ("is_instance_mask", _esc(explanation.get("is_instance_mask"))),
                ("kind", _esc(explanation.get("kind"))),
            ]))
        else:
            parts.append(_kv_table([
                ("argmax tile (row, col)",
                 _esc(explanation.get("tile_argmax_row_col"))),
                ("argmax tile pixel box (proxy raster)",
                 _esc(explanation.get("tile_argmax_pixel_box_xyxy"))
                 + " &mdash; a TILE extent, not an object extent"),
                ("tile_max / whole-frame template_dist",
                 _esc(explanation.get("tile_max_over_global_template_dist"))),
                ("would the whole-frame pass have cleared its own floor?",
                 _esc(explanation.get("global_pass_would_clear_its_own_floor"))),
                ("is_instance_mask", _esc(explanation.get("is_instance_mask"))),
                ("kind", _esc(explanation.get("kind"))),
            ]))
    else:
        parts.append(f"<h3>{_esc(heading)}</h3>")
        parts.append('<div class="caveat">No tile explanation is present in '
                     'the census for this candidate, so no spatial overlay is '
                     'drawn. It would have been an explanation of the detector '
                     'signal and not an instance mask in any case.</div>')

    raw = json.dumps({
        "candidate_id": cid,
        "window_source_frames": item["window_source_frames"],
        "cluster": cluster,
        "tile_explanation": (item["overlay"]["explanation"]
                             if item["overlay"] else None),
        "derived": {
            "bracket_source_frames": item["bracket"],
            "bracket_rule": "bracket = [candidate_frame - stride, candidate_frame]",
            "t_median_seconds": float(item["t_median"]),
            "t_median_seconds_exact": (f"{item['t_median'].numerator}/"
                                       f"{item['t_median'].denominator}"),
            "time_rule": "t = source_frame * denominator / numerator, exact",
        },
    }, indent=2, sort_keys=True)
    parts.append(f"<details><summary>Raw JSON provenance for {_esc(cid)} "
                 f"(census cluster + tile explanation, verbatim)</summary>"
                 f'<pre class="raw">{_esc(raw)}</pre></details>')

    parts.append(_controls_html(cid))
    parts.append("</section>")
    return "".join(parts)


def _legend_table(legend: list[dict]) -> str:
    head = ("<tr><th>camera</th><th>source frame</th><th>t (s)</th>"
            "<th>t exact (s)</th><th>phase</th><th>frame present</th></tr>")
    cells = []
    for row in legend:
        seconds = "%.4f" % float(row["t_seconds"])
        present = "yes" if row["present"] else "NO PROXY FRAME"
        cells.append(
            f'<tr><td>{_esc(row["camera"])}</td>'
            f'<td>{_esc(row["source_frame"])}</td>'
            f'<td>{_esc(seconds)}</td>'
            f'<td>{_esc(row["t_seconds_exact"])}</td>'
            f'<td>{_esc(row["phase"])}</td>'
            f'<td>{_esc(present)}</td></tr>')
    rows = "".join(cells)
    return ("<details><summary>Tile legend (camera, exact source frame, "
            "timestamp from the rational rate, phase)</summary>"
            f'<table class="idx">{head}{rows}</table></details>')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--census", required=True,
                        help="census JSON from imvid_event_proxy.py --mode census")
    parser.add_argument("--proxy-root", required=True,
                        help="proxy tree root (READ ONLY): <root>/<camNN>/frames/")
    parser.add_argument("--out", required=True, help="output bundle directory")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--max-candidates", type=int,
                        default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--max-cameras-per-candidate", type=int,
                        default=DEFAULT_MAX_CAMERAS_PER_CANDIDATE,
                        help="camera rows per montage PAGE; more cameras "
                             "produce more pages, they are not dropped until "
                             "--max-montage-pages bites")
    parser.add_argument("--max-montage-pages", type=int,
                        default=DEFAULT_MAX_MONTAGE_PAGES,
                        help="0 = unlimited")
    parser.add_argument("--context-steps", type=int,
                        default=DEFAULT_CONTEXT_STEPS,
                        help="proxy steps sampled either side of the anchor")
    parser.add_argument("--montage-tile-width", type=int,
                        default=DEFAULT_MONTAGE_TILE_WIDTH)
    parser.add_argument("--focus-tile-width", type=int,
                        default=DEFAULT_FOCUS_TILE_WIDTH)
    parser.add_argument("--overlay-width", type=int, default=DEFAULT_OVERLAY_WIDTH)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                        help="soft bundle budget; 0 = unlimited")
    parser.add_argument("--clean", action="store_true",
                        help="delete the output directory first")
    args = parser.parse_args(argv)

    manifest = build_gallery(
        census_path=Path(args.census), proxy_root=Path(args.proxy_root),
        out_dir=Path(args.out), scene=args.scene,
        max_candidates=args.max_candidates,
        max_cameras_per_candidate=args.max_cameras_per_candidate,
        max_montage_pages=args.max_montage_pages,
        context_steps=args.context_steps,
        montage_tile_width=args.montage_tile_width,
        focus_tile_width=args.focus_tile_width,
        overlay_width=args.overlay_width, jpeg_quality=args.jpeg_quality,
        max_bytes=args.max_bytes, clean=args.clean)

    out_dir = Path(args.out)
    print(f"[gallery] census detection schema: "
          f"{manifest['census_detection_mode']}", flush=True)
    print(f"[gallery] ranked by: {manifest['ranking']['order']}", flush=True)
    if manifest["per_tile_floor_provenance"]:
        floor = manifest["per_tile_floor_provenance"]
        print(f"[gallery] per-tile absolute floor F={floor['F']} x measured "
              f"noise scale median "
              f"{floor['noise_scale_median_grey_levels']:.6f} = "
              f"{floor['floor_grey_levels']:.6f} grey levels "
              f"({floor['n_tile_series']} tile series)", flush=True)
        if floor["floor_is_degenerate_zero"]:
            print("[gallery] WARNING: the measured noise scale median is "
                  "ZERO, so the absolute gate is INERT. The index page says "
                  "so in a red banner.", flush=True)
    print(f"[gallery] {manifest['n_candidates_rendered']} of "
          f"{manifest['n_candidates_in_census']} candidate clusters rendered",
          flush=True)
    print(f"[gallery] {manifest['n_images']} images, "
          f"{manifest['bytes_total'] / 2**20:.2f} MiB total "
          f"(index.html {manifest['bytes_index_html'] / 1024:.1f} KiB)",
          flush=True)
    drops = manifest["dropped"]
    any_drop = False
    for key in ("candidates_dropped_by_max_candidates",
                "candidates_dropped_by_size_budget",
                "camera_rows_dropped_by_page_cap"):
        if int(drops[key]):
            any_drop = True
            print(f"[gallery] DROPPED: {key} = {drops[key]}", flush=True)
    for item in drops["candidates_with_dropped_cameras"]:
        any_drop = True
        print(f"[gallery] DROPPED: {item['candidate_id']} shows "
              f"{item['cameras_shown']} of {item['cameras_supporting']} "
              f"cameras ({item['cameras_dropped']} dropped)", flush=True)
    for item in drops["cameras_missing_from_proxy_tree"]:
        any_drop = True
        print(f"[gallery] MISSING FROM PROXY TREE: {item['candidate_id']} -> "
              f"{', '.join(item['cameras'])}", flush=True)
    if drops["candidates_without_tile_explanation"]:
        print(f"[gallery] no tile explanation for "
              f"{len(drops['candidates_without_tile_explanation'])} candidate(s): "
              f"{', '.join(drops['candidates_without_tile_explanation'][:20])}",
              flush=True)
    if not any_drop:
        print("[gallery] nothing dropped: every census candidate and every "
              "supporting camera is rendered", flush=True)
    if manifest["n_candidates_in_census"] == 0:
        print("[gallery] THE CENSUS RETURNED ZERO CANDIDATES. That is a "
              "RESULT, not an error: the gallery states it plainly and shows "
              "the settings and the declared detection scale. It is not "
              "evidence that the scene contains no occlusion or reveal.",
              flush=True)
    print(f"[gallery] open {out_dir / 'index.html'} — CANDIDATES, NOT "
          f"EVENTS; no candidate here is claimed to be an event", flush=True)
    print(f"[gallery] manifest -> {out_dir / GALLERY_MANIFEST_NAME}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
