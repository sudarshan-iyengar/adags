#!/usr/bin/env python
"""Assemble the counterfactual-absence N3V fixture from an original scene plus an
SA4D-style edit tranche.

WHAT THIS IS. The original blender-format N3V scene is left byte-untouched. A
DERIVED scene is materialized next to it in which the frames of one authored
window ``[A, B]`` carry the EDITED imagery (the object removed and the
background behind it in-painted by the editing tool) and every other frame is
the ORIGINAL pixel data, shared by hardlink so the derived tree costs almost
nothing and cannot silently drift from its source.

WHAT THIS IS NOT. The absent interval is a TEACHER-RENDERED COUNTERFACTUAL, not
a measurement: nobody observed this scene with the object removed. The manifest
stamps ``teacher_rendered_counterfactual: true`` and ``evidence_bearing: false``
so no downstream page can quote a number from this fixture as evidence about
the world. It is an instrument for asking whether a representation can be made
to render absence, and nothing else.

THE PRECONDITION THAT MATTERS. A reading rule is not enough; the mechanism the
rule reads must be shown to have been exercised. Two are enforced here and both
are statements about the SETUP, never about a score:

  * the union silhouette over ``[A, B]`` must be non-empty, otherwise the event
    bounding box would be a bounding box of nothing and every downstream metric
    would be computed over an empty or arbitrary region while still returning a
    finite, healthy-looking number;
  * at least one frame of ``[A, B]`` must actually carry an edited PNG,
    otherwise the "absence" fixture is a byte-for-byte copy of the original and
    a comparison against it measures the copy.

Both refuse the build rather than annotate the output.

OUTPUT LAYOUT
    <out>/images/camXX_FFFF.png                  hardlink (original) or copy (edited)
    <out>/points3d.ply                           hardlink
    <out>/transforms_train.json                  byte copy
    <out>/transforms_test.json                   byte copy
    <out>/absfix/evaluation_rois/core/FFFFF.png  cam00, 5-digit ABSOLUTE frame
    <out>/absfix/evaluation_rois/ring/FFFFF.png
    <out>/absfix/evaluation_rois/object/FFFFF.png
    <out>/absfix/absence_event_masks.json        ccr-event-ray-masks-v1
    <out>/MANIFEST.absence_edit.json

``flow/``, ``motion_priors/`` and ``seg/`` are NEVER created: those are derived
products of the ORIGINAL imagery and carrying them into a scene whose pixels
have changed inside ``[A, B]`` would silently feed the trainer priors computed
from a world the images no longer show.

The 5-digit ROI filenames are the ABSOLUTE frame index, matching the
``"%05d.png"`` convention of ``scripts/eval_n3v_gated.render_filename`` and
``scripts/event_ray_metrics``, so an ROI directory can be handed to a
per-frame profiler without a second naming rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

TOOL_NAME = "scripts/build_absence_fixture_scene.py"
SCHEMA_VERSION = "ccr-event-ray-masks-v1"

# Frozen ROI geometry. These are the fixture's identity: changing one changes
# which pixels every downstream number is computed over.
SILHOUETTE_GUARD_DILATE_PX = 12   # construction-mask dilation the DEVA silhouette is clipped to
CORE_ERODE_PX = 4                 # erosion of (silhouette AND supported) -> core
RING_DILATE_PX = 6                # dilation of the silhouette that bounds the ring
EVENT_BBOX_PAD_PX = 8             # padding on the union silhouette bbox
MASK_THRESHOLD = 128              # uint8 masks are 0/255; >= this is "set"

FORBIDDEN_DIRS = ("flow", "motion_priors", "seg")

IMAGE_NAME_RE = re.compile(r"^cam(\d+)_(\d+)$")


# ---------------------------------------------------------------------------
# PURE HELPERS -- numpy + stdlib only, so the tests can exercise every reading
# rule without a real scene on disk.
# ---------------------------------------------------------------------------


def parse_image_stem(name):
    """``('cam03', 147)`` from ``cam03_0147.png`` / ``images/cam03_0147``.

    Returns ``None`` when the stem is not a ``cam<NN>_<FFFF>`` name. The camera
    is returned as the ZERO-PADDED string exactly as written, because that
    string is the directory key everywhere else in the fixture; the frame is an
    int because it is compared against window bounds.
    """
    stem = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    if stem.lower().endswith(".png"):
        stem = stem[:-4]
    match = IMAGE_NAME_RE.match(stem)
    if match is None:
        return None
    return "cam" + match.group(1), int(match.group(2))


def frame_token(frame, width=4):
    """The zero-padded frame token used by the ORIGINAL image names."""
    return "%0*d" % (int(width), int(frame))


def roi_filename(frame):
    """``<ABSFRAME:05d>.png`` -- the absolute-frame ROI name."""
    return "%05d.png" % int(frame)


def should_copy_edited(frame, window, edited_path):
    """The hardlink-vs-copy decision, isolated so it can be tested alone.

    A frame takes the EDITED bytes only when it is inside the authored window
    AND the editing tool actually produced a file for it. Everything else is
    shared with the original. Returning a bool rather than acting means the
    rule is auditable without a filesystem.
    """
    lo, hi = int(window[0]), int(window[1])
    if not (lo <= int(frame) <= hi):
        return False
    return Path(edited_path).is_file()


def _shift_or(mask, dy, dx):
    """``mask`` translated by ``(dy, dx)`` with False fill."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys = slice(max(dy, 0), h + min(dy, 0))
    xs = slice(max(dx, 0), w + min(dx, 0))
    yt = slice(max(-dy, 0), h + min(-dy, 0))
    xt = slice(max(-dx, 0), w + min(-dx, 0))
    out[ys, xs] = mask[yt, xt]
    return out


def dilate(mask, pixels):
    """Binary dilation by a ``(2k+1)`` SQUARE (Chebyshev) structuring element.

    Separable: one pass along rows, one along columns, so the cost is linear in
    ``k`` rather than quadratic. Square, not disc -- "N px" here means N pixels
    in the max-norm, and that choice is frozen because the ROI geometry depends
    on it.
    """
    k = int(pixels)
    if k < 0:
        raise ValueError("dilation radius must be >= 0; got %r" % (pixels,))
    out = np.asarray(mask, dtype=bool)
    if out.ndim != 2:
        raise ValueError("a mask must be 2-D; got %r" % (out.shape,))
    if k == 0:
        return out.copy()
    grown = out.copy()
    for dx in range(-k, k + 1):
        if dx:
            grown |= _shift_or(out, 0, dx)
    out = grown
    grown = out.copy()
    for dy in range(-k, k + 1):
        if dy:
            grown |= _shift_or(out, dy, 0)
    return grown


def erode(mask, pixels):
    """Erosion is dilation of the complement, complemented."""
    return ~dilate(~np.asarray(mask, dtype=bool), pixels)


def outline(mask):
    """The 1-px inner boundary of a mask -- what the montage draws."""
    m = np.asarray(mask, dtype=bool)
    return m & ~erode(m, 1)


def build_rois(silhouette_ids, construction, support):
    """The three evaluation ROIs for one cam00 frame.

    ``silhouette_ids``  bool, the union of the chosen DEVA ids
    ``construction``    bool, the editing tool's construction mask
    ``support``         bool, the background-behind-the-object alpha >= 128

    silhouette = DEVA ids AND dilate(construction, 12)
        The DEVA id map is a tracker output on the ORIGINAL video and it leaks:
        it will happily label the stool the object stands on. The construction
        mask is what the editor actually rewrote, so intersecting with a
        generously dilated copy of it keeps the tracker honest without eroding
        the true silhouette boundary.
    core = erode(silhouette AND support, 4)
        Pixels where the object was AND the tool knows what is behind it,
        pulled 4 px in from the boundary so feathering never lands in the core.
    ring = dilate(silhouette, 6) MINUS core
        The halo where a leaking representation shows up first.
    object = silhouette
    """
    sil = np.asarray(silhouette_ids, dtype=bool)
    con = np.asarray(construction, dtype=bool)
    sup = np.asarray(support, dtype=bool)
    if not (sil.shape == con.shape == sup.shape):
        raise ValueError(
            "ROI inputs disagree on shape: %r %r %r" % (sil.shape, con.shape, sup.shape))
    silhouette = sil & dilate(con, SILHOUETTE_GUARD_DILATE_PX)
    core = erode(silhouette & sup, CORE_ERODE_PX)
    ring = dilate(silhouette, RING_DILATE_PX) & ~core
    return {"core": core, "ring": ring, "object": silhouette}


def mask_bbox(mask):
    """``[x0, y0, x1, y1]`` half-open on the far edge, or ``None`` when empty."""
    m = np.asarray(mask, dtype=bool)
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1]


def union_bbox(boxes):
    """The bounding box of a list of ``[x0,y0,x1,y1]``; ``None`` entries skipped."""
    live = [b for b in boxes if b is not None]
    if not live:
        return None
    x0 = min(b[0] for b in live)
    y0 = min(b[1] for b in live)
    x1 = max(b[2] for b in live)
    y1 = max(b[3] for b in live)
    return [int(x0), int(y0), int(x1), int(y1)]


def pad_bbox(box, pad, width, height):
    """Grow a box by ``pad`` px and clamp it to the raster."""
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return [
        int(max(0, x0 - pad)),
        int(max(0, y0 - pad)),
        int(min(int(width), x1 + pad)),
        int(min(int(height), y1 + pad)),
    ]


def difference_confined(original, edited, allowed_mask):
    """Is every changed pixel inside ``allowed_mask``, and did anything change?

    Returns ``(confined, outside_changed_px, inside_mean_abs_diff)``. Byte
    arrays, not floats: "byte-equal outside" is the claim, so the comparison is
    exact and a 1/255 leak cannot round away.
    """
    a = np.asarray(original)
    b = np.asarray(edited)
    if a.shape != b.shape:
        raise ValueError("shape mismatch: %r vs %r" % (a.shape, b.shape))
    allowed = np.asarray(allowed_mask, dtype=bool)
    if allowed.shape != a.shape[:2]:
        raise ValueError(
            "mask shape %r does not match image %r" % (allowed.shape, a.shape[:2]))
    changed = (a != b)
    if changed.ndim == 3:
        changed = changed.any(axis=2)
    outside = int(np.count_nonzero(changed & ~allowed))
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    inside = float(diff[allowed].mean()) if allowed.any() else 0.0
    return outside == 0, outside, inside


def frames_have_expected_time(frames, fps=30.0, tol=1e-6):
    """``[(file_path, frame, time, ok), ...]`` for the ``time == frame/fps`` rule.

    The N3V blender export stamps ``time`` as the absolute frame divided by the
    nominal fps. A scene assembled by copying transforms byte-for-byte cannot
    break this, which is exactly why it is checked: it is the cheapest
    detector of the transforms having been REGENERATED rather than copied.
    """
    out = []
    for entry in frames:
        path = entry.get("file_path")
        parsed = parse_image_stem(path)
        frame = None if parsed is None else parsed[1]
        time = entry.get("time")
        ok = (
            frame is not None
            and isinstance(time, (int, float))
            and abs(float(time) - frame / float(fps)) <= tol
        )
        out.append((path, frame, time, bool(ok)))
    return out


def summarize_hole_fraction(payload):
    """Per-camera ``{min, median, max, n}`` from whatever shape the tool wrote.

    ``hole_fraction.json`` is a THIRD-PARTY artifact; its exact shape is not
    guaranteed by anything in this repository. Three plausible shapes are
    parsed (flat ``camXX_FFFF`` keys, camera-keyed maps/lists, and a list of
    records) and anything else is recorded as ``unparsed`` rather than being
    coerced into a summary that would read as a measurement.
    """
    values = {}

    def add(cam, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
        values.setdefault(str(cam), []).append(value)
        return True

    parsed_any = False
    if isinstance(payload, dict):
        for key, val in payload.items():
            parsed = parse_image_stem(key)
            if parsed is not None and not isinstance(val, (dict, list)):
                parsed_any |= add(parsed[0], val)
            elif isinstance(val, dict):
                for sub in val.values():
                    parsed_any |= add(key, sub)
            elif isinstance(val, list):
                for sub in val:
                    if isinstance(sub, dict):
                        parsed_any |= add(key, sub.get("hole_fraction", sub.get("value")))
                    else:
                        parsed_any |= add(key, sub)
            else:
                parsed_any |= add(key, val)
    elif isinstance(payload, list):
        for rec in payload:
            if not isinstance(rec, dict):
                continue
            cam = rec.get("camera") or rec.get("cam")
            if cam is None:
                parsed = parse_image_stem(rec.get("file", ""))
                cam = None if parsed is None else parsed[0]
            if cam is None:
                continue
            parsed_any |= add(cam, rec.get("hole_fraction", rec.get("value")))

    if not parsed_any:
        return {"unparsed": True, "payload_type": type(payload).__name__}
    summary = {}
    for cam in sorted(values):
        series = sorted(values[cam])
        summary[cam] = {
            "min": float(series[0]),
            "median": float(statistics.median(series)),
            "max": float(series[-1]),
            "n": len(series),
        }
    return summary


def build_event_manifest(prefix, bbox, window, control_window, raster, source_note):
    """The ``ccr-event-ray-masks-v1`` payload for this fixture.

    The five windows are FROZEN offsets from the authored window and are chosen
    before any number is read:
      ``_absence_gap``   [A+3, B-2]    inside the gap, clear of both edges
      ``_return_early``  [B+3, B+10]   the first observations after the return
      ``_return_late``   [B+11, B+20]  the settled return
      ``_pre``           [A-30, A-3]   the object present, before the gap
      ``_control``       [CA, CB]      an authored window with no edit at all
    """
    a, b = int(window[0]), int(window[1])
    ca, cb = int(control_window[0]), int(control_window[1])
    if bbox is None:
        raise ValueError("refusing to write an event manifest with an empty bbox")
    events = [
        (prefix + "_absence_gap", [[a + 3, b - 2]],
         "the authored absence: the object is gone from the imagery"),
        (prefix + "_return_early", [[b + 3, b + 10]],
         "the first frames after the object returns"),
        (prefix + "_return_late", [[b + 11, b + 20]],
         "the settled return"),
        (prefix + "_pre", [[a - 30, a - 3]],
         "the object present, before the gap"),
        (prefix + "_control", [[ca, cb]],
         "an unedited window; no counterfactual applied"),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source_note,
        "raster": [int(raster[0]), int(raster[1])],
        "camera": "cam00",
        "window_frames": [0, 299],
        "note": (
            "Counterfactual-absence fixture. The imagery inside the authored "
            "window is TEACHER-RENDERED, not observed, so every number scored "
            "on '%s_absence_gap' is a statement about the representation's "
            "ability to render an authored absence and NOT evidence about the "
            "world. Boxes are [x0,y0,x1,y1] in the cam00 raster, half-open on "
            "the far edge, and are the union silhouette over the whole gap "
            "padded by %d px. '%s_control' carries no edit and exists so a "
            "reader can tell an effect of the edit from an effect of the "
            "window." % (prefix, EVENT_BBOX_PAD_PX, prefix)
        ),
        "events": [
            {
                "name": name,
                "bbox": list(bbox),
                "frames": frames,
                "class": "counterfactual_absence",
                "confidence": "authored",
                "occluder": note,
            }
            for name, frames, note in events
        ],
    }


# ---------------------------------------------------------------------------
# filesystem / provenance
# ---------------------------------------------------------------------------


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def image_list_signature(images_dir):
    """A sha over ``name\\tsize\\tmtime_ns`` of every PNG in a directory.

    Recorded before AND after the build so the manifest can assert the raw tree
    was untouched. Hardlinking bumps the source's link count and ctime but
    leaves name, size and mtime alone, so this signature is stable under the
    operation it is meant to permit and changes under the one it is meant to
    catch.
    """
    lines = []
    for name in sorted(os.listdir(images_dir)):
        if not name.lower().endswith(".png"):
            continue
        st = os.stat(os.path.join(images_dir, name))
        lines.append("%s\t%d\t%d" % (name, st.st_size, st.st_mtime_ns))
    blob = "\n".join(lines).encode("utf-8")
    return {"sha256": hashlib.sha256(blob).hexdigest(), "n_files": len(lines)}


def git_provenance(repo_root):
    """``{commit, dirty}`` or ``{error}``. Never raises."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(repo_root),
            capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(status)}
    except Exception as exc:  # provenance is recorded, never enforced
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def read_mask(path, threshold=MASK_THRESHOLD):
    """A uint8 mask PNG as a bool array."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"))
    return arr >= threshold


def read_id_map(path):
    """A DEVA id map PNG as a label array (NOT thresholded).

    ``P``-mode files are read WITHOUT converting: the palette indices ARE the
    tracker ids, and ``convert("L")`` would push them through the palette and
    silently return colours instead of labels.
    """
    with Image.open(path) as img:
        if img.mode not in ("L", "I;16", "I", "P"):
            img = img.convert("L")
        arr = np.asarray(img)
    return arr


def write_mask(path, mask):
    Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), mode="L").save(path)


def link_or_copy(src, dst, copy):
    """Hardlink unless ``copy``. Returns ``'hardlink'`` or ``'copy'``.

    A failed ``os.link`` is FATAL without ``--copy``: silently degrading to a
    copy would produce a derived tree that looks right and no longer tracks its
    source, and the whole point of the hardlink is that it cannot drift.
    """
    if copy:
        shutil.copyfile(src, dst)
        return "copy"
    os.link(src, dst)
    return "hardlink"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orig", required=True, help="original blender-format scene")
    ap.add_argument("--edit", required=True, help="SA4D edit tranche")
    ap.add_argument("--deva_cam00", required=True,
                    help="DEVA run directory holding object_mask/FFFF.png for cam00")
    ap.add_argument("--deva_cam00_ids", required=True, nargs="+", type=int,
                    help="the DEVA ids whose union is the object silhouette")
    ap.add_argument("--out", required=True, help="derived scene directory (must not exist)")
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of hardlink (filesystems without link support)")
    ap.add_argument("--control_window", nargs=2, type=int, required=True,
                    metavar=("CA", "CB"), help="the unedited control window, inclusive")
    ap.add_argument("--event_prefix", default="X",
                    help="event-name prefix; events are <prefix>_absence_gap etc.")
    ap.add_argument("--expect_raster", nargs=2, type=int, default=[1352, 1014],
                    metavar=("W", "H"),
                    help="refuse to build if the imagery is not this size")
    ap.add_argument("--deva_frame_digits", type=int, default=4,
                    help="zero-padding of the DEVA object_mask filenames")
    args = ap.parse_args(argv)

    orig = Path(args.orig)
    edit = Path(args.edit)
    deva = Path(args.deva_cam00)
    out = Path(args.out)

    if out.exists():
        sys.exit("refusing to overwrite an existing output directory: %s" % out)
    for path, what in ((orig, "--orig"), (edit, "--edit"), (deva, "--deva_cam00")):
        if not path.is_dir():
            sys.exit("%s is not a directory: %s" % (what, path))

    orig_images = orig / "images"
    if not orig_images.is_dir():
        sys.exit("no images/ under --orig: %s" % orig_images)

    edit_params = json.loads((edit / "edit_params.json").read_text(encoding="utf-8"))
    window = [int(edit_params["window"][0]), int(edit_params["window"][1])]
    margin = int(edit_params.get("margin", 0))
    a, b = window
    if a > b:
        sys.exit("edit_params window is empty: %r" % (window,))
    margin_requested = [a - margin, b + margin]

    provenance_path = edit / "sa4d_provenance.json"
    sa4d_provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file() else {"error": "sa4d_provenance.json absent"}
    )
    hole_path = edit / "hole_fraction.json"
    hole_summary = (
        summarize_hole_fraction(json.loads(hole_path.read_text(encoding="utf-8")))
        if hole_path.is_file() else {"error": "hole_fraction.json absent"}
    )

    # --- signature of the raw tree BEFORE we touch anything -----------------
    raw_before = image_list_signature(orig_images)

    names = sorted(n for n in os.listdir(orig_images) if n.lower().endswith(".png"))
    if not names:
        sys.exit("no PNGs under %s" % orig_images)

    with Image.open(orig_images / names[0]) as probe:
        raster = (probe.width, probe.height)
    if list(raster) != [int(args.expect_raster[0]), int(args.expect_raster[1])]:
        sys.exit("imagery is %dx%d but --expect_raster is %dx%d"
                 % (raster[0], raster[1], args.expect_raster[0], args.expect_raster[1]))

    # The margin range is CLAMPED to the frames the original actually has, and
    # both the requested and the effective range are recorded. Clamping rather
    # than failing keeps a window near frame 0 usable; recording it keeps the
    # verifier and the builder quoting ONE range instead of two.
    available = sorted({p[1] for p in (parse_image_stem(n) for n in names) if p})
    if not available:
        sys.exit("no cam<NN>_<FFFF>.png names under %s" % orig_images)
    margin_lo = max(margin_requested[0], available[0])
    margin_hi = min(margin_requested[1], available[-1])
    if margin_lo > margin_hi:
        sys.exit("the margin range %r has no overlap with the frames present [%d, %d]"
                 % (margin_requested, available[0], available[-1]))

    # --- precondition: the edit tranche is not vacuous ----------------------
    edited_dir = edit / "images_edited"
    edited_present = []
    for name in names:
        parsed = parse_image_stem(name)
        if parsed is None:
            continue
        if should_copy_edited(parsed[1], window, edited_dir / name):
            edited_present.append(name)
    if not edited_present:
        sys.exit(
            "PRECONDITION FAILED: no edited PNG exists for any frame of the "
            "authored window [%d, %d] under %s -- the derived scene would be a "
            "byte-for-byte copy of the original and any comparison against it "
            "would measure the copy." % (a, b, edited_dir))

    # --- materialize the derived tree ---------------------------------------
    out_images = out / "images"
    out_images.mkdir(parents=True)
    counts = {"hardlinked": 0, "copied": 0, "edited": 0, "skipped_non_png": 0}
    for name in names:
        parsed = parse_image_stem(name)
        if parsed is None:
            counts["skipped_non_png"] += 1
            continue
        src_edited = edited_dir / name
        if should_copy_edited(parsed[1], window, src_edited):
            shutil.copyfile(src_edited, out_images / name)
            counts["edited"] += 1
        else:
            how = link_or_copy(orig_images / name, out_images / name, args.copy)
            counts["hardlinked" if how == "hardlink" else "copied"] += 1

    ply_src = orig / "points3d.ply"
    if not ply_src.is_file():
        sys.exit("no points3d.ply under --orig")
    link_or_copy(ply_src, out / "points3d.ply", args.copy)

    transform_shas = {}
    for tname in ("transforms_train.json", "transforms_test.json"):
        src = orig / tname
        if not src.is_file():
            sys.exit("no %s under --orig" % tname)
        shutil.copyfile(src, out / tname)
        transform_shas[tname] = sha256_file(out / tname)
        if transform_shas[tname] != sha256_file(src):
            sys.exit("byte copy of %s did not reproduce the source sha" % tname)

    # --- cam00 evaluation ROIs ----------------------------------------------
    roi_root = out / "absfix" / "evaluation_rois"
    for kind in ("core", "ring", "object"):
        (roi_root / kind).mkdir(parents=True)

    ids = np.asarray(sorted(set(int(i) for i in args.deva_cam00_ids)))
    silhouette_boxes = []
    roi_frames = []
    for frame in range(margin_lo, margin_hi + 1):
        tok4 = frame_token(frame, args.deva_frame_digits)
        deva_path = deva / "object_mask" / (tok4 + ".png")
        con_path = edit / "construction_masks" / ("cam00_%s.png" % frame_token(frame))
        sup_path = edit / "support" / ("cam00_%s.png" % frame_token(frame))
        for p, what in ((deva_path, "DEVA object_mask"),
                        (con_path, "construction mask"),
                        (sup_path, "support mask")):
            if not p.is_file():
                sys.exit("missing %s for frame %d: %s" % (what, frame, p))
        id_map = read_id_map(deva_path)
        silhouette_ids = np.isin(id_map, ids)
        rois = build_rois(silhouette_ids, read_mask(con_path), read_mask(sup_path))
        for kind, mask in rois.items():
            write_mask(roi_root / kind / roi_filename(frame), mask)
        roi_frames.append(frame)
        if a <= frame <= b:
            silhouette_boxes.append(mask_bbox(rois["object"]))

    union = union_bbox(silhouette_boxes)
    if union is None:
        sys.exit(
            "PRECONDITION FAILED: the object silhouette is EMPTY on every "
            "frame of the authored window [%d, %d]. The event bounding box "
            "would be a bounding box of nothing, and every downstream metric "
            "would still return a finite number. Check --deva_cam00_ids."
            % (a, b))
    bbox = pad_bbox(union, EVENT_BBOX_PAD_PX, raster[0], raster[1])

    events = build_event_manifest(
        args.event_prefix, bbox, window, args.control_window, raster,
        source_note="%s (%s)" % (TOOL_NAME, out.name))
    (out / "absfix" / "absence_event_masks.json").write_text(
        json.dumps(events, indent=1) + "\n", encoding="utf-8")

    # --- per-file shas over the margin range, all cameras --------------------
    file_shas = {}
    for name in names:
        parsed = parse_image_stem(name)
        if parsed is None:
            continue
        if not (margin_lo <= parsed[1] <= margin_hi):
            continue
        file_shas[name] = {
            "before": sha256_file(orig_images / name),
            "after": sha256_file(out_images / name),
            "edited": name in set(edited_present),
        }

    raw_after = image_list_signature(orig_images)

    manifest = {
        "tool": TOOL_NAME,
        "repo": git_provenance(REPO_ROOT),
        "orig": str(orig),
        "edit": str(edit),
        "out": str(out),
        "raster": [int(raster[0]), int(raster[1])],
        "window": window,
        "margin": margin,
        "margin_range": [margin_lo, margin_hi],
        "margin_range_requested": margin_requested,
        "margin_range_clamped": [margin_lo, margin_hi] != margin_requested,
        "control_window": [int(args.control_window[0]), int(args.control_window[1])],
        "event_prefix": args.event_prefix,
        "edit_params": edit_params,
        "sa4d_provenance": sa4d_provenance,
        "hole_fraction_summary": hole_summary,
        "deva_cam00": str(deva),
        "deva_cam00_ids": [int(i) for i in ids],
        "roi_geometry": {
            "silhouette_guard_dilate_px": SILHOUETTE_GUARD_DILATE_PX,
            "core_erode_px": CORE_ERODE_PX,
            "ring_dilate_px": RING_DILATE_PX,
            "event_bbox_pad_px": EVENT_BBOX_PAD_PX,
            "structuring_element": "square (Chebyshev)",
            "mask_threshold": MASK_THRESHOLD,
        },
        "roi_frames": [int(f) for f in roi_frames],
        "event_bbox": bbox,
        "file_sha256": file_shas,
        "points3d_ply_sha256": sha256_file(out / "points3d.ply"),
        "transforms_sha256": transform_shas,
        "counts": counts,
        "link_mode": "copy" if args.copy else "hardlink",
        "teacher_rendered_counterfactual": True,
        "evidence_bearing": False,
        "raw_tree_untouched": {
            "before": raw_before,
            "after": raw_after,
            "unchanged": raw_before == raw_after,
        },
        "forbidden_dirs_never_created": list(FORBIDDEN_DIRS),
    }
    (out / "MANIFEST.absence_edit.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    if not manifest["raw_tree_untouched"]["unchanged"]:
        sys.exit("the ORIGINAL images/ signature changed during the build; "
                 "the derived tree is written but the raw tree is NOT clean")

    print("wrote %s" % out)
    print("  images: %(hardlinked)d hardlinked, %(copied)d copied, %(edited)d edited" % counts)
    print("  rois:   %d frames [%d, %d]" % (len(roi_frames), margin_lo, margin_hi))
    print("  bbox:   %r" % (bbox,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
