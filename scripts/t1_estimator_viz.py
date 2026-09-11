#!/usr/bin/env python3
"""Visualise the T1 visibility-gap estimator's output over a scene's cam00.

WHAT THIS IS. A reusable, CPU-only renderer for the artefacts that
`scripts/estimate_episodes.py` writes: the report JSON
(`adags-episode-estimate-v1`) and, optionally, the standalone v2 program
(`adags-episode-program-v2`). It draws

  (a) a montage, one column per requested frame, of the scene's cam00 image
      with EVERY candidate voxel cell's 3D bounding box projected as its 2D
      bounding polygon, coloured by the estimator's outcome for that cell
      (gated, or the abstention reason), fired cells on top with a thick
      outline and their estimated boundaries printed; and

  (b) below the montage, one strip per fired group over the measured frame
      window.

WHAT THE STRIP CAN AND CANNOT SHOW. `estimate_episode_program` builds a
per-(group, camera, frame) series and returns it as `estimate["series"]`
(scripts/estimate_episodes.py:1039), but `main()` NEVER writes it to the
report: the report's `diagnostics` block carries only
`estimate["decisions"]` (scripts/estimate_episodes.py:1708-1710), and each
decision keeps summary statistics (`contrast`) plus the per-camera crossing
frames (`per_camera`, scripts/estimate_episodes.py:1109-1113). There is
therefore NO per-frame series on disk. The strip consequently shows the
per-camera onset/offset MARKERS, the pooled estimated gap, the evaluated
frames, and the frozen contrast statistics -- and this script prints an
explicit warning saying so. It never fabricates a series.

WHY THE POINT CLOUD IS AN INPUT. A candidate group is one cell of the voxel
grid, and the report identifies a group only by its INDEX. The index is the
rank of the cell's key among the kept keys (scripts/estimate_episodes.py:
709-732), and the v2 program records actual cell keys for the GATED groups
only (`spatial.group_cell_keys`, scripts/estimate_episodes.py:1204-1208).
Recovering the geometry of the abstained cells therefore requires the cloud
the estimate was computed on. `--xyz_npy` takes it as a plain array;
`--checkpoint` lazily imports torch and reads `_xyz`, which is element 1 of
the captured tuple (scene/gaussian_model.py:392-424).

The grouping is then recomputed here, in numpy, exactly as
`build_voxel_groups` computes it, and CROSS-CHECKED: every group's recomputed
row count must equal the `rows` the report recorded, and every gated group's
recomputed cell key must equal the key the v2 program recorded. A mismatch
aborts unless `--allow_row_mismatch` is given, in which case the figure
carries the failure in its provenance line.

CAMERA CONVENTION. N3V scenes are read through the "Blender" reader
(`sceneLoadTypeCallbacks["Blender"]`, scene/dataset_readers.py:711-713), so
the camera is reconstructed from `transforms_{test,train}.json` exactly as
that reader plus `loadCam` plus `Camera` build it:

  c2w = transform_matrix; c2w[:3, 1:3] *= -1      dataset_readers.py:389-391
  w2c = inv(c2w); R = w2c[:3,:3].T; T = w2c[:3,3] dataset_readers.py:394-397
  cx, cy, fl_x, fl_y are divided by the resolution scale
                                                  camera_utils.py:41-45
  world_view = getWorld2View2(R, T).T             cameras.py:66,
                                                  graphics_utils.py:39-50
  projection = getProjectionMatrixCenterShift(
      znear=0.01, zfar=far, cx, cy, fl_x, fl_y, W, H).T
                                                  cameras.py:67-68,
                                                  graphics_utils.py:74-92
  full_proj  = world_view @ projection            cameras.py:76
  clip = [X, 1] @ full_proj; ndc = clip[:2]/clip[3]
  x_px = (ndc_x + 1) * 0.5 * (W - 1)              motion_prior_utils.py:122-149
  y_px = (ndc_y + 1) * 0.5 * (H - 1)

which is the identical chain `build_footprints` projects through
(scripts/estimate_episodes.py:846, 864-867).

This module imports numpy, matplotlib and PIL only. torch is imported lazily
inside `load_xyz_from_checkpoint` and nowhere else, so the whole thing is
importable and testable on a workstation with no CUDA and no torch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

REPORT_SCHEMA = "adags-episode-estimate-v1"

#: Outcome classes, in legend order. Keys are the report's `abstain_reason`
#: values (scripts/estimate_episodes.py:187-193) plus the gated case.
GATED = "gated"
OTHER = "other"
OUTCOME_COLOURS = {
    GATED: "#d62728",                 # red
    "no_interior_gap": "#9e9e9e",     # grey
    "contrast": "#1f77b4",            # blue
    "empty_footprint": "#e6e6e6",     # light
    "camera_disagreement": "#ff7f0e",  # orange
    OTHER: "#9467bd",                 # purple: inadmissible_interval,
                                      # boundary_not_densely_evaluated, unknown
}
OUTCOME_ORDER = [GATED, "no_interior_gap", "contrast", "camera_disagreement",
                 "empty_footprint", OTHER]

NO_SERIES_WARNING = (
    "WARNING: the report schema %s stores NO per-frame series. "
    "estimate_episode_program returns one as estimate['series'] "
    "(scripts/estimate_episodes.py:1039) but main() writes only "
    "estimate['decisions'] into report['diagnostics'] "
    "(scripts/estimate_episodes.py:1708-1710). The per-group strips therefore "
    "show the per-camera onset/offset MARKERS, the pooled estimated gap and "
    "the frozen contrast statistics only -- no series is drawn, and none is "
    "invented." % REPORT_SCHEMA
)


class VizError(RuntimeError):
    """Any fail-closed condition in this tool."""


# ---------------------------------------------------------------------------
# report / program access
# ---------------------------------------------------------------------------


def load_report(path):
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    schema = report.get("schema")
    if schema != REPORT_SCHEMA:
        raise VizError("unexpected report schema %r (expected %r)"
                       % (schema, REPORT_SCHEMA))
    return report


def report_has_series(report):
    """True iff the report carries any per-frame series.

    Checked structurally rather than assumed, so that a future estimator that
    starts writing one is detected instead of silently ignored.
    """
    if "series" in report:
        return True
    diagnostics = report.get("diagnostics", {})
    if "series" in diagnostics:
        return True
    for record in diagnostics.get("decisions", []):
        for key in ("series", "values", "per_frame", "frames"):
            if key in record:
                return True
    return False


def outcome_of(decision):
    """The outcome class of one decision record."""
    if decision.get("gated"):
        return GATED
    reason = decision.get("abstain_reason") or "unknown"
    if reason in OUTCOME_COLOURS:
        return reason
    return OTHER


def outcome_counts(decisions):
    counts = {name: 0 for name in OUTCOME_ORDER}
    for decision in decisions:
        counts[outcome_of(decision)] += 1
    return counts


def grid_spec_of(report):
    """(cells_per_axis, lo, span) of the candidate grid, from the report."""
    grouping = report["grouping"]
    cells = int(grouping["cells_per_axis"])
    lo = grouping.get("grid_lo")
    span = grouping.get("grid_span")
    if lo is None or span is None:
        raise VizError(
            "the report records no grid_lo/grid_span, so the run used the raw "
            "bounding box of the cloud; pass a cloud and this tool will "
            "recompute it -- not implemented, because every lane run uses "
            "--grid_percentile")
    return cells, np.asarray(lo, dtype=np.float32), np.asarray(span, dtype=np.float32)


# ---------------------------------------------------------------------------
# cloud + grouping (numpy mirror of scripts/estimate_episodes.py:709-732)
# ---------------------------------------------------------------------------


def load_xyz_from_npy(path):
    xyz = np.load(path)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise VizError("--xyz_npy must hold an (N, 3) array; got %s"
                       % (xyz.shape,))
    return np.ascontiguousarray(xyz, dtype=np.float32)


def load_xyz_from_checkpoint(path):
    """`_xyz` out of an EL-GS/4D checkpoint. Lazily imports torch."""
    import torch  # lazy on purpose: this module stays CPU/torch-free to import

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only
        blob = torch.load(path, map_location="cpu")
    model_params = blob[0] if isinstance(blob, (tuple, list)) else blob
    # capture() returns active_sh_degree first and `_xyz` second, in both the
    # 3D and the 4D branch (scene/gaussian_model.py:328-341, 392-424).
    xyz = model_params[1]
    return np.ascontiguousarray(xyz.detach().cpu().numpy(), dtype=np.float32)


def voxel_keys(xyz, cells, lo, span):
    """Per-row absolute cell key, -1 outside the box.

    float32 throughout, mirroring `voxel_grid` (estimate_episodes.py:680-706)
    which runs on a float32 cloud, so the cell assignment is reproduced rather
    than approximated.
    """
    points = np.asarray(xyz, dtype=np.float32)
    lo = np.asarray(lo, dtype=np.float32)
    span = np.asarray(span, dtype=np.float32)
    voxel = np.clip((points - lo) / span * np.float32(cells), 0,
                    cells - 1).astype(np.int64)
    keys = voxel[:, 0] * cells * cells + voxel[:, 1] * cells + voxel[:, 2]
    inside = np.all((points >= lo) & (points <= (lo + span).astype(np.float32)),
                    axis=1)
    return np.where(inside, keys, -1)


def build_voxel_groups(xyz, cells, lo, span, min_rows):
    """(labels, kept_keys) -- the numpy mirror of `build_voxel_groups`.

    `labels[i]` is the group index of row i, -1 for substrate; `kept_keys[g]`
    is the absolute cell key of group g. Group indices are ranks among the
    kept keys in ascending key order, exactly as the torch version assigns
    them (estimate_episodes.py:722-732).
    """
    keys = voxel_keys(xyz, cells, lo, span)
    unique_keys, inverse_map = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse_map, minlength=unique_keys.size)
    keep = (counts >= int(min_rows)) & (unique_keys >= 0)
    remap = np.full(unique_keys.size, -1, dtype=np.int64)
    kept = int(keep.sum())
    if kept:
        remap[keep] = np.arange(kept, dtype=np.int64)
    labels = remap[inverse_map]
    return labels, unique_keys[keep].astype(np.int64)


def cell_box(key, cells, lo, span):
    """((x0,y0,z0), (x1,y1,z1)) world-unit corners of one cell key."""
    key = int(key)
    iz = key % cells
    iy = (key // cells) % cells
    ix = key // (cells * cells)
    index = np.array([ix, iy, iz], dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    span = np.asarray(span, dtype=np.float64)
    return lo + index / cells * span, lo + (index + 1.0) / cells * span


def box_corners(low, high):
    """The 8 corners of an axis-aligned box, as an (8, 3) array."""
    return np.array([[low[0] if i & 1 == 0 else high[0],
                      low[1] if i & 2 == 0 else high[1],
                      low[2] if i & 4 == 0 else high[2]] for i in range(8)],
                    dtype=np.float64)


def verify_grouping(report, program, labels, kept_keys):
    """Cross-check the recomputed grouping against the frozen artefacts.

    Returns (ok, message). Checks: the number of groups; every group's row
    count against the report's `rows`; and, when a v2 program is supplied,
    every gated group's cell key against `spatial.group_cell_keys`.
    """
    decisions = report["diagnostics"]["decisions"]
    problems = []
    if len(kept_keys) != len(decisions):
        problems.append("group count %d != report %d"
                        % (len(kept_keys), len(decisions)))
    counts = np.bincount(labels[labels >= 0], minlength=len(kept_keys))
    mismatched = 0
    for record in decisions:
        group = int(record["group"])
        if group >= counts.size:
            mismatched += 1
            continue
        if int(counts[group]) != int(record["rows"]):
            mismatched += 1
    if mismatched:
        problems.append("%d of %d groups disagree on the row count"
                        % (mismatched, len(decisions)))
    key_problems = 0
    if program is not None:
        declared = program.get("spatial", {}).get("group_cell_keys", {})
        for group_str, keys in declared.items():
            group = int(group_str)
            if group >= len(kept_keys) or [int(kept_keys[group])] != [int(k) for k in keys]:
                key_problems += 1
        if key_problems:
            problems.append("%d gated group(s) disagree on the cell key"
                            % key_problems)
    if problems:
        return False, "grouping cross-check FAILED: " + "; ".join(problems)
    detail = "%d/%d groups match row counts" % (len(decisions), len(decisions))
    if program is not None:
        detail += ", %d gated cell key(s) match" % len(
            program.get("spatial", {}).get("group_cell_keys", {}))
    return True, "grouping cross-check OK (%s)" % detail


# ---------------------------------------------------------------------------
# camera (numpy mirror of the Blender reader + loadCam + Camera)
# ---------------------------------------------------------------------------


def get_world_to_view(rotation, translation):
    """`getWorld2View2` with the default trans/scale (graphics_utils.py:39-50)."""
    world_view = np.zeros((4, 4), dtype=np.float64)
    world_view[:3, :3] = np.asarray(rotation, dtype=np.float64).T
    world_view[:3, 3] = np.asarray(translation, dtype=np.float64)
    world_view[3, 3] = 1.0
    return world_view


def get_projection_center_shift(znear, zfar, cx, cy, fl_x, fl_y, width, height):
    """`getProjectionMatrixCenterShift` (graphics_utils.py:74-92)."""
    top = cy / fl_y * znear
    bottom = -(height - cy) / fl_y * znear
    left = -(width - cx) / fl_x * znear
    right = cx / fl_x * znear
    projection = np.zeros((4, 4), dtype=np.float64)
    projection[0, 0] = 2.0 * znear / (right - left)
    projection[1, 1] = 2.0 * znear / (top - bottom)
    projection[0, 2] = (right + left) / (right - left)
    projection[1, 2] = (top + bottom) / (top - bottom)
    projection[3, 2] = 1.0
    projection[2, 2] = zfar / (zfar - znear)
    projection[2, 3] = -(zfar * znear) / (zfar - znear)
    return projection


class PinholeCamera(object):
    """Just enough of `scene.cameras.Camera` to project world points."""

    def __init__(self, rotation, translation, cx, cy, fl_x, fl_y, width, height,
                 znear=0.01, zfar=100.0, name=""):
        self.width = int(width)
        self.height = int(height)
        self.cx = float(cx)
        self.cy = float(cy)
        self.fl_x = float(fl_x)
        self.fl_y = float(fl_y)
        self.name = name
        self.world_view_transform = get_world_to_view(rotation, translation).T
        self.projection_matrix = get_projection_center_shift(
            znear, zfar, self.cx, self.cy, self.fl_x, self.fl_y,
            self.width, self.height).T
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix

    def project(self, points):
        """(xy pixels, valid) for an (N, 3) world-space array.

        `valid` mirrors `project_points_to_grid` (motion_prior_utils.py:
        122-140): a finite homogeneous w and NDC inside [-1, 1] on both axes.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        clip = hom @ self.full_proj_transform
        w = clip[:, 3]
        w_safe = np.where(np.abs(w) < 1e-6, 1e-6, w)
        ndc = clip[:, :2] / w_safe[:, None]
        valid = ((np.abs(w) >= 1e-6)
                 & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
                 & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0))
        x = (ndc[:, 0] + 1.0) * 0.5 * max(float(self.width - 1), 1.0)
        y = (ndc[:, 1] + 1.0) * 0.5 * max(float(self.height - 1), 1.0)
        return np.stack([x, y], axis=1), valid, w


def load_camera(scene_root, camera_name, image_width=None, image_height=None,
                zfar=100.0):
    """Build the named camera from the scene's transforms JSONs.

    Searched in `transforms_test.json` then `transforms_train.json`, matching
    the camera by the prefix of `file_path`'s basename -- which is how
    `camera_id_of` identifies a camera in the estimator itself
    (estimate_episodes.py:1402-1412).

    `image_width`/`image_height` are the size of the images actually being
    drawn; the intrinsics are divided by `w_json / image_width`, which is the
    `scale` of `loadCam` (camera_utils.py:22-45).
    """
    root = Path(scene_root)
    matrix = None
    contents = None
    for name in ("transforms_test.json", "transforms_train.json"):
        path = root / name
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        for frame in data.get("frames", []):
            if Path(frame["file_path"]).name.startswith(camera_name):
                matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
                contents = data
                break
        if matrix is not None:
            break
    if matrix is None:
        raise VizError("camera %r is in neither transforms_test.json nor "
                       "transforms_train.json under %s"
                       % (camera_name, scene_root))
    for key in ("w", "h", "fl_x", "fl_y", "cx", "cy"):
        if key not in contents:
            raise VizError("transforms file has no %r; this tool only handles "
                           "the fl_x/fl_y/cx/cy intrinsics branch "
                           "(dataset_readers.py:444-451)" % key)
    json_width = float(contents["w"])
    json_height = float(contents["h"])
    width = int(image_width or round(json_width))
    height = int(image_height or round(json_height))
    scale = json_width / float(width)
    if abs(json_height / float(height) - scale) > 1e-6:
        raise VizError("image aspect %dx%d does not match the transforms "
                       "%gx%g" % (width, height, json_width, json_height))

    c2w = matrix.copy()
    c2w[:3, 1:3] *= -1          # dataset_readers.py:391
    w2c = np.linalg.inv(c2w)    # dataset_readers.py:394
    rotation = w2c[:3, :3].T    # dataset_readers.py:396 (stored transposed)
    translation = w2c[:3, 3]    # dataset_readers.py:397
    return PinholeCamera(
        rotation, translation,
        cx=float(contents["cx"]) / scale, cy=float(contents["cy"]) / scale,
        fl_x=float(contents["fl_x"]) / scale, fl_y=float(contents["fl_y"]) / scale,
        width=width, height=height, zfar=zfar, name=camera_name,
    )


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def convex_hull(points):
    """Monotone-chain convex hull of an (N, 2) array, counter-clockwise."""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=np.float64).tolist())))
    if len(pts) <= 2:
        return np.asarray(pts, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def cell_polygon(camera, key, cells, lo, span):
    """The 2D bounding polygon of one cell, or None if it is not projectable.

    A cell with any corner at or behind the camera plane is skipped rather
    than drawn: its projection is not a bounded polygon.
    """
    low, high = cell_box(key, cells, lo, span)
    corners = box_corners(low, high)
    xy, _valid, w = camera.project(corners)
    if np.any(w <= 1e-6) or not np.all(np.isfinite(xy)):
        return None
    return convex_hull(xy)


def visible_polygon(polygon, camera, margin=0.0):
    """True iff the polygon's bounding box intersects the image rectangle."""
    if polygon is None or len(polygon) < 3:
        return False
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    return (x1 >= -margin and y1 >= -margin
            and x0 <= camera.width + margin and y0 <= camera.height + margin)


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def auto_frames(truth, count, pad=10):
    """`count` evenly spaced frames spanning [truth[0] - pad, truth[1] + pad]."""
    if count < 2:
        raise VizError("--n_frames must be at least 2")
    start = int(truth[0]) - int(pad)
    end = int(truth[1]) + int(pad)
    frames = np.unique(np.rint(np.linspace(start, end, int(count))).astype(int))
    return [int(f) for f in frames]


def frame_path(frames_dir, camera_name, frame, pattern="%s_%04d", extensions=(".png", ".jpg", ".jpeg")):
    base = pattern % (camera_name, int(frame))
    for extension in extensions:
        candidate = Path(frames_dir) / (base + extension)
        if candidate.is_file():
            return candidate
    raise VizError("no image for %s frame %d under %s (tried %s)"
                   % (camera_name, frame, frames_dir,
                      ", ".join(base + e for e in extensions)))


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


def draw_frame_axis(axis, image, decisions, polygons, camera, truth, frame,
                    fired, label_fired=True):
    """One montage column: the frame with every projectable cell drawn on it."""
    axis.imshow(image)
    axis.set_xlim(0, camera.width)
    axis.set_ylim(camera.height, 0)
    axis.set_xticks([])
    axis.set_yticks([])

    by_outcome = {}
    drawn = 0
    for record in decisions:
        group = int(record["group"])
        polygon = polygons.get(group)
        if not visible_polygon(polygon, camera):
            continue
        outcome = outcome_of(record)
        if outcome == GATED:
            continue  # drawn on top, below
        by_outcome.setdefault(outcome, []).append(polygon)
        drawn += 1
    for outcome in OUTCOME_ORDER:
        polys = by_outcome.get(outcome)
        if not polys:
            continue
        axis.add_collection(PolyCollection(
            polys, facecolors="none", edgecolors=OUTCOME_COLOURS[outcome],
            linewidths=0.45, alpha=0.75, zorder=2))

    for record in fired:
        group = int(record["group"])
        polygon = polygons.get(group)
        if polygon is None or len(polygon) < 3:
            continue
        axis.add_collection(PolyCollection(
            [polygon], facecolors=OUTCOME_COLOURS[GATED], edgecolors=OUTCOME_COLOURS[GATED],
            linewidths=2.4, alpha=0.16, zorder=4))
        axis.add_collection(PolyCollection(
            [polygon], facecolors="none", edgecolors=OUTCOME_COLOURS[GATED],
            linewidths=2.4, zorder=5))
        drawn += 1
        if label_fired:
            offset = record.get("offset_frame")
            onset = record.get("onset_frame")
            text = "g%d" % group
            if offset is not None and onset is not None:
                text += "  gap [%d, %d]" % (int(offset), int(onset) - 1)
            axis.text(float(polygon[:, 0].mean()), float(polygon[:, 1].min()) - 6.0,
                      text, color="white", fontsize=7.0, ha="center", va="bottom",
                      zorder=6,
                      bbox=dict(facecolor=OUTCOME_COLOURS[GATED], edgecolor="none",
                                alpha=0.85, pad=1.4))

    inside = truth is not None and int(truth[0]) <= int(frame) <= int(truth[1])
    title = "frame %d" % int(frame)
    if truth is not None:
        title += " -- %s" % ("IN the window" if inside else "outside")
    axis.set_title(title, fontsize=9,
                   color=("#b30000" if inside else "#222222"),
                   fontweight=("bold" if inside else "normal"))
    if inside:
        for spine in axis.spines.values():
            spine.set_edgecolor("#b30000")
            spine.set_linewidth(2.0)
    return drawn


def draw_group_strip(axis, record, report, truth, has_series):
    """One fired group's timeline: per-camera markers, the gap, the window."""
    sampling = report["sampling"]
    frame_range = sampling.get("frame_range") or [0, sampling["n_frames"] - 1]
    evaluated = [int(f) for f in sampling.get("evaluated_frames", [])]
    start, end = int(frame_range[0]), int(frame_range[1])
    group = int(record["group"])
    offset = record.get("offset_frame")
    onset = record.get("onset_frame")

    per_camera = record.get("per_camera", {}) or {}
    camera_ids = sorted(per_camera.keys(), key=lambda c: int(c))
    rows = max(1, len(camera_ids))

    axis.set_xlim(start - 1, end + 1)
    axis.set_ylim(-1.2, rows + 0.4)
    axis.set_yticks(range(rows))
    axis.set_yticklabels(["cam%02d" % int(c) for c in camera_ids] or ["--"],
                         fontsize=7)
    axis.tick_params(axis="x", labelsize=7)
    axis.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=0)

    if truth is not None:
        axis.axvspan(int(truth[0]) - 0.5, int(truth[1]) + 0.5, facecolor="#000000",
                     alpha=0.07, zorder=1)
        for edge in (int(truth[0]) - 0.5, int(truth[1]) + 0.5):
            axis.axvline(edge, color="#000000", linestyle="--", linewidth=1.2,
                         zorder=3)
        axis.text(0.5 * (int(truth[0]) + int(truth[1])), rows + 0.2,
                  "truth window [%d, %d]" % (int(truth[0]), int(truth[1])),
                  fontsize=6.5, color="#000000", ha="center", va="top")
    if offset is not None and onset is not None:
        axis.axvspan(int(offset) - 0.5, int(onset) - 0.5,
                     facecolor=OUTCOME_COLOURS[GATED], alpha=0.22, zorder=2)

    if evaluated:
        axis.plot(evaluated, [-0.75] * len(evaluated), marker="|", linestyle="none",
                  color="#555555", markersize=4, zorder=3)
        axis.text(start, -1.05, "evaluated frames (n=%d)" % len(evaluated),
                  fontsize=6, color="#555555", va="bottom", ha="left")

    for row, camera in enumerate(camera_ids):
        entry = per_camera[camera] or {}
        reason = entry.get("reason")
        if reason:
            axis.text(start + 0.5, row, "abstained: %s" % reason, fontsize=7,
                      color="#777777", va="center", ha="left", zorder=5)
            continue
        cam_offset = entry.get("offset_frame")
        cam_onset = entry.get("onset_frame")
        if cam_offset is None or cam_onset is None:
            continue
        axis.plot([int(cam_offset), int(cam_onset) - 1], [row, row], color="#333333",
                  linewidth=2.0, solid_capstyle="butt", zorder=5)
        axis.plot([int(cam_offset)], [row], marker="v", color="#333333",
                  markersize=6, zorder=6)
        axis.plot([int(cam_onset)], [row], marker="^", color="#333333",
                  markersize=6, zorder=6)
        axis.text(int(cam_onset) + 1.0, row, "off %d / on %d"
                  % (int(cam_offset), int(cam_onset)), fontsize=6.5,
                  color="#333333", va="center", ha="left", zorder=6)

    contrast = record.get("contrast") or {}
    parts = ["group %d" % group, "%d rows" % int(record.get("rows", 0)),
             "%d agreeing cameras" % int(record.get("agreeing_cameras", 0))]
    if offset is not None and onset is not None:
        parts.append("estimated gap [%d, %d]" % (int(offset), int(onset) - 1))
    if contrast:
        parts.append("contrast sep %.4g / within-mode scale %.4g (n_high %s, "
                     "n_low %s, range %.4g)"
                     % (contrast.get("separation") or float("nan"),
                        contrast.get("within_mode_scale") or float("nan"),
                        contrast.get("n_high"), contrast.get("n_low"),
                        contrast.get("range") or float("nan")))
    axis.set_title("  |  ".join(parts), fontsize=8, loc="left")
    if not has_series:
        axis.text(end, rows + 0.2,
                  "no per-frame series in the report: markers only",
                  fontsize=6.5, color="#b30000", ha="right", va="top")
    axis.set_xlabel("frame", fontsize=7)


def build_figure(report, program, decisions, polygons, camera, frames, images,
                 truth, out_path, title, provenance, has_series, dpi=130):
    import textwrap

    fired = [record for record in decisions if record.get("gated")]
    n_columns = len(frames)
    n_strips = max(1, len(fired))
    column_width = 4.0
    frame_height = column_width * camera.height / float(camera.width)
    strip_height = 1.6
    fig_width = column_width * n_columns
    # The header is measured in inches so the montage never collides with it,
    # whatever the provenance block's length or the figure's aspect.
    wrap_at = max(80, int(fig_width * 17))
    provenance_lines = []
    for paragraph in provenance.split("\n"):
        provenance_lines.extend(textwrap.wrap(paragraph, wrap_at) or [""])
    header = 0.50 + 0.145 * len(provenance_lines) + 0.30
    footer = 0.55
    fig_height = header + frame_height + n_strips * strip_height + footer
    figure = plt.figure(figsize=(fig_width, fig_height))
    grid = figure.add_gridspec(
        1 + n_strips, n_columns,
        height_ratios=[frame_height] + [strip_height] * n_strips,
        hspace=0.30, wspace=0.03,
        top=1.0 - (header + 0.22) / fig_height, bottom=footer / fig_height,
        left=0.030, right=0.995)

    drawn = 0
    for column, frame in enumerate(frames):
        axis = figure.add_subplot(grid[0, column])
        drawn = draw_frame_axis(axis, images[column], decisions, polygons,
                                camera, truth, frame, fired)

    for index in range(n_strips):
        axis = figure.add_subplot(grid[1 + index, :])
        if fired:
            draw_group_strip(axis, fired[index], report, truth, has_series)
        else:
            axis.axis("off")
            axis.text(0.5, 0.5, "no group was gated: no strip to draw",
                      fontsize=10, ha="center", va="center")

    counts = outcome_counts(decisions)
    handles = []
    for outcome in OUTCOME_ORDER:
        if counts[outcome] == 0 and outcome != GATED:
            continue
        label = "%s (%d)" % (outcome, counts[outcome])
        if outcome == GATED:
            handles.append(Patch(facecolor=OUTCOME_COLOURS[outcome], alpha=0.5,
                                 edgecolor=OUTCOME_COLOURS[outcome],
                                 linewidth=2.0, label=label))
        else:
            handles.append(Patch(facecolor="none", edgecolor=OUTCOME_COLOURS[outcome],
                                 linewidth=1.4, label=label))
    figure.legend(handles=handles, loc="upper right", ncol=len(handles),
                  fontsize=8.5, frameon=False,
                  bbox_to_anchor=(0.995, 1.0 - 0.14 / fig_height))
    figure.text(0.012, 1.0 - 0.14 / fig_height, title, fontsize=12, ha="left",
                va="top")
    figure.text(0.012, 1.0 - 0.58 / fig_height, "\n".join(provenance_lines),
                fontsize=7.4, va="top", ha="left", color="#333333",
                linespacing=1.45)
    figure.text(0.030, 0.06 / fig_height,
                "%d of %d cells drawn on the last column (a cell whose box has "
                "a corner at or behind the camera plane, or that falls wholly "
                "outside the image, is skipped)." % (drawn, len(decisions)),
                fontsize=7, va="bottom", ha="left", color="#555555")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="Visualise a T1 visibility-gap estimator report over cam00")
    parser.add_argument("--report", required=True,
                        help="the %s report JSON" % REPORT_SCHEMA)
    parser.add_argument("--program", default="",
                        help="the v2 program JSON (optional; cross-checks the "
                             "gated groups' cell keys)")
    parser.add_argument("--scene", required=True,
                        help="scene root holding transforms_{test,train}.json "
                             "and images/")
    parser.add_argument("--frames_dir", default="",
                        help="frame directory (default: <scene>/images)")
    parser.add_argument("--camera", default="cam00")
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        help="explicit frames for the montage columns")
    parser.add_argument("--n_frames", type=int, default=6,
                        help="how many frames to span the window with when "
                             "--frames is omitted")
    parser.add_argument("--pad", type=int, default=10,
                        help="frames to extend the montage either side of the "
                             "window")
    parser.add_argument("--truth", nargs=2, type=int, default=None,
                        metavar=("A", "B"),
                        help="the authored or curated absent window, INCLUSIVE")
    parser.add_argument("--truth_label", default="window",
                        help="what the --truth window is, for the title")
    parser.add_argument("--xyz_npy", default="",
                        help="(N, 3) float array of the cloud the estimate ran on")
    parser.add_argument("--checkpoint", default="",
                        help="checkpoint to read _xyz from (lazily imports torch)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--allow_row_mismatch", action="store_true",
                        help="draw anyway when the recomputed grouping "
                             "disagrees with the report, and say so on the "
                             "figure. Off by default: a silent mismatch means "
                             "the polygons are not the cells that were decided")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])

    report = load_report(args.report)
    program = None
    if args.program:
        with open(args.program, "r", encoding="utf-8") as handle:
            program = json.load(handle)
    decisions = report["diagnostics"]["decisions"]

    has_series = report_has_series(report)
    if not has_series:
        print(NO_SERIES_WARNING)

    if not args.xyz_npy and not args.checkpoint:
        raise VizError("one of --xyz_npy / --checkpoint is required: the "
                       "report identifies a group only by index, so the cloud "
                       "is needed to recover every cell's box")
    xyz = (load_xyz_from_npy(args.xyz_npy) if args.xyz_npy
           else load_xyz_from_checkpoint(args.checkpoint))
    if int(xyz.shape[0]) != int(report["grouping"]["n_rows"]):
        raise VizError("the cloud has %d rows but the report was computed on "
                       "%d" % (xyz.shape[0], report["grouping"]["n_rows"]))

    cells, lo, span = grid_spec_of(report)
    min_rows = int(report["grouping"].get("min_group_rows", 4))
    labels, kept_keys = build_voxel_groups(xyz, cells, lo, span, min_rows)
    ok, check_message = verify_grouping(report, program, labels, kept_keys)
    print(check_message)
    if not ok and not args.allow_row_mismatch:
        raise VizError(check_message + " -- pass --allow_row_mismatch to draw "
                       "anyway with the failure recorded on the figure")

    frames_dir = args.frames_dir or os.path.join(args.scene, "images")
    frames = args.frames
    if frames is None:
        if args.truth is None:
            raise VizError("pass --frames, or --truth so the frames can span it")
        frames = auto_frames(args.truth, int(args.n_frames), int(args.pad))
    paths = [frame_path(frames_dir, args.camera, frame) for frame in frames]
    images = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    height, width = images[0].shape[:2]
    camera = load_camera(args.scene, args.camera, image_width=width,
                         image_height=height)
    print("camera %s: %dx%d fl=(%.3f, %.3f) c=(%.1f, %.1f)"
          % (camera.name, camera.width, camera.height, camera.fl_x, camera.fl_y,
             camera.cx, camera.cy))

    polygons = {}
    for index, key in enumerate(kept_keys):
        polygons[index] = cell_polygon(camera, key, cells, lo, span)

    counts = outcome_counts(decisions)
    title = args.title or ("T1 visibility-gap estimator -- %s"
                           % Path(args.report).parent.name)
    provenance = (
        "report %s  |  program sha256 %s  |  %d cells (%d^3 grid over the %s "
        "percentile box)  |  measured frames %s, stride %s, cameras %s  |  "
        "%s  |  %s  |  drawn on %s (%s)"
        % (args.report, str(report.get("program_sha256"))[:16],
           report["grouping"]["n_groups"], cells,
           report["grouping"].get("grid_percentile"),
           report["sampling"].get("frame_range"),
           report["sampling"].get("coarse_stride"),
           report["sampling"].get("train_camera_ids_used"),
           check_message,
           "gated %d  |  " % counts[GATED] + ", ".join(
               "%s %d" % (name, counts[name]) for name in OUTCOME_ORDER[1:]
               if counts[name]),
           args.camera,
           "held out of training" if args.camera == "cam00" else "training camera"))
    if args.truth is not None:
        provenance += (
            "\n%s: frames [%d, %d] -- a montage column inside it is outlined "
            "red, and the strip marks it with a grey band between dashed "
            "lines; the RED band on the strip is the ESTIMATED gap"
            % (args.truth_label, int(args.truth[0]), int(args.truth[1])))
    if not has_series:
        provenance += "\n" + NO_SERIES_WARNING

    out = build_figure(report, program, decisions, polygons, camera, frames,
                       images, args.truth, args.out, title, provenance,
                       has_series, dpi=int(args.dpi))
    print("wrote %s" % out)
    return out


if __name__ == "__main__":
    main()
