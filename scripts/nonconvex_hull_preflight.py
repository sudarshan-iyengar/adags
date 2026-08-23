#!/usr/bin/env python3
"""CPU analytic preflight for the LRV5-NCX non-convex hull-completion falsifier.

PURE numpy + stdlib. ZERO GPU, no torch, no rendering, no training.

It proves, by computation, the five properties the frozen spec depends on:

  P1  the event object is CONNECTED at the actual voxel resolution
      (6-connectivity flood fill over analytically-occupied cells);
  P2  the object is genuinely NON-CONVEX at that resolution -- its
      axis-aligned CELL hull strictly contains at least one cell whose
      object volume is EXACTLY zero and whose object row count is zero;
  P3  the empty cell is not a grid artefact -- the concavity survives at
      cells_per_axis = 8 AND at cells_per_axis = 16 (2x resolution), and
      across a phase/scale sweep covering the plausible trained-cloud grid;
  P4  the object's volume fraction inside its hull cells (the LRV3 analogue
      was 12.20% at the fresh grid / 6.6% at the trained grid, for a sphere
      in 8 cells);
  P5  the background-row occupancy of the concavity cell -- if the concavity
      is empty of background too, hull completion costs nothing and the
      falsifier is toothless.

Both predeclared orientations (O1 primary, O2 second) are evaluated.

Cell volumes are computed ANALYTICALLY and EXACTLY (axis-aligned box /
axis-aligned cell intersection with inclusion-exclusion), never by sampling,
so "zero object volume in this cell" is an exact statement, not a sampling
artefact. A lattice is used only for the surface-supply proxy.

Grid construction mirrors scripts/estimate_episodes.py::voxel_grid exactly:
    lo   = cloud.min(axis=0)
    span = clamp_min(cloud.max(axis=0) - lo, 1e-6)
    idx  = clip(int((p - lo) / span * cells), 0, cells - 1)
    key  = i*cells*cells + j*cells + k
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

# ---------------------------------------------------------------------------
# FROZEN FIXTURE CONSTANTS -- must match nonconvex-hull-spec.md exactly.
# ---------------------------------------------------------------------------

SCENE_ID = "LRV5-NCX"

# Initialization cloud: unchanged from LRV1/LRV2/LRV3.
N_INIT_PTS = 50_000
INIT_HALF_WIDTH = 1.3
INIT_SEED = 0

# Voxel grouping: unchanged from the T1 estimator.
CELLS_PER_AXIS = 8
CELLS_PER_AXIS_2X = 16
MIN_GROUP_ROWS = 4

# ---- Orientation O1 (PRIMARY) --------------------------------------------
# Event object = union of two axis-aligned boxes joined along a corner block.
# Hull half-extent 0.64 in x and z; arm thickness 0.40; notch 0.88 x 0.88.
ARM_T = 0.40
HULL_H = 0.64
OBJ_Y = 0.30
NOTCH_LO = -HULL_H + ARM_T          # -0.24
NOTCH_HI = HULL_H                   # +0.64

EVENT_BOXES_O1 = (
    # arm A: the z-arm (long in z, thin in x)
    ((-HULL_H, -OBJ_Y, -HULL_H), (-HULL_H + ARM_T, OBJ_Y, HULL_H)),
    # arm B: the x-arm (long in x, thin in z)
    ((-HULL_H, -OBJ_Y, -HULL_H), (HULL_H, OBJ_Y, -HULL_H + ARM_T)),
)

# Notch filler: a STATIC vertical cross of two thin walls standing in the
# concavity, present at EVERY frame. It exists so that gating the concavity
# is measurably wrong: hull completion would suppress a persistent,
# always-visible object for the whole 27-frame absence gap.
# 0.12 clearance from both inner faces of the L.
FILLER_BOXES_O1 = (
    ((-0.12, -OBJ_Y, 0.14), (0.52, OBJ_Y, 0.26)),   # wall spanning x, thin in z
    ((0.14, -OBJ_Y, -0.12), (0.26, OBJ_Y, 0.52)),   # wall spanning z, thin in x
)

# ---- Orientation O2 (SECOND, PREDECLARED) --------------------------------
# mirror in x, then translate by half a fresh cell in x and z.
HALF_CELL = (2.0 * INIT_HALF_WIDTH / CELLS_PER_AXIS) / 2.0   # 0.1625


def _mirror_shift_box(box):
    (lx, ly, lz), (hx, hy, hz) = box
    nlx, nhx = -hx + HALF_CELL, -lx + HALF_CELL
    return ((nlx, ly, lz + HALF_CELL), (nhx, hy, hz + HALF_CELL))


EVENT_BOXES_O2 = tuple(_mirror_shift_box(b) for b in EVENT_BOXES_O1)
FILLER_BOXES_O2 = tuple(_mirror_shift_box(b) for b in FILLER_BOXES_O1)

# Static context spheres (identical in BOTH orientations; relocated from LRV3
# so that neither orientation intersects them).
STATIC_SPHERES = (
    ((-0.95, -0.20, 0.85), 0.28),
    ((0.95, 0.15, -0.95), 0.26),
    ((-0.30, -0.40, 1.05), 0.22),
)
GROUND_Y = -0.62
GROUND_HALF_EXTENT = 1.3

ORIENTATIONS = {
    "O1": (EVENT_BOXES_O1, FILLER_BOXES_O1),
    "O2": (EVENT_BOXES_O2, FILLER_BOXES_O2),
}

# LRV3 reference numbers, for the comparison lines only.
LRV3_SPHERE_R = 0.20
LRV3_SPHERE_C = (0.70, 0.10, 0.35)
LRV3_TRAINED_ROWS = 149_794
LRV3_OBJECT_ROWS_TRAINED = 10_650      # 11,330 gated at precision 0.9400
LRV3_AREAL_ROW_DENSITY = LRV3_OBJECT_ROWS_TRAINED / (4.0 * np.pi * LRV3_SPHERE_R ** 2)

FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print("  [%s] %s%s" % (tag, name, ("  -- " + detail) if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


# ---------------------------------------------------------------------------
# exact axis-aligned geometry
# ---------------------------------------------------------------------------


def box_intersection(a, b):
    """Intersection of two axis-aligned boxes, or None."""
    lo = np.maximum(np.asarray(a[0], float), np.asarray(b[0], float))
    hi = np.minimum(np.asarray(a[1], float), np.asarray(b[1], float))
    if np.any(hi <= lo):
        return None
    return (lo, hi)


def box_volume(box):
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    return float(np.prod(np.clip(hi - lo, 0.0, None)))


def union_volume(boxes):
    """Exact volume of a union of axis-aligned boxes (inclusion-exclusion)."""
    total = 0.0
    n = len(boxes)
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            cur = boxes[combo[0]]
            for idx in combo[1:]:
                cur = box_intersection(cur, boxes[idx])
                if cur is None:
                    break
            if cur is None:
                continue
            total += ((-1.0) ** (r + 1)) * box_volume(cur)
    return total


def axis_overlap_lengths(blo, bhi, edges):
    """Per-cell overlap length of [blo, bhi] with consecutive cell edges."""
    lo = np.maximum(blo, edges[:-1])
    hi = np.minimum(bhi, edges[1:])
    return np.clip(hi - lo, 0.0, None)


def per_cell_box_volume(box, edges_xyz):
    """(C,C,C) exact volume of one axis-aligned box inside each cell."""
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    lx = axis_overlap_lengths(lo[0], hi[0], edges_xyz[0])
    ly = axis_overlap_lengths(lo[1], hi[1], edges_xyz[1])
    lz = axis_overlap_lengths(lo[2], hi[2], edges_xyz[2])
    return lx[:, None, None] * ly[None, :, None] * lz[None, None, :]


def per_cell_union_volume(boxes, edges_xyz):
    """(C,C,C) exact volume of a UNION of axis-aligned boxes inside each cell."""
    shape = (len(edges_xyz[0]) - 1, len(edges_xyz[1]) - 1, len(edges_xyz[2]) - 1)
    total = np.zeros(shape, dtype=np.float64)
    n = len(boxes)
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            cur = boxes[combo[0]]
            for idx in combo[1:]:
                cur = box_intersection(cur, boxes[idx])
                if cur is None:
                    break
            if cur is None:
                continue
            total += ((-1.0) ** (r + 1)) * per_cell_box_volume(cur, edges_xyz)
    return total


def cell_edges(lo, span, cells):
    """Cell edges per axis for the estimator's absolute grid."""
    return [lo[a] + span[a] * np.arange(cells + 1) / float(cells) for a in range(3)]


def point_cell_index(points, lo, span, cells):
    """Exactly scripts/estimate_episodes.py::voxel_grid, in numpy."""
    v = (points - lo) / span * cells
    return np.clip(v.astype(np.int64), 0, cells - 1)


def boxes_contain(points, boxes):
    inside = np.zeros(points.shape[0], dtype=bool)
    for lo, hi in boxes:
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        inside |= np.all((points >= lo) & (points <= hi), axis=1)
    return inside


# ---------------------------------------------------------------------------
# cell-set operators
# ---------------------------------------------------------------------------


def flood_components(occ):
    """6-connected components of a boolean (C,C,C) occupancy array."""
    cells = occ.shape[0]
    labels = -np.ones(occ.shape, dtype=np.int64)
    comp = 0
    for start in zip(*np.nonzero(occ)):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = comp
        while stack:
            i, j, k = stack.pop()
            for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                               (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                n = (i + di, j + dj, k + dk)
                if not all(0 <= c < s for c, s in zip(n, occ.shape)):
                    continue
                if occ[n] and labels[n] < 0:
                    labels[n] = comp
                    stack.append(n)
        comp += 1
    return labels, comp


def shortest_path_cells(occ, start, goal):
    """Shortest 6-connected path through occupied cells (BFS)."""
    from collections import deque
    prev = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            n = (cur[0] + di, cur[1] + dj, cur[2] + dk)
            if not all(0 <= c < s for c, s in zip(n, occ.shape)):
                continue
            if occ[n] and n not in prev:
                prev[n] = cur
                queue.append(n)
    if goal not in prev:
        return []
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def index_bbox(occ):
    idx = np.nonzero(occ)
    return [(int(a.min()), int(a.max())) for a in idx]


def bbox_fill_mask(occ):
    """H1: axis-aligned CELL bounding-box fill, applied PER 6-CONNECTED
    COMPONENT of the accepted set, union of the results."""
    out = np.zeros_like(occ)
    labels, n = flood_components(occ)
    for c in range(n):
        comp = labels == c
        (i0, i1), (j0, j1), (k0, k1) = index_bbox(comp)
        out[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = True
    return out


def closing_mask(occ, radius=1):
    """H2: 3x3x3 morphological closing (dilate then erode), clipped to the
    accepted set's own index bbox so it can never grow outward."""
    def dilate(m):
        out = np.zeros_like(m)
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                for dk in range(-radius, radius + 1):
                    out |= np.roll(np.roll(np.roll(m, di, 0), dj, 1), dk, 2)
        return out

    def erode(m):
        out = np.ones_like(m)
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                for dk in range(-radius, radius + 1):
                    out &= np.roll(np.roll(np.roll(m, di, 0), dj, 1), dk, 2)
        return out

    pad = radius + 1
    padded = np.zeros([s + 2 * pad for s in occ.shape], dtype=bool)
    padded[pad:-pad, pad:-pad, pad:-pad] = occ
    closed = erode(dilate(padded))[pad:-pad, pad:-pad, pad:-pad]
    closed |= occ                                  # closing is extensive
    bbox = np.zeros_like(occ)
    (i0, i1), (j0, j1), (k0, k1) = index_bbox(occ)
    bbox[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = True
    return closed & bbox


# ---------------------------------------------------------------------------
# surface-supply proxy (lattice shell count)
# ---------------------------------------------------------------------------

LAT_N = 256
LAT_HALF = 1.35


def lattice_axes():
    return [np.linspace(-LAT_HALF, LAT_HALF, LAT_N) for _ in range(3)]


def lattice_mask(boxes, axes):
    m = np.zeros((LAT_N, LAT_N, LAT_N), dtype=bool)
    for lo, hi in boxes:
        mx = (axes[0] >= lo[0]) & (axes[0] <= hi[0])
        my = (axes[1] >= lo[1]) & (axes[1] <= hi[1])
        mz = (axes[2] >= lo[2]) & (axes[2] <= hi[2])
        m |= mx[:, None, None] & my[None, :, None] & mz[None, None, :]
    return m


def shell_mask(mask):
    """Boundary shell: occupied lattice sites with a non-occupied 6-neighbour."""
    inner = mask.copy()
    for ax in range(3):
        for sh in (1, -1):
            inner &= np.roll(mask, sh, axis=ax)
    return mask & ~inner


def shell_counts_per_cell(shell, axes, lo, span, cells):
    """(C,C,C) count of boundary-shell lattice sites falling in each cell."""
    idx = [np.clip((((axes[a] - lo[a]) / span[a]) * cells).astype(np.int64),
                   0, cells - 1) for a in range(3)]
    sites = np.nonzero(shell)
    keys = (idx[0][sites[0]] * cells * cells
            + idx[1][sites[1]] * cells + idx[2][sites[2]])
    counts = np.bincount(keys, minlength=cells ** 3)
    return counts.reshape(cells, cells, cells)


# ---------------------------------------------------------------------------
# camera rig (mirrors scripts/build_synthetic_reveal_scene.py::camera_poses)
# ---------------------------------------------------------------------------

N_CAMERAS = 20
TEST_CAMERAS = (2, 7, 12, 17)
RIG_RADIUS = 3.3   # CHANGED from LRV3's 2.4: the larger object needs it
RIG_ELEVATIONS_DEG = (12.0, 28.0)
ESTIMATOR_CAMERAS = 4          # scripts/estimate_episodes.py --cameras
MIN_AGREEING_CAMERAS = 3       # frozen in the T1 estimator


WIDTH, HEIGHT, FOCAL = 400, 300, 420.0


def camera_frames():
    """(eye, R) per camera, verbatim build_synthetic_reveal_scene::camera_poses."""
    out = []
    up = np.array([0.0, 1.0, 0.0])
    for i in range(N_CAMERAS):
        az = np.deg2rad(360.0 * i / N_CAMERAS)
        el = np.deg2rad(RIG_ELEVATIONS_DEG[i % len(RIG_ELEVATIONS_DEG)])
        eye = np.array([RIG_RADIUS * np.cos(el) * np.cos(az),
                        RIG_RADIUS * np.sin(el),
                        RIG_RADIUS * np.cos(el) * np.sin(az)])
        fwd = -eye / np.linalg.norm(eye)
        x_axis = np.cross(fwd, up)
        x_axis /= np.linalg.norm(x_axis)
        z_axis = -fwd
        y_axis = np.cross(z_axis, x_axis)
        out.append((eye, np.stack([x_axis, y_axis, z_axis], axis=1)))
    return out


def camera_eyes():
    return [f[0] for f in camera_frames()]


def project(points, eye, R):
    """(pixel_i, pixel_j, in_front) under the fixture's OpenGL intrinsics."""
    v = (points - eye[None, :]) @ R
    in_front = v[:, 2] < -1e-6
    depth = np.where(in_front, -v[:, 2], 1.0)
    a = v[:, 0] / depth
    b = v[:, 1] / depth
    return a * FOCAL + WIDTH / 2.0 - 0.5, -b * FOCAL + HEIGHT / 2.0 - 0.5, in_front


def frame_occupancy(points, rig_radius):
    """(worst normalised frame occupancy, list of clipping cameras)."""
    saved = globals()["RIG_RADIUS"]
    globals()["RIG_RADIUS"] = rig_radius
    try:
        worst, bad = 0.0, []
        for cam, (eye, R) in enumerate(camera_frames()):
            pi, pj, front = project(points, eye, R)
            worst = max(worst,
                        float(np.max(np.abs(pi - (WIDTH / 2.0 - 0.5)))
                              / (WIDTH / 2.0)),
                        float(np.max(np.abs(pj - (HEIGHT / 2.0 - 0.5)))
                              / (HEIGHT / 2.0)))
            if (not front.all() or pi.min() < 0 or pi.max() > WIDTH - 1
                    or pj.min() < 0 or pj.max() > HEIGHT - 1):
                bad.append(cam)
        return worst, bad
    finally:
        globals()["RIG_RADIUS"] = saved


def ray_box_t(origin, dirs, box):
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    d = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    t1 = (lo[None, :] - origin[None, :]) / d
    t2 = (hi[None, :] - origin[None, :]) / d
    tmin = np.minimum(t1, t2).max(axis=1)
    tmax = np.maximum(t1, t2).min(axis=1)
    t = np.where(tmin > 1e-4, tmin, tmax)
    return np.where((tmax >= np.maximum(tmin, 0.0)) & (t > 1e-4), t, np.inf)


def ray_sphere_t(origin, dirs, centre, radius):
    oc = origin - np.asarray(centre, float)
    b = dirs @ oc
    c = float(oc @ oc) - radius * radius
    disc = b * b - c
    hit = disc > 0.0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t0, t1 = -b - sq, -b + sq
    t = np.where(t0 > 1e-4, t0, t1)
    return np.where(hit & (t > 1e-4), t, np.inf)


def ray_ground_t(origin, dirs):
    den = dirs[:, 1]
    t = np.where(np.abs(den) > 1e-8, (GROUND_Y - origin[1]) / den, np.inf)
    t = np.where(t > 1e-4, t, np.inf)
    p = origin[None, :] + t[:, None] * dirs
    inside = (np.abs(p[:, 0]) < GROUND_HALF_EXTENT) & (np.abs(p[:, 2]) < GROUND_HALF_EXTENT)
    return np.where(np.isfinite(t) & inside, t, np.inf)


def silhouette_pixels(cam, ev, fl, event_present=True):
    """(event px, filler px) front-most hit counts for one camera."""
    eye, R = camera_frames()[cam]
    jj, ii = np.meshgrid(np.arange(HEIGHT), np.arange(WIDTH), indexing="ij")
    px = (ii + 0.5 - WIDTH / 2.0) / FOCAL
    py = -(jj + 0.5 - HEIGHT / 2.0) / FOCAL
    dc = np.stack([px, py, -np.ones_like(px)], axis=-1).reshape(-1, 3)
    dirs = dc @ R.T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ts, tag = [ray_ground_t(eye, dirs)], [0]
    for c, r in STATIC_SPHERES:
        ts.append(ray_sphere_t(eye, dirs, c, r)); tag.append(1)
    for b in fl:
        ts.append(ray_box_t(eye, dirs, b)); tag.append(2)
    if event_present:
        for b in ev:
            ts.append(ray_box_t(eye, dirs, b)); tag.append(3)
    stack = np.stack(ts, axis=0)
    best = np.argmin(stack, axis=0)
    finite = np.isfinite(np.take_along_axis(stack, best[None], axis=0)[0])
    lab = np.where(finite, np.asarray(tag)[best], -1)
    return int((lab == 3).sum()), int((lab == 2).sum())


def box_corners(box):
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    return np.array([[lo[0] if a else hi[0], lo[1] if b else hi[1],
                      lo[2] if c else hi[2]]
                     for a in (0, 1) for b in (0, 1) for c in (0, 1)])


def select_cameras(all_ids, count):
    """Verbatim scripts/estimate_episodes.py::select_cameras."""
    ordered = sorted(all_ids)
    if count >= len(ordered):
        return ordered
    step = len(ordered) / float(count)
    return [ordered[int(i * step)] for i in range(count)]


def segment_hits_boxes(origin, targets, boxes, eps=1e-4):
    """Boolean per target: does the segment origin->target enter any box?"""
    o = np.asarray(origin, float)[None, :]
    d = targets - o
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    hit = np.zeros(targets.shape[0], dtype=bool)
    for lo, hi in boxes:
        t1 = (np.asarray(lo, float)[None, :] - o) / d
        t2 = (np.asarray(hi, float)[None, :] - o) / d
        tmin = np.maximum(np.minimum(t1, t2).max(axis=1), 0.0)
        tmax = np.minimum(np.maximum(t1, t2).min(axis=1), 1.0 - eps)
        hit |= tmin <= tmax
    return hit


# ---------------------------------------------------------------------------
# analysis of one (orientation, grid) configuration
# ---------------------------------------------------------------------------


def analyse(event_boxes, filler_boxes, lo, span, cells, cloud=None):
    edges = cell_edges(lo, span, cells)
    vol_obj = per_cell_union_volume(list(event_boxes), edges)
    vol_fill = per_cell_union_volume(list(filler_boxes), edges)
    occ = vol_obj > 0.0
    res = {
        "edges": edges, "vol_obj": vol_obj, "vol_fill": vol_fill, "occ": occ,
        "n_occupied": int(occ.sum()),
    }
    if res["n_occupied"] == 0:
        res["ok"] = False
        return res
    (i0, i1), (j0, j1), (k0, k1) = index_bbox(occ)
    hull = np.zeros_like(occ)
    hull[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = True
    empty = hull & ~occ
    res.update({
        "bbox": ((i0, i1), (j0, j1), (k0, k1)),
        "hull": hull, "n_hull": int(hull.sum()),
        "empty": empty, "n_empty": int(empty.sum()),
        "ok": True,
    })
    if cloud is not None:
        cidx = point_cell_index(cloud, lo, span, cells)
        keys = cidx[:, 0] * cells * cells + cidx[:, 1] * cells + cidx[:, 2]
        rows_all = np.bincount(keys, minlength=cells ** 3).reshape(occ.shape)
        in_obj = boxes_contain(cloud, event_boxes)
        rows_obj = np.bincount(keys[in_obj],
                               minlength=cells ** 3).reshape(occ.shape)
        in_fill = boxes_contain(cloud, filler_boxes)
        rows_fill = np.bincount(keys[in_fill],
                                minlength=cells ** 3).reshape(occ.shape)
        res.update({"rows_all": rows_all, "rows_obj": rows_obj,
                    "rows_fill": rows_fill})
    return res


def fmt_cells(mask, limit=12):
    out = []
    for c in zip(*np.nonzero(mask)):
        key = c[0] * mask.shape[0] * mask.shape[0] + c[1] * mask.shape[0] + c[2]
        out.append("(%d,%d,%d)/key%d" % (c[0], c[1], c[2], key))
        if len(out) >= limit:
            out.append("...")
            break
    return ", ".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    print("=" * 78)
    print("LRV5-NCX non-convex hull-completion falsifier -- CPU analytic preflight")
    print("=" * 78)
    print("numpy %s | no torch | no GPU | no rendering" % np.__version__)

    # ---- the fresh seeding cloud, byte-reproducible ----------------------
    rng = np.random.default_rng(INIT_SEED)
    cloud = rng.uniform(-INIT_HALF_WIDTH, INIT_HALF_WIDTH, size=(N_INIT_PTS, 3))
    lo = cloud.min(axis=0)
    span = np.clip(cloud.max(axis=0) - lo, 1e-6, None)
    print("\nfresh seeding cloud: %d rows, seed %d" % (N_INIT_PTS, INIT_SEED))
    print("  grid lo   = [%.6f %.6f %.6f]" % tuple(lo))
    print("  grid span = [%.6f %.6f %.6f]" % tuple(span))
    print("  cell size = [%.6f %.6f %.6f]  (cells_per_axis %d)"
          % (span[0] / CELLS_PER_AXIS, span[1] / CELLS_PER_AXIS,
             span[2] / CELLS_PER_AXIS, CELLS_PER_AXIS))

    # ---- geometric sanity: no interpenetration ---------------------------
    print("\n--- geometry sanity ---")
    for name, (ev, fl) in ORIENTATIONS.items():
        clash = False
        for b in ev:
            for f in fl:
                if box_intersection(b, f) is not None:
                    clash = True
        check("%s: event object and notch filler are disjoint" % name, not clash)
        sphere_clash = []
        for ci, (c, r) in enumerate(STATIC_SPHERES):
            cbox = (tuple(np.asarray(c) - r), tuple(np.asarray(c) + r))
            for b in ev + fl:
                if box_intersection(b, cbox) is not None:
                    sphere_clash.append(ci)
        check("%s: static context spheres clear of object and filler" % name,
              not sphere_clash,
              "" if not sphere_clash else "clashes with sphere(s) %s" % sphere_clash)
        extent = np.max([np.max(np.abs(np.asarray(b))) for b in ev + fl])
        check("%s: all geometry inside the init cube (max |coord| %.4f < %.2f)"
              % (name, extent, INIT_HALF_WIDTH), extent < INIT_HALF_WIDTH)
        # framing. HARD requirement on the event object and its notch filler:
        # every corner must project inside the image from every camera, so the
        # event is fully observed. The static context spheres and the ground
        # plane are allowed to run off frame, exactly as LRV3's ground does.
        pts = np.concatenate([box_corners(b) for b in ev + fl])
        occ_frac, bad = frame_occupancy(pts, RIG_RADIUS)
        check("%s: event object + notch filler project fully inside %dx%d from "
              "all %d cameras at RIG_RADIUS %.2f"
              % (name, WIDTH, HEIGHT, N_CAMERAS, RIG_RADIUS), not bad,
              "worst normalised frame occupancy %.3f%s"
              % (occ_frac, "" if not bad else "; clipped in cameras %s" % bad))
        if name == "O1":
            rlrv3 = np.asarray(LRV3_SPHERE_C)[None, :] + LRV3_SPHERE_R * np.array(
                [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
                 [0, 0, -1]])
            f3, b3 = frame_occupancy(rlrv3, 2.4)
            print("      reference: LRV3's event sphere at RIG_RADIUS 2.40 -> "
                  "worst frame occupancy %.3f, clipped in %d camera(s)"
                  % (f3, len(b3)))
            sp = np.concatenate([np.asarray(c)[None, :] + r * np.array(
                [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
                 [0, 0, -1]]) for c, r in STATIC_SPHERES])
            fs, bs = frame_occupancy(sp, RIG_RADIUS)
            print("      static context spheres at RIG_RADIUS %.2f -> worst "
                  "frame occupancy %.3f, partially off-frame in %d camera(s) "
                  "(ALLOWED)" % (RIG_RADIUS, fs, len(bs)))

    axes = lattice_axes()

    for oname in ("O1", "O2"):
        ev, fl = ORIENTATIONS[oname]
        print("\n" + "=" * 78)
        print("ORIENTATION %s" % oname)
        print("=" * 78)
        print("  event boxes :")
        for b in ev:
            print("      x[%+.4f,%+.4f] y[%+.4f,%+.4f] z[%+.4f,%+.4f]"
                  % (b[0][0], b[1][0], b[0][1], b[1][1], b[0][2], b[1][2]))
        print("  filler boxes:")
        for b in fl:
            print("      x[%+.4f,%+.4f] y[%+.4f,%+.4f] z[%+.4f,%+.4f]"
                  % (b[0][0], b[1][0], b[0][1], b[1][1], b[0][2], b[1][2]))

        r8 = analyse(ev, fl, lo, span, CELLS_PER_AXIS, cloud=cloud)
        r16 = analyse(ev, fl, lo, span, CELLS_PER_AXIS_2X, cloud=cloud)

        # ---------------- P1 connectivity -----------------------------
        print("\n[P1] CONNECTEDNESS at the actual voxel resolution")
        for label, res, cpa in (("8^3", r8, CELLS_PER_AXIS),
                                ("16^3", r16, CELLS_PER_AXIS_2X)):
            _, ncomp = flood_components(res["occ"])
            check("%s %s: occupied cells form exactly 1 six-connected component"
                  % (oname, label), ncomp == 1,
                  "%d occupied cells, %d component(s)" % (res["n_occupied"], ncomp))

        # ---------------- P2 non-convexity ----------------------------
        print("\n[P2] NON-CONVEXITY at cells_per_axis = %d" % CELLS_PER_AXIS)
        (i0, i1), (j0, j1), (k0, k1) = r8["bbox"]
        print("  cell hull index bbox : i[%d,%d] j[%d,%d] k[%d,%d]"
              % (i0, i1, j0, j1, k0, k1))
        print("  hull cells %d | occupied %d | EMPTY-of-object %d"
              % (r8["n_hull"], r8["n_occupied"], r8["n_empty"]))
        print("  empty cells: %s" % fmt_cells(r8["empty"]))
        check("%s: cell hull STRICTLY contains >=1 zero-object-volume cell" % oname,
              r8["n_empty"] >= 1, "%d such cells" % r8["n_empty"])
        max_vol_empty = float(r8["vol_obj"][r8["empty"]].max()) if r8["n_empty"] else -1
        check("%s: every 'empty' cell has EXACTLY zero object volume" % oname,
              r8["n_empty"] > 0 and max_vol_empty == 0.0,
              "max object volume over empty cells = %.3e" % max_vol_empty)
        max_rows_empty = int(r8["rows_obj"][r8["empty"]].max()) if r8["n_empty"] else -1
        check("%s: every 'empty' cell has ZERO object rows (fresh cloud)" % oname,
              max_rows_empty == 0,
              "max object rows over empty cells = %d" % max_rows_empty)

        # hull operators
        h1 = bbox_fill_mask(r8["occ"])
        h2 = closing_mask(r8["occ"])
        h1_bad = h1 & ~r8["occ"]
        h2_bad = h2 & ~r8["occ"]
        print("\n  hull operator behaviour on the FULL occupied set:")
        print("    H1 bbox-fill  : adds %d zero-object cell(s)  %s"
              % (int(h1_bad.sum()), fmt_cells(h1_bad)))
        print("    H2 3x3x3 close: adds %d zero-object cell(s)  %s"
              % (int(h2_bad.sum()), fmt_cells(h2_bad) or "(none)"))
        check("%s: H1 (bbox fill) DOES enter the concavity" % oname,
              int(h1_bad.sum()) >= 1,
              "H1 is the operator under test; a 0 here would make the "
              "falsifier vacuous")

        # H1 on MINIMAL CONNECTED accepted sets. T1 accepts only a subset of
        # the object's cells (2 of 8 on LRV3) and candidate A2 expands it by
        # TRANSITIVE ADJACENCY, so the accepted set is a 6-connected component
        # by construction. The worst case for H1 is a SHORTEST 6-connected
        # path through occupied cells between two accepted cells.
        edges8 = r8["edges"]
        vol_a = per_cell_box_volume(ev[0], edges8)
        vol_b = per_cell_box_volume(ev[1], edges8)
        only_a = (vol_a > 0) & (vol_b == 0)
        only_b = (vol_b > 0) & (vol_a == 0)
        occ_cells = [tuple(int(v) for v in c) for c in
                     np.array(list(zip(*np.nonzero(r8["occ"]))))]

        def path_bite(a, b):
            path = shortest_path_cells(r8["occ"], a, b)
            if not path:
                return None
            m = np.zeros_like(r8["occ"])
            for c in path:
                m[c] = True
            return int((bbox_fill_mask(m) & ~r8["occ"]).sum()), len(path)

        all_bites = [path_bite(a, b) for i, a in enumerate(occ_cells)
                     for b in occ_cells[i + 1:]]
        all_bites = [b for b in all_bites if b is not None]
        min_any = min(b[0] for b in all_bites)
        cross_pairs = [(a, b) for a in occ_cells if only_a[a]
                       for b in occ_cells if only_b[b]]
        cross_bites = [path_bite(a, b) for a, b in cross_pairs]
        cross_bites = [b for b in cross_bites if b is not None]
        min_cross = min(b[0] for b in cross_bites)
        print("    H1 on MINIMAL CONNECTED accepted sets (shortest 6-conn paths):")
        print("      cells belonging to ARM A only: %d | ARM B only: %d | "
              "corner block (both): %d"
              % (int(only_a.sum()), int(only_b.sum()),
                 int(((vol_a > 0) & (vol_b > 0)).sum())))
        print("      over ALL %d occupied-cell pairs : MIN zero-object cells "
              "filled = %d   <-- VACUITY MODE" % (len(all_bites), min_any))
        print("      over the %d ARM-A x ARM-B pairs : MIN zero-object cells "
              "filled = %d" % (len(cross_bites), min_cross))
        check("%s: H1 bites for EVERY accepted component that spans both arms"
              % oname, min_cross >= 1,
              "minimum over arm-A x arm-B pairs = %d" % min_cross)
        if min_any == 0:
            print("      NOTE: a component confined to ONE arm makes H1 "
                  "vacuous. This is a declared PRECONDITION of the spec, not "
                  "a defect of the geometry.")

        # ---------------- filler visibility ---------------------------
        print("\n[P5b] THE ESTIMATOR CANNOT ACCEPT THE CONCAVITY ON ITS OWN")
        train_ids = [c for c in range(N_CAMERAS) if c not in TEST_CAMERAS]
        sampled = select_cameras(train_ids, ESTIMATOR_CAMERAS)
        eyes = camera_eyes()
        fl_shell = shell_mask(lattice_mask(fl, axes))
        sites = np.array(np.nonzero(fl_shell)).T
        stride = max(1, sites.shape[0] // 4000)
        sites = sites[::stride]
        pts = np.stack([axes[0][sites[:, 0]], axes[1][sites[:, 1]],
                        axes[2][sites[:, 2]]], axis=1)
        print("  sampled train cameras (select_cameras(train_ids, %d)) = %s"
              % (ESTIMATOR_CAMERAS, sampled))
        print("  filler surface samples: %d" % pts.shape[0])
        n_changing = 0
        for cam in sampled:
            occl = segment_hits_boxes(eyes[cam], pts, ev)
            frac = float(occl.mean())
            state = ("CHANGES (bump: hidden while the object is present)"
                     if frac > 0.02 else "CONSTANT (always visible)")
            print("    cam%02d: %.1f%% of filler surface occluded by the event "
                  "object -> %s" % (cam, 100 * frac, state))
            if frac > 0.02:
                n_changing += 1
        check("%s: fewer than MIN_AGREEING_CAMERAS=%d sampled cameras see the "
              "filler change at all" % (oname, MIN_AGREEING_CAMERAS),
              n_changing < MIN_AGREEING_CAMERAS,
              "%d camera(s) change" % n_changing)

        print("  front-most pixel counts (ray cast, %dx%d = %d px):"
              % (WIDTH, HEIGHT, WIDTH * HEIGHT))
        ev_px_min = 10 ** 9
        for cam in sampled:
            e_on, f_on = silhouette_pixels(cam, ev, fl, True)
            _, f_off = silhouette_pixels(cam, ev, fl, False)
            ev_px_min = min(ev_px_min, e_on)
            print("    cam%02d: event object %6d px (%.1f%%) | filler %5d px "
                  "present -> %5d px during the gap" % (cam, e_on,
                  100.0 * e_on / (WIDTH * HEIGHT), f_on, f_off))
        check("%s: the event object is resolvable from every sampled camera "
              "(>= 2000 front-most px)" % oname, ev_px_min >= 2000,
              "minimum %d px" % ev_px_min)

        # ---------------- P3 not a grid artefact ----------------------
        print("\n[P3] THE CONCAVITY IS NOT A GRID ARTEFACT")
        (a0, a1), (b0, b1), (c0, c1) = r16["bbox"]
        print("  at 2x resolution (cells_per_axis %d): hull %d, occupied %d, "
              "EMPTY %d" % (CELLS_PER_AXIS_2X, r16["n_hull"], r16["n_occupied"],
                            r16["n_empty"]))
        check("%s: concavity survives at 2x resolution" % oname,
              r16["n_empty"] >= 1, "%d empty in-hull cells" % r16["n_empty"])
        check("%s: 2x-resolution empty cells carry zero object rows" % oname,
              r16["n_empty"] > 0 and int(r16["rows_obj"][r16["empty"]].max()) == 0)

        # phase / scale sweep over the plausible TRAINED-cloud grid
        print("\n  phase/scale sweep (models the unknown TRAINED-cloud grid):")
        print("    %-10s %-8s %-8s %-9s %-9s" %
              ("cell", "configs", "with>=1", "median", "median"))
        print("    %-10s %-8s %-8s %-9s %-9s" %
              ("size", "", "empty", "n_empty", "bite r"))
        sweep_rows = []
        for cell in (0.30, 0.325, 0.35, 0.40, 0.45, 0.50):
            span_s = np.array([cell * CELLS_PER_AXIS] * 3)
            n_ok = 0
            n_tot = 0
            nempty = []
            bites = []
            for px in np.linspace(0.0, 1.0, 11)[:-1]:
                for pz in np.linspace(0.0, 1.0, 11)[:-1]:
                    lo_s = np.array([-1.3 - px * cell, -1.3 - 0.5 * cell,
                                     -1.3 - pz * cell])
                    r = analyse(ev, fl, lo_s, span_s, CELLS_PER_AXIS)
                    n_tot += 1
                    if not r["ok"]:
                        continue
                    if r["n_empty"] >= 1:
                        n_ok += 1
                    nempty.append(r["n_empty"])
                    base = float(r["vol_obj"][r["occ"]].sum())
                    add = float(r["vol_fill"][r["empty"]].sum()) if r["n_empty"] else 0.0
                    bites.append(add / base if base > 0 else 0.0)
            sweep_rows.append((cell, n_tot, n_ok,
                               float(np.median(nempty)) if nempty else 0.0,
                               float(np.median(bites)) if bites else 0.0))
            print("    %-10.3f %-8d %-8d %-9.1f %-9.3f" % sweep_rows[-1])
        worst = min(r[2] / float(r[1]) for r in sweep_rows)
        check("%s: >=1 empty in-hull cell at EVERY swept (cell size, phase)"
              % oname, worst == 1.0,
              "worst-case coverage %.3f of configurations" % worst)

        # ---------------- P4 volume fraction --------------------------
        print("\n[P4] VOLUME FRACTION WITHIN THE HULL CELLS")
        cell_vol = float(np.prod(span / CELLS_PER_AXIS))
        obj_vol = union_volume(list(ev))
        frac = obj_vol / (r8["n_hull"] * cell_vol)
        sph_vol = 4.0 / 3.0 * np.pi * LRV3_SPHERE_R ** 3
        print("  object volume            %.6f" % obj_vol)
        print("  hull cells x cell volume %d x %.6f = %.6f"
              % (r8["n_hull"], cell_vol, r8["n_hull"] * cell_vol))
        print("  LRV5 volume fraction     %.4f  (%.2f%%)" % (frac, 100 * frac))
        print("  LRV3 analogue (sphere r=0.20 in 8 fresh cells) %.4f (%.2f%%)"
              % (sph_vol / (8 * cell_vol), 100 * sph_vol / (8 * cell_vol)))
        print("  LRV3 recorded, TRAINED grid                    0.0660 (6.60%)")
        check("%s: volume fraction computed" % oname, 0.0 < frac < 1.0)

        # ---------------- P5 background occupancy ---------------------
        print("\n[P5] BACKGROUND OCCUPANCY OF THE CONCAVITY")
        empty_idx = list(zip(*np.nonzero(r8["empty"])))
        # designated concavity cell: the empty cell holding the most filler volume
        fv = [(float(r8["vol_fill"][c]), c) for c in empty_idx]
        fv.sort(reverse=True)
        des = fv[0][1]
        des_key = des[0] * CELLS_PER_AXIS ** 2 + des[1] * CELLS_PER_AXIS + des[2]
        print("  designated concavity cell (i,j,k) = %s  key = %d"
              % (str(des), des_key))
        print("  object rows in it (fresh cloud)          %d"
              % int(r8["rows_obj"][des]))
        print("  ALL rows in it (fresh cloud)             %d"
              % int(r8["rows_all"][des]))
        print("  filler rows in it (fresh cloud)          %d"
              % int(r8["rows_fill"][des]))
        print("  fresh-cloud rows inside the event object   %d"
              % int(r8["rows_obj"].sum()))
        print("  fresh-cloud rows inside the notch filler   %d"
              % int(r8["rows_fill"].sum()))
        tot_empty_rows = int(r8["rows_all"][r8["empty"]].sum())
        tot_occ_rows = int(r8["rows_all"][r8["occ"]].sum())
        print("  rows in ALL %d empty cells / rows in %d occupied cells"
              % (r8["n_empty"], r8["n_occupied"]))
        print("     %d / %d = %.4f" % (tot_empty_rows, tot_occ_rows,
                                       tot_empty_rows / float(tot_occ_rows)))
        check("%s: designated concavity cell holds ZERO object rows" % oname,
              int(r8["rows_obj"][des]) == 0)
        check("%s: designated concavity cell holds background rows "
              "(>= MIN_GROUP_ROWS=%d)" % (oname, MIN_GROUP_ROWS),
              int(r8["rows_all"][des]) >= MIN_GROUP_ROWS,
              "%d rows" % int(r8["rows_all"][des]))

        # trained-cloud surface-supply proxy
        occ_lat_obj = lattice_mask(ev, axes)
        occ_lat_fill = lattice_mask(fl, axes)
        sh_obj = shell_counts_per_cell(shell_mask(occ_lat_obj), axes, lo, span,
                                       CELLS_PER_AXIS)
        sh_fill = shell_counts_per_cell(shell_mask(occ_lat_fill), axes, lo, span,
                                        CELLS_PER_AXIS)
        base_shell = int(sh_obj[r8["occ"]].sum())
        add_shell = int(sh_fill[r8["empty"]].sum())
        bite = add_shell / float(base_shell)
        lat_cell_area = (2 * LAT_HALF / (LAT_N - 1)) ** 2
        print("  surface-supply proxy (%d^3 lattice boundary shell):" % LAT_N)
        print("     event-object shell sites in OCCUPIED cells   %d "
              "(~%.3f world area units)" % (base_shell, base_shell * lat_cell_area))
        print("     filler shell sites in EMPTY (hull) cells     %d "
              "(~%.3f world area units)" % (add_shell, add_shell * lat_cell_area))
        print("     bite ratio r = added/base                    %.4f" % bite)
        print("     projected trained rows added by H1 at LRV3's areal density "
              "(%.0f rows/unit^2): ~%d"
              % (LRV3_AREAL_ROW_DENSITY, add_shell * lat_cell_area
                 * LRV3_AREAL_ROW_DENSITY))
        print("     precision after H1 if base precision were 0.94: %.4f"
              % (0.94 / (1.0 + bite)))
        check("%s: bite ratio r >= 0.25 (falsifier is not toothless)" % oname,
              bite >= 0.25, "r = %.4f" % bite)
        check("%s: projected precision after H1 falls below the 0.80 floor "
              "(given base precision 0.94)" % oname, 0.94 / (1.0 + bite) < 0.80,
              "%.4f" % (0.94 / (1.0 + bite)))

    print("\n" + "=" * 78)
    if FAILURES:
        print("PREFLIGHT FAILED: %d check(s)" % len(FAILURES))
        for f in FAILURES:
            print("   - %s" % f)
        return 1
    print("PREFLIGHT PASSED: all checks green for both predeclared orientations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
