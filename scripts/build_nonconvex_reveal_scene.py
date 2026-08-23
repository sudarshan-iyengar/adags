#!/usr/bin/env python3
"""Build LRV5-NCX: the NON-CONVEX leave-and-return fixture that falsifies
axis-aligned cell-hull completion.

EXPLORATORY development fixture, ``evidence_bearing: false``. NOT a claim-grade
dataset and NOT admitted for event supply.

THE FROZEN CONTRACT
-------------------
This generator implements, and may not deviate from,
``research-wiki/operations/nonconvex-hull-falsifier-spec-2026-08-24.md``.
Every geometric number below is quoted from that document and is independently
validated by the CPU preflight ``scripts/nonconvex_hull_preflight.py``. The
``--self-test`` mode asserts, over a dense grid, that this file's ``is_inside``
predicate and the preflight's ``boxes_contain`` describe the SAME object, so
the two cannot silently drift apart.

WHAT IS DIFFERENT FROM LRV1/LRV2/LRV3/LRV4
------------------------------------------
LRV1-LRV4 are produced by ``scripts/build_synthetic_reveal_scene.py``, whose
event object is a SPHERE. A sphere is cell-convex at every grid resolution and
phase, so hull completion on LRV3 *cannot* gate a cell the object does not
occupy -- which is precisely why LRV3 cannot test the rule that was proposed on
it. LRV5's event object is an **L of two axis-aligned boxes**, whose
axis-aligned cell hull provably contains cells with zero object volume, and the
concavity is occupied by a **static vertical cross of two thin walls that is
present at EVERY frame**. If hull completion gates the concavity, a persistent,
always-visible object is deleted from the render for the whole 27-frame gap.

That is the falsifier's teeth, and it is why this generator needed a new
primitive: an axis-aligned box intersector (``box_hit``, slab method) and a
union primitive with correct nearest-hit semantics and per-object identity ids.

RELATIONSHIP TO ``build_synthetic_reveal_scene.py`` (READ-ONLY, LOAD-BEARING)
-----------------------------------------------------------------------------
That module produced LRV1/LRV2/LRV3/LRV4 and their historical reproducibility
depends on it being untouched, so it is NOT modified and NOT parameterised
here. It is imported, and the following are reused verbatim by import:

  * ``sphere_hit``, ``sphere_albedo``, ``ground_albedo``  -- pure functions;
  * ``pixel_rays``                                        -- reads only
    WIDTH/HEIGHT/FOCAL, all unchanged for LRV5;
  * ``store_ply``, ``sha256_file``, ``frame_times``;
  * ``gated_presence_admissible``, ``episode_2_duration``, ``FLOOR_LEN``;
  * the frame/camera/intrinsic constants that LRV5 keeps bit-identical
    (``N_FRAMES``, ``FPS``, ``TIME_DURATION``, ``N_CAMERAS``, ``TEST_CAMERAS``,
    ``RIG_ELEVATIONS_DEG``, ``WIDTH``, ``HEIGHT``, ``FOCAL``, ``GROUND_Y``,
    ``EPISODE_1_FRAMES``, ``EVENT_OBJECT_ID``, ``GROUND_ID``,
    ``BACKGROUND_ID``, ``LIGHT_DIR``, ``AMBIENT``).

Two functions are COPIED rather than imported, because they read module globals
that LRV5 must change and that module is read-only:

  * ``camera_poses``  -- reads ``RIG_RADIUS``; LRV5 moves it 2.4 -> 3.3
    (spec section 2.1: the enlarged object clips 11 of 20 cameras at 2.4);
  * ``plane_hit``     -- reads ``GROUND_HALF_EXTENT``; LRV5 freezes it at 1.3
    (spec section 1.3), which the LRV3 build passed on the command line.

The copies are otherwise line-for-line identical to the originals and the
``--self-test`` cross-checks the camera ring against the preflight's
independent implementation.

Determinism: no unseeded randomness. The only random draw is the initialization
point cloud, from an explicit ``numpy.random.default_rng(seed)``; the seed is
recorded in ``event_spec.json``.

USAGE
-----
    python scripts/build_nonconvex_reveal_scene.py --self-test
    python scripts/build_nonconvex_reveal_scene.py --out <dir> --dry-run
    python scripts/build_nonconvex_reveal_scene.py --out data/synthetic/lrv5_ncx    --orientation O1
    python scripts/build_nonconvex_reveal_scene.py --out data/synthetic/lrv5_ncx_o2 --orientation O2
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# The two sibling scripts this file is contractually tied to live next to it.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import build_synthetic_reveal_scene as lrv   # noqa: E402  READ-ONLY reuse

SPEC_DOC = "research-wiki/operations/nonconvex-hull-falsifier-spec-2026-08-24.md"
PREFLIGHT_SCRIPT = "scripts/nonconvex_hull_preflight.py"
GENERATOR_SCRIPT = "scripts/build_nonconvex_reveal_scene.py"

# --------------------------------------------------------------------------
# Constants LRV5 keeps BIT-IDENTICAL to LRV3. Imported, not restated, so they
# cannot drift.
# --------------------------------------------------------------------------

N_FRAMES = lrv.N_FRAMES                       # 60
FPS = lrv.FPS                                 # 6.0
TIME_DURATION = lrv.TIME_DURATION             # (0.0, 10.0)
N_CAMERAS = lrv.N_CAMERAS                     # 20
TEST_CAMERAS = lrv.TEST_CAMERAS               # (2, 7, 12, 17)
RIG_ELEVATIONS_DEG = lrv.RIG_ELEVATIONS_DEG   # (12.0, 28.0)
WIDTH, HEIGHT = lrv.WIDTH, lrv.HEIGHT         # 400, 300
FOCAL = lrv.FOCAL                             # 420.0
GROUND_Y = lrv.GROUND_Y                       # -0.62
LIGHT_DIR = lrv.LIGHT_DIR
AMBIENT = lrv.AMBIENT
EPISODE_1_FRAMES = lrv.EPISODE_1_FRAMES       # (0, 29)

GROUND_ID = lrv.GROUND_ID                     # 0
BACKGROUND_ID = lrv.BACKGROUND_ID             # -1
EVENT_OBJECT_ID = lrv.EVENT_OBJECT_ID         # 100

#: Identity id of the NOTCH FILLER. It MUST differ from EVENT_OBJECT_ID: the
#: whole experiment is whether a membership operator wrongly captures the
#: filler, and a shared id would make that indistinguishable by construction.
NOTCH_FILLER_ID = 200

#: Presence schedule, frozen by spec section 6 and bit-identical to LRV3:
#: episode 1 = frames 0-29, gap = 30-56, episode 2 = 57-59. This is NOT a flag.
#: The frozen value is deliberately not reachable from the command line so a
#: run cannot silently move it.
FIRST_RETURN_FRAME = 57
GAP_FRAMES = (EPISODE_1_FRAMES[1] + 1, FIRST_RETURN_FRAME - 1)   # (30, 56)
EPISODE_2_FRAMES = (FIRST_RETURN_FRAME, N_FRAMES - 1)            # (57, 59)

# --------------------------------------------------------------------------
# Constants LRV5 CHANGES, each declared with its spec reference.
# --------------------------------------------------------------------------

#: spec section 2.1. 2.4 -> 3.3. At 2.4 the enlarged object clips 11 of 20
#: cameras (worst normalised frame occupancy 1.428). 3.1 is the smallest 0.1
#: step with zero clipping; 3.3 is taken so the worst-case occupancy (0.723 O1
#: / 0.904 O2) sits below LRV3's own 0.925 rather than merely inside the frame.
RIG_RADIUS = 3.3

#: spec section 1.3. Equal to the initialization half-width, so LRV2's
#: uncovered-surface defect cannot recur.
GROUND_HALF_EXTENT = 1.3
INIT_HALF_WIDTH = 1.3

#: spec section 1.3. Relocated from LRV3 so that NEITHER predeclared
#: orientation intersects them. Colours and texture modes are index-for-index
#: the LRV1-LRV4 ones; centres and radii are the spec's table, which is what
#: the preflight validated.
STATIC_SPHERES = (
    ((-0.95, -0.20, 0.85), 0.28, (0.15, 0.62, 0.60), "stripe_y"),
    ((0.95, 0.15, -0.95), 0.26, (0.72, 0.20, 0.58), "stripe_x"),
    ((-0.30, -0.40, 1.05), 0.22, (0.55, 0.70, 0.18), "checker"),
)

# --------------------------------------------------------------------------
# THE EVENT OBJECT AND THE NOTCH FILLER -- spec sections 1.1, 1.2, 11.
#
# These derivations are written to mirror scripts/nonconvex_hull_preflight.py
# term for term. --self-test asserts float-exact equality against that file's
# EVENT_BOXES_O1/O2 and FILLER_BOXES_O1/O2, so a transcription error here is a
# hard failure, not a silent divergence.
# --------------------------------------------------------------------------

ARM_T = 0.40        # arm thickness
HULL_H = 0.64       # hull half-extent in x and z
OBJ_Y = 0.30        # extrusion half-height in y
NOTCH_LO = -HULL_H + ARM_T      # -0.24
NOTCH_HI = HULL_H               # +0.64

EVENT_BOXES_O1 = (
    # arm A: the z-arm (long in z, thin in x)
    ((-HULL_H, -OBJ_Y, -HULL_H), (-HULL_H + ARM_T, OBJ_Y, HULL_H)),
    # arm B: the x-arm (long in x, thin in z)
    ((-HULL_H, -OBJ_Y, -HULL_H), (HULL_H, OBJ_Y, -HULL_H + ARM_T)),
)
EVENT_BOX_NAMES = ("arm_a_z", "arm_b_x")

# A STATIC vertical cross of two thin walls standing in the concavity, present
# at EVERY frame. 0.12 clearance from both inner faces of the L and 0.12 to the
# hull edge. A cross of thin walls, not a solid block: trained rows sit on
# SURFACES, and a solid block would leave its own interior cells empty of rows,
# so gating them would cost nothing and the falsifier would be toothless.
FILLER_BOXES_O1 = (
    ((-0.12, -OBJ_Y, 0.14), (0.52, OBJ_Y, 0.26)),   # wall spanning x, thin in z
    ((0.14, -OBJ_Y, -0.12), (0.26, OBJ_Y, 0.52)),   # wall spanning z, thin in x
)
FILLER_BOX_NAMES = ("wall_f1_x", "wall_f2_z")

#: spec section 11: half a fresh cell, (2 * 1.3 / 8) / 2 = 0.1625.
CELLS_PER_AXIS = 8              # frozen in scripts/estimate_episodes.py
MIN_GROUP_ROWS = 4              # mirrored from scene.packet_birth
HALF_CELL = (2.0 * INIT_HALF_WIDTH / CELLS_PER_AXIS) / 2.0


def _mirror_shift_box(box):
    """O2's exact map: (x, y, z) -> (-x + 0.1625, y, z + 0.1625).

    A mirror through the plane x = 0 followed by a translation of half a fresh
    cell in x and z. Mirroring swaps the x bounds, so they are re-ordered.
    """
    (lx, ly, lz), (hx, hy, hz) = box
    nlx, nhx = -hx + HALF_CELL, -lx + HALF_CELL
    return ((nlx, ly, lz + HALF_CELL), (nhx, hy, hz + HALF_CELL))


EVENT_BOXES_O2 = tuple(_mirror_shift_box(b) for b in EVENT_BOXES_O1)
FILLER_BOXES_O2 = tuple(_mirror_shift_box(b) for b in FILLER_BOXES_O1)

ORIENTATIONS = {
    "O1": (EVENT_BOXES_O1, FILLER_BOXES_O1),
    "O2": (EVENT_BOXES_O2, FILLER_BOXES_O2),
}

# Appearance. Not frozen by the spec; chosen so a rendered frame is
# human-checkable at a glance. The event object keeps LRV1-LRV4's orange
# ``stripe_band``; the filler is a saturated blue with horizontal ``stripe_y``
# banding, which no other primitive in the scene resembles.
EVENT_COLOUR = lrv.EVENT_SPHERE_COLOUR          # (0.95, 0.45, 0.10)
EVENT_TEXTURE = lrv.EVENT_SPHERE_TEXTURE        # "stripe_band"
FILLER_COLOUR = (0.18, 0.34, 0.92)
FILLER_TEXTURE = "stripe_y"

#: Ray-parameter epsilon. Identical to build_synthetic_reveal_scene.sphere_hit's.
EPS_T = 1e-4


# --------------------------------------------------------------------------
# Axis-aligned box geometry
# --------------------------------------------------------------------------

def is_inside(points, boxes):
    """CLOSED membership predicate for a union of axis-aligned boxes.

    This is spec section 1.1's ``is_event`` (and section 1.2's filler analogue),
    and section 9's frozen metric definition applies it to row positions -- the
    exact analogue of LRV3's ``(xyz - centre).norm <= radius``. Boundaries are
    INCLUSIVE on both sides.

    ``points`` is (..., 3). Returns a boolean array of shape ``points.shape[:-1]``.
    """
    p = np.asarray(points, dtype=np.float64)
    inside = np.zeros(p.shape[:-1], dtype=bool)
    for lo, hi in boxes:
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        inside |= np.all((p >= lo) & (p <= hi), axis=-1)
    return inside


def boxes_aabb(boxes):
    """Axis-aligned bounding box of a union of boxes, as (lo, hi) arrays."""
    lo = np.min([np.asarray(b[0], dtype=np.float64) for b in boxes], axis=0)
    hi = np.max([np.asarray(b[1], dtype=np.float64) for b in boxes], axis=0)
    return lo, hi


def boxes_centre(boxes):
    lo, hi = boxes_aabb(boxes)
    return 0.5 * (lo + hi)


def box_corners(box):
    """The 8 corners of one box, as an (8, 3) array."""
    lo = np.asarray(box[0], dtype=np.float64)
    hi = np.asarray(box[1], dtype=np.float64)
    return np.array([[lo[0] if a else hi[0],
                      lo[1] if b else hi[1],
                      lo[2] if c else hi[2]]
                     for a in (0, 1) for b in (0, 1) for c in (0, 1)])


def _box_intersection(a, b):
    lo = np.maximum(np.asarray(a[0], float), np.asarray(b[0], float))
    hi = np.minimum(np.asarray(a[1], float), np.asarray(b[1], float))
    if np.any(hi <= lo):
        return None
    return (lo, hi)


def union_volume(boxes):
    """Exact volume of a union of axis-aligned boxes (inclusion-exclusion).

    Same construction as nonconvex_hull_preflight.union_volume; used only to
    stamp the analytic volume into event_spec.json.
    """
    total = 0.0
    for r in range(1, len(boxes) + 1):
        for combo in itertools.combinations(range(len(boxes)), r):
            cur = boxes[combo[0]]
            for idx in combo[1:]:
                cur = _box_intersection(cur, boxes[idx])
                if cur is None:
                    break
            if cur is None:
                continue
            lo, hi = np.asarray(cur[0], float), np.asarray(cur[1], float)
            total += ((-1.0) ** (r + 1)) * float(np.prod(np.clip(hi - lo, 0.0, None)))
    return total


def box_hit(origin, dirs, box, eps=EPS_T):
    """Nearest positive ray / axis-aligned-box intersection, by the slab method.

    Returns ``(t, normal)``:
      * ``t``      -- (...,) nearest hit distance strictly greater than ``eps``,
                      ``inf`` where the ray misses;
      * ``normal`` -- (..., 3) OUTWARD unit face normal of the hit face, the
                      zero vector where the ray misses.

    Degenerate cases are handled EXPLICITLY rather than being allowed to fall
    through as NaN:

    * a direction component of exactly zero makes ``1/d`` infinite, and if the
      origin ALSO lies exactly on that pair of slab planes the product is
      ``0 * inf = NaN``. That case means "the ray runs inside the closed slab
      forever", i.e. the axis imposes NO constraint, so its per-axis interval
      is replaced by ``(-inf, +inf)``. Every other axis-parallel case resolves
      through IEEE infinities to the right answer without a NaN: an origin
      strictly outside a slab yields two same-signed infinities whose interval
      is empty, which is a miss.
    * an all-zero direction yields ``t_near = -inf`` and ``t_far = +inf``; the
      finiteness test turns that into a miss instead of a spurious hit.
    * the ray origin INSIDE the box yields ``t_near <= eps < t_far``; the exit
      distance is returned, with the exit face's outward normal.

    ``--self-test`` asserts that no NaN can reach the output and cross-checks
    every returned distance against a bisected brute-force ground truth.
    """
    lo = np.asarray(box[0], dtype=np.float64)
    hi = np.asarray(box[1], dtype=np.float64)
    if np.any(hi < lo):
        raise ValueError("degenerate box, hi < lo: %r" % (box,))
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(dirs, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t_lo = (lo - o) * inv
        t_hi = (hi - o) * inv

    unconstrained = np.isnan(t_lo) | np.isnan(t_hi)
    per_near = np.where(unconstrained, -np.inf, np.minimum(t_lo, t_hi))
    per_far = np.where(unconstrained, np.inf, np.maximum(t_lo, t_hi))

    axis_near = np.argmax(per_near, axis=-1)
    axis_far = np.argmin(per_far, axis=-1)
    t_near = np.take_along_axis(per_near, axis_near[..., None], axis=-1)[..., 0]
    t_far = np.take_along_axis(per_far, axis_far[..., None], axis=-1)[..., 0]

    use_near = t_near > eps
    t = np.where(use_near, t_near, t_far)
    hit = (t_far >= t_near) & (t > eps) & np.isfinite(t)

    axis = np.where(use_near, axis_near, axis_far)
    d_sel = np.take_along_axis(d, axis[..., None], axis=-1)[..., 0]
    # Entry: the outward normal opposes the ray. Exit: it agrees with it.
    # On a real hit the selected axis always has a non-zero direction
    # component (a zero one is unconstrained, i.e. -inf/+inf, and can only be
    # selected when the hit test already failed), so the sign is never 0.
    nsign = np.where(use_near, -np.sign(d_sel), np.sign(d_sel))

    normal = np.zeros(d.shape, dtype=np.float64)
    np.put_along_axis(normal, axis[..., None],
                      np.where(hit, nsign, 0.0)[..., None], axis=-1)
    return np.where(hit, t, np.inf), normal


def box_union_hit(origin, dirs, boxes, eps=EPS_T):
    """Nearest-hit over a UNION of axis-aligned boxes, with the face normal.

    Nearest-hit over the members is the correct union semantics and never
    returns an interior surface: if box P's entry point lay strictly inside
    box Q, the ray would already have entered Q earlier, so Q's smaller entry
    distance wins. The surviving minimum is therefore always on the union's
    exterior (possibly on a shared face, which is the same surface).
    """
    t_best, n_best = None, None
    for b in boxes:
        t, n = box_hit(origin, dirs, b, eps=eps)
        if t_best is None:
            t_best, n_best = t, n
            continue
        take = t < t_best
        t_best = np.where(take, t, t_best)
        n_best = np.where(take[..., None], n, n_best)
    return t_best, n_best


# --------------------------------------------------------------------------
# Camera rig and ground plane.
#
# COPIED from build_synthetic_reveal_scene (camera_poses / plane_hit) because
# those functions read that module's RIG_RADIUS and GROUND_HALF_EXTENT globals,
# which LRV5 changes (spec sections 2.1 and 1.3). That module is read-only and
# load-bearing for LRV1-LRV4 reproducibility, so it is neither edited nor
# mutated at run time. Everything else in these two functions is unchanged.
# --------------------------------------------------------------------------

def camera_poses():
    """(N_CAMERAS, 4, 4) camera-to-world matrices, OpenGL convention."""
    poses = np.zeros((N_CAMERAS, 4, 4), dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0])
    for i in range(N_CAMERAS):
        az = np.deg2rad(360.0 * i / N_CAMERAS)
        el = np.deg2rad(RIG_ELEVATIONS_DEG[i % len(RIG_ELEVATIONS_DEG)])
        eye = np.array([
            RIG_RADIUS * np.cos(el) * np.cos(az),
            RIG_RADIUS * np.sin(el),
            RIG_RADIUS * np.cos(el) * np.sin(az),
        ])
        fwd = -eye / np.linalg.norm(eye)          # look at the origin
        x_axis = np.cross(fwd, up)
        x_axis /= np.linalg.norm(x_axis)
        z_axis = -fwd                              # OpenGL: camera looks down -Z
        y_axis = np.cross(z_axis, x_axis)
        poses[i, :3, 0] = x_axis
        poses[i, :3, 1] = y_axis
        poses[i, :3, 2] = z_axis
        poses[i, :3, 3] = eye
        poses[i, 3, 3] = 1.0
    return poses


def plane_hit(origin, dirs):
    denom = dirs[..., 1]
    t = np.where(np.abs(denom) > 1e-8, (GROUND_Y - origin[1]) / denom, np.inf)
    t = np.where(t > EPS_T, t, np.inf)
    p = origin + t[..., None] * dirs
    inside = ((np.abs(p[..., 0]) < GROUND_HALF_EXTENT)
              & (np.abs(p[..., 2]) < GROUND_HALF_EXTENT))
    return np.where(np.isfinite(t) & inside, t, np.inf)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _const_normal(vec):
    v = np.asarray(vec, dtype=np.float64)
    return lambda pts: np.broadcast_to(v, pts.shape)


def _sphere_normal(centre, radius):
    c = np.asarray(centre, dtype=np.float64)
    return lambda pts: (pts - c) / radius


def _array_normal(arr):
    return lambda pts: arr


def scene_primitives(event_boxes, filler_boxes, origin, dirs, event_present):
    """Ordered primitive list for one ray bundle.

    Each entry is ``(name, t, identity_id, normal_fn, albedo_fn)``. The notch
    filler and the event object are UNIONS with DISTINCT identity ids; the two
    arms of the L share EVENT_OBJECT_ID and the two walls of the cross share
    NOTCH_FILLER_ID, because each is one object.
    """
    prims = [("ground", plane_hit(origin, dirs), GROUND_ID,
              _const_normal((0.0, 1.0, 0.0)),
              lambda pts: lrv.ground_albedo(pts))]

    for i, (centre, radius, base, mode) in enumerate(STATIC_SPHERES):
        prims.append((
            "static_sphere_%d" % i,
            lrv.sphere_hit(origin, dirs, centre, radius),
            i + 1,
            _sphere_normal(centre, radius),
            (lambda pts, c=centre, b=base, m=mode: lrv.sphere_albedo(pts, c, b, m)),
        ))

    t_fill, n_fill = box_union_hit(origin, dirs, filler_boxes)
    fill_org = boxes_centre(filler_boxes)
    prims.append((
        "notch_filler", t_fill, NOTCH_FILLER_ID, _array_normal(n_fill),
        (lambda pts, c=fill_org: lrv.sphere_albedo(pts, c, FILLER_COLOUR,
                                                   FILLER_TEXTURE)),
    ))

    if event_present:
        t_ev, n_ev = box_union_hit(origin, dirs, event_boxes)
        ev_org = boxes_centre(event_boxes)
        prims.append((
            "event_object", t_ev, EVENT_OBJECT_ID, _array_normal(n_ev),
            (lambda pts, c=ev_org: lrv.sphere_albedo(pts, c, EVENT_COLOUR,
                                                     EVENT_TEXTURE)),
        ))
    return prims


def render(c2w, event_boxes, filler_boxes, event_present):
    """Return (rgb float32 HxWx3 in [0,1], identity int32 HxW, depth float32 HxW)."""
    origin, dirs = lrv.pixel_rays(c2w)
    prims = scene_primitives(event_boxes, filler_boxes, origin, dirs, event_present)

    stack = np.stack([p[1] for p in prims], axis=0)
    best = np.argmin(stack, axis=0)
    depth = np.take_along_axis(stack, best[None], axis=0)[0]
    hit = np.isfinite(depth)
    ids = np.asarray([p[2] for p in prims], dtype=np.int32)
    identity = np.where(hit, ids[best], BACKGROUND_ID).astype(np.int32)

    points = origin + np.where(hit, depth, 0.0)[..., None] * dirs
    albedo = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    normal = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    for pi, (_name, _t, _pid, normal_fn, albedo_fn) in enumerate(prims):
        m = hit & (best == pi)
        if m.any():
            albedo[m] = albedo_fn(points)[m]
            normal[m] = normal_fn(points)[m]

    light = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
    lambert = np.clip(normal @ light, 0.0, 1.0)
    shade = AMBIENT + (1.0 - AMBIENT) * lambert
    rgb = np.clip(albedo * shade[..., None], 0.0, 1.0)
    rgb[~hit] = 0.0
    return rgb.astype(np.float32), identity, np.where(hit, depth, 0.0).astype(np.float32)


def event_present_at(frame):
    """The presence schedule applies to the L ONLY. The filler is always present."""
    return (EPISODE_1_FRAMES[0] <= frame <= EPISODE_1_FRAMES[1]
            or EPISODE_2_FRAMES[0] <= frame <= EPISODE_2_FRAMES[1])


def phase_of(frame):
    if EPISODE_1_FRAMES[0] <= frame <= EPISODE_1_FRAMES[1]:
        return "episode_1"
    if GAP_FRAMES[0] <= frame <= GAP_FRAMES[1]:
        return "gap"
    return "episode_2"


# --------------------------------------------------------------------------
# event_spec.json
# --------------------------------------------------------------------------

def _boxes_json(boxes, names):
    out = []
    for name, (lo, hi) in zip(names, boxes):
        out.append({"name": name,
                    "lo": [float(v) for v in lo],
                    "hi": [float(v) for v in hi]})
    return out


def build_spec(orientation, scene_id, seed, num_init_pts, event_boxes,
               filler_boxes, visible_px, filler_px, init_rows):
    """Every analytic quantity a downstream membership scorer needs.

    Membership is recorded as the EXACT CLOSED BOX UNION, so a scorer reads
    geometry from the artifact and never from a hard-coded constant.
    """
    test_set = set(TEST_CAMERAS)
    ev_lo, ev_hi = boxes_aabb(event_boxes)
    fl_lo, fl_hi = boxes_aabb(filler_boxes)
    notch_lo = (NOTCH_LO, -OBJ_Y, NOTCH_LO)
    notch_hi = (NOTCH_HI, OBJ_Y, NOTCH_HI)
    if orientation == "O2":
        notch_lo, notch_hi = _mirror_shift_box((notch_lo, notch_hi))

    spec = {
        "scene_id": scene_id,
        "kind": "synthetic_nonconvex_leave_and_return",
        "orientation": orientation,
        "spec_document": SPEC_DOC,
        "preflight_script": PREFLIGHT_SCRIPT,
        "generator_script": GENERATOR_SCRIPT,
        "evidence_bearing": False,
        "ground_half_extent": GROUND_HALF_EXTENT,
        "n_frames": N_FRAMES, "fps": FPS, "time_duration": list(TIME_DURATION),
        "width": WIDTH, "height": HEIGHT, "focal_px": FOCAL,
        "n_cameras": N_CAMERAS, "test_cameras": list(TEST_CAMERAS),
        "train_cameras": [c for c in range(N_CAMERAS) if c not in test_set],
        "rig": {
            "radius": RIG_RADIUS,
            "elevations_deg": list(RIG_ELEVATIONS_DEG),
            "lrv3_radius": 2.4,
            "changed_because": (
                "spec 2.1: the enlarged non-convex object clips 11 of 20 "
                "cameras at 2.4; at 3.3 the worst normalised frame occupancy "
                "is 0.723 (O1) / 0.904 (O2) against LRV3's own 0.925. Every "
                "RENDERED quantity is therefore not comparable to a recorded "
                "LRV3 number; only membership precision/recall transfer."),
        },
        "initialization": {
            "num_points": int(num_init_pts),
            "seed": int(seed),
            "half_width": INIT_HALF_WIDTH,
            "rng": "numpy.random.default_rng",
            "rows_inside_event_object": int(init_rows["event"]),
            "rows_inside_notch_filler": int(init_rows["filler"]),
        },
        "event_object": {
            "id": EVENT_OBJECT_ID,
            "kind": "axis_aligned_box_union",
            "membership_predicate": "closed_box_union",
            "membership_note": (
                "is_event(p) = OR over boxes of (lo <= p <= hi), componentwise "
                "and INCLUSIVE on both sides. This is spec 1.1 and spec 9's "
                "frozen metric definition."),
            "boxes": _boxes_json(event_boxes, EVENT_BOX_NAMES),
            "aabb": {"lo": [float(v) for v in ev_lo],
                     "hi": [float(v) for v in ev_hi]},
            "volume": union_volume(list(event_boxes)),
            "arm_thickness": ARM_T,
            "hull_half_extent": HULL_H,
            "extrusion_half_height_y": OBJ_Y,
            "notch": {"lo": [float(v) for v in notch_lo],
                      "hi": [float(v) for v in notch_hi],
                      "size": [0.88, 0.60, 0.88]},
            "sphere_centre_radius_omitted": (
                "DELIBERATE. LRV1-LRV4 recorded 'centre' and 'radius' and "
                "consumers such as scripts/estimate_episodes.py::score_program "
                "build a SPHERE membership test from them. LRV5's event object "
                "is not a sphere, so emitting those keys would let a "
                "sphere-shaped predicate silently mis-score this fixture. They "
                "are omitted so such a consumer fails loudly and the box "
                "predicate above must be adopted explicitly."),
        },
        "notch_filler": {
            "id": NOTCH_FILLER_ID,
            "kind": "axis_aligned_box_union",
            "membership_predicate": "closed_box_union",
            "static": True,
            "present_frames": [0, N_FRAMES - 1],
            "boxes": _boxes_json(filler_boxes, FILLER_BOX_NAMES),
            "aabb": {"lo": [float(v) for v in fl_lo],
                     "hi": [float(v) for v in fl_hi]},
            "volume": union_volume(list(filler_boxes)),
            "clearance_to_inner_faces": 0.12,
            "role": (
                "spec 1.2: the falsifier's teeth. Present at EVERY frame, "
                "never part of the event object. If hull completion gates the "
                "concavity, this persistent always-visible object is driven to "
                "presence ~0 over the whole 27-frame gap."),
        },
        "static_spheres": [
            {"id": i + 1, "centre": list(c), "radius": r,
             "colour": list(col), "texture": mode}
            for i, (c, r, col, mode) in enumerate(STATIC_SPHERES)
        ],
        "ground": {"id": GROUND_ID, "y": GROUND_Y,
                   "half_extent": GROUND_HALF_EXTENT},
        "background_id": BACKGROUND_ID,
        "presence_frames": {
            "episode_1": list(EPISODE_1_FRAMES),
            "gap": list(GAP_FRAMES),
            "episode_2": list(EPISODE_2_FRAMES),
        },
        "first_return_frame": FIRST_RETURN_FRAME,
        "gated_presence_admissible": bool(
            lrv.gated_presence_admissible(FIRST_RETURN_FRAME)),
        "episode_2_duration_s": lrv.episode_2_duration(FIRST_RETURN_FRAME),
        "floor_len_s": lrv.FLOOR_LEN,
        # Bit-identical to LRV3's schedule, so these are the same oracle
        # programs LRV3's event_spec.json points at.
        "oracle_episodes": "configs/lrv1/oracle_correct.json",
        "wrong_time_episodes": "configs/lrv1/oracle_wrong.json",
        "return_frames": list(range(EPISODE_2_FRAMES[0], EPISODE_2_FRAMES[1] + 1)),
        "estimator_reference": {
            "cells_per_axis": CELLS_PER_AXIS,
            "min_group_rows": MIN_GROUP_ROWS,
            "note": ("spec 3: frozen constants of scripts/estimate_episodes.py, "
                     "recorded here for the reader. This generator does not "
                     "build the voxel grid; the estimator builds it from the "
                     "TRAINED cloud's own bounding box."),
        },
        "event_object_pixels_per_test_view_frame": {
            "cam{:02d}_f{:03d}".format(c, f): int(visible_px[(c, f)])
            for c in TEST_CAMERAS for f in range(N_FRAMES)
        },
        "notch_filler_pixels_per_test_view_frame": {
            "cam{:02d}_f{:03d}".format(c, f): int(filler_px[(c, f)])
            for c in TEST_CAMERAS for f in range(N_FRAMES)
        },
    }
    return spec


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def init_cloud(seed, num_init_pts):
    """Byte-identical construction to LRV1/LRV2/LRV3 (spec section 5)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-INIT_HALF_WIDTH, INIT_HALF_WIDTH, size=(num_init_pts, 3))


def generate(args, event_boxes, filler_boxes):
    out = Path(args.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    (out / "gt_identity").mkdir(parents=True, exist_ok=True)

    poses = camera_poses()
    times = lrv.frame_times()
    test_set = set(TEST_CAMERAS)

    frames_train, frames_test = [], []
    visible_px, filler_px = {}, {}

    t_start = time.time()
    for cam in range(N_CAMERAS):
        split = "test" if cam in test_set else "train"
        for f in range(N_FRAMES):
            rgb, identity, _ = render(poses[cam], event_boxes, filler_boxes,
                                      event_present_at(f))
            stem = "cam{:02d}_f{:03d}".format(cam, f)
            Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(
                out / split / (stem + ".png"))
            visible_px[(cam, f)] = int((identity == EVENT_OBJECT_ID).sum())
            filler_px[(cam, f)] = int((identity == NOTCH_FILLER_ID).sum())
            if cam in test_set:
                # front-most identity buffer, held-out views only
                np.save(out / "gt_identity" / (stem + ".npy"),
                        identity.astype(np.int16))
            (frames_test if cam in test_set else frames_train).append({
                "file_path": "./{}/{}".format(split, stem),
                "time": float(times[f]),
                "transform_matrix": poses[cam].tolist(),
                "camera_index": cam,
                "frame_index": f,
            })
        print("  camera %2d/%d done (%.1fs elapsed)"
              % (cam + 1, N_CAMERAS, time.time() - t_start))

    intrinsics = {
        "fl_x": FOCAL, "fl_y": FOCAL,
        "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0,
        "w": WIDTH, "h": HEIGHT,
        "camera_angle_x": float(2.0 * np.arctan(WIDTH / (2.0 * FOCAL))),
    }
    for name, frames in (("transforms_train.json", frames_train),
                         ("transforms_test.json", frames_test)):
        (out / name).write_text(json.dumps({**intrinsics, "frames": frames}, indent=1))

    xyz = init_cloud(args.seed, args.num_init_pts)
    rgb = np.full((args.num_init_pts, 3), 0.5)
    lrv.store_ply(out / "points3d.ply", xyz, rgb)

    init_rows = {"event": int(is_inside(xyz, event_boxes).sum()),
                 "filler": int(is_inside(xyz, filler_boxes).sum())}
    spec = build_spec(args.orientation, args.scene_id, args.seed,
                      args.num_init_pts, event_boxes, filler_boxes,
                      visible_px, filler_px, init_rows)
    (out / "event_spec.json").write_text(json.dumps(spec, indent=1))

    ret = spec["return_frames"]
    tot = sum(visible_px[(c, f)] for c in TEST_CAMERAS for f in ret)
    ep1 = sum(visible_px[(c, f)] for c in TEST_CAMERAS
              for f in range(EPISODE_1_FRAMES[0], EPISODE_1_FRAMES[1] + 1))
    gap = sum(visible_px[(c, f)] for c in TEST_CAMERAS
              for f in range(GAP_FRAMES[0], GAP_FRAMES[1] + 1))
    fgap = sum(filler_px[(c, f)] for c in TEST_CAMERAS
               for f in range(GAP_FRAMES[0], GAP_FRAMES[1] + 1))
    print("held-out EVENT pixel-times:  return=%d episode1=%d gap=%d" % (tot, ep1, gap))
    print("held-out FILLER pixel-times during the gap: %d "
          "(must be > 0: it is the deletion the falsifier detects)" % fgap)
    print("initialization rows inside the event object: %d | inside the filler: %d"
          % (init_rows["event"], init_rows["filler"]))
    print("points3d.ply sha256: %s" % lrv.sha256_file(out / "points3d.ply"))
    print("wrote %s" % out)
    return 0


# --------------------------------------------------------------------------
# --dry-run
# --------------------------------------------------------------------------

def dry_run(args, event_boxes, filler_boxes):
    poses = camera_poses()
    n_test = len(TEST_CAMERAS)
    n_train = N_CAMERAS - n_test

    print("=" * 78)
    print("DRY RUN -- nothing is written")
    print("=" * 78)
    print("orientation     : %s" % args.orientation)
    print("scene id        : %s" % args.scene_id)
    print("output directory: %s" % (args.out or "(none given)"))
    print("seed            : %d   init points: %d" % (args.seed, args.num_init_pts))
    print("resolution      : %dx%d, focal %.1f px" % (WIDTH, HEIGHT, FOCAL))
    print("rig radius      : %.2f (LRV3: 2.40)   elevations %s"
          % (RIG_RADIUS, list(RIG_ELEVATIONS_DEG)))
    print("frames          : %d @ %.1f fps   presence ep1 %s gap %s ep2 %s"
          % (N_FRAMES, FPS, list(EPISODE_1_FRAMES), list(GAP_FRAMES),
             list(EPISODE_2_FRAMES)))
    print("cameras         : %d (%d train, %d held out %s)"
          % (N_CAMERAS, n_train, n_test, list(TEST_CAMERAS)))
    print("identity ids    : ground %d, static spheres 1..%d, notch filler %d, "
          "event object %d, background %d"
          % (GROUND_ID, len(STATIC_SPHERES), NOTCH_FILLER_ID, EVENT_OBJECT_ID,
             BACKGROUND_ID))

    print("\ngeometry (%s):" % args.orientation)
    for name, (lo, hi) in zip(EVENT_BOX_NAMES, event_boxes):
        print("  event  %-9s x[%+.4f,%+.4f] y[%+.4f,%+.4f] z[%+.4f,%+.4f]"
              % (name, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    for name, (lo, hi) in zip(FILLER_BOX_NAMES, filler_boxes):
        print("  filler %-9s x[%+.4f,%+.4f] y[%+.4f,%+.4f] z[%+.4f,%+.4f]"
              % (name, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    print("  event volume %.6f | filler volume %.6f"
          % (union_volume(list(event_boxes)), union_volume(list(filler_boxes))))

    # Real measurements, not guesses: encode two representative frames.
    t0 = time.time()
    rgb_on, id_on, _ = render(poses[0], event_boxes, filler_boxes, True)
    rgb_off, id_off, _ = render(poses[0], event_boxes, filler_boxes, False)
    render_s = (time.time() - t0) / 2.0

    def _png_bytes(rgb):
        buf = io.BytesIO()
        Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(buf, format="PNG")
        return buf.tell()

    png_on, png_off = _png_bytes(rgb_on), _png_bytes(rgb_off)
    n_present = ((EPISODE_1_FRAMES[1] - EPISODE_1_FRAMES[0] + 1)
                 + (EPISODE_2_FRAMES[1] - EPISODE_2_FRAMES[0] + 1))
    n_absent = N_FRAMES - n_present
    png_per_cam = n_present * png_on + n_absent * png_off

    nbuf = io.BytesIO()
    np.save(nbuf, id_on.astype(np.int16))
    npy_bytes = nbuf.tell()

    sample = init_cloud(args.seed, 1000)
    ply_line = sum(len("{:.6f} {:.6f} {:.6f} 0.000000 0.000000 0.000000 "
                       "127 127 127\n".format(*sample[i]))
                   for i in range(1000)) / 1000.0
    ply_bytes = int(ply_line * args.num_init_pts) + 220

    # transforms: one frame entry is ~ 700 bytes of pretty-printed JSON.
    tr_train = 700 * n_train * N_FRAMES
    tr_test = 700 * n_test * N_FRAMES

    # event_spec.json: ~6 kB of analytic geometry plus two per-test-view-frame
    # pixel dictionaries of n_test * N_FRAMES entries at ~22 bytes each.
    spec_bytes = 6_000 + 2 * 22 * n_test * N_FRAMES

    total_png = png_per_cam * N_CAMERAS
    total_npy = npy_bytes * n_test * N_FRAMES
    total = total_png + total_npy + ply_bytes + tr_train + tr_test + spec_bytes

    print("\nWOULD WRITE:")
    print("  train/*.png        %5d files   %8.1f MB"
          % (n_train * N_FRAMES, png_per_cam * n_train / 1e6))
    print("  test/*.png         %5d files   %8.1f MB"
          % (n_test * N_FRAMES, png_per_cam * n_test / 1e6))
    print("  gt_identity/*.npy  %5d files   %8.1f MB  (%dx%d int16, held-out "
          "cameras only)" % (n_test * N_FRAMES, total_npy / 1e6, HEIGHT, WIDTH))
    print("  points3d.ply       %5d files   %8.1f MB  (%d ascii vertices, "
          "estimated)" % (1, ply_bytes / 1e6, args.num_init_pts))
    print("  transforms_train.json / transforms_test.json  %8.1f MB (estimated)"
          % ((tr_train + tr_test) / 1e6))
    print("  event_spec.json        1 file    %8.3f MB (estimated)"
          % (spec_bytes / 1e6))
    print("  ---------------------------------------------")
    print("  TOTAL              %5d files   %8.1f MB (estimated)"
          % (N_CAMERAS * N_FRAMES + n_test * N_FRAMES + 4, total / 1e6))
    print("\n  measured PNG size: %d bytes with the event object, %d without"
          % (png_on, png_off))
    print("  measured render:   %.3f s/frame  ->  %d frames = %.1f min "
          "(single CPU process)"
          % (render_s, N_CAMERAS * N_FRAMES,
             render_s * N_CAMERAS * N_FRAMES / 60.0))
    ids_on = sorted(int(v) for v in np.unique(id_on))
    ids_off = sorted(int(v) for v in np.unique(id_off))
    print("  identity ids present, cam00: event present %s | event absent %s"
          % (ids_on, ids_off))
    return 0


# --------------------------------------------------------------------------
# --self-test
# --------------------------------------------------------------------------

_FAILURES = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not ok:
        _FAILURES.append(name)
    return bool(ok)


def _import_preflight():
    import nonconvex_hull_preflight as pre
    return pre


def _brute_force_box_t(origin, direction, box, t_max=12.0, step=1.0e-3,
                       eps=EPS_T, t_min=None):
    """Ground-truth first surface crossing, by dense marching then bisection.

    Independent of box_hit: it only evaluates the CLOSED inside predicate. The
    answer is the first t > eps at which the inside state differs from the
    state at t = eps -- which is the entry point for an origin outside the box
    and the exit point for an origin inside it, exactly matching box_hit's
    contract.

    A uniform march CANNOT see a chord shorter than ``step``: a ray that
    grazes a corner enters and leaves between two consecutive samples. That is
    a limitation of this ground truth, not of ``box_hit``, and the caller
    resolves it by re-running over a narrow ``t_min``/``t_max`` window at a
    much finer step. The window only localises where to look; the verdict is
    still the independent inside-predicate test.
    """
    if t_min is None:
        t_min = eps
    ts = np.arange(t_min, t_max, step)
    if ts.size < 2:
        return np.inf
    pts = origin[None, :] + ts[:, None] * direction[None, :]
    ins = is_inside(pts, [box])
    change = np.nonzero(ins != ins[0])[0]
    if change.size == 0:
        return np.inf
    k = int(change[0])
    a, b = ts[k - 1], ts[k]
    sa = ins[k - 1]
    for _ in range(80):
        m = 0.5 * (a + b)
        if bool(is_inside(origin + m * direction, [box])) == bool(sa):
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def _test_box_intersector():
    print("\n[T1] BOX INTERSECTOR vs BRUTE-FORCE GROUND TRUTH")
    rng = np.random.default_rng(20260824)
    boxes = list(EVENT_BOXES_O1) + list(FILLER_BOXES_O1) + [
        ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
        ((0.1, -0.9, 0.2), (0.13, 0.9, 0.25)),      # very thin slab
    ]
    n_free, n_aimed, n_in = 300, 700, 200
    n_rays = n_free + n_aimed + n_in
    max_dt = 0.0
    n_hit = n_miss = n_disagree = n_inside = n_grazing = 0
    max_face_err = 0.0
    max_surface_err = 0.0
    for box in boxes:
        lo_b = np.asarray(box[0], float)
        hi_b = np.asarray(box[1], float)
        centre = 0.5 * (lo_b + hi_b)
        half = 0.5 * (hi_b - lo_b)
        # (a) free rays -- mostly misses, exercises the rejection paths;
        # (b) rays aimed at a random point in a 1.15x-expanded AABB, so a large
        #     fraction hit and many pass close to a face or an edge;
        # (c) rays whose origin is strictly INSIDE the box, exercising the
        #     exit-distance branch.
        origins = np.concatenate([
            centre + rng.uniform(-2.0, 2.0, size=(n_free + n_aimed, 3)),
            centre + rng.uniform(-0.98, 0.98, size=(n_in, 3)) * half,
        ])
        dirs = rng.normal(size=(n_rays, 3))
        target = centre + rng.uniform(-1.15, 1.15, size=(n_aimed, 3)) * half
        dirs[n_free:n_free + n_aimed] = target - origins[n_free:n_free + n_aimed]
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        t_an, n_an = box_hit(origins, dirs, box)
        if np.isnan(t_an).any() or np.isnan(n_an).any():
            check("no NaN in box_hit output", False, "box %r" % (box,))
            return
        for i in range(n_rays):
            t_bf = _brute_force_box_t(origins[i], dirs[i], box)
            an_hit = np.isfinite(t_an[i]) and t_an[i] < 12.0
            bf_hit = np.isfinite(t_bf)
            if an_hit and not bf_hit:
                # Candidate grazing ray: re-march a narrow window at 1e-4 the
                # step. If the box really is entered there, the coarse march
                # simply stepped over a chord shorter than 1e-3.
                t_bf = _brute_force_box_t(
                    origins[i], dirs[i], box, step=1.0e-7,
                    t_min=max(EPS_T, float(t_an[i]) - 2.0e-3),
                    t_max=float(t_an[i]) + 2.0e-3)
                bf_hit = np.isfinite(t_bf)
                if bf_hit:
                    n_grazing += 1
                else:
                    n_disagree += 1
                    continue
            elif an_hit != bf_hit:
                # Brute force found a crossing the analytic test missed. This
                # is never excusable.
                n_disagree += 1
                continue
            if not an_hit:
                n_miss += 1
                continue
            n_hit += 1
            if is_inside(origins[i], [box]):
                n_inside += 1
            max_dt = max(max_dt, abs(float(t_an[i]) - float(t_bf)))
            p = origins[i] + t_an[i] * dirs[i]
            lo = np.asarray(box[0], float)
            hi = np.asarray(box[1], float)
            # the hit point lies on the closed box
            max_surface_err = max(max_surface_err,
                                  float(np.max(np.maximum(lo - p, p - hi))))
            # the returned normal names the face the point actually touches
            ax = int(np.argmax(np.abs(n_an[i])))
            face = lo[ax] if n_an[i][ax] < 0 else hi[ax]
            max_face_err = max(max_face_err, abs(float(p[ax]) - float(face)))
    check("randomized rays: analytic and brute-force agree on hit/miss",
          n_disagree == 0,
          "%d hits, %d misses, %d unresolved disagreements over %d rays x "
          "%d boxes (%d grazing rays needed the refined ground truth: their "
          "chord is shorter than the 1e-3 march step)"
          % (n_hit, n_miss, n_disagree, n_rays, len(boxes), n_grazing))
    check("randomized rays: max |t_analytic - t_bruteforce| below 1e-9",
          max_dt < 1e-9, "max discrepancy %.3e (%d origin-inside rays covered)"
          % (max_dt, n_inside))
    check("hit points lie on the closed box (max outward excursion < 1e-9)",
          max_surface_err < 1e-9, "%.3e" % max_surface_err)
    check("returned normal names the face the hit point touches (< 1e-9)",
          max_face_err < 1e-9, "%.3e" % max_face_err)

    # ---- explicit degenerate battery -----------------------------------
    box = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    cases = [
        ("axis-parallel, through the box",
         np.array([-3.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 2.0),
        ("axis-parallel, misses the slab",
         np.array([-3.0, 5.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.inf),
        ("axis-parallel, origin exactly ON the face plane (0/0 case)",
         np.array([-3.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), 2.0),
        ("origin exactly at a corner, pointing inward",
         np.array([-1.0, -1.0, -1.0]), np.array([1.0, 0.0, 0.0]), 2.0),
        ("origin strictly inside -> exit distance",
         np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 1.0),
        ("ray points away from the box",
         np.array([-3.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.inf),
        ("zero-length direction",
         np.array([-3.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.inf),
        ("two zero components",
         np.array([-3.0, 1.0, 1.0]), np.array([1.0, 0.0, 0.0]), 2.0),
    ]
    ok_all, nan_free = True, True
    for name, o, d, expect in cases:
        t, n = box_hit(o, d[None, :], box)
        if np.isnan(t).any() or np.isnan(n).any():
            nan_free = False
        got = float(t[0])
        good = (np.isinf(got) and np.isinf(expect)) or abs(got - expect) < 1e-12
        if not good:
            ok_all = False
            print("      degenerate case FAILED: %s -> t=%r expected %r"
                  % (name, got, expect))
    check("degenerate rays (axis-parallel, on-face 0/0, corner, inside, "
          "zero-direction) give the hand-computed answer", ok_all,
          "%d cases" % len(cases))
    check("no NaN reaches box_hit's output on any degenerate case", nan_free)

    # ---- union semantics -------------------------------------------------
    # A ray entering arm B along -z at an x inside arm A must land on arm A's
    # z = +0.64 exterior face, NOT on arm B's z = -0.24 face, which is interior
    # to arm A. This is the case that a naive per-box minimum would get wrong
    # if nearest-hit semantics were mis-implemented.
    o = np.array([-0.40, 0.0, 3.0])
    d = np.array([[0.0, 0.0, -1.0]])
    t_u, n_u = box_union_hit(o, d, EVENT_BOXES_O1)
    p = o + t_u[0] * d[0]
    check("union nearest-hit returns the EXTERIOR surface, not an interior face",
          abs(float(p[2]) - HULL_H) < 1e-9 and abs(float(n_u[0][2]) - 1.0) < 1e-12,
          "hit z=%.6f (expected %+.4f), normal %s" % (p[2], HULL_H,
                                                      n_u[0].tolist()))


def _test_predicate_against_preflight():
    print("\n[T2] is_inside AGREES EXACTLY WITH scripts/nonconvex_hull_preflight.py")
    try:
        pre = _import_preflight()
    except Exception as exc:                                   # pragma: no cover
        check("import nonconvex_hull_preflight", False, repr(exc))
        return

    # (a) the box tuples themselves are float-exact
    pairs = [("EVENT_BOXES_O1", EVENT_BOXES_O1, pre.EVENT_BOXES_O1),
             ("EVENT_BOXES_O2", EVENT_BOXES_O2, pre.EVENT_BOXES_O2),
             ("FILLER_BOXES_O1", FILLER_BOXES_O1, pre.FILLER_BOXES_O1),
             ("FILLER_BOXES_O2", FILLER_BOXES_O2, pre.FILLER_BOXES_O2)]
    for name, mine, theirs in pairs:
        same = (len(mine) == len(theirs)
                and all(np.array_equal(np.asarray(a, float), np.asarray(b, float))
                        for ma, tb in zip(mine, theirs)
                        for a, b in zip(ma, tb)))
        check("%s is float-exactly the preflight's" % name, same)

    # (b) the shared rig / scene constants
    const_pairs = [
        ("RIG_RADIUS", RIG_RADIUS, pre.RIG_RADIUS),
        ("N_CAMERAS", N_CAMERAS, pre.N_CAMERAS),
        ("TEST_CAMERAS", tuple(TEST_CAMERAS), tuple(pre.TEST_CAMERAS)),
        ("RIG_ELEVATIONS_DEG", tuple(RIG_ELEVATIONS_DEG),
         tuple(pre.RIG_ELEVATIONS_DEG)),
        ("WIDTH/HEIGHT/FOCAL", (WIDTH, HEIGHT, FOCAL),
         (pre.WIDTH, pre.HEIGHT, pre.FOCAL)),
        ("GROUND_Y", GROUND_Y, pre.GROUND_Y),
        ("GROUND_HALF_EXTENT", GROUND_HALF_EXTENT, pre.GROUND_HALF_EXTENT),
        ("INIT half-width / points / seed",
         (INIT_HALF_WIDTH, 50_000, 0),
         (pre.INIT_HALF_WIDTH, pre.N_INIT_PTS, pre.INIT_SEED)),
        ("CELLS_PER_AXIS / MIN_GROUP_ROWS", (CELLS_PER_AXIS, MIN_GROUP_ROWS),
         (pre.CELLS_PER_AXIS, pre.MIN_GROUP_ROWS)),
        ("HALF_CELL", HALF_CELL, pre.HALF_CELL),
    ]
    for name, mine, theirs in const_pairs:
        check("%s matches the preflight" % name, mine == theirs,
              "%r vs %r" % (mine, theirs))
    sph_same = (len(STATIC_SPHERES) == len(pre.STATIC_SPHERES)
                and all(tuple(a[0]) == tuple(b[0]) and a[1] == b[1]
                        for a, b in zip(STATIC_SPHERES, pre.STATIC_SPHERES)))
    check("STATIC_SPHERES centres and radii match the preflight", sph_same)

    # (c) the PREDICATE, over a dense grid plus every exact face/edge/corner
    grid_n = 161
    axis = np.linspace(-1.35, 1.35, grid_n)
    total_disagree = 0
    total_samples = 0
    for oname in ("O1", "O2"):
        ev, fl = ORIENTATIONS[oname]
        for label, boxes in (("event", ev), ("filler", fl)):
            dis = 0
            for xi in range(grid_n):
                yy, zz = np.meshgrid(axis, axis, indexing="ij")
                pts = np.stack([np.full(yy.shape, axis[xi]), yy, zz], axis=-1)
                flat = pts.reshape(-1, 3)
                mine = is_inside(flat, boxes)
                theirs = pre.boxes_contain(flat, boxes)
                dis += int(np.count_nonzero(mine != theirs))
                total_samples += flat.shape[0]
            # exact boundary coordinates: faces, edges and corners
            coords = sorted({float(v) for b in boxes for e in b for v in e})
            extra = sorted(set(coords) | {c + 1e-12 for c in coords}
                           | {c - 1e-12 for c in coords} | {0.0})
            grid = np.array(list(itertools.product(extra, repeat=3)))
            mine = is_inside(grid, boxes)
            theirs = pre.boxes_contain(grid, boxes)
            dis += int(np.count_nonzero(mine != theirs))
            total_samples += grid.shape[0]
            total_disagree += dis
            check("%s %s: is_inside == preflight boxes_contain" % (oname, label),
                  dis == 0, "%d disagreeing samples" % dis)
    check("TOTAL predicate disagreements over %d samples is ZERO" % total_samples,
          total_disagree == 0, "%d" % total_disagree)


def _test_schedule_and_split():
    print("\n[T3] SCHEDULE, CAMERAS AND SPLIT ARE BIT-IDENTICAL TO LRV3")
    check("episode 1 = frames [0, 29]", tuple(EPISODE_1_FRAMES) == (0, 29),
          str(tuple(EPISODE_1_FRAMES)))
    check("gap = frames [30, 56]", tuple(GAP_FRAMES) == (30, 56),
          str(tuple(GAP_FRAMES)))
    check("episode 2 = frames [57, 59]", tuple(EPISODE_2_FRAMES) == (57, 59),
          str(tuple(EPISODE_2_FRAMES)))
    check("first return frame 57 clears floor_len (gated-presence admissible)",
          lrv.gated_presence_admissible(FIRST_RETURN_FRAME),
          "episode 2 lasts %.4f s against floor %.4f s"
          % (lrv.episode_2_duration(FIRST_RETURN_FRAME), lrv.FLOOR_LEN))
    check("57 is the admissible maximum",
          FIRST_RETURN_FRAME == lrv.ADMISSIBLE_MAX_FIRST_RETURN_FRAME)
    check("20 cameras, held out (2, 7, 12, 17)",
          N_CAMERAS == 20 and tuple(TEST_CAMERAS) == (2, 7, 12, 17))
    check("event object id %d and notch filler id %d are DISTINCT"
          % (EVENT_OBJECT_ID, NOTCH_FILLER_ID),
          EVENT_OBJECT_ID != NOTCH_FILLER_ID)

    lrv3 = Path(__file__).resolve().parents[1] / "data/synthetic/lrv3/event_spec.json"
    if not lrv3.is_file():
        print("  [SKIP] data/synthetic/lrv3/event_spec.json not present; the "
              "on-disk LRV3 comparison could not be run")
        return
    s3 = json.loads(lrv3.read_text())
    same = (s3["presence_frames"]["episode_1"] == list(EPISODE_1_FRAMES)
            and s3["presence_frames"]["gap"] == list(GAP_FRAMES)
            and s3["presence_frames"]["episode_2"] == list(EPISODE_2_FRAMES)
            and s3["test_cameras"] == list(TEST_CAMERAS)
            and s3["n_cameras"] == N_CAMERAS and s3["n_frames"] == N_FRAMES
            and s3["fps"] == FPS and s3["width"] == WIDTH
            and s3["height"] == HEIGHT and s3["focal_px"] == FOCAL
            and s3["ground_half_extent"] == GROUND_HALF_EXTENT)
    check("matches the ON-DISK LRV3 event_spec.json (schedule, split, "
          "intrinsics, ground extent)", same)


def _test_framing():
    print("\n[T4] CAMERA RING AT RIG_RADIUS %.2f" % RIG_RADIUS)
    try:
        pre = _import_preflight()
    except Exception as exc:                                   # pragma: no cover
        check("import nonconvex_hull_preflight", False, repr(exc))
        return

    poses = camera_poses()
    ref = pre.camera_frames()
    dmax = 0.0
    for i in range(N_CAMERAS):
        dmax = max(dmax, float(np.abs(poses[i, :3, 3] - ref[i][0]).max()))
        dmax = max(dmax, float(np.abs(poses[i, :3, :3] - ref[i][1]).max()))
    check("camera ring reproduces the preflight's independent construction",
          dmax < 1e-12, "max |difference| %.3e" % dmax)

    expected = {"O1": 0.723, "O2": 0.904}
    for oname in ("O1", "O2"):
        ev, fl = ORIENTATIONS[oname]
        pts = np.concatenate([box_corners(b) for b in tuple(ev) + tuple(fl)])
        worst, clipped = 0.0, []
        for cam in range(N_CAMERAS):
            eye = poses[cam, :3, 3]
            R = poses[cam, :3, :3]
            v = (pts - eye[None, :]) @ R
            in_front = v[:, 2] < -1e-6
            depth = np.where(in_front, -v[:, 2], 1.0)
            pi = v[:, 0] / depth * FOCAL + WIDTH / 2.0 - 0.5
            pj = -v[:, 1] / depth * FOCAL + HEIGHT / 2.0 - 0.5
            worst = max(worst,
                        float(np.max(np.abs(pi - (WIDTH / 2.0 - 0.5))) / (WIDTH / 2.0)),
                        float(np.max(np.abs(pj - (HEIGHT / 2.0 - 0.5))) / (HEIGHT / 2.0)))
            if (not in_front.all() or pi.min() < 0 or pi.max() > WIDTH - 1
                    or pj.min() < 0 or pj.max() > HEIGHT - 1):
                clipped.append(cam)
        check("%s: worst normalised frame occupancy %.4f matches the spec's "
              "%.3f (tol 1e-3)" % (oname, worst, expected[oname]),
              abs(worst - expected[oname]) < 1e-3,
              "measured %.6f" % worst)
        check("%s: zero cameras clip the event object + notch filler" % oname,
              not clipped, "clipped in %s" % clipped if clipped else "")


def _test_init_rows():
    print("\n[T5] FRESH SEEDING CLOUD ROW COUNTS (spec section 5)")
    expected = {"O1": (1589, 244), "O2": (1565, 241)}
    xyz = init_cloud(0, 50_000)
    for oname in ("O1", "O2"):
        ev, fl = ORIENTATIONS[oname]
        n_ev = int(is_inside(xyz, ev).sum())
        n_fl = int(is_inside(xyz, fl).sum())
        exp_ev, exp_fl = expected[oname]
        check("%s: %d rows inside the event object (spec: %d)"
              % (oname, n_ev, exp_ev), n_ev == exp_ev)
        check("%s: %d rows inside the notch filler (spec: %d)"
              % (oname, n_fl, exp_fl), n_fl == exp_fl)
    check("event object and notch filler are disjoint in both orientations",
          all(_box_intersection(b, f) is None
              for o in ORIENTATIONS.values() for b in o[0] for f in o[1]))
    for oname in ("O1", "O2"):
        ev, fl = ORIENTATIONS[oname]
        clash = [i for i, (c, r, _, _) in enumerate(STATIC_SPHERES)
                 for b in tuple(ev) + tuple(fl)
                 if _box_intersection(b, (tuple(np.asarray(c) - r),
                                          tuple(np.asarray(c) + r))) is not None]
        check("%s: static context spheres clear of object and filler" % oname,
              not clash, "clashes with %s" % sorted(set(clash)) if clash else "")


def _test_identity_buffers():
    print("\n[T6] RENDERED IDENTITY BUFFERS, ONE FRAME PER PHASE")
    poses = camera_poses()
    # The estimator's own sampled training cameras (spec section 7).
    cams = [0, 5, 10, 15]
    probes = [("episode_1", 10), ("gap", 40), ("episode_2", 58)]
    allowed = {BACKGROUND_ID, GROUND_ID, 1, 2, 3, NOTCH_FILLER_ID, EVENT_OBJECT_ID}

    for oname in ("O1", "O2"):
        ev, fl = ORIENTATIONS[oname]
        print("  orientation %s   (event id %d, filler id %d)"
              % (oname, EVENT_OBJECT_ID, NOTCH_FILLER_ID))
        print("    %-10s %-6s %8s %8s %8s %8s %8s %8s"
              % ("phase", "cam", "bg(-1)", "grnd(0)", "sph1", "sph2/3",
                 "FILL200", "EVT100"))
        bad_ids, ev_ok, fill_gap_ok, fill_any_all_phase = set(), True, True, {}
        for phase, frame in probes:
            present = event_present_at(frame)
            if phase_of(frame) != phase:
                check("probe frame %d is in phase %s" % (frame, phase), False)
            for cam in cams:
                _rgb, identity, _d = render(poses[cam], ev, fl, present)
                vals, counts = np.unique(identity, return_counts=True)
                hist = {int(v): int(c) for v, c in zip(vals, counts)}
                bad_ids |= set(hist) - allowed
                n_ev = hist.get(EVENT_OBJECT_ID, 0)
                n_fl = hist.get(NOTCH_FILLER_ID, 0)
                print("    %-10s cam%02d %8d %8d %8d %8d %8d %8d"
                      % (phase, cam, hist.get(BACKGROUND_ID, 0),
                         hist.get(GROUND_ID, 0), hist.get(1, 0),
                         hist.get(2, 0) + hist.get(3, 0), n_fl, n_ev))
                if present and n_ev <= 0:
                    ev_ok = False
                if (not present) and n_ev != 0:
                    ev_ok = False
                if (not present) and n_fl <= 0:
                    fill_gap_ok = False
                fill_any_all_phase.setdefault(cam, []).append(n_fl)
        check("%s: identity buffer contains ONLY the expected ids %s"
              % (oname, sorted(allowed)), not bad_ids,
              "unexpected %s" % sorted(bad_ids) if bad_ids else "")
        check("%s: the event object is visible in episode 1 and at the return, "
              "and ABSENT in the gap" % oname, ev_ok)
        check("%s: the notch filler is visible from EVERY sampled camera during "
              "the gap (the deletion the falsifier detects)" % oname, fill_gap_ok)
        n_all = sum(1 for v in fill_any_all_phase.values() if all(c > 0 for c in v))
        check("%s: the notch filler is visible in ALL THREE phases from at "
              "least one sampled camera" % oname, n_all >= 1,
              "%d of %d sampled cameras" % (n_all, len(cams)))


def self_test():
    print("=" * 78)
    print("LRV5-NCX generator self-test -- NOTHING IS WRITTEN")
    print("=" * 78)
    print("numpy %s | Pillow %s | python %s"
          % (np.__version__, Image.__version__ if hasattr(Image, "__version__")
             else "?", sys.version.split()[0]))
    print("spec      : %s" % SPEC_DOC)
    print("preflight : %s" % PREFLIGHT_SCRIPT)

    _test_box_intersector()
    _test_predicate_against_preflight()
    _test_schedule_and_split()
    _test_framing()
    _test_init_rows()
    _test_identity_buffers()

    print("\n" + "=" * 78)
    if _FAILURES:
        print("SELF-TEST FAILED: %d check(s)" % len(_FAILURES))
        for f in _FAILURES:
            print("   - %s" % f)
        return 1
    print("SELF-TEST PASSED: every check green.")
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build the LRV5-NCX non-convex leave-and-return fixture.")
    ap.add_argument("--out", help="output scene directory")
    ap.add_argument("--orientation", choices=sorted(ORIENTATIONS), default="O1",
                    help="O1 is the primary orientation; O2 is the PREDECLARED "
                         "second one, (x,y,z) -> (-x + 0.1625, y, z + 0.1625). "
                         "Spec section 11: O2 is built only if H1 SURVIVES "
                         "ROUND 1.")
    ap.add_argument("--scene-id", default="LRV5",
                    help="scene id stamped into event_spec.json (spec 1: LRV5)")
    ap.add_argument("--num-init-pts", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what WOULD be written and stop. Writes nothing.")
    ap.add_argument("--self-test", action="store_true",
                    help="run every correctness check and stop. Writes nothing.")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    event_boxes, filler_boxes = ORIENTATIONS[args.orientation]
    if args.dry_run:
        return dry_run(args, event_boxes, filler_boxes)
    if not args.out:
        raise SystemExit("--out is required unless --dry-run or --self-test")
    if not lrv.gated_presence_admissible(FIRST_RETURN_FRAME):
        raise SystemExit(
            "first return frame %d does not clear floor_len; LRV5's schedule is "
            "frozen at 57 and must remain gated-presence admissible"
            % FIRST_RETURN_FRAME)
    return generate(args, event_boxes, filler_boxes)


if __name__ == "__main__":
    raise SystemExit(main())
