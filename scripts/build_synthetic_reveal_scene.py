"""Build the LRV1 controlled leave-and-return multiview event scene.

EXPLORATORY development fixture for the EL-GS oracle-headroom question. NOT a
claim-grade dataset and NOT admitted for event supply.

The scene is a deterministic analytic ray-trace of five primitives seen by a
fixed 20-camera surround ring over 60 frames. One object -- the EVENT OBJECT --
is present, then genuinely absent (removed from the world, not occluded), then
present again at the same pose with the same appearance. Because the world is
authored rather than measured, every quantity the oracle-headroom experiment
needs is exact: presence intervals, per-pixel front-most identity, and the
newly-revealed pixel-times at return.

Emits the "Blender" layout that ``scene/dataset_readers.readNerfSyntheticInfo``
already consumes (``transforms_train.json`` / ``transforms_test.json`` /
``points3d.ply``), plus ground-truth identity buffers and a frozen
``event_spec.json``.

Determinism: no unseeded randomness. The only random draw is the initialization
point cloud, from an explicit ``numpy.random.default_rng(seed)``.

Observation supply is the fixture's one free variable
--------------------------------------------------------------------------
``--first-return-frame`` sets how many times the returned surface is observed.
LRV3 (first return frame 57) gives 3 return frames x 16 training cameras = 48
training view-frames. That supply is the subject of a PRE-IDENTIFIED
experiment, not a rescue of a negative result: the 2026-08-23 block falsified
two consolidation payloads (DC and opacity) on LRV3 with oracle-correct links,
and the recorded reading of that pair of nulls is that LRV3's return is wrong
about NOTHING -- identical pose, colour and texture, densely observed -- so no
payload can carry anything into it. That reading is a mechanism claim and it is
falsifiable: if headroom is a function of observation SUPPLY rather than of
which tensor is carried, starving the return must create headroom, and if it
does not, the claim is wrong.

LRV4 is that test, and it is one variable: first return frame 59, so the return
is a SINGLE frame and the supply drops 3x to 16 training view-frames. Cameras,
held-out split, resolution, frame count, fps, ground extent, object centre,
radius, colour, texture, lighting, seed and initialization cloud are all
unchanged. Because a 1-frame return cannot clear ``FLOOR_LEN``, LRV4 is
INADMISSIBLE for gated-presence / EL-GS work and is a PAYLOAD fixture only;
``--allow-short-return`` is the default-off flag that permits it and stamps the
refusal into ``event_spec.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# Frozen scene constants. Changing any of these changes the scene identity.
# --------------------------------------------------------------------------

N_FRAMES = 60
FPS = 6.0                      # 60 frames spanning [0, 9.8333] s, window 10 s
TIME_DURATION = (0.0, 10.0)

N_CAMERAS = 20
TEST_CAMERAS = (2, 7, 12, 17)  # evenly spaced held-out views
RIG_RADIUS = 2.4
RIG_ELEVATIONS_DEG = (12.0, 28.0)   # alternating, for vertical diversity

WIDTH, HEIGHT = 400, 300
FOCAL = 420.0

# Event object presence, in FRAME indices (inclusive).
# The RETURN length is the event's difficulty knob: it sets how many
# observations the model gets of the returned surface. LRV1/LRV2 used 6 frames
# and the temporal control reconstructed the return to within 0.96 dB of its
# continuously-present performance -- below the gate's 1.0 dB minimum, i.e. too
# easy to discriminate. Admissibility bound: episode 2's duration is
# 10.5 - first_return_frame/6 and must STRICTLY exceed floor_len = 0.8333 s,
# so the first return frame may not exceed 57.
EPISODE_1_FRAMES = (0, 29)
DEFAULT_FIRST_RETURN_FRAME = 54
GAP_FRAMES = (30, DEFAULT_FIRST_RETURN_FRAME - 1)
EPISODE_2_FRAMES = (DEFAULT_FIRST_RETURN_FRAME, 59)

# The gated-presence admissibility bound, in seconds. Episode 2's duration is
# TIME_DURATION[1] + 0.5 - first_return_frame/FPS and must STRICTLY exceed
# FLOOR_LEN for the EL-GS interval representation to be able to hold it; the
# cap of 57 is that bound solved for the first return frame. `--allow-short-
# return` relaxes the CAP ONLY, and stamps the resulting scene INADMISSIBLE.
FLOOR_LEN = 10.0 / 12.0                # 0.8333 s
ADMISSIBLE_MAX_FIRST_RETURN_FRAME = 57
#: The presence window's end, in frames: (TIME_DURATION[1] + 0.5) * FPS = 63.
WINDOW_END_FRAMES = int(round((TIME_DURATION[1] + 0.5) * FPS))
#: FLOOR_LEN in frames: 0.8333 s * 6 fps = 5 frames.
FLOOR_LEN_FRAMES = int(round(FLOOR_LEN * FPS))


def episode_2_duration(first_return_frame):
    """Episode 2's duration in SECONDS -- for reporting only.

    Do not compare this against FLOOR_LEN: see
    :func:`gated_presence_admissible`.
    """
    return (TIME_DURATION[1] + 0.5) - first_return_frame / FPS


def episode_2_duration_frames(first_return_frame):
    """Episode 2's duration in FRAMES: an exact integer."""
    return WINDOW_END_FRAMES - int(first_return_frame)


def gated_presence_admissible(first_return_frame):
    """True iff episode 2 STRICTLY clears FLOOR_LEN.

    Tested in FRAMES, and it must be. At first_return_frame = 58 -- a
    2-frame return -- episode 2 lasts EXACTLY floor_len, and the LRV3
    spec refuses that case rather than silently accepting it. In float64
    the same comparison is `10.5 - 58/6 > 10/12`, whose left side rounds
    to 0.8333333333333339 against a right side of 0.8333333333333334, so
    the float form ADMITS the boundary the spec refuses. The integer form
    `63 - f0 > 5` is exact and agrees with the frozen cap of 57.
    """
    return episode_2_duration_frames(first_return_frame) > FLOOR_LEN_FRAMES


LIGHT_DIR = np.array([0.5, 1.0, 0.4])
AMBIENT = 0.25
GROUND_Y = -0.62
# LRV1 shipped 3.0 and that was a DEFECT: the initialization cloud is uniform in
# [-1.3, 1.3]^3, so a ground plane reaching to 3.0 puts 13.94% of every image on
# surface with no primitive anywhere near it. Densification clones and splits
# EXISTING primitives; it cannot create them from nothing. Those pixels are
# exactly where the optimizer is free to place floaters that satisfy the 16
# training views and are wrong from the 4 held-out ones -- LRV1's A0 reached
# 34.23 dB on training views and 19.31 dB held out. LRV2 sets it to the init
# half-width, which measures 0.00% uncovered.
DEFAULT_GROUND_HALF_EXTENT = 3.0
GROUND_HALF_EXTENT = DEFAULT_GROUND_HALF_EXTENT

# (centre, radius, base_colour, texture_mode)
STATIC_SPHERES = (
    ((-0.55, -0.25, 0.15), 0.35, (0.15, 0.62, 0.60), "stripe_y"),
    ((0.10, 0.30, -0.45), 0.28, (0.72, 0.20, 0.58), "stripe_x"),
    ((0.05, -0.35, 0.55), 0.25, (0.55, 0.70, 0.18), "checker"),
)
EVENT_SPHERE_CENTRE = (0.70, 0.10, 0.35)
EVENT_SPHERE_RADIUS = 0.20
EVENT_SPHERE_COLOUR = (0.95, 0.45, 0.10)
EVENT_SPHERE_TEXTURE = "stripe_band"

EVENT_OBJECT_ID = 100          # id used in the front-most identity buffer
BACKGROUND_ID = -1
GROUND_ID = 0


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def frame_times():
    return np.arange(N_FRAMES, dtype=np.float64) / FPS


def camera_poses():
    """Return (N_CAMERAS, 4, 4) camera-to-world matrices, OpenGL convention."""
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


def pixel_rays(c2w):
    """Return (origin (3,), directions (H, W, 3)) in world space."""
    jj, ii = np.meshgrid(np.arange(HEIGHT), np.arange(WIDTH), indexing="ij")
    px = (ii + 0.5 - WIDTH / 2.0) / FOCAL
    py = -(jj + 0.5 - HEIGHT / 2.0) / FOCAL
    dir_cam = np.stack([px, py, -np.ones_like(px)], axis=-1)
    dir_world = dir_cam @ c2w[:3, :3].T
    dir_world = dir_world / np.linalg.norm(dir_world, axis=-1, keepdims=True)
    return c2w[:3, 3].copy(), dir_world


def sphere_hit(origin, dirs, centre, radius):
    """Nearest positive ray-sphere intersection distance, inf where missed."""
    oc = origin - np.asarray(centre)
    b = dirs @ oc
    c = float(oc @ oc) - radius * radius
    disc = b * b - c
    hit = disc > 0.0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t0 = -b - sq
    t1 = -b + sq
    t = np.where(t0 > 1e-4, t0, t1)
    return np.where(hit & (t > 1e-4), t, np.inf)


def plane_hit(origin, dirs):
    denom = dirs[..., 1]
    t = np.where(np.abs(denom) > 1e-8, (GROUND_Y - origin[1]) / denom, np.inf)
    t = np.where(t > 1e-4, t, np.inf)
    p = origin + t[..., None] * dirs
    inside = ((np.abs(p[..., 0]) < GROUND_HALF_EXTENT)
              & (np.abs(p[..., 2]) < GROUND_HALF_EXTENT))
    return np.where(np.isfinite(t) & inside, t, np.inf)


def sphere_albedo(points, centre, base, mode):
    local = points - np.asarray(centre)
    base = np.asarray(base, dtype=np.float64)
    if mode == "stripe_y":
        s = np.sin(local[..., 1] * 26.0)
    elif mode == "stripe_x":
        s = np.sin(local[..., 0] * 26.0)
    elif mode == "stripe_band":
        s = np.sin(local[..., 1] * 46.0 + local[..., 0] * 18.0)
    elif mode == "checker":
        s = np.sign(np.sin(local[..., 0] * 22.0) * np.sin(local[..., 2] * 22.0))
    else:
        raise ValueError(mode)
    modulation = 0.68 + 0.32 * (s > 0.0)
    return np.clip(base[None, None, :] * modulation[..., None], 0.0, 1.0)


def ground_albedo(points):
    checker = (np.floor(points[..., 0] * 2.2) + np.floor(points[..., 2] * 2.2)) % 2
    light = np.array([0.58, 0.58, 0.62])
    dark = np.array([0.26, 0.27, 0.31])
    return np.where(checker[..., None] > 0.5, light[None, None, :], dark[None, None, :])


def render(c2w, event_present):
    """Return (rgb float32 HxWx3 in [0,1], identity int32 HxW, depth float32 HxW)."""
    origin, dirs = pixel_rays(c2w)

    depths = [plane_hit(origin, dirs)]
    ids = [GROUND_ID]
    for idx, (centre, radius, _, _) in enumerate(STATIC_SPHERES):
        depths.append(sphere_hit(origin, dirs, centre, radius))
        ids.append(idx + 1)
    if event_present:
        depths.append(sphere_hit(origin, dirs, EVENT_SPHERE_CENTRE, EVENT_SPHERE_RADIUS))
        ids.append(EVENT_OBJECT_ID)

    stack = np.stack(depths, axis=0)
    best = np.argmin(stack, axis=0)
    depth = np.take_along_axis(stack, best[None], axis=0)[0]
    hit = np.isfinite(depth)
    identity = np.where(hit, np.asarray(ids, dtype=np.int32)[best], BACKGROUND_ID)

    points = origin + np.where(hit, depth, 0.0)[..., None] * dirs
    albedo = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    normal = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)

    ground_mask = identity == GROUND_ID
    if ground_mask.any():
        albedo[ground_mask] = ground_albedo(points)[ground_mask]
        normal[ground_mask] = np.array([0.0, 1.0, 0.0])

    for idx, (centre, radius, base, mode) in enumerate(STATIC_SPHERES):
        m = identity == (idx + 1)
        if m.any():
            albedo[m] = sphere_albedo(points, centre, base, mode)[m]
            n = (points - np.asarray(centre)) / radius
            normal[m] = n[m]
    if event_present:
        m = identity == EVENT_OBJECT_ID
        if m.any():
            albedo[m] = sphere_albedo(points, EVENT_SPHERE_CENTRE,
                                      EVENT_SPHERE_COLOUR, EVENT_SPHERE_TEXTURE)[m]
            n = (points - np.asarray(EVENT_SPHERE_CENTRE)) / EVENT_SPHERE_RADIUS
            normal[m] = n[m]

    light = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
    lambert = np.clip(normal @ light, 0.0, 1.0)
    shade = AMBIENT + (1.0 - AMBIENT) * lambert
    rgb = np.clip(albedo * shade[..., None], 0.0, 1.0)
    rgb[~hit] = 0.0
    return rgb.astype(np.float32), identity, np.where(hit, depth, 0.0).astype(np.float32)


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------

def event_present_at(frame):
    return (EPISODE_1_FRAMES[0] <= frame <= EPISODE_1_FRAMES[1]
            or EPISODE_2_FRAMES[0] <= frame <= EPISODE_2_FRAMES[1])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def store_ply(path, xyz, rgb):
    n = xyz.shape[0]
    header = (
        "ply\nformat ascii 1.0\n"
        "element vertex {}\n".format(n) +
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    lines = [header]
    for i in range(n):
        lines.append(
            "{:.6f} {:.6f} {:.6f} 0.000000 0.000000 0.000000 {} {} {}\n".format(
                xyz[i, 0], xyz[i, 1], xyz[i, 2], rgb8[i, 0], rgb8[i, 1], rgb8[i, 2]))
    path.write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output scene directory")
    ap.add_argument("--num-init-pts", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene-id", default="LRV1")
    ap.add_argument("--first-return-frame", type=int,
                    default=DEFAULT_FIRST_RETURN_FRAME,
                    help="first frame of the return episode; smaller return = "
                         "harder event. Capped at 57 by floor_len.")
    ap.add_argument("--allow-short-return", action="store_true",
                    help="DEFAULT OFF. Relax the floor_len CAP so the return "
                         "may be shorter than the EL-GS interval "
                         "representation can hold (up to a SINGLE frame). The "
                         "resulting scene is INADMISSIBLE for any gated-"
                         "presence / EL-GS experiment and its event_spec.json "
                         "says so; it is a PAYLOAD fixture only, built to vary "
                         "OBSERVATION SUPPLY. Nothing else about the scene "
                         "changes, and with the flag off generation is "
                         "byte-identical to before the flag existed.")
    ap.add_argument("--ground-half-extent", type=float,
                    default=DEFAULT_GROUND_HALF_EXTENT,
                    help="ground plane half extent; MUST NOT exceed the "
                         "initialization cloud half-width, or visible surface "
                         "is left with no primitive to grow from")
    args = ap.parse_args()

    global GROUND_HALF_EXTENT, GAP_FRAMES, EPISODE_2_FRAMES
    GROUND_HALF_EXTENT = float(args.ground_half_extent)
    f0 = int(args.first_return_frame)
    floor = ADMISSIBLE_MAX_FIRST_RETURN_FRAME
    if args.allow_short_return:
        # The cap, and ONLY the cap, is relaxed. The structural minimum
        # (a return must start at least one frame after the gap opens) and
        # the last-frame bound are unchanged.
        floor = N_FRAMES - 1
    if not EPISODE_1_FRAMES[1] + 2 <= f0 <= floor:
        raise SystemExit(
            "first return frame %d is out of range [%d, %d]: %s" % (
                f0, EPISODE_1_FRAMES[1] + 2, floor,
                "the last frame is %d" % (N_FRAMES - 1)
                if args.allow_short_return else
                "episode 2 must clear floor_len, which caps it at %d "
                "(pass --allow-short-return to build an observation-STARVED "
                "payload-only fixture)"
                % ADMISSIBLE_MAX_FIRST_RETURN_FRAME))
    GAP_FRAMES = (EPISODE_1_FRAMES[1] + 1, f0 - 1)
    EPISODE_2_FRAMES = (f0, N_FRAMES - 1)
    admissible = gated_presence_admissible(f0)
    if not admissible:
        print("WARNING: episode 2 lasts %.4f s, which does NOT clear "
              "floor_len = %.4f s. Scene %r is INADMISSIBLE for gated-presence "
              "/ EL-GS experiments and is a PAYLOAD fixture only."
              % (episode_2_duration(f0), FLOOR_LEN, args.scene_id))

    out = Path(args.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    (out / "gt_identity").mkdir(parents=True, exist_ok=True)

    poses = camera_poses()
    times = frame_times()
    test_set = set(TEST_CAMERAS)

    frames_train, frames_test = [], []
    visible_px = {}          # (cam, frame) -> count of event-object front-most pixels

    for cam in range(N_CAMERAS):
        split = "test" if cam in test_set else "train"
        for f in range(N_FRAMES):
            rgb, identity, _ = render(poses[cam], event_present_at(f))
            stem = "cam{:02d}_f{:03d}".format(cam, f)
            Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(
                out / split / (stem + ".png"))
            visible_px[(cam, f)] = int((identity == EVENT_OBJECT_ID).sum())
            if cam in test_set:
                # front-most identity buffer, held-out views only: reveal-mask source
                np.save(out / "gt_identity" / (stem + ".npy"), identity.astype(np.int16))
            entry = {
                "file_path": "./{}/{}".format(split, stem),
                "time": float(times[f]),
                "transform_matrix": poses[cam].tolist(),
                "camera_index": cam,
                "frame_index": f,
            }
            (frames_test if cam in test_set else frames_train).append(entry)

    intrinsics = {
        "fl_x": FOCAL, "fl_y": FOCAL,
        "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0,
        "w": WIDTH, "h": HEIGHT,
        "camera_angle_x": float(2.0 * np.arctan(WIDTH / (2.0 * FOCAL))),
    }
    for name, frames in (("transforms_train.json", frames_train),
                         ("transforms_test.json", frames_test)):
        (out / name).write_text(json.dumps({**intrinsics, "frames": frames}, indent=1))

    rng = np.random.default_rng(args.seed)
    xyz = rng.uniform(-1.3, 1.3, size=(args.num_init_pts, 3))
    rgb = np.full((args.num_init_pts, 3), 0.5)
    store_ply(out / "points3d.ply", xyz, rgb)

    spec = {
        "scene_id": args.scene_id,
        "kind": "synthetic_leave_and_return",
        "ground_half_extent": GROUND_HALF_EXTENT,
        "evidence_bearing": False,
        "n_frames": N_FRAMES, "fps": FPS, "time_duration": list(TIME_DURATION),
        "width": WIDTH, "height": HEIGHT, "focal_px": FOCAL,
        "n_cameras": N_CAMERAS, "test_cameras": list(TEST_CAMERAS),
        "train_cameras": [c for c in range(N_CAMERAS) if c not in test_set],
        "event_object": {
            "id": EVENT_OBJECT_ID,
            "centre": list(EVENT_SPHERE_CENTRE),
            "radius": EVENT_SPHERE_RADIUS,
        },
        "presence_frames": {
            "episode_1": list(EPISODE_1_FRAMES),
            "gap": list(GAP_FRAMES),
            "episode_2": list(EPISODE_2_FRAMES),
        },
        # The oracle EPISODE PROGRAMS deliberately do NOT live here. They depend
        # on trainer-side constants this builder has no business knowing -- the
        # smoothstep half-width w and the latch endpoints +-w_m, both derived
        # from the frozen structural prereg -- and duplicating them here once
        # already produced three artifacts that disagreed about what the
        # wrong-time control was. The single source of truth is
        # configs/lrv1/oracle_{correct,wrong}.json; this file states only the
        # ground-truth presence FRAMES, which is what the scene actually knows.
        "oracle_episodes": "configs/lrv1/oracle_correct.json",
        "wrong_time_episodes": "configs/lrv1/oracle_wrong.json",
        "return_frames": list(range(EPISODE_2_FRAMES[0], EPISODE_2_FRAMES[1] + 1)),
        "event_object_pixels_per_test_view_frame": {
            "cam{:02d}_f{:03d}".format(c, f): visible_px[(c, f)]
            for c in TEST_CAMERAS for f in range(N_FRAMES)
        },
    }
    if not admissible:
        # Emitted ONLY for a scene the floor_len cap would have refused, so
        # an admissible scene's event_spec.json is byte-identical to the
        # one the generator produced before --allow-short-return existed.
        # Any consumer that gates presence MUST read this and refuse.
        spec["gated_presence_admissibility"] = {
            "admissible": False,
            "episode_2_duration_s": episode_2_duration(EPISODE_2_FRAMES[0]),
            "floor_len_s": FLOOR_LEN,
            "built_with": "--allow-short-return",
            "WARNING": (
                "INADMISSIBLE FOR GATED PRESENCE. Episode 2 is shorter than "
                "floor_len, so the EL-GS interval representation cannot hold "
                "it and no gated-presence / episodic-presence experiment may "
                "use this scene. It is a PAYLOAD fixture only: it exists to "
                "vary OBSERVATION SUPPLY at the return while holding pose, "
                "colour, texture, cameras, lighting and initialization fixed."
            ),
        }
    (out / "event_spec.json").write_text(json.dumps(spec, indent=1))

    ret = spec["return_frames"]
    tot = sum(visible_px[(c, f)] for c in TEST_CAMERAS for f in ret)
    ep1 = sum(visible_px[(c, f)] for c in TEST_CAMERAS
              for f in range(EPISODE_1_FRAMES[0], EPISODE_1_FRAMES[1] + 1))
    gap = sum(visible_px[(c, f)] for c in TEST_CAMERAS
              for f in range(GAP_FRAMES[0], GAP_FRAMES[1] + 1))
    print("held-out event pixel-times: return={} episode1={} gap={}".format(tot, ep1, gap))
    per_view = {c: sum(visible_px[(c, f)] for f in ret) for c in TEST_CAMERAS}
    print("per held-out view over return frames: {}".format(per_view))
    print("points3d.ply sha256: {}".format(sha256_file(out / "points3d.ply")))
    if not admissible:
        n_train = N_CAMERAS - len(test_set)
        print("return observation supply: {} frames x {} training cameras = {} "
              "training view-frames (gated-presence INADMISSIBLE)".format(
                  len(ret), n_train, len(ret) * n_train))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
