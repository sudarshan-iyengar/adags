"""Observation-history census on the synthetic Absence-Return Suite (exact, CPU).

For held-out camera pixels that show a STATIC surface point at frame t, decide whether any training camera sees
that same surface point at frame t (seen now), only at earlier frames (seen only earlier), only at later frames
(seen only later), or never. Visibility is exact: static self-visibility by min-depth ray casting from the training
camera, then blocking by every non-static part present at that frame (cook, rotator, target, occupant).

Census v1 labels static surface points only (moving parts are reported as 'dynamic, unlabelled').
"""
import argparse, importlib.util, json, os, sys, zipfile, io
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--scene_dir", required=True)
ap.add_argument("--repo_root", default="D:/adags")
ap.add_argument("--train", required=True, help="comma-separated training cameras")
ap.add_argument("--test", default="cam00")
ap.add_argument("--stride", type=int, default=8)
ap.add_argument("--out", required=True)
ap.add_argument("--arc", action="store_true", help="synthetic arc rig: cameras arcNN at angle NN deg around the table")
a = ap.parse_args()

spec_mod = importlib.util.spec_from_file_location("ars", os.path.join(a.repo_root, "scripts", "build_absence_suite_scene.py"))
ars = importlib.util.module_from_spec(spec_mod)
spec_mod.loader.exec_module(ars)

spec = json.load(open(os.path.join(a.scene_dir, "event_spec.json")))
cams = {c["name"]: c for c in ars.load_rig(spec, a.repo_root)}
if a.arc:
    import math
    ref = cams["cam00"]
    centre = np.array([0.5, 7.8, 11.0]); up = np.array([0.0, -1.0, 0.0])
    cams = {}
    for ang in range(-80, 81, 4):
        th = math.radians(ang)
        C = centre + np.array([10.0 * math.sin(th), -6.5, -10.0 * math.cos(th)])
        fwd = centre - C; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up); right /= np.linalg.norm(right)   # columns: right, down, forward
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd], axis=1)
        cams[f"arc{ang:+03d}"] = {"name": f"arc{ang:+03d}", "R": R, "C": C, "f": ref["f"], "W": ref["W"], "H": ref["H"]}
train = a.train.split(",")
tc = cams[a.test]
W, H = tc["W"], tc["H"]
static = ars.scene_parts(spec, 0, ars.ROLES_STATIC)
static_ids = set(p["id"] for p, *_ in static)

# 1. static hit points of the sampled test pixels
vs, us = np.mgrid[a.stride // 2:H:a.stride, a.stride // 2:W:a.stride]
us, vs = us.ravel(), vs.ravel()
d = ars.cam_rays(tc, us, vs, ars.OFFS[ars.CENTRE:])
D = np.full(d.shape[:2], np.inf); I = np.full(d.shape[:2], -1, np.int16); Wn = np.full(d.shape[:2], -1, np.int64)
ars.composite(static, tc["C"], d, D, I, Wn)
X = tc["C"][None] + D[:, 0:1] * d[:, 0]
Ist = I[:, 0]
n = X.shape[0]


def ray_to(cam, X):
    v = X - cam["C"][None]
    L = np.sqrt((v * v).sum(1))
    return v / L[:, None], L


nonstatic_roles = ars.ROLES_ORDINARY + ars.ROLES_EVENT
NF = ars.N_FRAMES
seen = np.zeros((NF, n), bool)
per_cam_static_vis = {}
for cn in train:
    c = cams[cn]
    dirs, L = ray_to(c, X)
    uv, z = ars.project(c, X)
    infr = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < c["W"]) & (uv[:, 1] >= 0) & (uv[:, 1] < c["H"])
    Ds = np.full((n, 1), np.inf); Is = np.full((n, 1), -1, np.int16); Ws = np.full((n, 1), -1, np.int64)
    ars.composite(static, c["C"], dirs[:, None, :], Ds, Is, Ws)
    svis = infr & (Is[:, 0] == Ist) & (np.abs(Ds[:, 0] - L) <= 1e-6 * L + 1e-6)
    per_cam_static_vis[cn] = float(svis.mean())
    idx = np.flatnonzero(svis)
    for f in range(NF):
        blocked = np.zeros(idx.size, bool)
        for part, _o, R, t in ars.scene_parts(spec, f, nonstatic_roles):
            b = ars.hull_bbox(c, part, R, t)
            if b is None:
                continue
            u, v = uv[idx, 0], uv[idx, 1]
            inb = (u >= b[0] - 1) & (u < b[2] + 1) & (v >= b[1] - 1) & (v < b[3] + 1) & ~blocked
            if not inb.any():
                continue
            j = idx[inb]
            tt = ars.part_hit(part, R, t, c["C"], dirs[j][:, None, :])[:, 0]
            blocked[np.flatnonzero(inb)[tt < L[j] * (1 - 1e-6)]] = True
        seen[f, idx[~blocked]] = True

# 2. what the test camera shows at each frame (stored identity buffer, pixel-centre ray)
shows_static = np.zeros((NF, n), bool)
if a.arc:
    rr = ars.CameraRenderer(spec, tc, identity_only=True)
    for f in range(NF):
        ident = rr.frame(f)[1]
        shows_static[f] = ident[vs, us] == Ist
else:
  with zipfile.ZipFile(os.path.join(a.scene_dir, "gt", "identity", f"{a.test}.npz")) as zf:
    for f in range(NF):
        ident = np.load(io.BytesIO(zf.read(f"f{f:04d}.npy")))
        shows_static[f] = ident[vs, us] == Ist

# 3. labels (v2, after the 2026-10-16 Codex review: split past-only / past-and-future / future-only; memory-episode
#    durations counted only over frames that are in a memory class)
cum_before = np.cumsum(seen, 0) - seen            # count of seen frames strictly before
cum_after = seen[::-1].cumsum(0)[::-1] - seen      # strictly after
hidden = shows_static & ~seen
now = shows_static & seen
past_only = hidden & (cum_before > 0) & (cum_after == 0)
bracketed = hidden & (cum_before > 0) & (cum_after > 0)
future_only = hidden & (cum_before == 0) & (cum_after > 0)
never = hidden & (cum_before == 0) & (cum_after == 0)
earlier = past_only | bracketed                   # causal memory: seen at some earlier frame

last = np.full(n, -10**6)
age_hist = np.zeros(4, np.int64)   # frames since last seen, over 'earlier' pixel-frames: 1-5, 6-15, 16-30, 31+
run = np.zeros(n, np.int64); episodes = []
for f in range(NF):
    e = earlier[f]
    age = f - last
    for k, (lo, hi) in enumerate([(1, 5), (6, 15), (16, 30), (31, 10**9)]):
        age_hist[k] += int((e & (age >= lo) & (age <= hi)).sum())
    last = np.where(seen[f], f, last)
    ended = (run > 0) & ~e
    episodes.extend(run[ended].tolist())
    run = np.where(e, run + 1, 0)
episodes.extend(run[run > 0].tolist())
ep = np.asarray(episodes, dtype=np.int64)

tot = int(shows_static.sum())
res = {
    "census_version": 2,
    "scene": spec["scene_id"], "test": a.test, "train": train, "stride": a.stride, "n_sampled_px": int(n),
    "static_pixel_frames": tot, "dynamic_pixel_frames": int(NF * n - tot),
    "seen_now": int(now.sum()), "seen_before_only": int(past_only.sum()), "seen_before_and_after": int(bracketed.sum()),
    "seen_after_only": int(future_only.sum()), "never_seen": int(never.sum()),
    "frac_earlier": round(earlier.sum() / tot, 6), "frac_before_only": round(past_only.sum() / tot, 6),
    "frac_bracketed": round(bracketed.sum() / tot, 6), "frac_never": round(never.sum() / tot, 6),
    "earlier_age_hist_1_5_6_15_16_30_31p": age_hist.tolist(),
    "pixels_ever_memory": int(earlier.any(0).sum()),
    "memory_episodes": int(ep.size), "memory_episodes_ge10": int((ep >= 10).sum()), "memory_episodes_ge30": int((ep >= 30).sum()),
    "memory_episode_len_median": float(np.median(ep)) if ep.size else 0.0, "memory_episode_len_max": int(ep.max()) if ep.size else 0,
    "earlier_per_frame": earlier.sum(1).tolist(),
    "never_per_frame": never.sum(1).tolist(),
    "per_cam_static_visible_frac": per_cam_static_vis,
}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if not k.endswith("per_frame") and k != "per_cam_static_visible_frac"}))
