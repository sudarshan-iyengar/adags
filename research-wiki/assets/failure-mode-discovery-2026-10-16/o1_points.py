"""Oracle O1 (plan page section 30): starting points from MANY frames, training cameras only.

mode layout: a derived scene with the TRAINING cameras only (cam00 excluded, so the held-out view never enters
  triangulation) at frames 0, 30, ..., 270, renamed to consecutive indices 0..9 so that the Spacetime Gaussians
  per-frame COLMAP route (agent-control/realdata/stg/prep_derived.py -> pre_n3d + getcolmapsinglen3d, known poses,
  point_triangulator) runs on a contiguous range. ORIG4 = poses_bounds rows of those cameras + mp4 name placeholders.
mode merge: read colmap_k/sparse/0/points3D.bin for k = 0..9, concatenate (xyz, rgb) and write multiframe.ply
  (all three methods keep cameras in the raw poses_bounds world frame, checked 2026-10-19).
mode union: OUT = the method's own starting cloud + multiframe.ply (one change: points from other frames ADDED).
"""
import argparse, glob, os, struct, sys
import numpy as np
from plyfile import PlyData, PlyElement

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["layout", "merge", "union"])
ap.add_argument("--native"); ap.add_argument("--orig"); ap.add_argument("--root")
ap.add_argument("--own"); ap.add_argument("--multi"); ap.add_argument("--out")
ap.add_argument("--step", type=int, default=30); ap.add_argument("--n", type=int, default=10)
a = ap.parse_args()


def read_points_bin(p):
    xyz, rgb = [], []
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8); xyz.append(struct.unpack("<3d", f.read(24))); rgb.append(struct.unpack("<3B", f.read(3))); f.read(8)
            tl = struct.unpack("<Q", f.read(8))[0]; f.read(8 * tl)
    return np.array(xyz, np.float32).reshape(-1, 3), np.array(rgb, np.uint8).reshape(-1, 3)


def write_ply(p, xyz, rgb):
    dt = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    el = np.empty(len(xyz), dtype=dt)
    el["x"], el["y"], el["z"] = xyz.T; el["nx"] = el["ny"] = el["nz"] = 0; el["red"], el["green"], el["blue"] = rgb.T
    PlyData([PlyElement.describe(el, "vertex")]).write(p)


if a.mode == "layout":
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.orig, "cam*.mp4")))
    train = [c for c in names if c != "cam00"]
    sc, og = os.path.join(a.root, "scene", "images"), os.path.join(a.root, "orig")
    os.makedirs(sc, exist_ok=True); os.makedirs(og, exist_ok=True)
    for c in train:
        for k in range(a.n):
            s = os.path.join(a.native, c, f"{k * a.step}.png"); assert os.path.exists(s), s
            d = os.path.join(sc, f"{c}_{k:04d}.png")
            if not os.path.lexists(d): os.symlink(s, d)
    pb = np.load(os.path.join(a.orig, "poses_bounds.npy")); assert pb.shape[0] == len(names)
    np.save(os.path.join(og, "poses_bounds.npy"), pb[[names.index(c) for c in train]])
    for c in train:
        d = os.path.join(og, c + ".mp4")
        if not os.path.lexists(d): os.symlink(os.path.join(a.orig, c + ".mp4"), d)
    print("layout: train cams", len(train), "frames", [k * a.step for k in range(a.n)])
elif a.mode == "merge":
    X, C = [], []
    for k in range(a.n):
        x, c = read_points_bin(os.path.join(a.root, "stg", f"colmap_{k}", "sparse", "0", "points3D.bin"))
        print(f"frame {k * a.step}: {len(x)} points"); X.append(x); C.append(c)
    X, C = np.concatenate(X), np.concatenate(C)
    write_ply(os.path.join(a.root, "multiframe.ply"), X, C); print("multiframe.ply", len(X))
else:
    own = PlyData.read(a.own)["vertex"]; m = PlyData.read(a.multi)["vertex"]
    xo = np.stack([own["x"], own["y"], own["z"]], 1).astype(np.float32)
    co = np.stack([own["red"], own["green"], own["blue"]], 1)
    if co.dtype != np.uint8: co = np.clip(co * (255 if co.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    xm = np.stack([m["x"], m["y"], m["z"]], 1).astype(np.float32); cm = np.stack([m["red"], m["green"], m["blue"]], 1).astype(np.uint8)
    write_ply(a.out, np.concatenate([xo, xm]), np.concatenate([co, cm]))
    print("union", a.out, "own", len(xo), "+ multiframe", len(xm))
