#!/usr/bin/env python
"""ARS -- absence-return suite scene builder (ARS fast-path plan rev 3 section 3 and
step E5; mechanism spec in the rev 2 plan sections 3-6).

ARS is fully synthetic: only the camera matrices come from a real N3V rig
(``poses_bounds.npy``); every pixel is an analytic ray cast. A scene is defined by
ONE frozen, renderer-independent JSON spec (rig reference, objects, per-part
shapes and object-frame textures, analytic trajectories, presence intervals, the
designed clock (A*, B*) and the scoring windows). This file writes those specs,
renders them, and checks them.

Subcommands
-----------
specs     write the wave-1 specs (layouts L1-L3 x conditions A0, R2, G1)
render    render a spec: canonical frame store ``frames/camXX/NNNN.png``, the
          front-most identity of every pixel centre (``gt/identity/camXX.npz``,
          int16, one entry per frame), per-camera statistics (target and occupant
          pixel counts, V6 counts); ``--identity_only`` skips colour and PNGs
gt        cam00 scoring masks, per-frame part poses and the evaluator window
          package (``scripts/eval_event_windows.py`` format)
check     V1, V2, V4, V5, V6, V10, V11 under the rev 2 section 6 applicability
          matrix, plus R0 (optimised render == brute-force render on sampled
          (camera, frame) pairs); writes ``checks.json``
montage   cam00 key-frame montages with every scored region outlined
manifest  sha256 of every file, the generator's sha256 and git state, the spec
          and trajectory hashes

Renderer (plan rev 3 section 3)
-------------------------------
* primitives: sphere, oriented box, capped cylinder, textured plane, each with a
  per-frame SE(3) pose (object root pose, optional one-axis sinusoidal joint);
* textures: procedural value-noise fBm evaluated at the OBJECT-FRAME hit point;
* shading: Lambertian plus ambient, one directional light, NO cast shadows;
* 2x2 supersampling (sub-pixel offsets 0.25/0.75) with a box filter; the identity
  buffer is the front-most part id of the pixel-CENTRE ray;
* numerics: float64 throughout, 8-bit output = floor(clip(x, 0, 1) * 255 + 0.5);
  every rotation and dot product is written elementwise (no BLAS), so the
  optimised path (static layer cached per camera, dynamic parts composited
  only inside their projected hulls) is bit-identical to a brute-force render,
  which ``check`` asserts (R0).

Camera model: pinhole, principal point at (W/2, H/2), pixel u covers continuous
coordinates [u, u+1) with its centre at u + 0.5 (the 3DGS / COLMAP convention).
``poses_bounds.npy`` rows are LLFF [down, right, back | centre | h w f]; the
raster focal is f * W / w_native. World +y is DOWN (gravity), as in the N3V
frames; ``UP = (0, -1, 0)``.
"""

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

SPEC_VERSION = "ars-1"
N_FRAMES = 300
FPS = 30.0
RASTER = (1352, 1014)
UP = np.array([0.0, -1.0, 0.0])
R_UP = np.diag([1.0, -1.0, -1.0])       # object-local +y (up) -> world up; det +1
EPS_T = 1e-9
# sub-pixel ray offsets: four supersamples, then the pixel centre (identity ray)
OFFS = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75], [0.5, 0.5]])
N_SUB = 4
CENTRE = 4
ROLES_STATIC = ("static",)
ROLES_ORDINARY = ("ordinary",)
ROLES_EVENT = ("target", "occupant")
V1_CAM00 = (3000, 8000)
V1_MIN_TRAIN_CAMS = 15
V11_MIN_PIXEL_TIMES = 5000
V10_MAX_SAMPLES = 20000   # per part; larger parts are sampled coarser (their spacing enters the certificate)
NEAR_FLOOR = 5.0          # 1.25 x Ex4DGS "near": 4.0 (CUDA min_depth cull); real N3V nearest surfaces 5.6-7.8
KEY_FRAMES = (59, 60, 75, 89, 90, 100)


# ==========================================================================
# small math, all elementwise
# ==========================================================================


def mat_vec(R, v):
    """R @ v for v (..., 3), written elementwise."""
    return v[..., 0:1] * R[:, 0] + v[..., 1:2] * R[:, 1] + v[..., 2:3] * R[:, 2]


def matT_vec(R, v):
    """R.T @ v for v (..., 3), written elementwise."""
    return v[..., 0:1] * R[0, :] + v[..., 1:2] * R[1, :] + v[..., 2:3] * R[2, :]


def dot3(a, b):
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def rot_axis(axis, ang):
    a = np.asarray(axis, dtype=np.float64)
    a = a / math.sqrt(float(a @ a))
    x, y, z = a
    c, s = math.cos(ang), math.sin(ang)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def r_upright(yaw_deg):
    """Object-local +y to world up, then a yaw about world up."""
    return rot_axis(UP, math.radians(yaw_deg)) @ R_UP


def smoothstep(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ==========================================================================
# rig
# ==========================================================================


def load_rig(spec, repo_root="."):
    rig = spec["rig"]
    p = rig["poses_bounds"]
    p = p if os.path.isabs(p) else os.path.join(repo_root, p)
    if sha256_file(p) != rig["sha256"]:
        raise SystemExit(f"rig {p} does not match the spec sha256")
    pb = np.load(p)
    W, H = rig["raster"]
    names = rig["camera_names"]
    if pb.shape != (len(names), 17):
        raise SystemExit(f"rig has {pb.shape[0]} rows, spec names {len(names)} cameras")
    cams = []
    for i, name in enumerate(names):
        M = pb[i, :15].reshape(3, 5)
        h_n, w_n, f_n = M[:, 4]
        R = np.stack([M[:, 1], M[:, 0], -M[:, 2]], axis=1)   # columns: right, down, forward
        cams.append({"name": name, "R": R, "C": M[:, 3].copy(), "f": float(f_n) * W / float(w_n),
                     "W": int(W), "H": int(H)})
    return cams


def cam_rays(cam, us, vs, offs):
    """World ray directions (m, k, 3) for pixels (us, vs) and sub-pixel offsets (k, 2)."""
    W, H, f, R = cam["W"], cam["H"], cam["f"], cam["R"]
    x = (us[:, None] + offs[None, :, 0] - W / 2.0) / f
    y = (vs[:, None] + offs[None, :, 1] - H / 2.0) / f
    d = x[..., None] * R[:, 0] + y[..., None] * R[:, 1] + R[:, 2]
    n = np.sqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2])
    return d / n[..., None]


def project(cam, X):
    """World points (n, 3) -> continuous pixel coordinates (n, 2) and camera depth (n,)."""
    Xc = matT_vec(cam["R"], X - cam["C"])
    u = cam["f"] * Xc[:, 0] / Xc[:, 2] + cam["W"] / 2.0
    v = cam["f"] * Xc[:, 1] / Xc[:, 2] + cam["H"] / 2.0
    return np.stack([u, v], axis=1), Xc[:, 2]


# ==========================================================================
# shapes (object-local frames). cylinder: axis +y, base y = 0, top y = h, capped.
# plane: local y = 0, |x| <= hx, |z| <= hz, normal +y (shaded two-sided).
# ==========================================================================


def isect(shape, o, d):
    """Nearest positive hit distance (inf where missed). o broadcastable to d (..., 3)."""
    typ = shape["type"]
    ox, oy, oz = o[..., 0], o[..., 1], o[..., 2]
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if typ == "sphere":
            r = shape["r"]
            b = ox * dx + oy * dy + oz * dz
            c = (ox * ox + oy * oy + oz * oz) - r * r
            disc = b * b - c
            sq = np.sqrt(np.maximum(disc, 0.0))
            t0 = -b - sq
            t1 = -b + sq
            t = np.where(t0 > EPS_T, t0, t1)
            return np.where((disc >= 0.0) & (t > EPS_T), t, np.inf)
        if typ == "box":
            h = shape["half"]
            tmin = np.full(np.broadcast(ox, dx).shape, -np.inf)
            tmax = np.full(tmin.shape, np.inf)
            for oo, dd, hh in ((ox, dx, h[0]), (oy, dy, h[1]), (oz, dz, h[2])):
                t1 = (-hh - oo) / dd
                t2 = (hh - oo) / dd
                tmin = np.fmax(tmin, np.fmin(t1, t2))
                tmax = np.fmin(tmax, np.fmax(t1, t2))
            hit = tmax >= np.maximum(tmin, EPS_T)
            t = np.where(tmin > EPS_T, tmin, tmax)
            return np.where(hit, t, np.inf)
        if typ == "cylinder":
            r, hh = shape["r"], shape["h"]
            a = dx * dx + dz * dz
            b = ox * dx + oz * dz
            c = (ox * ox + oz * oz) - r * r
            disc = b * b - a * c
            ts = (-b - np.sqrt(np.maximum(disc, 0.0))) / a
            ys = oy + ts * dy
            ok = (disc >= 0.0) & (a > 0.0) & (ts > EPS_T) & (ys >= 0.0) & (ys <= hh)
            t = np.where(ok, ts, np.inf)
            for yc in (0.0, hh):
                tc = (yc - oy) / dy
                px = ox + tc * dx
                pz = oz + tc * dz
                ok = (tc > EPS_T) & ((px * px + pz * pz) <= r * r)
                t = np.minimum(t, np.where(ok, tc, np.inf))
            return t
        if typ == "plane":
            hx, hz = shape["half"]
            t = -oy / dy
            px = ox + t * dx
            pz = oz + t * dz
            ok = (t > EPS_T) & (np.abs(px) <= hx) & (np.abs(pz) <= hz)
            return np.where(ok, t, np.inf)
    raise ValueError(typ)


def normal_local(shape, p):
    typ = shape["type"]
    n = np.zeros_like(p)
    if typ == "sphere":
        return p / shape["r"]
    if typ == "box":
        h = np.asarray(shape["half"], dtype=np.float64)
        q = np.abs(p) / h
        ax = np.argmax(q, axis=-1)
        rows = np.arange(p.shape[0])
        n[rows, ax] = np.sign(p[rows, ax])
        return n
    if typ == "cylinder":
        r, hh = shape["r"], shape["h"]
        rad = np.sqrt(p[:, 0] * p[:, 0] + p[:, 2] * p[:, 2])
        side = np.abs(rad - r) <= np.minimum(np.abs(p[:, 1]), np.abs(p[:, 1] - hh))
        safe = np.where(rad > 0.0, rad, 1.0)
        n[:, 0] = np.where(side, p[:, 0] / safe, 0.0)
        n[:, 2] = np.where(side, p[:, 2] / safe, 0.0)
        n[:, 1] = np.where(side, 0.0, np.where(p[:, 1] > hh / 2.0, 1.0, -1.0))
        return n
    if typ == "plane":
        n[:, 1] = 1.0
        return n
    raise ValueError(typ)


def hull_corners(shape):
    """Eight local points whose convex hull contains the shape (4 for a plane)."""
    typ = shape["type"]
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=np.float64)
    if typ == "sphere":
        return s * shape["r"]
    if typ == "box":
        return s * np.asarray(shape["half"], dtype=np.float64)
    if typ == "cylinder":
        c = s * np.array([shape["r"], 1.0, shape["r"]])
        c[:, 1] = np.where(s[:, 1] > 0, shape["h"], 0.0)
        return c
    if typ == "plane":
        hx, hz = shape["half"]
        return np.array([[-hx, 0, -hz], [-hx, 0, hz], [hx, 0, -hz], [hx, 0, hz]], dtype=np.float64)
    raise ValueError(typ)


def sdf_local(shape, p):
    """Exact signed distance (negative inside) in the local frame; planes are half-spaces (+y = free)."""
    typ = shape["type"]
    if typ == "sphere":
        return np.sqrt(dot3(p, p)) - shape["r"]
    if typ == "box":
        q = np.abs(p) - np.asarray(shape["half"], dtype=np.float64)
        outside = np.sqrt(dot3(np.maximum(q, 0.0), np.maximum(q, 0.0)))
        return outside + np.minimum(np.max(q, axis=-1), 0.0)
    if typ == "cylinder":
        r, hh = shape["r"], shape["h"]
        rad = np.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2)
        dr = rad - r
        dy = np.abs(p[..., 1] - hh / 2.0) - hh / 2.0
        out = np.sqrt(np.maximum(dr, 0.0) ** 2 + np.maximum(dy, 0.0) ** 2)
        return out + np.minimum(np.maximum(dr, dy), 0.0)
    if typ == "plane":
        return p[..., 1]
    raise ValueError(typ)


def surface_samples(shape, spacing):
    """Deterministic surface samples (n, 3), spacing <= `spacing` (planes are not sampled)."""
    typ = shape["type"]
    if typ == "sphere":
        r = shape["r"]
        n = max(64, int(math.ceil(4 * math.pi * r * r / spacing ** 2)))
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        th = math.pi * (1 + 5 ** 0.5) * i
        return np.stack([r * np.cos(th) * np.sin(phi), r * np.cos(phi), r * np.sin(th) * np.sin(phi)], 1)
    if typ == "box":
        h = np.asarray(shape["half"], dtype=np.float64)
        pts = []
        for ax in range(3):
            o1, o2 = [a for a in range(3) if a != ax]
            g1 = np.linspace(-h[o1], h[o1], max(2, int(math.ceil(2 * h[o1] / spacing)) + 1))
            g2 = np.linspace(-h[o2], h[o2], max(2, int(math.ceil(2 * h[o2] / spacing)) + 1))
            A, B = np.meshgrid(g1, g2, indexing="ij")
            for sgn in (-1.0, 1.0):
                q = np.zeros((A.size, 3))
                q[:, ax] = sgn * h[ax]
                q[:, o1] = A.ravel()
                q[:, o2] = B.ravel()
                pts.append(q)
        return np.concatenate(pts)
    if typ == "cylinder":
        r, hh = shape["r"], shape["h"]
        na = max(16, int(math.ceil(2 * math.pi * r / spacing)))
        ny = max(2, int(math.ceil(hh / spacing)) + 1)
        a = np.arange(na) * 2 * math.pi / na
        ys = np.linspace(0.0, hh, ny)
        A, Y = np.meshgrid(a, ys, indexing="ij")
        side = np.stack([r * np.cos(A.ravel()), Y.ravel(), r * np.sin(A.ravel())], 1)
        caps = []
        nr = max(2, int(math.ceil(r / spacing)) + 1)
        for rr in np.linspace(0.0, r, nr):
            k = max(1, int(math.ceil(2 * math.pi * rr / spacing)))
            aa = np.arange(k) * 2 * math.pi / k
            for yc in (0.0, hh):
                caps.append(np.stack([rr * np.cos(aa), np.full(k, yc), rr * np.sin(aa)], 1))
        return np.concatenate([side] + caps)
    raise ValueError(typ)


def support_point(shape, R, t, u):
    """World point of the shape extremal in world direction u."""
    ul = matT_vec(R, np.asarray(u, dtype=np.float64)[None])[0]
    typ = shape["type"]
    if typ == "sphere":
        pl = shape["r"] * ul / math.sqrt(float(ul @ ul))
    elif typ == "box":
        pl = np.sign(ul) * np.asarray(shape["half"], dtype=np.float64)
    elif typ == "cylinder":
        rad = math.hypot(ul[0], ul[2])
        pl = np.array([0.0, shape["h"] if ul[1] > 0 else 0.0, 0.0])
        if rad > 0:
            pl[0] = shape["r"] * ul[0] / rad
            pl[2] = shape["r"] * ul[2] / rad
    else:
        raise ValueError(typ)
    return mat_vec(R, pl[None])[0] + t


# ==========================================================================
# procedural textures, evaluated at the object-frame point
# ==========================================================================

_K1 = np.uint64(0x9E3779B97F4A7C15)
_K2 = np.uint64(0xC2B2AE3D27D4EB4F)
_K3 = np.uint64(0x165667B19E3779F9)
_M1 = np.uint64(0xFF51AFD7ED558CCD)
_M2 = np.uint64(0xC4CEB9FE1A85EC53)
_S33 = np.uint64(33)
_S11 = np.uint64(11)


def _hash01(ix, iy, iz, seed):
    h = (ix.astype(np.uint64) * _K1) ^ (iy.astype(np.uint64) * _K2) ^ (iz.astype(np.uint64) * _K3)
    h = h ^ np.uint64((seed * 0x27D4EB2F165667C5) & 0xFFFFFFFFFFFFFFFF)
    h = h ^ (h >> _S33)
    h = h * _M1
    h = h ^ (h >> _S33)
    h = h * _M2
    h = h ^ (h >> _S33)
    return (h >> _S11).astype(np.float64) * (1.0 / 9007199254740992.0)


def value_noise(p, seed):
    fl = np.floor(p)
    i = fl.astype(np.int64)
    f = p - fl
    u = f * f * (3.0 - 2.0 * f)
    ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    c000 = _hash01(ix, iy, iz, seed)
    c100 = _hash01(ix + 1, iy, iz, seed)
    c010 = _hash01(ix, iy + 1, iz, seed)
    c110 = _hash01(ix + 1, iy + 1, iz, seed)
    c001 = _hash01(ix, iy, iz + 1, seed)
    c101 = _hash01(ix + 1, iy, iz + 1, seed)
    c011 = _hash01(ix, iy + 1, iz + 1, seed)
    c111 = _hash01(ix + 1, iy + 1, iz + 1, seed)
    x00 = c000 + (c100 - c000) * ux
    x10 = c010 + (c110 - c010) * ux
    x01 = c001 + (c101 - c001) * ux
    x11 = c011 + (c111 - c011) * ux
    y0 = x00 + (x10 - x00) * uy
    y1 = x01 + (x11 - x01) * uy
    return y0 + (y1 - y0) * uz


def fbm(p, seed, octaves):
    acc = np.zeros(p.shape[0])
    amp, norm, scale = 1.0, 0.0, 1.0
    for k in range(octaves):
        acc = acc + amp * value_noise(p * scale, seed * 131 + k)
        norm += amp
        amp *= 0.5
        scale *= 2.0
    return acc / norm


def tex_eval(tex, p, nl):
    """Albedo (m, 3) of object-frame points p (m, 3) with local normals nl."""
    typ = tex["type"]
    if typ == "noise":
        q = p * np.asarray(tex.get("aniso", [1.0, 1.0, 1.0]), dtype=np.float64) / tex["scale"]
        w = fbm(q, tex["seed"], tex.get("octaves", 5))
        w = np.clip((w - 0.5) * tex.get("contrast", 2.5) + 0.5, 0.0, 1.0)
        c0 = np.asarray(tex["c0"], dtype=np.float64)
        c1 = np.asarray(tex["c1"], dtype=np.float64)
        col = c0 + (c1 - c0) * w[:, None]
        sp = tex.get("speckle")
        if sp:
            s = fbm(p / sp["scale"], sp["seed"], 3)
            col = col * (1.0 - sp["amount"] + 2.0 * sp["amount"] * s)[:, None]
        return col
    if typ == "banded":
        col = tex_eval(tex["base"], p, nl)
        for band in tex["bands"]:
            sel = (p[:, 1] >= band["y"][0]) & (p[:, 1] <= band["y"][1])
            if band.get("phi_deg") is not None:
                phi = np.degrees(np.arctan2(p[:, 0], p[:, 2]))
                lo, hi = band["phi_deg"]
                sel &= (phi >= lo) & (phi <= hi)
            if band.get("r_min") is not None:
                sel &= np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) >= band["r_min"]
            if sel.any():
                col[sel] = tex_eval(band["tex"], p[sel], nl[sel])
        return col
    if typ == "faces":
        ax = np.argmax(np.abs(nl), axis=1)
        sg = np.sign(nl[np.arange(nl.shape[0]), ax])
        col = np.zeros((p.shape[0], 3))
        for key, sub in tex["faces"].items():
            a = "xyz".index(key[1])
            s = 1.0 if key[0] == "+" else -1.0
            sel = (ax == a) & (sg == s)
            if sel.any():
                col[sel] = tex_eval(sub, p[sel], nl[sel])
        return col
    if typ == "sectors":
        k = len(tex["texs"])
        phi = np.mod(np.arctan2(p[:, 0], p[:, 2]), 2 * math.pi)
        idx = np.minimum((phi / (2 * math.pi / k)).astype(np.int64), k - 1)
        cap = np.abs(nl[:, 1]) > 0.5
        col = np.zeros((p.shape[0], 3))
        for j, sub in enumerate(tex["texs"]):
            sel = (idx == j) & ~cap
            if sel.any():
                col[sel] = tex_eval(sub, p[sel], nl[sel])
        if cap.any():
            col[cap] = tex_eval(tex["cap"], p[cap], nl[cap])
        return col
    raise ValueError(typ)


# ==========================================================================
# scene model: objects, parts, motion
# ==========================================================================


def root_pose(motion, frame):
    typ = motion["type"]
    t = frame / FPS
    if typ == "static":
        R = np.asarray(motion["R"], dtype=np.float64) if "R" in motion else r_upright(motion.get("yaw_deg", 0.0))
        return R, np.asarray(motion["pos"], dtype=np.float64)
    if typ == "keyed":
        for k in motion["keys"]:
            if k["frames"][0] <= frame <= k["frames"][1]:
                return r_upright(k.get("yaw_deg", 0.0)), np.asarray(k["pos"], dtype=np.float64)
        raise ValueError(f"keyed motion has no key at frame {frame}")
    if typ in ("slide", "roll"):
        a, b = motion["frames"]
        s = smoothstep((frame - a) / float(b - a))
        p0 = np.asarray(motion["from"], dtype=np.float64)
        p1 = np.asarray(motion["to"], dtype=np.float64)
        pos = p0 + (p1 - p0) * s
        R = r_upright(motion.get("yaw_deg", 0.0))
        if typ == "roll":
            dist = float(np.linalg.norm(p1 - p0))
            axis = np.cross(UP, (p1 - p0) / dist)
            R = rot_axis(axis, dist * s / motion["radius"]) @ R
        return R, pos
    if typ == "spin":
        return r_upright(motion["yaw0_deg"] + 360.0 * t / motion["period_s"]), np.asarray(motion["pos"], dtype=np.float64)
    if typ == "sway":
        off = motion["amp"] * math.sin(2 * math.pi * t / motion["period_s"] + math.radians(motion["phase_deg"]))
        pos = np.asarray(motion["pos"], dtype=np.float64) + off * np.asarray(motion["dir"], dtype=np.float64)
        return r_upright(motion.get("yaw_deg", 0.0)), pos
    raise ValueError(typ)


def part_pose(part, Rr, tr, frame):
    loc = part.get("local", {})
    Rl = np.asarray(loc["R"], dtype=np.float64) if "R" in loc else np.eye(3)
    tl = np.asarray(loc.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    # joint chain, outermost first (e.g. [shoulder, elbow]); pivots and axes are in the
    # object's rest frame, so the innermost joint is applied first
    for j in reversed(part.get("joints", [])):
        ang = j.get("offset_deg", 0.0) + j["amp_deg"] * math.sin(
            2 * math.pi * (frame / FPS) / j["period_s"] + math.radians(j["phase_deg"]))
        Rj = rot_axis(j["axis"], math.radians(ang))
        pv = np.asarray(j["pivot"], dtype=np.float64)
        Rl = Rj @ Rl
        tl = Rj @ (tl - pv) + pv
    return Rr @ Rl, Rr @ tl + tr


def present(obj, frame):
    pr = obj.get("presence")
    return pr is None or any(a <= frame <= b for a, b in pr)


def scene_parts(spec, frame, roles):
    """[(part, obj, R, t)] for present objects with a role in `roles`, in spec order."""
    out = []
    for obj in spec["objects"]:
        if obj["role"] not in roles or not present(obj, frame):
            continue
        Rr, tr = root_pose(obj["motion"], frame)
        for part in obj["parts"]:
            R, t = part_pose(part, Rr, tr, frame)
            out.append((part, obj, R, t))
    return out


def ids_by_role(spec):
    out = {}
    for obj in spec["objects"]:
        out.setdefault(obj["role"], []).extend(p["id"] for p in obj["parts"])
    return out


# ==========================================================================
# rendering core
# ==========================================================================


def part_hit(part, R, t, o, d):
    ol = matT_vec(R, (o - t)[None])[0]
    dl = matT_vec(R, d)
    return isect(part["shape"], ol, dl)


def shade(part, R, t, o, d, tt, light, ambient):
    ol = matT_vec(R, (o - t)[None])[0]
    dl = matT_vec(R, d)
    pl = ol + tt[:, None] * dl
    nl = normal_local(part["shape"], pl)
    n = mat_vec(R, nl)
    flip = dot3(n, d) > 0.0
    n = np.where(flip[:, None], -n, n)
    lam = np.maximum(dot3(n, light[None]), 0.0)
    s = ambient + (1.0 - ambient) * lam
    return tex_eval(part["tex"], pl, nl) * s[:, None]


def composite(plist, o, d, D, I, Wn, base_index=0):
    """Min-depth composite of plist over rays d (m, k, 3); strict < keeps the earlier part."""
    for j, (part, _obj, R, t) in enumerate(plist):
        tt = part_hit(part, R, t, o, d)
        closer = tt < D
        if closer.any():
            D[closer] = tt[closer]
            I[closer] = part["id"]
            Wn[closer] = base_index + j


def shade_winners(plist, o, d, D, Wn, C, light, ambient, base_index=0):
    """Colour the supersample rays (first N_SUB) won by each part of plist into C (m, N_SUB, 3)."""
    for j, (part, _obj, R, t) in enumerate(plist):
        sel = Wn[:, :N_SUB] == base_index + j
        if sel.any():
            r, k = np.nonzero(sel)
            C[r, k] = shade(part, R, t, o, d[r, k], D[r, k], light, ambient)


def box_mean(C):
    return (C[:, 0] + C[:, 1] + C[:, 2] + C[:, 3]) * 0.25


def to8(x):
    return np.floor(np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def hull_bbox(cam, part, R, t):
    """Half-open pixel box [u0, v0, u1, v1] containing the part's projection (+1 px), or the full frame."""
    W, H = cam["W"], cam["H"]
    X = mat_vec(R, hull_corners(part["shape"])) + t
    Xc = matT_vec(cam["R"], X - cam["C"])
    if (Xc[:, 2] <= 1e-3).any():
        return [0, 0, W, H]
    u = cam["f"] * Xc[:, 0] / Xc[:, 2] + W / 2.0
    v = cam["f"] * Xc[:, 1] / Xc[:, 2] + H / 2.0
    u0 = max(0, int(math.floor(u.min())) - 1)
    u1 = min(W, int(math.floor(u.max())) + 2)
    v0 = max(0, int(math.floor(v.min())) - 1)
    v1 = min(H, int(math.floor(v.max())) + 2)
    if u0 >= u1 or v0 >= v1:
        return None
    return [u0, v0, u1, v1]


def render_brute(spec, cam, frame, roles=ROLES_STATIC + ROLES_ORDINARY + ROLES_EVENT, identity_only=False):
    """Reference render: every part against every ray of the frame. Returns (rgb8 or None, identity int16)."""
    W, H = cam["W"], cam["H"]
    light = np.asarray(spec["render"]["light_dir"], dtype=np.float64)
    light = light / math.sqrt(float(light @ light))
    amb = spec["render"]["ambient"]
    # same part order as the optimised path: static, then ordinary, then event objects
    plist = [x for grp in (ROLES_STATIC, ROLES_ORDINARY, ROLES_EVENT)
             for x in scene_parts(spec, frame, [r for r in grp if r in roles])]
    rgb = np.zeros((H, W, 3))
    ident = np.zeros((H, W), dtype=np.int16)
    offs = OFFS[CENTRE:] if identity_only else OFFS
    for v0 in range(0, H, 64):
        vs, us = np.mgrid[v0:min(H, v0 + 64), 0:W]
        us, vs = us.ravel(), vs.ravel()
        d = cam_rays(cam, us, vs, offs)
        D = np.full(d.shape[:2], np.inf)
        I = np.full(d.shape[:2], -1, dtype=np.int16)
        Wn = np.full(d.shape[:2], -1, dtype=np.int64)
        composite(plist, cam["C"], d, D, I, Wn)
        ident[vs, us] = I[:, -1]
        if not identity_only:
            C = np.zeros((d.shape[0], N_SUB, 3))
            shade_winners(plist, cam["C"], d, D, Wn, C, light, amb)
            rgb[vs, us] = box_mean(C)
    return (None if identity_only else to8(rgb)), ident


class CameraRenderer:
    """Optimised renderer for one camera: static layer cached, dynamic parts composited in their hulls."""

    def __init__(self, spec, cam, identity_only=False):
        self.spec, self.cam, self.idonly = spec, cam, identity_only
        self.offs = OFFS[CENTRE:] if identity_only else OFFS
        self.k = self.offs.shape[0]
        L = np.asarray(spec["render"]["light_dir"], dtype=np.float64)
        self.light = L / math.sqrt(float(L @ L))
        self.amb = spec["render"]["ambient"]
        W, H = cam["W"], cam["H"]
        self.static = scene_parts(spec, 0, ROLES_STATIC)
        self.sD = np.empty((H * W, self.k))
        self.sI = np.empty((H * W, self.k), dtype=np.int16)
        self.sC = None if identity_only else np.empty((H * W, N_SUB, 3))
        for v0 in range(0, H, 64):
            vs, us = np.mgrid[v0:min(H, v0 + 64), 0:W]
            us, vs = us.ravel(), vs.ravel()
            d = cam_rays(cam, us, vs, self.offs)
            D = np.full(d.shape[:2], np.inf)
            I = np.full(d.shape[:2], -1, dtype=np.int16)
            Wn = np.full(d.shape[:2], -1, dtype=np.int64)
            composite(self.static, cam["C"], d, D, I, Wn)
            if not np.isfinite(D).all():
                raise SystemExit(f"{cam['name']}: static world leaves rays with no hit")
            sl = slice(v0 * W, v0 * W + us.size)
            self.sD[sl], self.sI[sl] = D, I
            if not identity_only:
                C = np.zeros((d.shape[0], N_SUB, 3))
                shade_winners(self.static, cam["C"], d, D, Wn, C, self.light, self.amb)
                self.sC[sl] = C
        self.static_rgb = None if identity_only else box_mean(self.sC).reshape(H, W, 3)
        self.static_id = self.sI[:, -1].reshape(H, W).copy()
        self.dz = self._centre_dz(np.arange(H * W))
        self.static_zmin = float((self.sD[:, -1] * self.dz).min())

    def _centre_dz(self, pix):
        """Camera-frame z of the unit pixel-centre ray, so camera depth = t * dz."""
        W, H, f = self.cam["W"], self.cam["H"], self.cam["f"]
        x = (pix % W + 0.5 - W / 2.0) / f
        y = (pix // W + 0.5 - H / 2.0) / f
        return 1.0 / np.sqrt(x * x + y * y + 1.0)

    def frame(self, f, v6=False):
        """(rgb8 or None, identity int16 (H, W), v6 dict or None, dirty pixel count)."""
        cam, W, H = self.cam, self.cam["W"], self.cam["H"]
        ordn = scene_parts(self.spec, f, ROLES_ORDINARY)
        evt = scene_parts(self.spec, f, ROLES_EVENT)
        boxes = [hull_bbox(cam, p, R, t) for p, _o, R, t in ordn + evt]
        dirty = np.zeros((H, W), dtype=bool)
        for b in boxes:
            if b is not None:
                dirty[b[1]:b[3], b[0]:b[2]] = True
        pix = np.flatnonzero(dirty)
        ident = self.static_id.copy()
        rgb = None if self.idonly else self.static_rgb.copy()
        v6r = None
        if pix.size:
            us, vs = pix % W, pix // W
            d = cam_rays(cam, us, vs, self.offs)
            D = self.sD[pix].copy()
            I = self.sI[pix].copy()
            Wn = np.full(D.shape, -1, dtype=np.int64)
            pos = np.full(H * W, -1, dtype=np.int64)
            pos[pix] = np.arange(pix.size)
            pos = pos.reshape(H, W)

            def comp(j, item):
                b = boxes[j]
                if b is None:
                    return
                sub = pos[b[1]:b[3], b[0]:b[2]].ravel()
                Ds, Is, Ws = D[sub], I[sub], Wn[sub]
                composite([item], cam["C"], d[sub], Ds, Is, Ws, base_index=j)
                D[sub], I[sub], Wn[sub] = Ds, Is, Ws

            for j, item in enumerate(ordn):
                comp(j, item)
            Db, Wb = D.copy(), Wn.copy()          # the base world: static + ordinary dynamics
            for j, item in enumerate(evt):
                comp(len(ordn) + j, item)
            ident.ravel()[pix] = I[:, -1]
            if not self.idonly:
                Cb = self.sC[pix].copy()
                shade_winners(ordn, cam["C"], d, Db, Wb, Cb, self.light, self.amb, base_index=0)
                C = Cb.copy()
                shade_winners(evt, cam["C"], d, D, Wn, C, self.light, self.amb, base_index=len(ordn))
                rgb.reshape(-1, 3)[pix] = box_mean(C)
                if v6:
                    v6r = self._v6(evt, pix, to8(box_mean(C)), to8(box_mean(Cb)))
        elif v6 and not self.idonly:
            v6r = {"diff_px": 0, "outside_support_px": 0}
        self.last_zmin = min(self.static_zmin, float((D[:, -1] * self.dz[pix]).min())) if pix.size else self.static_zmin
        return (None if self.idonly else to8(rgb)), ident, v6r, int(pix.size)

    def _v6(self, evt, pix, scene8, base8):
        """Pixels where the scene differs from the base world must lie within 1 px of an
        event-object silhouette. The silhouette is recomputed here by brute force at the 3x3
        neighbourhood of every differing pixel, independent of the hull boxes."""
        cam, W, H = self.cam, self.cam["W"], self.cam["H"]
        diff = (scene8 != base8).any(axis=1)
        dp = pix[diff]
        out = {"diff_px": int(dp.size), "outside_support_px": 0}
        if dp.size == 0:
            return out
        u, v = dp % W, dp // W
        nb = []
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                nb.append(np.stack([np.clip(u + du, 0, W - 1), np.clip(v + dv, 0, H - 1)], 1))
        nb = np.stack(nb, 1)                                   # (n, 9, 2)
        flat = (nb[..., 1] * W + nb[..., 0]).ravel()
        uniq, inv = np.unique(flat, return_inverse=True)
        d = cam_rays(cam, uniq % W, uniq // W, OFFS[:N_SUB])
        hit = np.zeros(uniq.size, dtype=bool)
        for p, _o, R, t in evt:
            hit |= np.isfinite(part_hit(p, R, t, cam["C"], d)).any(axis=1)
        ok = hit[inv].reshape(dp.size, 9).any(axis=1)
        out["outside_support_px"] = int((~ok).sum())
        return out


# ==========================================================================
# render subcommand
# ==========================================================================


def _npy_bytes(a):
    b = io.BytesIO()
    np.lib.format.write_array(b, np.ascontiguousarray(a), allow_pickle=False)
    return b.getvalue()


def _render_camera(args):
    spec_path, out_dir, ci, frames, identity_only, v6, repo_root = args
    spec = json.load(open(spec_path))
    cam = load_rig(spec, repo_root)[ci]
    ids = ids_by_role(spec)
    tgt = np.array(ids.get("target", []), dtype=np.int16)
    occ = np.array(ids.get("occupant", []), dtype=np.int16)
    t0 = time.time()
    rr = CameraRenderer(spec, cam, identity_only)
    t_static = time.time() - t0
    fdir = os.path.join(out_dir, "frames", cam["name"])
    idir = os.path.join(out_dir, "gt", "identity")
    os.makedirs(idir, exist_ok=True)
    if not identity_only:
        os.makedirs(fdir, exist_ok=True)
    tag = "" if len(frames) == N_FRAMES else "_partial"
    zpath = os.path.join(idir, f"{cam['name']}{tag}.npz")
    stats = {"camera": cam["name"], "frames": [], "target_px": [], "occupant_px": [], "dirty_px": [],
             "v6_diff_px": [], "v6_outside_support_px": []}
    with zipfile.ZipFile(zpath + ".tmp", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in frames:
            rgb, ident, v6r, ndirty = rr.frame(f, v6=v6 and not identity_only)
            zf.writestr(f"f{f:04d}.npy", _npy_bytes(ident))
            if rgb is not None:
                Image.fromarray(rgb).save(os.path.join(fdir, f"{f:04d}.png"))
            stats["frames"].append(f)
            stats["target_px"].append(int(np.isin(ident, tgt).sum()) if tgt.size else 0)
            stats["occupant_px"].append(int(np.isin(ident, occ).sum()) if occ.size else 0)
            stats["dirty_px"].append(ndirty)
            stats.setdefault("min_camera_depth", []).append(round(rr.last_zmin, 4))
            if v6r is not None:
                stats["v6_diff_px"].append(v6r["diff_px"])
                stats["v6_outside_support_px"].append(v6r["outside_support_px"])
    os.replace(zpath + ".tmp", zpath)
    stats["seconds_static"] = round(t_static, 2)
    stats["seconds_total"] = round(time.time() - t0, 2)
    stats["identity_only"] = identity_only
    sdir = os.path.join(out_dir, "gt", "stats")
    os.makedirs(sdir, exist_ok=True)
    with open(os.path.join(sdir, f"{cam['name']}{tag}.json"), "w", newline="\n") as fh:
        json.dump(stats, fh)
    return cam["name"], stats["seconds_total"]


def cmd_render(a):
    spec = json.load(open(a.spec))
    cams = load_rig(spec, a.repo_root)
    names = [c["name"] for c in cams]
    sel = list(range(len(cams))) if not a.cams else [names.index(c) for c in a.cams.split(",")]
    frames = list(range(N_FRAMES)) if not a.frames else [int(x) for x in a.frames.split(",")]
    os.makedirs(os.path.join(a.out, "rig"), exist_ok=True)
    with open(a.spec, "rb") as src, open(os.path.join(a.out, "event_spec.json"), "wb") as dst:
        dst.write(src.read())
    rig_src = spec["rig"]["poses_bounds"]
    rig_src = rig_src if os.path.isabs(rig_src) else os.path.join(a.repo_root, rig_src)
    with open(rig_src, "rb") as src, open(os.path.join(a.out, "rig", "poses_bounds.npy"), "wb") as dst:
        dst.write(src.read())            # load_rig above already verified this file's sha256 against the spec
    jobs = [(os.path.abspath(a.spec), a.out, ci, frames, a.identity_only, not a.no_v6, a.repo_root) for ci in sel]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs))) as ex:
        for name, secs in ex.map(_render_camera, jobs):
            print(f"  {spec['scene_id']} {name}: {secs:.1f} s", flush=True)
    print(f"{spec['scene_id']}: {len(sel)} cameras x {len(frames)} frames in {time.time() - t0:.1f} s")


# ==========================================================================
# gt subcommand: cam00 masks, part poses, window package
# ==========================================================================


def load_identity(out_dir, cam_name, frame, partial=False):
    tag = "_partial" if partial else ""
    with np.load(os.path.join(out_dir, "gt", "identity", f"{cam_name}{tag}.npz")) as z:
        return z[f"f{frame:04d}"]


def alone_mask(spec, cam, role, frame, pose_frame=None, offs=OFFS[CENTRE:]):
    """Pixels whose rays hit an object of `role` alone (no occlusion), at the pose of `pose_frame`
    (default: `frame`), ignoring presence at `frame`. offs = centre ray (default) or supersamples."""
    pf = frame if pose_frame is None else pose_frame
    W, H = cam["W"], cam["H"]
    parts = []
    for obj in spec["objects"]:
        if obj["role"] != role:
            continue
        Rr, tr = root_pose(obj["motion"], pf)
        for p in obj["parts"]:
            R, t = part_pose(p, Rr, tr, pf)
            parts.append((p, obj, R, t))
    m = np.zeros((H, W), dtype=bool)
    for p, _o, R, t in parts:
        b = hull_bbox(cam, p, R, t)
        if b is None:
            continue
        vs, us = np.mgrid[b[1]:b[3], b[0]:b[2]]
        d = cam_rays(cam, us.ravel(), vs.ravel(), offs)
        hit = np.isfinite(part_hit(p, R, t, cam["C"], d)).any(axis=1)
        m[vs.ravel()[hit], us.ravel()[hit]] = True
    return m


def windows_of(spec):
    return spec["windows"]


def cmd_gt(a):
    spec = json.load(open(os.path.join(a.out, "event_spec.json")))
    cams = load_rig(spec, a.repo_root)
    cam = cams[0]
    if cam["name"] != "cam00":
        raise SystemExit("first rig camera is not cam00")
    ids = ids_by_role(spec)
    A, B = spec["clock"]["A"], spec["clock"]["B"]
    tgt = np.array(ids["target"], dtype=np.int16)
    occ = np.array(ids.get("occupant", []), dtype=np.int16)
    ordi = np.array(ids.get("ordinary", []), dtype=np.int16)
    stat = np.array(ids["static"], dtype=np.int16)
    ghost = alone_mask(spec, cam, "target", A - 1)
    # union of every event-object silhouette over the sequence (target poses + occupant path), +2 px
    from scipy.ndimage import binary_dilation
    ev_union = np.zeros_like(ghost)
    tobj = [o for o in spec["objects"] if o["role"] == "target"][0]
    tposes = sorted({k["frames"][0] for k in tobj["motion"]["keys"]})
    for pf in tposes:
        ev_union |= alone_mask(spec, cam, "target", pf, offs=OFFS[:N_SUB])
    has_occ = any(o["role"] == "occupant" for o in spec["objects"])
    if has_occ:
        for f in range(N_FRAMES):
            ev_union |= alone_mask(spec, cam, "occupant", f, offs=OFFS[:N_SUB])
    ev_union = binary_dilation(ev_union, structure=np.ones((5, 5), dtype=bool))
    mroot = os.path.join(a.out, "gt", "masks")
    regions = ["ghost", "target", "ordinary", "static_complement"]
    if spec["condition"] == "R2":
        regions.append("vacated")
    if has_occ:
        regions.append("occupant")
    for r in regions:
        os.makedirs(os.path.join(mroot, r), exist_ok=True)
    counts = {r: [] for r in regions}
    part_ids, Rs, ts, pres = [], [], [], []
    all_parts = [(p, o) for o in spec["objects"] for p in o["parts"]]
    for f in range(N_FRAMES):
        ident = load_identity(a.out, "cam00", f)
        cur_t = alone_mask(spec, cam, "target", f) if present(tobj, f) else np.zeros_like(ghost)
        masks = {
            "ghost": ghost,
            "target": np.isin(ident, tgt),
            "ordinary": np.isin(ident, ordi),
            "static_complement": np.isin(ident, stat) & ~ev_union,
        }
        if "vacated" in regions:
            masks["vacated"] = ghost & ~cur_t
        if "occupant" in regions:
            masks["occupant"] = ghost & np.isin(ident, occ)
        for r in regions:
            Image.fromarray((masks[r] * 255).astype(np.uint8)).save(os.path.join(mroot, r, f"{f:05d}.png"))
            counts[r].append(int(masks[r].sum()))
        Rf, tf, pf_ = [], [], []
        for p, o in all_parts:
            Rr, tr = root_pose(o["motion"], f) if (present(o, f) or o["motion"]["type"] != "keyed") else (np.eye(3), np.zeros(3))
            R, t = part_pose(p, Rr, tr, f)
            Rf.append(R)
            tf.append(t)
            pf_.append(present(o, f))
        Rs.append(Rf)
        ts.append(tf)
        pres.append(pf_)
    for p, o in all_parts:
        part_ids.append(p["id"])
    np.savez_compressed(os.path.join(a.out, "gt", "part_poses.npz"), part_ids=np.array(part_ids, dtype=np.int16),
                        R=np.array(Rs), t=np.array(ts), present=np.array(pres))
    with open(os.path.join(a.out, "gt", "cam00_region_counts.json"), "w", newline="\n") as fh:
        json.dump(counts, fh)
    pkg = {
        "package_version": "1",
        "scene": spec["scene_id"],
        "camera": "cam00",
        "raster": spec["rig"]["raster"],
        "gt": {"dir": "../frames/cam00", "pattern": "%04d.png"},
        "frames": [0, N_FRAMES - 1],
        "regions": dict({"whole": {"kind": "whole"}},
                        **{r: {"kind": "mask_dir", "dir": f"masks/{r}", "pattern": "%05d.png", "threshold": 127}
                           for r in regions}),
        "windows": spec["windows"],
        "source": {"tool": "scripts/build_absence_suite_scene.py gt", "spec_sha256": sha256_file(os.path.join(a.out, "event_spec.json"))},
    }
    with open(os.path.join(a.out, "gt", "window_package.json"), "w", newline="\n") as fh:
        json.dump(pkg, fh, indent=1)
    print(f"{spec['scene_id']}: masks {regions}; window package written")


# ==========================================================================
# check subcommand
# ==========================================================================


def load_stats(out_dir, cams, partial=False):
    tag = "_partial" if partial else ""
    return {c["name"]: json.load(open(os.path.join(out_dir, "gt", "stats", f"{c['name']}{tag}.json"))) for c in cams}


def shape_area(shape):
    typ = shape["type"]
    if typ == "sphere":
        return 4.0 * math.pi * shape["r"] ** 2
    if typ == "box":
        hx, hy, hz = shape["half"]
        return 8.0 * (hx * hy + hy * hz + hx * hz)
    if typ == "cylinder":
        return 2.0 * math.pi * shape["r"] * shape["h"] + 2.0 * math.pi * shape["r"] ** 2
    return math.inf


def check_v10(spec):
    """Collision and free space, every frame, every pair of distinct present objects.

    Each non-plane part is sampled on its surface at spacing s_p = max(s, sqrt(area / V10_MAX_SAMPLES)).
    For a part pair and one direction (samples of part x against the exact SDF of part y), the minimum
    sampled SDF d_xy certifies the true distance >= d_xy - s_x (the SDF is 1-Lipschitz). The part pair's
    certificate is the best such bound over its two directions, or its bounding-sphere separation when
    that already exceeds c + 0.05; any negative sample in either direction is penetration and the pair
    fails. An object pair passes iff min over its part pairs of the certificate >= clearance c.
    Declared support contacts must instead touch (analytic lowest point on the support surface,
    |gap| <= 1e-9) and not penetrate (sampled SDF >= -1e-9); declared mounted contacts must touch
    (|min sampled SDF| <= 1e-9, which holds exactly for samples on the shared face)."""
    c = spec["clearance"]
    s = spec.get("v10_spacing", 0.02)
    supports = {tuple(p) for p in spec["support_contacts"]}
    mounted = {tuple(p) for p in spec.get("mounted_contacts", [])}
    mounted |= {(b, a) for a, b in mounted}
    objs = spec["objects"]
    byname = {o["name"]: o for o in objs}
    spacing, samples = {}, {}
    for o in objs:
        for p in o["parts"]:
            if p["shape"]["type"] != "plane":
                spacing[id(p)] = max(s, math.sqrt(shape_area(p["shape"]) / V10_MAX_SAMPLES))
                samples[id(p)] = surface_samples(p["shape"], spacing[id(p)])
    cert_by_pair, raw_by_pair = {}, {}
    fails = []
    support_rows = []

    def bound(p, R, t):
        if p["shape"]["type"] == "plane":
            return None, math.inf
        hc = hull_corners(p["shape"])
        cl = (hc.min(axis=0) + hc.max(axis=0)) / 2.0
        return mat_vec(R, cl[None])[0] + t, float(np.sqrt(((hc - cl) ** 2).sum(axis=1)).max())

    static_cache = {}
    for f in range(N_FRAMES):
        posed = {}
        for o in objs:
            if not present(o, f):
                continue
            if o["role"] == "static" and o["name"] in static_cache:
                posed[o["name"]] = static_cache[o["name"]]
                continue
            Rr, tr = root_pose(o["motion"], f)
            rows = []
            for p in o["parts"]:
                R, t = part_pose(p, Rr, tr, f)
                bc, br = bound(p, R, t)
                W_ = mat_vec(R, samples[id(p)]) + t if id(p) in samples else None
                rows.append((p, R, t, bc, br, W_))
            posed[o["name"]] = rows
            if o["role"] == "static":
                static_cache[o["name"]] = rows
        names = list(posed)
        for i, na in enumerate(names):
            for nb in names[i + 1:]:
                if byname[na]["role"] == "static" and byname[nb]["role"] == "static" and f > 0:
                    continue
                cert, raw = math.inf, math.inf
                for pa in posed[na]:
                    for pb in posed[nb]:
                        if pa[3] is not None and pb[3] is not None:
                            lower = float(np.linalg.norm(pa[3] - pb[3])) - pa[4] - pb[4]
                            if lower > c + 0.05:
                                cert = min(cert, lower)
                                continue
                        best, pen = -math.inf, math.inf
                        for x, y in ((pa, pb), (pb, pa)):
                            if x[5] is None:
                                continue
                            sx = spacing[id(x[0])]
                            pts = x[5]
                            if y[3] is not None:
                                pts = pts[np.sqrt(dot3(pts - y[3], pts - y[3])) <= y[4] + c + sx + 0.05]
                            if pts.size == 0:
                                best = max(best, c + 0.05)      # every sample is beyond y's bound + c + s_x + 0.05
                                continue
                            dd = float(sdf_local(y[0]["shape"], matT_vec(y[1], pts - y[2])).min())
                            raw = min(raw, dd)
                            pen = min(pen, dd)
                            best = max(best, dd - sx)
                        cp = pen if pen < 0.0 else (best if best > -math.inf else math.inf)
                        cert = min(cert, cp)
                k = (na, nb)
                cert_by_pair[k] = min(cert_by_pair.get(k, math.inf), cert)
                raw_by_pair[k] = min(raw_by_pair.get(k, math.inf), raw)
                key = (na, nb) if (na, nb) in supports else (nb, na) if (nb, na) in supports else None
                if key is not None:
                    sup_obj, base_obj = key
                    lows = []
                    for p, R, t, _bc, _br, _w in posed[sup_obj]:
                        if p["shape"]["type"] != "plane":
                            lows.append(support_point(p["shape"], R, t, -UP)[1])
                    gap = spec["support_surfaces"][base_obj] - max(lows)
                    ok = raw >= -1e-9 and abs(gap) <= 1e-9
                    support_rows.append((f, sup_obj, base_obj, gap, raw))
                    if not ok:
                        fails.append({"frame": f, "pair": [sup_obj, base_obj], "kind": "support", "gap": gap, "min_sdf": raw})
                elif (na, nb) in mounted:
                    if not (-1e-9 <= raw <= 1e-9):
                        fails.append({"frame": f, "pair": [na, nb], "kind": "mounted", "min_sdf": raw})
                elif cert < c:
                    fails.append({"frame": f, "pair": [na, nb], "kind": "clearance", "certified_distance": cert, "min_sdf": raw})
    contact = supports | mounted | {(b, a) for a, b in supports}
    closest = [(f"{a}|{b}", round(v, 4)) for (a, b), v in sorted(cert_by_pair.items(), key=lambda kv: kv[1])
               if (a, b) not in contact and v < math.inf]
    return {
        "pass": not fails,
        "clearance": c, "base_spacing": s, "max_samples_per_part": V10_MAX_SAMPLES,
        "rule": "certified distance (sampled SDF minus the sampling spacing, best direction; any negative sample fails) >= "
                "clearance; support contacts touch (|gap|<=1e-9) and do not penetrate; mounted contacts touch (|min SDF|<=1e-9)",
        "n_failures": len(fails), "failures": fails[:20],
        "closest_non_contact_pairs": dict(closest[:12]),
        "support_contacts_checked": len(support_rows),
        "support_max_abs_gap": max((abs(r[3]) for r in support_rows), default=0.0),
    }


def check_scene(a):
    spec = json.load(open(os.path.join(a.out, "event_spec.json")))
    cams = load_rig(spec, a.repo_root)
    partial = a.partial
    if partial:
        cams = [c for c in cams if os.path.isfile(os.path.join(a.out, "gt", "stats", f"{c['name']}_partial.json"))]
    stats = load_stats(a.out, cams, partial)
    A, B = spec["clock"]["A"], spec["clock"]["B"]
    cond = spec["condition"]
    tobj = [o for o in spec["objects"] if o["role"] == "target"][0]
    train = [c["name"] for c in cams if c["name"] not in spec["rig"]["held_out"]]
    res = {"scene": spec["scene_id"], "condition": cond, "partial_frames": partial}
    frames = stats["cam00"]["frames"]
    if partial:
        res["partial_scope"] = {"cameras": [c["name"] for c in cams], "frames": frames,
                                "note": "partial render: only V6 and R0 are evaluated"}
    full = not partial
    # ---- V1: target visibility in present frames
    pf = [f for f in frames if present(tobj, f)] if full else []
    cam00 = dict(zip(frames, stats["cam00"]["target_px"]))
    ntrain = {f: sum(1 for n in train if dict(zip(stats[n]["frames"], stats[n]["target_px"]))[f] > 0) for f in pf}
    c00 = [cam00[f] for f in pf]
    v1 = {} if not full else {"present_frames": len(pf), "cam00_min": min(c00), "cam00_max": max(c00),
          "train_cams_min": min(ntrain.values()), "train_cams_total": len(train),
          "range": list(V1_CAM00), "min_train_cams": V1_MIN_TRAIN_CAMS}
    if full:
        v1["pass"] = V1_CAM00[0] <= v1["cam00_min"] and v1["cam00_max"] <= V1_CAM00[1] and v1["train_cams_min"] >= V1_MIN_TRAIN_CAMS
        res["V1"] = v1
    # ---- V2: world absence and zero pixels in every camera over [A, B]
    gap = [f for f in frames if A <= f <= B]
    in_graph = [f for f in range(A, B + 1) if present(tobj, f)]
    px = {n: sum(dict(zip(stats[n]["frames"], stats[n]["target_px"]))[f] for f in gap) for n in stats}
    if full:
        res["V2"] = {"frames_checked": len(gap), "target_in_scene_graph_frames": in_graph,
                     "target_px_over_gap_all_cams": int(sum(px.values())),
                     "pass": not in_graph and sum(px.values()) == 0}
    # ---- V4 (R2): pose change visible
    cam = cams[0]
    if cond == "R2" and full:
        old = alone_mask(spec, cam, "target", A - 1)
        new = alone_mask(spec, cam, "target", B + 1)
        inter, uni = int((old & new).sum()), int((old | new).sum())
        newf = [cam00[f] for f in frames if f > B]
        res["V4"] = {"iou_old_new_cam00": inter / uni, "old_alone_px": int(old.sum()), "new_alone_px": int(new.sum()),
                     "new_frontmost_min": min(newf) if newf else None, "new_frontmost_max": max(newf) if newf else None,
                     "pass": inter == 0 and bool(newf) and V1_CAM00[0] <= min(newf) and max(newf) <= V1_CAM00[1]}
    # ---- V5 (G1): occupant covers the ghost and passes through the former volume
    if cond == "G1" and full:
        ghost = alone_mask(spec, cam, "target", A - 1)
        ids = ids_by_role(spec)
        occ = np.array(ids["occupant"], dtype=np.int16)
        cov = {}
        for f in range(A, B + 1):
            ident = load_identity(a.out, "cam00", f, partial) if f in frames else None
            if ident is None:
                continue
            cov[f] = float((ghost & np.isin(ident, occ)).sum()) / float(ghost.sum())
        # former volume: grid samples inside the target at its pre-gap pose
        Rr, tr = root_pose(tobj["motion"], A - 1)
        pts = []
        for p in tobj["parts"]:
            R, t = part_pose(p, Rr, tr, A - 1)
            lo = hull_corners(p["shape"]).min(axis=0)
            hi = hull_corners(p["shape"]).max(axis=0)
            g = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], 24) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
            g = g[sdf_local(p["shape"], g) < 0]
            pts.append(mat_vec(R, g) + t)
        pts = np.concatenate(pts)
        oobj = [o for o in spec["objects"] if o["role"] == "occupant"][0]
        inside = np.zeros(len(pts), dtype=bool)
        for f4 in np.arange(A * 4, B * 4 + 1):
            fr = f4 / 4.0
            Rr, tr = root_pose(oobj["motion"], fr)
            for p in oobj["parts"]:
                R, t = part_pose(p, Rr, tr, fr)
                inside |= sdf_local(p["shape"], matT_vec(R, pts - t)) < 0
        n50 = sum(1 for v in cov.values() if v >= 0.5)
        res["V5"] = {"ghost_px": int(ghost.sum()), "coverage_by_frame": {str(k): round(v, 4) for k, v in cov.items()},
                     "frames_ge_50pct": n50, "max_coverage": max(cov.values()) if cov else None,
                     "former_volume_fraction_passed": float(inside.mean()), "volume_samples": int(len(pts)),
                     "pass": bool(n50 >= 3 and inside.mean() >= 0.5)}
    # ---- V6: differences from the base world lie inside the event-object support (+1 px)
    if not stats["cam00"]["identity_only"]:
        tot = sum(sum(stats[n]["v6_outside_support_px"]) for n in stats)
        dif = sum(sum(stats[n]["v6_diff_px"]) for n in stats)
        res["V6"] = {"diff_px_total": int(dif), "outside_support_px_total": int(tot), "pass": tot == 0 and dif > 0}
    else:
        res["V6"] = {"status": "not run (identity-only render)"}
    # ---- NEAR: no surface nearer than 1.25 x Ex4DGS's near cull (4.0) in any camera and frame
    if full and all("min_camera_depth" in stats[n] for n in stats):
        zmin = min(min(stats[n]["min_camera_depth"]) for n in stats)
        res["NEAR"] = {"min_camera_depth": zmin, "floor": NEAR_FLOOR, "pass": bool(zmin >= NEAR_FLOOR)}
    # ---- V10
    if full:
        res["V10"] = check_v10(spec)
    # ---- V11: denominators of the primary endpoints
    if not partial:
        counts = json.load(open(os.path.join(a.out, "gt", "cam00_region_counts.json")))
        v11 = {}
        for w in spec["primary_endpoints"]:
            win = spec["windows"][w]
            lo, hi = win["frames"]
            n = sum(counts[win["region"]][lo:hi + 1])
            empty = [f for f in range(lo, hi + 1) if counts[win["region"]][f] == 0]
            v11[w] = {"region": win["region"], "frames": [lo, hi], "pixel_times": n, "empty_frames": len(empty)}
        ok = all(v["pixel_times"] >= V11_MIN_PIXEL_TIMES for v in v11.values())
        ok &= all(v11[w]["empty_frames"] == 0 for w in v11 if spec["windows"][w].get("require_nonempty_every_frame"))
        res["V11"] = {"min_pixel_times": V11_MIN_PIXEL_TIMES, "endpoints": v11, "pass": bool(ok)}
    # ---- R0: optimised render == brute force on sampled (camera, frame) pairs
    if a.r0:
        rng = np.random.default_rng(spec["seed"])
        pairs = []
        for _ in range(a.r0):
            ci = int(rng.integers(len(cams)))
            fi = int(rng.choice(frames))
            pairs.append((ci, fi))
        rows = []
        for ci, fi in pairs:
            c = cams[ci]
            rgb_b, id_b = render_brute(spec, c, fi, identity_only=stats[c["name"]]["identity_only"])
            id_s = load_identity(a.out, c["name"], fi, partial)
            ok = bool((id_b == id_s).all())
            if rgb_b is not None:
                with Image.open(os.path.join(a.out, "frames", c["name"], f"{fi:04d}.png")) as im:
                    ok &= bool((np.asarray(im) == rgb_b).all())
            rows.append({"camera": c["name"], "frame": fi, "identical": ok})
        res["R0"] = {"pairs": rows, "pass": all(r["identical"] for r in rows)}
    app = spec["applicability"] if full else [k for k in ("V6", "R0") if k in spec["applicability"]]
    missing = [k for k in app if k not in res or "pass" not in res[k]]
    res["applicable"] = app
    res["missing_applicable"] = missing
    res["all_pass"] = (not missing) and all(res[k]["pass"] is True for k in app)
    name = "checks_partial.json" if partial else "checks.json"
    with open(os.path.join(a.out, name), "w", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    summ = {k: res[k].get("pass") for k in ["V1", "V2", "V4", "V5", "V6", "NEAR", "V10", "V11", "R0"] if k in res}
    print(f"{spec['scene_id']}: {summ} all_pass={res['all_pass']}")
    return res


def cmd_check(a):
    check_scene(a)


# ==========================================================================
# montage subcommand
# ==========================================================================

REGION_COLOURS = {"ghost": (255, 40, 40), "target": (40, 230, 40), "vacated": (255, 220, 0),
                  "occupant": (0, 220, 255), "ordinary": (230, 0, 230)}


def _outline(img, mask, colour):
    from scipy.ndimage import binary_erosion
    edge = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    img[edge] = colour


def cmd_montage(a):
    spec = json.load(open(os.path.join(a.out, "event_spec.json")))
    frames = [int(x) for x in a.frames.split(",")] if a.frames else list(KEY_FRAMES)
    mroot = os.path.join(a.out, "gt", "masks")
    regs = [r for r in ("ghost", "vacated", "occupant", "target") if os.path.isdir(os.path.join(mroot, r))]
    ghost0 = np.asarray(Image.open(os.path.join(mroot, "ghost", f"{frames[0]:05d}.png"))) > 127
    # crop box: union of ghost and every event-object support in cam00 over the montage frames
    boxm = ghost0.copy()
    for f in frames:
        for r in regs:
            boxm |= np.asarray(Image.open(os.path.join(mroot, r, f"{f:05d}.png"))) > 127
    ys, xs = np.nonzero(boxm)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    hw = max(200, (xs.max() - xs.min()) // 2 + 60)
    hh = max(150, (ys.max() - ys.min()) // 2 + 60)
    x0, x1 = max(0, cx - hw), min(spec["rig"]["raster"][0], cx + hw)
    y0, y1 = max(0, cy - hh), min(spec["rig"]["raster"][1], cy + hh)
    tiles_full, tiles_crop, tiles_raw = [], [], []
    tw = 450
    for f in frames:
        img = np.asarray(Image.open(os.path.join(a.out, "frames", "cam00", f"{f:04d}.png")).convert("RGB")).copy()
        raw = img[y0:y1, x0:x1].copy()
        for r in regs:
            m = np.asarray(Image.open(os.path.join(mroot, r, f"{f:05d}.png"))) > 127
            if m.any():
                _outline(img, m, REGION_COLOURS[r])
        full = Image.fromarray(img).resize((tw, int(tw * img.shape[0] / img.shape[1])), Image.BILINEAR)
        d = ImageDraw.Draw(full)
        d.rectangle([x0 * tw / img.shape[1], y0 * tw / img.shape[1], x1 * tw / img.shape[1], y1 * tw / img.shape[1]], outline=(255, 255, 255))
        d.text((6, 4), f"{spec['scene_id']} cam00 f{f}", fill=(255, 255, 0))
        tiles_full.append(full)
        crop = Image.fromarray(img[y0:y1, x0:x1]).resize((tw, int(tw * (y1 - y0) / (x1 - x0))), Image.NEAREST)
        tiles_crop.append(crop)
        tiles_raw.append(Image.fromarray(raw).resize((tw, int(tw * (y1 - y0) / (x1 - x0))), Image.NEAREST))
    rows = [tiles_full, tiles_raw, tiles_crop]
    Hs = [max(t.size[1] for t in r) for r in rows]
    sheet = Image.new("RGB", (tw * len(frames), sum(Hs) + 18), (20, 20, 20))
    y = 0
    for r, hgt in zip(rows, Hs):
        for i, t in enumerate(r):
            sheet.paste(t, (i * tw, y))
        y += hgt
    ImageDraw.Draw(sheet).text((6, sum(Hs) + 3), "rows: full frame with outlines | crop raw | crop with outlines.  "
                               "red ghost (pre-gap target-alone), green target front-most, yellow vacated, cyan occupant in ghost",
                               fill=(230, 230, 230))
    os.makedirs(os.path.join(a.out, "montage"), exist_ok=True)
    p = os.path.join(a.out, "montage", f"{spec['scene_id']}_cam00_keyframes.png")
    sheet.save(p)
    print(p)


# ==========================================================================
# manifest subcommand
# ==========================================================================


def _git(args, cwd):
    try:
        return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def cmd_manifest(a):
    spec_p = os.path.join(a.out, "event_spec.json")
    spec = json.load(open(spec_p))
    files = {}
    for root, _dirs, fnames in os.walk(a.out):
        for fn in sorted(fnames):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, a.out).replace("\\", "/")
            if rel == "MANIFEST.json":
                continue
            files[rel] = sha256_file(p)
    gen = os.path.abspath(__file__)
    man = {
        "scene_id": spec["scene_id"],
        "spec_sha256": sha256_file(spec_p),
        "spec_canonical_sha256": sha256_bytes(canon(spec).encode()),
        "trajectory_sha256": sha256_bytes(canon(spec["objects"]).encode()),
        "generator": "scripts/build_absence_suite_scene.py",
        "generator_sha256": sha256_file(gen),
        "git_head": _git(["rev-parse", "HEAD"], a.repo_root),
        "generator_tracked_clean": _git(["status", "--porcelain", "--", "scripts/build_absence_suite_scene.py"], a.repo_root) == "",
        "n_files": len(files),
        "files": files,
    }
    with open(os.path.join(a.out, "MANIFEST.json"), "w", newline="\n") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    print(f"{spec['scene_id']}: MANIFEST {len(files)} files, spec {man['spec_sha256'][:12]}, trajectory {man['trajectory_sha256'][:12]}")


# ==========================================================================
# wave-1 specs: layouts L1-L3 x conditions A0, R2, G1
# ==========================================================================

RIGS = {
    "L1": ("cut_roasted_beef", ["cam%02d" % i for i in range(21) if i != 4]),
    "L2": ("flame_steak", ["cam%02d" % i for i in range(21)]),
    "L3": ("sear_steak", ["cam%02d" % i for i in range(21)]),
}


def _noise(seed, c0, c1, scale, contrast=2.5, octaves=5, aniso=None, speckle=None):
    t = {"type": "noise", "seed": seed, "c0": c0, "c1": c1, "scale": scale, "contrast": contrast, "octaves": octaves}
    if aniso:
        t["aniso"] = aniso
    if speckle:
        t["speckle"] = {"seed": seed + 17, "scale": speckle[0], "amount": speckle[1]}
    return t


def _faces(seed, pal, scale):
    keys = ["+x", "-x", "+y", "-y", "+z", "-z"]
    return {"type": "faces", "faces": {k: _noise(seed + i, pal[i][0], pal[i][1], scale, contrast=3.0) for i, k in enumerate(keys)}}


def _align_y(v):
    """Rotation taking local +y to unit vector v (minimal rotation)."""
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)
    y = np.array([0.0, 1.0, 0.0])
    ax = np.cross(y, v)
    s = np.linalg.norm(ax)
    if s < 1e-12:
        return np.eye(3) if v[1] > 0 else np.diag([1.0, -1.0, -1.0])
    return rot_axis(ax, math.atan2(s, float(y @ v)))



def _two_link(S, H, L1, L2, pole):
    """Elbow position of a two-segment arm from shoulder S to hand H, bending toward `pole`."""
    d = float(np.linalg.norm(H - S))
    if not abs(L1 - L2) < d < L1 + L2:
        raise ValueError(f"hand unreachable: |H - S| = {d:.3f}, links {L1}, {L2}")
    u = (H - S) / d
    a = (L1 * L1 - L2 * L2 + d * d) / (2.0 * d)
    w = pole - (pole @ u) * u
    w = w / np.linalg.norm(w)
    return S + a * u + math.sqrt(L1 * L1 - a * a) * w

LAYOUT = {
    "L1": {
        "seed": 1100, "y_table": 7.8, "table_x": (-16.0, 17.0), "table_z": (4.0, 14.0), "wall_z": 18.3,
        "floor_below_table": 10.6, "ceiling_y": -15.0, "side_x": (-19.0, 20.0), "front_z": -8.0,
        "wall_pal": ([0.84, 0.80, 0.72], [0.76, 0.72, 0.64]), "table_pal": ([0.52, 0.34, 0.19], [0.74, 0.54, 0.33]),
        "target": {"kind": "bottle", "r": 0.32, "h": 2.24, "xz": (6.25, 11.6)},
        "r2": {"xz": (4.0, 11.3)},
        "g1": {"kind": "box", "half": (0.65, 0.85, 0.55), "from": (8.2, 11.6), "to": (4.3, 11.6), "frames": (61, 88)},
        "props": [
            {"name": "prop_ball", "shape": {"type": "sphere", "r": 0.6}, "xz": (-2.8, 9.6), "pal": ([0.95, 0.55, 0.12], [0.75, 0.30, 0.05])},
            {"name": "prop_jar", "shape": {"type": "cylinder", "r": 0.5, "h": 1.5}, "xz": (2.3, 9.4), "pal": ([0.25, 0.55, 0.58], [0.55, 0.80, 0.78])},
            {"name": "prop_block", "shape": {"type": "box", "half": [0.9, 0.5, 0.6]}, "xz": (-8.8, 12.6), "yaw": 20.0, "pal": ([0.45, 0.25, 0.55], [0.70, 0.50, 0.75])},
            {"name": "prop_plate", "shape": {"type": "cylinder", "r": 1.1, "h": 0.1}, "xz": (9.8, 9.0), "pal": ([0.92, 0.92, 0.95], [0.35, 0.45, 0.80])},
        ],
        "cook": {"x": 0.4, "z": 16.3, "sway_amp": 0.8, "sway_period": 4.0, "sway_phase": 0.0,
                 "hand_dx": 1.3, "hand_z": 12.5, "hand_above_table": 1.4,
                 "shoulder": (4.0, 2.2, (0.0, 140.0)), "elbow": (6.0, 1.5, (40.0, 220.0)),
                 "shirt": ([0.24, 0.30, 0.42], [0.32, 0.38, 0.50]), "apron": ([0.16, 0.16, 0.18], [0.22, 0.22, 0.24]),
                 "trousers": ([0.20, 0.20, 0.22], [0.26, 0.26, 0.28]), "hair": ([0.10, 0.08, 0.06], [0.18, 0.14, 0.10])},
        "rotator": {"shape": {"type": "box", "half": [0.55, 0.8, 0.45]}, "xz": (-5.8, 10.8), "period": 6.0, "yaw0": 10.0},
        "decor": {"window": {"x": 0.0, "w": 10.0, "h": 8.0, "bottom": 7.5, "blind_frac": 0.55},
                  "poster": {"x": -10.75, "w": 3.5, "h": 4.5, "bottom": 10.0, "pal": ([0.85, 0.40, 0.20], [0.20, 0.40, 0.70])},
                  "shelf": {"x": 11.25, "w": 4.5, "d": 1.1, "height": 9.5,
                            "items": [("cyl", 0.40, 1.1, -1.4), ("box", (0.35, 0.8, 0.40), 0.0), ("sph", 0.35, 1.4)]}},
    },
    "L2": {
        "seed": 2200, "y_table": 7.8, "table_x": (-16.0, 17.0), "table_z": (4.0, 14.0), "wall_z": 18.4,
        "floor_below_table": 10.6, "ceiling_y": -15.0, "side_x": (-19.0, 20.0), "front_z": -8.0,
        "wall_pal": ([0.76, 0.82, 0.78], [0.68, 0.74, 0.70]), "table_pal": ([0.62, 0.44, 0.26], [0.82, 0.66, 0.44]),
        "target": {"kind": "carton", "half": (0.47, 0.62, 0.31), "xz": (5.7, 11.5), "yaw": 25.0},
        "r2": {"xz": (8.0, 12.0)},
        "g1": {"kind": "ball", "r": 0.7, "from": (3.2, 11.5), "to": (8.3, 11.5), "frames": (61, 88)},
        "props": [
            {"name": "prop_ball", "shape": {"type": "sphere", "r": 0.5}, "xz": (-3.5, 9.2), "pal": ([0.30, 0.70, 0.20], [0.60, 0.85, 0.35])},
            {"name": "prop_jar", "shape": {"type": "cylinder", "r": 0.45, "h": 1.8}, "xz": (1.4, 9.8), "pal": ([0.70, 0.25, 0.25], [0.90, 0.60, 0.50])},
            {"name": "prop_block", "shape": {"type": "box", "half": [0.7, 0.6, 0.7]}, "xz": (-9.5, 11.8), "yaw": -15.0, "pal": ([0.85, 0.75, 0.30], [0.60, 0.45, 0.15])},
            {"name": "prop_plate", "shape": {"type": "cylinder", "r": 1.0, "h": 0.1}, "xz": (-1.2, 7.4), "pal": ([0.95, 0.95, 0.90], [0.80, 0.35, 0.35])},
        ],
        "cook": {"x": -0.6, "z": 16.3, "sway_amp": 1.1, "sway_period": 4.6, "sway_phase": 60.0,
                 "hand_dx": 1.4, "hand_z": 12.4, "hand_above_table": 1.4,
                 "shoulder": (4.5, 2.6, (30.0, 200.0)), "elbow": (6.0, 1.7, (80.0, 300.0)),
                 "shirt": ([0.50, 0.24, 0.22], [0.58, 0.30, 0.28]), "apron": ([0.82, 0.80, 0.74], [0.74, 0.72, 0.66]),
                 "trousers": ([0.18, 0.20, 0.30], [0.24, 0.26, 0.36]), "hair": ([0.30, 0.20, 0.10], [0.40, 0.28, 0.16])},
        "rotator": {"shape": {"type": "cylinder", "r": 0.7, "h": 1.2}, "xz": (-5.0, 11.4), "period": -5.0, "yaw0": 0.0},
        "decor": {"window": {"x": -1.0, "w": 9.0, "h": 8.5, "bottom": 7.0, "blind_frac": 0.45},
                  "poster": {"x": 11.0, "w": 3.0, "h": 4.2, "bottom": 10.5, "pal": ([0.20, 0.60, 0.35], [0.90, 0.80, 0.25])},
                  "shelf": {"x": -11.5, "w": 4.0, "d": 1.1, "height": 9.0,
                            "items": [("box", (0.30, 0.7, 0.40), -1.2), ("cyl", 0.35, 1.3, 0.2), ("box", (0.40, 0.45, 0.28), 1.3)]}},
    },
    "L3": {
        "seed": 3300, "y_table": 7.3, "table_x": (-16.0, 17.0), "table_z": (4.0, 14.0), "wall_z": 18.2,
        "floor_below_table": 10.6, "ceiling_y": -15.0, "side_x": (-19.0, 20.0), "front_z": -8.0,
        "wall_pal": ([0.86, 0.78, 0.74], [0.78, 0.70, 0.66]), "table_pal": ([0.40, 0.30, 0.22], [0.60, 0.48, 0.36]),
        "target": {"kind": "mug", "r": 0.47, "h": 1.05, "xz": (4.9, 12.0), "yaw": 30.0},
        "r2": {"xz": (2.7, 11.6)},
        "g1": {"kind": "tin", "r": 0.62, "h": 1.4, "from": (7.4, 12.0), "to": (3.0, 12.0), "frames": (61, 88)},
        "props": [
            {"name": "prop_ball", "shape": {"type": "sphere", "r": 0.55}, "xz": (-2.2, 9.0), "pal": ([0.85, 0.80, 0.20], [0.60, 0.55, 0.10])},
            {"name": "prop_jar", "shape": {"type": "cylinder", "r": 0.55, "h": 1.3}, "xz": (-6.8, 9.8), "pal": ([0.35, 0.35, 0.70], [0.65, 0.65, 0.90])},
            {"name": "prop_block", "shape": {"type": "box", "half": [0.8, 0.45, 0.55]}, "xz": (9.8, 12.8), "yaw": 35.0, "pal": ([0.30, 0.55, 0.35], [0.55, 0.78, 0.55])},
            {"name": "prop_plate", "shape": {"type": "cylinder", "r": 1.05, "h": 0.1}, "xz": (0.2, 8.4), "pal": ([0.93, 0.90, 0.85], [0.30, 0.60, 0.55])},
        ],
        "cook": {"x": 0.6, "z": 16.0, "sway_amp": 0.7, "sway_period": 3.8, "sway_phase": 120.0,
                 "hand_dx": 1.0, "hand_z": 13.2, "hand_above_table": 1.4,
                 "shoulder": (3.5, 2.0, (90.0, 250.0)), "elbow": (6.5, 1.4, (10.0, 170.0)),
                 "shirt": ([0.26, 0.42, 0.32], [0.32, 0.50, 0.38]), "apron": ([0.40, 0.26, 0.16], [0.48, 0.32, 0.20]),
                 "trousers": ([0.30, 0.28, 0.24], [0.36, 0.34, 0.30]), "hair": ([0.12, 0.10, 0.08], [0.20, 0.16, 0.12])},
        "rotator": {"shape": {"type": "box", "half": [0.4, 1.0, 0.4]}, "xz": (-4.4, 12.0), "period": 7.0, "yaw0": 45.0},
        "decor": {"window": {"x": 1.0, "w": 11.0, "h": 7.5, "bottom": 8.0, "blind_frac": 0.6},
                  "poster": {"x": -11.5, "w": 3.2, "h": 4.0, "bottom": 9.5, "pal": ([0.70, 0.20, 0.45], [0.95, 0.85, 0.70])},
                  "shelf": {"x": 11.5, "w": 4.2, "d": 1.1, "height": 8.5,
                            "items": [("sph", 0.38, -1.3), ("cyl", 0.33, 1.2, 0.0), ("box", (0.30, 0.75, 0.30), 1.3)]}},
    },
}


def _yaw_facing(pos, cam_centre):
    """Yaw (deg, rounded to 1e-3) that turns object-local +z toward cam_centre in the horizontal plane.
    r_upright(theta) maps local +z to (sin theta, 0, -cos theta)."""
    d = np.asarray(cam_centre, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    yaw = round(math.degrees(math.atan2(d[0], -d[2])), 3)
    got = mat_vec(r_upright(yaw), np.array([[0.0, 0.0, 1.0]]))[0]
    h = np.array([d[0], 0.0, d[2]]) / math.hypot(d[0], d[2])
    assert float(got @ h) > 0.99999, (got, h)
    return yaw


def make_spec(layout, condition, rig_sha, cam00_centre):
    L = LAYOUT[layout]
    scene, names = RIGS[layout]
    seed = L["seed"]
    yT = L["y_table"]
    A, B = 60, 89
    objs = []

    def static(name, pid, shape, pos, tex, R=None, yaw=None):
        m = {"type": "static", "pos": list(map(float, pos))}
        if R is not None:
            m["R"] = np.asarray(R).tolist()
        else:
            m["yaw_deg"] = float(yaw or 0.0)
        objs.append({"name": name, "role": "static", "parts": [{"id": pid, "shape": shape, "tex": tex}], "motion": m})

    # ---- room: planes with local +y = inward normal; calm, large-scale, low-contrast textures
    x0, x1 = L["side_x"]
    zf, zw = L["front_z"], L["wall_z"]
    yc, yfl = L["ceiling_y"], yT + L["floor_below_table"]
    big = 80.0
    wall = L["wall_pal"]

    def calm(sd, c0, c1, scale=3.0):
        return _noise(sd, c0, c1, scale, contrast=1.3, octaves=3)

    static("room_floor", 1, {"type": "plane", "half": [big, big]}, (0.0, yfl, 0.0),
           calm(seed + 1, [0.36, 0.34, 0.31], [0.30, 0.28, 0.26], 2.0), R=rot_axis([1, 0, 0], math.pi))
    static("room_back_wall", 2, {"type": "plane", "half": [big, big]}, (0.0, 0.0, zw),
           calm(seed + 2, wall[0], wall[1]), R=rot_axis([1, 0, 0], -math.pi / 2))
    static("room_left_wall", 3, {"type": "plane", "half": [big, big]}, (x0, 0.0, 0.0),
           calm(seed + 3, [c * 0.92 for c in wall[0]], [c * 0.92 for c in wall[1]]), R=rot_axis([0, 0, 1], -math.pi / 2))
    static("room_right_wall", 4, {"type": "plane", "half": [big, big]}, (x1, 0.0, 0.0),
           calm(seed + 4, [c * 0.88 for c in wall[0]], [c * 0.88 for c in wall[1]]), R=rot_axis([0, 0, 1], math.pi / 2))
    static("room_ceiling", 5, {"type": "plane", "half": [big, big]}, (0.0, yc, 0.0),
           calm(seed + 5, [0.90, 0.90, 0.88], [0.84, 0.84, 0.82]), R=np.eye(3))
    static("room_front_wall", 6, {"type": "plane", "half": [big, big]}, (0.0, 0.0, zf),
           calm(seed + 6, [0.5, 0.5, 0.5], [0.44, 0.44, 0.44]), R=rot_axis([1, 0, 0], math.pi / 2))
    # ---- table: a slab, top at y_table
    tx0, tx1 = L["table_x"]
    tz0, tz1 = L["table_z"]
    th = 0.25
    static("table", 10, {"type": "box", "half": [(tx1 - tx0) / 2, th, (tz1 - tz0) / 2]},
           ((tx0 + tx1) / 2, yT + th, (tz0 + tz1) / 2),
           _noise(seed + 10, L["table_pal"][0], L["table_pal"][1], 0.5, contrast=3.0, aniso=[0.25, 1.0, 1.0], speckle=(0.05, 0.12)),
           R=np.eye(3))
    # ---- props
    for i, p in enumerate(L["props"]):
        shp = p["shape"]
        tex = _noise(seed + 20 + i, p["pal"][0], p["pal"][1], 0.15, contrast=3.0, speckle=(0.04, 0.12))
        if shp["type"] == "box":
            objs.append({"name": p["name"], "role": "static",
                         "parts": [{"id": 20 + i, "shape": shp, "tex": tex, "local": {"pos": [0.0, shp["half"][1], 0.0]}}],
                         "motion": {"type": "static", "pos": [p["xz"][0], yT, p["xz"][1]], "yaw_deg": p.get("yaw", 0.0)}})
        elif shp["type"] == "sphere":
            objs.append({"name": p["name"], "role": "static",
                         "parts": [{"id": 20 + i, "shape": shp, "tex": tex, "local": {"pos": [0.0, shp["r"], 0.0]}}],
                         "motion": {"type": "static", "pos": [p["xz"][0], yT, p["xz"][1]], "yaw_deg": 0.0}})
        else:
            objs.append({"name": p["name"], "role": "static", "parts": [{"id": 20 + i, "shape": shp, "tex": tex}],
                         "motion": {"type": "static", "pos": [p["xz"][0], yT, p["xz"][1]], "yaw_deg": p.get("yaw", 0.0)}})
    # ---- wall fixtures (static, sparse strong features): a window with a blind, a poster, a shelf with items.
    # Local frames: +x along the wall, +y up (height above the table top), +z out of the wall toward the cameras.
    dec = L["decor"]
    mounted = []
    wn = dec["window"]
    Wd, Hd, b0, fb = wn["w"], wn["h"], wn["bottom"], wn["blind_frac"]
    t, dd = 0.45, 0.175
    white = calm(seed + 60, [0.93, 0.93, 0.91], [0.88, 0.88, 0.86], 1.0)
    glass = _noise(seed + 61, [0.10, 0.12, 0.17], [0.20, 0.23, 0.30], 1.5, contrast=1.5, octaves=4)
    rng = np.random.default_rng(seed + 62)
    hb = Hd * fb
    folds, yy = [], -hb / 2 + 0.5
    while yy < hb / 2 - 0.3:
        folds.append({"y": [round(yy, 4), round(yy + 0.09, 4)], "phi_deg": None,
                      "tex": calm(seed + 63, [0.60, 0.58, 0.53], [0.55, 0.53, 0.48], 1.0)})
        yy += float(rng.uniform(0.7, 1.0))
    blind = {"type": "banded", "base": calm(seed + 64, [0.87, 0.85, 0.79], [0.81, 0.79, 0.73], 1.5), "bands": folds}
    ymid = b0 + Hd / 2
    ylow = b0 + Hd * (1 - fb) / 2
    wparts = [
        {"id": 11, "shape": {"type": "box", "half": [Wd / 2 + t, t / 2, dd]}, "tex": white, "local": {"pos": [0.0, b0 + Hd + t / 2, dd]}},
        {"id": 11, "shape": {"type": "box", "half": [Wd / 2 + t + 0.3, t / 2, 0.25]}, "tex": white, "local": {"pos": [0.0, b0 - t / 2, 0.25]}},
        {"id": 11, "shape": {"type": "box", "half": [t / 2, Hd / 2, dd]}, "tex": white, "local": {"pos": [-Wd / 2 - t / 2, ymid, dd]}},
        {"id": 11, "shape": {"type": "box", "half": [t / 2, Hd / 2, dd]}, "tex": white, "local": {"pos": [Wd / 2 + t / 2, ymid, dd]}},
        {"id": 12, "shape": {"type": "box", "half": [Wd / 2, Hd / 2, 0.01]}, "tex": glass, "local": {"pos": [0.0, ymid, 0.01]}},
        {"id": 13, "shape": {"type": "box", "half": [Wd / 2, hb / 2, 0.04]}, "tex": blind, "local": {"pos": [0.0, b0 + Hd - hb / 2, 0.06]}},
        {"id": 13, "shape": {"type": "box", "half": [Wd / 2, 0.1, 0.08]}, "tex": white, "local": {"pos": [0.0, b0 + Hd - hb, 0.10]}},
        {"id": 11, "shape": {"type": "box", "half": [0.12, Hd * (1 - fb) / 2, 0.03]}, "tex": white, "local": {"pos": [0.0, ylow, 0.05]}},
        {"id": 11, "shape": {"type": "box", "half": [Wd / 2, 0.12, 0.03]}, "tex": white, "local": {"pos": [0.0, ylow, 0.05]}},
    ]
    objs.append({"name": "window", "role": "static", "parts": wparts,
                 "motion": {"type": "static", "pos": [wn["x"], yT, zw], "yaw_deg": 0.0}})
    mounted.append(["window", "room_back_wall"])
    ps = dec["poster"]
    pw, ph, pb = ps["w"], ps["h"], ps["bottom"]
    image = {"type": "banded", "base": _noise(seed + 70, ps["pal"][0], ps["pal"][1], 0.6, contrast=4.0, octaves=4),
             "bands": [{"y": [ph / 2 - 1.0, ph / 2 - 0.45], "phi_deg": None,
                        "tex": calm(seed + 71, [0.95, 0.94, 0.90], [0.90, 0.89, 0.85], 1.0)}]}
    objs.append({"name": "poster", "role": "static", "parts": [
        {"id": 14, "shape": {"type": "box", "half": [pw / 2, ph / 2, 0.02]}, "tex": calm(seed + 72, [0.12, 0.12, 0.13], [0.16, 0.16, 0.17], 1.0),
         "local": {"pos": [0.0, pb + ph / 2, 0.02]}},
        {"id": 15, "shape": {"type": "box", "half": [pw / 2 - 0.2, ph / 2 - 0.2, 0.01]}, "tex": image,
         "local": {"pos": [0.0, pb + ph / 2, 0.05]}}],
        "motion": {"type": "static", "pos": [ps["x"], yT, zw], "yaw_deg": 0.0}})
    mounted.append(["poster", "room_back_wall"])
    sh = dec["shelf"]
    sd = sh["d"]
    objs.append({"name": "shelf", "role": "static", "parts": [
        {"id": 16, "shape": {"type": "box", "half": [sh["w"] / 2, 0.12, sd / 2]},
         "tex": calm(seed + 73, [0.55, 0.40, 0.26], [0.48, 0.34, 0.22], 0.8), "local": {"pos": [0.0, sh["height"], sd / 2]}}],
        "motion": {"type": "static", "pos": [sh["x"], yT, zw], "yaw_deg": 0.0}})
    mounted.append(["shelf", "room_back_wall"])
    shelf_top = yT - (sh["height"] + 0.12)
    shelf_supports = []
    for k, it in enumerate(sh["items"]):
        itex = _noise(seed + 80 + k, [0.80 - 0.2 * k, 0.35 + 0.2 * k, 0.30 + 0.15 * k], [0.95 - 0.2 * k, 0.60 + 0.15 * k, 0.45 + 0.1 * k],
                      0.12, contrast=3.0, octaves=4)
        if it[0] == "cyl":
            shp, xo, loc = {"type": "cylinder", "r": it[1], "h": it[2]}, it[3], [0.0, 0.0, 0.0]
        elif it[0] == "sph":
            shp, xo, loc = {"type": "sphere", "r": it[1]}, it[2], [0.0, it[1], 0.0]
        else:
            shp, xo, loc = {"type": "box", "half": list(it[1])}, it[2], [0.0, it[1][1], 0.0]
        nm = f"shelf_item{k}"
        objs.append({"name": nm, "role": "static", "parts": [{"id": 17 + k, "shape": shp, "tex": itex, "local": {"pos": loc}}],
                     "motion": {"type": "static", "pos": [sh["x"] + xo, shelf_top, zw - sd / 2], "yaw_deg": 15.0 * k}})
        shelf_supports.append([nm, "shelf"])
    # ---- ordinary dynamics: a cook proxy with human proportions (moving body) and a rotating object.
    # Cook local frame: origin on the floor under the body, +y up, +z toward the cameras. One unit is ~8.5 cm.
    ck = L["cook"]
    floor_y = yfl
    T_h = L["floor_below_table"]                        # table-top height above the floor
    hip, sh_top = T_h - 0.4, T_h + 6.4                  # hip just below the table top; shoulder top

    def cloth(pal, sd_):
        return _noise(sd_, pal[0], pal[1], 0.6, contrast=1.8, octaves=4)

    shirt, apron, trousers = cloth(ck["shirt"], seed + 40), cloth(ck["apron"], seed + 41), cloth(ck["trousers"], seed + 42)
    skin = _noise(seed + 43, [0.86, 0.68, 0.55], [0.80, 0.62, 0.50], 0.4, contrast=1.5, octaves=3)
    hair = _noise(seed + 44, ck["hair"][0], ck["hair"][1], 0.2, contrast=2.0, octaves=4)
    Rx = rot_axis([0, 0, 1], -math.pi / 2)              # local +y -> +x: a horizontal cylinder along x
    parts = [
        {"id": 40, "shape": {"type": "box", "half": [1.5, hip / 2, 0.9]}, "tex": trousers, "local": {"pos": [0.0, hip / 2, 0.0]}},
        {"id": 41, "shape": {"type": "box", "half": [2.0, (sh_top - 0.9 - hip) / 2, 1.1]}, "tex": shirt,
         "local": {"pos": [0.0, hip + (sh_top - 0.9 - hip) / 2, 0.0]}},
        {"id": 42, "shape": {"type": "cylinder", "r": 0.9, "h": 4.3}, "tex": shirt, "local": {"pos": [-2.15, sh_top - 0.9, 0.0], "R": Rx.tolist()}},
        {"id": 41, "shape": {"type": "box", "half": [1.55, 3.4, 0.03]}, "tex": apron, "local": {"pos": [0.0, hip + 2.4, 1.13]}},
        {"id": 43, "shape": {"type": "cylinder", "r": 0.5, "h": 1.0}, "tex": skin, "local": {"pos": [0.0, sh_top - 0.2, 0.0]}},
        {"id": 44, "shape": {"type": "sphere", "r": 1.15}, "tex": skin, "local": {"pos": [0.0, sh_top + 1.85, 0.0]}},
        {"id": 44, "shape": {"type": "sphere", "r": 1.2}, "tex": hair, "local": {"pos": [0.0, sh_top + 2.1, -0.3]}},
    ]
    L1_, L2_ = 3.5, 3.3                                  # upper arm, forearm lengths
    hand_h = T_h + ck["hand_above_table"]
    for k, sgn in enumerate((-1.0, 1.0)):
        S = np.array([sgn * 2.15, sh_top - 0.9, 0.1])
        H = np.array([sgn * ck["hand_dx"], hand_h, ck["z"] - ck["hand_z"]])
        E = _two_link(S, H, L1_, L2_, np.array([sgn * 0.4, -1.0, -0.5]))
        elbow_axis = np.cross(E - S, H - E)
        elbow_axis = (elbow_axis / np.linalg.norm(elbow_axis)).tolist()
        sa, sp, sph_ = ck["shoulder"]
        ea, ep, eph = ck["elbow"]
        jsh = {"pivot": S.tolist(), "axis": [1.0, 0.0, 0.0], "amp_deg": sa, "period_s": sp, "phase_deg": sph_[k], "offset_deg": 0.0}
        jel = {"pivot": E.tolist(), "axis": elbow_axis, "amp_deg": ea, "period_s": ep, "phase_deg": eph[k], "offset_deg": 0.0}
        parts += [
            {"id": 45 + k, "shape": {"type": "cylinder", "r": 0.52, "h": L1_}, "tex": shirt,
             "local": {"pos": S.tolist(), "R": _align_y(E - S).tolist()}, "joints": [jsh]},
            {"id": 47 + k, "shape": {"type": "cylinder", "r": 0.42, "h": float(np.linalg.norm(H - E))}, "tex": skin,
             "local": {"pos": E.tolist(), "R": _align_y(H - E).tolist()}, "joints": [jsh, jel]},
            {"id": 49 + k, "shape": {"type": "sphere", "r": 0.45}, "tex": skin, "local": {"pos": H.tolist()}, "joints": [jsh, jel]},
        ]
    objs.append({"name": "cook", "role": "ordinary", "parts": parts,
                 "motion": {"type": "sway", "pos": [ck["x"], floor_y, ck["z"]], "dir": [1.0, 0.0, 0.0], "amp": ck["sway_amp"],
                            "period_s": ck["sway_period"], "phase_deg": ck["sway_phase"], "yaw_deg": 0.0}})
    ro = L["rotator"]
    rs = ro["shape"]
    if rs["type"] == "box":
        rtex = _faces(seed + 50, [([0.85, 0.20, 0.20], [0.95, 0.60, 0.50]), ([0.20, 0.35, 0.85], [0.55, 0.65, 0.95]),
                                  ([0.95, 0.95, 0.90], [0.70, 0.70, 0.65]), ([0.3, 0.3, 0.3], [0.2, 0.2, 0.2]),
                                  ([0.95, 0.80, 0.15], [0.75, 0.55, 0.10]), ([0.20, 0.70, 0.30], [0.55, 0.85, 0.50])], 0.1)
        rpart = {"id": 60, "shape": rs, "tex": rtex, "local": {"pos": [0.0, rs["half"][1], 0.0]}}
    else:
        pal = [([0.85, 0.2, 0.2], [0.95, 0.5, 0.45]), ([0.95, 0.85, 0.2], [0.8, 0.65, 0.1]), ([0.2, 0.6, 0.25], [0.5, 0.8, 0.5]),
               ([0.2, 0.4, 0.85], [0.5, 0.65, 0.95]), ([0.6, 0.25, 0.7], [0.8, 0.55, 0.85]), ([0.95, 0.95, 0.95], [0.7, 0.7, 0.7])]
        rtex = {"type": "sectors", "texs": [_noise(seed + 50 + i, a_, b_, 0.1, contrast=3.0) for i, (a_, b_) in enumerate(pal)],
                "cap": _noise(seed + 59, [0.9, 0.9, 0.9], [0.5, 0.5, 0.5], 0.1, contrast=3.0)}
        rpart = {"id": 60, "shape": rs, "tex": rtex}
    objs.append({"name": "rotator", "role": "ordinary", "parts": [rpart],
                 "motion": {"type": "spin", "pos": [ro["xz"][0], yT, ro["xz"][1]], "yaw0_deg": ro["yaw0"], "period_s": ro["period"]}})
    # ---- target
    tg = L["target"]
    tpos = [tg["xz"][0], yT, tg["xz"][1]]
    if tg["kind"] == "bottle":
        yaw = _yaw_facing(tpos, cam00_centre)
        body = _noise(seed + 100, [0.10, 0.28, 0.14], [0.22, 0.46, 0.26], 0.12, contrast=2.5, speckle=(0.03, 0.10))
        label = _noise(seed + 101, [0.93, 0.89, 0.78], [0.70, 0.16, 0.12], 0.06, contrast=6.0)
        tex = {"type": "banded", "base": body,
               "bands": [{"y": [0.25 * tg["h"], 0.72 * tg["h"]], "phi_deg": [-75.0, 75.0], "tex": label},
                         {"y": [0.93 * tg["h"], tg["h"]], "phi_deg": None, "tex": _noise(seed + 102, [0.85, 0.75, 0.25], [0.6, 0.5, 0.1], 0.05)}]}
        tparts = [{"id": 100, "shape": {"type": "cylinder", "r": tg["r"], "h": tg["h"]}, "tex": tex}]
    elif tg["kind"] == "carton":
        yaw = tg["yaw"]
        h = tg["half"]
        pal = [([0.85, 0.30, 0.25], [0.95, 0.65, 0.55]), ([0.25, 0.40, 0.80], [0.60, 0.70, 0.95]), ([0.95, 0.95, 0.92], [0.75, 0.75, 0.70]),
               ([0.4, 0.4, 0.4], [0.3, 0.3, 0.3]), ([0.95, 0.80, 0.20], [0.80, 0.55, 0.10]), ([0.25, 0.65, 0.35], [0.60, 0.85, 0.55])]
        tparts = [{"id": 100, "shape": {"type": "box", "half": list(h)}, "tex": _faces(seed + 100, pal, 0.07), "local": {"pos": [0.0, h[1], 0.0]}}]
    else:  # mug
        yaw = tg["yaw"]
        mtex = {"type": "banded", "base": _noise(seed + 100, [0.90, 0.90, 0.94], [0.30, 0.42, 0.80], 0.07, contrast=4.0),
                "bands": [{"y": [0.62 * tg["h"], 0.80 * tg["h"]], "phi_deg": None, "tex": _noise(seed + 101, [0.85, 0.20, 0.20], [0.60, 0.10, 0.10], 0.06)}]}
        tparts = [{"id": 100, "shape": {"type": "cylinder", "r": tg["r"], "h": tg["h"]}, "tex": mtex},
                  {"id": 101, "shape": {"type": "box", "half": [0.17, 0.30, 0.07]}, "tex": _noise(seed + 102, [0.90, 0.90, 0.94], [0.30, 0.42, 0.80], 0.07, contrast=4.0),
                   "local": {"pos": [tg["r"] + 0.12, 0.58 * tg["h"], 0.0]}}]
    if condition == "R2":
        keys = [{"frames": [0, A - 1], "pos": tpos, "yaw_deg": yaw},
                {"frames": [B + 1, N_FRAMES - 1], "pos": [L["r2"]["xz"][0], yT, L["r2"]["xz"][1]], "yaw_deg": yaw}]
    else:
        keys = [{"frames": [0, A - 1], "pos": tpos, "yaw_deg": yaw}, {"frames": [B + 1, N_FRAMES - 1], "pos": tpos, "yaw_deg": yaw}]
    objs.append({"name": "target", "role": "target", "presence": [[0, A - 1], [B + 1, N_FRAMES - 1]],
                 "parts": tparts, "motion": {"type": "keyed", "keys": keys}})
    supports = ([["target", "table"]] + [[p["name"], "table"] for p in L["props"]] + [["rotator", "table"], ["cook", "room_floor"]]
                + shelf_supports)
    # ---- G1 occupant
    if condition == "G1":
        g = L["g1"]
        p0 = [g["from"][0], yT, g["from"][1]]
        p1 = [g["to"][0], yT, g["to"][1]]
        if g["kind"] == "box":
            opart = {"id": 110, "shape": {"type": "box", "half": list(g["half"])},
                     "tex": _faces(seed + 110, [([0.95, 0.55, 0.10], [0.75, 0.35, 0.05])] * 2 + [([0.95, 0.90, 0.60], [0.70, 0.60, 0.30])] * 2
                                   + [([0.20, 0.55, 0.85], [0.10, 0.30, 0.60])] * 2, 0.06),
                     "local": {"pos": [0.0, g["half"][1], 0.0]}}
            om = {"type": "slide", "frames": list(g["frames"]), "from": p0, "to": p1, "yaw_deg": 0.0}
        elif g["kind"] == "ball":
            opart = {"id": 110, "shape": {"type": "sphere", "r": g["r"]},
                     "tex": _noise(seed + 110, [0.95, 0.45, 0.10], [0.20, 0.20, 0.85], 0.12, contrast=5.0)}
            p0[1] -= g["r"]
            p1[1] -= g["r"]
            om = {"type": "roll", "frames": list(g["frames"]), "from": p0, "to": p1, "radius": g["r"], "yaw_deg": 0.0}
        else:  # tin
            opart = {"id": 110, "shape": {"type": "cylinder", "r": g["r"], "h": g["h"]},
                     "tex": {"type": "banded", "base": _noise(seed + 110, [0.75, 0.78, 0.82], [0.45, 0.48, 0.52], 0.08, contrast=3.0),
                             "bands": [{"y": [0.2 * g["h"], 0.8 * g["h"]], "phi_deg": None,
                                        "tex": _noise(seed + 111, [0.95, 0.85, 0.20], [0.20, 0.50, 0.25], 0.07, contrast=5.0)}]}}
            om = {"type": "slide", "frames": list(g["frames"]), "from": p0, "to": p1, "yaw_deg": 0.0}
        objs.append({"name": "occupant", "role": "occupant", "parts": [opart], "motion": om})
        supports.append(["occupant", "table"])
    wins = {
        "P1": {"region": "ghost", "frames": [A + 3, B - 2], "require_nonempty_every_frame": True},
        "T": {"region": "target", "frames": [B + 1, B + 2]},
        "P2": {"region": "target", "frames": [B + 3, B + 10], "require_nonempty_every_frame": True},
        "S1": {"region": "target", "frames": [B + 11, B + 20]},
        "H1": {"region": "target", "frames": [A - 30, A - 3]},
        "C1": {"region": "target", "frames": [230, 259]},
        "H2": {"region": "whole", "frames": [0, N_FRAMES - 1]},
        "ORD": {"region": "ordinary", "frames": [0, N_FRAMES - 1]},
        "STAT": {"region": "static_complement", "frames": [0, N_FRAMES - 1]},
    }
    prim = ["P1", "P2"]
    app = ["V1", "V2", "V6", "V10", "V11", "NEAR", "R0"]
    if condition == "R2":
        wins["V"] = {"region": "vacated", "frames": [B + 1, B + 20], "require_nonempty_every_frame": True}
        prim.append("V")
        app.insert(2, "V4")
    if condition == "G1":
        wins["OCC"] = {"region": "occupant", "frames": [A + 3, B - 2]}
        prim.append("OCC")
        app.insert(2, "V5")
    spec = {
        "ars_spec_version": SPEC_VERSION,
        "scene_id": f"ARS-{layout}-{condition}",
        "layout": layout, "condition": condition, "seed": seed,
        "plan": "research-wiki/operations/ars-fast-path-plan-2026-09-23.md (rev 3); spec: absence-return-suite-and-baseline-lanes-plan-2026-09-23.md (rev 2) s3-s6",
        "rig": {"poses_bounds": f"data/{scene}/poses_bounds.npy", "sha256": rig_sha, "source_scene": scene,
                "camera_names": names, "held_out": ["cam00"], "raster": list(RASTER),
                "camera_model": "pinhole, principal point (W/2, H/2), pixel centre at +0.5, f_raster = f_native * W / w_native"},
        "time": {"n_frames": N_FRAMES, "fps": FPS},
        "render": {"supersample": "2x2, offsets 0.25/0.75, box filter", "identity": "front-most part id of the pixel-centre ray",
                   "shading": "lambert + ambient, one directional light, no cast shadows", "light_dir": [-0.35, -1.0, -0.55],
                   "ambient": 0.35, "numerics": "float64; 8-bit = floor(clip(x,0,1)*255+0.5)", "world_up": UP.tolist()},
        "clock": {"A": A, "B": B, "kind": "instant", "note": "designed reference clock: target removed from the world over [A, B]"},
        "windows": wins,
        "primary_endpoints": prim,
        "applicability": app,
        "clearance": 0.05, "v10_spacing": 0.02,
        "support_contacts": supports,
        "mounted_contacts": mounted,
        "support_surfaces": {"table": yT, "room_floor": yfl, "shelf": shelf_top},
        "objects": objs,
    }
    return spec


def cmd_specs(a):
    os.makedirs(a.out_dir, exist_ok=True)
    for layout in a.layouts.split(","):
        scene = RIGS[layout][0]
        pbp = os.path.join(a.repo_root, "data", scene, "poses_bounds.npy")
        rig_sha = sha256_file(pbp)
        pb = np.load(pbp)
        cam00_c = pb[0, :15].reshape(3, 5)[:, 3]
        for cond in a.conditions.split(","):
            spec = make_spec(layout, cond, rig_sha, cam00_c)
            p = os.path.join(a.out_dir, f"{spec['scene_id']}.json")
            with open(p, "w", newline="\n") as fh:
                json.dump(spec, fh, indent=1, sort_keys=True)
                fh.write("\n")
            print(f"{p}: sha256 {sha256_file(p)[:16]} trajectory {sha256_bytes(canon(spec['objects']).encode())[:16]}")


# ==========================================================================
# CLI
# ==========================================================================


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("specs")
    s.add_argument("--out_dir", default="configs/ars/wave1")
    s.add_argument("--layouts", default="L1,L2,L3")
    s.add_argument("--conditions", default="A0,R2,G1")
    s.add_argument("--repo_root", default=".")
    s.set_defaults(fn=cmd_specs)
    r = sub.add_parser("render")
    r.add_argument("--spec", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--cams", default=None, help="comma-separated camera names (default all)")
    r.add_argument("--frames", default=None, help="comma-separated frames (default all 300)")
    r.add_argument("--identity_only", action="store_true")
    r.add_argument("--no_v6", action="store_true")
    r.add_argument("--workers", type=int, default=20)
    r.add_argument("--repo_root", default=".")
    r.set_defaults(fn=cmd_render)
    g = sub.add_parser("gt")
    g.add_argument("--out", required=True)
    g.add_argument("--repo_root", default=".")
    g.set_defaults(fn=cmd_gt)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    c.add_argument("--partial", action="store_true", help="read *_partial stats/identity (a frame subset)")
    c.add_argument("--r0", type=int, default=6, help="number of sampled (camera, frame) pairs for R0")
    c.add_argument("--repo_root", default=".")
    c.set_defaults(fn=cmd_check)
    m = sub.add_parser("montage")
    m.add_argument("--out", required=True)
    m.add_argument("--frames", default=None)
    m.set_defaults(fn=cmd_montage)
    mf = sub.add_parser("manifest")
    mf.add_argument("--out", required=True)
    mf.add_argument("--repo_root", default=".")
    mf.set_defaults(fn=cmd_manifest)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
