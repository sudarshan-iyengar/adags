#!/usr/bin/env python3
"""Background-plate absence-fixture candidate census (CPU only).

N3V cameras are static.  If an object occupies an image region during a
30-frame window ``W`` but is elsewhere at other times, the true background
behind it during ``W`` is available as REAL pixels from the same camera at
those other frames (a "plate").  This script scouts, per scene, which DEVA
pseudo-label ids are stable enough over some window and have a usable plate
on every camera.

It reads only SA4D's DEVA pseudo labels

    <deva_root>/<scene>/camXX/pseudo_label/object_mask/FFFF.png   (uint8 id map)
    <deva_root>/<scene>/camXX/images/FFFF.png                     (RGB frame)

plus the LLFF ``poses_bounds.npy`` for the cross-camera id harmonisation.
No GPU, no torch, no checkpoints.

Scoring definitions are documented in ``census.json['definitions']`` and in
the module docstring of each stage below.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------- constants

N_FRAMES = 300
DEVA_BACKGROUND_ID = 0          # DEVA writes 0 for "no object here"
MIN_TRACK_AREA = 500            # px; below this an id is "absent" in a frame
MIN_CAND_AREA = 2000            # px; the task's size floor
WINDOW_LEN = 30
WINDOW_STARTS = list(range(30, 251, 10))
MARGIN_BEFORE = 20              # M = [a - 20, a + 29 + 20]
MARGIN_AFTER = 20
FOOTPRINT_DILATE = 10           # px, task-specified
DYN_TAU = 12                    # grey levels; |frame - temporal median| > TAU
DYN_BLUR = 5                    # median blur on the abs-diff before threshold
PLATE_USABLE = 0.02             # coverage floor for "usable plate"
SELF_OCCUPANCY_MAX = 0.35       # median-contamination guard (see stage 3)
COMP_MIN_AREA = 50              # px; ignore specks when counting components
GRAN_LO, GRAN_HI = 0.30, 3.50   # accepted harmonised/reference area ratio
CV_SCALE = 0.35                 # area CV that scores 0
DRIFT_SCALE = 40.0              # px of bbox-centre drift that scores 0
F_SAMPLES = 7                   # frames of W sampled to build the footprint
DEPTH_SAMPLES = 64              # epipolar depth sweep resolution
HARMONISE_VOTE_MIN = 0.50       # per-camera vote fraction to accept an id

# Known cam15 -> cam00 correspondences from the 2026-09-10 absence-fixture
# lane (wine bottle), used to validate the harmoniser.  Nothing downstream
# depends on them; they are a self-test only.
KNOWN_PAIRS = {
    "cut_roasted_beef": (117, 136),
    "flame_steak": (79, 65),
    "sear_steak": (114, 67),
}


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------------- stage 1: per camera

def camera_paths(deva_root, scene, cam):
    base = os.path.join(deva_root, scene, cam)
    return (os.path.join(base, "pseudo_label", "object_mask"),
            os.path.join(base, "images"))


def _read_mask(path):
    arr = np.array(Image.open(path))
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ValueError("unexpected id map %s: ndim=%d dtype=%s"
                         % (path, arr.ndim, arr.dtype))
    return arr


def camera_pass(job):
    """One camera: per-id per-frame area and bbox, plus the dynamic proxy.

    The dynamic proxy is ``|grey(t) - median_t grey| > DYN_TAU`` after a
    ``DYN_BLUR`` median filter, where the temporal median is taken over every
    third frame of the whole take.  It is the occupancy test that needs no id,
    so it is the one criterion available on every camera.
    """
    deva_root, scene, cam, keep_masks = job
    mdir, idir = camera_paths(deva_root, scene, cam)

    areas = np.zeros((256, N_FRAMES), dtype=np.int32)
    bboxes = np.full((256, N_FRAMES, 4), -1, dtype=np.int32)  # y0 y1 x0 x1
    masks = [] if keep_masks else None

    grey = None
    for f in range(N_FRAMES):
        a = _read_mask(os.path.join(mdir, "%04d.png" % f))
        if keep_masks:
            masks.append(a)
        counts = np.bincount(a.ravel(), minlength=256)
        areas[:, f] = counts[:256]
        slices = ndimage.find_objects(a)
        for lab, sl in enumerate(slices, start=1):
            if sl is None or counts[lab] < MIN_TRACK_AREA:
                continue
            bboxes[lab, f] = (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)

        g = np.asarray(Image.open(os.path.join(idir, "%04d.png" % f)).convert("L"))
        if grey is None:
            grey = np.empty((N_FRAMES,) + g.shape, dtype=np.uint8)
        grey[f] = g

    areas[DEVA_BACKGROUND_ID] = 0
    median = np.median(grey[::3], axis=0).astype(np.uint8)

    h, w = median.shape
    dyn_packed = np.empty((N_FRAMES, (h * w + 7) // 8), dtype=np.uint8)
    for f in range(N_FRAMES):
        diff = cv2.absdiff(grey[f], median)
        diff = cv2.medianBlur(diff, DYN_BLUR)
        dyn_packed[f] = np.packbits((diff > DYN_TAU).ravel())
    del grey

    ids_present = sorted(int(i) for i in np.where(areas.max(axis=1) >= MIN_TRACK_AREA)[0])
    return {
        "cam": cam, "shape": (int(h), int(w)),
        "areas": areas, "bboxes": bboxes, "dyn": dyn_packed,
        "ids": ids_present, "masks": masks,
    }


def unpack_rows(packed_frame, width, y0, y1):
    """Unpack only image rows [y0, y1) of one packed dynamic mask."""
    b0 = (y0 * width) // 8
    bit0 = y0 * width - b0 * 8
    nbits = (y1 - y0) * width
    seg = np.unpackbits(packed_frame[b0: b0 + (bit0 + nbits + 7) // 8])
    return seg[bit0: bit0 + nbits].reshape(y1 - y0, width)


# --------------------------------------------------- stage 2: window scoring

def window_frames(a):
    w = list(range(a, a + WINDOW_LEN))
    m = list(range(max(0, a - MARGIN_BEFORE),
                   min(N_FRAMES, a + WINDOW_LEN + MARGIN_AFTER)))
    return w, m


def stability_row(areas, bboxes, masks, ident, a, shape):
    """Stability of one id over one window.

    presence  p = frames of the margin window M with area >= MIN_TRACK_AREA,
                  divided by |M|
    area CV   c = std/mean of the id's area over the frames of W where present
    drift     d = max over W of the distance (px) from the bbox centre to the
                  median bbox centre of W
    comps     k = max over F_SAMPLES sampled frames of W of the number of
                  connected components of the id's mask with area >= 50 px
    score       = p^2 * (0.40*clip(1-c/0.35) + 0.40*clip(1-d/40) + 0.20/k)
    """
    w, m = window_frames(a)
    ar_m = areas[ident, m]
    p = float((ar_m >= MIN_TRACK_AREA).mean())

    ar_w = areas[ident, w].astype(np.float64)
    present = ar_w >= MIN_TRACK_AREA
    if present.sum() < 2:
        return None
    vals = ar_w[present]
    c = float(vals.std() / max(vals.mean(), 1.0))

    bb = bboxes[ident, w][present]
    cy = (bb[:, 0] + bb[:, 1]) / 2.0
    cx = (bb[:, 2] + bb[:, 3]) / 2.0
    med = np.array([np.median(cy), np.median(cx)])
    d = float(np.max(np.hypot(cy - med[0], cx - med[1])))

    samp = [w[int(round(i * (WINDOW_LEN - 1) / (F_SAMPLES - 1)))] for i in range(F_SAMPLES)]
    k = 1
    for f in samp:
        if areas[ident, f] < MIN_TRACK_AREA:
            continue
        mm = (masks[f] == ident).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mm, 8)
        k = max(k, int((stats[1:, cv2.CC_STAT_AREA] >= COMP_MIN_AREA).sum()) or 1)

    s = (p ** 2) * (0.40 * float(np.clip(1.0 - c / CV_SCALE, 0.0, 1.0))
                    + 0.40 * float(np.clip(1.0 - d / DRIFT_SCALE, 0.0, 1.0))
                    + 0.20 * (1.0 / k))
    return {"presence_frac": round(p, 4), "area_cv": round(c, 4),
            "centre_drift_px": round(d, 2), "max_components": k,
            "median_area": int(np.median(vals)), "stability": round(float(s), 4)}


def footprint_from_masks(masks, areas, ident, a, shape):
    """F = dilate(union over W of the id's mask, FOOTPRINT_DILATE px).

    The union is taken over F_SAMPLES frames evenly spaced across W rather
    than all 30 (read budget); the 10 px dilation absorbs the gaps.
    """
    w, _ = window_frames(a)
    samp = [w[int(round(i * (WINDOW_LEN - 1) / (F_SAMPLES - 1)))] for i in range(F_SAMPLES)]
    acc = np.zeros(shape, dtype=bool)
    for f in samp:
        if areas is not None and areas[ident, f] < MIN_TRACK_AREA:
            continue
        acc |= (masks[f] == ident)
    if not acc.any():
        return None
    k = 2 * FOOTPRINT_DILATE + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(acc.astype(np.uint8), ker).astype(bool)


def plate_scan(F, dyn_packed, shape, a, bboxes=None, ident=None):
    """Best (lowest) dynamic coverage of F over the frames outside M.

    coverage(t) = |F & dynamic(t)| / |F|, i.e. the fraction of the footprint
    that differs from the camera's temporal median at frame t.  Because the
    median is the background wherever the object occupies the pixel for less
    than half the take, coverage(t) is exactly "how much of F is NOT showing
    background at t" -- object, hand, or anything else.

    ``self_occupancy`` guards the one way that reading can lie: if the id sits
    inside F for most of the take the median IS the object, so the object
    would read as non-dynamic.  It is the fraction of all frames whose id bbox
    covers >= 30% of F's bbox.
    """
    area = float(F.sum())
    _, m = window_frames(a)
    m = set(m)
    ys, xs = np.where(F)
    fy0, fy1, fx0, fx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    Fc = F[fy0:fy1, fx0:fx1]
    h, w = shape

    best = (2.0, -1)
    cov_all = {}
    for t in range(N_FRAMES):
        if t in m:
            continue
        d = unpack_rows(dyn_packed[t], w, fy0, fy1)[:, fx0:fx1]
        cov = float(np.count_nonzero(d & Fc) / area)
        cov_all[t] = cov
        if cov < best[0]:
            best = (cov, t)

    occ = None
    if bboxes is not None and ident is not None:
        fb_area = float((fy1 - fy0) * (fx1 - fx0))
        hits = 0
        for t in range(N_FRAMES):
            y0, y1, x0, x1 = bboxes[ident, t]
            if y0 < 0:
                continue
            iy = max(0, min(y1, fy1) - max(y0, fy0))
            ix = max(0, min(x1, fx1) - max(x0, fx0))
            if iy * ix / fb_area >= 0.30:
                hits += 1
        occ = hits / float(N_FRAMES)

    return {"plate_coverage": round(best[0], 5), "plate_frame": int(best[1]),
            "footprint_px": int(area), "self_occupancy": occ,
            "coverage_sorted": sorted(cov_all.values())[:5]}


# --------------------------------------- stage 3: cross-camera harmonisation

def load_cameras_llff(poses_bounds, cams, img_shape):
    """LLFF poses_bounds.npy -> per-camera (R_c2w, t, f, cx, cy, near, far).

    Rows are the usual 3x5 [down, right, backwards | hwf] blocks; they are
    rotated to the [right, up, backwards] (OpenGL) convention exactly as the
    reference LLFF loader does.  hwf is quoted at the full 2028x2704 raster,
    so the focal length is rescaled to the mask raster.
    """
    arr = np.load(poses_bounds)
    if arr.shape[0] != len(cams):
        raise ValueError("poses_bounds has %d rows, %d cameras present"
                         % (arr.shape[0], len(cams)))
    poses = arr[:, :15].reshape(-1, 3, 5)
    bds = arr[:, 15:17]
    poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], axis=2)
    out = {}
    H, W = img_shape
    for i, cam in enumerate(cams):
        R = poses[i, :, :3]
        t = poses[i, :, 3]
        hh, ww, ff = poses[i, :, 4]
        s = W / float(ww)
        out[cam] = {"R": R, "t": t, "f": ff * s, "cx": W / 2.0, "cy": H / 2.0,
                    "near": float(bds[i, 0]), "far": float(bds[i, 1])}
    return out


def project(cam, X, shape):
    """World points -> pixel coordinates in the OpenGL c2w convention."""
    H, W = shape
    xc = (X - cam["t"]) @ cam["R"]            # R^T (X - t)
    z = -xc[:, 2]
    u = cam["cx"] + cam["f"] * xc[:, 0] / np.where(z == 0, 1e-9, z)
    v = cam["cy"] - cam["f"] * xc[:, 1] / np.where(z == 0, 1e-9, z)
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, ok


def backproject(cam, uv, depth):
    d = np.stack([(uv[:, 0] - cam["cx"]) / cam["f"],
                  -(uv[:, 1] - cam["cy"]) / cam["f"],
                  -np.ones(len(uv))], axis=1)
    return cam["t"][None, :] + (d * depth) @ cam["R"].T


def harmonise(ref_cam, ref_pix, cams_geom, id_maps, shape, near, far):
    """Plane-sweep the reference mask's pixels into every other camera.

    For each of DEPTH_SAMPLES depths the reference pixels are back-projected
    and re-projected into every camera; the id under the projected points is
    voted.  The depth that maximises the summed best-vote over cameras is
    chosen globally (one compact object lies at one depth), and each camera
    keeps its winning id together with the vote fraction.  A camera with a
    vote fraction below HARMONISE_VOTE_MIN is reported unharmonised.
    """
    depths = np.linspace(near, far, DEPTH_SAMPLES)
    H, W = shape
    best = None
    for d in depths:
        X = backproject(cams_geom[ref_cam], ref_pix, d)
        total, per_cam = 0.0, {}
        for cam, geom in cams_geom.items():
            if cam == ref_cam or cam not in id_maps:
                continue
            u, v, ok = project(geom, X, shape)
            if ok.sum() < 0.5 * len(X):
                per_cam[cam] = (None, 0.0)
                continue
            ids = id_maps[cam][v[ok].astype(np.int32), u[ok].astype(np.int32)]
            ids = ids[ids != DEVA_BACKGROUND_ID]
            if ids.size == 0:
                per_cam[cam] = (None, 0.0)
                continue
            vals, counts = np.unique(ids, return_counts=True)
            j = int(np.argmax(counts))
            frac = float(counts[j] / ok.sum())
            per_cam[cam] = (int(vals[j]), frac)
            total += frac
        if best is None or total > best[0]:
            best = (total, float(d), per_cam)
    return {"depth": best[1], "score": best[0],
            "per_camera": {c: {"id": i, "vote": round(f, 3)}
                           for c, (i, f) in best[2].items()}}


# ----------------------------------------------------------------- montage

def draw_tile(img, mask, colour, label, tile_w):
    out = img.copy()
    if mask is not None and mask.any():
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, colour, 3)
    scale = tile_w / float(out.shape[1])
    out = cv2.resize(out, (tile_w, int(round(out.shape[0] * scale))))
    cv2.rectangle(out, (0, 0), (tile_w, 18), (0, 0, 0), -1)
    cv2.putText(out, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_montage(deva_root, scene, rows, cam_data, out_path, tile_w=330):
    """One block per candidate: cam00 and cam15 at a, a+15, a+29 (id outlined)
    and at the plate frame t* (footprint outlined)."""
    blocks = []
    for r in rows:
        a = r["window"][0]
        ident15 = r["id_cam15"]
        ident00 = r.get("id_cam00")
        band = []
        for cam, ident in (("cam15", ident15), ("cam00", ident00)):
            if cam not in cam_data:
                continue
            d = cam_data[cam]
            idir = camera_paths(deva_root, scene, cam)[1]
            tstar = r["per_camera"].get(cam, {}).get("plate_frame", -1)
            F = r["footprints"].get(cam)
            tiles = []
            for f, kind in [(a, "id"), (a + 15, "id"), (a + 29, "id"), (tstar, "F")]:
                if f < 0 or f >= N_FRAMES:
                    tiles.append(np.zeros((int(tile_w * 0.75), tile_w, 3), np.uint8))
                    continue
                img = cv2.cvtColor(np.asarray(Image.open(
                    os.path.join(idir, "%04d.png" % f))), cv2.COLOR_RGB2BGR)
                if kind == "id" and ident is not None and d["masks"] is not None:
                    m = (d["masks"][f] == ident)
                    col, lab = (0, 255, 0), "%s f%03d id%s" % (cam, f, ident)
                else:
                    m = F
                    col = (0, 160, 255)
                    lab = "%s PLATE f%03d cov=%.4f" % (
                        cam, f, r["per_camera"].get(cam, {}).get("plate_coverage", -1))
                tiles.append(draw_tile(img, m, col, lab, tile_w))
            h = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
            band.append(np.hstack(tiles))
        if not band:
            continue
        blk = np.vstack(band)
        hdr = np.zeros((26, blk.shape[1], 3), np.uint8)
        cv2.putText(hdr, "#%d %s id15=%s id00=%s W=[%d,%d] stab=%.3f occ15=%.2f "
                         "pav(vis)=%.2f pav(matched)=%.2f/%dcams worstcov=%s "
                         "area15=%d vis=%.2f"
                    % (r["rank"], scene, ident15, ident00, r["window"][0], r["window"][1],
                       r["stability"], r["cam15_self_occupancy"] or 0.0,
                       r["plate_availability_visible"], r["plate_availability_matched"],
                       r["cameras_granularity_ok"], r["worst_coverage_visible"],
                       r["median_area_cam15"], r["visibility"]),
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        blocks.append(np.vstack([hdr, blk]))
    if not blocks:
        raise RuntimeError("no montage rows")
    w = max(b.shape[1] for b in blocks)
    blocks = [cv2.copyMakeBorder(b, 0, 8, 0, w - b.shape[1], cv2.BORDER_CONSTANT,
                                 value=(40, 40, 40)) for b in blocks]
    cv2.imwrite(out_path, np.vstack(blocks), [int(cv2.IMWRITE_JPEG_QUALITY), 88])


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--deva_root", required=True)
    ap.add_argument("--poses_bounds", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_cam", default="cam15")
    ap.add_argument("--held_cam", default="cam00")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cross_top", type=int, default=24,
                    help="candidates carried to the 21-camera plate test")
    ap.add_argument("--montage_top", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scene_dir = os.path.join(args.deva_root, args.scene)
    cams = sorted(d for d in os.listdir(scene_dir)
                  if d.startswith("cam") and os.path.isdir(
                      os.path.join(scene_dir, d, "pseudo_label", "object_mask")))
    log("scene=%s cameras=%d %s" % (args.scene, len(cams), ",".join(cams)))
    if args.ref_cam not in cams:
        raise SystemExit("reference camera %s absent" % args.ref_cam)

    t0 = time.time()
    jobs = [(args.deva_root, args.scene, c, c in (args.ref_cam, args.held_cam))
            for c in cams]
    with Pool(args.workers) as pool:
        results = pool.map(camera_pass, jobs)
    cam_data = {r["cam"]: r for r in results}
    shape = cam_data[args.ref_cam]["shape"]
    log("stage 1 done in %.1f s; raster %s" % (time.time() - t0, shape))

    # ---- stage 2: windows on the reference camera
    ref = cam_data[args.ref_cam]
    rows = []
    for ident in ref["ids"]:
        for a in WINDOW_STARTS:
            w, _ = window_frames(a)
            if np.median(ref["areas"][ident, w]) < MIN_CAND_AREA:
                continue
            st = stability_row(ref["areas"], ref["bboxes"], ref["masks"], ident, a, shape)
            if st is None or st["presence_frac"] < 0.80:
                continue
            F = footprint_from_masks(ref["masks"], ref["areas"], ident, a, shape)
            if F is None:
                continue
            pl = plate_scan(F, ref["dyn"], shape, a, ref["bboxes"], ident)
            st.update({"id_cam15": ident, "window": [a, a + WINDOW_LEN - 1]})
            st["cam15_plate"] = pl
            rows.append((st, F))
    log("stage 2: %d (id, W) rows survive presence>=0.80 and median area>=%d"
        % (len(rows), MIN_CAND_AREA))

    def plate_possible(r):
        """The object leaves F often enough that the temporal median in F is
        the background -- the precondition for the coverage reading to mean
        anything at all.  A permanently static id fails it by construction."""
        occ = r[0]["cam15_plate"]["self_occupancy"]
        return occ is None or occ <= SELF_OCCUPANCY_MAX

    def ref_ok(r):
        return (r[0]["cam15_plate"]["plate_coverage"] < PLATE_USABLE
                and plate_possible(r))

    # Rows that pass the cam15 plate test first; then the rows that could in
    # principle have a plate, best coverage first (these are the informative
    # near misses); static ids last.
    rows.sort(key=lambda r: (not ref_ok(r), not plate_possible(r),
                             r[0]["cam15_plate"]["plate_coverage"],
                             -r[0]["stability"], -r[0]["median_area"]))
    short = rows[: args.cross_top]
    log("stage 2: %d rows pass the cam15 plate test, %d are plate-possible "
        "(self_occupancy <= %.2f); %d carried to cross-camera"
        % (sum(1 for r in rows if ref_ok(r)),
           sum(1 for r in rows if plate_possible(r)), SELF_OCCUPANCY_MAX, len(short)))

    # ---- stage 3: harmonisation + per-camera plate
    pb = args.poses_bounds or os.path.join(scene_dir, "poses_bounds.npy")
    geom = load_cameras_llff(pb, cams, shape)

    selftest = None
    full = []
    for rank, (st, F) in enumerate(short, start=1):
        a = st["window"][0]
        ident = st["id_cam15"]
        centre = a + WINDOW_LEN // 2
        id_maps = {}
        for cam in cams:
            if cam == args.ref_cam:
                continue
            d = cam_data[cam]
            if d["masks"] is not None:
                id_maps[cam] = d["masks"][centre]
            else:
                id_maps[cam] = _read_mask(os.path.join(
                    camera_paths(args.deva_root, args.scene, cam)[0], "%04d.png" % centre))
        ref_mask = (ref["masks"][centre] == ident)
        ys, xs = np.where(ref_mask)
        if len(ys) > 3000:
            sel = np.random.RandomState(0).choice(len(ys), 3000, replace=False)
            ys, xs = ys[sel], xs[sel]
        ref_pix = np.stack([xs, ys], axis=1).astype(np.float64)
        near = min(g["near"] for g in geom.values())
        far = max(g["far"] for g in geom.values())
        harm = harmonise(args.ref_cam, ref_pix, geom, id_maps, shape, near, far)

        per_cam = {args.ref_cam: dict(st["cam15_plate"], id=ident, vote=1.0,
                                      median_area=st["median_area"])}
        footprints = {args.ref_cam: F}
        for cam in cams:
            if cam == args.ref_cam:
                continue
            hid = harm["per_camera"].get(cam, {})
            d = cam_data[cam]
            if hid.get("id") is None or hid.get("vote", 0) < HARMONISE_VOTE_MIN:
                per_cam[cam] = {"id": hid.get("id"), "vote": hid.get("vote", 0.0),
                                "plate_coverage": None, "plate_frame": -1,
                                "median_area": None, "harmonised": False}
                continue
            cid = hid["id"]
            if d["masks"] is not None:
                masks = d["masks"]
            else:
                mdir = camera_paths(args.deva_root, args.scene, cam)[0]
                w, _ = window_frames(a)
                samp = [w[int(round(i * (WINDOW_LEN - 1) / (F_SAMPLES - 1)))]
                        for i in range(F_SAMPLES)]
                masks = {f: _read_mask(os.path.join(mdir, "%04d.png" % f)) for f in samp}
            Fc = footprint_from_masks(masks, d["areas"], cid, a, shape)
            if Fc is None:
                per_cam[cam] = {"id": cid, "vote": hid["vote"], "plate_coverage": None,
                                "plate_frame": -1, "median_area": None, "harmonised": True,
                                "note": "empty footprint"}
                continue
            pl = plate_scan(Fc, d["dyn"], shape, a, d["bboxes"], cid)
            w, _ = window_frames(a)
            per_cam[cam] = dict(pl, id=cid, vote=hid["vote"], harmonised=True,
                                median_area=int(np.median(d["areas"][cid, w])))
            footprints[cam] = Fc

        usable = [c for c, v in per_cam.items()
                  if v.get("plate_coverage") is not None and v["plate_coverage"] < PLATE_USABLE
                  and (v.get("self_occupancy") is None
                       or v["self_occupancy"] <= SELF_OCCUPANCY_MAX)]
        covs = [v["plate_coverage"] for v in per_cam.values()
                if v.get("plate_coverage") is not None]
        vis = [c for c, v in per_cam.items()
               if v.get("median_area") is not None and v["median_area"] >= MIN_CAND_AREA]
        # DEVA's segment granularity is not the same on every camera: an object
        # that is its own id on the reference camera is often swallowed by a
        # much larger segment elsewhere.  The cameras all sit on one frontal
        # rig at similar range, so a harmonised id whose area is far from the
        # reference area is a granularity mismatch, and its plate coverage is
        # a statement about that larger region, not about this object.
        ref_area = float(max(st["median_area"], 1))
        for c, v in per_cam.items():
            ma = v.get("median_area")
            v["area_ratio"] = (round(ma / ref_area, 3) if ma else None)
            v["granularity_ok"] = bool(ma and GRAN_LO <= ma / ref_area <= GRAN_HI)
        gran = [c for c, v in per_cam.items() if v["granularity_ok"]]
        # A camera on which the object is not visible needs no edit, so the
        # binding plate question is asked over the visible cameras; the
        # all-camera figure is reported alongside it.
        vis_usable = [c for c in usable if c in vis]
        cov_vis = [per_cam[c]["plate_coverage"] for c in vis
                   if per_cam[c].get("plate_coverage") is not None]

        row = dict(st)
        occ15 = row["cam15_plate"]["self_occupancy"]
        row.pop("cam15_plate", None)
        row.update({
            "rank": rank,
            "plate_possible": bool(occ15 is None or occ15 <= SELF_OCCUPANCY_MAX),
            "cam15_self_occupancy": occ15,
            "id_cam00": per_cam.get(args.held_cam, {}).get("id"),
            "harmonise_depth": round(harm["depth"], 4),
            "harmonise_score": round(harm["score"], 3),
            "cameras_total": len(cams),
            "cameras_harmonised": sum(1 for c, v in per_cam.items()
                                      if v.get("harmonised", c == args.ref_cam)),
            "plate_availability": round(len(usable) / float(len(cams)), 4),
            "plate_availability_visible": (round(len(vis_usable) / float(len(vis)), 4)
                                           if vis else 0.0),
            "plate_availability_matched": (
                round(len([c for c in usable if c in gran]) / float(len(gran)), 4)
                if gran else 0.0),
            "cameras_visible": len(vis),
            "cameras_granularity_ok": len(gran),
            "worst_coverage": round(max(covs), 5) if covs else None,
            "worst_coverage_visible": round(max(cov_vis), 5) if cov_vis else None,
            "cam15_plate_coverage": per_cam[args.ref_cam]["plate_coverage"],
            "cam15_plate_frame": per_cam[args.ref_cam]["plate_frame"],
            "median_area_cam15": st["median_area"],
            "median_area_cam00": per_cam.get(args.held_cam, {}).get("median_area"),
            "visibility": round(len(vis) / float(len(cams)), 4),
            "per_camera": per_cam,
            "footprints": footprints,
        })
        full.append(row)
        log("  cand %2d id15=%3d W=[%d,%d] stab=%.3f plate_avail=%.2f "
            "cam15cov=%.4f harm_cams=%d" %
            (rank, ident, a, a + 29, st["stability"], row["plate_availability"],
             row["cam15_plate_coverage"], row["cameras_harmonised"]))

    # harmoniser self-test on the known wine-bottle pair
    if args.scene in KNOWN_PAIRS and args.held_cam in cams:
        k15, k00 = KNOWN_PAIRS[args.scene]
        centre = 75
        id_maps = {}
        for cam in cams:
            if cam == args.ref_cam:
                continue
            d = cam_data[cam]
            id_maps[cam] = (d["masks"][centre] if d["masks"] is not None else
                            _read_mask(os.path.join(camera_paths(
                                args.deva_root, args.scene, cam)[0], "%04d.png" % centre)))
        m = (ref["masks"][centre] == k15)
        if m.any():
            ys, xs = np.where(m)
            pix = np.stack([xs, ys], axis=1).astype(np.float64)
            h = harmonise(args.ref_cam, pix, geom, id_maps, shape,
                          min(g["near"] for g in geom.values()),
                          max(g["far"] for g in geom.values()))
            got = h["per_camera"].get(args.held_cam, {})
            selftest = {"known_cam15_id": k15, "expected_cam00_id": k00,
                        "predicted_cam00_id": got.get("id"), "vote": got.get("vote"),
                        "depth": round(h["depth"], 4),
                        "pass": got.get("id") == k00}
        else:
            selftest = {"known_cam15_id": k15, "note": "id absent at frame 75"}
        log("harmoniser self-test: %s" % json.dumps(selftest))

    # Rank: a row whose reference self_occupancy fails the guard cannot have a
    # plate at all, so it goes last whatever its stability; then plate
    # availability, then stability, then size.
    full.sort(key=lambda r: (not r["plate_possible"],
                             -r["plate_availability_visible"],
                             -r["plate_availability_matched"],
                             -r["stability"], -r["median_area_cam15"]))
    for i, r in enumerate(full, start=1):
        r["rank"] = i

    # ---- montage before the footprints are dropped from the JSON
    montage_rows = full[: args.montage_top]
    if montage_rows:
        mp = os.path.join(args.out, "montage.jpg")
        try:
            build_montage(args.deva_root, args.scene, montage_rows, cam_data, mp)
            log("montage -> %s" % mp)
        except Exception as exc:                       # noqa: BLE001 - reported
            log("MONTAGE FAILED: %r" % (exc,))

    for r in full:
        r.pop("footprints", None)

    doc = {
        "scene": args.scene, "cameras": cams, "raster": list(shape),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "definitions": {
            "window": "W = [a, a+29] for a in range(30, 251, 10); margin M = [a-20, a+49]",
            "presence_frac": "frames of M with id area >= %d px, / |M|" % MIN_TRACK_AREA,
            "area_cv": "std/mean of the id area over the present frames of W",
            "centre_drift_px": "max over W of |bbox centre - median bbox centre of W|",
            "max_components": "max over %d sampled frames of W of connected components >= %d px"
                              % (F_SAMPLES, COMP_MIN_AREA),
            "stability": "p^2 * (0.40*clip(1-cv/%.2f) + 0.40*clip(1-drift/%.0f) + 0.20/comps)"
                         % (CV_SCALE, DRIFT_SCALE),
            "footprint": "F = dilate(union of the id mask over %d frames spanning W, %d px)"
                         % (F_SAMPLES, FOOTPRINT_DILATE),
            "plate_coverage": "min over t outside M of |F & (|grey(t)-median_t grey| > %d "
                              "after a %dx%d median blur)| / |F|" % (DYN_TAU, DYN_BLUR, DYN_BLUR),
            "plate_frame": "the t achieving that minimum",
            "self_occupancy": "fraction of all frames whose id bbox covers >= 30%% of F's bbox; "
                              "a guard against the temporal median containing the object",
            "usable_plate": "plate_coverage < %.2f and self_occupancy <= %.2f"
                            % (PLATE_USABLE, SELF_OCCUPANCY_MAX),
            "plate_availability": "cameras with a usable plate / all cameras",
            "granularity_ok": "the harmonised id's median area over W is within [%.2f, %.2f] "
                              "of the reference camera's; outside that band DEVA has segmented "
                              "a different-sized region and its coverage is not about this "
                              "object" % (GRAN_LO, GRAN_HI),
            "plate_availability_matched": "cameras with a usable plate / cameras whose "
                                          "harmonised segment passes granularity_ok",
            "plate_availability_visible": "cameras with a usable plate / cameras where the "
                                          "object is harmonised and its median area over W is "
                                          ">= %d px (a camera that cannot see the object needs "
                                          "no edit); this is the ranked criterion" % MIN_CAND_AREA,
            "visibility": "cameras where the harmonised id's median area over W >= %d px / all"
                          % MIN_CAND_AREA,
            "harmonisation": "plane sweep over %d depths in [near, far]: the reference mask's "
                             "pixels are back-projected and re-projected into every camera and "
                             "the DEVA id under them is voted; one global depth maximises the "
                             "summed vote; a camera needs vote >= %.2f" % (DEPTH_SAMPLES,
                                                                           HARMONISE_VOTE_MIN),
            "ranking": "usable plate on every camera first, then stability, then median area",
        },
        "harmoniser_selftest": selftest,
        "n_rows_scanned": len(rows),
        "n_rows_cam15_plate_ok": sum(1 for r in rows if ref_ok(r)),
        "n_rows_plate_possible": sum(1 for r in rows if plate_possible(r)),
        "near_miss": [dict(r[0], cam15_plate=r[0]["cam15_plate"])
                      for r in rows if plate_possible(r)][:20],
        "cam15_rows": [r[0] for r in rows],
        "ranking": full,
    }
    with open(os.path.join(args.out, "census.json"), "w") as fh:
        json.dump(doc, fh, indent=1, default=lambda o: int(o) if isinstance(o, np.integer)
                  else (float(o) if isinstance(o, np.floating) else str(o)))
    log("census -> %s  (%.1f s total)" % (os.path.join(args.out, "census.json"),
                                          time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
