"""N3V evaluation / oracle-study scorer: one protocol for every method's held-out cam00 renders.
(Frozen 2026-10-19; extended 2026-10-20 with --frames_list and the per-camera colour fit O4a.)

Per method, over the chosen frames:
  psnr_own      vs the method's own ground truth (when given)
  psnr_<f>      vs the native 2704x2028 cam00 frame halved with filter f in {lanczos, bicubic, bilinear, area}
  psnr_cc       per-FRAME colour fit (gain + offset per channel, refitted every frame) vs Lanczos ground truth
  psnr_ccam     per-CAMERA colour fit (O4a): ONE gain + offset per channel fitted over all scored frames and pixels,
                computed exactly from per-frame sufficient statistics (clipping to [0,1] is ignored for this one)
  regions       raw / per-frame cc / per-camera cc PSNR in: dynamic (cells departing from the Lanczos ground truth's
                temporal median by > 0.05 mean abs, dilated 16 px), static (the rest), each --rect, each --mask_dir
                (binary PNG per frame, {f} pattern; e.g. SAM2 masks for tools, hands, flames)
  per_frame     per-frame raw PSNR (whole) for frame-level comparisons
PSNRs are means of per-frame PSNR.
Usage: eval_n3v.py --scene S --native PAT --method "NAME=render_pat[|own_gt_pat]" ... [--rect n=x0,x1,y0,y1]
                   [--mask n=PAT] [--frames 0:300 --step 1 | --frames_list 10,20,30] --out scores.json
"""
import argparse, json
import numpy as np
from PIL import Image
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True); ap.add_argument("--native", required=True)
ap.add_argument("--method", action="append", required=True); ap.add_argument("--rect", action="append", default=[])
ap.add_argument("--mask", action="append", default=[]); ap.add_argument("--frames", default="0:300")
ap.add_argument("--step", type=int, default=1); ap.add_argument("--frames_list", default="")
ap.add_argument("--median_from", default="", help="frames for the temporal median (default: 60 spread over 0..299)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
if a.frames_list:
    T = [int(x) for x in a.frames_list.split(",")]
else:
    f0, f1 = (int(x) for x in a.frames.split(":")); T = list(range(f0, f1, a.step))
FILT = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "area": None}
meths = []
for m in a.method:
    name, rest = m.split("=", 1); rp, _, gp = rest.partition("|"); meths.append((name, rp, gp or None))
rects = {r.split("=")[0]: [int(v) for v in r.split("=")[1].split(",")] for r in a.rect}
mpats = {m.split("=", 1)[0]: m.split("=", 1)[1] for m in a.mask}


def load(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float64) / 255


def half(native, f):
    if f == "area":
        return np.asarray(native).astype(np.float64).reshape(1014, 2, 1352, 2, 3).mean((1, 3)) / 255
    return np.asarray(native.resize((1352, 1014), FILT[f])).astype(np.float64) / 255


def nat(t):
    im = Image.open(a.native.format(f=t)).convert("RGB"); assert im.size == (2704, 2028), im.size; return im


def mse(x, y, m=None):
    d = ((x - y) ** 2).mean(-1)
    return float(d[m].mean()) if m is not None else float(d.mean())


def cfit(r, g):
    o = np.empty_like(r)
    for c in range(3):
        A = np.stack([r[..., c].ravel(), np.ones(r[..., c].size)], 1)
        o[..., c] = (A @ np.linalg.lstsq(A, g[..., c].ravel(), rcond=None)[0]).reshape(r.shape[:2])
    return np.clip(o, 0, 1)


def stats(r, g, m):
    """per-channel sufficient statistics [n, Sr, Sg, Srr, Srg, Sgg] inside mask m"""
    out = np.zeros((3, 6))
    for c in range(3):
        x, y = r[..., c][m], g[..., c][m]
        out[c] = [x.size, x.sum(), y.sum(), (x * x).sum(), (x * y).sum(), (y * y).sum()]
    return out


med_frames = [int(x) for x in a.median_from.split(",")] if a.median_from else list(range(0, 300, 5))
med = np.median(np.stack([half(nat(t), "lanczos").astype(np.float32) for t in med_frames]), 0)
regions = ["whole", "dynamic", "static"] + list(rects) + list(mpats)
acc = {n: {k: [] for k in ["own"] + [f"f_{f}" for f in FILT]} for n, _, _ in meths}
reg = {n: {r: {"raw": [], "cc": [], "st": []} for r in regions} for n, _, _ in meths}
perf = {n: [] for n, _, _ in meths}; dynfrac = []
for t in T:
    GTt = {f: half(nat(t), f) for f in FILT}; g = GTt["lanczos"]
    dyn = ndimage.binary_dilation(np.abs(g - med).mean(-1) > 0.05, iterations=16); dynfrac.append(dyn.mean())
    masks = {"whole": np.ones(g.shape[:2], bool), "dynamic": dyn, "static": ~dyn}
    for rn, (x0, x1, y0, y1) in rects.items():
        m = np.zeros(g.shape[:2], bool); m[y0:y1, x0:x1] = True; masks[rn] = m
    for rn, pat in mpats.items():
        masks[rn] = np.asarray(Image.open(pat.format(f=t)).convert("L")) > 127
    for name, rp, gp in meths:
        r = load(rp.format(f=t)); rc = cfit(r, g)
        if gp: acc[name]["own"].append(mse(r, load(gp.format(f=t))))
        for f in FILT: acc[name][f"f_{f}"].append(mse(r, GTt[f]))
        perf[name].append(round(-10 * np.log10(max(mse(r, g), 1e-12)), 3))
        for rn, m in masks.items():
            if m.any():
                reg[name][rn]["raw"].append(mse(r, g, m)); reg[name][rn]["cc"].append(mse(rc, g, m))
                reg[name][rn]["st"].append(stats(r, g, m))
ps = lambda v: round(float(np.mean([-10 * np.log10(max(x, 1e-12)) for x in v])), 3) if v else None


def cam_fit_psnr(st_list):
    """O4a: one (k, b) per channel fitted over all frames, per-frame MSE from the statistics, mean per-frame PSNR"""
    if not st_list: return None, None
    S = np.sum(st_list, 0); kb = []
    for c in range(3):
        n, sr, sg, srr, srg, _ = S[c]
        A = np.array([[srr, sr], [sr, n]]); y = np.array([srg, sg]); kb.append(np.linalg.solve(A, y))
    per = []
    for st in st_list:
        e = 0.0; N = 0.0
        for c in range(3):
            n, sr, sg, srr, srg, sgg = st[c]; k, b = kb[c]
            e += k * k * srr + b * b * n + sgg + 2 * k * b * sr - 2 * k * srg - 2 * b * sg; N += n
        per.append(max(e / N, 1e-12))
    return round(float(np.mean([-10 * np.log10(x) for x in per])), 3), [[round(float(k), 4), round(float(b), 4)] for k, b in kb]


out = {"scene": a.scene, "frames": T if len(T) <= 20 else [T[0], T[-1], len(T)], "dynamic_frac": round(float(np.mean(dynfrac)), 4), "methods": {}}
for name, _, _ in meths:
    res = {"psnr_own": ps(acc[name]["own"]), **{f"psnr_{f}": ps(acc[name][f"f_{f}"]) for f in FILT}, "regions": {}, "per_frame": perf[name]}
    for rn, v in reg[name].items():
        p_cam, kb = cam_fit_psnr(v["st"])
        res["regions"][rn] = {"raw": ps(v["raw"]), "cc_frame": ps(v["cc"]), "cc_camera": p_cam}
        if rn == "whole": res["camera_colour_fit"] = kb
    out["methods"][name] = res
    print(name, json.dumps({k: v for k, v in res.items() if k != "per_frame"}), flush=True)
json.dump(out, open(a.out, "w"), indent=1)
