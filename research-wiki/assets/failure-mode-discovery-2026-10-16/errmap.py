"""Where do errors cluster? Per-pixel error maps of held-out renders, across methods, in space and time.

For one scene and one held-out camera, given renders from several methods and the ground-truth frames:
  * per-frame, per-pixel error e = mean |render - gt| over RGB (0..1), pooled to an 8x8 grid (127x169 cells);
  * per-method: whole-frame PSNR, the error curve over time, the mean error map, and CONCENTRATION: the share of total
    squared error carried by the worst 1 % / 5 % of cells (space-time);
  * SHARED error = the minimum over methods of the cell error: what every method gets wrong;
  * hotspots = 3-D connected components (time x rows x cols) of SHARED error above a threshold (default: the 99th
    percentile of the shared error, and at least 0.08), ranked by excess error mass;
  * cue per hotspot, from the ground truth only: how much the cell changes over time (|gt_t - temporal median|, a
    motion/change cue) at the hotspot vs the frame average;
  * crops for each top hotspot at its peak frame: GT | each method's render | the shared-error heat map.
Usage: errmap.py --scene NAME --gt PATTERN --method LABEL=PATTERN [--method ...] --frames 0:300 --out DIR
Patterns use {f} (python format, e.g. {f:04d}).
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--gt", required=True)
ap.add_argument("--method", action="append", required=True)
ap.add_argument("--frames", default="0:300")
ap.add_argument("--out", required=True)
ap.add_argument("--cell", type=int, default=8)
ap.add_argument("--pct", type=float, default=99.0)
ap.add_argument("--floor", type=float, default=0.08)
ap.add_argument("--topk", type=int, default=12)
a = ap.parse_args()
f0, f1 = (int(x) for x in a.frames.split(":"))
T = list(range(f0, f1))
meths = [m.split("=", 1) for m in a.method]
os.makedirs(a.out, exist_ok=True)
C = a.cell


def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0


def pool(x):
    h, w = x.shape[0] // C * C, x.shape[1] // C * C
    return x[:h, :w].reshape(h // C, C, w // C, C).mean((1, 3))


g0 = load(a.gt.format(f=T[0]))
H, W = g0.shape[0] // C, g0.shape[1] // C
E = {lab: np.zeros((len(T), H, W), np.float32) for lab, _ in meths}
SE = {lab: np.zeros((len(T), H, W), np.float32) for lab, _ in meths}
G = np.zeros((len(T), H, W, 3), np.float32)
psnr = {lab: [] for lab, _ in meths}
for i, t in enumerate(T):
    g = load(a.gt.format(f=t))
    G[i] = np.stack([pool(g[..., k]) for k in range(3)], -1)
    for lab, pat in meths:
        r = load(pat.format(f=t))
        assert r.shape == g.shape, (lab, r.shape, g.shape)
        d = np.abs(r - g).mean(-1)
        E[lab][i] = pool(d)
        SE[lab][i] = pool(((r - g) ** 2).mean(-1))
        psnr[lab].append(float(-10 * np.log10(((r - g) ** 2).mean())))
gmed = np.median(G, 0)
change = np.abs(G - gmed).mean(-1)                       # cue: how much the GT cell departs from its temporal median
shared = np.min(np.stack([E[l] for l, _ in meths]), 0)
thr = max(float(np.percentile(shared, a.pct)), a.floor)
lab3, n = ndimage.label(shared > thr, structure=np.ones((3, 3, 3)))
comps = []
for k, sl in enumerate(ndimage.find_objects(lab3), 1):
    m = lab3[sl] == k
    vals = shared[sl][m]
    mass = float((vals - thr).sum())
    tt = np.where(m.any((1, 2)))[0] + sl[0].start
    peak = int(sl[0].start + np.argmax((shared[sl] * m).sum((1, 2))))
    comps.append({"id": k, "mass": round(mass, 3), "cells": int(m.sum()),
                  "frames": [T[int(tt.min())], T[int(tt.max())]], "n_frames": int(len(tt)), "peak_frame": T[peak],
                  "box_px": [int(sl[1].start * C), int(sl[1].stop * C), int(sl[2].start * C), int(sl[2].stop * C)],
                  "mean_shared_err": round(float(vals.mean()), 4),
                  "gt_change_in": round(float(change[sl][m].mean()), 4),
                  "per_method_err": {l: round(float(E[l][sl][m].mean()), 4) for l, _ in meths}})
comps.sort(key=lambda c: -c["mass"])
tot_change = float(change.mean())
summary = {"scene": a.scene, "frames": [f0, f1 - 1], "cell_px": C, "threshold": round(thr, 4), "n_hotspots": n,
           "gt_change_mean": round(tot_change, 4), "methods": {}}
for l, _ in meths:
    s = np.sort(SE[l].ravel())[::-1]
    summary["methods"][l] = {"psnr_mean": round(float(np.mean(psnr[l])), 2),
                             "worst_frame": T[int(np.argmin(psnr[l]))], "worst_frame_psnr": round(float(np.min(psnr[l])), 2),
                             "sq_err_share_top1pct": round(float(s[: len(s) // 100].sum() / s.sum()), 3),
                             "sq_err_share_top5pct": round(float(s[: len(s) // 20].sum() / s.sum()), 3),
                             "share_of_sq_err_in_hotspots": round(float(SE[l][lab3 > 0].sum() / SE[l].sum()), 3)}
summary["hotspots"] = comps[: a.topk]
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
np.savez_compressed(os.path.join(a.out, "maps.npz"), shared=shared, change=change, **{f"err_{l}": E[l] for l, _ in meths})
# figures: mean shared-error map and per-method curves (as an image strip)
mm = shared.mean(0); mm = (np.clip(mm / max(mm.max(), 1e-6), 0, 1) * 255).astype(np.uint8)
Image.fromarray(mm).resize((W * 4, H * 4), Image.NEAREST).save(os.path.join(a.out, "shared_mean.png"))
for j, c in enumerate(comps[: a.topk]):
    t = c["peak_frame"]; y0, y1, x0, x1 = c["box_px"]
    pad = 48; y0, x0 = max(0, y0 - pad), max(0, x0 - pad); y1, x1 = min(g0.shape[0], y1 + pad), min(g0.shape[1], x1 + pad)
    tiles = [("GT", load(a.gt.format(f=t)))] + [(l, load(p.format(f=t))) for l, p in meths]
    e = shared[T.index(t)]; e = np.kron(e, np.ones((C, C)))[: g0.shape[0], : g0.shape[1]]
    tiles.append(("shared err", np.repeat(np.clip(e / max(e.max(), 1e-6), 0, 1)[..., None], 3, -1)))
    ims = []
    for name, im in tiles:
        crop = Image.fromarray((im[y0:y1, x0:x1] * 255).astype(np.uint8))
        s = 260 / max(crop.size); crop = crop.resize((max(1, int(crop.size[0] * s)), max(1, int(crop.size[1] * s))))
        ImageDraw.Draw(crop).text((3, 3), name, fill=(255, 255, 0)); ims.append(np.asarray(crop))
    h = max(i.shape[0] for i in ims)
    row = np.concatenate([np.pad(i, ((0, h - i.shape[0]), (0, 2), (0, 0))) for i in ims], 1)
    Image.fromarray(row).save(os.path.join(a.out, f"hot{j:02d}_f{t:04d}.png"))
print(json.dumps({k: v for k, v in summary.items() if k != "hotspots"}, indent=1))
for c in comps[:a.topk]:
    print(c["id"], "mass", c["mass"], "frames", c["frames"], "peak", c["peak_frame"], "box", c["box_px"],
          "err", c["mean_shared_err"], "gt_change", c["gt_change_in"], "vs mean", round(tot_change, 4))
