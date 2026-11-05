"""Ghost vs static ring on a REAL authored-absence fixture (Multi4D monocular sweep runs).

Inputs (pack_real.sh layout): render/, fix/ (fixture = ground truth with the object removed over the gap),
orig/ (original footage, object present), mask/ (exact vacated-region masks, fixture != orig, gap frames).
Per held-out camera, pooled over gap frames:
  ghost      PSNR(render, fixture) inside the vacated mask
  ring       PSNR(render, fixture) in a 6-30 px ring around the mask, excluding pixels that move in the original
             footage (|orig_t - median_t orig| > 20 on the max channel) and pixels inside any gap mask
  obj        PSNR(render, original) inside the mask: high = the render shows the departed object
  closer     share of mask pixels where the render is closer to the original (object) than to the fixture
  pre_obj    PSNR(render, original) inside the union gap mask over pre-gap frames F0-57 (object present, seen)
Optional args: CAMS (comma list, default cam00,cam05,cam17) and F0 (default 40).
"""
import sys, os, glob, json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

root, label = sys.argv[1], sys.argv[2]
CAMS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["cam00", "cam05", "cam17"]
F0 = int(sys.argv[4]) if len(sys.argv) > 4 else 40   # first frame available (pre-gap window F0..57)
G0, G1 = 60, 89
def im(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]; return x * x + y * y <= r * r
def ps(se, n): return float(10 * np.log10(1.0 / (se / max(n, 1)))) if n else float("nan")
out = {}
for c in CAMS:
    orig = {t: im(f"{root}/orig/{c}/{t:04d}.png") for t in range(F0, 100)}
    med = np.median(np.stack([orig[t] for t in range(F0, 100, 2)]), 0)
    masks = {t: np.asarray(Image.open(f"{root}/mask/{c}/{t:05d}.png")) > 0 for t in range(G0, G1 + 1)}
    U = np.any(np.stack(list(masks.values())), 0)
    if U.sum() == 0:
        out[c] = {"mask_px": 0}; continue
    ys, xs = np.where(binary_dilation(U, disk(31)))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    acc = dict(g=0.0, gn=0, r=0.0, rn=0, o=0.0, cl=0, pre=0.0, pn=0)
    for t in range(G0, G1 + 1):
        m = masks[t][y0:y1, x0:x1]
        if m.sum() == 0: continue
        r = im(f"{root}/render/{c}/{t:04d}.png")[y0:y1, x0:x1]; f = im(f"{root}/fix/{c}/{t:04d}.png")[y0:y1, x0:x1]
        o = orig[t][y0:y1, x0:x1]
        mov = np.abs(o - med[y0:y1, x0:x1]).max(-1) > 20 / 255
        ring = binary_dilation(m, disk(30)) & ~binary_dilation(m, disk(6)) & ~mov & ~U[y0:y1, x0:x1]
        ef = ((r - f) ** 2).sum(-1) / 3; eo = ((r - o) ** 2).sum(-1) / 3
        acc["g"] += ef[m].sum(); acc["gn"] += m.sum(); acc["o"] += eo[m].sum(); acc["cl"] += int((eo[m] < ef[m]).sum())
        acc["r"] += ef[ring].sum(); acc["rn"] += ring.sum()
    u = U[y0:y1, x0:x1]
    for t in range(F0, 58):
        r = im(f"{root}/render/{c}/{t:04d}.png")[y0:y1, x0:x1]; o = orig[t][y0:y1, x0:x1]
        acc["pre"] += (((r - o) ** 2).sum(-1) / 3)[u].sum(); acc["pn"] += u.sum()
    g, ri = ps(acc["g"], acc["gn"]), ps(acc["r"], acc["rn"])
    out[c] = {"mask_px_mean": round(acc["gn"] / 30), "ghost": round(g, 2), "ring": round(ri, 2), "ghost_minus_ring": round(g - ri, 2),
              "obj": round(ps(acc["o"], acc["gn"]), 2), "closer_to_object": round(acc["cl"] / acc["gn"], 3),
              "pre_obj": round(ps(acc["pre"], acc["pn"]), 2), "ring_px_mean": round(acc["rn"] / 30)}
    print(label, c, json.dumps(out[c]))
os.makedirs("D:/adags-ars/fmd/scores", exist_ok=True)
json.dump(out, open(f"D:/adags-ars/fmd/scores/real_{label}.json", "w"), indent=1)
