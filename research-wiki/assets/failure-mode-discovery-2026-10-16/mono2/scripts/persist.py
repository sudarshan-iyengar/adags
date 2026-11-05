"""fmd mono2: registration-free persistence check on a ghost_ring_real pack.

Inside the union vacated mask of each held-out camera:
  render_change  PSNR(render_t, mean render over pre-gap frames 40-57), pooled over gap frames 60-89.
                 HIGH (e.g. > 30 dB) = the render of the vacated region did not change when the object left.
  truth_change   the same for the footage the method saw: PSNR(fixture_t (gap), mean original 40-57).
                 What a correct reconstruction would show (the object leaving = a large change, low PSNR).
  closer_to_own_object  share of gap mask pixels where the render is closer to the method's OWN pre-gap render
                 (the object as this method draws it) than to the fixture (the empty surface).
  pre_noise      PSNR(render_t, pre-gap mean render) over pre-gap frames 40-57: the render's own frame-to-frame
                 variation while nothing changes (a reference for render_change).
Because render_change compares a method's renders with each other, a fixed misregistration of the held-out camera
cancels; the cook's arm crossing the region lowers it for every method alike (it lowers truth_change too).
Usage: persist.py PACK LABEL"""
import sys, json, numpy as np
from PIL import Image

d, L = sys.argv[1:3]
def im(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255
def ps(se, n): return float(10 * np.log10(n / se)) if se > 0 else 99.0
out = {}
for c in ["cam00", "cam05", "cam17"]:
    U = np.any(np.stack([np.asarray(Image.open(f"{d}/mask/{c}/{t:05d}.png")) > 0 for t in range(60, 90)]), 0)
    pre_r = np.mean([im(f"{d}/render/{c}/{t:04d}.png")[U] for t in range(40, 58)], 0)
    pre_o = np.mean([im(f"{d}/orig/{c}/{t:04d}.png")[U] for t in range(40, 58)], 0)
    rc = tc = pn = 0.0; n = U.sum() * 3; cl = 0
    for t in range(60, 90):
        r = im(f"{d}/render/{c}/{t:04d}.png")[U]; f = im(f"{d}/fix/{c}/{t:04d}.png")[U]
        rc += ((r - pre_r) ** 2).sum(); tc += ((f - pre_o) ** 2).sum()
        cl += int((((r - pre_r) ** 2).sum(-1) < ((r - f) ** 2).sum(-1)).sum())
    for t in range(40, 58):
        pn += ((im(f"{d}/render/{c}/{t:04d}.png")[U] - pre_r) ** 2).sum()
    out[c] = {"render_change": round(ps(rc, n * 30), 2), "truth_change": round(ps(tc, n * 30), 2), "pre_noise": round(ps(pn, n * 18), 2),
              "closer_to_own_object": round(cl / (U.sum() * 30), 3)}
    print(L, c, out[c])
json.dump(out, open(f"D:/adags-ars/fmd/scores/persist_{L}.json", "w"), indent=1)
