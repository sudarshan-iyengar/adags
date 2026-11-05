"""How much of a held-out render's error is a global colour/exposure mismatch? Per frame, fit a per-channel affine
map (gain, offset) from render to GT by least squares over all pixels; report PSNR before and after, and the fitted
gains/offsets. Usage: colorfit.py LABEL GT_PATTERN RENDER_PATTERN (patterns with {f})."""
import sys, json
import numpy as np
from PIL import Image
lab, gp, rp = sys.argv[1:4]
fr = list(range(0, 300, 30))
out = []
for f in fr:
    g = np.asarray(Image.open(gp.format(f=f)).convert("RGB")).astype(np.float64) / 255
    r = np.asarray(Image.open(rp.format(f=f)).convert("RGB")).astype(np.float64) / 255
    p0 = -10 * np.log10(((r - g) ** 2).mean())
    rc = np.empty_like(r); gains = []
    for c in range(3):
        A = np.stack([r[..., c].ravel(), np.ones(r[..., c].size)], 1)
        k, b = np.linalg.lstsq(A, g[..., c].ravel(), rcond=None)[0]
        rc[..., c] = k * r[..., c] + b; gains.append((round(k, 3), round(b, 3)))
    p1 = -10 * np.log10(((np.clip(rc, 0, 1) - g) ** 2).mean())
    out.append((f, round(p0, 2), round(p1, 2), gains))
d = np.array([o[2] - o[1] for o in out])
print(lab, "mean PSNR", round(np.mean([o[1] for o in out]), 2), "-> after global affine colour fit", round(np.mean([o[2] for o in out]), 2),
      "gain +", round(d.mean(), 2), "dB; frame0 gains/offsets", out[0][3])
