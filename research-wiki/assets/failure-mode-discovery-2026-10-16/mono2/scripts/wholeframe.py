import sys, json, numpy as np
from PIL import Image
d, L = sys.argv[1:3]; out = {}
for c in ["cam00","cam05","cam17"]:
    p = []
    for t in range(40,100):
        r = np.asarray(Image.open(f"{d}/render/{c}/{t:04d}.png").convert("RGB")).astype(np.float32)/255
        f = np.asarray(Image.open(f"{d}/fix/{c}/{t:04d}.png").convert("RGB")).astype(np.float32)/255
        p.append(-10*np.log10(((r-f)**2).mean()))
    out[c] = {"wholeframe_psnr_40_99": round(float(np.mean(p)),2), "pre_gap_40_57": round(float(np.mean(p[:18])),2), "gap_60_89": round(float(np.mean(p[20:50])),2)}
    print(L, c, out[c])
json.dump(out, open(f"D:/adags-ars/fmd/scores/wholeframe_{L}.json","w"), indent=1)
