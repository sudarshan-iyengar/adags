"""fmd dy Pilot 2 day-0 gate (no GPU training): share of held-out squared error inside the silhouette band.

For each scene and held-out camera, over all held-out frames, inside the official co-visibility mask:
  share_band     = SSE(band) / SSE(co-visible)      band = within 4 px (max-norm) of the DYNAMIC-mask boundary
  share_bandfast = the same around the FAST mask (hands, held objects, blades, wheel)
  share_dynamic  = SSE(dynamic AND co-visible) / SSE(co-visible);  area shares for comparison
Band width is 8 px at the method's evaluation resolution: MoSca 360x480 (masks as frozen); Shape of Motion 720x960
(dynamic/fast masks nearest-upsampled x2, band rebuilt at 1x, its own 1x co-visibility masks).
Usage: band_gate.py REGIONS_DIR OUT_JSON
"""
import sys, os, glob, json, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
REG, OUT = sys.argv[1:3]
X = "/leonardo_scratch/large/userexternal/siyengar/fmd"
def im(p, size=None):
    a = Image.open(p).convert("RGB")
    return np.asarray(a).astype(np.float32) / 255
def m(p): return np.asarray(Image.open(p).convert("L")) > 127
def dil(x, r):
    t = torch.from_numpy(x.astype(np.float32))[None, None]
    return F.max_pool2d(t, 2 * r + 1, stride=1, padding=r)[0, 0].numpy() > 0.5
def band(x, cov):
    return dil(x, 4) & dil(~x, 4) & cov if x.any() else np.zeros_like(x)
res = {}
for meth in ["mosca", "som"]:
    for scene in ["apple", "block", "paper-windmill", "space-out", "spin", "teddy", "wheel"]:
        if meth == "mosca":
            ld = sorted(glob.glob(f"{X}/runs/mosca_iphone/{scene}/logs/iphone_fit_*"))[-1]
            rdir, gdir, cdir = f"{ld}/tto_test", f"{X}/runs/mosca_iphone/{scene}/test_images", f"{X}/runs/mosca_iphone/{scene}/test_covisible"
        else:
            rdir = f"{X}/runs/som_iphone/{scene}/results/rgb"
            if not os.path.isdir(rdir): continue
            gdir, cdir = f"{X}/data/dycheck/som/scenes/{scene}/rgb/1x", f"{X}/data/dycheck/som/scenes/{scene}/flow3d_preprocessed/covisible/1x/val"
        names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{REG}/{scene}/dynamic/*.png"))
        for cam in sorted({n.split("_")[0] for n in names}):
            acc = dict(cov=0., band=0., bandfast=0., dyn=0., a_cov=0, a_band=0, a_bandfast=0, a_dyn=0, n=0)
            for n in [n for n in names if n.startswith(cam + "_")]:
                if not os.path.exists(f"{rdir}/{n}.png"): continue
                g = im(f"{gdir}/{n}.png"); r = im(f"{rdir}/{n}.png")
                if r.shape != g.shape: continue
                cov = m(f"{cdir}/{n}.png")
                dyn = m(f"{REG}/{scene}/dynamic/{n}.png"); fast = m(f"{REG}/{scene}/fast/{n}.png")
                if dyn.shape != cov.shape:   # Shape of Motion at 1x
                    dyn = np.kron(dyn, np.ones((2, 2), bool)); fast = np.kron(fast, np.ones((2, 2), bool))
                b, bf = band(dyn, cov), band(fast, cov)
                e = ((g - r) ** 2).mean(-1)
                acc["cov"] += e[cov].sum(); acc["band"] += e[b].sum(); acc["bandfast"] += e[bf].sum(); acc["dyn"] += e[dyn & cov].sum()
                acc["a_cov"] += cov.sum(); acc["a_band"] += b.sum(); acc["a_bandfast"] += bf.sum(); acc["a_dyn"] += (dyn & cov).sum(); acc["n"] += 1
            if acc["n"] == 0: continue
            o = {"frames": acc["n"], "share_band": acc["band"] / acc["cov"], "share_bandfast": acc["bandfast"] / acc["cov"],
                 "share_dynamic": acc["dyn"] / acc["cov"], "area_band": acc["a_band"] / acc["a_cov"],
                 "area_bandfast": acc["a_bandfast"] / acc["a_cov"], "area_dynamic": acc["a_dyn"] / acc["a_cov"]}
            o = {k: (round(float(v), 4) if isinstance(v, float) or isinstance(v, np.floating) else v) for k, v in o.items()}
            res[f"{meth}/{scene}/{cam}"] = o; print(meth, scene, cam, o, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
