"""fmd dy: region metrics for MoSca DyCheck runs on the FROZEN region masks (regions_v4_final, manifest sha256 afd674c2...).

Per run (runs/dy/<run>/logs/iphone_fit_*/tto_test vs the run's test_images), per held-out camera, pooled over frames:
  fast_psnr       PSNR inside fast (hands, held objects, blades, wheel) AND co-visibility   [Pilot 1 fast-part metric]
  fast_psnr_all   PSNR inside fast, not restricted to co-visibility
  dyn_psnr        PSNR inside the dynamic mask (ViDAR-style: dynamic in place of co-visibility)
  static_psnr     PSNR inside static (co-visible minus dilated dynamic)
  band_psnr / band_sse_share   8-px silhouette band (Pilot 2 metric)
  blur            Laplacian band-energy ratio inside dynamic AND co-visible: sum|Lap(render)|^2 / sum|Lap(GT)|^2 (<1 = softer)
Writes runs/dy/<run>/region_scores.json (skips runs that already have it). Usage: region_score.py PREFIX
"""
import sys, os, glob, json, numpy as np
from PIL import Image
X = "/leonardo_scratch/large/userexternal/siyengar/fmd"; REG = f"{X}/data/dycheck/regions_v4_final"
LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
def lap(g):
    p = np.pad(g, 1, mode="edge"); return sum(LAP[i, j] * p[i:i + g.shape[0], j:j + g.shape[1]] for i in range(3) for j in range(3))
def im(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255
def mk(p): return np.asarray(Image.open(p).convert("L")) > 127
def ps(se, n): return float(-10 * np.log10(se / n)) if n > 0 and se > 0 else None
for ws in sorted(glob.glob(f"{X}/runs/dy/{sys.argv[1]}*")):
    out = f"{ws}/region_scores.json"
    if os.path.exists(out): continue
    ld = sorted(glob.glob(f"{ws}/logs/iphone_fit_*/tto_test"))
    if not ld or not os.path.exists(f"{ws}/provenance.txt") or "logdir=" not in open(f"{ws}/provenance.txt").read(): continue
    scene = open(f"{ws}/provenance.txt").read().split("scene=")[1].split()[0]
    td = ld[-1]; names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{REG}/{scene}/dynamic/*.png"))
    res = {}
    for cam in sorted({n.split("_")[0] for n in names}):
        a = {k: 0.0 for k in ["fast", "fasta", "dyn", "stat", "band", "cov", "lr", "lg"]}; c = {k: 0 for k in ["fast", "fasta", "dyn", "stat", "band", "cov"]}
        for n in [n for n in names if n.startswith(cam + "_")]:
            if not os.path.exists(f"{td}/{n}.png"): continue
            g = im(f"{ws}/test_images/{n}.png"); r = im(f"{td}/{n}.png"); e = ((g - r) ** 2).mean(-1)
            cov = mk(f"{ws}/test_covisible/{n}.png"); fast = mk(f"{REG}/{scene}/fast/{n}.png"); dyn = mk(f"{REG}/{scene}/dynamic/{n}.png")
            st = mk(f"{REG}/{scene}/static/{n}.png"); bd = mk(f"{REG}/{scene}/band/{n}.png")
            for k, m in [("fast", fast & cov), ("fasta", fast), ("dyn", dyn), ("stat", st), ("band", bd), ("cov", cov)]:
                a[k] += float(e[m].sum()); c[k] += int(m.sum())
            md = dyn & cov
            if md.any():
                a["lr"] += float((lap(r.mean(-1)) ** 2)[md].sum()); a["lg"] += float((lap(g.mean(-1)) ** 2)[md].sum())
        res[f"cam{cam}"] = {"fast_psnr": ps(a["fast"], c["fast"]), "fast_psnr_all": ps(a["fasta"], c["fasta"]), "dyn_psnr": ps(a["dyn"], c["dyn"]),
                            "static_psnr": ps(a["stat"], c["stat"]), "band_psnr": ps(a["band"], c["band"]),
                            "band_sse_share": a["band"] / a["cov"] if a["cov"] else None, "blur": a["lr"] / a["lg"] if a["lg"] else None,
                            "cov_psnr_pooled": ps(a["cov"], c["cov"]), "px": c}
    json.dump({"scene": scene, "regions_manifest": open(f"{REG}/FROZEN.txt").read().split()[1], **res}, open(out, "w"), indent=1)
    print(os.path.basename(ws), {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "px"} for k, v in res.items()}, flush=True)
