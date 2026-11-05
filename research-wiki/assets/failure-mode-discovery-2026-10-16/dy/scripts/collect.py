"""fmd dy: collect official MoSca metrics (tto_dycheck_metrics.xlsx: per-frame mpsnr/mssim/mlpips over the co-visible
mask) for every finished run under runs/dy, per held-out camera and overall, plus active primitive counts.
Usage: collect.py PREFIX OUT_JSON   (PREFIX e.g. p0_ or d1_)"""
import sys, os, glob, json, re
import pandas as pd
X = "/leonardo_scratch/large/userexternal/siyengar/fmd"
pre, out = sys.argv[1:3]
res = {}
for ws in sorted(glob.glob(f"{X}/runs/dy/{pre}*")):
    run = os.path.basename(ws); f = glob.glob(f"{ws}/logs/iphone_fit_*/tto_dycheck_metrics.xlsx")
    if not f: continue
    df = pd.read_excel(f[0]); df = df[df.fn.astype(str).str.match(r"^\d_")]
    df["cam"] = df.fn.astype(str).str[0]
    r = {"n": len(df), **{k: round(float(df[k].mean()), 4) for k in ["mpsnr", "mssim", "mlpips"]}}
    for c, g in df.groupby("cam"): r[f"cam{c}"] = {k: round(float(g[k].mean()), 4) for k in ["mpsnr", "mssim", "mlpips"]}
    pv = open(f"{ws}/provenance.txt").read()
    m = re.search(r"prims static=(\d+) static_active=(\d+) dynamic=(\d+) dynamic_active=(\d+)", pv)
    if m: r["prims"] = dict(zip(["static", "static_active", "dynamic", "dynamic_active"], map(int, m.groups())))
    res[run] = r
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items(): print(k, v["mpsnr"], v["mssim"], v["mlpips"], v.get("prims"))
