"""fmd dy Pilot 1 (D1) readout against the frozen thresholds of 06_pilots_designed.md.
Inputs (pulled from Leonardo): collect_p0.json, collect_d1.json (official MoSca metrics), regions/<run>.json
(region_score.py on the frozen masks). Baseline = the P0 published reruns of the scene (section 33 update); the
patched-copy variant 1 is shown as the no-op check. Spread = max - min over the baseline reruns (paired spread)."""
import json, glob, os, numpy as np
p0 = json.load(open("collect_p0.json")); d1 = json.load(open("collect_d1.json"))
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob("regions/*.json")}
SC = ["apple", "paper-windmill", "spin"]
V = {"v1": "published (patched copy, support off)", "v2": "window 8 frames", "v3": "window 32 frames",
     "v4": "soft reliability weighting", "v5": "reliability-selected support", "v6": "shuffled support (placebo)"}
def runs(prefix): return sorted(k for k in list(p0) + list(d1) if k.startswith(prefix))
def off(k): return (p0.get(k) or d1.get(k))
def agg(keys):
    o = {}
    if not keys: return None
    o["n"] = len(keys)
    o["mpsnr"] = np.mean([off(k)["mpsnr"] for k in keys]); o["mpsnr_min"] = min(off(k)["mpsnr"] for k in keys); o["mpsnr_max"] = max(off(k)["mpsnr"] for k in keys)
    for cam in ["cam1", "cam2"]:
        o[cam] = np.mean([off(k)[cam]["mpsnr"] for k in keys if cam in off(k)]) if any(cam in off(k) for k in keys) else None
    for m in ["fast_psnr", "static_psnr", "dyn_psnr", "band_psnr", "blur"]:
        for cam in ["cam1", "cam2"]:
            vals = [R[k][cam][m] for k in keys if k in R and cam in R[k] and R[k][cam][m] is not None]
            o[f"{m}_{cam}"] = np.mean(vals) if vals else None
    o["dyn_active"] = np.mean([off(k).get("prims", {}).get("dynamic_active", np.nan) for k in keys])
    return o
table = {}
for s in SC:
    table[s] = {"base": agg(runs(f"p0_{s}_"))}
    for v in V: table[s][v] = agg(runs(f"d1_{s}_{v}_"))
def f(x, d=2): return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"
lines = ["| Scene | Variant | runs | mPSNR (range) | d vs base | cam1 / cam2 | fast-part PSNR cam1 / cam2 (d) | static PSNR d | blur ratio | active moving prims |", "|" + "---|" * 10]
for s in SC:
    b = table[s]["base"]
    for v in ["base"] + list(V):
        a = table[s][v]
        if a is None: continue
        name = "P0 published reruns (baseline)" if v == "base" else f"{v} {V[v]}"
        dd = a["mpsnr"] - b["mpsnr"]
        fp = "/".join(f(a[f"fast_psnr_{c}"]) for c in ["cam1", "cam2"])
        fd = "/".join(f(a[f"fast_psnr_{c}"] - b[f"fast_psnr_{c}"]) if a[f"fast_psnr_{c}"] is not None and b[f"fast_psnr_{c}"] is not None else "-" for c in ["cam1", "cam2"])
        sd = "/".join(f(a[f"static_psnr_{c}"] - b[f"static_psnr_{c}"]) if a[f"static_psnr_{c}"] is not None and b[f"static_psnr_{c}"] is not None else "-" for c in ["cam1", "cam2"])
        bl = "/".join(f(a[f"blur_{c}"], 3) for c in ["cam1", "cam2"])
        lines.append(f"| {s} | {name} | {a['n']} | {f(a['mpsnr'])} ({f(a['mpsnr_min'])}-{f(a['mpsnr_max'])}) | {f(dd)} | {f(a['cam1'])} / {f(a['cam2'])} | {fp} ({fd}) | {sd} | {bl} | {f(a['dyn_active'],0)} |")
open("d1_table.md", "w").write("\n".join(lines)); print("\n".join(lines))
# thresholds
print("\nThree-scene mean d official mPSNR vs baseline:")
for v in V:
    ds = [table[s][v]["mpsnr"] - table[s]["base"]["mpsnr"] for s in SC if table[s][v]]
    print(f"  {v}: {np.mean(ds):+.3f} over {len(ds)} scenes ({', '.join(f'{x:+.2f}' for x in ds)})")
json.dump(table, open("d1_table.json", "w"), indent=1, default=float)

# ---------------- frozen thresholds (06_pilots_designed.md, Pilot 1) ----------------
def spread(s, key, cam=None):
    ks = runs(f"p0_{s}_")
    if cam is None: v = [off(k)[key] for k in ks]
    else: v = [R[k][cam][key] for k in ks if k in R and R[k][cam].get(key) is not None]
    return (max(v) - min(v)) if len(v) > 1 else 0.0
cams = {"apple": ["cam1", "cam2"], "paper-windmill": ["cam1", "cam2"], "spin": ["cam1", "cam2"]}
def d(s, v, key, cam=None):
    a, b = table[s][v], table[s]["base"]
    if cam is None: return a[key] - b[key]
    x, y = a.get(f"{key}_{cam}"), b.get(f"{key}_{cam}")
    return None if x is None or y is None else x - y
out = []
v5o = [d(s, "v5", "mpsnr") for s in SC]; sp = [spread(s, "mpsnr") for s in SC]
out.append(f"T1 three-scene mean official >= +0.4 dB over published: v5 {np.mean(v5o):+.3f} (per scene {', '.join(f'{x:+.2f}' for x in v5o)}; rerun spread {', '.join(f'{x:.2f}' for x in sp)}) -> {'PASS' if np.mean(v5o) >= 0.4 else 'FAIL'}")
fp = {s: [d(s, "v5", "fast_psnr", c) for c in cams[s]] for s in SC}
n_ok = sum(1 for s in SC if all(x is not None and x >= 1.0 for x in fp[s]))
out.append(f"T2 fast-part >= +1 dB on two scenes, both cameras: v5 " + "; ".join(f"{s} {'/'.join(f'{x:+.2f}' for x in fp[s])}" for s in SC) + f" -> {n_ok} scenes -> {'PASS' if n_ok >= 2 else 'FAIL'}")
best = max(["v2", "v3", "v4"], key=lambda v: np.mean([d(s, v, "mpsnr") for s in SC]))
o_vs = np.mean([table[s]["v5"]["mpsnr"] - table[s][best]["mpsnr"] for s in SC])
f_vs = np.mean([np.mean([(table[s]["v5"][f"fast_psnr_{c}"] - table[s][best][f"fast_psnr_{c}"]) for c in cams[s]]) for s in SC])
out.append(f"T3 over the strongest control ({best}): official {o_vs:+.3f} (need +0.2), fast-part {f_vs:+.3f} (need +0.5) -> {'PASS' if o_vs >= 0.2 and f_vs >= 0.5 else 'FAIL'}")
st = {s: [d(s, "v5", "static_psnr", c) for c in cams[s]] for s in SC}
worst = min(x for s in SC for x in st[s] if x is not None)
out.append(f"T4 static-region loss <= 0.1 dB: v5 " + "; ".join(f"{s} {'/'.join(f'{x:+.2f}' for x in st[s])}" for s in SC) + f" -> worst {worst:+.2f} -> {'PASS' if worst >= -0.1 else 'FAIL'}")
k1 = [v for v in ["v2", "v3"] if np.mean([d(s, v, "mpsnr") for s in SC]) >= np.mean(v5o) - 0.05]
out.append(f"K1 a fixed window explains everything: windows {', '.join(f'{v} {np.mean([d(s, v, chr(109)+chr(112)+chr(115)+chr(110)+chr(114)) for s in SC]):+.3f}' for v in ['v2','v3'])} vs v5 {np.mean(v5o):+.3f} -> {'TRIGGERED' if k1 else 'not triggered'} ({', '.join(k1) or 'no window at or above v5'})")
v6o = [d(s, "v6", "mpsnr") for s in SC]
out.append(f"K2 shuffled placebo matches v5: v6 {np.mean(v6o):+.3f} vs v5 {np.mean(v5o):+.3f} (per scene v6-v5 {', '.join(f'{a-b:+.2f}' for a, b in zip(v6o, v5o))}) -> {'TRIGGERED (placebo >= rule)' if np.mean(v6o) >= np.mean(v5o) - max(sp) else 'not triggered'}")
open("d1_verdict.txt", "w").write("\n".join(out)); print("\n" + "\n".join(out))
