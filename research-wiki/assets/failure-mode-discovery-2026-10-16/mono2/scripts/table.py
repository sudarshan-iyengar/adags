"""fmd mono2: collect scores (ghost_ring_real, whole-frame PSNR, persistence) into a markdown table + JSON."""
import json, os
S = "D:/adags-ars/fmd/scores"
M = [("Shape of Motion", "som"), ("MoSca (published eval: pose-aligned)", "mosca"), ("MoSca (no alignment)", "moscaNA"),
     ("Multi4D monocular (sec. 24)", "m4d")]
SC = ["cut_roasted_beef", "flame_steak", "sear_steak"]; CAMS = ["cam00", "cam05", "cam17"]
rows, allj = [], {}
for name, p in M:
    for sc in SC:
        L = f"{p}_monoreal_{sc}_rep1"
        f = f"{S}/real_{L}.json"
        if not os.path.exists(f): continue
        g = json.load(open(f)); w = json.load(open(f"{S}/wholeframe_{L}.json")) if os.path.exists(f"{S}/wholeframe_{L}.json") else {}
        q = json.load(open(f"{S}/persist_{L}.json")) if os.path.exists(f"{S}/persist_{L}.json") else {}
        allj[L] = {"ghost_ring": g, "wholeframe": w, "persist": q}
        j = lambda k, d=g: " / ".join(f"{d[c][k]:.1f}" if isinstance(d.get(c, {}).get(k), float) else "-" for c in CAMS)
        pct = lambda k, d: " / ".join(f"{100 * d[c][k]:.0f}" if c in d else "-" for c in CAMS)
        rows.append(f"| {name} | {sc} | **{j('ghost_minus_ring')}** | {j('ghost')} vs {j('ring')} | {j('obj')} | {j('pre_obj')} | "
                    f"{pct('closer_to_object', g)} % | {pct('closer_to_own_object', q) if q else '-'} % | "
                    f"{j('render_change', q) if q else '-'} (truth {j('truth_change', q) if q else '-'}) | "
                    f"{j('wholeframe_psnr_40_99', w) if w else '-'} |")
hdr = ("| Method | Scene | Ghost minus ring, cam00 / 05 / 17 (dB) | Ghost vs ring (dB) | Render vs departed bottle, gap (dB) | "
       "Same, before the gap (dB) | Vacated px closer to the bottle | Closer to the method's own pre-gap render | "
       "Render change in the region, gap vs pre-gap (dB) | Whole-frame PSNR, frames 40-99 (dB) |\n|" + "---|" * 10)
out = hdr + "\n" + "\n".join(rows)
open("D:/adags-ars/fmd/mono2/table.md", "w").write(out); json.dump(allj, open(f"{S}/mono2_summary.json", "w"), indent=1)
print(out)
