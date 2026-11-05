"""Which transient rows paint the object interior before the gap, and when are they centred in time? (descriptive, 2026-10-16)

Tests the reading that base Multi4D's transient "detail" on the object is the temporal tail of a gap occluder: for each
run given, at cam00 pre-gap frames (default 40 and 57), the per-row transient contribution mass sum(alpha*T) inside the
target interior (target mask eroded 4 px) by the CER instrument (scene/cer.py mass, unarmed for base runs, the trained gate
re-attached for CER runs, as cer_m4d_banks.py does); each transient row's time centre in frames (t * 300 / duration); and
its contribution at the gap frame 74 inside the gap region (the target mask at frame A-1 = 59). Reports, mass-weighted over
the interior-painting transient rows: the shares centred before the gap (< 60), in it (60-89) and after it (> 89); the share
that also paints the gap region at frame 74; and the transient share of all interior mass.
Run from the root of a CER Multi4D tree (here repo/multi4d-cer-early-tr, read-only) with its own extensions directory.
Usage: python transient_time_interior.py --out <json> --scratch <dir> --runs <run dir> ... [--frames 40 57 --gap_frame 74]
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

S = "/leonardo_scratch/large/userexternal/siyengar/ars/wave1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("cer_m4d_banks", os.path.join(HERE, "cer_m4d_banks.py"))
banks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(banks)  # defines link_model / load_model; wraps cer_gate.apply with a no-op multiplier map
from scene import cer  # noqa: E402
from utils import cer_gate  # noqa: E402


def scene_of(run):
    b = os.path.basename(run.rstrip("/"))
    return b.split("_")[2]


def mask(sc, f, erode):
    m = np.asarray(Image.open("%s/%s/gt/masks/target/%05d.png" % (S, sc, f)).convert("L")) > 127
    if erode:
        m = ndimage.binary_erosion(m, iterations=erode)
    return torch.from_numpy(m).float().cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[40, 57])
    ap.add_argument("--gap_frame", type=int, default=74)
    ap.add_argument("--iteration", type=int, default=20000)
    a = ap.parse_args()
    res = {}
    for run in a.runs:
        sc = scene_of(run)
        label = os.path.basename(run.rstrip("/"))
        tree = "%s/trees/multi4d/%s" % (S, sc)
        cer_gate.ARMED = False
        p, fg, bg, tr, scene = banks.load_model(banks.link_model(run, os.path.join(a.scratch, label), a.iteration), tree, a.iteration)
        state_p = os.path.join(run, "model", "point_cloud", "iteration_%d" % a.iteration, "cer_state.pt")
        if os.path.exists(state_p):
            st = torch.load(state_p)
            for m, lab in ((fg, st["fg_labels"]), (bg, st["bg_labels"])):
                m.cer_labels, m.cer_clock, m.cer_q = lab.cuda(), tuple(st["clock"]), st["q"]
            tr.active_sh_degree_t = st["active_sh_degree_t"]
            cer_gate.ARMED = True
        cams = {cer_gate.frame(c): c for c in scene.getTestCameras()}
        dur = tr.time_duration[1] - tr.time_duration[0]
        centre = ((tr.get_t[:, 0] - tr.time_duration[0]) / dur * 300.0).detach()
        g_in, _ = cer.mass(fg, bg, tr, p, [(cams[a.gap_frame], mask(sc, 59, 0))])
        paints_gap = g_in[2] > 0
        rec = {"scene": sc, "cer": os.path.exists(state_p), "rows_tr": int(tr.get_xyz.shape[0]), "frames": {}}
        for f in a.frames:
            inside, _ = cer.mass(fg, bg, tr, p, [(cams[f], mask(sc, f, 4))])
            w = inside[2]
            tot_tr, tot = float(w.sum()), float(sum(x.sum() for x in inside))
            out = {"transient_share_of_interior_mass": tot_tr / tot if tot > 0 else None}
            if tot_tr > 0:
                out.update({"centre_before_gap": float(w[centre < 60].sum()) / tot_tr,
                            "centre_in_gap": float(w[(centre >= 60) & (centre <= 89)].sum()) / tot_tr,
                            "centre_after_gap": float(w[centre > 89].sum()) / tot_tr,
                            "also_paints_gap_region_at_%d" % a.gap_frame: float(w[paints_gap].sum()) / tot_tr,
                            "mean_abs_centre_offset_frames": float((w * (centre - f).abs()).sum()) / tot_tr,
                            "rows_with_mass": int((w > 0).sum())})
            rec["frames"][str(f)] = out
        res[label] = rec
        print(label, json.dumps(rec), flush=True)
        del p, fg, bg, tr, scene
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
