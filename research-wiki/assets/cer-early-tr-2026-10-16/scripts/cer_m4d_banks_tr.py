"""EXPLORATORY copy of cer_m4d_banks.py for the early gate with transient copies (cer_transient_gate; plan
research-wiki/operations/cer-multi4d-early-tr-plan-2026-10-16.md, D8). The one difference: when cer_state.pt holds
`tr_labels`, they are re-attached to the transient bank with the clock and the strength, and cer_gate.TRANSIENT is set for
that run (and reset to False before every run, since one job may hold runs of different arms), so the `full` render still
reproduces the saved final render. cer_m4d_banks.py itself is unchanged. Run it from the root of repo/multi4d-cer-early-tr.

ARS wave 1, CER on Multi4D: what each bank paints on the held-out camera (descriptive, looking material, never a score).

For each Multi4D run given (base or CER), reload the saved model (iteration --iteration, default 20000) and render cam00 at the frames asked
for, three ways, on a black background with the same call the final render makes (render_rawall3pc, stage fine):
  full        all three banks (CER gate as trained);
  persistent  foreground + background banks alone (transient opacity set to 0);
  transient   transient bank alone (foreground + background opacity factor 0, after the CER gate).
Saves the three as 8-bit PNGs (torchvision's rounding) plus each bank-alone accumulated alpha, and per frame the mean
alpha of each bank alone inside the ghost mask (P1 region) and the target mask.

Faithfulness check, required: the `full` render must reproduce the run's saved final render
(<run>/final/iteration_20000/render/<f>.png) on every frame; the per-frame number of differing 8-bit values and the
largest difference are recorded, and the run is marked unfaithful if any value differs by more than 1.

Reload: a CER model refuses Multi4D's load path (cer_state.pt guard), by design. Here the point clouds are linked into
a scratch directory WITHOUT cer_state.pt, loaded with the census loader (test_cer.py LOADER), and the saved labels,
clock and strength are re-attached from cer_state.pt; the gate is then armed. Nothing is written into a run.
Run from the root of the tree that trained the runs, with its TORCH_EXTENSIONS_DIR.
Usage: python cer_m4d_banks_tr.py --out <dir> --frames 56 63 75 87 92 --runs <run dir> ... [--scratch <dir>]
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

S = "/leonardo_scratch/large/userexternal/siyengar/ars/wave1"
sys.path.insert(0, os.getcwd())

import mmcv  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams  # noqa: E402
from utils.params_utils import merge_hparams  # noqa: E402
from scene import GaussianModel, GaussianModel_dynamic, Scene2gs_mixed  # noqa: E402
from scene.gaussian_model import GaussianModelTransient  # noqa: E402
from gaussian_renderer import render_rawall3pc  # noqa: E402
from utils import cer_gate  # noqa: E402

_orig_apply = cer_gate.apply
KEEP = {}  # id(bank) -> opacity multiplier applied after the CER gate (bank-alone renders)


def _apply(pc, cam, opacity):
    out = _orig_apply(pc, cam, opacity)
    k = KEEP.get(id(pc))
    return out if k is None else out * k


cer_gate.apply = _apply  # gaussian_renderer looks cer_gate.apply up at call time


def link_model(run_dir, scratch, it):
    """scratch/model/point_cloud/iteration_<it> holding links to every saved file except cer_state.pt."""
    src = os.path.join(run_dir, "model", "point_cloud", "iteration_%d" % it)
    dst = os.path.join(scratch, "model", "point_cloud", "iteration_%d" % it)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f != "cer_state.pt" and not os.path.exists(os.path.join(dst, f)):
            os.symlink(os.path.join(src, f), os.path.join(dst, f))
    return scratch


def load_model(run_dir, tree, it):
    """The census loader of test_cer.py (Multi4D's own load path, transient time-SH degree at its maximum)."""
    parser = argparse.ArgumentParser()
    lp, op, pp, hp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser), ModelHiddenParams(parser)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    model_dir = os.path.join(run_dir, "model")
    args = parser.parse_args(["-s", tree, "--model_path", model_dir, "--configs", "arguments/dynerf.py"])
    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    d, o, p, h = lp.extract(args), op.extract(args), pp.extract(args), hp.extract(args)
    d.sh_degree = 3
    fg = GaussianModel_dynamic(d.sh_degree, h)
    bg = GaussianModel(d.sh_degree)
    tr = GaussianModelTransient(d.sh_degree, gaussian_dim=4,
                                time_duration=list(getattr(o, "transient_time_duration", [0.0, 10.0])),
                                rot_4d=True, sh_degree_t=2)
    d.model_path = model_dir
    scene = Scene2gs_mixed(d, fg, load_iteration=model_dir, gaussians_second=bg, gaussians_transient=tr)
    assert scene.loaded_iter == it, (scene.loaded_iter, it)
    tr.active_sh_degree_t = tr.max_sh_degree_t
    return p, fg, bg, tr, scene


def u8(img):
    """torchvision.utils.save_image's rounding: [3,H,W] float -> [H,W,3] uint8."""
    return img.clamp(0, 1).mul(255).add(0.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def mask(scene_name, kind, f):
    p = "%s/%s/gt/masks/%s/%05d.png" % (S, scene_name, kind, f)
    return (np.asarray(Image.open(p).convert("L")) > 127) if os.path.isfile(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--iteration", type=int, default=20000, help="saved fine iteration to load (point_cloud/iteration_<n>, final/iteration_<n>)")
    a = ap.parse_args()
    black = torch.zeros(3, device="cuda")
    summary = []
    for run in a.runs:
        t0 = time.time()
        label = os.path.basename(run.rstrip("/"))
        prov = dict(l.strip().split("=", 1) for l in open(os.path.join(run, "provenance.txt")) if "=" in l)
        sc, tree = prov["scene"], prov["tree"]
        state_p = os.path.join(run, "model", "point_cloud", "iteration_%d" % a.iteration, "cer_state.pt")
        is_cer = os.path.isfile(state_p)
        cer_gate.ARMED = False
        cer_gate.TRANSIENT = False  # reset before every run: one job may hold runs of different arms
        p, fg, bg, tr, scene = load_model(link_model(run, os.path.join(a.scratch, label), a.iteration), tree, a.iteration)
        if is_cer:
            st = torch.load(state_p)
            for m, lab in ((fg, st["fg_labels"]), (bg, st["bg_labels"])):
                assert lab.shape[0] == m.get_xyz.shape[0], (label, lab.shape, m.get_xyz.shape)
                m.cer_labels, m.cer_clock, m.cer_q = lab.cuda(), tuple(st["clock"]), st["q"]
            tr.active_sh_degree_t = st["active_sh_degree_t"]
            if "tr_labels" in st:  # the transient gate: labels, clock and strength of the transient bank
                assert st["tr_labels"].shape[0] == tr.get_xyz.shape[0], (label, st["tr_labels"].shape, tr.get_xyz.shape)
                tr.cer_labels, tr.cer_clock, tr.cer_q = st["tr_labels"].cuda(), tuple(st["clock"]), st["q"]
                cer_gate.TRANSIENT = True
            cer_gate.ARMED = True
        od = os.path.join(a.out, label)
        os.makedirs(od, exist_ok=True)
        rec = {"run": run, "label": label, "scene": sc, "cer": is_cer, "frames": {},
               "rows": {"fg": fg.get_xyz.shape[0], "bg": bg.get_xyz.shape[0], "tr": tr.get_xyz.shape[0]}}
        if is_cer:
            rec["cohorts"] = {"bg_memory": int((bg.cer_labels == 1).sum()), "bg_residual": int((bg.cer_labels == 2).sum()),
                              "fg_memory": int((fg.cer_labels == 1).sum()), "clock": list(bg.cer_clock), "q": bg.cer_q,
                              "transient_gate": cer_gate.TRANSIENT,
                              "tr_memory": int((tr.cer_labels == 1).sum()) if cer_gate.TRANSIENT else None}
        tr_op = tr._opacity.data.clone()
        faithful = True
        with torch.no_grad():
            for f in a.frames:
                cam = scene.getTestCameras()[f]
                assert int(cam.image_name) == f, (cam.image_name, f)
                fr = {}
                KEEP.clear()
                full = render_rawall3pc(cam, fg, bg, tr, p, bg_color=black, stage="fine")
                img = u8(full["render"])
                saved = np.asarray(Image.open(os.path.join(run, "final", "iteration_%d" % a.iteration, "render", "%d.png" % f)).convert("RGB"))
                diff = np.abs(img.astype(int) - saved.astype(int))
                fr["vs_saved"] = {"n_values_differ": int((diff > 0).sum()), "max_abs_diff": int(diff.max())}
                faithful &= int(diff.max()) <= 1
                Image.fromarray(img).save(os.path.join(od, "full_%03d.png" % f))
                # persistent banks alone: transient opacity 0
                tr._opacity.data.fill_(-1e4)
                r = render_rawall3pc(cam, fg, bg, tr, p, bg_color=black, stage="fine")
                tr._opacity.data.copy_(tr_op)
                Image.fromarray(u8(r["render"])).save(os.path.join(od, "persistent_%03d.png" % f))
                alpha_p = r["alpha"].reshape(r["render"].shape[1:]).clamp(0, 1).cpu().numpy()
                # transient bank alone: foreground and background opacity 0 after the gate
                KEEP[id(fg)], KEEP[id(bg)] = 0.0, 0.0
                r = render_rawall3pc(cam, fg, bg, tr, p, bg_color=black, stage="fine")
                KEEP.clear()
                Image.fromarray(u8(r["render"])).save(os.path.join(od, "transient_%03d.png" % f))
                alpha_t = r["alpha"].reshape(r["render"].shape[1:]).clamp(0, 1).cpu().numpy()
                for nm, al in (("persistent", alpha_p), ("transient", alpha_t)):
                    Image.fromarray((al * 255 + 0.5).astype(np.uint8)).save(os.path.join(od, "alpha_%s_%03d.png" % (nm, f)))
                for kind in ("ghost", "target"):
                    m = mask(sc, kind, f)
                    if m is not None and m.any():
                        fr["alpha_in_%s" % kind] = {"persistent": float(alpha_p[m].mean()),
                                                     "transient": float(alpha_t[m].mean()), "pixels": int(m.sum())}
                rec["frames"][str(f)] = fr
        rec["faithful_to_saved_render"] = bool(faithful)
        rec["seconds"] = round(time.time() - t0, 1)
        json.dump(rec, open(os.path.join(od, "banks.json"), "w"), indent=1)
        summary.append({k: rec[k] for k in ("label", "faithful_to_saved_render", "seconds")})
        print(json.dumps(summary[-1]), {f: v["vs_saved"] for f, v in rec["frames"].items()}, flush=True)
        del p, fg, bg, tr, scene
        torch.cuda.empty_cache()
    json.dump(summary, open(os.path.join(a.out, "summary_%d.json" % os.getpid()), "w"), indent=1)


if __name__ == "__main__":
    main()
