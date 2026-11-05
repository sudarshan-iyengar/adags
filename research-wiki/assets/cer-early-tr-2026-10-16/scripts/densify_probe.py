"""ARS wave 1, Multi4D early gate: are memory rows densified less in the early arm? (exploratory diagnostic, never evidence)

Plan: research-wiki/operations/cer-multi4d-early-tr-plan-2026-10-16.md, section 9 (task B), rev 2.

Runs Multi4D's reviewed train.py unmodified (runpy, as m4d_final_render.py does) from the repo given (cer-early@7b30ef8)
and records, without changing any computation, at every call of `densify(max_grad, extent)` on the foreground and the
background bank, before the original runs, per CER label group (0 other, 1 memory, 2 residual; "all" before labels exist):
row count; denom sum and mean; the average gradient xyz_gradient_accum / denom (NaN as 0, as densify does), mean and
quantiles; the rows the clone and the split will select (the same masks, recomputed read-only from the same tensors); and
the same under a deletion sensitivity: accum / (denom - zero_inc), where zero_inc counts, per row, the denom increments
made in batches in which every view that saw the row was a gap view (for a gated memory row exactly those increments
carry zero gradient; denom counts batches in which a row is visible in any view, train.py:593). Rows whose corrected
denominator is 0 get 0 and are counted apart. Coarser counts are kept beside it: increments in all-gap batches, and in
mixed batches. The sensitivity is exact for a gated row; it is not a model of an ungated run.

Per-view visibility: every training render (render_background, render_foreground, render_rawall3pc called by train.py)
with a camera other than cam00 records, per bank, its frame and the bank's per-row visibility (visibility_filter of
render_background / render_foreground; visibility_filter_fg / visibility_filter_bg of render_rawall3pc). At a bank's
add_densification_stats call, the source whose union over views equals the update filter exactly is the one that fed the
statistics; if none matches, the call is counted as unmatched and only the frame counts are made. The records are
cleared after the background bank's call (the iteration's last) and each source keeps at most 16 views.

The per-row counters follow the rows: filtered by every prune_points, reset when densification_postfix resets the
accumulators (both persistent banks reset xyz_gradient_accum and denom for every row there), and zero-padded for rows
appended at the end without it (the residual rows at begin).

Two modes:
  early   train.py's own CER (--cer_scene ... --cer_begin_iter 3000 in the training arguments) labels the rows at begin.
  shadow  CER is off (no --cer_scene). At the end of fine iteration --shadow_iter, right after run_evaluation (where an
          early begin would run), the memory rule of begin is computed with one ungated cer.mass pass and attached as
          cer_labels (and a clock and strength, which Scene2gs_mixed.save reads) to both persistent banks WITHOUT arming
          the gate: cer_gate.ARMED stays False, so no render, no lift and no check reads them; the cer-early propagation
          carries them through clone, split and prune (E3: carrying labels changes nothing in the model).

Snapshots of the row count per label group, in both modes: after the labelling at 3,000 (early: after begin; shadow:
after the shadow labels) and at run_evaluation(--snapshot_iter, default 16000), which precedes the early arm's freeze in
the same iteration. A NaN restart (train.py os.execv) would replace this process with an uninstrumented one, so the
probe makes os.execv raise. <out>/complete.json is written only after train.py returns with both snapshots present.

Writes <out>/densify_probe.jsonl (one record per densify call), <out>/labels.json and <out>/complete.json; refuses a
non-empty <out>.
Usage: python densify_probe.py --repo <tree> --out <dir> --scene <canonical scene> --mode early|shadow
           [--shadow_iter 3000] -- <train.py arguments>
"""
import argparse
import collections
import json
import os
import runpy
import sys

P = {}  # module state, filled by install()


def _note(cam, bank_vis):
    """Record one training render: per bank (id), per source, (frame, visibility)."""
    from utils import cer_gate
    c, f = cer_gate.parse(cam.image_path)
    if c == 0:
        return
    for bank, src, vis in bank_vis:
        P["views"].setdefault(id(bank), {}).setdefault(src, collections.deque(maxlen=16)).append((f, vis))


def _state(m):
    """Per-row counters of bank m, aligned with its rows (zero-padded at the end for rows appended without a postfix)."""
    import torch
    n = m.get_xyz.shape[0]
    st = P["state"].setdefault(id(m), {k: torch.zeros(n, device="cuda") for k in ("zero_inc", "all_gap", "mixed")})
    k = st["zero_inc"].shape[0]
    assert k <= n, "probe counters (%d) longer than the bank (%d)" % (k, n)
    if k < n:
        for key in st:
            st[key] = torch.cat([st[key], torch.zeros(n - k, device="cuda")])
    return st


def _fresh(m):
    import torch
    n = m.get_xyz.shape[0]
    P["state"][id(m)] = {k: torch.zeros(n, device="cuda") for k in ("zero_inc", "all_gap", "mixed")}


def count_increment(m, update_filter, stage=None):
    """The probe's bookkeeping for one add_densification_stats call of bank m (after the original has run)."""
    import torch
    A, B = P["A"], P["B"]
    st = _state(m)
    srcs = P["views"].get(id(m), {})
    chosen = None
    for src, views in srcs.items():
        if views and views[0][1].shape == update_filter.shape:
            union = torch.stack([v for _, v in views]).any(dim=0)
            if torch.equal(union, update_filter):
                chosen = views
                break
    if chosen is None:
        key = "%s/%s" % (type(m).__name__, stage)
        P["unmatched"][key] = P["unmatched"].get(key, 0) + 1
        frames = sorted({f for views in srcs.values() for f, _ in views})
    else:
        frames = [f for f, _ in chosen]
        nongap = [v for f, v in chosen if not (A <= f <= B)]
        seen_outside = torch.stack(nongap).any(dim=0) if nongap else torch.zeros_like(update_filter)
        st["zero_inc"][update_filter & ~seen_outside] += 1
    if frames:
        gap = [A <= f <= B for f in frames]
        if all(gap):
            st["all_gap"][update_filter] += 1
        elif any(gap):
            st["mixed"][update_filter] += 1
    P["views"].pop(id(m), None)


def _q(x, p):
    import torch
    return float(torch.quantile(x, p)) if x.numel() else None


def densify_record(m, max_grad, extent, iteration=None, stage=None):
    """The record of one densify call, from the tensors densify is about to read (read-only)."""
    import torch
    import scene.gaussian_model as gm
    from utils import cer_gate
    with torch.no_grad():
        st = _state(m)
        denom = m.denom[:, 0]
        g = (m.xyz_gradient_accum / m.denom)[:, 0]
        g[g.isnan()] = 0.0
        du = denom - st["zero_inc"]
        assert bool((du >= 0).all()), "zero increments exceed denom"
        gu = torch.where(du > 0, m.xyz_gradient_accum[:, 0] / du.clamp_min(1), torch.zeros_like(du))
        small = torch.max(m.get_scaling, dim=1).values <= m.percent_dense * extent
        clone, split = (g >= max_grad) & small, (g >= max_grad) & ~small
        clone_u, split_u = (gu >= max_grad) & small, (gu >= max_grad) & ~small
        lab = getattr(m, "cer_labels", None)
        groups = {"all": None} if lab is None else {str(c): c for c in (0, 1, 2)}
        out = {}
        for k, c in groups.items():
            sel = torch.ones_like(g, dtype=torch.bool) if c is None else (lab == c)
            seen = sel & (denom > 0)
            seen_u = sel & (du > 0)
            out[k] = {"rows": int(sel.sum()), "rows_seen": int(seen.sum()), "rows_seen_undiluted": int(seen_u.sum()),
                      "rows_zero_corrected_denom": int((seen & ~(du > 0)).sum()),
                      "denom_sum": float(denom[sel].sum()), "denom_mean_seen": float(denom[seen].mean()) if seen.any() else None,
                      "zero_inc_sum": float(st["zero_inc"][sel].sum()), "all_gap_sum": float(st["all_gap"][sel].sum()),
                      "mixed_sum": float(st["mixed"][sel].sum()),
                      "grad_mean_seen": float(g[seen].mean()) if seen.any() else None,
                      "grad_q50_seen": _q(g[seen], 0.5), "grad_q90_seen": _q(g[seen], 0.9),
                      # the same rows as grad_mean_seen (denom > 0), corrected zeros kept as 0
                      "grad_u_mean_seen": float(gu[seen].mean()) if seen.any() else None,
                      "frac_ge_thr_seen": float((g[seen] >= max_grad).float().mean()) if seen.any() else None,
                      "clone": int((clone & sel).sum()), "split": int((split & sel).sum()),
                      "clone_undiluted": int((clone_u & sel).sum()), "split_undiluted": int((split_u & sel).sum())}
        return {"iteration": iteration, "stage": stage, "bank": "bg" if isinstance(m, gm.GaussianModel) else "fg",
                "threshold": float(max_grad), "extent": float(extent), "percent_dense": float(m.percent_dense),
                "armed": bool(cer_gate.ARMED), "labelled": lab is not None, "groups": out,
                "masks": (clone, split)}


def install(scene_dir, mode, out_dir, shadow_iter=3000, log=None, snapshot_iter=16000):
    """Install the probe's wrappers (renders, add_densification_stats, prune_points, densification_postfix, densify,
    run_evaluation, cer.freeze). Returns the state dict."""
    import torch
    import gaussian_renderer as grend
    import scene.gaussian_model as gm
    from scene import cer
    from utils import cer_gate
    import utils.eval_utils as eu

    A, B, _ = cer.event(scene_dir)
    P.update({"A": A, "B": B, "views": {}, "state": {}, "unmatched": {}, "log": log,
              "labels": {"mode": mode, "A": A, "B": B}, "out": out_dir})

    orig_bg, orig_fg, orig_all = grend.render_background, grend.render_foreground, grend.render_rawall3pc

    def render_background(viewpoint_camera, pc, *args, **kwargs):
        r = orig_bg(viewpoint_camera, pc, *args, **kwargs)
        _note(viewpoint_camera, [(pc, "render_background", r["visibility_filter"])])
        return r

    def render_foreground(viewpoint_camera, pc, *args, **kwargs):
        r = orig_fg(viewpoint_camera, pc, *args, **kwargs)
        _note(viewpoint_camera, [(pc, "render_foreground", r["visibility_filter"])])
        return r

    def render_rawall3pc(viewpoint_camera, pc_foreground, pc_background, *args, **kwargs):
        r = orig_all(viewpoint_camera, pc_foreground, pc_background, *args, **kwargs)
        _note(viewpoint_camera, [(pc_foreground, "rawall_fg", r["visibility_filter_fg"]),
                                 (pc_background, "rawall_bg", r["visibility_filter_bg"])])
        return r

    # train.py imports these names from gaussian_renderer when runpy executes it; cer and eval_utils keep the originals
    grend.render_background, grend.render_foreground, grend.render_rawall3pc = render_background, render_foreground, render_rawall3pc

    orig_add = gm.PersistentBranchMixin.add_densification_stats

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        orig_add(self, viewspace_point_tensor, update_filter)
        count_increment(self, update_filter, sys._getframe(1).f_locals.get("stage"))
        if isinstance(self, gm.GaussianModel):  # the background's call is the iteration's last
            P["views"].clear()

    gm.PersistentBranchMixin.add_densification_stats = add_densification_stats

    for cls in (gm.GaussianModel, gm.GaussianModel_dynamic):
        orig_prune, orig_post = cls.prune_points, cls.densification_postfix

        def prune_points(self, mask, _orig=orig_prune):
            st = _state(self)
            _orig(self, mask)
            keep = ~mask
            for key in st:
                st[key] = st[key][keep]
            assert st["zero_inc"].shape[0] == self.get_xyz.shape[0]

        def densification_postfix(self, *args, _orig=orig_post, **kwargs):
            _orig(self, *args, **kwargs)  # resets xyz_gradient_accum and denom for every row
            _fresh(self)

        cls.prune_points, cls.densification_postfix = prune_points, densification_postfix

    orig_densify = gm.PersistentBranchMixin.densify

    def densify(self, max_grad, extent):
        caller = sys._getframe(1).f_locals
        rec = densify_record(self, max_grad, extent, caller.get("iteration"), caller.get("stage"))
        rec.pop("masks")
        rec["unmatched_so_far"] = dict(P["unmatched"])
        if P["log"] is not None:
            P["log"].write(json.dumps(rec) + "\n")
            P["log"].flush()
        return orig_densify(self, max_grad, extent)

    gm.PersistentBranchMixin.densify = densify

    orig_eval = eu.run_evaluation

    def run_evaluation(iteration, scene, foreground_gaussians, background_gaussians, transient_gaussian,
                       pipeline_config, optimization_params, expname, stage, current_best_psnr=-1.0):
        res = orig_eval(iteration, scene, foreground_gaussians, background_gaussians, transient_gaussian,
                        pipeline_config, optimization_params, expname, stage, current_best_psnr)
        if mode == "shadow" and stage == "fine" and int(iteration) == shadow_iter:
            assert not cer_gate.ARMED and not scene.cer_scene, "shadow labelling with CER on"
            assert not hasattr(background_gaussians, "cer_labels"), "shadow labels attached twice"
            fg, bg = foreground_gaussians, background_gaussians
            cer.check_finite(fg, bg)
            scene.cer_scene = scene_dir  # read by cer._region only, restored before training continues
            try:
                inside, total = cer.mass(fg, bg, transient_gaussian, pipeline_config, cer._region(scene, range(0, A)))
            finally:
                scene.cer_scene = ""
            memory = [(t > 0) & (i >= 0.5 * t) for i, t in zip(inside[:2], total[:2])]
            for m, mem in zip((fg, bg), memory):
                m.cer_labels, m.cer_clock, m.cer_q = mem.to(torch.uint8), (A, B), 1.0
            assert not cer_gate.ARMED
            P["labels"].update({"shadow_iter": int(iteration),
                                "memory_rows": {"fg": int(memory[0].sum()), "bg": int(memory[1].sum())},
                                "memory_share_pregap": ((inside[0][memory[0]].sum() + inside[1][memory[1]].sum())
                                                        / sum(x.sum() for x in inside)).item(),
                                "rows": {"fg": fg.get_xyz.shape[0], "bg": bg.get_xyz.shape[0]}})
            _snapshot("snapshot_labelling", fg, bg, iteration)
            print("[densify_probe] shadow labels at fine %d: %s" % (iteration, P["labels"]["memory_rows"]), flush=True)
        if stage == "fine" and int(iteration) == snapshot_iter:
            _snapshot("snapshot_%d" % snapshot_iter, foreground_gaussians, background_gaussians, iteration)
        return res

    eu.run_evaluation = run_evaluation  # train.py imports it from utils.eval_utils when runpy executes it

    orig_begin = cer.begin

    def begin(fg, bg, tr, scene, pipe, opt, iteration):
        r = orig_begin(fg, bg, tr, scene, pipe, opt, iteration)
        _snapshot("snapshot_labelling", fg, bg, iteration)
        return r

    cer.begin = begin  # train.py calls cer.begin through the module

    def execv(*args, **kwargs):
        raise SystemExit("densify_probe: train.py asked for a NaN restart (os.execv); this diagnostic run is invalid")

    os.execv = execv

    orig_freeze = cer.freeze

    def freeze(fg, bg, tr, scene, pipe, opt, iteration):
        P["labels"].update({"freeze_iter": int(iteration), "rows_at_freeze": {
            n: {str(c): int((m.cer_labels == c).sum()) for c in (0, 1, 2)} for n, m in (("fg", fg), ("bg", bg))}})
        _dump_labels()
        return orig_freeze(fg, bg, tr, scene, pipe, opt, iteration)

    cer.freeze = freeze  # train.py calls cer.freeze through the module
    return P


def _snapshot(key, fg, bg, iteration):
    """Rows per label group of both persistent banks (None where a bank carries no labels)."""
    def groups(m):
        lab = getattr(m, "cer_labels", None)
        return None if lab is None else {str(c): int((lab == c).sum()) for c in (0, 1, 2)}
    P["labels"][key] = {"iteration": int(iteration), "fg": groups(fg), "bg": groups(bg),
                        "rows": {"fg": fg.get_xyz.shape[0], "bg": bg.get_xyz.shape[0]}}
    _dump_labels()


def _dump_labels():
    if P.get("out"):
        P["labels"]["unmatched"] = dict(P["unmatched"])
        json.dump(P["labels"], open(os.path.join(P["out"], "labels.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", required=True, help="canonical ARS scene dir (event_spec.json, gt/identity)")
    ap.add_argument("--mode", required=True, choices=("early", "shadow"))
    ap.add_argument("--shadow_iter", type=int, default=3000)
    ap.add_argument("--snapshot_iter", type=int, default=16000)
    ap.add_argument("train_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    train_args = a.train_args[1:] if a.train_args[:1] == ["--"] else a.train_args
    if a.mode == "shadow":
        assert "--cer_scene" not in train_args, "shadow mode runs with CER off"
    else:
        assert "--cer_scene" in train_args and "--cer_begin_iter" in train_args, "early mode needs train.py's own CER"
    if os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit("refusing: %s is not empty" % a.out)
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, a.repo)
    os.chdir(a.repo)
    log = open(os.path.join(a.out, "densify_probe.jsonl"), "w")
    install(a.scene, a.mode, a.out, a.shadow_iter, log, a.snapshot_iter)
    sys.argv = ["train.py"] + train_args
    runpy.run_path("train.py", run_name="__main__")
    log.close()
    _dump_labels()
    if a.mode == "shadow" and "shadow_iter" not in P["labels"]:
        raise SystemExit("densify_probe: the shadow labels were never attached (no fine evaluation at %d)" % a.shadow_iter)
    missing = [k for k in ("snapshot_labelling", "snapshot_%d" % a.snapshot_iter)
               if k not in P["labels"] or P["labels"][k]["bg"] is None or P["labels"][k]["fg"] is None]
    if missing:
        raise SystemExit("densify_probe: snapshots missing or unpopulated: %s" % missing)
    fine_unmatched = {k: v for k, v in P["unmatched"].items() if not k.endswith("/coarse")}
    if fine_unmatched:
        raise SystemExit("densify_probe: fine-stage statistics calls without a matching visibility source: %s" % fine_unmatched)
    json.dump({"complete": True, "mode": a.mode, "unmatched": P["unmatched"], "snapshot_iter": a.snapshot_iter},
              open(os.path.join(a.out, "complete.json"), "w"))
    print("[densify_probe] done: %s" % json.dumps(P["labels"]), flush=True)


if __name__ == "__main__":
    main()
