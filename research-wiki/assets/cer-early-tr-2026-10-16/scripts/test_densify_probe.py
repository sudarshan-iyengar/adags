"""Test of the task-B densification probe (plan cer-multi4d-early-tr-plan-2026-10-16.md section 9, "Test of the probe first").

Run from the root of repo/multi4d-cer-early on one GPU (debug QOS), with the early extensions directory:
    python test_densify_probe.py --probe <agent-control/ars/densify_probe.py> --run <wave-1 Multi4D run dir> \
        --tree <multi4d tree> --scene <canonical scene> --out <json>
(a) zero-increment counting: an all-gap batch, a mixed batch where a row is seen only in the gap view, a mixed batch where
    it is seen in both, give zero_inc 1, 1, 0 for that row; a real render_background pair (non-gap then gap view) is
    recorded under its source and matched to the update filter;
(b) the counters stay aligned through prune_points, densification_postfix and an end append;
(c) on both banks, the recomputed clone and split masks are exactly the rows densify copies and splits (tagged rows);
(d) on both banks, the wrapped calls leave every model tensor, the optimizer state and the RNG bit-identical to a twin run
    with the original class methods restored;
(e) a zero corrected denominator gives 0 and is counted.
Exits non-zero if any check fails.
"""
import argparse
import contextlib
import copy
import importlib.util
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--probe", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--tree", required=True)
ap.add_argument("--scene", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
import test_cer as T  # noqa: E402  (the census loader; this tree's code)
import scene.gaussian_model as gm  # noqa: E402
import gaussian_renderer as grend  # noqa: E402
from utils import cer_gate  # noqa: E402

R = []


def check(name, ok, detail=""):
    R.append({"name": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (" :: " + str(detail) if detail else ""), flush=True)


SLOTS = {"add": (gm.PersistentBranchMixin, "add_densification_stats"), "densify": (gm.PersistentBranchMixin, "densify"),
         "prune_bg": (gm.GaussianModel, "prune_points"), "post_bg": (gm.GaussianModel, "densification_postfix"),
         "prune_fg": (gm.GaussianModel_dynamic, "prune_points"), "post_fg": (gm.GaussianModel_dynamic, "densification_postfix")}
ORIG = {k: getattr(owner, attr) for k, (owner, attr) in SLOTS.items()}  # before the probe is installed

d, o, p, fg, bg, tr, scene = T.load_model(a.run, a.tree)
spec = importlib.util.spec_from_file_location("densify_probe", a.probe)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
P = probe.install(a.scene, "early", None)
A, B = P["A"], P["B"]
cams = list(scene.getTrainCameras())
gap_cam = next(c for c in cams if A <= cer_gate.frame(c) <= B)
# a different camera, so that the background rows seen by the two views differ (background rows have no time)
non_cam = next(c for c in cams if cer_gate.frame(c) < A and cer_gate.parse(c.image_path)[0] != cer_gate.parse(gap_cam.image_path)[0])
fa, fn = cer_gate.frame(gap_cam), cer_gate.frame(non_cam)
N = bg.get_xyz.shape[0]


def vis(rows):
    v = torch.zeros(N, dtype=torch.bool, device="cuda")
    v[rows] = True
    return v


# (a) hand-built batches; row 0 is the row under test, row 1 is a bystander seen only in the non-gap view
probe._fresh(bg)
cases = [("all-gap", [(fa, vis([0])), (fa, vis([0]))], 1), ("mixed, seen only in gap", [(fa, vis([0])), (fn, vis([1]))], 1),
         ("mixed, seen in both", [(fa, vis([0])), (fn, vis([0, 1]))], 0)]
for name, views, want in cases:
    probe._fresh(bg)
    P["views"][id(bg)] = {"hand": list(views)}
    filt = torch.stack([v for _, v in views]).any(dim=0)
    probe.count_increment(bg, filt)
    st = probe._state(bg)
    got = int(st["zero_inc"][0])
    check("a %s: zero_inc row 0 = %d" % (name, want), got == want, {"got": got, "all_gap": int(st["all_gap"][0]),
                                                                       "mixed": int(st["mixed"][0]),
                                                                       "bystander_zero_inc": int(st["zero_inc"][1])})
# real renders through the wrapped render_background (the phase-2 standalone background rasterizer)
probe._fresh(bg)
P["views"].clear()
black = torch.zeros(3, device="cuda")
with torch.no_grad():
    pm = lambda c: torch.ones((int(c.image_height), int(c.image_width)), dtype=torch.float32, device="cuda")
    r_non = grend.render_background(non_cam, bg, p, black, stage="coarse", prob_mask=pm(non_cam))
    r_gap = grend.render_background(gap_cam, bg, p, black, stage="coarse", prob_mask=pm(gap_cam))
srcs = P["views"].get(id(bg), {})
check("a real renders recorded under render_background", list(srcs) == ["render_background"] and len(srcs["render_background"]) == 2,
      {k: len(v) for k, v in srcs.items()})
v_non, v_gap = r_non["visibility_filter"], r_gap["visibility_filter"]
filt = v_non | v_gap
probe.count_increment(bg, filt)
st = probe._state(bg)
want = (v_gap & ~v_non)
check("a real renders: zero_inc = seen only in the gap view", torch.equal(st["zero_inc"] > 0, want) and int(want.sum()) > 0
      and not P["unmatched"], {"rows_gap_only": int(want.sum()), "unmatched": P["unmatched"]})

# (b) alignment through prune, postfix and an end append (counters tagged with their row index)
probe._fresh(bg)
st = probe._state(bg)
st["zero_inc"] = torch.arange(N, device="cuda", dtype=torch.float32)
g = torch.Generator(device="cuda").manual_seed(1)
mask = torch.rand(N, device="cuda", generator=g) < 0.01
bg.prune_points(mask)
st = probe._state(bg)
check("b prune keeps the counters aligned", torch.equal(st["zero_inc"], torch.arange(N, device="cuda", dtype=torch.float32)[~mask])
      and st["zero_inc"].shape[0] == bg.get_xyz.shape[0], {"pruned": int(mask.sum())})
n1 = bg.get_xyz.shape[0]
k = 5
with torch.no_grad():
    new = {"xyz": bg._xyz[:k], "f_dc": bg._features_dc[:k], "f_rest": bg._features_rest[:k], "opacity": bg._opacity[:k],
           "scaling": bg._scaling[:k], "rotation": bg._rotation[:k]}
t = bg.cat_tensors_to_optimizer(new)
bg._xyz, bg._features_dc, bg._features_rest = t["xyz"], t["f_dc"], t["f_rest"]
bg._opacity, bg._scaling, bg._rotation = t["opacity"], t["scaling"], t["rotation"]
bg.max_radii2D = torch.cat([bg.max_radii2D, torch.zeros(k, device="cuda")])
bg.xyz_gradient_accum = torch.cat([bg.xyz_gradient_accum, torch.zeros((k, 1), device="cuda")])
bg.denom = torch.cat([bg.denom, torch.zeros((k, 1), device="cuda")])
st = probe._state(bg)
check("b end append is zero-padded", st["zero_inc"].shape[0] == n1 + k and bool((st["zero_inc"][n1:] == 0).all())
      and torch.equal(st["zero_inc"][:n1], torch.arange(N, device="cuda", dtype=torch.float32)[~mask]))

# (c) and (e), both banks: forced statistics; rows tagged by a unique id in _opacity (clone and split copy it exactly and
# the selection does not read it); the final tags must be [orig[~split], orig[clone], orig[split] x 2]
thr, ext = 0.0004, scene.cameras_extent
for name, m in (("bg", bg), ("fg", fg)):
    n0 = m.get_xyz.shape[0]
    g = torch.Generator(device="cuda").manual_seed(2)
    m.denom = torch.randint(1, 5, (n0, 1), device="cuda", generator=g).float()
    m.xyz_gradient_accum = torch.rand((n0, 1), device="cuda", generator=g) * 0.001 * m.denom
    probe._fresh(m)
    st = probe._state(m)
    st["zero_inc"][:10] = m.denom[:10, 0]  # rows 0-9: every increment a zero increment, corrected denominator 0
    rec = probe.densify_record(m, thr, ext)
    clone, split = rec["masks"]
    gs = rec["groups"]["all"]
    if name == "bg":
        check("e zero corrected denominator counted", gs["rows_zero_corrected_denom"] == 10, gs["rows_zero_corrected_denom"])
        ref = (m.xyz_gradient_accum / m.denom)[:, 0]
        ref[ref.isnan()] = 0
        seen = m.denom[:, 0] > 0
        check("e the ordinary average over the rows with denom > 0", gs["rows_seen"] == int(seen.sum()) and
              abs(gs["grad_mean_seen"] - float(ref[seen].mean())) < 1e-9, gs["rows_seen"])
    check("c %s clone and split both non-empty" % name, int(clone.sum()) > 0 and int(split.sum()) > 0,
          (int(clone.sum()), int(split.sum())))
    tags = torch.arange(n0, device="cuda", dtype=torch.float32)
    assert n0 < 2 ** 24
    with torch.no_grad():
        m._opacity.data[:] = tags[:, None]
    torch.manual_seed(3)
    m.densify(thr, ext)
    want = torch.cat([tags[~split], tags[clone], tags[split].repeat(2)])
    got = m._opacity.data[:, 0]
    check("c %s the recomputed masks are the rows densify copies and splits" % name,
          got.shape == want.shape and torch.equal(got, want), {"rows": int(got.shape[0]), "want": int(want.shape[0])})
    check("c %s counters reset by the postfix" % name, probe._state(m)["zero_inc"].shape[0] == m.get_xyz.shape[0]
          and float(probe._state(m)["zero_inc"].sum()) == 0.0)


# (e) the corrected average over the same rows (Codex round 3's example): accumulators [2, 0], denominators [1, 1],
# zero increments [0, 1]: both averages are 1 and one corrected denominator is 0
class _Tiny:
    percent_dense = 0.01

    def __init__(self):
        self.get_xyz = torch.zeros((2, 3), device="cuda")
        self.get_scaling = torch.full((2, 3), 0.001, device="cuda")
        self.denom = torch.ones((2, 1), device="cuda")
        self.xyz_gradient_accum = torch.tensor([[2.0], [0.0]], device="cuda")


tiny = _Tiny()
probe._fresh(tiny)
probe._state(tiny)["zero_inc"][1] = 1
gt = probe.densify_record(tiny, 1.0, 1.0)["groups"]["all"]
check("e corrected and ordinary averages over the same rows ([2,0] example)",
      gt["grad_mean_seen"] == 1.0 and gt["grad_u_mean_seen"] == 1.0 and gt["rows_zero_corrected_denom"] == 1, gt)

# (d) neutrality, both banks: a twin with the original class methods restored (no wrapper active)
@contextlib.contextmanager
def unwrapped():
    now = {k: getattr(owner, attr) for k, (owner, attr) in SLOTS.items()}
    for k, (owner, attr) in SLOTS.items():
        setattr(owner, attr, ORIG[k])
    try:
        yield
    finally:
        for k, (owner, attr) in SLOTS.items():
            setattr(owner, attr, now[k])


def _flat(prefix, x, out):
    """Every tensor and plain value reachable through dicts, lists and tuples, keyed by its path."""
    if torch.is_tensor(x):
        out[prefix] = x.detach().clone()
    elif isinstance(x, dict):
        for k in sorted(x, key=str):
            _flat("%s.%s" % (prefix, k), x[k], out)
    elif isinstance(x, (list, tuple)):
        for i_, v in enumerate(x):
            _flat("%s[%d]" % (prefix, i_), v, out)
    elif isinstance(x, (int, float, str, bool)) or x is None:
        out[prefix] = x


def digest(m):
    """Every tensor attribute of the bank (parameters and buffers), every module's state_dict (the foreground's
    deformation network), and the optimizer's complete state_dict (per-parameter state, step counters, group metadata)."""
    out = {}
    for k, v in vars(m).items():
        if torch.is_tensor(v):
            out["attr." + k] = v.detach().clone()
        elif isinstance(v, torch.nn.Module):
            _flat("module." + k, v.state_dict(), out)
    _flat("optimizer", m.optimizer.state_dict(), out)
    return out


def same(x, y):
    if torch.is_tensor(x) and torch.is_tensor(y):
        return x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y)
    return type(x) == type(y) and x == y


d_, o_, p_, fg2, bg2, tr2, scene2 = T.load_model(a.run, a.tree)
for name, m in (("bg", bg2), ("fg", fg2)):
    probe.P["views"].clear()
    twin = copy.deepcopy(m)
    n = m.get_xyz.shape[0]
    g = torch.Generator(device="cuda").manual_seed(4)
    vpt = torch.rand((n, 3), device="cuda", generator=g) * 1e-3
    filt = torch.rand(n, device="cuda", generator=g) < 0.5
    for gr in m.optimizer.param_groups:  # nonzero Adam moments on both
        gr["params"][0].grad = torch.full_like(gr["params"][0], 1e-6)
    m.optimizer.step()
    m.optimizer.zero_grad(set_to_none=True)
    for gr in twin.optimizer.param_groups:
        gr["params"][0].grad = torch.full_like(gr["params"][0], 1e-6)
    twin.optimizer.step()
    twin.optimizer.zero_grad(set_to_none=True)

    def seq(x):
        torch.manual_seed(5)
        torch.cuda.manual_seed_all(5)
        x.add_densification_stats(vpt, filt)
        x.densify(0.0004, scene.cameras_extent)
        k2 = x.get_xyz.shape[0]
        pm = torch.zeros(k2, dtype=torch.bool, device="cuda")
        pm[::97] = True
        x.prune_points(pm)
        return torch.cuda.get_rng_state(), torch.get_rng_state()

    rng_w = seq(m)
    with unwrapped():
        rng_u = seq(twin)
    dw, du_ = digest(m), digest(twin)
    diff = [k for k in set(dw) | set(du_) if k not in dw or k not in du_ or not same(dw[k], du_[k])]
    check("d %s wrapped calls leave the model, the optimizer state and the RNG bit-identical" % name,
          not diff and set(dw) == set(du_) and all(torch.equal(x, y) for x, y in zip(rng_w, rng_u)),
          {"differ": diff, "tensors": len(dw)})

json.dump(R, open(a.out, "w"), indent=1)
bad = [r["name"] for r in R if not r["ok"]]
print("ALL PASS" if not bad else "FAILED: %s" % bad, flush=True)
sys.exit(1 if bad else 0)
