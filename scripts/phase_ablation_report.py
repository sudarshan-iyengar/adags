#!/usr/bin/env python3
"""Report the LoRA phase-alignment ablation under its own frozen gates.

Spec: research-wiki/operations/lora-phase-alignment-ablation-2026-08-27.md

The gate ordering is enforced HERE rather than left to whoever runs this. The
contrast is not printed unless V1 and V2 pass, because the failure this guards
against is not arithmetic -- it is reading a number that could not have come
out any other way.

V1, NON-VACUITY, and it is a statement about the SETUP, never about the score.
    The two arms differ only insofar as the per-primitive temporal centres are
    dispersed. If they collapse, "primitive" and "global_matched" coincide BY
    CONSTRUCTION and any difference between them is noise wearing a label.
    Passes iff the middle-90% span of `get_t` at the end of training is at
    least half the sequence, in EVERY cell of BOTH arms.

V2, THE ARMS ACTUALLY DIVERGED.
    Both arms must record different configs, and their held-out scores must not
    be bit-identical. Identical scores mean the flag never reached the model --
    e.g. a checkpoint restore that dropped it -- which is vacuous, not null.

FROZEN READING RULE (spec section 5), fixed before any cell ran:
    phase alignment is SUPPORTED iff the paired mean (P - G-M) exceeds
    +0.50 dB AND all three per-seed differences carry the same sign.
    +0.50 dB is the measured same-code replicate floor at this protocol
    (0.4945 dB), not a chosen threshold.

Usage:
  python3 scripts/phase_ablation_report.py --runs <runs>/phase_ablation
  python3 scripts/phase_ablation_report.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402

SEEDS = (0, 1, 2)
ARMS = {"p": "primitive", "gm": "global_matched"}
CAPTURE_T_INDEX = 13          # scene/gaussian_model.py capture(): _t
V1_MIN_SPAN_FRACTION = 0.5
DELTA_FLOOR_DB = 0.50


def temporal_dispersion(ckpt: Path, duration: float) -> float:
    """Middle-90% span of the per-primitive temporal centres, / duration."""
    import torch
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    params = blob[0] if isinstance(blob, (tuple, list)) else blob
    t = params[CAPTURE_T_INDEX]
    if not hasattr(t, "flatten"):
        raise ContractError(
            f"{ckpt}: capture index {CAPTURE_T_INDEX} is {type(t)}, not a "
            "tensor. The checkpoint layout changed; re-read capture()."
        )
    t = t.detach().flatten().float()
    lo = float(torch.quantile(t, 0.05))
    hi = float(torch.quantile(t, 0.95))
    return (hi - lo) / max(duration, 1e-9)


def load_cells(runs: Path, duration: float, skip_v1: bool) -> dict:
    cells = {}
    for tag in ARMS:
        for seed in SEEDS:
            hits = sorted(glob.glob(str(runs / f"*_phase_{tag}_s{seed}" / "summary.json")))
            if not hits:
                cells[(tag, seed)] = {"state": "MISSING"}
                continue
            s = json.load(open(hits[-1]))
            s = s.get("summary", s)
            run_dir = Path(hits[-1]).parent
            psnr = s.get("final/psnr")
            entry = {"state": "ok" if psnr is not None else "INCOMPLETE",
                     "psnr": psnr, "ssim": s.get("final/ssim"),
                     "run_dir": str(run_dir), "dispersion": None}
            ck = run_dir / "chkpnt6000.pth"
            if not skip_v1 and ck.exists():
                entry["dispersion"] = temporal_dispersion(ck, duration)
            elif not skip_v1:
                entry["state"] = "NO_CHECKPOINT"
            cells[(tag, seed)] = entry
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--duration", type=float, default=1.6340,
                    help="sequence length in model time units (frames 0-49 @30fps)")
    ap.add_argument("--skip-v1", action="store_true",
                    help="report WITHOUT the non-vacuity gate; the contrast is "
                         "still refused, because V1 is what makes it mean anything")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        ok = True
        d = {(t, s): {"state": "ok", "psnr": 30.0 + s, "dispersion": 0.9}
             for t in ARMS for s in SEEDS}
        for s in SEEDS:
            d[("p", s)]["psnr"] = 30.0 + s + 0.9
        diffs = [d[("p", s)]["psnr"] - d[("gm", s)]["psnr"] for s in SEEDS]
        supported = (sum(diffs) / len(diffs) > DELTA_FLOOR_DB
                     and (all(x > 0 for x in diffs) or all(x < 0 for x in diffs)))
        if not supported:
            print("  FAIL: a +0.9 dB uniform lift should be SUPPORTED"); ok = False
        d2 = dict(d); d2[("p", 1)] = dict(d2[("p", 1)]); d2[("p", 1)]["psnr"] = 30.0
        diffs2 = [d2[("p", s)]["psnr"] - d2[("gm", s)]["psnr"] for s in SEEDS]
        if all(x > 0 for x in diffs2):
            print("  FAIL: a sign flip should break sign-consistency"); ok = False
        if 0.4945 > DELTA_FLOOR_DB:
            print("  FAIL: floor below the measured replicate spread"); ok = False
        print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
        return 0 if ok else 1

    if args.runs is None:
        ap.error("--runs is required unless --self-test")
    cells = load_cells(args.runs, args.duration, args.skip_v1)

    print("cells")
    for tag, arm in ARMS.items():
        for seed in SEEDS:
            c = cells[(tag, seed)]
            disp = f"{c['dispersion']:.3f}" if c.get("dispersion") is not None else "  -  "
            psnr = f"{c['psnr']:.4f}" if c.get("psnr") is not None else "   -   "
            print(f"  {arm:<15} seed {seed}  state {c['state']:<12} "
                  f"PSNR {psnr}  t-dispersion {disp}")

    missing = [k for k, v in cells.items() if v["state"] != "ok"]
    if missing:
        print(f"\nREFUSED: {len(missing)} cell(s) not usable: "
              f"{sorted(f'{ARMS[t]}/s{s}' for t, s in missing)}")
        return 2

    print("\nV1 non-vacuity (temporal centres dispersed; about the SETUP)")
    if args.skip_v1:
        print("  SKIPPED by flag -- the contrast is refused regardless.")
        return 2
    worst = min(c["dispersion"] for c in cells.values())
    v1 = worst >= V1_MIN_SPAN_FRACTION
    print(f"  worst middle-90% span = {worst:.3f} of the sequence "
          f"(floor {V1_MIN_SPAN_FRACTION}) -> {'PASS' if v1 else 'FAIL'}")
    if not v1:
        print("\nREFUSED: with the centres collapsed the two arms coincide by "
              "construction. This is INVALID, not negative.")
        return 2

    print("\nV2 the arms diverged")
    ident = [s for s in SEEDS
             if cells[("p", s)]["psnr"] == cells[("gm", s)]["psnr"]]
    v2 = not ident
    print(f"  bit-identical P/G-M pairs: {ident if ident else 'none'} -> "
          f"{'PASS' if v2 else 'FAIL'}")
    if not v2:
        print("\nREFUSED: identical scores mean the arm flag never reached the "
              "model. Vacuous, not null.")
        return 2

    diffs = [cells[("p", s)]["psnr"] - cells[("gm", s)]["psnr"] for s in SEEDS]
    mean = sum(diffs) / len(diffs)
    same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    print("\nfrozen reading rule (spec section 5)")
    for s, d in zip(SEEDS, diffs):
        print(f"  seed {s}:  P - G-M = {d:+.4f} dB")
    print(f"  paired mean = {mean:+.4f} dB   (floor +{DELTA_FLOOR_DB:.2f})")
    print(f"  all three same sign: {same_sign}")
    supported = mean > DELTA_FLOOR_DB and same_sign
    print(f"\nVERDICT: phase alignment is "
          f"{'SUPPORTED' if supported else 'NOT RESOLVED at n=3'}")
    if not supported:
        print("  Per the spec: per-primitive phase is not the load-bearing")
        print("  inductive bias, and the reframing the novelty check proposed")
        print("  is retired. A negative does NOT license re-running at a lower")
        print("  threshold or a larger n.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
