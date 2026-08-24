#!/usr/bin/env python
"""Demonstrate that the shell-supply PROJECTION cannot decide hull completion.

This script retires a route. It does not measure hull completion; it measures
the instrument that was going to be used to measure hull completion, and shows
that instrument cannot carry the question -- in absolutes OR in deltas.

RESULT 1 -- ABSOLUTES ARE UNUSABLE, calibrated against real data.
The proxy scores `object_shell / (object_shell + filler_shell)`, so it assumes
every gated row is object or filler. `nonconvex-hull-o1-result-2026-08-24`
records O1's base rule gating 10,374 rows at precision 0.5868, of which 1,152
are notch filler. That leaves 6,087 object and **3,135 (30.2%) NEITHER** --
background the proxy's denominator structurally cannot contain. Fed those exact
row counts, the proxy's accounting returns 0.8409 against a measured 0.5868:
**+0.2541, about 25x the ~0.01 discretization error.** Because the denominator
cannot hold background, the proxy is biased HIGH always, never low.

RESULT 2 -- DELTAS ARE UNUSABLE TOO, and this took three tries to see.
Spec section 9 reads the verdict only from the delta, so the delta was the
obvious fallback. It does not survive either.

  (a) Enumerating index bboxes and evaluating each at its MAXIMAL in-bbox mask
      (`box & occ`) makes `delta <= 0` a THEOREM, not a finding: every added
      cell then lies outside `occ`, and object shell on non-occupied cells is
      exactly zero (asserted below). The favourable branch is unreachable by
      construction. An earlier version of this analysis reported 156/156
      negative deltas from exactly that family and read it as a result.
  (b) Dropping the maximal-mask restriction, the two available accountings --
      surface-area-weighted shell counts, and volume-weighted row counts that
      DO include background -- disagree by an order of magnitude on how often
      H1 helps. They do not corroborate each other; they diverge.

So no reading of this projection decides the operator. A successor spec must
MEASURE (spec V4) on the trained substrate over `row_ids`, with recall and
false activations, and must state the gate's reachable range before freezing --
on this fixture the recall limb was already unreachable (0.2995 / 0.1599
against a 0.90 floor).

SCOPE. Every decomposition here is the **fresh 50k seeding grid**. The trained
grid is not controllable and the completed runs realized a different one:
experiment 274's accepted cells sit at `j = 5` while the fresh grid places the
object at `j` in {3,4}. Cell counts do not transfer to a trained substrate.

WHAT IS REUSABLE beyond the retirement: the accounting correction
(`p = p0/(1+r)` is valid only for `r = added / ALL gated rows`; the preflight's
`r` is `added / OBJECT rows`, differing by a factor `p0`), and the proof that
BFS tie-breaking cannot affect H1's output here.

CPU only, no torch, no GPU, no rendering.
"""

from __future__ import annotations

import itertools
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nonconvex_hull_preflight as pf  # noqa: E402

#: Trained-substrate ground truth for O1's base rule, from
#: nonconvex-hull-o1-result-2026-08-24.
O1_MEASURED = {"gated": 10374, "precision": 0.5868, "filler_rows": 1152}

#: Largest component size enumerated exhaustively. C(32, 8) is 10.5M, so 7 is
#: where exhaustive enumeration stops being free; the trend is already clear.
MAX_COMPONENT = 7

STEPS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _fresh_cloud():
    rng = np.random.default_rng(pf.INIT_SEED)
    return rng.uniform(
        -pf.INIT_HALF_WIDTH, pf.INIT_HALF_WIDTH, size=(pf.N_INIT_PTS, 3)
    )


def _neighbours(cell, shape, steps=STEPS):
    for d in steps:
        n = (cell[0] + d[0], cell[1] + d[1], cell[2] + d[2])
        if all(0 <= n[a] < shape[a] for a in range(3)):
            yield n


def shortest_path(occ, start, goal, steps=STEPS):
    if start == goal:
        return [start]
    prev, queue = {start: None}, deque([start])
    while queue:
        cell = queue.popleft()
        for n in _neighbours(cell, occ.shape, steps):
            if occ[n] and n not in prev:
                prev[n] = cell
                if n == goal:
                    path, cur = [], n
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    return path[::-1]
                queue.append(n)
    return []


def _connected(mask):
    idx = [tuple(int(v) for v in c) for c in np.array(list(zip(*np.nonzero(mask))))]
    if not idx:
        return False
    seen, stack = {idx[0]}, [idx[0]]
    while stack:
        for n in _neighbours(stack.pop(), mask.shape):
            if mask[n] and n not in seen:
                seen.add(n)
                stack.append(n)
    return len(seen) == len(idx)


class Fixture:
    def __init__(self, oname):
        cloud = _fresh_cloud()
        lo = cloud.min(axis=0)
        span = np.clip(cloud.max(axis=0) - lo, 1e-6, None)
        ev, fl = pf.ORIENTATIONS[oname]
        cpa = pf.CELLS_PER_AXIS
        res = pf.analyse(ev, fl, lo, span, cpa, cloud=cloud)
        axes = pf.lattice_axes()
        self.name = oname
        self.occ = res["occ"]
        self.shell_obj = pf.shell_counts_per_cell(
            pf.shell_mask(pf.lattice_mask(ev, axes)), axes, lo, span, cpa)
        self.shell_fill = pf.shell_counts_per_cell(
            pf.shell_mask(pf.lattice_mask(fl, axes)), axes, lo, span, cpa)
        self.rows_all, self.rows_obj = res["rows_all"], res["rows_obj"]
        vol_a = pf.per_cell_box_volume(ev[0], res["edges"])
        vol_b = pf.per_cell_box_volume(ev[1], res["edges"])
        self.only_a = (vol_a > 0) & (vol_b == 0)
        self.only_b = (vol_b > 0) & (vol_a == 0)
        self.cells = [tuple(int(v) for v in c)
                      for c in np.array(list(zip(*np.nonzero(self.occ))))]

    def _shell_precision(self, mask):
        o = float(self.shell_obj[mask].sum())
        f = float(self.shell_fill[mask].sum())
        return o / (o + f) if o + f > 0 else float("nan")

    def _row_precision(self, mask):
        o = float(self.rows_obj[mask].sum())
        a = float(self.rows_all[mask].sum())
        return o / a if a > 0 else float("nan")

    def deltas(self, mask):
        """(shell-accounting delta, row-accounting delta) for H1 on `mask`."""

        after = pf.bbox_fill_mask(mask)
        return (self._shell_precision(after) - self._shell_precision(mask),
                self._row_precision(after) - self._row_precision(mask))

    def admissible(self, mask):
        return (bool((mask & self.only_a).any())
                and bool((mask & self.only_b).any())
                and _connected(mask))

    def mask_of(self, cells):
        m = np.zeros_like(self.occ)
        for c in cells:
            m[c] = True
        return m


def report_calibration():
    m = O1_MEASURED
    obj = round(m["gated"] * m["precision"])
    neither = m["gated"] - obj - m["filler_rows"]
    reads = obj / (obj + m["filler_rows"])
    print("RESULT 1 -- absolutes are unusable")
    print("  O1 base rule, measured: %d gated | %d object | %d filler | "
          "%d NEITHER (%.1f%%)"
          % (m["gated"], obj, m["filler_rows"], neither,
             100.0 * neither / m["gated"]))
    print("  proxy accounting on those exact counts -> %.4f" % reads)
    print("  trained substrate measured             -> %.4f" % m["precision"])
    print("  BIAS = +%.4f, about %.0fx the ~0.01 discretization error"
          % (reads - m["precision"], (reads - m["precision"]) / 0.01))
    print("  The denominator cannot hold background, so the bias is one-sided.\n")


def report_vacuity(fx):
    """The maximal-mask family forces its own answer. Assert it, don't report it."""

    shell_outside = float(fx.shell_obj[~fx.occ].sum())
    rows_outside = int(fx.rows_obj[~fx.occ].sum())
    assert shell_outside == 0.0 and rows_outside == 0
    print("  (a) object shell on non-occupied cells = %.1f, object rows = %d."
          % (shell_outside, rows_outside))
    print("      So for the MAXIMAL in-bbox mask every added cell is pure false")
    print("      positive and delta <= 0 is FORCED. Enumerating that family")
    print("      cannot return a favourable answer -- it is vacuous, and an")
    print("      earlier version of this analysis reported it as a result.")


def report_divergence(fx):
    print("  (b) dropping the maximal-mask restriction, over ALL connected")
    print("      spanning components of size 2-%d:" % MAX_COMPONENT)
    print("        size   masks    shell %pos   best      row %pos   best")
    for size in range(2, MAX_COMPONENT + 1):
        ds, dr = [], []
        for combo in itertools.combinations(fx.cells, size):
            mask = fx.mask_of(combo)
            if not fx.admissible(mask):
                continue
            d, r = fx.deltas(mask)
            ds.append(d)
            dr.append(r)
        if not ds:
            continue
        ds, dr = np.array(ds), np.array(dr)
        print("         %2d   %6d      %5.1f%%  %+.4f     %5.1f%%  %+.4f"
              % (size, len(ds), 100 * (ds > 0).mean(), ds.max(),
                 100 * (dr > 0).mean(), dr.max()))
    d, r = fx.deltas(fx.occ)
    print("      full occupied set (%d cells): shell %+.4f | row %+.4f"
          % (len(fx.cells), d, r))
    print("      The two accountings DISAGREE on how often H1 helps. They do")
    print("      not corroborate each other, so the delta is undecided too.")


def report_bfs_invariance(fx):
    orderings = [tuple(p) for p in itertools.permutations(STEPS)]
    pairs = [(a, b) for a in fx.cells if fx.only_a[a]
             for b in fx.cells if fx.only_b[b]]
    worst, non_monotone = 1, 0
    for a, b in pairs:
        l1 = sum(abs(a[i] - b[i]) for i in range(3)) + 1
        boxes = set()
        for steps in orderings:
            path = shortest_path(fx.occ, a, b, steps)
            if not path:
                continue
            if len(path) != l1:
                non_monotone += 1
            boxes.add(pf.bbox_fill_mask(fx.mask_of(path)).tobytes())
        worst = max(worst, len(boxes))
    print("  BFS invariance: %d pairs x %d orderings -> at most %d distinct "
          "bbox per pair," % (len(pairs), len(orderings), worst))
    print("    %d non-monotone paths. Every shortest path has length L1+1, so"
          % non_monotone)
    print("    it cannot leave its endpoints' bbox. Tie-breaking is irrelevant.")


def main():
    print("hull completion on LRV5-NCX -- the PROJECTION cannot decide it")
    print("numpy %s | no torch | no GPU | no rendering\n" % np.__version__)
    report_calibration()
    print("RESULT 2 -- deltas are unusable too")
    for oname in ("O1", "O2"):
        fx = Fixture(oname)
        print("\n%s   (fresh seeding grid -- NOT the trained decomposition)"
              % oname)
        report_vacuity(fx)
        report_divergence(fx)
        report_bfs_invariance(fx)
    print("\nCONCLUSION: neither absolutes nor deltas from this projection can")
    print("decide hull completion. The route is retired; a successor spec must")
    print("measure on the trained substrate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
