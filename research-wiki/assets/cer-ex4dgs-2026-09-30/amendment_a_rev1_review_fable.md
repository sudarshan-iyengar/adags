# Fable 5.1 review of CER Ex4DGS amendment (a), rev 1 (2026-10-01)

Read-only subagent review; condensed transcription. Line numbers are the `cer` worktree's.

| # | dimension | score | short reason |
|---|---|---|---|
| 1 | rule and exactness | 8 | q = 0 holds with copies appended at the end of the dynamic bank; deductions: the duration cap is not asserted (F3); verbatim copies inherit non-finite evaluated opacity (F4) |
| 2 | A2 decision | 8 | keep `b_j` is right; the "second code path" rationale is overstated (F5); copies whose donor plateau covers the gap get exactly zero gradient to centre/var (F6) |
| 3 | A3 bookkeeping | 9 | all nine non-parameter dynamic tensors right; K inherited by slicing; one `cat` call suffices (F7) |
| 4 | lifecycle after 30,000 | 9 | only `prune_invisible`, `prune_small`, `prune_nan_points` touch rows; copies protected by A ≥ 0 |
| 5 | minimality | 8 | verbatim static copies are simpler than the snapshot; table-driven single `cat`; the donor-index bank split is the one hazard (F2) |
| 6 | A6 tests | 6 | T2c's byte equality not guaranteed (F1); T2c needs the donor map (F2); T2b's shape heuristic does not generalise (F8) |

**F1 (major).** A copy sorts at the end of its equal-(tile, depth) group, its donor at the donor's index; with a third
row of identical depth bits in a shared tile the per-pixel sequence changes. Criterion: `torch.equal` on every parameter
row pair plus the evaluated per-row quantities (`get_xyz_at_t`, `get_rotation_at_t`, `get_opacity_at_t`, scaling,
features) on the gap frames; render comparison reported (differing pixels, max |diff|). "Residual rows off" at q = 1 by
relabelling them 1 temporarily, no new code path.

**F2 (major).** Donor indices are global: split `ds = donors[donors < n_s]`, `dd = donors[donors >= n_s] − n_s` with the
pre-birth `n_s`; `begin` must record the map for T2c (the copies change the gated mass, so donors cannot be recomputed).

**F3.** Assert the cap, `g.duration == dataset.duration + g.time_shift + 1` (313), or drop the claim.
**F4.** `check_finite` in `begin` before copying.
**F5.** Dropping `b_j` would not need a second code path (parameter choice could make `b ≈ 1`); the decision stands on
"the copy would not start as its donor on fading frames".
**F6.** State that copies whose donor plateau covers the gap receive exactly zero gradient to their temporal
parameters; `var` NaN gradients are handled by `train.py:248-250` for copies as for donors.
**F7.** One `cat_tensors_to_optimizer` call with the 15 group names; remove `t_m` and the unused `inverse_sigmoid`.
**F8.** T2b: banks by capture-tuple position (1–15 static per-row, 23–39 dynamic per-row), each extended by its own D,
both banks' new-row moments zero.
**F9 (nits).** RAdam's shared step; per-row reductions for longer rotation tensors; per-bank residual mass in `gates`;
A1's "renders exactly as its donor" is per-row; cited line numbers are the pinned file's.

**Verdict: ready to implement after F1, F2, F3, F4, F8.**
