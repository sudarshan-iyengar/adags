# Fable 5.1 adversarial review of the CER Ex4DGS spec, rev 1 (2026-09-30)

Reviewer: a Fable 5.1 subagent, read-only, against a clone of Ex4DGS at 7fac649 (line numbers from `git show HEAD:`,
because the working tree started receiving the draft implementation during the review). Transcribed from the
subagent's report; the reviewer's own words, lightly condensed where it repeated itself.

## Scores

| # | dimension | score | reason |
|---|---|---|---|
| 1 | mathematics / exactness | 7 | Exactness consequences true; the q=0 proof omits the stable-sort step; "zero gradient" overstated (regularisers, RAdam momentum); t_m = 74.5 is not a frame; the donor rule selects the memory rows themselves (F1). |
| 2 | attachment points / substrate facts | 6 | Lines right within 1–3; chkpnt30000 is one RAdam step before begin; "no growth after 30,000" holds for a different reason (F2); resume unaddressed (F3); position-lr floor unmentioned (F1). |
| 3 | tensor shapes | 9 | All shapes check out. |
| 4 | cohort lifecycle | 6 | Prunes covered; promotion closed only by an unreachable trigger threshold that depends on interp_type=cube (F2); resume missing (F3); cer.pt route unnecessary (F8). |
| 5 | minimality | 8 | Only the cer.pt ply sidecar is unnecessary. |
| 6 | surgical scope | 8 | Not calling densification_postfix_onlystatic is right; name scene/__init__.py if the ply fallback is refused there. |
| 7 | tests | 7 | T2 vacuous with an empty residual cohort; T7 never prunes after begin; gates 3 and 5 are tautologies; per-row split tolerance too tight; no gradient-leak check in the spec's T3. |
| 8 | CUDA | 9 | No change needed (opacity enters as `opacities[idx]*cov.w`, forward.cu:267; skips at forward.cu:380-381, backward.cu:589-590; zero-initialised grad buffers). Caveat: `min(0.99f, NaN)` returns 0.99, so a NaN opacity renders opaque regardless of the factor (F12). |

## Findings

**F1 (major). On Ex4DGS the donor rule copies the memory cohort, and the copy cannot move.** Ex4DGS has no per-row
temporal appearance and a single-interval plateau, so at 30,000 the rows whose gap mass lies inside the silhouette are
the ghosting target rows (D ≈ K ∩ visible-in-gap). The rationale (same rows paint target and counterfactual background)
was measured on EL-GS and cannot occur here. At birth the gap render is unchanged; the static position learning rate at
≥ 30,000 is `position_lr_final × scale` = 1.6e-6 × scale (get_expon_lr_func clips at position_lr_max_steps = 30,000), so
residual rows cannot relocate to the background depth, only fade or recolour in place in ≈ 1,000 gap-frame iterations.
Fading exposes the wall rows and then fails gate 4; the alternative is a background-coloured billboard at the target's
depth, parallax-wrong in cam00. Fix (principal's decision): measure donors on the memory-gated model, so donors are the
rows that paint the gap region when the memory cohort is off, at the correct depth, and D ∩ K = ∅ by construction; or
keep the ghost-copy design and redefine gate 4.

**F2 (major). "After 30,000 the lifecycle can only remove rows" is true for this config, but not for the stated
reason.** `sample_len` reaches its cap 312 at 12,800; `mark_last` is set at 12,500 and stays True to the end, because its
consumer (train.py:127-130) needs a camera with timestamp ≥ sample_len − interval = 302 and none exceeds 299. Promotion is
blocked only because time_shift (12, cube) > interval (10); with linear interpolation it would fire after 30,000. Fix:
begin asserts `max timestamp < sample_len − interval`.

**F3 (major).** `--start_checkpoint` from chkpnt30000 skips begin and trains a base model under a CER run id. Refuse it.

**F4 (major).** chkpnt30000 is written before the optimizer step and NaN prune of iteration 30,000; begin sees one step
later. Tests may use it as input; cer_begin.json of a real run is not reproducible from it. State it.

**F5.** The q=0 proof needs: duplicateWithKeys emits keys in row order, the radix sort is stable, residual rows sit
between the static and dynamic blocks, so existing rows keep their tie order; last_contributor counts only contributing
rows.

**F6.** "Zero gradient" is the rendering-loss gradient only: static_reg and motion_reg reach every row, and RAdam applies
momentum.

**F7.** Use t_m = (A+B)//2 = 74, a trained frame.

**F8.** Refuse the ply fallback for a CER model instead of the cer.pt sidecar.

**F9.** T2 must require at least one residual row.

**F10.** T7 at 14,000 iterations never runs a prune after begin (epoch = 5,700 iterations; prunes at refills > 6,000:
11,401 < 13,000). Use 17,200 iterations so 17,101 prunes through the labelled prune_points.

**F11.** Per-row split tolerance is below float32 atomic-add noise; use an aggregate. Optional cross-check: the l1_accum
hook already accumulates αT per row in viewspace_l1points.grad[:,0].

**F12.** Assert finite opacities for the cohorts (a NaN opacity times 0 is NaN and renders at alpha 0.99; prune_nan_points
reads only positions).

**F13.** Gates 3 and 5 are tautologies once T3 passes; keep as guards, not evidence. Gate 2 can fail on a same-pose scene
if the base already split the lineage (a clone whose plateau starts at B+1 has no pre-gap mass and is not memory); record
that as such, not as a CER defect.

**F14.** Minor line corrections (render.py:128/129, backward.cu:589-590); the GUI call passes scaling_modifer into
timestamp (upstream bug, inert); begin ≈ 1,710 and gates ≈ 2,280 render+backward passes (≈ 5–12 min) are not in the
1.2 GPU-h line.

**F15.** prune_small can cull residual rows under 5 px in one gap view at 34,201/39,901; gate 4 is where it shows.

## Confirmed correct

Raw rotation copy (CUDA does not normalise); scaling pre-activation; features split; opacity logit round trip; torch
2.1.2 norm backward at zero gives 0, not NaN; autograd.grad on override_color touches no parameter gradient, render()
consumes no RNG; one opacity junction, no caller with mode ≠ 0; the final model equals chkpnt40000 and render.py restores
the 41-tuple; xyz_error_min_timestamp = A protects newborns; 40-element capture.

**Verdict: ready to implement after fixes; F1 needs one explicit decision from the principal before any CER cell is
submitted.**
