# STAR-GS v9 — Preregistered Experiment Plan (preserved, not scheduled)

Date: 2026-08-08
Status: complete claim-driven plan for testing [[operations/star-gs-v9-method]]
if and when its implementation is approved. No job in this plan has been
submitted; no code has been written. This page is the durable, self-contained
version of the plan (the generated working copies in `refine-logs/` are
transient history).
Review provenance: [[operations/star-gs-v9-review-history]].
Closest-baseline obligation: [[papers/kang2025_cec_4dgs]].

## Claim map

| Claim | Minimum convincing evidence | Blocks |
|---|---|---|
| C1 (primary): SRC's depth-free multiview residual carving places corrective capacity correctly where rendered-depth localization is unreliable (disocclusions, missing content), improving dynamic-region and event-region quality at matched capacity AND matched compute | (a) localization accuracy vs CEC-style placement on synthetic injections and audited real sites; (b) dev-scene wins vs CEC-faithful AND CEC-budget-matched on dynamic-mask/event metrics, 3 seeds; (c) component ablations (no occlusion gating / no rank normalization / no consensus / depth-based placement swap) failing as predicted | B1, B3, B4 |
| C2 (supporting): under a fixed budget, deficit birth + audited donor retirement beats unbudgeted growth and all reallocation/churn controls at matched counts | full B3 control matrix with preregistered equivalence margins; capacity/compute ledgers; donor factorial | B3 |
| Anti-claims ruled out | capacity growth (matched counts), extra compute (iso-compute lane), churn/reset artifacts (generic churn, identical resets), any-time birth (temporal controls), localization-agnostic error targeting (CEC/STG controls) | B3 |

## Experiment blocks

### B1 — Phase-A constructor gate (MUST-RUN FIRST; ~10-15 GPU-h)
Run the SRC constructor on EXISTING baseline checkpoints (dev scene,
iterations 1500/3000/4500 + final) and on synthetic-injection sets built
by deleting primitive clusters (stratified by bank/lifespan/size) from a
trained checkpoint with pre-deletion renders as reference (renderer-
matched unit test, labeled as such). Compare SRC against CEC-style
placement (single-view rendered-depth backprojection) on identical error
inputs. Preregistered floors (committed before running): median synthetic
localization error <= 1 voxel radius; >= 70% audit-genuine real
candidates (blinded multiview-RGB audit, n >= 30, fixture-calibrated —
existing census/audit tooling); yield >= 20 candidates per event window;
shifted-input yield collapse >= 5x (sanity diagnostic, not a p-value).
Expected: CEC-style placement shows larger localization error in
occluder-adjacent strata. FAIL -> stop before any training lane (one
preregistered parameter-revision cycle allowed, with new floors).

### B2 — Main anchor result (dev scenes)
From-scratch 6000-it/600k lanes on cut_roasted_beef + cook_spinach,
3 seeds: base (route0) vs full STAR-GS. Primary metric: dynamic-mask
PSNR; co-reported under a fixed multiplicity policy with hierarchical
scene+seed analysis: dynamic-mask LPIPS, event-region PSNR/LPIPS
(annotated tracks, evaluation-only, output-blind per
[[operations/phase9-annotation-contract-draft]]), tOF, static-region PSNR
(non-inferiority margin -0.05 dB), global PSNR/SSIM/LPIPS
(non-inferiority), realized capacity/compute ledgers.

### B3 — Decisive control matrix (the paper's core table)
cut_roasted_beef, same protocol. All lanes with IDENTICAL accepted-birth
counts, cadence, optimizer-state resets, and donor policy where
applicable; preregistered equivalence margins for every "matches"
verdict; 3 seeds on the decisive contrasts (STAR vs CEC-budget-matched;
STAR vs iso-compute), 1 seed + replication-on-signal elsewhere:
1. CEC-4DGS faithful reimplementation on the ADAGS backbone (unbudgeted,
   their thresholds; their code is public — port at code level);
2. CEC budget-matched (their targeting, our donor accounting);
3. STG-style guided sampling (single-view error + self-depth);
4. FreeTimeGS-style relocation (0.5*grad+0.5*opacity to existing regions);
5. generic churn (round-1 L4 machinery, identical resets);
6. iso-compute base (constructor compute spent as extra optimization);
7. temporal controls: shifted centers matched within strata of
   visibility / training exposure / residual severity / motion occupancy
   (permitted signals only — no annotations), and lifespan shuffle at
   matched presence mass;
8. donor factorial (utility vs random donors x SRC vs generic targets).
Kill rules: matches relocation / guided-sampling / CEC-budget at the
equivalence margin -> mechanism collapses into prior art, stop; matches
iso-compute -> compute-not-allocation, stop; temporal-shift match ->
WHEN claim dies, spatial-only scope reported.

### B4 — Localization evidence + attribution (make-or-break per novelty check)
(a) synthetic-injection localization extended to occluder-adjacent strata
(where CEC's rendered depth is provably wrong); (b) baseline reachability
measurement: fraction of audit-defined deficit sites (defined
independently of SRC) that ever receive capacity under base training;
(c) cluster-randomized held-out candidates during full STAR training
(exclusion radii, persistent assignment) -> policy-level local direct
effect via difference-in-differences on audit-fold cameras (no per-birth
causal language); (d) component ablations with predicted failure
signatures: no occlusion gating; no rank normalization; single-view
(consensus off); depth-based placement swap.

### B5 — Transfer, finals, failure analysis
Freeze all constants after B3, before transfer. Locked-pair single
post-freeze evaluation (flame_steak, sear_steak). Six-scene full-length
finals: STAR-GS vs base vs CEC-budget-matched at published-comparable
iteration budgets, plus a published-number context table (FreeTimeGS,
SharpTimeGS, TAD-GS) with protocol pinning (1352x1014, cam00 held out,
300 frames, LPIPS-Alex, coffee_martini cam13 handling disclosed) and
capacity/compute ledgers; 3 seeds on the two headline systems if budget
allows, else 2 + variance disclosure. Stress tier (coffee_martini,
flame_salmon_1): preregistered-checkpoint evaluation only, reveal panels,
failure analysis (specular/blur-type residual false positives, donor
errors, per-scene consistency).

## Run order, cost, gates

| Milestone | Goal | Decision gate | Cost (GPU-h) |
|---|---|---|---|
| M0 | constructor unit tests; metric definitions frozen (dynamic-mask source, event metrics, tOF implementation, ghost score); annotation pilot started | tests green; metric spec committed | ~5 |
| M1 | B1 Phase-A gate | preregistered floors pass | ~10-15 |
| M2 | CEC-4DGS port + control smokes | port fidelity; activation checks | ~40 |
| M3 | B2 + decisive B3 contrasts (3 seeds) | STAR beats CEC-budget + iso-compute outside margins | ~120 |
| M4 | remaining B3 controls + B4 | kill rules evaluated | ~80 |
| M5 | freeze -> transfer -> six-scene finals -> stress | Gate-B-style verdicts; tables complete | ~150-200 |

Total ~405-460 GPU-h (within the 400-600 envelope; M5 trimmed to 2 seeds
on overrun). Discipline (unchanged from the objective): configs committed
before submission, job IDs recorded and sacct-verified, activation
diagnostics mandatory, R009 untouched, cam00 never used in construction,
auditing, or tuning.

## Risks

Constructor yield/precision fails on real residuals (B1 kills cheaply);
effect concentrated in event metrics with flat global PSNR (co-primary
structure carries scope honestly; global non-inferiority still required);
CEC port fidelity contested (publish port; report faithful-config and
common-backbone variants); birth interference (cluster randomization;
policy-level language); six-scene power (hierarchical analysis, per-scene
effects, locked transfer as honesty check).
