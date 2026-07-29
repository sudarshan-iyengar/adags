# Phase 0 Census-v2 Preregistration — certification vs carrier discrimination

Date: 2026-07-29
Status: preregistered before execution; user-approved 2026-07-29. This page,
`configs/depth_visibility/phase0_census2_v1.json`, and the implementation are
committed before the census-v2 job is submitted or its outputs inspected.
Parent: [[operations/phase9-csvl-vpl-v2-direction]] (Phase 0, second cycle)
Preserves: [[operations/phase0-census-result]] (PHASE0_NO_GO) unchanged — v2
is a new cycle with new floors, not a revision of v1.

## Questions

**Q1 (certification).** Does a reveal-certification rule designed to require
evidence-temporal structure — occluder-gap magnitude, occluder-depth
coherence, anchored hysteresis, static-parallax exclusion — separate valid
evidence from frame-shuffled evidence where the v1 naive rule (ratio 0.950)
could not?

**Q2 (carrier, user-added hypothesis).** Is the route0 checkpoint an adequate
carrier of stable primitive identity/geometry for census purposes, or did v1's
F4 failure arise (wholly or partly) because primitive motion/jitter, not the
rule, destroys separability? v2 must distinguish certification-rule failure,
carrier/checkpoint failure, and their interaction.

## Design: 2x2 decomposition, checkpoint axis, quality stratification

**Positions axis.** (a) *moving*: `get_dynamic_xyz(t)` as in v1;
(b) *frozen*: each primitive fixed at its canonical position `_xyz` (the LoRA
offset at a primitive's own temporal center is exactly zero by construction),
with temporal presence still per-frame. Frozen positions remove all carrier
motion; any remaining state transitions come from evidence temporal structure.

**Evidence axis.** (a) *valid*; (b) *shuffled* (same per-camera frame
permutation, seed 20260729, as v1).

**Checkpoint axis (carrier comparison, "where available").**
- Primary: `20260722_102349_cut...600k_9000/chkpnt6000.pth` (as v1).
- Maturity: same run `chkpnt3000.pth` and `chkpnt9000.pth`.
- Independent run: `20260701_012706_cut...600k/chkpnt6000.pth` (same config
  family, independent optimization ~3 weeks earlier). Fallback preregistered:
  if its capture tuple fails to restore, record the failure and proceed
  without it; no substitute is selected after seeing results.
Passes: primary checkpoint runs all four cells; comparison checkpoints run
moving x {valid, shuffled} only (the frozen probe is a primary-config
diagnostic). No hard-static points expected in any checkpoint (asserted).

**Primitive-quality stratification (computed on the primary moving x valid
pass, then applied to stratified counts in all primary cells).** Per-primitive
metrics: activated opacity; maximum spatial scale; motion range
(max_t ||xyz_t - _xyz||); presence breadth (number of present frames);
near-surface stability (near-surface fraction of evaluable tuples); margin
proximity (median |z - d|/margin). Strata: quartiles of motion range,
quartiles of near-surface stability, opacity >= 0.5 vs < 0.5, quartiles of
scale. Stratified separation ratios are diagnostic outputs, not floors.

## Frozen certification rule (replaces the v1 naive transition count)

All evaluated per (primitive, camera) on the 17 included cameras
(`cam12`/`cam19` are excluded by preregistration — their single-member P01
coverage cannot form a consensus; fixing P01 grouping is out of scope for
this cycle). Baseline window frames 0-29; certification window frames 30-299.

1. **Gap-occluded frame**: BEHIND with witness AND `(z - d) >= k_gap x margin`
   with `k_gap = 3.0`, AND occluder-depth coherence: if a previous accepted
   gap-occluded frame exists in the current run, `|d_now - d_prev_occ| <=
   max(0.2 x d_now, 5.0 x sigma_now)`.
2. **Anchor**: >= 2 consecutive near-surface frames before entry.
3. **Entry**: >= 2 consecutive gap-occluded frames while anchored.
4. **Run**: total gap-occluded frames >= 3; at most one interruption frame of
   grace (a single near or other frame); two consecutive non-qualifying
   frames abort the run without certification.
5. **Certification**: >= 2 consecutive near-surface frames immediately after
   the run (the same stretch re-anchors the pair).
6. **Static-parallax exclusion**: pairs whose baseline (frames 0-29, >= 5
   evaluable) is majority-BEHIND are ineligible for certification.

The shuffle control destroys exactly the temporal coherence that rules 1-5
require, so a genuine occlusion signal must separate strongly under this rule
or the evidence-temporal-structure hypothesis is wrong at this granularity.

## Frozen floors — CENSUS2_GO requires all four, evaluated on the primary
configuration (chkpnt6000, moving positions, all eligible primitives)

- **G1 certified abundance**: >= 2,000 certified (primitive, camera) pairs,
  spanning >= 10 distinct end frames and >= 8 of the 17 cameras.
- **G2 control separation**: rho = certified pairs (valid) / certified pairs
  (shuffled) >= 3.0. If shuffled pairs = 0, G2 passes iff valid pairs >= G1's
  2,000.
- **G3 evidence validity**: cross-view conflict fraction <= 15% and >= 90% of
  the 17-camera consensus maps pass the member/valid-pixel checks.
- **G4 non-degeneracy**: no single camera contributes > 60% of certified
  pairs.

## Preregistered diagnosis rules (independent of GO; these answer Q2)

- **D1 rule failure confirmed (carrier adequate)**: rho_moving >= 3.0. The v1
  F4 failure is attributed to the naive rule; the checkpoint carries stable
  enough identity/geometry for census purposes.
- **D2 carrier implicated**: rho_frozen >= 3.0 AND rho_moving < 3.0. Evidence
  temporal structure is certifiable, but primitive motion destroys it: carrier
  failure or rule-carrier interaction. The stratified ratios localize it: if
  the top near-surface-stability or bottom motion-range stratum reaches
  rho >= 3.0 while pooled rho_moving < 3.0, classify as INTERACTION
  (rule works on well-behaved primitives; carrier quality gates coverage);
  otherwise classify as CARRIER failure.
- **D3 residual evidence/rule failure**: rho_frozen < 3.0. Even motionless
  carriers cannot separate: the failure is in the evidence signal or the rule
  family, and the carrier hypothesis is exonerated for v1's F4.
- **D4 checkpoint dependence**: max/min of rho_moving across successfully
  restored checkpoints >= 2.0 flags carrier-maturity sensitivity
  (descriptive; it refines but does not override D1-D3).

## Outputs, jobs, discipline

Output root: `$WORK/proj_adags/runs/phase9-depth-visibility-capacity/`
`phase0-census2-v1/` (`census2-v1.json` with canonical scientific SHA-256,
per-cell summaries, stratified tables, floors, diagnosis; capped transition
samples). One smoke job (frame-limit 40, separate output root, non-scientific,
validates all checkpoint restores) then one scientific job (<= 3 h,
boost_usr_prod pattern, logs under `logs/` with job ID). Job IDs captured
immediately; `squeue`/`sacct` checked before any resubmission. No floor or
rule adjustment after outcome inspection: any census-v3 needs a new
preregistration. No RGB, annotations, evaluator masks, R009 crop pixels, or
W&B. The v1 result page and artifacts are immutable.

## Decision rule

CENSUS2_GO iff G1-G4 all pass on the primary configuration. The formal
verdict (either way) plus the D1-D4 diagnosis is recorded in a result page,
and execution stops there; Phase 1 remains unauthorized regardless of the
verdict until the user approves.
