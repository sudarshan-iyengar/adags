# Phase 0 Census Preregistration — CSVL-VPL v2

Date: 2026-07-29
Status: preregistered before execution. This page, the frozen config
`configs/depth_visibility/phase0_census_v1.json`, and the implementation are
committed before the census job is submitted or its outputs inspected.
Parent: [[operations/phase9-csvl-vpl-v2-direction]] (Phase 0)
Constraint honored: C-6 (opportunity census before evidence machinery),
C-7 (control separation), C-8 (pre-registered effect-size floors).

## Question

Does the primitive-centric E1 visibility representation contain abundant,
temporally structured, control-separated occlusion/reveal opportunities on
`cut_roasted_beef` — the quantity Stage 1C showed the P03 bin representation
lacks entirely (zero cross-order candidates in 19/19 windows)?

## Exact inputs (hashes recorded in the output at runtime)

- Checkpoint: `$WORK/proj_adags/runs/fixed_budget_lora_route0_600k_9000/`
  `20260722_102349_cut_roasted_beef_fixed_budget_lora_route0_600k_9000/chkpnt6000.pth`
  — the canonical 6000-iteration route0 endpoint of the R038 relaunch
  (job 50073065). Model constructed as
  `GaussianModel(sh_degree=3, gaussian_dim=4, time_duration=[0.0, 10.0],
  rot_4d=False, force_sh_3d=True, sh_degree_t=0)`, restored with
  `restore(model_args, None)` (no optimizer). Dynamic bank only; the run has
  zero hard-static points (asserted).
- P01 DA3 sidecar: `$WORK/proj_adags/runs/phase9-depth-visibility-capacity/`
  `cycle-v5/preprocess/cut_roasted_beef/da3/` (manifest SHA-256 recorded).
  19 groups x 6 member views per frame at 378x504, per-member aligned_w2c and
  processed intrinsics, target camera cam00 never present.
- Frames 0..299; timestamp of frame f is `f/30` seconds.
- Cameras: the 19 training cameras present in P01 member views (cut drops
  cam04; cam00 excluded by construction).
- No RGB, no annotations, no evaluator masks, no R009 crop pixels, no W&B.

## Frozen method

1. **Consensus evidence depth** per (camera, frame): collect all P01 member
   views of that camera across groups (expected 6; require >= 3); per-pixel
   median depth `d`, robust sigma `1.4826 * MAD`, median confidence. Valid
   pixel: finite, `d > 0`, member count >= 3, median confidence >= the 20th
   percentile of that map's finite median-confidence values.
2. **Primitive positions**: `get_dynamic_xyz(t)` (repo code, not a
   reimplementation); temporal presence `get_marginal_t(t) >= 0.05`;
   primitives with activated opacity `< 0.005` (the prune floor) excluded and
   counted.
3. **Projection**: `x_cam = w2c @ [X;1]`; require `z > 0.01`; pixel via the
   member-consistent processed intrinsics (first occurrence per camera-frame;
   nearest-neighbor depth sample; in-bounds required).
4. **Margin**: `margin(px) = max(tau_rel * d, 2.5 * sigma(px))` with primary
   `tau_rel = 0.03`. Sensitivity variants `tau_rel in {0.01, 0.05}` are
   computed and reported as descriptive only; floors are evaluated only at
   0.03 (frozen before execution; no outcome-conditioned selection).
5. **States** per (primitive, camera, frame), on evaluable tuples (present,
   in-view, valid pixel): near-surface `|z - d| <= margin`; behind
   `z - d > margin`; in-front `d - z > margin`.
6. **Occluded-with-witness**: behind in camera c AND near-surface in >= 1
   other camera at the same frame.
7. **Completed reveal** per (primitive, camera): a maximal run of >= 3
   consecutive frames in occluded-with-witness immediately followed by a
   near-surface frame in the same camera. Any other state or a non-evaluable
   frame resets the run (strict primary rule). A relaxed variant (runs
   survive behind-without-witness frames) is reported as descriptive only.
8. **Shuffle control**: per camera, a fixed pseudorandom permutation
   (seed 20260729) of the frame-to-consensus-depth assignment; the full
   pipeline including witnesses is recomputed at the primary margin.
9. **Cross-view consistency**: on frames 0,10,...,290 and a fixed list of 20
   camera pairs, back-project a stride-60 grid of valid pixels of camera A at
   depth `d_A`, project into camera B, and classify consistent / occluded /
   conflict (conflict = the point lands more than margin in front of B's
   surface). This doubles as the sign/convention/z-buffer fixture on real
   data; a synthetic projection fixture is in the unit tests.

## Frozen floors — PHASE0_GO requires all five

- **F1 abundance**: occluded-with-witness tuples >= 0.5% of evaluable tuples.
- **F2 reveals**: >= 5,000 distinct (primitive, camera) pairs with >= 1
  completed reveal (strict rule), spanning >= 10 distinct end frames and
  >= 5 cameras.
- **F3 non-degeneracy**: per-camera occluded-with-witness fraction has median
  in [0.1%, 40%] and no camera above 60%.
- **F4 control separation**: valid completed-reveal pair count >= 2x the
  shuffle-control pair count.
- **F5 evidence validity**: cross-view conflict fraction <= 15% (mean over
  sampled pairs and frames), and >= 90% of (camera, frame) consensus maps
  reach >= 3 members and >= 50% valid pixels.

Failure attribution is preregistered: an F5 failure is an evidence-validity
no-go (the P01-derived consensus is unusable for E1 and the evidence source
must be fixed before any opportunity conclusion); an F1/F2/F3/F4 failure with
F5 passing is an opportunity/structure no-go for the E1 representation on
this scene, which triggers the bounded read-only dataset assessment approved
in the Phase 0 authorization. Quantities below floors are reported as null
results, never as "directional".

Descriptive-only outputs (explicitly not floors): overlap of reveal end
frames with the two historical cut R009 window frame ranges (95-110,
140-155); the relaxed-rule reveal count; sensitivity-margin variants;
occlusion-run-length distribution; per-camera and per-frame distributions;
excluded-primitive counts; consensus-dispersion statistics.

## Outputs

`$WORK/proj_adags/runs/phase9-depth-visibility-capacity/phase0-census-v1/`:
`census-v1.json` (all counts, distributions as histograms, floor verdicts,
input hashes, config echo, canonical scientific-content SHA-256 excluding
timestamp/job-id/output-root) and `transitions-sample.json` (capped 20,000
rows). Slurm stdout/stderr under repo `logs/` with the job ID in the
filename. One job; job ID captured immediately; `squeue`/`sacct` checked
before any resubmission; a failed run gets a new run ID.

## Decision rule

PHASE0_GO if and only if F1-F5 all pass. The GO/NO_GO verdict, whichever it
is, is recorded in a result page with the same evidence discipline as the
Stage 1 pages. PHASE0_GO does not authorize Phase 1; it makes Phase 1
scientifically admissible pending user approval.
