# Phase 9 CSVL-ISR v1 claim-driven experiment plan

Status: preregistered before new Phase 9 outcomes
Date: 2026-07-15
Method: [[operations/phase9-csvl-isr-v1-method]]
Objective: [[objectives/depth-visibility-capacity-v1]]
Slice B contract: [[operations/phase9-csvl-isr-v1-slice-b-contract]]
Split manifest: `configs/depth_visibility/n3v_split_v1.json`
Atomic run matrix: `configs/depth_visibility/phase9_run_matrix_v1.json`

This plan maps every permitted run to a claim, decision, and artifact. It uses an
adaptive fidelity ladder; 6000 iterations are reserved for convergence-matched
representation comparisons, not plumbing or geometry diagnosis. A run may start
only after its predecessors have terminal success and its exact command/config/
source hashes are checkpointed. Existing state, squeue, sacct, logs, and outputs
must be checked before submission or retry.

## Data and claim discipline

- Development and any selection: `cut_roasted_beef` only.
- Locked human-labeled transfer: `flame_steak` and `sear_steak` once after freeze.
- Final admitted representation comparison: all six N3V scenes after checksum
  freeze; no silent retuning from the same evaluation cycle.
- R009: historical posthoc continuity only; never training, selection, or tuning.
- Genuine human fields begin empty. Missing labels make label-dependent criteria
  `not_evaluable`; no synthetic, target-RGB, or model-derived substitute is valid.
- Expensive runs require a registered purpose and decision rule. Scientific
  failures are not retried as infrastructure failures or followed by arbitrary
  sweeps.

## Atomic execution authority

The machine-readable matrix is authoritative over the family-level tables. It
contains 181 atomic producer, execution, reuse, external-ingest, and decision
entries. Every internal prerequisite is another registered run ID; the only
external dependency is a typed genuine-human annotation return consumed by a
scene label-freeze producer. No boolean completion aliases are permitted.

The matrix separately registers:

- local deterministic tests and independent code-review admission;
- P9-I01-IMPLEMENTATION-FREEZE, which seals the pushed commit, environment,
  launcher, command, config, split, annotation, method, plan, and schema hashes;
- six full DA3 producers, six existing-flow validation/adaptation producers,
  six complete CSVL ledgers, and six training-sidecar freezes;
- blinded annotation packet, three human label freezes, target-consuming
  R031/R032/R033/R031-MT producers frozen over every candidate camera/window
  before labels open, plus cut/transfer score and decision nodes;
- scene evaluator freezes, cut oracle sidecar, operator conformance, every
  training lane, and explicit B01/B02/B03/B04 decision producers;
- a K=2048 cut null-reset endpoint, both preregistered median-moment repair
  lanes, an unconditional typed repair branch join, a B02 optimizer-policy
  artifact consumed by every later mutation lane, exact reuse verification,
  and the conditionally selected seed-1/2 matrix.

Before implementation, executable launcher/config/code hashes are intentionally
unresolved and submission_ready=false. No Slurm entry may be submitted until
I01 resolves the command template, merged config, environment, implementation,
sidecar/input artifact, and output inventory bindings. This is a fail-closed
implementation boundary, not a placeholder scientific decision. Each execution
then records distinct input/output checkpoints, run directory, W&B IDs when
applicable, Slurm terminal state/exit code, and every required artifact hash.

Cycle v1 uses seed 0, an iteration-5000 common checkpoint, a fixed 5250 pilot,
a 6000 comparable endpoint, K=256 pilot, K=2048 comparable, and a 600k point
ceiling. Seeds 1 and 2 are fully enumerated but run only after the registered
seed-0 decision passes. The conservative registered maximum is 840 GPU-hours:
416 unconditional in the all-gates-pass execution path plus 424 conditional
hours (4 for the sole repair and 420 for seed expansion). These maxima are
reservation bounds, not targets; failed gates prevent downstream spend.

The non-concurrent storage-bound sums are 7,196 GiB temporary, 8,759 GiB durable
run products, and 2,300 GiB checkpoint outputs. Actual free space is checked
before each wave and cleanup follows the retention manifest; raw arrays,
checkpoints, renders, and logs remain outside Git. GPU entries request one
Slurm a100 on boost_usr_prod, account euhpc_d21_034, QoS boost_qos_lprod;
per-entry CPU, host memory, wall time, storage, and job-ID log paths are
explicit. The 2026-07-15 partition probe observed four A100s, 514,000 MiB node
memory, 32 CPUs, and a one-day limit. I01 must verify 64-GiB GPU memory before
submission.

The matrix source hashes bind the objective, method, Slice B contract, this
plan, split manifest, 54-window annotation manifest, frozen scientific config,
frozen Slice B mode config, and generator. Any
post-outcome change creates a new cycle and matrix.

## Gate A registry

| ID | Claim or uncertainty | Data / comparison | Budget | Decisive result | Decision and artifacts |
| --- | --- | --- | --- | --- | --- |
| A00 | Code matches the frozen schemas, camera math, state/risk hierarchy, matching, and metrics. | Hand JSON and analytic arrays; exact expected outputs. | Login CPU, <5 min. | All deterministic schema, round-trip, covariance, order, transition, spatial-FP, calibration, hash, and failure-path tests pass twice. | Failure blocks Slurm. Track tests; ignore temp outputs. |
| A01 | Target-free multiview-temporal fusion recovers controlled front/rear/reveal structure. | Deterministic pinhole two-plane fixture; full CSVL versus no-temporal, sign-error, and wrong-flow controls. | CPU <10 min; Slurm if larger. | Correct order/events/regions within raster tolerance; each corrupted control fails its targeted assertion. | Failure returns to implementation. Preserve tracked fixture and ignored generated report. |
| A02 | Exact read-only DA3 weights are pinned. | 6.76 GB model file versus expected size/config/code pins. | CPU Slurm <=30 min. | COMPLETED/0:0 plus one SHA-256 of exact file. | Seal hash/job/command; failure blocks inference. |
| A03 | Pinned DA3 API integrates supplied K/w2c and has the declared depth/resize/repeatability behavior. | Exact analytic camera fixture plus cut frame-0 overlapping-group conformance. | One GPU Slurm <=30 min. | All fixed camera/z/K/finite/positive/MAD/repeatability thresholds pass. | Failure blocks real sidecars; preserve ignored arrays and durable conclusion. |
| A04 | Real cut processing is nondegenerate, target-free, numerically stable, and deterministic. | Frozen three-frame cut interval; CSVL, no-temporal, misaligned camera/flow controls. | One GPU Slurm <=2 h. | No cam00 image provenance; finite/nonzero support; valid projection/source-count/cycle/order/risk distributions; controls fail correctly; repeat hashes match. | Software/label-free admission only. Diagnose one causal defect at a time; do not tune gates. |
| A05 | A genuine blinded reference can be collected without leakage. | Exact frozen 54 raw-RGB windows; empty human fields; no predictions/depth/residuals. | Slurm contact-sheet rendering <=2 h if substantial. | R009-disjoint proof, fixed split/double-annotation assignment, valid schema/hashes, all human labels empty. | Handoff packet for humans; never auto-fill. |
| A06 | CSVL identifies real visibility/order/events and transfers. | Before labels, produce target-consuming R031/R032/R033/R031-MT for cam00 plus every train camera in all candidate windows; scoring later selects frozen annotated rows. Fit only on cut calibration, score cut development once, then locked flame/sear. | Three GPU baseline producers <=15 h each; CPU score/decision <=4 h each. | Every engineering or claim-grade conjunction in method v1. | Explicit score/provenance and cut/transfer decisions; missing labels = not evaluable. |
| P01 | Full calibrated DA3 evidence exists, not only a tiny diagnostic. | One scene per job, all 300 times and valid training-camera groups with complete target-conditioned ancestry. | Six one-A100 jobs <=24 h each; <=250 GiB temporary and 200 GiB retained per scene. | Complete exact K/w2c/group/array/provenance manifests. | Failure blocks that scene ledger; no partial scoring. |
| P02 | Existing flow can be used without guessing direction or validity. | Validate/adapt every scene flow NPZ against source images and analytic direction/cycle fixtures. | Six CPU Slurm jobs <=12 h; no flow generation in cycle v1. | Every used array and semantic field is sealed. | Invalid legacy flow requires a new registered generation cycle. |
| P03 | Complete label-free CSVL ledger is available. | Each scene P01+P02, all registered target cameras/times. | Six CPU Slurm jobs <=12 h. | Ancestry, risk, state, track, schema, and exact payload validation. | Correct geometry before labels or renders are opened. |
| P04/P05 | Training and evaluation inputs freeze before outcomes. | Six leakage-checked train-sidecar freezes; six evaluator freezes; cut oracle freeze after real labels. | CPU Slurm <=4 h per producer. | Prohibited-read, K-feasibility, formula, mask, LPIPS, label-alias, and hash checks pass. | Block only the dependent representation/scoring nodes. |

### A00 required fixtures

Camera transform and target exclusion; native/processed K round trips; off-axis z
versus ray distance; group selection and degeneracy; duplicate weighted-median
ties; covariance transport; reciprocal fusion/exclusivity; two-layer z order;
patch/raster ownership and spatial false positives; flow schema/direction/cycle;
split/merge and bounded reappearance; risk `[0,1]`; event-window FP assignment;
annotation aperture/adjudication; Hungarian matching; threshold fraction ties;
calibration bins/bootstrap seed; canonical manifests; malformed/nonfinite/crash
paths.

### A04 interpretation

A04 may establish camera/schema correctness, repeatability, valid projection,
label-free cross-view/cycle agreement, and support compactness. It cannot pass
real ordering, event, spatial, or calibration gates without independent human
rows. If those labels are unavailable, A05 is produced and A06 remains pending.

## Slice B admission registry

Slice B does not begin until its operator contract receives independent code and
method review. Without a human Gate A pass, all inferred-coupling results are
explicitly exploratory and cannot establish the coupled claim.

| ID | Claim or uncertainty | Data / comparison | Fidelity | Decisive result | Decision and artifacts |
| --- | --- | --- | --- | --- | --- |
| B00 | In-place reassignment is point-neutral, optimizer-complete, and restart-safe. | Tiny Gaussian/optimizer tensors; no-op, null reset, exact reassign; injected crash at pre/applying/applied. | Login CPU plus tiny GPU Slurm <=30 min if device-specific. | Exact row/total budget; every per-row state handled; selected moments reset; tensor step/RNG/sampler/scheduler restored; atomic replay deterministic. | Failure blocks training. Track operator/tests; ignore smoke checkpoint. |
| B01 | Capacity-only operation is stable before visibility is credited. | Cut seed-0 iteration-5000 common checkpoint; no-op versus rate-matched generic trigger. | Fixed 5001-5250, K=256 for capacity. | Finite optimization, exact budget, valid render, no catastrophic early static/global harm. | Failure diagnoses operator/budget/optimization before coupling. |
| B02 | The mandatory correctly reprojected surface-level human oracle can make the fixed operator affect intended intermittent surfaces. | Frozen cut development-test human tracks; target xyz/RGB comes only from annotated visible training-camera polygons plus calibrated DA3. Cam00 supplies evaluation only. | Fixed 5001-5250, K=256. | Feasible assignments/target use; intended render direction; exact budget; no early static catastrophe. | Failure attributes sidecar, feasibility, operator, or optimization. At most the contract's moment-initialization repair before pivot. |
| B03 | Inferred evidence and reassignment have a causal benefit beyond each factor or any sparse signal. | Frozen cut seed-0 iteration-5000 checkpoint: route0, null-reset, capacity-only, oracle-capacity, visibility-only, full, shuffled. Requires genuine A06 engineering pass and B02 admission. | Fixed 5001-6000, K=2048 for mutation lanes. | Full beats route0 and single-factor controls; null-reset isolates moment surgery; shuffle does not reproduce gain; all static/flicker/ghost/budget conditions pass. | Registered B03 decision emits admitted checksum or exact failure attribution; no arbitrary sweep. |
| B04 | Frozen admitted method generalizes rather than overfits cut. | Exact six scenes in the split manifest; route0/capacity-only/visibility-only/full/shuffled. Seed 0 first; seeds 1 and 2 only under the frozen matrix predicate. | Fixed 6000 iterations, <=600k, per-run resource maxima in matrix. | Available per-scene/aggregate Gate B, no scene-wide static failure, quality-budget and flicker/ghost; human event claims only on cut/flame/sear. | Complete matched seeds, start a versioned revision, Route 3 fallback, or retire. |

## Gate B causal controls and metrics

The Slice B contract freezes the operator, oracle construction, donor/target
rules, K, trigger, initialization, transaction, masks, event horizon, metrics,
seed repair, and the only permitted optimization repair.

The common checkpoint, data exposure, optimizer steps, learning-rate schedule,
trainable parameter count, K, point ceiling, and realized/integrated budget are
matched or disclosed. B01/B02 pilots begin with zero moments. The unconditional
B02 trigger emits a typed `run_repairs` or terminal `not_applicable` record;
the unconditional join consumes the appropriate branch and B02 emits exactly
`zero` or `coordinatewise_lower_median_v1`. Every B03/B04 mutation and B03
null-reset consumes that policy artifact. Every continuation consumes the exact hash from its
registered common-checkpoint producer and writes a distinct lane checkpoint;
reuse entries prove exact source/config/evaluator hashes and consume no GPU. Capacity-only uses the same operator/schedule with a
generic rate-matched trigger. Visibility-only changes observation weighting but
not topology/capacity. Misaligned evidence keeps the rate and operator while
destroying camera/time identity. The human oracle sidecar uses visible training-camera polygons and calibrated
DA3 for target xyz/RGB; cam00 labels remain evaluation-only and it never
supervises the reported non-oracle lane.

Checkpoint-backed metrics are event-region PSNR/LPIPS, fraction of event windows
improved on both, all-300-frame static PSNR/perceptual/reconstruction-L1
no-harm, flow-relative flicker, reveal ghost trails, global metrics,
realized/integrated point count, active splats,
memory/wall time, assignments/preservations/reinitializations, scene/seed
consistency, and secondary R009/oracle-gap diagnostics. Practical targets remain
`+0.20 dB` and `-5% LPIPS`; hard conditions and static bounds remain those in the
canonical objective.

PSNR uses the frozen MSE floor; zero-denominator relative error, empty masks,
minimum valid static frames/pairs, required event offsets, and non-evaluable
propagation follow the exact Slice B/config rules. Visibility weight is
`1+0.5*(1-state_risk)` on accepted source-camera visible/reveal raster pixels
with max-confidence/lower-track-ID overlap. The shuffled control uses the
Slice B contract's library-independent SHA-256 domain-separated cyclic offset
to permute complete camera/time confidence maps without fixed points, preserving
their value/compactness multiset while destroying identity.

If the oracle-capacity lane fails, first inspect sidecar alignment,
assignment feasibility, total-budget accounting, operator realization, gradients,
and checkpoint/render validity. Permit one registered optimization repair only.
If oracle-capacity succeeds and inferred evidence fails, attribute evidence or
calibration. If capacity-only helps but coupling hurts, diagnose association. If
all signals fail, change representation only under the documented Route 3 pivot,
not by widening masks or sweeping lifecycle parameters.

## Promotion, freeze, and retries

1. A00 and A01 must pass before substantial work.
2. A02 and A03 must pass before real DA3 sidecars.
3. A04 can admit software, never the real Gate A claim.
4. A06 alone decides real Gate A; absent labels leave it pending/not evaluable.
5. B00 must pass before training.
6. B01 tests operator stability. B02 is the exact human-oracle capacity lane.
7. A06 engineering admission and B02 admission are both required before B03;
   any pre-A06 capacity diagnostic is exploratory and cannot change Gate A.
8. Only a B03-admitted checksum may enter B04.
9. After the first complete B04 comparison, spend compute preferentially on
   matched seeds, intervals, decisive causal controls, and scene robustness.

Infrastructure retry requires a terminal non-scientific failure, diagnosed log,
new run ID, lineage, and an squeue/sacct/output duplicate check. A threshold or
configuration revision after observed results creates a new named cycle and
cannot reuse the same claim-grade holdout as fresh confirmation.

## Expected artifact classes

Tracked: method/gate/schema/config/launcher/test sources, immutable manifests
without arrays, annotation schema/empty packet manifest, run matrix, analyses,
decisions, wiki pages, and graph/log updates.

Ignored/outside Git: raw/generated depth and flow arrays, ledgers, contact-sheet
images unless intentionally promoted to tracked wiki assets, checkpoints,
renders, scheduler logs, W&B payloads, bootstrap samples, and temporary
verification outputs. Every ignored result referenced scientifically has a hash,
path, job/run provenance, and durable conclusion in the report.
