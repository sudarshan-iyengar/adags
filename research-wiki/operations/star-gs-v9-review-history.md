# STAR-GS v9 — Adversarial Review History (durable record)

Date: 2026-08-08. Reviewer: GPT-5.6-Sol via Codex MCP at xhigh reasoning;
fresh context per adversarial round (no shared thread with the refinement
loop). This page preserves the substantive scientific content of the
review chain; verbatim transcripts remain in transient `refine-logs/`
(round-1..5-review.md, adversarial-round-1..4-review/redesign.md) and are
NOT required to understand or reproduce the outcome.

## Stage 1 — research-refine loop (5 rounds, one reviewer thread)

Candidate: "Support-Matched Supervision" (residual routing -> support-
matched momentum). Score trajectory 6.1 RETHINK -> 7.0 -> 7.8 -> 8.0 ->
9.1 READY. Decisive technical findings, all verified and accepted:

- The claimed sequence-gradient exactness of residual routing is false:
  with time-varying Jacobians J_{t,i} correlated with residuals, the
  correct per-primitive statistic is the render-weight-weighted temporal
  sum, not the unweighted mean residual; a per-camera persistent-residual
  buffer is not any primitive's support projection and can bake an
  average foreground "veil" into background primitives.
- An EMA of rendered images mixes historical parameter states and is not
  the current model's time-mean render.
- Per-primitive gradient blending cannot pass through shared LoRA
  parameters after aggregation; L1's sign-gradients and D-SSIM are not
  covered by an L2 argument.
- Reformulation as per-row Adam momentum horizons (SMM) was judged
  design-complete at the loop's end (9.1) with a filter-replay gate — a
  verdict subsequently overturned by fresh-context review (below),
  demonstrating why the project requires fresh-context adversarial
  passes after any same-thread refinement loop.

## Stage 2 — fresh-context adversarial rounds

### Round 1: SMM (support-matched momentum horizons) — SINKS
Fatal findings (checked and accepted): (a) the support->horizon mapping
direction is unidentified and arguably reversed — for observation
probability p the proposed mapping yields < 1 informative observation per
horizon at small support, while sparse-supervision logic wants wall-clock
horizon ∝ 1/p; (b) dense Adam actively moves occluded/invisible rows
along stale momentum for ~1/(1-beta1) iterations (residual step decays as
(beta1/sqrt(beta2))^k), so long horizons WORSEN hidden-surface corruption;
(c) "Adam's second moment absorbs visibility fraction" is false
(E[m]/sqrt(E[v]) ~ sqrt(p)*sign(a)); (d) off-policy filter replay cannot
validate the causal claim; (e) novelty ceiling: parameter-group Adam,
TTUR/two-timescale SA, SparseAdam/LazyAdam, AggMo/QHAdam/AdEMAMix,
LARS/LAMB, Taming-3DGS selective Adam; the two-bank fallback is ordinary
parameter groups.

### Round 2: OC-GS (observation-clocked optimizer state) — SINKS
Fatal findings: (a) under a fixed training length, clocked updates give
rarely-observed primitives LESS total optimization (E[update/iter] ~
alpha*p < dense's ~ alpha*sqrt(p)) — the correction does not follow from
the diagnosed pathology; (b) fully clocked moments preserve arbitrarily
stale momentum to reveal time; the natural alternative (skip steps, decay
or reset moments) is exactly existing selective/sparse Adam; (c)
supervision gaps in optimizer time arise from SAMPLING, not occlusion —
all-view occlusion is rare on this rig (consistent with the project's own
census), so the occlusion narrative fails; (d) the proposed Phase-A
diagnostics were tautologies of the optimizer equations, not evidence of
harm; (e) unseparated implementation-level collision with gsplat
sparse_adam / Taming selective Adam. Conclusion recorded: optimizer-
semantics interventions are causally unidentified pre-experiment and have
a low novelty ceiling; family closed.

### Round 3: STAR-GS v7 (residual-epipolar triangulation) — SINKS
Fatal findings: scalar residual clusters carry no correspondence signal —
epipolar geometry constrains where a match may lie but cannot say which
residual cluster matches which; repairing with RGB/feature matching
collapses into the STG/VAD-GS/ConeGS birth family; "structurally
unreachable" overclaimed (finite rasterizer support, migration,
multi-step reachability); residual-self-consistency precision gates are
circular; temporal proximity-linking contradicted the
no-temporal-correspondence claim. The review explicitly left open a
"coherent third option": a volumetric K-view residual-consensus objective
with explicit null/occlusion treatment.

### Round 4: STAR-GS v8 (FDR-calibrated residual consensus) — SINKS
Fatal findings: per-camera cyclic time-shift permutations lack
exchangeability for nonstationary cooking video, so the "FDR" is not a
valid error rate; the tested null is residual synchrony, not missing
capacity; the controlled family (voxels vs components vs births vs the
adaptive event sequence) was undefined; empirical p-value resolution
infeasible at the required multiplicity. Confirmed positively: "no
demonstrated exact collision with the named systems" (VAD-GS, STG,
ConeGS, MCMC, FreeTimeGS relocation, space carving). Fixable-item
catalogue (all adopted into v9): construction/audit camera folds,
iso-compute control, interference-aware policy-level attribution,
WHEN-control matching strata, synthetic-injection labeling, hierarchical
statistics.

### Round 5: STAR-GS v9 (claims cut to a conservative proposer) —
SURVIVES-WITH-RISKS
"At the design level, this now survives... No fatal method or evaluation
defect remains. Strong results could, in principle, validate the system."
Binding obligations recorded in [[operations/star-gs-v9-method]]:
exact-conjunction wording ("we introduce", never "the first" —
FreeTimeGS relocation and SharpTimeGS stage-2 occupy budgeted
reallocation; CEC-4DGS occupies error-driven time-local birth);
count-matching to realized accepted births with identical
resets/init/retirement; WHEN-control strata from permitted signals only;
within-run attribution labeled "local direct effect under the mixed
deployment policy"; shifted-input collapse preregistered as diagnostic
only; occlusion abstention never called "conservative" unqualified.
Top risks (severity order): (1) "space carving + churn" incrementality
perception; (2) residual consensus finding non-capacity-fixable errors
(motion mismatch, specularity, blur, calibration) with ineffective
zero-motion births and donor erasure of rare content; (3) six-scene
statistical power and external validity.

## Stage 3 — novelty check (fresh Codex thread + targeted searches)

New decisive collision found by search and verified from full text:
[[papers/kang2025_cec_4dgs]] (CEC-4DGS, SIGGRAPH Asia 2025) — error-
clustered, cross-view-checked, TIME-LOCAL 4D birth localized at
single-view rendered depth, unbudgeted, on Ex4DGS, evaluated on N3V
(+0.12 dB global) and Technicolor (+0.42 dB). Referee verdict on STAR-GS
v9 given this: per-claim novelty C1 MEDIUM (depth-free multiview residual
carving vs rendered-depth backprojection is "a real algorithmic
distinction, not automatically an implementation detail" — but only with
localization-failure evidence), C2 LOW (composition), C3 MEDIUM
(benchmark half stronger). Overall **5.5/10, PROCEED WITH CAUTION**.
Viability conditions (binding on any future test): beat faithful AND
budget-matched CEC reimplementations on the same backbone; event-level
gains materially larger than global-PSNR gains; direct localization
evidence; component ablations failing as predicted. "~0.1-0.3 dB with no
localization evidence" = too incremental; cannot beat capacity-matched
CEC = abandon the dominant claim. Approved positioning: "budget-neutral
correction of depth-deficient dynamic Gaussian models through multiview
residual-space carving"; hostile summary to preempt: "CEC-4DGS with
voxelized aggregation plus a pruning swap"; honest rebuttal: CEC's
placement is conditioned on model depth that is systematically unreliable
at newly disoccluded surfaces; STAR-GS infers locations from
visibility-gated multiview residual evidence, with capacity- and
compute-matched controls isolating the localization change.

## Cross-cutting conclusion (durable)

Across ten reviewer passes, every claim inflation — exactness proofs,
statistical calibration guarantees, impossibility language, priority
claims — was independently fatal; conservative-proposer +
causal-validation framings survived. Same-thread refinement approval
(9.1 READY) did not predict fresh-context survival; fresh-context
adversarial review is the binding standard for this project.

## Links

- [[operations/star-gs-v9-method]] — the surviving method
- [[operations/star-gs-v9-experiment-plan]] — the preserved test plan
- [[operations/rejected-approaches-2026-08]] — rejected families + revisit conditions
- [[operations/sota-sweep-2026-08]] — literature/code findings backing the review chain
