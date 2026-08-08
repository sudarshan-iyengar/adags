# Rejected Approaches — 2026-08-08 method-discovery run

Ledger of serious candidates examined and rejected during the STAR-GS
selection run. Substantive review record:
[[operations/star-gs-v9-review-history]] (raw transcripts remain in
transient `refine-logs/` but are not required). Reviewer: GPT-5.6-Sol at
xhigh, fresh context per adversarial round. Each rejection lists whether
it is fundamental or revisitable with specific new evidence.

## 1. SMS-GS v1: timescale-routed residual attribution (per-camera
persistent-residual blending)

- Mechanism: decompose per-view residual into persistent (time-mean
  anchored) + transient parts; blend per-primitive by temporal support via
  two backward passes.
- Why it looked promising: unified account of ghost/corruption at event
  regions; exact under L2 for time-invariant contributors (so it seemed).
- Rejection (refine round 1, RETHINK): the sequence-gradient equivalence
  is false (time-varying Jacobians correlated with residuals — the correct
  statistic is render-weight-weighted, not unweighted mean); EMA-of-renders
  is stale w.r.t. the current model; per-primitive routing cannot pass
  through shared LoRA parameters after aggregation; L1/SSIM not covered;
  shared persistent residual can bake a foreground "veil" into background.
- Status: FUNDAMENTAL as formulated. A support- and contribution-weighted
  variant collapses into per-primitive gradient averaging (→ family 2).
- Reconsider only if: a formulation produces per-primitive
  support-projected residual targets WITHOUT shared-parameter routing and
  WITHOUT stale render averages (e.g., exact per-primitive accumulators
  proven unbiased under the actual sampler), AND a measured
  variance-reduction benefit exists at fixed topology.

## 2. SMM: support-matched per-primitive momentum horizons (per-row Adam
β1 from temporal support)

- Mechanism: β1_i horizons matched to temporal-coverage fraction; refined
  over 4 rounds to READY 9.1 with filter-replay gates.
- Why it looked promising: implementable, zero components, plausible
  variance/staleness story; refine-loop reviewer approved the design.
- Rejection (fresh-context adversarial round 1, SINKS): the
  support→horizon direction is unidentified and arguably reversed (for
  observation probability p the mapping gives <1 informative observation
  per horizon at small support; sparse-supervision logic wants H ∝ 1/p);
  dense Adam ACTIVELY moves occluded rows along stale momentum for
  ~1/(1−β1) iterations, so long horizons worsen the hidden-surface
  corruption the method claimed to fix; "second moment absorbs visibility
  fraction" is false (E[m]/√E[v] ≈ √p·sign(a)); novelty ceiling low
  (parameter-group Adam, TTUR, SparseAdam, AdEMAMix, LARS/LAMB,
  selective Adam); two-bank fallback = ordinary parameter groups.
- Status: FUNDAMENTAL for the mapping-based form. Revisitable only as a
  narrow empirical study with shadow-copy counterfactual instrumentation
  (per adversarial reviewer 2's "most plausible surviving paper").

## 3. OC-GS: observation-clocked optimizer state (tick-based moments/
steps/bias correction on a supervision clock)

- Mechanism: per-primitive Adam advances only on supervised observations
  (ω = Σw² above floor); clocked densification statistics on same clock.
- Why it looked promising: derivable 1/p wall-clock scaling; freezes
  hidden rows (subsumes round-1 protection); one clock object unifies
  optimizer + densifier; closed-form stale-momentum pathology analysis.
- Rejection (fresh-context adversarial round 2, SINKS): under a fixed
  training length, clocked updates give rare primitives LESS total
  optimization (E[update/iter] ≈ αp < dense's ≈ α√p) — the correction
  does not follow from the pathology; full moment clocking preserves
  arbitrarily stale momentum to reveal time; supervision gaps in
  optimizer time come from SAMPLING, not occlusion (all-view occlusion
  rare on this rig — consistent with our census), so the occlusion
  narrative fails; unseparated collision with gsplat sparse_adam /
  Taming selective Adam (delta ≈ per-row bias correction + threshold);
  Phase-A diagnostics were tautologies (optimizer equations), not harm
  evidence.
- Status: FUNDAMENTAL as a headline. The step-skipping HALF (freeze
  parameters, decay-or-reset moments) remains a sensible engineering
  option ≈ selective Adam, usable as a control, never a contribution.
- Reconsider only if: shadow-copy counterfactual instrumentation (a
  frozen twin of each row during unsupervised gaps) demonstrates
  measurable next-observation damage from dense-clock semantics on
  dynamic scenes, AND the clocked-bias-correction delta over gsplat
  sparse_adam / Taming selective Adam is shown to matter empirically —
  and even then, as an empirical-study paper, not a mechanism claim.

## 4. STAR-GS v7/v8 constructor variants (residual-epipolar triangulation;
FDR-calibrated residual consensus)

- v7 rejection (adversarial round 3, SINKS): scalar residual clusters
  carry no correspondence signal — epipolar matching of residuals is
  underdetermined; adding RGB/feature matching collapses into
  STG/VAD-GS/ConeGS territory; "structural unreachability" overclaimed;
  precision gate circular; temporal proximity-linking contradicted the
  no-temporal-correspondence claim.
- v8 rejection (adversarial round 4, SINKS): the per-camera cyclic
  time-shift permutation null lacks exchangeability for nonstationary
  video; the tested null is residual synchrony, not missing capacity;
  the FDR family (voxels/components/births/adaptive events) undefined;
  empirical p-value resolution infeasible at the required scale.
- Status: FUNDAMENTAL for the calibration CLAIMS; the underlying carving
  machinery survives in v9 with claims cut to a conservative proposer +
  causal validation (adversarial round 5: SURVIVES-WITH-RISKS; novelty
  check 5.5/10 PROCEED WITH CAUTION). See
  [[operations/star-gs-v9-method]].
- Reconsider the calibration claims only if: a null with
  exchangeability-by-construction is found (or conservative bounds
  calibrated on synthetic injections are accepted as the weaker claim),
  AND the FDR family is defined over the actual birth decisions including
  the adaptive event sequence. Until then, v9's uncalibrated conservative
  selection is the standing form.

## Cross-cutting lessons (recorded for future method work)

- In 2026's saturated dynamic-GS field, every SIMPLE training-side
  mechanism axis we examined is occupied at mechanism level:
  presence-weighted densification (TAD-GS, 4D-Scaffold-GS), budgeted
  relocation (FreeTimeGS, 3DGS-MCMC, SharpTimeGS stage-2), error-driven
  time-local 4D birth (CEC-4DGS), visibility-gated gradient masking
  (WildRayZer; PackUV/MAPo freezing), learned uncertainty gradients
  (U-4DGS), pixel rejection (robust-GS), sparse optimizer semantics
  (gsplat/Taming), temporal anti-aliasing (Alias-free 4DGS), lifespan
  shaping (SharpTimeGS), time warps (TAD-GS TOW).
- Claim inflation is what dies under adversarial review: exactness
  proofs, statistical calibration, impossibility language, and priority
  ("the first") claims were each independently fatal; conservative
  proposer + causal-validation framings survive.
- Pre-experiment causal identification of optimizer-level interventions
  is extremely hard; capacity-allocation interventions have shorter,
  auditable causal paths (site-localized, attributable).
- The project's matched-capacity/causal-control discipline and the L5
  time-shift lesson repeatedly earned reviewer credibility — they are
  assets to keep at the center of any future proposal.
