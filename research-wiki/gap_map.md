# Gap Map

Updated 2026-08-08 (night) after Loop 2 ([[operations/elgs-method]]).
Earlier updates remain binding history.

## Audit-Instrument Update — 2026-08-18 (block 4)

**G13's absence limb does not get its decision this block, and the reason is
worth more than the decision would have been.**

The two-stage M1-A0b audit — the only instrument that could convert the
DiVa-360 absence question from open to decided — reached revision 3 with its
sequence universe reconciled and its applicable-camera set adopted, and was
then **BLOCKED at round 2** by a fresh-context statistical review
([[operations/elgs-audit-prereg-2026-08-18]] REVIEW ROUND 2). **Stage 1 was
not executed; no candidate frame was rendered, displayed or transmitted.**

The reconciliation and the camera decision both stand, and are new durable
structure:

* the sequence universe is **20 coverage-population sequences → 18 eligible →
  12 with `N_s > 0` → `E_select` = 10**, and the simultaneous-bound divisor is
  **m = 10**. The "16 more eligible", the 10, and the 11-vs-12 were three
  different sets, not a contradiction;
* the applicable-camera set is **D3**, the sealed per-candidate frustum rule,
  adopted by the user. The estimand narrows to *unobservability across the
  cameras the frozen candidate generator geometrically considered applicable,
  followed by same-identity reappearance* — **not** physical absence, **not** a
  rig-wide set, **not** generator-independent supply.

**The new negative knowledge is about inheritance, and it generalizes past this
audit.** Two of the four blocking findings are cases of a preregistration
inheriting a sample and an instrument from an earlier frozen document, carrying
the sample's *identity* forward, and silently dropping the *design facts* that
came with it:

* **the sampling design was not carried.** The frozen 73-window sample is a
  round-robin over strata, 3 rounds, without replacement. Per-sequence
  Clopper-Pearson on `(k_s, n_s)` assumes equal within-sequence inclusion
  probabilities, which requires `3·N_s/n_s` to be an integer. **Verified: it is
  not, for five sequences including `pour_tea` and `tea` — two of the four that
  decide the outcome.** So the decision-bearing bound is not the confidence
  bound the kill rule treats it as, while the design-consistent
  Horvitz-Thompson estimator sits in the same file with no decision weight;
* **a disclosed bias was not carried.** The parent document discloses that
  audit frames are drawn over a window that *includes bridged frames, at which
  the identity is associated by construction of the flicker rule* — so a
  bridged window can never be A3-positive whatever the scene contains.
  Deflated A3 → **kill easier**, the wrongful-kill direction. **Verified: the
  word appeared zero times in either audit document.**

**The lesson to carry into every future instrument:** citing a frozen artifact
by name inherits its identity, not its assumptions. A preregistration that
reuses a sample must restate that sample's design, and must re-import every
disclosed bias, inside its own frozen text — or its estimator is being applied
to a design it has never seen.

Two further verified facts that bound what any repair can claim:

* **the finite-population omission is not a rounding matter.** `U_s` routinely
  exceeds the logically attainable maximum `k + (N − n)` — for `tea` at every
  `k`. Capping it moves the kill from **2,580 to 3,141 of 7,000**, a 21.7%
  shift, and `tea`'s published "6 of 9" threshold is unattainable in fact;
* **stage-1 decidability was never general.** It exists only because `scissor`
  and `poker` are excluded, on a frozen sensitivity reading sitting 0.003 and
  0.015 below a threshold. Verified in git that the exclusion is **not**
  post-hoc: the freeze commit predates every coverage figure.

**Not changed:** G-OA's FAIL is not reopened; no floor, threshold or estimand
moved; `scissor` and `poker` remain `indeterminate`; and the 0-of-597
corroboration figure stands exactly as it did. What changed is that the route's
*decision* is now blocked on statistics rather than on compute, and the block
is repairable.

## Dataset-Reachability Update — 2026-08-18 (block 4)

**ImViD's full release is reachable after all, and the previous block's
contrary finding is corrected rather than rewritten**
([[operations/dataset-admission-matrix-2026-08-18]] block-4 append). It is
world-readable with no credential: 325 files, 1.181 TB, verified by the primary
and enumerated two independent ways. But only **7 of 16 published takes** are
there (47.2% of the published bytes), `scene3_classroom` is missing a camera,
and a 122 GiB `moving_rig` folder is unaccounted for in the README's own total.
**This changes cost and reachability, not admissibility** — ImViD remains NOT
ADMITTED for event supply.

**Google Immersive's 46-videos-vs-45-calibrations gap is resolved against the
sealed artifacts**: the uncalibrated file is `camera_0046`, and `camera_0001` —
STG's held-out test view — is present and calibrated. So it is a per-scene
calibration failure whose index **moves between scenes**, and any visibility
ledger built by enumerating `camera_*.mp4` will mis-index differently on
different scenes. The dome's event-supply exclusion now carries a number: the
whole 46-camera rig spans ~50° of parallax at 1 m against a surround rig's
180°, so the multi-view corroboration the absence instrument needs is not
merely hard there — **it is undefined**.

## Coverage-Instrument Update — 2026-08-18

**G13's absence limb: the coverage floor that eliminated the two
richest sequences was an artifact of the falsified visibility gate.**
Retracts nothing; it changes which sequences the absence question can
still be asked about, and it sharpens a coupling this map already
recorded.

The 2026-08-14 entry above notes that `track_coverage_upper_bound` and the
absence limb share the `v >= 0.5` constant, so part of their
anticorrelation is an instrument identity. M-2 then showed `v` is BINARY,
which makes every threshold in `(0, 1]` equivalent and the
"lower-the-threshold" repair vacuous. The repair the evidence supports is
a component-membership gate, and its effect on coverage has now been
measured ([[operations/elgs-coverage-bounding-pair-result]], Determined
experiment 154, all five contract checks exact):

| sequence | frozen `v >= 0.5` | anchor-agreeing | any-report (upper) |
|---|---:|---:|---:|
| `scissor` | 0.441 | **0.852** | 0.916 |
| `poker` | 0.382 | **0.796** | 0.834 |
| `put_candy` | 0.507 | 0.727 | 0.861 |
| `pour_tea` | 0.591 | 0.710 | 0.795 |

So **the 0.5 coverage floor did not eliminate scissor and poker because
their foreground was untracked** — it eliminated them because a per-point
self-occlusion flag was being read as an association signal. Both clear the
floor by more than they previously fell short of it.

**They are nevertheless classed `indeterminate`**, because the frozen
convention-dependence rule demotes on a sensitivity crossing and the
transposed-anchor reading lands at 0.497 and 0.485. The rule was frozen
before the reducer existed and was not changed after the numbers; the
demotion is conservative and the transposed variant is diagnosed on the
page as a null rather than a rival convention (it admits 4.5-11x fewer
reports).

**The consequence is structural and it is new knowledge about the route,
not about the scenes.** Excluding scissor and poker removes 452 of the 597
candidate windows, and that exclusion is the only reason the amended
audit's stage-1 kill rule is decidable at all: with scissor, poker and
pour_tea all admitted, no possible audit outcome can fire the kill, because
`U_s = N_s * UB_s` scales with the candidate count while the bound is
floored by the sample size ([[operations/elgs-audit-prereg-2026-08-18]]
section 4). **A larger candidate population makes the kill harder.** That
is a preregistered tripwire now, and it means any future rule that admits
those two must re-specify stage 1 first.

**Nothing here reopens G-OA's FAIL, changes a floor, or admits any sequence
to evidence use.** Admission requires a fresh preregistration under a
corrected instrument.

**One further open item, recorded because it bounds every A3 and A_S count
the audit could produce:** the audit's presence decoys bound the
instrument's false-positive side only, and nothing estimates its miss rate
on genuine full-view absence, because no DiVa-360 window is known to be
absent. That is the specific hole
[[operations/kubric-testbed-scope-2026-08-18]] scopes, and it is not
closable on real data.

## Renderer Update — 2026-08-18

**The admitted image is reproducible run-to-run; the old one was not.**
Three runs of the repaired kernel at fixed seed agree to **3.3e-4 dB** of
held-out PSNR, against **0.10 dB** (same metric) and **0.36 dB**
(training-log metric) between two old-image runs
([[operations/renderer-integrity-admission-2026-08-18]] Appendix C). This
resolves that page's open question in the direction of the repair, and it
bounds the `atomicAdd` explanation empirically: if float-summation order
were the dominant source here, the repaired image would vary similarly.

Consequence for every lane: single-run-per-arm comparisons **on the
admitted image** no longer inherit the 0.36 dB penalty in that
configuration. The figure does not transfer to 15k DiVa-360 runs, and the
matched presence spec still measures its own spread.

## Measurement-Closure Update — 2026-08-14

**G13's ABSENCE limb loses its measured supply; its OCCLUSION limb keeps
it.** A frozen, four-times-reviewed diagnostic over all 597 corrected
DiVa-360 tranche-1 true-absence windows returned **status_2 (material
defect), UNANIMOUS across 144 sensitivity readings**
([[operations/elgs-absence-diagnostic-result]]):

- **ZERO of 597** scored true-absence windows are corroborated as genuine
  full-multiview disappearance — zero pooled and zero in every one of the
  twelve sequences.
- **96.6%** are windows where an eligible foreground component sustained
  multi-view-consistent occupancy of the instrument's own frozen anchor
  while the tracker's report failed to qualify: **87.6%** of the evidence
  is a per-point visibility flag below 0.5, **12.2%** is cameras in the
  applicable set that were never queried and can never associate. The
  tracker never LOST the point (C2 = 0 everywhere).
- Because `track_coverage_upper_bound` uses the SAME `v >= 0.5` threshold,
  coverage and absence are coupled through one constant — part of the
  measured coverage/absence anticorrelation (r = -0.765) is an instrument
  identity, not a scene fact. Occlusion, which requires association in
  >= 2 cameras (tracking WORKING), is barely coupled (r = -0.178).

**New negative knowledge.** A tracker visibility flag is a per-point
self-occlusion signal, not an existence signal: on a surround rig a surface
point can be self-occluded in every queried camera while the object is
plainly present. Any future presence/absence instrument must separate
"unobserved" from "absent" by evidence that does not reduce to the tracker's
own confidence, and must not let its applicable-camera set include cameras
it never queried.

**Not concluded:** that the objects were physically present. C2/C3 cannot
separate "component still there, untracked" from "identity left and the
manipulating hand covers the vacated site". Only the frozen M1-A0b audit
can, and its 73-window stratified sample has been emitted but NOT run.

G14/CC4 is unchanged by this: it already had zero measured supply
([[operations/elgs-substrate-remeasurement-result]]). G9's
uncertainty/abstention need is sharpened — the instrument had no abstention
class at all until this diagnostic introduced one.

## Loop-2 Update - 2026-08-08 (night)

Under user-relaxed constraints (external priors; any public dataset;
per-scene fixed), three verified sweeps ([[operations/loop2-sweep-2026-08]])
found: (i) tracker visibility states have NEVER been consumed as
representation-level presence/identity (MoSca/SoM loss masks only);
(ii) no existence inference with an observation model exists in
differentiable rendering (CIF nearest, segmentation-scoped); (iii) the
occlusion-order+memory conjunction remains unoccupied even on
surround/ego rigs (ST-NeRF verified to lack persistent hidden state;
no GS successor); (iv) non-rigid permanence through occlusion on
surround capture and egocentric hidden-state dynamic GS are open;
(v) DiVa-360 is the event-dense benchmark with no GS baselines.
EL-GS occupies (i)+(ii) with the LGS substrate — G13/G14 now have a
candidate occupant with a calibrated 8.0 conditional novelty; the
remaining gate is formal (v8 write-out + one fresh adversarial round).
G9 (uncertainty/occlusion confidence) is directly addressed by the
censored-evidence ontology (conditional claims, no calibration).
New negative knowledge: track-state⇒existence naive mappings are
invalid measurement semantics; e-process validity claims for adaptive
structural acceptance are unsupportable; per-segment bridge selection
creates chimera evidence (all ledgered in
[[operations/elgs-review-history]]).

## Post-Representation-Run Update - 2026-08-08 (evening)

Five verified representation-level sweeps + nine deep-dives + five
fresh-context adversarial rounds + a calibrated novelty check
([[operations/repr-sweep-2026-08]], [[operations/lgs-novelty-record]])
tightened the map:

- **G13 (visibility events)**: the representation-level boundary is now
  precise: per-primitive multi-interval/reactivating presence, latched
  presence, per-primitive changepoints, and exact compact-support
  absence are VERIFIED UNOCCUPIED across all eight dynamic-GS families;
  discrete lifecycle exists only in streaming methods and always births
  NEW rows. [[operations/lgs-method]] occupies this slice (selected
  candidate, 6.5/10 novelty, awaiting user decision). Near misses to
  cite: Ex4DGS (single flat-top), CTRL-GS (scene-global segments),
  TOM-GS (presence-only single bump), AD-GS/TRiGS (single window),
  CLOTH-HUGS (order without memory), CIF (occupancy × semantic
  identity), PersistGS/4DPM/MoPe (pose/log-odds permanence), CubifyGS
  (frozen assets, discrete maintenance), TSA (2D slot activation).
- **G14 (identity-conserving promotion)**: reactivation-with-own-content
  is confirmed unoccupied AND mechanistically opposite to the entire
  relocation/respawn family (3DGS-MCMC donor-clone overwrite verified
  at code level; FreeTimeGS++ 2605.03337 ablates that family). LGS's
  lineage tying + reactivation is the candidate occupant; its
  irreversible-fragmentation limitation (no merge) is recorded.
- **G5 (capacity)**: counterfactual trial-render structural acceptance
  is unoccupied (closest: L2D2-GS 2606.29374, offline policy reward);
  it enters LGS as supporting machinery only.
- **New negative knowledge** ([[operations/rejected-representations-2026-08]]):
  occlusion-order layer stacks are rig-hostile here (P03); ratio-based
  description-cost economics is undefined/gameable as a principle; soft
  content assignment over a candidate library has no coherent geometry
  under migration and contradicts strict scalar caps; same-thread refine
  approval again failed to predict fresh-context survival.

## Post-Method-Discovery Update - 2026-08-08

Five verified literature sweeps + nine paper deep-dives + four
fresh-context adversarial reviews substantially tightened the map:

- **G5 (capacity allocation)**: [[operations/star-gs-v9-method]] is the
  preserved training-side candidate on this axis (deficit-carved,
  budget-neutral spacetime birth) — NOT the approved lead direction; the
  next phase is representation-level discovery. Its test plan is
  preserved at [[operations/star-gs-v9-experiment-plan]], review record
  at [[operations/star-gs-v9-review-history]], sweep evidence at
  [[operations/sota-sweep-2026-08]]. Occupied neighbors verified at mechanism level:
  CEC-4DGS ([[papers/kang2025_cec_4dgs]]) = error-driven time-local 4D
  birth at single-view rendered depth (unbudgeted); FreeTimeGS = periodic
  budget-neutral relocation to existing high-score regions; SharpTimeGS
  stage-2 = fixed-count error/motion densification; TAD-GS +
  [[papers/cho2026_4d_scaffold_gs]] = presence-weighted statistics. The
  residual open slice: depth-free multiview deficit localization +
  audited budget accounting + causal/event validation.
- **G13 (visibility events)**: WildRayZer (CVPR 2026 Highlight) occupies
  learned transient-mask gradient gating; [[papers/mazur2026_4dpm]]
  (CVPR 2026 Oral) occupies primitive permanence via motion extrapolation
  (monocular, rigid). Optimizer-level "protection" approaches were
  examined and rejected this run ([[operations/rejected-approaches-2026-08]]).
- **G7 (evaluation)**: externally corroborated — ViDAR
  ([[papers/nazarczuk2025_vidar]]) quantifies co-visibility-mask static
  bias (mean 26% dynamic pixels) and establishes -D dynamic-mask metrics;
  TAD-GS's M-PSNR is precedent. The field has no standard temporal
  metric (tOF/tPSNR borrowed ad hoc) — adopt, don't invent.
- **New negative knowledge** (review-derived, recorded in the rejected
  ledger): per-primitive optimizer-timescale interventions are causally
  unidentified pre-experiment and collide with sparse/selective-Adam
  tooling; residual images carry no cross-view correspondence signal;
  time-shift permutation nulls lack exchangeability for nonstationary
  video; static-scene densification theory now has three distinct
  accounts (SteepGS saddle points, GDAGS direction coherence,
  Structure-Aware aliasing) — cite, don't re-derive.

## Post-Stage-1 Update - 2026-07-29

Stage 1 of CSVL-VPL was executed on 2026-07-26 and returned three no-gos
([[operations/phase9-csvl-vpl-stage1-result]],
[[operations/phase9-csvl-vpl-stage1b-result]],
[[operations/phase9-csvl-vpl-stage1c-result]]). This falsifies the 2026-07-25
statement below that "the first unresolved gap is uncertainty-bearing temporal
surface association and abstention": the sealed P03 evidence layer contains
zero front/rear cross-order candidates in all 19 scanned windows, so there is
nothing for a temporal association to associate. The binding constraint is the
evidence representation itself — multilayer bin occupancy requires multi-camera
co-support of two depth layers in one bin, which a frontal rig almost never
produces (93.4% of 3.07M bins rejected for insufficient camera co-support).

The approved direction is [[operations/phase9-csvl-vpl-v2-direction]]:
primitive-centric reprojection visibility (E1/E1-int/E2) replacing the P03 bin
route, a from-scratch lifecycle that never touches rendered opacity, a restored
oracle-capacity attribution lane, and an evidence-opportunity census (Phase 0)
gating all further evidence investment. G9 and G13 remain the target gaps; the
first unresolved question is now empirical opportunity abundance in the
primitive-centric representation, not association design.

Additional negative evidence recorded for G13: the Stage-1 association scored
camera-swapped flow above valid flow; flow was non-causal for its output; and
the R034 synthetic fixture (AUC 1.0) predicted nothing about real admission
(R035 accepted 0/72). Fixture passage must never again be a Go criterion.

## Post-B01 Update - 2026-07-25

The corrected 256-slot B01 continuation is an operator-stability control, not
mechanism evidence: global PSNR improved by only `+0.048315468 dB`, dynamic-mask
PSNR by `+0.011161912 dB`, and static PSNR by `+0.055157407 dB`. It used an
event-blind target rule and therefore did not test calibrated visibility-guided
allocation.

The selected direction is [[operations/phase9-post-b01-csvl-vpl-direction]]:
CSVL-VPL, a calibrated surface visibility ledger coupled to a surface-owned
primitive lifecycle. This refines G5, G9, G13, and G14 together. Fixed-count
reassignment remains a matched-count control and optimizer-safe transaction
substrate; generic extra capacity is mandatory as the capacity control.

The sealed P03 artifact supplies calibrated multilayer opportunity evidence but
does not propagate persistent surface identity. The first unresolved gap is
therefore uncertainty-bearing temporal surface association and abstention, not
another capacity intervention.

Implementation-level novelty pressure is stronger than the earlier paper-only
map: temporal-visibility densification, opacity modulation, proxy-guided growth,
multi-bank promotion, and layered representations all have close precedents.
The narrower open hypothesis is calibrated non-rigid front/rear surface identity
plus abstaining evidence and controlled surface-owned lifecycle changes.

## Tournament Update - 2026-06-30

The selected ADAGS direction is [[ideas/self-calibrated-prior-reliability-field]]. It addresses G1/G2/G3/G7/G8/G9/G11 by making reliability the mechanism that decides where masks, flow, tracks, static exclusion, and detail priors may act. [[ideas/boundary-aware-static-anchor-negative-space]] is the deterministic baseline/static-leakage safeguard. [[ideas/adags-failure-atlas-mechanism-screen]] is the backup and required reporting frame.

New blocking gaps:

- Reliability must be calibrated as an error/usefulness predictor, not just a mask recipe.
- Evaluation masks must be independent enough to avoid circular validation.
- Reliability must retain hard dynamic-core pixels; easy-pixel selection invalidates the method.
- Wins must be shown against LoRA route0 under matched realized budget and static-quality preservation.

## Problem-First Redo Update - 2026-06-30

The problem-first redo deliberately treats ADAGS as prototype infrastructure, not the method boundary. It demotes reliability-gated priors to the safe ADAGS fallback and elevates representation-level questions that still remain after Multi4D, RiGS, SharpTimeGS, AdaGaR, MAPo, PaMoSplat, MoE-GS, USplat4D, Ground4D, MoSca, and Prior-Enhanced GS.

New high-upside candidate directions:

- [[ideas/event-causal-visibility-gaussians]]: visibility events for occlusion, disocclusion, birth, split, merge, and retirement.
- [[ideas/identity-conserving-detail-carriers]]: parented transient detail carriers that preserve identity while recovering high-frequency motion detail.
- [[ideas/frequency-adaptive-temporal-support]]: temporal support bandwidth tied to dynamic frequency/detail and uncertainty.
- [[ideas/counterfactual-prior-usefulness-routing]]: route priors by estimated downstream usefulness, not confidence alone.

New blocking gaps:

- G13: occlusion/disocclusion are still often modeled as smooth deformation or implicit lifespan effects rather than causal visibility events.
- G14: dynamic detail can be recovered by transient capacity, but identity-preserving promotion/demotion rules remain underdeveloped.
- G15: prior confidence is not the same as prior usefulness; the field lacks counterfactual tests for when masks, tracks, flow, depth, or geometry priors should be trusted.

## G1 - Dynamic-Region Sharpness Needs A Direct Objective

Global PSNR can hide the failure ADAGS cares about: food, hands, and heads remain smeared even when full-image metrics improve. Recent papers make this gap explicit: MAPo targets blurred high-dynamic regions, SharpTimeGS targets sharp and stable temporal visibility, PaMoSplat targets substantial intricate motions, AdaGaR makes high-frequency dynamic detail explicit, and Multi4D frames the tradeoff between oversmoothed deformation fields and overparameterized 4D primitives.

Status: open
Priority: high
Literature pressure: [[papers/jiao2026_mapo]], [[papers/liao2026_sharptimegs]], [[papers/deng2026_pamosplat]], [[papers/chan2026_adagar]], [[papers/wang2026_multi4d]], [[papers/jiang2024_motiongs]]
Related ideas: [[ideas/dynamic-mask-static-exclusion]], [[ideas/rendered-flow-gated-supervision]], [[ideas/dynamic-region-diagnostic-benchmark]]

## G2 - Static/Dynamic Leakage Is A Representation And Evaluation Problem

Static/dynamic separation is no longer novel by itself. SWinGS has static/dynamic weighting, SplatFlow decomposes static background and dynamic objects, 4DGS-SLAM classifies static and dynamic Gaussian sets, SharpTimeGS uses temporal lifespan to balance long-lived static and short-lived dynamic regions, Hybrid 3D-4DGS uses distinct static/dynamic representation capacity, and RiGS explicitly separates coherent rigid transformations from residual deformation. ADAGS needs a more precise claim around reducing static-branch ghosting under reversible routing.

Status: open
Priority: high
Literature pressure: [[papers/shaw2024_swings]], [[papers/sun2025_splatflow]], [[papers/li2025_4dgs_slam]], [[papers/liao2026_sharptimegs]], [[papers/oh2025_hybrid_3d_4dgs]], [[papers/wu2026_rigs]], [[papers/wang2026_flow4dgs_slam]]
Related ideas: [[ideas/dynamic-mask-static-exclusion]], [[ideas/static-anchor-negative-space]]

## G3 - Long-Range Tracks And Depth Priors Are Becoming Table Stakes

MoSca, Shape of Motion, and Prior-Enhanced GS all use long-range tracks and/or depth/foundation priors. ADAGS already has a track-flow hook, but current configs leave `lambda_track_flow: 0.0`. A publishable method needs either to activate and improve this path or explain why a lighter alternative works.

Status: open
Priority: high
Literature pressure: [[papers/lei2025_mosca]], [[papers/wang2025_shape_of_motion]], [[papers/shih2025_prior_enhanced_gs]]
Related ideas: [[ideas/track-prior-scaffold-motion]], [[ideas/rendered-flow-gated-supervision]]

## G4 - Scaffold Residual Motion Is Crowded By MoSca And Prior-Enhanced GS

Plain "motion scaffolds for dynamic Gaussian reconstruction" is occupied territory. MoSca uses 4D motion scaffolds, and Prior-Enhanced GS adds scaffold-projection loss tying motion nodes to tracks. ADAGS should not pitch scaffold residual motion alone; the gap is a lighter reversible LoRA plus scaffold variant with diagnostic proof, or a training-only prior/flow-gated version.

Status: open
Priority: high
Literature pressure: [[papers/lei2025_mosca]], [[papers/shih2025_prior_enhanced_gs]]
Related ideas: [[ideas/track-prior-scaffold-motion]], [[ideas/rendered-flow-gated-supervision]]

## G5 - Capacity Allocation Must Be Matched And Dynamic-Aware

HiCoM, SharpTimeGS, SpeeDe3DGS, MAPo, Hybrid 3D-4DGS, Disentangled4DGS, and Multi4D all treat dynamic capacity, temporal pruning, grouping, partitioning, or representation allocation as central. ADAGS fixed-budget screens are useful, but realized point counts, dynamic-region point density, and high-frequency/detail retention must be audited before claiming allocation gains.

Status: open
Priority: medium-high
Literature pressure: [[papers/gao2024_hicom]], [[papers/liao2026_sharptimegs]], [[papers/tu2026_speede3dgs]], [[papers/jiao2026_mapo]], [[papers/oh2025_hybrid_3d_4dgs]], [[papers/feng2025_disentangled4dgs]], [[papers/wang2026_multi4d]]
Related ideas: [[ideas/motion-aware-densification-budget]]

## G6 - Single Global Motion Models Are A Known Weakness

MAPo partitions high-dynamic Gaussians, MoE-GS routes to specialized experts, PaMoSplat uses part-aware motion, HiCoM uses hierarchical coherent motion, RiGS separates rigid transforms from residual deformations, MotionScale scales motion/geometry reconstruction, Multi4D uses multi-level competitive allocation, and the SE(3) B-spline paper models continuous motion with explicit bases. ADAGS LoRA route0 is stable, but the novelty gap is specialized motion without losing stability.

Status: open
Priority: high
Literature pressure: [[papers/jiao2026_mapo]], [[papers/jin2026_moegs]], [[papers/deng2026_pamosplat]], [[papers/gao2024_hicom]], [[papers/wu2026_rigs]], [[papers/zhou2026_motionscale]], [[papers/wang2026_multi4d]], [[papers/zhang2026_continuous_motion]]
Related ideas: [[ideas/part-aware-reversible-routing]], [[ideas/track-prior-scaffold-motion]]

## G7 - A Benchmark/Diagnostic Claim Is Necessary

MonoDyGauBench argues monocular dynamic Gaussian results are brittle and scene-dependent, and it standardizes apples-to-apples comparisons. D4RT also raises the speed/generalization baseline outside per-scene optimization, while MotionScale and Mono4DGS-HDR remind that geometry, motion, exposure, and photometric artifacts can be mixed together. ADAGS should report dynamic-mask PSNR, static ghost score, track-flow error, edge/sharpness proxies, realized point count, and qualitative panels, not just global PSNR.

Status: open
Priority: high
Literature pressure: [[papers/liang2025_monodygaubench]], [[papers/zhang2025_d4rt]], [[papers/zhou2026_motionscale]], [[papers/liu2025_mono4dgs_hdr]]
Related ideas: [[ideas/dynamic-region-diagnostic-benchmark]]

## Renderer note (2026-08-18) — read before interpreting any flow or gradient gap

G6 and G8 below are unchanged as SCIENTIFIC gaps: making the rendered-flow
gradient live does not gate it for reliability, and the track-flow hook is
still inert at `lambda_track_flow: 0.0`.

Two engineering facts now bound how any gradient-based result in this
repository may be read ([[operations/renderer-integrity-admission-2026-08-18]],
[[operations/rasterizer-backward-two-defects-2026-08-17]]):

* until 2026-08-18 the ACTIVE backward render kernel gated itself on
  UNINITIALISED device memory, so its behaviour depended on allocator
  history rather than on the scene. No prior run's gradients are known
  reproducible. This does NOT establish that any recorded result is
  wrong, and the reproducibility bound measuring the old image's own
  spread is recorded on the admission page.
* rendered-flow supervision was non-functional before 2026-08-18 — the
  VJP lived in a kernel that was never launched. It is now live and
  correctly routed. That closes an INSTRUMENT blocker, not a gap: the
  flow VJP's numerical correctness is still unestablished, and per the
  2026-08-18 decision memo flow supervision is recommended to stay
  shelved because no EL-GS claim has a flow term and the primary dataset
  has no flow.

Nothing below is retracted on this basis. The historical flow-lane
readings cited in G8 used the track-flow path, not rendered flow, so they
are not affected by the rendered-flow repair.

## G8 - Flow Supervision Needs Reliability Gating

MotionGS, PaMoSplat, SplatFlow, and Flow4DGS-SLAM all support explicit flow/motion guidance in some form, but ADAGS W&B suggests naive flow lanes underperform while render-gated flow looks more plausible. The gap is not "add flow"; it is robustly gating flow to reliable dynamic cores and boundaries.

Status: open
Priority: high
Literature pressure: [[papers/jiang2024_motiongs]], [[papers/deng2026_pamosplat]], [[papers/sun2025_splatflow]], [[papers/wang2026_flow4dgs_slam]]
Related ideas: [[ideas/rendered-flow-gated-supervision]]

## G9 - Uncertainty And Occlusion Confidence Are Underused In ADAGS

USplat4D shows uncertainty can improve monocular 4D reconstruction and motion tracking. ADAGS masks, residuals, tracks, and flow losses currently lack a principled confidence model for occlusion, disocclusion, and mask noise.

Status: open
Priority: medium
Literature pressure: [[papers/guo2026_usplat4d]]
Related ideas: [[ideas/rendered-flow-gated-supervision]], [[ideas/track-prior-scaffold-motion]]

## G10 - Practical N3V Cooking-Scene Niche Is Still Available

Many recent methods target autonomous driving, compression, SLAM, HDR/low-light, sparse multi-view capture, or general monocular reconstruction. ADAGS can still own a narrower claim if it demonstrates fast cooking-scene motion improvement under fixed budgets and training-only priors.

Status: open
Priority: medium
Literature pressure: [[papers/sun2025_splatflow]], [[papers/song2025_coda4dgs]], [[papers/kumar2026_l2dgs]], [[papers/aerodgs2026]], [[papers/zhou2026_4c4d]], [[papers/liu2025_mono4dgs_hdr]]
Related ideas: [[ideas/dynamic-region-diagnostic-benchmark]], [[ideas/dynamic-mask-static-exclusion]]

## G11 - Representation Frequency Is A New Sharpness Axis

AdaGaR, Multi4D, and frequency-oriented dynamic reconstruction framing make dynamic blur a representation-frequency problem, not just a missing loss or bad mask. ADAGS currently logs dynamic edge magnitude, but it does not yet claim or evaluate frequency/detail preservation directly.

Status: open
Priority: high
Literature pressure: [[papers/chan2026_adagar]], [[papers/wang2026_multi4d]]
Related ideas: [[ideas/dynamic-region-diagnostic-benchmark]], [[ideas/motion-aware-densification-budget]]

## G12 - Feedforward 4D Models Raise The Baseline

D4RT-style feedforward 4D reconstruction changes the comparison space for speed and generalization. ADAGS can still be valuable as a per-scene, fixed-budget, diagnostic-driven method, but should avoid broad "fast 4D reconstruction" claims unless compared against amortized 4D baselines.

Status: open
Priority: medium-high
Literature pressure: [[papers/zhang2025_d4rt]]
Related ideas: [[ideas/dynamic-region-diagnostic-benchmark]]

## G13 - Visibility Events Are Not Smooth Deformation

Many dynamic GS methods improve motion smoothness, lifespan, partitioning, or transient capacity, but occlusion and disocclusion are event-like changes. Treating them as smooth deformation can create boundary blur, ghost trails, and flicker.

Status: open
Priority: high
Literature pressure: [[papers/wang2026_multi4d]], [[papers/wu2026_rigs]], [[papers/liao2026_sharptimegs]], [[papers/zhao2026_ground4d]]
Related ideas: [[ideas/event-causal-visibility-gaussians]]

Negative evidence: R017 actual opacity gating, R025 non-oracle candidate-local refinement, and R027 non-oracle boundary-gated micro-densification all failed the frozen R009 event-crop gate. R027 produced only small directional gains over route0 (`+0.0569 dB` PSNR, `-0.0000903` L1) and recovered less than 1% of the oracle crop upper bound. R028 posthoc audit found the R026 boundary support essentially missed the frozen crops. R029 route0 continuation worsened route0, so R027's tiny positive movement was not generic continuation. R030 oracle-support micro-densification also failed with mean PSNR `29.9021`, mean L1 `0.0158770`, and `0/5` route0 PSNR+L1 wins. This preserves the visibility-event gap but rejects support-only continuation of the current posthoc micro-densification recipe.

## G14 - Detail Needs Identity-Conserving Promotion Rules

Persistent primitives preserve correspondence but can oversmooth high-frequency detail. Transient capacity can sharpen detail but risks fragmenting identity. The missing piece is a promotion/demotion rule that says when detail should remain attached, become new geometry, or retire.

Status: open
Priority: high
Literature pressure: [[papers/wang2026_multi4d]], [[papers/wu2026_rigs]], [[papers/chan2026_adagar]], [[papers/jin2026_moegs]]
Related ideas: [[ideas/identity-conserving-detail-carriers]], [[ideas/frequency-adaptive-temporal-support]]

## G15 - Prior Usefulness Needs Counterfactual Calibration

Masks, tracks, flow, depth, and geometry priors can be confident but wrong near occlusions, boundaries, static/dynamic leakage, or out-of-distribution geometry. A routing field should estimate whether trusting a prior improves future reconstruction, not merely whether the prior appears confident.

Status: open
Priority: high
Literature pressure: [[papers/guo2026_usplat4d]], [[papers/shih2025_prior_enhanced_gs]], [[papers/zhao2026_ground4d]], [[papers/sun2025_splatflow]]
Related ideas: [[ideas/counterfactual-prior-usefulness-routing]], [[ideas/self-calibrated-prior-reliability-field]]
