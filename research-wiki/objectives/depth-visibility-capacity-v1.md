# Depth Visibility And Intermittent-Surface Capacity — Objective v1

Date: 2026-07-15

Status: approved scientific contract awaiting method refinement. Approval of
this contract does not authorize code implementation, Slurm submission, W&B
writes, experiments, commits, or pushes.

## Recommended objective

Develop and validate a non-oracle, calibrated multiview-temporal surface
visibility model that combines aligned depth, camera geometry, appearance, and
temporal correspondence to infer foreground/background order and
occluded/hidden/revealed states with uncertainty; then couple that evidence to a
budget-matched Gaussian representation that can create, preserve, or reassign
capacity for intermittently visible surfaces, learn their content only from
views/times where it is visible, and render it correctly after reveal without
degrading static or already well-reconstructed regions.

The two halves are independent. Gate A can pass while Gate B fails.

## Focused route comparison

### Route 1 — Geometry-first visibility ledger plus intermittent-surface capacity (recommended)

**Exact mechanism hypothesis.** Known N3V cameras, depth aligned to the scene
frame, multiview reprojection/z-order, temporal correspondence, and appearance
agreement can produce a soft surface visibility ledger: surface identity,
first-surface order, visible/occluded/hidden/revealed state, and uncertainty for
each usable camera-time observation. If the route0 representation uses that
ledger to learn a surface only from first-visible observations and to
create/preserve/reassign capacity under a fixed budget, reappearing content will
improve.

**Relationship to prior experiments.**

- R031-R033 become a monocular-edge baseline, not the starting mechanism.
- R030's unwarped rectangles and posthoc clone/split pressure are replaced by
  calibrated surface support and capacity that need not inherit an existing
  primitive.
- R037's opacity attenuation is retained only as a negative control.
- R030's result motivates a capacity-changing representation; R037 motivates
  separating the occluder from the hidden surface.

**Changed failed assumption.** Visibility is a view/time-conditioned surface
relation, not a 2D crop or depth edge; missing hidden surfaces may require
distinct capacity, not more gradient on or less opacity from the existing bank.

**Dominant contribution.** The experimentally verified coupling from calibrated
surface visibility to budgeted intermittent-surface capacity.

**Smallest adequate implementation.** A deterministic geometry teacher that
emits a surface-level visibility/uncertainty ledger plus one new trainable
capacity component attached to the route0 backbone. The first implementation
must preserve intermittently hidden capacity and perform budget-neutral
reassignment/reinitialization: every reassigned slot is paired with retirement
of an existing slot, so the operation does not grow the 600k bank. The exact
candidate, slot, surface-state, and reinitialization parameterization remains a
method-refinement decision.

**Required supervision.** Known cameras; aligned depth/confidence; RGB or
feature correspondence; optional flow/scene-flow consistency; ordinary training
images. No frozen event windows, GT crop residuals, or test-view labels enter
training.

**Trainable components.** At most one genuinely new component beyond route0:
the intermittent-surface representation/capacity component. The geometry
teacher is initially deterministic or frozen.

**Decisive Gate A test.** Withhold a camera or subset of views and predict
surface order/visibility from the remaining calibrated observations. Beat
single-frame depth edges, R031-style support, and geometry without temporal
correspondence on temporal event recall, boundary/region localization, ordering,
cross-view/temporal consistency, compactness, and calibrated risk.

**Decisive Gate B test.** Under the same 6000-iteration and realized point
budget, beat route0 and matched controls on checkpoint-backed event renders
while passing static no-harm. Ablations must separate visibility evidence from
capacity change.

**Working novelty hypothesis.** VAD-GS already turns visibility and MVS into new
Gaussian geometry; Proxy-GS already uses proxy occlusion depth to guide
densification; 4C4D already conditions opacity learning on spatial-temporal
visibility; PackUV-GS already targets temporal consistency through
disocclusions; and OccluGaussian already uses co-visibility to organize Gaussian
reconstruction. Route 1 may still differ by combining calibrated non-rigid
surface order/reveal state and uncertainty with budget-neutral preservation and
reassignment, but this is a hypothesis to test, not an established absence in
prior work.

**Feasibility.** High enough for a staged program because N3V already has known
cameras, synchronized frames, route0 checkpoints, depth tooling, and
checkpoint-backed evaluation. The geometry teacher can be falsified before
representation work.

**Compute cost.** Moderate Gate A preprocessing and low-cost scoring; moderate
Gate B pilot; high only if expanded to seeds and a second dataset.

**Principal failure mode.** Dynamic depth/correspondence errors create incorrect
surface identities, so the allocator protects or creates the wrong capacity.

### Route 2 — Geometry-distilled learned visibility field plus adaptive capacity

**Exact mechanism hypothesis.** A compact view/time-conditioned visibility
field distilled from calibrated geometry can generalize beyond incomplete or
noisy pseudo-labels, and its uncertainty can control a capacity allocator.

**Relationship to prior experiments.** Replaces R031's fixed score fusion with a
learned surface/ray field and replaces R037's fixed rectangle gate with
continuous view/time-conditioned evidence. Retains R030 as evidence that the
capacity side cannot be posthoc clone/split alone.

**Changed failed assumption.** A fixed handcrafted union of cues is not the
right representation of visibility; visibility should be learned as a
conditional field while preserving its geometric teacher and calibration.

**Dominant contribution.** A geometry-distilled, uncertainty-calibrated dynamic
visibility field coupled to representation capacity.

**Smallest adequate implementation.** One compact visibility field and one
minimal capacity component; no additional motion network beyond route0.

**Required supervision.** Route 1 geometry pseudo-labels, held-out-view
consistency, RGB/features, and ordinary reconstruction loss.

**Trainable components.** Two new components: visibility field and capacity
component. This is the maximum allowed and makes causal attribution harder.

**Decisive Gate A test.** The learned field must outperform its deterministic
teacher on disjoint held-out cameras/times without losing calibration or
compactness.

**Decisive Gate B test.** It must beat both the deterministic-teacher capacity
route and the learned field with no capacity change under matched compute.

**Novelty relative to literature.** NeuRay is static source-view visibility;
USplat4D is per-Gaussian uncertainty; neither learns non-rigid reveal states and
uses them to allocate new capacity.

**Feasibility.** Medium. It adds label-noise, identifiability, and training
stability risks before Gate B.

**Compute cost.** Medium-high due to an additional training stage and ablations.

**Principal failure mode.** The field learns photometric shortcuts or the
teacher's errors and looks calibrated in-distribution while failing on actual
reveals.

### Route 3 — Explicit layered surface-memory Gaussians

**Exact mechanism hypothesis.** When calibrated rays support multiple ordered
surface hypotheses, an explicit foreground/hidden-surface memory can learn the
rear surface from other views/times and reveal it without suppressing or
overwriting it.

**Relationship to prior experiments.** Directly changes R030's single-bank
clone/split assumption and R037's attenuation of all candidate-local dynamic
Gaussians. It uses R031 only as a weak depth cue baseline.

**Changed failed assumption.** One bank with soft static/dynamic routing is
insufficient when two surfaces occupy the same projected neighborhood at
different depths and times.

**Dominant contribution.** An occlusion-conditioned layered Gaussian surface
memory for dynamic content.

**Smallest adequate implementation.** Two ordered local surface hypotheses only
where Gate A provides strong evidence, with shared route0 motion elsewhere and
a competitive budget between ordinary and hidden-surface capacity.

**Required supervision.** Calibrated depth order, cross-view surface
correspondence, and ordinary visible-view RGB/features.

**Trainable components.** One layered representation component; Gate A remains
deterministic or frozen.

**Decisive Gate A test.** Same as Route 1, with an added requirement that a
second surface hypothesis is supported by another view/time and not merely a
depth discontinuity.

**Decisive Gate B test.** Beat a single-bank capacity allocator at the same
point count, especially on background and dynamic-behind-dynamic reveals.

**Novelty relative to literature.** Deep 3D Mask Volume assumes a static
background MPI; this route would target non-rigid Gaussian surfaces and
evidence-driven local layering.

**Feasibility.** Medium. It is conceptually direct but risks locking the problem
to two layers and complicating compositing/association.

**Compute cost.** Moderate.

**Principal failure mode.** Layer ambiguity duplicates geometry, creates
floaters, or spends budget on unsupported rear surfaces.

## Approved route hierarchy

Route 1 is approved as the lead. It offers the cleanest causal program: a
deterministic/frozen Gate A can be rejected before training, and one
capacity-changing component can then test Gate B. Route 3 is the approved
fallback if Gate A evidence is strong but a general allocator is under-specified.
Route 2 is permitted only if deterministic Gate A passes but remains too
incomplete for a useful representation.

## Closest-work addendum and novelty status

| Work | Visibility or disocclusion mechanism | Representation/capacity action | Difference that Route 1 must test rather than assume |
| --- | --- | --- | --- |
| VAD-GS | Voxel first-visibility plus calibrated cross-frame MVS | Initializes new Gaussians for missing geometry | Route 1 targets non-rigid N3V without urban LiDAR/boxes and uses a fixed bank, but novelty is not established until the mechanisms are compared directly. |
| Proxy-GS | Proxy-mesh occlusion-depth maps | Culls structured Gaussians and guides surface-aligned anchor densification | Static proxy geometry and net densification differ from dynamic reveal-state inference and budget-neutral reassignment. |
| 4C4D | View-intersection and temporal-span activity of 4D Gaussians | Applies learned versus constant opacity decay to visible/invisible subsets | It conditions optimization on visibility but does not explicitly infer front/back order or reassign hidden-surface capacity. |
| PackUV-GS | Flow-guided keyframes and projected dynamic-Gaussian labels | Sequential per-frame UV fitting, static freezing, and UV density control | It handles disocclusion and temporal coherence in dense capture, but does not explicitly output ordered occluder/hidden/reveal states. |
| OccluGaussian | Static camera-position and co-visibility graph | Partitions large scenes and culls region-invisible Gaussians | Its visibility is camera/region-level rather than non-rigid surface/time-level. |

The defensible novelty statement is therefore provisional: **Route 1 tests
whether calibrated, uncertainty-bearing, non-rigid surface visibility can drive
budget-neutral preservation and reassignment for intermittently hidden content
better than visibility weighting, opacity modulation, proxy-guided
densification, keyframing, or scene partitioning.** Do not write that no reviewed
work combines visibility and capacity until a mechanism-by-mechanism audit of
these methods and any newer close work is complete.

# Research contract

## 1. Problem Anchor

Dynamic Gaussian reconstruction currently treats occlusion and reveal as
image-space support, smooth motion, or opacity modulation. The unresolved
problem is to infer which physical surface is first-visible, hidden, or
reappearing from calibrated multiview-temporal evidence, then ensure the
representation retains adequate capacity and supervision for that surface
without corrupting static or already solved regions.

## 2. Two-part scientific question

**Occlusion inference:** Can a non-oracle model using calibrated cameras,
aligned depth, appearance, and temporal correspondence identify genuine
surface-level occlusion/reveal structure with useful ordering and uncertainty?

**Representation and reconstruction:** Does coupling that evidence to
budget-matched capacity creation, preservation, or reassignment improve
checkpoint-backed reconstruction of intermittently visible surfaces without
static harm?

## 3. Evidence ledger

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| R013/R015 oracle compositing | Frozen regions contain a large recoverable image-space gap | A Gaussian method can learn the content |
| R017 | Runtime opacity gating on existing Gaussians is harmful | All visibility modeling is harmful |
| R025/R020 | High-recall 2D boxes can overlap windows, but posthoc local refinement fails | Calibrated surface support or new capacity fails |
| R026/R027 | Thin boundary support plus micro-densification failed | Good 3D support fails; R026 missed the windows |
| R029 | Generic 6000-to-6400 continuation is not the source of the small R027 movement | A different full-training representation cannot work |
| R030 | Unwarped crop-weighted, 400-step clone/split micro-densification fails | Correctly reprojected 3D support or distinct hidden-surface capacity fails |
| R031-R033 | Single-camera normalized edge/change support and capped component selection are weak under a confounded overlap audit | Calibrated multiview-temporal depth, ordering, or depth-driven representation fails |
| R036/R037 | R020-supported fixed opacity attenuation fails in checkpoint-backed full training | DA3 support, surface ordering, preservation, layering, or capacity creation fails |
| Current code | Route0 has strong reusable motion/routing and a fixed-budget baseline | It has an occlusion-aware surface visibility state or hidden-surface memory |

## 4. What prior experiments falsified

- The specific R031-R033 recipe--single-camera, independently normalized depth
  edges/change, capped component selection, and its confounded overlap
  audit--does not justify use as reveal support. Because the audit compared
  `cam00` masks with `cam10` rectangles without reprojection, it did not cleanly
  falsify even monocular depth-edge localization by itself.
- Adding more high-scoring 2D support or filling selected tiles does not fix
  mislocalization.
- Frozen or non-oracle 2D rectangles cannot be copied across cameras as though
  coordinates identify the same surface.
- Another posthoc ROI/refinement/micro-densification pass on the existing
  clone/split representation is not justified.
- Direct opacity attenuation of all projected candidate-local dynamic
  primitives is not justified.
- Extra iterations alone are not a repair mechanism.

## 5. What remains untested

- Known-camera DA3 or another depth source aligned across synchronized N3V
  cameras and time.
- Temporal comparison after optical/scene-flow warping or 3D reprojection.
- Explicit foreground/background order and surface identity.
- Occluded, persistently hidden, and newly revealed state inference.
- Calibration and abstention for unreliable depth/visibility.
- Learning hidden content from other cameras/times where the surface is visible.
- Budgeted capacity creation, preservation, or reassignment driven by verified
  visibility evidence.
- A correctly reprojected surface-level oracle diagnostic that isolates Gate B.

## 6. Dominant falsifiable claim

> Under matched training and realized point budgets, calibrated
> multiview-temporal surface visibility evidence can drive an
> intermittent-surface Gaussian capacity mechanism that recovers a meaningful
> fraction of held-out occlusion/reveal reconstruction error while preserving
> static-region quality; the specific R031-R033 edge-support, R037 attenuation,
> and R030 posthoc clone/split recipes did not do so.

The claim fails if Gate A does not identify visibility structure or if a
Gate-A-passing signal does not improve Gate B under matched controls.

## 7. Gate A criteria: occlusion inference

### Data discipline

- Use `cut_roasted_beef` exclusively for method development, threshold fitting,
  debugging, and qualitative iteration.
- Lock `flame_steak` and `sear_steak` for transfer evaluation. Do not inspect
  outputs on their new claim-grade event tracks until Route 1 and thresholds are
  frozen.
- Use frozen R009 windows once, after method and thresholds are fixed, as a
  historical posthoc benchmark only. They already informed R013-R037 and this
  contract, so they are not an unbiased confirmatory test. Lock a new disjoint
  claim-grade event set before implementation; reveal its scores once and do
  not use them to revise the method in the same claim cycle.
- Preserve camera identity in every record. Any cross-view pixel comparison
  requires reprojection.
- Include a small synthetic or controlled-geometry fixture with true visibility
  order to catch sign, convention, and z-buffer errors before real N3V scoring.
- Estimate any depth scale/shift, pose correction, or confidence calibration
  from training/development views only, never from a held-out test camera or
  locked event set.
- Build a human-audited reference set with at least 24 event tracks, targeting
  30-36. Stratify tracks across all three scenes while exposing only the
  `cut_roasted_beef` development portion. At least 20% of tracks must be selected
  for independent double annotation before labels are produced; report
  agreement and adjudicate disagreements under a frozen rubric.

### Reference and denominator protocol

- Before producing method scores, freeze the unit of analysis: a surface/ray
  observation for ordering and state, a frame transition for temporal events,
  and a pixel set for boundary/interior localization.
- On real N3V development data, form reference events from evidence withheld
  from the predictor: held-out-camera first-surface observations, independent
  camera/time consensus, and forward/backward checks. A cue may not grade
  itself; for example, DA3-derived predictions cannot be declared correct only
  because another normalization of the same DA3 output agrees.
- Exact ordering and boundary denominators come from the controlled fixture and,
  if approved, a small disjoint human-audited development set. Annotation is an
  evaluation reference only; it is not an input, loss, mask, or selection cue.
- Report coverage/abstention with every conditional metric and macro-average by
  scene and event, so a method cannot pass by scoring only easy non-abstained
  rays or long events.
- Define positives and negatives from the frozen reference annotations or
  controlled geometry; evaluate boundaries at 2, 4, and 8 pixels at native
  evaluation resolution (4 pixels primary); use 15 equal-mass ECE bins; and
  make scene/event macro-averages primary, with pooled micro-averages secondary.

### Non-oracle validation signals

- Held-out-camera consistency: infer from other calibrated views and compare
  predicted first-surface depth/appearance with the observed held-out view.
- Cross-view z-order agreement: a candidate rear surface must lie behind the
  observed first surface in occluding views and agree in views where visible.
- Temporal forward/backward cycle consistency after flow or 3D correspondence.
- Appearance/feature consistency only among observations predicted visible.
- Consensus pseudo-labels formed from independent camera/time evidence, with
  abstention where geometry is underconstrained.
- The approved disjoint human audit assesses contours and ordering. Its labels
  never enter inference and cannot include the historical R009 windows.

### Required metrics

Report all metrics separately, not a single support score:

1. temporal event precision, recall, and F1;
2. boundary precision/recall/F1 at predeclared pixel tolerances;
3. interior/region precision, recall, and IoU;
4. foreground/background ordering accuracy and AUROC;
5. cross-view visibility-state agreement after reprojection;
6. temporal state consistency and forward/backward cycle violation;
7. support/candidate compactness and precision-recall versus selected fraction;
8. uncertainty Brier score, expected calibration error, and risk-coverage curve.

### Engineering-admission tier

This tier admits representation engineering on `cut_roasted_beef`; it is not a
scientific claim. All conditions are required on the frozen development tracks:

- controlled-geometry sign, camera, z-buffer, and temporal-cycle fixtures pass;
- ordering accuracy at least 0.70 and AUROC at least 0.75 at coverage of at
  least 0.60 of evaluable ordering pairs;
- temporal event F1 at least 0.45 and recall at least 0.60;
- primary 4-pixel boundary F1 and region IoU each improve by at least 0.05
  absolute over the strongest R031-style/monocular baseline at matched selected
  fraction;
- cross-view and temporal inconsistency each fall by at least 15% relative to
  the strongest non-geometric baseline;
- ECE is at most 0.15 and risk decreases monotonically as low-confidence
  evidence is rejected;
- every development event has evaluable support and the method has nonzero
  recall on at least 80% of tracks.

### Claim-grade tier

After the method is frozen, evaluate once on the locked `flame_steak` and
`sear_steak` tracks. Both scenes must satisfy the scene-level requirements:

- ordering accuracy at least 0.75 and AUROC at least 0.80 at coverage of at
  least 0.70;
- temporal event F1 at least 0.60 and recall at least 0.70;
- primary boundary F1 and region IoU each improve by at least 0.10 absolute over
  the strongest matched baseline;
- cross-view and temporal inconsistency each fall by at least 25%;
- ECE is at most 0.10 with a monotonic risk-coverage curve;
- no transfer scene or annotated event family is completely missed, and paired
  bootstrap confidence intervals by event/scene are reported.

Abstention counts as a miss for temporal and region recall. Conditional
ordering/calibration metrics are invalid below the applicable coverage threshold.
Failure of claim-grade transfer does not retroactively invalidate engineering
admission, but it blocks the scientific claim and triggers the stop rules.

Gate A fails if improvements require frozen-window tuning, copied pixel
coordinates between cameras, broad selection that wins only by coverage, or
confidence that does not predict errors.

## 8. Gate B criteria: representation benefit

### Required evidence

- Real checkpoint-backed Gaussian training and rendering.
- Exact route0 comparison at 6000 iterations and a 600k ceiling, reusing
  authoritative route0 checkpoints/results where scientifically matched.
- Realized point count, peak capacity, and training budget reported.
- Controls that isolate: visibility with no capacity change, capacity change
  without visibility, and the coupled method.
- A correctly reprojected, surface-level oracle-evidence-plus-capacity
  diagnostic on a disjoint development set. It is an upper-bound attribution
  tool, never training support for the reported non-oracle method.
- R037 opacity attenuation remains a historical negative control; it is not
  silently rebranded as the new method.

### Required metrics

- event-region PSNR and LPIPS computed from rendered and ground-truth pixels,
  with the current L1/proxy retained only for continuity;
- fraction of event windows improved on both fidelity metrics;
- PSNR and perceptual oracle-gap recovery;
- static-region PSNR/quality and masked reconstruction L1 no-harm;
- temporal flicker and reveal-transition ghost trails;
- realized points, reassignments/reinitializations/preservations, and capacity
  by surface state; record births only if a later approved route introduces them;
- scene and seed consistency.

Before training, freeze event interiors/boundaries, the static mask, reveal
transitions, and all formulas. Historical R009 rectangles retain their original
rectangle metrics only for continuity. Claim-grade event masks come from the
new locked test set; the static mask is the method-independent complement of a
predeclared dilated dynamic/event mask. Temporal flicker is the error of the
flow-warped render difference relative to the corresponding ground-truth
difference. Reveal ghost is residual foreground/trail error inside the locked
newly revealed region for a fixed number of frames after the transition.

### Practical Gate B targets

- mean event-region PSNR at least +0.20 dB versus route0;
- mean perceptual error improves by at least 5% versus route0;

### Hard admission conditions

- a majority of events in the applicable new audited set improve on both PSNR
  and LPIPS;
- static no-harm criteria in Section 9 pass;
- flicker and ghosting do not regress by more than 5% in mean and neither has a
  scene-wide failure;
- for the primary matched-budget comparison, final and integrated point-count
  budgets are each within +/-2% of the control and never exceed 600k; a method
  using materially less capacity is reported separately on a quality-budget
  Pareto curve rather than called matched;
- the full coupled method beats the evidence-only and capacity-only ablations,
  not only route0, and shuffled/misaligned evidence does not reproduce the gain.

The `+0.20 dB` and `5% LPIPS` values are practical targets for deciding whether
the effect merits transfer evaluation, not universal publication thresholds.
R009 `3/5` continuity and oracle-gap recovery are reported as secondary
diagnostics only and cannot independently pass or fail Gate B. Oracle-gap
recovery remains undefined when the oracle does not beat route0.

For each event report paired deltas and confidence intervals. Oracle-gap
recovery is `(method - route0) / (oracle - route0)` in the improvement
direction, only where the oracle gap is positive; otherwise report it undefined
and retain raw deltas. Predeclare a catastrophic event as either PSNR at least
0.50 dB worse or LPIPS at least 10% worse than route0. A scene-wide failure is
any scene with no event improved or with a failed per-scene static bound.

The `visibility-only` control must change observation weighting/state without
changing primitive birth, preservation, reassignment, or point count. The
`capacity-only` control must use the identical capacity operator and schedule
but replace inferred visibility with a predeclared generic or rate-matched
trigger. A shuffled/misaligned-evidence control tests whether any sparse signal
would produce the same gain. Optimizer steps, learning-rate schedule, trainable
parameter count, wall-clock/GPU budget, and data exposures must be matched or
reported as explicit differences.

### Gate B failure attribution

A failed coupled result is not assigned to "depth" without this diagnostic
ladder:

1. Re-score the exact Gate A evidence on the Gate B observations to verify its
   coverage, ordering, calibration, and camera/time alignment.
2. Run the same frozen capacity mechanism with the correctly reprojected,
   surface-level oracle evidence on a disjoint development set. If this succeeds
   while learned/inferred evidence fails, the failure is support discovery,
   calibration, or transfer.
3. If both visibility signals fail but capacity-only succeeds, the coupling or
   visibility-conditioned parameterization is wrong.
4. If no evidence variant succeeds, inspect realized births, preservation,
   reassignment, gradients, budget saturation, and convergence. Failure to
   express additional rear-surface hypotheses indicates insufficient or wrong
   capacity; failure to activate a capable mechanism indicates optimization.
5. At most one predeclared optimization repair is allowed before a
   representation pivot. Frozen R009 rectangles never become training support
   in this ladder.

One-scene/one-seed results can admit expansion but cannot support the dominant
claim. Claim-grade evidence should cover all three N3V scenes and at least two
seeds where feasible.

## 9. Static-region no-harm criteria

- Mean static-region PSNR loss no worse than -0.05 dB versus route0.
- Mean static perceptual error and masked static reconstruction L1 no more than 2% worse.
- Every claim-grade scene satisfies each bound; aggregate success cannot hide a
  failed scene. Report R009 static continuity separately without using it as a
  pass/fail condition.
- No broad background floaters, duplicated surfaces, or persistent ghost layer
  in blinded qualitative review.
- Capacity gained for intermittent surfaces must come from the declared budget,
  not hidden total-point expansion.

## 10. Baselines and datasets

### Required baselines

- canonical fixed-budget LoRA route0;
- R031-style single-camera depth edge/change;
- calibrated geometry without temporal correspondence;
- calibrated geometry with temporal correspondence but no learned visibility;
- visibility evidence with no capacity change;
- capacity mechanism with no visibility or with shuffled/misaligned evidence;
- full coupled method;
- correctly reprojected surface-level oracle evidence with the identical
  capacity mechanism, as an attribution upper bound only;
- relevant historical R030 and R037 negatives;
- derived hide/reveal oracle as an upper bound only.
- Proxy-GS, 4C4D, PackUV-GS, OccluGaussian, and VAD-GS as mechanism-level
  closest-work comparisons; runnable code baselines are required only when
  scientifically compatible with N3V and separately approved.

### Datasets

- Development: N3V `cut_roasted_beef` only. Live transforms metadata establishes
  that `cam00` is the held-out test camera in the active loader and `cam10` is a
  training camera. The exact train/test record manifest and development
  cameras/times must be frozen and hashed before Gate A execution.
- Locked transfer: N3V `flame_steak` and `sear_steak`; the new claim-grade event
  annotations and rendered results remain sealed until method lock.
- Cross-dataset evaluation is deferred until N3V Gate B admission. PanopticSports
  is not implied by the existing smoke configuration and would require separate
  scientific approval.
- Synthetic fixture: small controlled occluder/reveal geometry for Gate A
  correctness, not a paper result by itself.

## 11. Phase 9 compute policy

The earlier Phase 8B one-scene/five-training/approximately 80-GPU-hour envelope
was a pre-implementation planning cap. The user superseded it for Phase 9:

- there is no artificial GPU-hour ceiling, but every expensive execution must
  have a preregistered purpose, exact configuration, compute estimate, success/
  failure interpretation, and downstream decision;
- plumbing, geometry, and numerical checks use the shortest adequate fidelity;
- 6000 iterations are used only for converged, comparable representation runs;
- after one-scene causal admission and configuration freeze, final admitted
  comparisons cover all six N3V scenes;
- subsequent compute prioritizes matched seeds, uncertainty, decisive causal
  controls, and robustness rather than arbitrary sweeps.

The 600k point ceiling, matched realized/integrated budgets, static no-harm, and
Gate A/B separation remain unchanged. Cross-dataset work remains deferred.

## 12. Non-goals

- Proving that DA3 is the best depth model.
- Tuning another depth-edge mask, tile fill, ROI weight, opacity gate, or
  densification multiplier.
- Using frozen event rectangles as training support.
- Claiming metric 3D accuracy from an unvalidated foundation-depth scale.
- Solving surfaces never visible in any training camera/time.
- Adding more than two genuinely new trainable components.
- Improving only global PSNR while event or static diagnostics regress.
- Claiming generic dynamic-scene SOTA from three N3V scenes.
- Cross-dataset evaluation before N3V Gate B admission.

## 13. Stop and pivot rules

- **Stop Gate A** if calibrated evidence cannot beat the strongest
  monocular/flow baseline on ordering plus localization, or if gains disappear
  at matched compactness.
- **Stop learned visibility** if it does not beat the deterministic teacher or
  its calibration worsens; retain the geometry teacher and pivot to Route 1/3.
- **Stop Gate B parameter tuning** after one serious optimization repair if a
  Gate-A-passing signal still cannot beat capacity-only and visibility-only
  controls.
- **Pivot representation** if visibility-only helps observation weighting but
  not reconstruction; test the approved preservation/reassignment operation,
  then layering or birth only under renewed approval, rather than a stronger
  gate.
- **Pivot evidence** if capacity-only helps but coupled evidence hurts; diagnose
  surface association/calibration before more training.
- **Retire the direction** if the coupled method fails static no-harm in two
  serious parameterizations or across two scenes.
- A failed family may be revisited only with a written changed assumption and a
  test that isolates that assumption.

## 14. Research Freedom clause

Within an approved Phase 8B scope, the agent may propose and implement a novel
Gaussian representation, replace existing routing, densification, lifecycle,
or visibility mechanisms, and choose a different concrete mask/loss/field/layer
parameterization when evidence requires it. The agent is not required to retain
R030 or R037 machinery.

This freedom is bounded by the two-part objective, non-oracle data discipline,
at most one dominant contribution and two genuinely new trainable components,
matched compute/capacity, static no-harm, and the approval boundaries below. A
previously failed family may be revisited only after documenting which prior
assumption changed, why the new mechanism isolates it, and what result would
falsify the revisit.

## 15. Phase 9 approval boundaries

The user explicitly authorized ordinary Phase 9 implementation, configuration,
new depth/geometry sidecars, Slurm execution, creation of new W&B runs/metrics,
narrow commits, and pushes on the tracked branch. These actions still require
the method/plan reviews, preregistration, validation, job ledger, and gate rules
in this contract.

Further user approval remains required for a material change to the two-part
objective or route hierarchy, cross-dataset evaluation, destructive action,
mutation of historical W&B runs, rewriting branch history, or an irreconcilable
repository/credential decision. Human labels may never be fabricated.

## 16. Resumability/checkpoint protocol

- Transient Phase 9 execution status lives in
  `$WORK/proj_adags/agent-control/phase9-depth-visibility-capacity/STATE.json`.
- After later implementation/job approval, each major Gate A/B stage gets an
  agent-control checkpoint containing inputs, exact commands, outputs, job IDs,
  and next action.
- Every submitted job ID must be recorded immediately and checked through
  `squeue` and `sacct` before resubmission.
- Generated maps, logs, arrays, and checkpoints stay ignored/outside Git.
- Durable hypotheses, negative results, gate decisions, and changed assumptions
  are promoted to `research-wiki/` and appended to its log.
- A run ID may not be reused. Every experiment manifest records config hash,
  code commit, dataset/camera split, depth source/version, seeds, point budget,
  and frozen evaluation version.
- W&B sync, if approved later, is a distinct resumable step after local
  checkpoint-backed evidence exists.

## 17. Approved decisions and remaining method refinement

The nine decision groups are reconciled and approved:

1. **Route hierarchy:** Route 1 leads; Route 3 is fallback; Route 2 is permitted
   only after a passing but incomplete deterministic Gate A.
2. **Gate A tiers:** engineering admission and claim-grade transfer use the
   separate thresholds in Section 7.
3. **Gate B interpretation:** `+0.20 dB` and `5% LPIPS` are practical targets;
   R009 continuity and oracle-gap recovery are secondary diagnostics.
4. **Human reference:** at least 24 tracks, target 30-36, with at least 20%
   independently double annotated.
5. **Scene split:** `cut_roasted_beef` is development-only;
   `flame_steak` and `sear_steak` are locked transfer scenes.
6. **Compute:** the Phase 8B conditional five-training, one-scene envelope
   was the approved pre-implementation starting point. Phase 9 supersedes its
   approximately 80-GPU-hour aggregate cap as specified in Section 11; the
   6000-iteration comparable endpoint, 600k ceiling, and preregistered
   per-execution resource bounds remain.
7. **Cross-dataset:** deferred until N3V Gate B admission.
8. **Initial capacity operation:** preservation plus budget-neutral
   reassignment/reinitialization.
9. **Evaluation lock:** create a new claim-grade event set; R009 remains
   historical continuity only.

Phase 9 method refinement must freeze the annotation/adjudication protocol,
surface-track representation, independent depth alignment/calibration,
oracle-sidecar construction, exact reassignment transaction, and budget
instrumentation before the corresponding execution. The implementation-ready
method and claim-driven matrix are versioned operational contracts under
`research-wiki/operations/`; they may change concrete engineering within the
Research Freedom clause but may not weaken approved data separation or gates.
Every post-outcome revision requires a new experimental cycle.
