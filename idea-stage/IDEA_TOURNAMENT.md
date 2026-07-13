# ADAGS Research Idea Tournament

Date: 2026-06-30
Scope: idea selection only. No training jobs, pilots, or code implementation were launched.

## Decision

Winner: **Self-Calibrated Reliability-Gated Priors for Dynamic Gaussian Splatting**.

This is not a claim that the method already works. It is the most paper-worthy direction to pursue next because it turns the crowded "add flow/masks/scaffolds" space into a narrower falsifiable mechanism:

- C11 is the primary method: a predeclared reliability field decides where external priors are allowed to act.
- C14 is the deterministic baseline and safeguard: dynamic core, uncertain boundary, and static-anchor negative space.
- I1 is the first concrete instantiation: rendered/track-flow supervision only in reliable dynamic cores.
- C15 is the reporting frame: failure atlas, diagnostic protocol, and causal mechanism screen.

Backup: **ADAGS Failure Atlas and Mechanism Screen with a compact reliability-gated case study**. This is the fallback if the reliability field is scientifically informative but the full method does not produce a large enough dynamic-region improvement for a main-method paper.

Hard kill condition: if reliability mostly selects easy pixels, fails to retain hard dynamic-core regions, or cannot beat LoRA route0 on dynamic quality while preserving static quality and matched realized budget, the winner is invalidated.

## Evidence Ledger

- Local report: `idea-stage/IDEA_REPORT.md`.
- Research memory: `research-wiki/query_pack.md`, `research-wiki/gap_map.md`, `research-wiki/log.md`, `research-wiki/ideas/`, `research-wiki/experiments/`, `research-wiki/papers/`, `research-wiki/literature_map_dynamic_gs_2026.md`.
- Code assets inspected: `main.py`, `scene/gaussian_model.py`, `utils/motion_prior_utils.py`, `gaussian_renderer/__init__.py`, `arguments/__init__.py`, configs, Slurm helpers, branch log and remote Codex branches.
- W&B read-only context: `models-ku-leuven/adags`, 126 total runs, 126 finished, no running/crashed/failed/killed runs. Mechanism-screen and eval-6000 aggregates match the wiki. Diagnostic metrics exist for 15 eval-6000 runs but no runs with tag `diagnostics:dynamic_static`.
- Literature pressure: MoSca, Prior-Enhanced GS, MotionGS, SplatFlow, Flow4DGS-SLAM, Shape of Motion, USplat4D, RiGS, MAPo, MoE-GS, PaMoSplat, HiCoM, SharpTimeGS, AdaGaR, Multi4D, D4RT, and MonoDyGauBench.

## Candidate Pool

The tournament started from all 10 candidates in `IDEA_REPORT.md` and added 5 challengers because the inherited pool had strong diagnostics but many method ideas were crowded or incremental.

| ID  | Candidate                                                | Source      | Tournament status                                 |
| --- | -------------------------------------------------------- | ----------- | ------------------------------------------------- |
| I1  | Reliability-Gated Rendered Flow for Dynamic Cores        | IDEA_REPORT | Final package component                           |
| I2  | Dynamic-Region Diagnostic Benchmark as Paper Backbone    | IDEA_REPORT | Folded into C15                                   |
| I3  | Route0-Coherent plus Residual-Specialist Motion          | IDEA_REPORT | Semifinal loss, backup seed only                  |
| I4  | Confidence-Aware Priors for Masks, Tracks, and Flow      | IDEA_REPORT | Merged into C11                                   |
| I5  | Dynamic Frequency/Detail Preservation under Fixed Budget | IDEA_REPORT | Eliminated as standalone, retained as metric axis |
| I6  | Track-Anchored Scaffold Residual Motion                  | IDEA_REPORT | Eliminated/conditional component                  |
| I7  | Unanchored Scaffold Residual Motion                      | IDEA_REPORT | Eliminated negative control                       |
| I8  | Naive Broad Flow Supervision                             | IDEA_REPORT | Eliminated negative control                       |
| I9  | Hard Static Conversion                                   | IDEA_REPORT | Eliminated ablation only                          |
| I10 | Blur Curriculum and Early Part-Basis Tried Forms         | IDEA_REPORT | Eliminated historic controls                      |
| C11 | Self-Calibrated Prior Reliability Field                  | Challenger  | Winner                                            |
| C12 | Reliable Temporal Detail Transport                       | Challenger  | Qualifier loss, future high-risk idea             |
| C13 | Competitive Route0-Residual Budget Allocator             | Challenger  | Semifinal loss, backup seed only                  |
| C14 | Boundary-Aware Static-Anchor Negative Space              | Challenger  | Folded into winner                                |
| C15 | ADAGS Failure Atlas plus Mechanism Screen                | Challenger  | Backup and required frame                         |

## Candidate Dossiers

### I1 - Reliability-Gated Rendered Flow for Dynamic Cores

- Problem anchor: cached flow can help fast moving food/hands only where correspondence is reliable; broad flow injects occlusion and boundary noise.
- Concrete method: LoRA route0 base; apply rendered/track-flow loss in reliable dynamic-core pixels; weaken or exclude uncertain boundary/static regions.
- Dominant contribution: reliability-gated use of rendered flow for dynamic-core sharpness.
- Nearest related work: MotionGS, SplatFlow, PaMoSplat, Flow4DGS-SLAM, Shape of Motion, Prior-Enhanced GS.
- Why not merely incremental: only if the paper proves broad flow fails, render gating recovers, and reliability predicts where flow is useful.
- Minimum decisive experiment: route0 vs naive flow vs core-mask flow vs render-gated flow on identical scenes, budgets, checkpoints, masks, and metrics.
- Likely failure mode: global PSNR recovers but dynamic sharpness/static ghosting does not improve.
- Implementation burden: medium. Flow/render hooks exist, but best lane lives on remote branches and diagnostic coverage is sparse.
- Top-venue acceptance path: use as the concrete instantiation of C11, not as standalone "flow plus masks."

### I2 - Dynamic-Region Diagnostic Benchmark as Paper Backbone

- Problem anchor: global PSNR hides fast-motion blur, static-branch ghosts, and capacity allocation failures.
- Concrete method: standardized diagnostics: dynamic-mask PSNR, static-region PSNR, static ghost score, dynamic edge/detail, flow L1, routing/capacity/point-count metrics, and crops.
- Dominant contribution: evaluation protocol and failure taxonomy.
- Nearest related work: MonoDyGauBench, D4RT, MotionScale, AdaGaR, Multi4D.
- Why not merely incremental: only if diagnostics expose ranking reversals or causal failures not visible in global PSNR.
- Minimum decisive experiment: eval-only diagnostic pass over existing mechanism-screen/eval-6000 checkpoints.
- Likely failure mode: benchmark looks like internal tooling and metrics are noisy.
- Implementation burden: low-medium.
- Top-venue acceptance path: fold into C15 and pair with a compact method case study.

### I3 - Route0-Coherent plus Residual-Specialist Motion

- Problem anchor: LoRA route0 is stable but may oversmooth local non-rigid/high-frequency dynamic failures.
- Concrete method: route0 handles coherent base motion; a residual/phase-2 LoRA lane activates in high-error/high-motion dynamic regions.
- Dominant contribution: controlled coherent/residual decomposition.
- Nearest related work: RiGS, MAPo, MoE-GS, PaMoSplat, HiCoM, continuous SE(3) motion bases, Multi4D.
- Why not merely incremental: only if route0/residual specialization is measured, not narrated after the fact.
- Minimum decisive experiment: route0 vs phase-2/residual variants with residual activation maps, dynamic-detail gains, static-quality preservation, and matched budgets.
- Likely failure mode: residual capacity is competitive but not better; decomposition is post hoc.
- Implementation burden: medium-high.
- Top-venue acceptance path: future backup if diagnostics reveal localized route0 failure and residual specialization becomes measurable.

### I4 - Confidence-Aware Priors for Masks, Tracks, and Flow

- Problem anchor: binary masks and broad priors are brittle near occlusions, boundaries, and static/dynamic leakage.
- Concrete method: weight dynamic ROI, static exclusion, flow, and track losses by confidence from prior validity and agreement cues.
- Dominant contribution: confidence-weighted supervision.
- Nearest related work: USplat4D, Prior-Enhanced GS, Shape of Motion, SplatFlow.
- Why not merely incremental: only if confidence is calibrated and improves hard dynamic regions rather than hiding them.
- Minimum decisive experiment: binary prior vs confidence prior with calibration, cue ablations, and dynamic/static metrics.
- Likely failure mode: heuristic soup or confidence selects easy pixels.
- Implementation burden: medium-high.
- Top-venue acceptance path: merged into C11 with a smaller predeclared reliability field.

### I5 - Dynamic Frequency/Detail Preservation under Fixed Budget

- Problem anchor: moving-object blur is a detail/frequency failure, not just aggregate color error.
- Concrete method: audit and optionally optimize high-frequency dynamic detail under matched realized point budgets.
- Dominant contribution: metric and budget framing.
- Nearest related work: AdaGaR, Multi4D, SharpTimeGS, MAPo.
- Why not merely incremental: only if detail metrics are validated and drive method decisions.
- Minimum decisive experiment: equal-budget dynamic-detail table plus visual crops and artifact/flicker checks.
- Likely failure mode: edge/detail proxies reward artifacts.
- Implementation burden: medium.
- Top-venue acceptance path: retained as an evaluation axis, not a standalone lead.

### I6 - Track-Anchored Scaffold Residual Motion

- Problem anchor: unanchored scaffold residuals underperform; track/flow anchoring might make them useful.
- Concrete method: LoRA route0 plus scaffold residuals supervised by dense long-range track-flow in dynamic regions.
- Dominant contribution: correspondence-anchored scaffold residuals.
- Nearest related work: MoSca, Prior-Enhanced GS, Shape of Motion, MotionGS.
- Why not merely incremental: difficult. Needs a clear distinction from scaffold-plus-track prior work.
- Minimum decisive experiment: route0 vs scaffold-no-flow vs scaffold-track-flow with active `lambda_track_flow`, valid masks, scaffold norms/entropy, and dynamic/static metrics.
- Likely failure mode: scaffold adds no value beyond route0 plus flow; track priors fail under occlusion/flames.
- Implementation burden: high.
- Top-venue acceptance path: not selected; revisit only if C11 diagnostics show scaffold is needed.

### I7 - Unanchored Scaffold Residual Motion

- Problem anchor: add scaffold residual motion without active correspondence.
- Concrete method: KNN-attached scaffold nodes with smoothness/regularization.
- Dominant contribution: none strong enough.
- Nearest related work: MoSca, Prior-Enhanced GS.
- Why not merely incremental: it is incremental and locally weak.
- Minimum decisive experiment: none as a lead; use as negative control.
- Likely failure mode: residual remains unused or hurts route0.
- Implementation burden: medium.
- Top-venue acceptance path: eliminated.

### I8 - Naive Broad Flow Supervision

- Problem anchor: flow supervision applied broadly.
- Concrete method: cached/file-mask flow loss without reliability, core, boundary, or static-anchor separation.
- Dominant contribution: negative control.
- Nearest related work: MotionGS, SplatFlow, PaMoSplat, Flow4DGS-SLAM.
- Why not merely incremental: it is incremental.
- Minimum decisive experiment: use only to show broad flow fails.
- Likely failure mode: occlusion/boundary noise damages optimization.
- Implementation burden: low-medium.
- Top-venue acceptance path: eliminated.

### I9 - Hard Static Conversion

- Problem anchor: static/dynamic leakage under soft routing.
- Concrete method: thresholded irreversible conversion between dynamic and static Gaussian sets.
- Dominant contribution: negative ablation against reversible routing.
- Nearest related work: SWinGS, SplatFlow, SharpTimeGS, Hybrid 3D-4DGS, RiGS.
- Why not merely incremental: it is crowded and brittle.
- Minimum decisive experiment: use only as ablation showing why reversible routing and anchors are preferable.
- Likely failure mode: early irreversible mistakes damage motion and background.
- Implementation burden: medium.
- Top-venue acceptance path: eliminated.

### I10 - Blur Curriculum and Early Part-Basis Tried Forms

- Problem anchor: early blur/part-basis attempts did not solve fast-motion smear.
- Concrete method: blurred training targets or unsupervised part bases without route0/track conditioning.
- Dominant contribution: historic controls.
- Nearest related work: MAPo, PaMoSplat, MoE-GS, HiCoM for part specialization.
- Why not merely incremental: tried forms are not new and were weak.
- Minimum decisive experiment: none for these exact forms.
- Likely failure mode: blur attacks symptoms; part assignments unstable.
- Implementation burden: low for blur, medium for part bases.
- Top-venue acceptance path: eliminated; part-aware ideas require a new route0/track-conditioned formulation.

### C11 - Self-Calibrated Prior Reliability Field

- Problem anchor: the key problem is not lack of priors but lack of a principled decision about when priors are trustworthy.
- Concrete method: compute a small reliability field from predeclared cues: prior validity, dynamic-core membership, boundary distance, rendered-flow/prior-flow agreement, and static-anchor consistency. Use it to weight dynamic ROI, static exclusion, rendered-flow, track-flow, and optional detail/residual supervision.
- Dominant contribution: calibrated reliability-gated training-time priors for hard dynamic regions.
- Nearest related work: USplat4D, Prior-Enhanced GS, MotionGS, SplatFlow, Shape of Motion, PaMoSplat.
- Why not merely incremental: C11 makes reliability the mechanism and requires calibration/coverage evidence, rather than adding another prior loss.
- Minimum decisive experiment: route0, binary priors, render-gate, boundary/static-anchor, and self-calibrated reliability on matched scenes/budgets, with cue-drop ablations and calibration.
- Likely failure mode: reliability becomes heuristic soup or avoids hard pixels.
- Implementation burden: medium-high but bounded if the cue set is small.
- Top-venue acceptance path: show that calibrated reliability predicts prior usefulness, retains hard dynamic pixels, improves dynamic metrics, preserves static quality, and explains failures of broad flow/scaffold.

### C12 - Reliable Temporal Detail Transport

- Problem anchor: fast moving dynamic details vanish because high-frequency evidence is not temporally transported through reliable correspondence.
- Concrete method: use reliable correspondences and route0 rendered motion to transport dynamic detail/residual supervision across frames into a localized residual/detail lane.
- Dominant contribution: temporal detail persistence.
- Nearest related work: AdaGaR, Multi4D, PaMoSplat, Shape of Motion.
- Why not merely incremental: potentially more novel than flow loss if it transports detail rather than only supervising displacement.
- Minimum decisive experiment: transport vs no-transport with detail/flicker metrics and dynamic crops.
- Likely failure mode: flicker, ghost trails, or hallucinated sharpness.
- Implementation burden: high; current repo lacks a clear detail-transport lane.
- Top-venue acceptance path: eliminated for now as too speculative; preserve for later.

### C13 - Competitive Route0-Residual Budget Allocator

- Problem anchor: dynamic detail may be underallocated under a fixed Gaussian budget.
- Concrete method: competitively allocate capacity between stable route0/coherent motion and ephemeral high-frequency residual Gaussians using dynamic error/detail/temporal-consistency signals.
- Dominant contribution: fixed-budget route0/residual allocation.
- Nearest related work: Multi4D, SharpTimeGS, MAPo, SpeeDe3DGS.
- Why not merely incremental: only if allocation is algorithmically clean and auditable.
- Minimum decisive experiment: equal realized point counts, allocation curves/maps, dynamic-detail gains, static/flicker checks.
- Likely failure mode: dynamic densification heuristic with more bookkeeping.
- Implementation burden: high.
- Top-venue acceptance path: semifinal loss; future backup seed if reliability is not enough.

### C14 - Boundary-Aware Static-Anchor Negative Space

- Problem anchor: flow/mask losses are most dangerous at motion boundaries and near static background that can become ghosted.
- Concrete method: decompose pixels into dynamic core, uncertain boundary ring, and static-anchor negative space; apply flow/detail losses only in core, neutral/soft boundary treatment, and static-anchor constraints outside core.
- Dominant contribution: clean deterministic reliability baseline and static-leakage safeguard.
- Nearest related work: SWinGS, SplatFlow, Flow4DGS-SLAM, Prior-Enhanced GS.
- Why not merely incremental: too modest alone, but valuable as the interpretable ablation bridge into C11.
- Minimum decisive experiment: core-only vs boundary-ring vs static-anchor vs C11 reliability with boundary crops and static ghost metrics.
- Likely failure mode: ring width/mask morphology overfits; suppresses legitimate boundary motion.
- Implementation burden: medium-low; remote branches already suggest boundary/static-anchor lanes.
- Top-venue acceptance path: component of C11, not standalone lead.

### C15 - ADAGS Failure Atlas plus Mechanism Screen

- Problem anchor: ADAGS has many runs but no claim-ready failure taxonomy tying mechanisms to visible failures.
- Concrete method: diagnostic failure atlas over cooking-scene dynamic GS with causal mechanism screen and a compact reliability-gated method case study.
- Dominant contribution: transparent diagnostic protocol plus method-backed case study.
- Nearest related work: MonoDyGauBench, D4RT, MotionScale, AdaGaR, Multi4D.
- Why not merely incremental: only if it teaches the field reusable failure modes and validates metrics beyond this codebase.
- Minimum decisive experiment: completed diagnostic table, qualitative panels, metric validity checks, and one compact method that fixes a diagnosed failure.
- Likely failure mode: internal ablation report, too narrow for a top venue.
- Implementation burden: medium.
- Top-venue acceptance path: backup direction and required reporting frame for C11.

## Independent Review Panel

Reviewer roles:

- Reviewer 1: skeptical novelty reviewer.
- Reviewer 2: CVPR/ICCV/NeurIPS computer vision and methodology reviewer.
- Reviewer 3: pragmatic implementation reviewer.

### Panel Scores

| ID | R1 novelty | R2 method | R3 implementation | Mean | Panel accept chance if executed well |
| --- | ---: | ---: | ---: | ---: | ---: |
| I1 | 6.0 | 6.5 | 7.5 | 6.67 | 35-45% |
| I2 | 5.0 | 6.0 | 6.5 | 5.83 | 25-30% |
| I3 | 5.0 | 6.0 | 6.5 | 5.83 | 25-35% |
| I4 | 5.0 | 6.5 | 6.0 | 5.83 | 30% |
| I5 | 4.0 | 5.5 | 5.5 | 5.00 | 18-25% |
| I6 | 4.0 | 5.0 | 5.5 | 4.83 | 15-25% |
| I7 | 2.0 | 3.0 | 2.5 | 2.50 | 3-5% |
| I8 | 2.0 | 2.0 | 2.0 | 2.00 | 3% |
| I9 | 2.0 | 2.0 | 2.0 | 2.00 | 3-5% |
| I10 | 2.0 | 3.0 | 2.5 | 2.50 | 5% |
| C11 | 6.0 | 7.0 | 7.0 | 6.67 | 38-40% |
| C12 | 6.0 | 6.5 | 5.0 | 5.83 | 25-35% |
| C13 | 5.0 | 6.0 | 6.0 | 5.67 | 25-32% |
| C14 | 5.0 | 7.0 | 7.0 | 6.33 | 28-38% |
| C15 | 6.0 | 6.5 | 7.0 | 6.50 | 35-40% |

### Reviewer Memory

Recurring objections:

- Flow, scaffolds, static/dynamic separation, uncertainty, allocation, and motion specialization are all crowded.
- Global PSNR is not a valid claim surface for the visible failure.
- Dynamic masks and diagnostic metrics may be noisy or circular.
- Route0 is strong and must not be beaten only by extra budget or hidden metric choices.
- Reliability can degenerate into heuristic soup or easy-pixel selection.
- Diagnostic-only papers risk looking internal unless the failure taxonomy generalizes.

Top criticisms requiring rebuttal:

1. "This is just optical flow or masks added to dynamic GS."
2. "The method does not beat route0 on dynamic-region metrics while preserving static quality."
3. "The diagnostics are self-serving or noisy."
4. "Reliability suppresses hard pixels instead of solving them."
5. "Boundary/static-anchor zoning is mask morphology, not a paper."

### Rebuttal Rulings

All three reviewers received the same rebuttal and were asked to rule.

| Criticism | Ruling | Required resolution |
| --- | --- | --- |
| Just flow/masks | Partially accepted | Select C11+C14, not I1 alone; predefine reliability; show ablations and calibration. |
| No route0 win on right metrics | Accepted | Make route0/dynamic-region/static-quality/matched-budget win a hard gate. |
| Self-serving diagnostics | Accepted | No residual fallback for eval, mask sensitivity, manual/file masks, visual disagreement cases. |
| C11 heuristic soup/easy pixels | Partially accepted | Report coverage over hard dynamic pixels and reliability-bin stratification; kill if easy-pixel selection dominates. |
| C14 mask morphology | Accepted as component, rejected as lead | Use C14 as deterministic baseline/static safeguard, not headline novelty. |

Consensus after rebuttal: **C11 wins conditionally**, packaged with C14, I1, and C15. No-winner is too harsh for idea selection because C11 now has a falsifiable mechanism and evidence plan, but no result should be claimed yet.

## Tournament Rounds

### Qualifiers

Advanced:

- I1 - best existing empirical signal but too crowded alone.
- I2 - necessary diagnostic infrastructure but too thin alone.
- I3 - plausible higher-novelty residual story but crowded.
- I4 - confidence version of prior gating, merged into C11.
- C11 - strongest mechanism if calibrated and falsifiable.
- C13 - strong budget/allocation framing, but crowded by Multi4D/SharpTimeGS/MAPo.
- C14 - clean boundary/static-anchor mechanism, not enough alone.
- C15 - best framing and transparency.

Eliminated:

- I5 lost because detail/frequency is an evaluation axis without a concrete enough method.
- I6 lost because track-anchored scaffold is too close to MoSca/Prior-Enhanced GS and higher burden.
- I7 lost because unanchored scaffold is locally mixed-negative and crowded.
- I8 lost because naive broad flow is a negative control.
- I9 lost because hard conversion is brittle and crowded.
- I10 lost because blur/early part-basis are tried weak forms.
- C12 lost because detail transport is interesting but too speculative and high burden for the next ADAGS paper.

### Quarterfinals

1. C11 vs I4: C11 wins. It preserves the confidence idea but makes reliability a predeclared, calibrated mechanism.
2. C14 vs I1: no standalone winner. C14 wins as the deterministic baseline/safeguard; I1 becomes the rendered-flow instantiation.
3. C15 vs I2: C15 wins. It adds causal mechanism screen and method case study to diagnostics.
4. C13 vs I3: C13 narrowly wins as an allocation story, but both are relegated because the field is crowded and implementation burden is high.

### Semifinals

1. C11 package vs C14/I1: C11 wins by absorbing C14 and I1. C14 and I1 are essential components, not separate leads.
2. C15 vs C13: C15 wins as the safer paper frame. C13 remains a future seed if reliability reveals capacity allocation as the bottleneck.

### Finals

Final: C11 package vs C15.

C11 wins because it can become a method paper with a clear mechanistic thesis:

> Dynamic Gaussian Splatting does not need more priors everywhere; it needs a calibrated decision about where priors are trustworthy enough to supervise motion and static/dynamic separation.

C15 is the backup because it is necessary infrastructure and may itself become publishable if the failure taxonomy and mechanism screen are rigorous, but it is weaker as the primary top-venue direction without a compact method.

## Final Selection

### Primary Direction

**Self-Calibrated Reliability-Gated Priors for Dynamic Gaussian Splatting**

Proposed paper shape:

1. Diagnose that route0 is stable but broad priors fail because masks/flow/tracks are unreliable around boundaries, occlusions, and static/dynamic leakage.
2. Define a small reliability field over dynamic-core pixels using predeclared cues.
3. Use reliability to gate rendered/track-flow, dynamic ROI, static exclusion, and optional detail/residual losses.
4. Include C14 dynamic-core/boundary/static-anchor zoning as the deterministic baseline and static-leakage safeguard.
5. Evaluate with C15 diagnostics: dynamic quality, static quality, static ghosting, flow consistency, detail/flicker, realized budget, and visual panels.

### Backup Direction

**ADAGS Failure Atlas and Mechanism Screen**

Use if C11's reliability field is informative but does not deliver enough dynamic-region improvement. The backup paper would emphasize:

- failure taxonomy for N3V cooking-scene dynamic GS,
- route0 vs flow vs scaffold vs boundary/static-anchor mechanism screen,
- metric validity and failure cases,
- compact reliability-gated case study,
- reusable guidance for when priors help or hurt dynamic GS.

## Negative Knowledge Preserved

- Unanchored scaffold residual motion is archived as a negative control.
- Naive broad flow supervision is archived as the required failure case for reliability-gated flow.
- Hard static conversion is ablation-only.
- Blur curriculum and early part-basis are historic controls, not leads.
- Dynamic detail/frequency and fixed budgets remain necessary evaluation axes, not standalone paper contributions.
- Track-anchored scaffold remains conditional only if reliability diagnostics show a residual scaffold is necessary and active.
- Route0/residual specialization remains a future seed only if residual activation can be measured.

## Minimum Decisive Experiment For The Winner

Do not run yet; this is the required future experiment before any paper claim.

Scene/control:

- Same three mechanism-screen scenes: `cut_roasted_beef`, `flame_steak`, `sear_steak`.
- Same checkpoint policy, same realized point-budget policy, same data splits.
- Evaluation masks cannot use residual fallback.

Methods:

1. LoRA route0.
2. Binary dynamic ROI/static exclusion.
3. Naive broad flow.
4. Render-gated flow.
5. C14 boundary/static-anchor deterministic baseline.
6. C11 self-calibrated reliability field.

Metrics:

- global PSNR/SSIM/LPIPS,
- dynamic-mask PSNR,
- static-region PSNR,
- static ghost score,
- dynamic detail and artifact/flicker checks,
- track/rendered-flow L1,
- route/dynamic probability diagnostics,
- realized point counts and dynamic point density,
- reliability calibration/error-prediction,
- coverage of hard dynamic-core pixels by reliability bins,
- qualitative panels and failure cases.

Win condition:

- C11 must improve dynamic quality over route0 and render-gate while preserving static-region quality and static ghosting under matched realized budget.
- Reliability must predict where priors help and must retain enough hard dynamic pixels.

Kill condition:

- If reliability selects only easy pixels, suppresses boundaries without fixing them, degrades static quality, or fails to beat route0 on dynamic metrics, stop and pivot to C15/no-winner restart.

## Literature Links Used

- MoSca: https://arxiv.org/abs/2405.17421
- Prior-Enhanced GS: https://arxiv.org/abs/2512.11356
- MotionGS: https://arxiv.org/abs/2410.07707
- SplatFlow: https://arxiv.org/abs/2411.15482
- RiGS: https://arxiv.org/abs/2605.23672
- MAPo: https://arxiv.org/abs/2508.19786
- MoE-GS: https://arxiv.org/abs/2510.19210
- AdaGaR: https://arxiv.org/abs/2601.00796
- Multi4D: https://arxiv.org/abs/2606.22197
- SharpTimeGS: https://arxiv.org/abs/2602.02989
- D4RT: https://arxiv.org/abs/2512.08924
- MonoDyGauBench: https://brownvc.github.io/MonoDyGauBench.github.io/
- USplat4D: https://tamu-visual-ai.github.io/usplat4d/
- PaMoSplat: https://arxiv.org/abs/2605.10307
