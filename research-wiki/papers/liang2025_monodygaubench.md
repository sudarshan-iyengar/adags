---
type: paper
node_id: paper:liang2025_monodygaubench
title: "Monocular Dynamic Gaussian Splatting is Fast, Brittle and Scene-Dependent: Quantifying Failure Modes and Smoothness as a Robustness Prior"
authors: ["Yixuan Liang", "Junyi Yin", "Viktor Larsson", "Daniel Barath"]
year: 2025
venue: "TMLR"
external_ids:
  arxiv: "2412.04457"
tags: [benchmark, dynamic-gs, robustness, evaluation]
status: deep-dived
---

Note: the frontmatter `authors` field does not match the actual arXiv record for
2412.04457 (it lists "Yixuan Liang, Junyi Yin, Viktor Larsson, Daniel Barath",
apparently carried over from an earlier seed-list entry). The paper's real
authors are Yiqing Liang, Mikhail Okunev, Mikaela Angelina Uy, Runfeng Li,
Leonidas Guibas, James Tompkin, Adam W. Harley (Brown, Stanford, NVIDIA). The
paper's own title also changed between versions: v1 (2024-12) is "Monocular
Dynamic Gaussian Splatting is Fast and Brittle but Smooth Motion Helps" (the
version this deep dive draws on, since it is closest to the frontmatter title
and to what ADAGS cares about), and v2 (2025-06, the TMLR camera-ready) is
retitled "Monocular Dynamic Gaussian Splatting: Fast, Brittle, and Scene
Complexity Rules". The frontmatter `authors`/`title` fields were left
unchanged per the preservation rule for this pass; flag for correction in a
future metadata cleanup.

# Monocular Dynamic Gaussian Splatting Is Fast, Brittle And Scene-Dependent

**Paper:** https://arxiv.org/abs/2412.04457
**Code:** https://github.com/lynl7130/MonoDyGauBench_code
**Base method:** Benchmark/analysis paper, not a new method. Unifies and
compares 8 existing monocular dynamic Gaussian Splatting approaches
(EffGS, SpaceTimeGaussians/STG, DeformableGS, 4DGS, RTGS, and others) plus the
NeRF-based TiNeuVox and static 3DGS as baselines, all reimplemented in one
shared codebase.

## One-line thesis

Monocular dynamic Gaussian Splatting methods are collectively less robust than
a hybrid voxel NeRF baseline (TiNeuVox) on image quality, and within the GS
family, low-dimensional/constrained motion representations (low-order
polynomial/Fourier bases, small field MLPs) implicitly produce smoother
trajectories and are more robust to narrow baselines and fast motion than
per-Gaussian unconstrained or 4D representations — not because of an explicit
smoothness loss, but because architectural capacity constraints act as an
implicit regularizer on an ill-posed monocular optimization problem.

## Problem / Gap

Concurrent monocular dynamic GS papers (DeformableGS, 4DGS, STG, EffGS, RTGS,
etc.) each report state-of-the-art numbers under different codebases,
datasets, and evaluation splits, so claimed rankings are not comparable and
aggregate PSNR/SSIM hides scene-by-scene failure. In particular, full-image
metrics are dominated by static background pixels (3DGS, a purely static
method, scores competitively on full-image LPIPS by reconstructing the
background well while failing on the dynamic subject), and no prior work
isolates which factor — camera baseline, object motion speed, adaptive
densification, or motion-representation choice — actually drives quality and
failure.

## Method

The authors reimplement 8 dynamic GS methods plus TiNeuVox and static 3DGS in
a single codebase so all methods share data loading, camera handling, and
metric computation, removing implementation confounds. They evaluate on 50
scenes from 5 existing datasets (D-NeRF, Nerfies, HyperNeRF, NeRF-DS, iPhone/
DyCheck) plus a new controlled synthetic "instructive" dataset (SlidingCube /
RotatingCube) that sweeps camera baseline and object-motion distance
independently. Methods are categorized along three axes — motion reference
frame (iterative frame-to-frame vs. canonical offset-from-rest), motion
locality (per-Gaussian/local vs. shared field/global), and motion complexity
(low-order basis vs. MLP vs. full 4D) — and metrics are computed both on the
full image and on SAM-Track-derived dynamic-region masks (mPSNR, mSSIM,
mLPIPS) to separate static-background performance from actual dynamic-subject
reconstruction quality.

## Assumptions

Strictly monocular or quasi-monocular video input (single moving or static
camera per scene, no multi-view rig), non-rigid or rigid object motion against
a mostly-static background, and availability of per-scene camera poses (COLMAP
or dataset-provided, with HyperNeRF poses re-estimated by the authors after
masking dynamic content).

## Limitations / Failure Modes

Adaptive density control (clone/split/prune) is a major brittleness source:
Gaussian counts vary 2-3 orders of magnitude across scenes with no predictable
pattern, and once opacity-thresholded pruning deletes too many Gaussians early
the scene can become empty and never recover (a catastrophic, unrecoverable
failure, not a gradual quality loss). On the strictly-monocular iPhone/DyCheck
dataset, all Gaussian methods score 15-17 PSNR versus TiNeuVox's 19.35 —
multi-view or near-duplicate viewpoints across frames appear necessary for GS
methods to disambiguate structure. RTGS (a full 4D-Gaussian, no explicit
motion-model method) additionally hits out-of-memory crashes at narrow camera
baselines (B ≤ 5 in the synthetic benchmark). All methods plateau around
20-23 PSNR on NeRF-DS's specular scenes, contradicting some individual papers'
claims of handling reflective surfaces well. Reconstruction quality degrades
systematically as camera baseline narrows and/or object motion speed
increases, worst for RTGS/EffGS and best for DeformableGS, but this clean
ranking from the controlled synthetic data does not transfer cleanly to real
datasets, where scene-specific factors overwhelm method identity.

## Reusable Ingredients

- **Masked dynamic-region metrics (mPSNR/mSSIM/mMS-SSIM/mLPIPS) via SAM-Track**
  — full-image metrics are dominated by static background; masking to the
  moving-object region is necessary to see true reconstruction differences.
  Directly actionable for [[research-wiki/gap_map]] G7 (event/static
  diagnostics needed, global PSNR insufficient).
- **Controlled synthetic sweep (SlidingCube/RotatingCube) over independent
  camera-baseline and object-motion-distance axes** — isolates which capture
  factor drives failure, rather than confounding it with scene content.
- **Single shared codebase for multi-method reimplementation** — removes
  implementation/data-loading confounds when comparing published methods.
- **Motion-representation taxonomy (reference frame × locality × complexity)**
  as a classification scheme for categorizing any new motion model against
  prior work.
- **Frequency-domain (FFT) scene-complexity proxy** correlated with required
  Gaussian count and training time — a candidate scene-difficulty signal, with
  the caveat noted below that DyCheck's own difficulty metric failed to
  correlate with quality.
- **Train/test LPIPS gap as an overfitting diagnostic** — adaptive-densification
  GS methods show 2-3x wider train-test gaps than TiNeuVox, usable as an
  early-warning brittleness signal independent of final metrics.

---

### Deep Dive

#### Core Novelty
This is a benchmark/analysis contribution, not an architectural one: its
"novelty" is (1) apples-to-apples reimplementation of 8 dynamic-GS methods in
one codebase to remove confounds, and (2) the empirical finding that
constraining motion-representation capacity (low-order polynomial/Fourier
bases, small shared MLP fields) correlates with more robust reconstruction
under an ill-posed monocular setup than giving each Gaussian unconstrained or
4D degrees of freedom. The key insight is that monocular dynamic
reconstruction is fundamentally underconstrained, so the choice of motion
function family acts as an implicit prior/regularizer even with no explicit
smoothness loss term.

#### Mathematical Formulation
No novel loss is introduced; the paper's math is limited to characterizing
existing motion-representation families for its taxonomy:

- **Per-Gaussian iterative (local, frame-to-frame) motion:**
  $$f(i, 0) = G_i; \quad f(i, t) = f(i, t-1) + \delta G_{i,t}$$
  Each Gaussian's state at frame $t$ is its state at $t-1$ plus a learned
  per-frame delta $\delta G_{i,t}$ (position/rotation/scale/opacity offset).
  Evaluated once per Gaussian per frame during training/rendering, no shared
  parameters across Gaussians.

- **Per-Gaussian canonical-offset motion:**
  $$G_{i,t} = G_i + \delta G_{i,t}$$
  Each frame's Gaussian state is a direct offset from one canonical rest-state
  $G_i$, rather than chained from the previous frame — avoids iterative error
  accumulation but requires the offset function to span the full temporal
  range directly.

- **Field-based (shared/global) motion:**
  $$f_\theta(z_i, t) = \delta G_{i,t}; \quad G_{i,t} = G_i + \delta G_{i,t}$$
  where $z_i$ is a per-Gaussian embedding (e.g., canonical position) and
  $f_\theta$ is a shared network (MLP for DeformableGS, HexPlane
  factorization for 4DGS) queried per-Gaussian, per-frame. Parameters are
  shared across all Gaussians in the scene, which is the source of the
  implicit smoothness/regularization the paper attributes to field methods.

- **4D representation (RTGS-style):** Gaussians are parameterized directly in
  $\mathbb{R}^4$ (3D position + time), with no explicit motion function;
  spatial 3D Gaussians are obtained by conditioning/slicing at render time.
  No explicit smoothness formulation — this is the most unconstrained,
  least robust family in their results.

The paper does not define a formal smoothness metric or penalty; "smoothness"
is used descriptively to mean the reduced degrees of freedom of low-order
basis and shared-field representations relative to fully unconstrained
per-Gaussian or 4D motion.

#### Algorithm / Pipeline Changes
Not applicable in the usual sense (no new pipeline stage is added to a base
method). The benchmark's procedural contribution is the evaluation pipeline
itself:
1. Reimplement each of the 8 methods' motion model inside one shared
   Gaussian-Splatting scaffold (shared data loader, camera handling, densification
   controls where applicable), holding non-motion components fixed across methods.
2. Train each method on each of the 50 dataset scenes plus the synthetic
   SlidingCube/RotatingCube sweep, 3 independent runs per scene for
   mean ± std reporting.
3. Run SAM-Track over rendered/GT frames to obtain per-frame dynamic-object
   masks.
4. Compute both full-image and masked (mPSNR/mSSIM/mMS-SSIM/mLPIPS) metrics,
   plus FPS and wall-clock training time (RTX 3090) for every method/scene pair.
5. Post-hoc analyses: FFT magnitude spectra per scene (complexity proxy),
   Gaussian-count and train/test LPIPS-gap tracking across training, and
   correlation of DyCheck's existing difficulty metric ($\omega$) against
   observed quality (found to not correlate).

#### Key Hyperparameters & Design Choices
- Motion basis orders tested: polynomial order ≤ 2, Fourier order 2 (EffGS),
  RBF order 3 + polynomial order 1 (STG). Exact learning rates and further
  per-method hyperparameters are reported in the paper's Appendix C, not
  reproduced in the fetched summary — Not specified in the material extracted
  for this deep dive.
- Synthetic benchmark sweep: camera baseline $B \in \{1, 3, 5, 10, 20\}$,
  object motion distance $D \in \{0, 5, 10\}$, 60-frame sequences,
  RotatingCube adds a $\pi$-radian rotation.
  Hardware: RTX 3090 for all timing/FPS numbers.
- 3 independent runs per scene, mean ± std reported.
- Not specified in paper (from available extraction): exact densification
  gradient/opacity thresholds per method, MLP hidden-layer width/depth for
  DeformableGS's field, HexPlane resolution for 4DGS.

#### Ablation Summary
Not a component ablation (this is a cross-method comparison, not one method
with parts removed). The closest analogue — the controlled synthetic sweep —
functions as the ablation:
- Narrowing camera baseline (low $B$) and/or increasing object motion ($D$)
  monotonically degrades masked LPIPS for all methods; DeformableGS degrades
  least, RTGS and EffGS degrade most, and STG fails outright (OOM) at
  $B \le 5$.
- On full-image LPIPS, static 3DGS ranks near the top (0.358, competitive
  with dynamic methods) purely by exploiting static-background dominance;
  the same method ranks at the bottom once metrics are masked to the dynamic
  region — the single largest swing in the paper, underscoring that
  full-image metrics are not a valid signal for dynamic-reconstruction quality.
- No single numeric delta-PSNR "most impactful component" is reported since
  this is not an ablation of one architecture.

#### Implementation Reality
- **Framework:** PyTorch, built as a unified benchmark harness extending the
  original 3D Gaussian Splatting (Kerbl et al.) CUDA rasterizer, wrapping the
  8 evaluated methods' motion models inside it.
- **Key files:** Not verified by direct repository inspection in this pass —
  the repo (`lynl7130/MonoDyGauBench_code`) exists and is public with
  installation/run instructions per web search results, but its file
  structure was not read for this deep dive. Treat as unconfirmed; verify
  directly before relying on specific file paths.
- **Notable implementation details:** HyperNeRF camera poses were
  re-estimated by the authors (masking dynamic content then re-running
  COLMAP) because the original poses were found to be inconsistent; this
  correction improved static-region rendering but had mixed, method-dependent
  effects on dynamic-region quality.

#### Failure Modes & Limitations
The paper explicitly flags: (1) no scene-difficulty metric independent of
method performance exists — they tried DyCheck's $\omega$ metric and found it
does not correlate with observed reconstruction quality, a "chicken-and-egg"
problem for defining scene hardness without already knowing method outcomes;
(2) dense ground-truth depth/motion is unavailable for real dynamic scenes,
so some failure attributions are correlational rather than causal; (3) the
survey cannot be exhaustive given how fast the field is growing, so some
concurrent methods are missed; (4) even after camera-pose correction on
HyperNeRF, the effect on dynamic-region quality was inconsistent across
methods and the cause was not resolved.

---

## Relevance to ADAGS

Strong justification for dynamic-region diagnostic benchmark before claiming new method wins.

## Connections


## Sources

- arXiv abstract/listing: https://arxiv.org/abs/2412.04457
- arXiv HTML v1 ("...Fast and Brittle but Smooth Motion Helps"): https://arxiv.org/html/2412.04457v1
- arXiv HTML v2 ("...Fast, Brittle, and Scene Complexity Rules"): https://arxiv.org/html/2412.04457v2
- Code: https://github.com/lynl7130/MonoDyGauBench_code
- Project/qualitative-results page: https://brownvc.github.io/MonoDyGauBench.github.io/
