---
type: paper
node_id: paper:cho2026_4d_scaffold_gs
title: "4D Scaffold Gaussian Splatting with Dynamic-Aware Anchor Growing for Efficient and High-Fidelity Dynamic Scene Reconstruction"
authors: ["Woong Oh Cho", "In Cho", "Seoha Kim", "Jeongmin Bae", "Youngjung Uh", "Seon Joo Kim"]
year: 2026
venue: "AAAI 2026"
external_ids:
  arxiv: "2411.17044"
tags: [dynamic-gs, densification, anchor, temporal-coverage, scaffold]
status: deep-dived
---

# 4D Scaffold Gaussian Splatting with Dynamic-Aware Anchor Growing

**Paper:** https://arxiv.org/abs/2411.17044
**Code:** https://github.com/raikuma/4D-Scaffold-GS
**Base method:** Scaffold-GS (Lu et al. 2024) — grid-aligned anchor points that
decode into neural Gaussians via shared MLPs — extended from 3D to a 4D
(space+time) anchor grid. Borrows the K-offsets-per-anchor convention directly
from Scaffold-GS. Positioned against direct 4D Gaussian methods (4DGS/Yang et
al. 2024, STG, Ex4DGS) as the storage-efficient alternative.

## One-line thesis
Scaffold-GS's anchor-growing rule accumulates 2D positional gradients as a
plain per-view average, which under-weights anchors that are only visible/
active for a few frames; reweighting that gradient accumulation by each
Gaussian's temporal-opacity duration (short-lived content gets amplified
gradients) fixes under-reconstruction of brief dynamic events without
increasing total storage.

## Problem / Gap
Direct 4D Gaussian methods (4DGS) reconstruct dynamic content well but cost
>6GB for a 10-second multi-view video, and naive compression via aggressive
pruning or motion interpolation removes Gaussians from dynamic regions,
sacrificing exactly the expressiveness needed for complex temporal change. A
naive 3D-to-4D extension of Scaffold-GS also fails on its own: its anchor
growing rule accumulates gradients uniformly across all frames, so dynamic
regions that only appear in a handful of frames contribute a diluted average
gradient and are never flagged for anchor growth, leaving short-lived dynamic
content under-reconstructed.

## Method
Anchors are seeded from the SfM point cloud at the initial timestamp and
placed on a spatiotemporal voxel grid; each anchor spawns K=10 neural 4D
Gaussians via learned 4D offsets (3D position + time). At a given render time,
each Gaussian's 3D center is computed by adding a learned linear motion term
(velocity × elapsed time) to its offset position, and its opacity is the
anchor-decoded base opacity multiplied by a generalized-Gaussian temporal
envelope centered on the Gaussian's own timestamp. Anchor growth is driven by
accumulated 2D positional gradients exactly as in Scaffold-GS, except each
per-frame gradient sample is weighted by the Gaussian's instantaneous temporal
opacity raised through an inverse-duration term, so anchors dominated by
short-lived (small-σ) Gaussians accumulate disproportionately large growth
signal. After training, anchors whose Gaussians all end up with negative
(pruned) opacity are removed.

## Assumptions
Designed for calibrated multi-view video capture (Neural 3D Video / N3DV and
Technicolor style rigs), not monocular video — the paper states applying it to
monocular video "can introduce additional challenges." Assumes an SfM point
cloud is available for anchor initialization and that scene motion is well
approximated by a per-Gaussian linear (constant-velocity) trajectory over its
active temporal window.

## Limitations / Failure Modes
The paper states the method "still suffers from reconstructing elements that
appear very shortly (1 or 2 frames)" — i.e., the dynamic-aware reweighting
mitigates but does not eliminate the short-duration under-reconstruction
failure mode it targets. It is explicitly scoped to multi-view capture; the
authors flag monocular video as introducing unaddressed additional
challenges.

## Reusable Ingredients
- **Temporal-coverage-weighted gradient accumulation** — reweight the
  densification/anchor-growth gradient signal by each primitive's temporal
  opacity duration so short-lived content is not diluted by long-lived
  content in the same accumulation window.
- **Generalized-Gaussian temporal opacity envelope** — `exp(-(|Δt|/σ)^β)` as a
  drop-in replacement for the standard temporal Gaussian used in prior 4DGS
  work, giving an extra shape parameter (β) to fit piecewise-persistent
  (on/off) dynamic content.
- **Linear (velocity-only) per-Gaussian motion model** — a 3-parameter motion
  representation as a cheap alternative to full deformation MLPs when paired
  with anchor-level (rather than per-Gaussian) feature storage.
- **Post-training anchor pruning by negative decoded opacity** — a cheap
  cleanup pass that removes anchors whose entire spawned Gaussian set never
  became visible.

---

### Deep Dive

#### Core Novelty
Relative to a straightforward 3D→4D extension of Scaffold-GS, the paper
changes exactly one mechanism: the anchor-growing gradient statistic. Instead
of Scaffold-GS's unweighted per-view average of 2D positional gradients, it
weights each contributing gradient sample by the Gaussian's temporal-opacity
value and an inverse-power of its temporal duration. The insight is that
under a naive average, a Gaussian that is only "on" for a few of N total
frames contributes a near-zero gradient on most frames, so its true per-frame
signal is drowned out by frames where it contributes nothing — reweighting by
temporal coverage restores that signal's influence on the grow/no-grow
decision.

#### Mathematical Formulation
- **Anchor initialization** (per-anchor point, 4D): 
  $$\mathbf{p} = (x_v, y_v, z_v, t_0), \; \forall v \in \mathbf{V}$$
  `V` is the SfM point set; every anchor is stamped with the same initial
  timestamp $t_0$. Evaluated once, at anchor initialization.

- **Neural Gaussian spawning** (per anchor, before rendering):
  $$\mathbf{x}_k = \mathbf{p} + \Delta\mathbf{x}_k, \quad k \in \{1,\dots,K\}$$
  $\Delta\mathbf{x}_k$ is a learned 4D offset (3 spatial + 1 temporal) decoded
  from the anchor feature; $K=10$ neural Gaussians are spawned per anchor.

- **Rendered Gaussian center** (per-Gaussian, at render time $t_r$, before
  rasterization):
  $$\mu_k = \mathbf{x}_k^{xyz} + h(t_r, \mathbf{x}_k^{t}, \mathbf{u})$$

- **Rendered Gaussian opacity** (per-Gaussian, at render time $t_r$, before
  rasterization):
  $$\alpha_k = \rho_k \cdot g(t_r, \mathbf{x}_k^{t}, \sigma_k)$$
  $\rho_k$ is the (temporally-flat) base opacity decoded by the opacity MLP;
  $g(\cdot)$ modulates it by temporal proximity.

- **Linear motion model**:
  $$h(t, \mathbf{x}^{t}, \mathbf{u}) = (t - \mathbf{x}^{t})\,\mathbf{u}$$
  $\mathbf{u}$ is a learned per-Gaussian velocity vector — only 3 scalar
  parameters, i.e. constant-velocity motion rather than a deformation field.

- **Temporal opacity envelope** (generalized Gaussian):
  $$g(t, \mathbf{x}^{t}, \sigma) = \exp\!\left(-\left(\frac{|t - \mathbf{x}^{t}|}{\sigma}\right)^{\beta}\right)$$
  $\sigma$ is the per-Gaussian temporal extent, $\beta$ a shape exponent
  (generalizes the standard temporal Gaussian used in prior 4DGS work; steeper
  falloff than a plain Gaussian is intended to better fit piecewise-persistent
  dynamic intervals).

- **Baseline (Scaffold-GS) anchor-growth gradient statistic** — unweighted
  mean over $N$ contributing frames/views, accumulated across training before
  each densification step:
  $$\nabla^{g} = \frac{\sum^{N} \lVert \nabla_{2D} \rVert}{N}$$

- **Proposed dynamic-aware gradient statistic** (replaces Eq. 7 at every
  densification step):
  $$\nabla^{g} = \frac{\sum^{N} w(\alpha', \sigma)\, \lVert \nabla_{2D} \rVert}{\sum^{N} w(\alpha', \sigma)}$$
  $\alpha' = g(t_r, \mathbf{x}^{t}, \sigma)$ is the time-variant opacity
  component (the same $g(\cdot)$ as the temporal envelope above) evaluated at
  each contributing frame.

- **Coverage weight function**:
  $$w(\alpha', \sigma) = \alpha' \left(\frac{1}{\sigma}\right)^{\gamma}$$
  $\gamma$ controls how aggressively short-$\sigma$ (short-duration) Gaussians
  are up-weighted relative to long-duration ones. New anchors are placed at
  the centers of voxels whose $\nabla^{g}$ exceeds a growth threshold (exact
  numeric threshold value not stated in the extracted text).

- **Training objective**:
  $$\mathcal{L} = (1-\lambda_{SSIM})\mathcal{L}_1 + \lambda_{SSIM}\mathcal{L}_{SSIM} + \lambda_{vol}\mathcal{L}_{vol}$$
  Standard reconstruction terms plus a volume regularizer; only the weights
  are novel choices (see Hyperparameters below), the loss form itself is not
  a contribution.

#### Algorithm / Pipeline Changes
1. Build the SfM point cloud as usual, then stamp every point with the
   scene's initial timestamp $t_0$ to form the 4D anchor set (Eq. 1).
2. Place anchors on a 4D voxel grid: spatial grid size 0.001, temporal grid
   size = one frame interval (0.0333 for N3DV, 0.02 for Technicolor).
3. Each anchor decodes, via four shared 2-layer MLPs (opacity, shape/
   covariance, color, velocity; hidden width 32, matching the 32-D anchor
   feature), $K=10$ neural Gaussians with learned 4D offsets (Eq. 2).
4. At render time $t_r$, compute each Gaussian's rendered center via the
   linear motion model (Eq. 5) and rendered opacity via the base opacity times
   the temporal envelope (Eqs. 4, 6), then rasterize as in standard 3DGS.
5. During training, accumulate the 2D positional gradient at each contributing
   frame, but weight each frame's contribution by the coverage weight
   $w(\alpha',\sigma)$ (Eqs. 8-9) instead of averaging uniformly; this replaces
   Scaffold-GS's Eq. 7 statistic used to trigger anchor growth.
6. Grow new anchors at voxel centers where the weighted gradient statistic
   exceeds the growth threshold (mechanism identical to Scaffold-GS; only the
   input statistic changes).
7. After training completes, prune any anchor for which every one of its
   spawned Gaussians has gone to negative (invalid) opacity, then cache MLP
   outputs to speed up inference-time rendering.

#### Key Hyperparameters & Design Choices
- Training length: 120K iterations, single NVIDIA A6000 GPU, ~3 hours.
- $\beta = 2$ (temporal opacity exponent), $\gamma = 1$ (coverage-weight
  exponent).
- Spatial voxel grid size: 0.001 (all scenes). Temporal voxel/grid size: one
  frame interval — 0.0333 for N3DV, 0.02 for Technicolor.
- $K = 10$ neural Gaussians per anchor (following Lu et al. 2024a /
  Scaffold-GS convention), fixed across experiments (a lighter "Ours-light"
  variant is also reported but its differing setting is not specified in the
  extracted text).
- Anchor feature dimension: 32; each of the four MLPs (opacity, shape, color,
  velocity) is a 2-layer network with hidden width 32.
- Activations: tanh (opacity head), exp (scaling/$\sigma$ head), sigmoid
  (color head). Color MLP additionally takes the viewing direction
  concatenated with the anchor feature as input.
- Loss weights: $\lambda_{SSIM} = 0.2$, $\lambda_{vol} = 0.01$.
- Learning rates and their decay schedules for anchor position/offset,
  anchor feature, and MLP parameters: "follow those of Scaffold-GS" — not
  restated numerically in this paper; exact values not specified here.
- Anchor-growth gradient threshold (numeric value that triggers a new anchor
  at a voxel): Not specified in paper (text only states growth occurs where
  $\nabla^g$ exceeds "each threshold" without giving the number).
  Densification start/stop iterations and the growth-check interval: Not
  specified in the extracted text.
- Anchor pruning rule: post-training removal of anchors whose Gaussians all
  have negative opacity (no numeric threshold needed — pure sign test).

#### Ablation Summary
From Table 3 (flame_steak scene), isolating the two novel components:
1. **Dynamic-aware (DA) anchor growing — largest single contributor.**
   Removing DA (keeping linear motion + the paper's own opacity model) drops
   PSNR from 29.57 to 25.77 (**-3.80 dB**) and LPIPS from 0.050 to 0.153.
2. **Compact/modified temporal opacity model — secondary contributor.**
   Replacing the paper's temporal opacity model with the standard 4DGS
   opacity (while keeping DA and linear motion) drops PSNR from 29.57 to
   27.93 (**-1.64 dB**) and LPIPS from 0.050 to 0.065, while increasing
   storage from 149 MB to 195 MB.
   Flag: dynamic-aware anchor growing is clearly the dominant component
   (roughly 2.3x the PSNR impact of the opacity-model change), and it is also
   the component this wiki's gap-map connection cares about.

#### Implementation Reality
- **Framework:** PyTorch; the public repo states "most of the code is built
  upon the excellent work of Scaffold-GS" (i.e., extends the Scaffold-GS /
  gaussian-splatting codebase rather than a from-scratch implementation).
- **Key files:** `train.py` (training loop), `render.py` (rendering + FPS
  benchmarking), `metrics.py` (quality metrics), `gaussian_renderer/` (core
  rasterization/rendering pipeline), `scene/` (scene and anchor data
  handling), `arguments/` (configuration), `utils/`. The README does not
  give file-level detail on exactly which module implements the
  dynamic-aware anchor-growing statistic specifically — this could not be
  verified beyond the directory structure without opening individual source
  files, which was not done here.
- **Notable implementation details:** Dependencies are managed via Conda
  (`environment.yml`); training automatically produces rendering results,
  FPS, and quality metrics without a separate evaluation step; FPS is
  measured with CUDA synchronization and is noted by the authors as "roughly
  estimated." No specific paper-vs-code discrepancies (e.g., differing MLP
  depth or disabled densification windows) were confirmed from the README
  alone — verifying those would require reading the actual source files.

#### Failure Modes & Limitations
The paper explicitly states the method "still suffers from reconstructing
elements that appear very shortly (1 or 2 frames)" despite the dynamic-aware
reweighting targeting exactly this regime — the fix reduces but does not
solve extreme-short-duration under-reconstruction. It is also explicitly
scoped to multi-view video capture; the authors note that "applying it to
monocular videos can introduce additional challenges" without further
elaboration on what those challenges are.

---

## Relevance to ADAGS

Second occupant (beside [[papers/sandu2026_temporally_aware_densification]])
of the time-aware densification-gradient normalization axis. Any ADAGS
densification contribution must be positioned against BOTH: TAD-GS normalizes
by per-frame temporal opacity on flat 4D primitives; 4D Scaffold-GS reweights
anchor growth by temporal coverage in a scaffold/anchor representation.
Public code (github.com/raikuma/4D-Scaffold-GS) makes it a candidate
reimplementable baseline.

## Connections

- Pressures [[gap_map#G5 - Capacity Allocation Must Be Matched And Dynamic-Aware]]
- Pressures [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]

## Sources

- https://arxiv.org/abs/2411.17044
- https://github.com/raikuma/4D-Scaffold-GS
