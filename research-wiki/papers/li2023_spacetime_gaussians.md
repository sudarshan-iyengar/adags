---
type: paper
node_id: paper:li2023_spacetime_gaussians
title: "Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis"
authors: ["Zhan Li", "Zhang Chen", "Zhong Li", "Yi Xu"]
year: 2023
venue: "CVPR 2024"
external_ids:
  arxiv: "2312.16812"
  doi: null
  s2: null
tags: ["dynamic-gaussians", "temporal-opacity", "capacity-allocation", "densification"]
added: 2026-07-14T22:18:30Z
---

# Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis

**Paper:** https://arxiv.org/abs/2312.16812
**Code:** https://github.com/oppo-us-research/SpacetimeGaussians
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023), extended with per-Gaussian temporal parameterization and a feature-splatting color model in place of spherical harmonics.

## One-line thesis

Giving each Gaussian a temporal-radial-basis opacity (so it can be born and die at a specific time) plus low-degree polynomial motion/rotation trajectories lets a single static-3DGS-style representation cover an entire multi-view video without a per-frame model or a deformation MLP, and a separate error/depth-guided sampling step adds genuinely new Gaussians in under-covered regions instead of only splitting existing ones.

## Problem / Gap

Prior per-frame 3DGS reconstructions of dynamic multi-view video store an independent Gaussian set per timestep, which is storage-heavy and has no temporal coherence between frames. Deformation-field approaches (canonical 3DGS + MLP warp) instead assume one static template deforms smoothly over time, which cannot represent content that only exists for part of the sequence (steam, flames, objects entering/leaving frame) — the MLP is forced to smear such transient content across the whole time range. Clone/split densification also only ever produces new Gaussians as local perturbations of ones the optimizer already placed, so it cannot recover coverage in regions the initial point cloud never populated.

## Method

Each Gaussian carries a temporal center and duration, a spatial opacity, a low-degree polynomial trajectory for position, a low-degree polynomial for rotation, and a 9-dimensional feature vector (instead of spherical-harmonic color) that is rasterized and decoded to RGB by a small shared MLP conditioned on view direction. At render time for a given frame, each Gaussian's opacity is evaluated by a 1D temporal Gaussian centered on its temporal center, so far-away-in-time Gaussians vanish smoothly rather than being included at full opacity. Separately from standard clone/split densification, a guided sampling step periodically identifies image patches with persistently high training error, back-projects rays through those patch centers using a coarse depth map produced by the current model, and seeds new Gaussians along those rays (with added positional noise), letting the optimizer prune the unhelpful ones later.

## Assumptions

Assumes calibrated multi-view (not monocular) video with a shared SfM point cloud across all timestamps for initialization, and that content that is transient in time is still observed from multiple synchronized cameras when it exists — the method is built and evaluated on multi-camera capture rigs (Neural 3D Video, Technicolor, Google Immersive), not casual single-camera footage.

## Limitations / Failure Modes

The paper states the method cannot train on-the-fly/streaming — it requires the full multi-view sequence available for per-scene optimization, unlike online/incremental dynamic reconstruction. It is scoped to multi-view capture; monocular dynamic video is not extensively explored. The authors note it would benefit from better initialization for faster convergence, implying the SfM-point-cloud start is a bottleneck. Guided sampling is driven by photometric training error and a self-produced coarse depth map — neither is an occlusion or visibility label, so it can only recover missing capacity where existing views already disagree with the render, not where a surface is currently unobserved from all cameras.

## Reusable Ingredients

- **Temporal-radial-basis opacity**: `σ(t) = σ_spatial · exp(-s_t · |t - μ_t|²)` — gives each primitive a soft birth/death window without any explicit lifecycle logic, purely through gradient-based optimization of `μ_t` and `s_t`.
- **Low-degree polynomial motion/rotation instead of an MLP**: cheap, per-Gaussian, closed-form trajectory that avoids the smoothing bias of a shared deformation network.
- **Feature-splatting color model**: splat a compact learned feature vector and decode color with a small shared MLP post-rasterization, rather than storing full per-Gaussian spherical-harmonic coefficients — cuts params per Gaussian roughly 48 → 9.
- **Error-and-depth-guided new-primitive sampling**: an explicit alternative to clone/split that seeds entirely new Gaussians in high-persistent-error regions using a coarse rendered depth to constrain 3D placement — a genuine "add capacity where the model currently has none" mechanism rather than only "grow from what's already there."
- **Lite/no-MLP color variant**: dropping the view/time-dependent color MLP and feature terms for a maximum-speed configuration — demonstrates the color-decoder can be optional without discarding the temporal-opacity/motion mechanism.

---

### Deep Dive

#### Core Novelty
Relative to static 3DGS, the paper's changes are (1) making opacity itself a function of time via a 1D Gaussian radial basis so primitives can locally exist only in a sub-interval of the sequence, (2) replacing per-Gaussian static position/rotation with low-degree polynomials in time so a single Gaussian traces a smooth trajectory instead of being re-optimized per frame or warped by a shared MLP, and (3) replacing per-Gaussian spherical harmonics with a compact splatted feature vector decoded by a small shared MLP. The key insight is that transient, time-local phenomena (things appearing/disappearing, non-rigid motion) are better modeled by giving *individual primitives* their own temporal extent and trajectory than by asking one canonical shape plus a global deformation field to cover the whole timeline — the temporal-opacity ablation is by far the largest single contributor in their results, confirming this is the load-bearing mechanism.

#### Mathematical Formulation

**Temporal opacity** (evaluated per-Gaussian, per-frame, before rasterization — replaces static per-Gaussian opacity):
$$\sigma_i(t) = \sigma_i^s \cdot \exp\!\left(-s_i^t \, |t - \mu_i^t|^2\right)$$
where $\sigma_i^s$ is the (time-independent) spatial opacity, $\mu_i^t$ is the learned temporal center of Gaussian $i$, and $s_i^t$ is a learned per-Gaussian scale controlling how quickly the Gaussian fades in/out around $\mu_i^t$ (its temporal "duration").

**Polynomial motion trajectory** (degree $n_p = 3$; evaluated per-Gaussian, per-frame, to obtain the position used for projection/rasterization):
$$\mu_i(t) = \sum_{k=0}^{3} b_{i,k}\,(t - \mu_i^t)^k$$
with learnable coefficients $b_{i,k}$ per Gaussian. The scaling matrix $S_i$ is kept time-independent (found not to need temporal variation experimentally).

**Polynomial rotation** (degree $n_r = 1$, i.e. linear-in-time quaternion interpolation; evaluated alongside the position to build the per-frame covariance):
$$q_i(t) = \sum_{k=0}^{1} c_{i,k}\,(t - \mu_i^t)^k$$
with learnable coefficients $c_{i,k}$.

**Splatted feature vector** (9-dimensional, replaces per-Gaussian SH coefficients; evaluated per-Gaussian before rasterization, then the *feature* — not color — is alpha-blended):
$$\mathbf{f}_i(t) = \left[\, \mathbf{f}_i^{base},\ \mathbf{f}_i^{dir},\ (t-\mu_i^t)\,\mathbf{f}_i^{time} \,\right]^T$$
where $\mathbf{f}_i^{base}$ (3D) is the base RGB-like term, and $\mathbf{f}_i^{dir}$, $\mathbf{f}_i^{time}$ (3D each) carry view- and time-dependent information. After rasterizing/blending to get per-pixel splatted features $\mathbf{F}^{base}, \mathbf{F}^{dir}, \mathbf{F}^{time}$, the final pixel color is
$$\mathbf{I} = \mathbf{F}^{base} + \Phi\!\left(\mathbf{F}^{dir}, \mathbf{F}^{time}, \mathbf{r}\right)$$
where $\Phi$ is a shallow 2-layer MLP evaluated once per pixel post-rasterization and $\mathbf{r}$ is the viewing direction. This is the mechanism that lets a single shared decoder absorb view/time-dependent appearance instead of storing full SH coefficients per Gaussian.

#### Algorithm / Pipeline Changes
1. **Initialization**: build Gaussians from an SfM sparse point cloud computed across all timestamps of the multi-view sequence (replaces per-frame or single-frame SfM init in static 3DGS).
2. **Per-frame forward pass**: for the current time $t$, evaluate each Gaussian's temporal opacity $\sigma_i(t)$, polynomial position $\mu_i(t)$, and polynomial rotation $q_i(t)$; Gaussians whose $\sigma_i(t)$ is effectively zero contribute nothing to that frame's render — this is evaluated before projection/rasterization, replacing the static-attribute lookup of vanilla 3DGS.
3. **Feature rasterization + decode**: rasterize the 9D feature vectors (instead of SH-evaluated RGB) to get per-pixel $\mathbf{F}^{base}, \mathbf{F}^{dir}, \mathbf{F}^{time}$, then run the shared 2-layer MLP $\Phi$ once per pixel to produce final color — inserted as a post-rasterization stage that replaces SH color evaluation.
4. **Standard density control** (clone/split/prune) runs as in 3DGS, but with more aggressive pruning to keep the temporally-extended model compact.
5. **Guided sampling** (runs periodically, up to 3 times per training sequence, as a step separate from clone/split): (a) after error stabilizes, identify image patches with persistently high training loss; (b) cast rays through the high-error patch centers; (c) use a coarse depth map (produced by the current model's own splatting) to constrain where along each ray new Gaussians are placed; (d) initialize new Gaussians at those depths with added positional noise; (e) let subsequent optimization/pruning remove any that turn out unhelpful.

#### Key Hyperparameters & Design Choices
- Motion polynomial degree $n_p = 3$; rotation polynomial degree $n_r = 1$.
- Feature vector dimensionality: 9 (3 base + 3 dir + 3 time), vs. ~48 SH coefficients in static 3DGS.
- Color decoder $\Phi$: shallow 2-layer MLP (exact hidden width not specified in the accessible text).
- Optimizer: Adam; loss: $L_1$ + D-SSIM, following standard 3DGS practice; explicit loss weight between the two terms: not specified in the accessible text.
- Learning rates/schedules for temporal center $\mu_i^t$, temporal scale $s_i^t$, and polynomial coefficients $b_{i,k}$, $c_{i,k}$: not specified in the accessible text.
- Guided sampling frequency: at most 3 times over a training run; exact iteration schedule, error threshold, and number of points added per round: not specified in the accessible text.
- Density-control pruning thresholds: paper states pruning is "more aggressive" than baseline 3DGS to keep the model compact; exact threshold values not specified in the accessible text.
- Training time: 40–60 minutes for a 50-frame multi-view sequence on an NVIDIA A6000.
- Lite variant: drops the feature/MLP color decoder entirely and uses only $\mathbf{F}^{base}$, trading appearance fidelity for maximum speed (reported 8K@60 FPS on an RTX 4090).

#### Ablation Summary
(Technicolor dataset; PSNR / D-SSIM / LPIPS, full model vs. one component removed at a time)
1. **w/o Temporal Opacity: 31.0 dB (−2.6 dB vs. full 33.6 dB), D-SSIM 0.063, LPIPS 0.153** — by far the largest single-component drop; flagged as the most impactful mechanism.
2. w/o Polynomial Motion: 32.6 dB (−1.0 dB), D-SSIM 0.045, LPIPS 0.099.
3. w/o Feature Splatting: 33.0 dB (−0.6 dB), D-SSIM 0.044, LPIPS 0.097.
4. w/o Guided Sampling: 33.3 dB (−0.3 dB), D-SSIM 0.041, LPIPS 0.085.
5. w/o Polynomial Rotation: 33.4 dB (−0.2 dB), D-SSIM 0.042, LPIPS 0.085 — smallest measured contribution of the five ablated components.
Full model: 33.6 dB, D-SSIM 0.040, LPIPS 0.084.

#### Implementation Reality
- **Framework:** PyTorch, extending the original `graphdeco-inria/gaussian-splatting` CUDA rasterizer codebase (standard for the 3DGS family).
- **Key files:** repository is `oppo-us-research/SpacetimeGaussians`; exact novel-logic file layout not verified beyond the repo's existence — a full file-by-file audit was not performed in this pass.
- **Notable implementation details:** not verified against the paper text in this pass; none captured beyond what the paper itself states.

#### Failure Modes & Limitations
The paper explicitly states the method does not support on-the-fly/streaming training — it requires the complete multi-view sequence up front for per-scene optimization. It is developed and evaluated for multi-view (not monocular) capture, and the authors note monocular dynamic video is not extensively explored. They also note the approach "would benefit from" better initialization strategies for faster convergence, implicitly flagging SfM-point-cloud initialization as a limiting factor.

---

## Relevance to This Project

It is a close capacity precedent for Phase 8, but the proposed contribution must be the visibility-to-capacity coupling rather than temporal opacity alone.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Sources

- https://arxiv.org/abs/2312.16812
- https://github.com/oppo-us-research/SpacetimeGaussians
