---
type: paper
node_id: paper:sun2025_splatflow
title: "SplatFlow: Self-Supervised Dynamic Gaussian Splatting in Neural Motion Flow Field for Autonomous Driving"
authors: ["Yangyang Sun", "Ziwei Zhu", "Yunsong Zhou", "Hongzi Zhu", "Jingwei Huang", "Yuhui Xu"]
year: 2025
venue: "CVPR"
external_ids:
  arxiv: "2411.15482"
tags: [dynamic-gs, flow, self-supervised, autonomous-driving]
status: deep-dived
---

# SplatFlow: Self-Supervised Dynamic Gaussian Splatting in Neural Motion Flow Field for Autonomous Driving

**Paper:** https://arxiv.org/abs/2411.15482
**Code:** Not found (no official repository located on arXiv, GitHub search, or Papers With Code as of this review; a project page with videos exists at sites.google.com/view/splatflow but no code link)
**Base method:** 4D/dynamic Gaussian Splatting with per-Gaussian deformation (general dynamic-GS line), combined with a LiDAR-supervised neural scene-flow field (analogous to NSFF-style implicit motion fields) and SEA-RAFT optical flow distillation.

## One-line thesis

A Neural Motion Flow Field (NMFF) — implicit MLPs that predict per-point 3D translation and rotation deltas between timestamps, pre-trained on LiDAR via bidirectional Chamfer distance — supervises both static/dynamic point decomposition and per-Gaussian warping, replacing manual 3D bounding-box tracks as the source of dynamic-object supervision in urban driving scenes.

## Problem / Gap

Existing dynamic Gaussian splatting methods for urban driving scenes (e.g., box-track-conditioned dynamic GS pipelines) require expensive manually-labeled, tracked 3D bounding boxes to separate and pose dynamic objects, which limits scalability to unlabeled driving logs. NeRF-based self-supervised alternatives avoid the labeling requirement but are slow to train and render. SplatFlow targets self-supervised dynamic decomposition and motion recovery without tracked boxes while retaining Gaussian splatting's speed.

## Method

SplatFlow pre-trains a Neural Motion Flow Field on raw LiDAR point clouds using bidirectional Chamfer distance to establish frame-to-frame correspondence, giving each 3D point a predicted translation and rotation delta. This flow thresholded per-point yields a 3D dynamic/static mask: static points are merged across time using ego-motion only, dynamic points are merged using ego-motion plus the predicted motion flow, and the two point sets initialize 3D Gaussians (static background) and 4D Gaussians (dynamic objects, aggregated into a common reference frame), respectively. During joint optimization, NMFF's motion deltas warp each dynamic Gaussian's mean and rotation across time, while an optical-flow distillation loss against SEA-RAFT predictions (matched directly and via warp-based photometric consistency) further refines dynamic-region identification and motion accuracy.

## Assumptions

Requires synchronized, calibrated multi-sensor input: camera imagery plus LiDAR point clouds, with known or odometry-estimated ego-vehicle trajectory and a removable ground plane (via pseudo-labels). Assumes low temporal frame rates such that per-object point aggregation across frames does not create severe occlusion conflicts, i.e., relatively slow/simple urban object motion rather than fast non-rigid deformation.

## Limitations / Failure Modes

The paper reports no explicit occlusion handling in the dynamic-point aggregation step, meaning objects with significant self-occlusion or occlusion by others across the aggregation window are not specifically modeled. Performance depends on LiDAR quality during NMFF pre-training and degrades with sparse or noisy point clouds. The method still requires an external optical-flow estimator (SEA-RAFT) at training time, and rendering runs at 40 FPS on Waymo (1920×1280) versus ~63 FPS for static 3DGS, i.e., a real throughput cost for the dynamic/NMFF machinery.

## Reusable Ingredients

- **Bidirectional Chamfer-distance LiDAR pre-training**: establishes frame-to-frame point correspondence and motion supervision purely from geometry, before any RGB optimization begins.
- **Motion-flow thresholding for static/dynamic decomposition**: turns a continuous predicted 3D flow field into a hard per-point dynamic mask, used to route points into separate static (3D Gaussian) vs. dynamic (4D Gaussian) initialization sets.
- **Optical-flow distillation loss with dual terms**: combines direct flow-vector matching with warp-then-photometric-compare, useful as a general recipe for injecting a 2D flow foundation model's signal into a 3D/4D representation.
- **Common-reference-frame aggregation for dynamic object initialization**: pools points from multiple timestamps (once de-warped by predicted motion) into one canonical point cloud per dynamic object to get denser initialization than any single frame provides.

---

### Deep Dive

#### Core Novelty

Relative to prior label-dependent dynamic driving GS methods, SplatFlow's change is to replace tracked-box supervision with an implicit LiDAR-pretrained motion field (NMFF) that simultaneously (a) produces the static/dynamic point partition and (b) supplies the per-Gaussian warp deltas used during rendering-time optimization. The key insight is that geometric point-correspondence (via Chamfer distance on LiDAR, which is comparatively unambiguous and label-free) is a more reliable self-supervision signal for both segmentation and motion than trying to infer either from photometric loss alone, and this geometric signal can then be reinforced/refined by a complementary 2D optical-flow distillation loss during the photometric optimization stage.

#### Mathematical Formulation

Neural Motion Flow Field — an implicit function predicting per-point motion between two timestamps, evaluated during LiDAR pre-training and again to warp Gaussians during joint optimization:
$$\phi_{t_1:t_2}(x_{t_1}, y_{t_1}, z_{t_1}) = [\Delta x, \Delta y, \Delta z, \Delta R]$$
where $(x_{t_1}, y_{t_1}, z_{t_1})$ is a 3D point at time $t_1$ and the output is a translation delta plus a rotation delta $\Delta R$ mapping the point/Gaussian to its pose at $t_2$.

Bidirectional Chamfer distance (Eq. 5) — LiDAR correspondence loss used to pre-train NMFF on raw point clouds $P, Q$ from two frames, evaluated before any Gaussian optimization begins:
$$d_{CD}(P,Q) = \sum_{p\in P}\min_{q\in Q}\|p-q\|^2 + \sum_{q\in Q}\min_{p\in P}\|q-p\|^2$$

4D Gaussian warping (Eqs. 6-8) — applied per-dynamic-Gaussian, per-timestep, before rasterization, using NMFF's predicted deltas to move a Gaussian from its canonical reference-frame pose to its pose at a queried time:
$$\mathcal{G}_i(t_2) = \{\mu(t_1) + \Delta\mu_{t_1:t_2},\; R(t_1)\cdot\Delta R_{t_1:t_2},\; S,\; \alpha,\; c\}$$
(scale $S$, opacity $\alpha$, color $c$ held fixed; only mean $\mu$ and rotation $R$ are warped).

Optical-flow distillation loss (Eq. 13) — a rendering-time loss combining direct 2D flow matching against SEA-RAFT predictions $\mathcal{F}$ and warp-based photometric consistency, applied after rendering each frame pair:
$$\mathcal{L}_F = \lambda_f \mathcal{L}_1(\mathcal{F}, \hat{\mathcal{F}}) + (1-\lambda_f)\,\mathcal{L}_1(\mathcal{I}_{next}, \mathcal{T}(\hat{\mathcal{I}}\,|\,\hat{\mathcal{F}}))$$
where $\hat{\mathcal{F}}$ is the rendered/predicted flow, $\mathcal{T}(\cdot|\hat{\mathcal{F}})$ warps the rendered image $\hat{\mathcal{I}}$ by that flow, and $\mathcal{I}_{next}$ is the actual next-frame image.

Total training loss (Eq. 14), summed over the joint optimization stage:
$$\mathcal{L} = \mathcal{L}_I + \lambda_1 \mathcal{L}_D + \lambda_2 \mathcal{L}_F + \lambda_3 \mathcal{L}_{sky} + \lambda_4 \mathcal{L}_{reg}$$
($\mathcal{L}_I$ is the standard photometric/SSIM image loss, $\mathcal{L}_D$ a depth term, $\mathcal{L}_{sky}$ a sky-region term, $\mathcal{L}_{reg}$ a regularizer.)

#### Algorithm / Pipeline Changes

1. Pre-train NMFF (eight ReLU-MLP stacks, hidden dim 128) directly on consecutive-frame LiDAR point clouds using bidirectional Chamfer distance (Eq. 5), for up to 4000 iterations at LR 8e-3 — this happens entirely before any Gaussian is instantiated.
2. Threshold each LiDAR point's predicted 3D motion-flow magnitude from NMFF to produce a binary dynamic/static point mask.
3. Merge static points across all timestamps using ego-motion compensation only; merge dynamic points across timestamps using ego-motion plus the NMFF-predicted per-point motion flow (i.e., de-warp dynamic points into a common reference frame).
4. Initialize static 3D Gaussians from the merged static point cloud; initialize dynamic 4D Gaussians per object from the aggregated, de-warped dynamic points in their common reference frame — this augments/replaces standard SfM-point Gaussian initialization.
5. During joint optimization, at each queried timestamp, warp every dynamic Gaussian's mean and rotation from the canonical reference frame to that timestamp via NMFF (Eqs. 6-8), then rasterize static 3D Gaussians and warped 4D Gaussians together.
6. Render optical flow alongside RGB and apply the SEA-RAFT distillation loss (Eq. 13) to refine both the rendered motion and, implicitly, the dynamic-region identification.
7. Standard 3DGS densification/pruning continues on this combined Gaussian set (densification every 500 iterations, opacity reset every 3000 iterations).

#### Key Hyperparameters & Design Choices

- NMFF architecture: 8 ReLU-MLP stacks, hidden dimension 128.
- Spherical harmonics degree: 3 per Gaussian.
- Densification interval: 500 iterations; opacity reset interval: 3000 iterations.
- Loss weights: $\lambda_1$ (depth) = 0.1, $\lambda_2$ (flow) = 0.005, $\lambda_3$ (sky) = 0.05, $\lambda_4$ (reg) = 0.001, $\lambda_{ssim}$ = 0.2, $\lambda_f$ (flow-loss internal mix) = 0.8.
- Learning rates: Gaussian position LR 1.6e-5 decaying to 1.6e-6; NMFF LR 1e-4 during joint optimization; NMFF pre-training LR 8e-3.
- NMFF LiDAR pre-training length: up to 4000 iterations.
- Dynamic-mask threshold on 3D motion flow magnitude: not specified in paper (exact numeric threshold not given).

#### Ablation Summary

Waymo dataset, full-scene metrics:

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| w/o NMFF prior (no LiDAR-pretrained motion prior) | 27.69 | 0.863 | 0.282 |
| w/o NMFF optimization (no warping during joint training) | 28.14 | 0.874 | 0.269 |
| w/o optical-flow distillation | 28.28 | 0.877 | 0.252 |
| **Full SplatFlow** | **28.99** | **0.880** | **0.249** |

Dynamic-region-only LPIPS: 0.317 (w/o NMFF prior) vs. 0.231 (full) — the largest single reported delta, indicating the LiDAR-pretrained NMFF prior is the most impactful component, especially for dynamic-object quality specifically (not just overall scene averages).

#### Failure Modes & Limitations

The paper states dynamic-point aggregation has no explicit occlusion-conflict handling, so significant occlusion during the aggregation window is not modeled. Quality is sensitive to LiDAR sparsity/noise since NMFF pre-training is geometry-driven. The method also depends on an external SEA-RAFT optical-flow estimator at training time and is slower than static 3DGS at inference (40 FPS vs. ~63 FPS on Waymo at 1920×1280), reflecting the added cost of NMFF-based per-Gaussian warping.

## Relevance to ADAGS

Warns that flow and decomposition are already occupied; ADAGS needs reliable gating and cooking-scene diagnostics.

## Connections

## Sources

- https://arxiv.org/abs/2411.15482
- https://arxiv.org/html/2411.15482
