---
type: paper
node_id: paper:wang2024_shape_of_motion
title: "Shape of Motion: 4D Reconstruction from a Single Video"
authors: ["Qianqian Wang", "Vickie Ye", "Hang Gao", "Weijia Zeng", "Jake Austin", "Zhengqi Li", "Angjoo Kanazawa"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2407.13764"
  doi: "10.48550/arXiv.2407.13764"
  s2: null
tags: [dynamic-reconstruction, tracks, motion-bases]
status: deep-dived
---

# Shape of Motion: 4D Reconstruction from a Single Video

**Paper:** https://arxiv.org/abs/2407.13764
**Code:** https://github.com/vye16/shape-of-motion (official, MIT license; ICCV 2025)
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023), extended with persistent per-Gaussian SE(3) trajectories driven by a shared low-rank motion-basis field; uses `gsplat` as the rasterizer backend and optionally 2D Gaussian Splatting for geometry.

## One-line thesis

A scene's full space of per-Gaussian 3D motion trajectories is regularized to lie in a low-dimensional subspace spanned by B ≪ N shared SE(3) basis trajectories, so that long-range 2D tracks (from an off-the-shelf tracker) plus monocular depth are sufficient to supervise persistent, globally 3D-consistent motion for every Gaussian, even ones that are only sparsely and non-simultaneously visible.

## Problem / Gap

Deformation-field methods (e.g. HyperNeRF, Deformable-3DGS) model motion as a single continuous per-frame warp, which has no explicit persistent-identity structure across time and cannot recover accurate long-range 3D trajectories for points that go through occlusion or leave/re-enter the camera frustum. Prior dynamic-GS methods that fit motion independently per-Gaussian per-frame are underconstrained from a single monocular view and drift because there is no mechanism to pool motion evidence across many Gaussians that move together. The paper targets accurate, complete, long-range 3D point trajectories (not just visually plausible novel-view rendering) as the core deliverable, which prior monocular 4D methods do not evaluate or achieve well.

## Method

Gaussians are defined once in a canonical frame t0 (position, rotation, scale, opacity, color) and their 3D pose at any other time t is obtained via a weighted combination of B=10 shared learnable SE(3) basis trajectories, with per-Gaussian fixed blend weights (motion coefficients) that sum after normalization — i.e. motion is factored into "what the B characteristic motions in the scene are doing" (basis trajectories) and "how much each Gaussian participates in each" (per-Gaussian coefficients). The model is supervised by standard photometric/depth/mask rendering losses plus two track-based losses that rasterize each Gaussian's 3D position at a target frame, project it back to 2D, and compare against externally computed long-range 2D tracks (TAPIR) and their associated monocular depth. A rigidity loss additionally penalizes k-NN pairwise 3D distance changes over time to keep locally rigid neighborhoods coherent. Initialization uses k-means clustering of track velocities to seed basis assignment and weighted Procrustes alignment to initialize the basis trajectories themselves, followed by a short L1 + temporal-smoothness pre-optimization stage before joint training.

## Assumptions

Monocular casual video with known (estimated) camera intrinsics/extrinsics per frame, scale-aligned monocular depth, long-range 2D point tracks, and a binary moving-object mask are all required as precomputed inputs (via MegaSaM, Depth Anything, TAPIR, and Track-Anything/SAM respectively). The method assumes scene motion is well-approximated by a small number (B=10) of shared globally-coherent rigid motion patterns — i.e. it is best suited to scenes with a modest number of independently-moving rigid or piecewise-rigid parts, not highly unstructured per-point deformation.

## Limitations / Failure Modes

Requires per-scene test-time optimization (no generalization/feed-forward inference), which blocks streaming use. Quality is bounded by the accuracy of the off-the-shelf depth, tracking, and masking priors it depends on — it degrades in textureless regions where TAPIR tracks are unreliable, and needs a manually/automatically produced moving-object mask as an explicit input rather than inferring dynamic regions itself. The ablation shows large degradation (EPE 0.082 → 0.141, δ₃D^0.05 43.0 → 30.4) when 2D track supervision is removed, meaning the method is heavily dependent on external correspondence quality rather than deriving motion structure purely from rendering losses.

## Reusable Ingredients

- **Shared low-rank SE(3) motion-basis factorization** — regularizes per-Gaussian motion into B shared rigid trajectories + per-Gaussian blend coefficients, giving persistent 3D identity and coherent motion pooling across sparsely-co-visible Gaussians.
- **3D trajectory rasterization for track supervision** — rasterize each Gaussian's rendered 3D position at a queried target time, project to 2D, and directly supervise against externally tracked correspondences and their depth (a technique for injecting long-range 2D correspondence into a 3D Gaussian pipeline).
- **k-NN rigidity loss** — penalizes changes in pairwise distance between nearby Gaussians across time, a cheap local-rigidity regularizer usable in any deformable-Gaussian setting.
- **Track-velocity k-means + weighted Procrustes initialization** — a principled way to seed a small number of rigid motion clusters and their basis trajectories before joint optimization, rather than random initialization.

---

### Deep Dive

#### Core Novelty

Relative to per-Gaussian independent deformation (raw per-point trajectories) or a single global deformation MLP, this paper factorizes motion into a shared low-rank basis: B=10 SE(3) trajectories shared across the whole scene, with each Gaussian holding only a fixed set of blend weights. The key insight is that most real dynamic scenes contain far fewer independent rigid-motion patterns than there are Gaussians, so pooling all Gaussians' evidence (photometric + track + depth) into a small shared basis set makes the per-basis trajectory well-constrained even though any single Gaussian may be visible/tracked in only a few frames — this is what lets the method recover complete long-range trajectories through partial occlusion instead of only where direct per-point evidence exists.

#### Mathematical Formulation

Per-Gaussian transform from canonical frame t0 to frame t, evaluated per-Gaussian before rasterization at each queried time:
$$T_{0\to t} = \sum_{b=1}^{B} w^{(b)} T_{0\to t}^{(b)}, \quad \sum_b \|w^{(b)}\| = 1$$
where $T_{0\to t}^{(b)}$ is the b-th shared learnable SE(3) basis trajectory (parameterized as 6D rotation + translation) and $w^{(b)}$ is the fixed per-Gaussian motion coefficient for basis b. This produces each Gaussian's mean $\mu_t$ (and orientation) at time t used by the standard 3DGS projection/rasterization:
$$\mu_0'(K,E) = \Pi(KE\mu_0) \in \mathbb{R}^2,\qquad \Sigma_0'(K,E) = J_{KE}\Sigma_0 J_{KE}^T \in \mathbb{R}^{2\times2}$$
$$\hat I(p) = \sum_{i\in H(p)} T_i \alpha_i c_i,\qquad \hat D(p) = \sum_{i\in H(p)} T_i \alpha_i d_i$$
(standard alpha-compositing over Gaussians $H(p)$ hit at pixel $p$).

3D trajectory rasterization (evaluated after rendering, at a queried target frame t′, to produce the correspondence supervision signal): the 3D world-space position each Gaussian reaches at t′ is itself alpha-composited per pixel from the canonical frame's rendering weights,
$$\hat X^{t\to t'}_w(p) = \sum_{i\in H(p)} T_i \alpha_i \mu_{i,t'}$$
then reprojected to obtain 2D correspondence and depth at t′:
$$\hat U_{t\to t'}(p) = \Pi\big(K_{t'} \hat X^{t\to t'}_c(p)\big),\qquad \hat D_{t\to t'}(p) = \big[\hat X^{t\to t'}_c(p)\big]_3$$

Loss terms (all evaluated after rendering, as training objectives):
$$\mathcal{L}_{recon} = \|\hat I - I\|_1 + \lambda_{depth}\|\hat D - D\|_1 + \lambda_{mask}\|\hat M - M\|_1$$
$$\mathcal{L}_{track\text{-}2d} = \|U_{t\to t'} - \hat U_{t\to t'}\|_1,\qquad \mathcal{L}_{track\text{-}depth} = \|d_{t\to t'} - \hat D(U_{t\to t'})\|_1$$
$$\mathcal{L}_{rigidity} = \big\|\,\mathrm{dist}(\hat X_t, C_k(\hat X_t)) - \mathrm{dist}(\hat X_{t'}, C_k(\hat X_{t'}))\,\big\|_2^2$$
where $U_{t\to t'}$/$d_{t\to t'}$ are the externally-computed 2D track and its depth (TAPIR + monocular depth), and $C_k(\cdot)$ returns the k-nearest-neighbor set of a Gaussian in 3D — the rigidity loss penalizes any change in pairwise neighbor distance over time.

#### Algorithm / Pipeline Changes

1. Precompute per-frame camera poses (MegaSaM), monocular depth aligned to metric/SfM scale, long-range 2D point tracks on a grid every 4 pixels (TAPIR), and a moving-object mask (Track-Anything/SAM).
2. Select canonical frame t0 as the frame with the maximum number of visible 3D tracks; initialize N Gaussian means (40k dynamic + 100k static) from sampled 3D track locations at t0.
3. Run k-means on vectorized per-track velocity vectors to form B=10 clusters; initialize each basis trajectory $T_{0\to t}^{(b)}$ via weighted Procrustes alignment between canonical-frame and target-frame track positions weighted toward that cluster; initialize each Gaussian's motion coefficients $w^{(b)}$ to decay exponentially with the Gaussian's distance from each cluster center.
4. Pre-optimize for 1000 iterations with an L1 photometric loss plus temporal-smoothness regularization (L2 on acceleration of basis quaternion/translation) before joint training, to stabilize basis trajectories.
5. Joint training for 500 epochs: sample 8 query frames per iteration, each paired with 4 target frames; for each pair, render photometric/depth/mask losses at the query frame and additionally rasterize/reproject 3D positions to the target frame for the 2D-track and track-depth losses, plus the k-NN rigidity loss between query and target 3D positions.
6. Static Gaussians (100k) are optimized as ordinary (non-time-varying) 3DGS points outside the moving-object mask; only the 40k dynamic Gaussians are driven by the motion-basis field.

#### Key Hyperparameters & Design Choices

- Motion basis count: B = 10 (all experiments)
- Gaussian counts: 40k dynamic + 100k static
- Pre-optimization: 1000 iterations (L1 + temporal smoothness)
- Main training: 500 epochs
- Learning rates (Adam): μ0 = 1.6×10⁻⁴, opacity = 1×10⁻², scale = 5×10⁻³, rotation R0 = 1×10⁻³, color = 1×10⁻², SE(3) motion bases = 1.6×10⁻⁴, motion coefficients = 1×10⁻²
- Loss weights: λ_depth = 0.5, λ_mask = 1.0, λ_track-2d = 2.0, λ_track-depth = 0.1, λ_rigidity = 0.1
- Batch composition: 8 query frames/iteration, 4 target frames per query
- 2D track grid spacing: every 4 pixels
- Temporal smoothness: L2 regularization on acceleration of quaternion and translation
- Training time: ~2 hours for 300 frames at 960×720 on one A100; rendering ~140 FPS

#### Ablation Summary

Measured on 3D tracking (EPE↓, δ₃D^0.05↑, δ₃D^0.10↑):

| Variant | EPE↓ | δ₃D^0.05↑ | δ₃D^0.10↑ |
|---|---|---|---|
| Full method | 0.082 | 43.0 | 73.3 |
| Translation-only bases (no SE(3)) | 0.093 | 42.3 | 69.9 |
| Per-Gaussian SE(3) (no shared basis) | 0.083 | 43.6 | 70.2 |
| Per-Gaussian translation only | 0.087 | 41.2 | 69.2 |
| No SE(3)/Procrustes initialization | 0.111 | 39.3 | 65.7 |
| No 2D track supervision | 0.141 | 30.4 | 57.8 |

Most impactful component: removing 2D track supervision entirely, by far the largest degradation (EPE +0.059, δ₃D^0.05 −12.6 points), confirming the method's dependence on external long-range correspondence rather than photometric/depth signal alone. Proper SE(3)/Procrustes initialization is the second-largest factor (EPE +0.029 if removed).

#### Implementation Reality

- **Framework:** PyTorch, built on the `gsplat` rasterizer (nerfstudio-project), with an optional 2D Gaussian Splatting mode (`--use_2dgs`).
- **Key files (repo `flow3d/`):** `scene_model.py` (Gaussian scene representation), `trajectories.py` (SE(3) motion-basis trajectories), `transforms.py` (SE(3)/coordinate transform utilities), `loss_utils.py` (reconstruction/track/rigidity losses), `init_utils.py` (k-means + Procrustes initialization), `trainer.py` (training loop), `renderer.py` (rendering), `params.py` and `tensor_dataclass.py` (parameter/tensor container definitions), `mesh_extractor.py` and `normal_utils.py` (geometry export), `metrics.py` (evaluation).
- Preprocessing (camera poses via MegaSaM, depth via Depth Anything, tracks via TAPIR, masks via Track-Anything/SAM) is documented as a separate pipeline (`preproc/README.md`) rather than folded into the main training script — these are treated as offline precomputed inputs, not differentiable modules in the training loop.

#### Failure Modes & Limitations

The paper explicitly states the method "still requires per-scene test-time optimization, hindering streamable applications," unlike feed-forward approaches. It is bounded by the quality of its input priors: depth and tracking degrade in textureless regions, and the moving-object mask must be supplied (via SAM/Track-Anything) rather than inferred, so segmentation errors propagate directly into which Gaussians receive motion-basis supervision at all.

## Relevance to ADAGS

Primary motivation for activating `lambda_track_flow`.

## Connections

## Sources

- https://arxiv.org/abs/2407.13764
- https://github.com/vye16/shape-of-motion
