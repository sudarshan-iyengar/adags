---
type: paper
node_id: paper:wang2025_shape_of_motion
title: "Shape of Motion: 4D Reconstruction from a Single Video"
authors:
  - Qianqian Wang
  - Vickie Ye
  - Hang Gao
  - Weijia Zeng
  - Jake Austin
  - Zhengqi Li
  - Angjoo Kanazawa
year: 2025
venue: ICCV
external_ids:
  arxiv: "2407.13764"
tags:
  - tracks
  - 4d-reconstruction
  - dynamic-reconstruction
status: deep-dived
---

# Shape of Motion: 4D Reconstruction from a Single Video

**Paper:** https://arxiv.org/abs/2407.13764
**Code:** https://github.com/vye16/shape-of-motion/
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023) rasterization/optimization, combined with off-the-shelf monocular depth, 2D long-range track, and camera-pose priors fused into a single dynamic scene representation.

## One-line thesis
Real-world scene motion is largely low-rank (rigid or near-rigid parts moving together), so representing every dynamic Gaussian's trajectory as a per-point blend of a small shared set of SE(3) basis trajectories — initialized from noisy off-the-shelf 3D tracks rather than learned from scratch — turns single-video 4D reconstruction from an ill-posed per-point motion-estimation problem into a well-regularized one.

## Problem / Gap
Prior monocular dynamic-scene methods depend on category-specific templates, are effective only in quasi-static scenes, or don't model 3D motion explicitly (e.g., deformation-field NeRFs that warp a canonical volume without persistent per-point trajectories). Giving every 3D Gaussian a free-form per-point trajectory, or routing it through a per-point deformation MLP, has too many degrees of freedom to fit reliably from a single casually captured video — the problem is inherently underconstrained without additional structure.

## Method
The paper attaches to each dynamic 3D Gaussian a canonical (position, orientation, scale, opacity, color) tuple and a rigid transform to any frame $t$ computed as a weighted blend of $B=10$ shared, learned SE(3) basis trajectories. Off-the-shelf monocular depth and long-range 2D tracks are lifted into noisy, uncertainty-weighted 3D tracks, which are clustered (k-means over per-track velocity vectors) and aligned via weighted Procrustes to initialize both the basis trajectories and the per-Gaussian blend weights before joint optimization begins. During training, Gaussians are rasterized at every frame with 3DGS/gsplat, and the combined photometric, depth, mask, 2D/depth-track, and rigidity losses jointly update canonical attributes, basis trajectories, and blend weights. A separate, non-moving static Gaussian set models the background alongside the dynamic Gaussians.

## Assumptions
Assumes a casually captured monocular video with externally supplied camera poses, monocular depth, long-range 2D point tracks, and a binary dynamic/static mask. Assumes scene motion is well-approximated by a small number of shared rigid or near-rigid motion groups rather than fully independent per-point motion.

## Limitations / Failure Modes
Requires per-scene test-time optimization with no feed-forward inference, which the authors note "hinder[s] streamable applications." Quality depends on and degrades with the off-the-shelf pose/depth/track priors, specifically in textureless regions or under large/fast motion. The method also needs a user-provided or externally computed dynamic/static mask — it does not discover moving regions on its own, though the authors note recent segmentation advances could automate this step.

## Reusable Ingredients
- **Low-rank SE(3) motion bases with per-point blend weights** — regularizes per-point 3D motion estimation by sharing a small set of global rigid trajectories across all points, instead of fitting free-form per-point trajectories or an MLP warp field.
- **Track-based motion initialization (k-means + weighted Procrustes)** — turns noisy, uncertainty-weighted 3D tracks into a non-random initial guess for both the shared motion bases and per-point blend weights, avoiding a hard from-scratch optimization.
- **Uncertainty/visibility-weighted track losses** — down-weights occluded or low-confidence tracks in both initialization and the 2D/depth track-supervision losses rather than trusting all tracker output equally.
- **As-rigid-as-possible (ARAP) pairwise regularizer over k-nearest neighbors** — penalizes changes in relative pairwise distances between nearby dynamic points across frames as a lightweight physical-plausibility prior.
- **Separate static vs. dynamic Gaussian pools** — keeps a non-moving background Gaussian set decoupled from the motion-basis machinery, avoiding wasted capacity/complexity on content that never moves.

---

### Deep Dive

#### Core Novelty
Instead of giving every Gaussian its own free-form trajectory (underconstrained from monocular video) or relying on a per-point deformation MLP, the paper represents the motion of the *entire dynamic scene* as a weighted linear combination of a small number ($B=10$) of shared SE(3) basis trajectories. This exploits the low-rank structure of real-world motion (rigid/near-rigid parts moving together) to regularize an otherwise ill-posed problem, and lets every dynamic Gaussian's persistent identity (its canonical attributes) be transported through time by a compact, globally shared motion field rather than per-point free parameters. The basis trajectories and per-Gaussian blend weights are additionally initialized from noisy, uncertainty-weighted 3D tracks via clustering and weighted Procrustes alignment, rather than being learned from a random start, which is the second key mechanism driving stability.

#### Mathematical Formulation
- Canonical Gaussian parameters (evaluated once, persistent object state): $\bar g_0 \equiv (\mu_0, R_0, s, o, c)$ — position $\mu_0 \in \mathbb{R}^3$, orientation $R_0 \in SO(3)$, scale $s \in \mathbb{R}^3$, opacity $o \in \mathbb{R}$, color $c \in \mathbb{R}^3$.
- Per-Gaussian rigid transform to time $t$, computed per-Gaussian before rasterization at each frame, as a weighted combination of $B$ shared learned basis transforms:
$$T_{0\to t} = \sum_{b=1}^{B} w^{(b)} T_{0\to t}^{(b)}, \qquad \sum_b \lVert w^{(b)} \rVert = 1$$
  Each basis $T_{0\to t}^{(b)}$ is parameterized as 6D rotation + translation; the same per-Gaussian weight $w^{(b)}$ blends rotation and translation components separately (linear blend skinning-style composition, but over rigid bases rather than a mesh skeleton).
- Resulting per-Gaussian pose at time $t$ (per-Gaussian, before rasterization):
$$\mu_t = R_{0\to t}\mu_0 + t_{0\to t}, \qquad R_t = R_{0\to t}R_0$$
- Reconstruction loss (after rendering, per frame): $\mathcal{L}_{recon} = \lVert \hat I - I\rVert_1 + \lambda_{depth}\lVert \hat D - D\rVert_1 + \lambda_{mask}\lVert \hat M - M\rVert_1$, with $\lambda_{depth}=0.5$, $\lambda_{mask}=1.0$.
- 2D/depth track-supervision losses (after rendering, tying rendered correspondence back to off-the-shelf 2D tracks): $\mathcal{L}_{track\text{-}2d} = \lVert U_{t\to t'} - \hat U_{t\to t'}\rVert_1$ and $\mathcal{L}_{track\text{-}depth} = \lVert \hat d_{t\to t'} - \hat D(U_{t\to t'})\rVert_1$, weights $\lambda_{track\text{-}2d}=2.0$, $\lambda_{track\text{-}depth}=0.1$.
- Rigidity/as-rigid-as-possible regularizer on dynamic Gaussians (sampled pairwise over k-nearest neighbors, as a loss term): $\mathcal{L}_{rigidity} = \lVert \mathrm{dist}(\hat X_t, C_k(\hat X_t)) - \mathrm{dist}(\hat X_{t'}, C_k(\hat X_{t'}))\rVert_2^2$, weight $\lambda_{rigidity}=0.1$, where $C_k(\cdot)$ is the k-nearest-neighbor set.

#### Algorithm / Pipeline Changes
1. Select a canonical reference frame $t_0$ as the frame where the most 3D tracks (lifted from off-the-shelf 2D tracks + monocular depth) are visible.
2. Initialize dynamic Gaussian means $\mu_0$ directly from N randomly sampled noisy 3D track locations at $t_0$ (no random init).
3. Run k-means clustering on the vectorized per-track velocity vectors to propose $B$ candidate rigid motion groups.
4. Initialize each of the $B$ SE(3) basis trajectories via weighted Procrustes alignment between the 3D point set at $t_0$ and at each other frame $t$, restricted to each cluster; weights come from per-track uncertainty and visibility scores (down-weighting occluded/unreliable tracks).
5. Initialize per-Gaussian basis blend weights $w^{(b)}$ to decay exponentially with the Gaussian's distance from its cluster's center.
6. Pre-optimize this initialization for 1000 iterations using an $\ell_1$ fitting loss plus temporal smoothness constraints, before joint scene optimization begins.
7. During main training, at every frame the per-Gaussian transform is recomputed via the basis-blend equation above, Gaussians are rasterized (3DGS/gsplat), and the combined photometric + depth + mask + track + rigidity losses are backpropagated jointly to canonical attributes, basis trajectories, and blend weights. Static background is modeled with a separate, non-moving Gaussian set (100k Gaussians) rendered alongside the 40k dynamic Gaussians.

#### Key Hyperparameters & Design Choices
- Number of SE(3) motion bases $B$: 10.
- Dynamic Gaussians: 40k; static Gaussians: 100k.
- Initialization pre-optimization: 1000 iterations.
- Main training: 500 epochs (300-frame sequences at 960×720).
- Learning rates: canonical means $\mu_0$ and basis parameters $1.6\times10^{-4}$; motion coefficients (blend weights) $1\times10^{-2}$; canonical rotation $R_0$ $1\times10^{-3}$; scale $s$ $5\times10^{-3}$.
- Loss weights: $\lambda_{depth}=0.5$, $\lambda_{mask}=1.0$, $\lambda_{track\text{-}2d}=2.0$, $\lambda_{track\text{-}depth}=0.1$, $\lambda_{rigidity}=0.1$; depth-gradient loss weight 1.0; motion smoothness weight 0.1.
- Reported rendering speed: ~140 FPS at inference.

#### Ablation Summary
iPhone dataset, full method: EPE 0.082, $\delta_{3D}^{.05}$ 43.0, $\delta_{3D}^{.10}$ 73.3.
- Removing 2D track supervision entirely is the single most damaging ablation: EPE 0.082 → 0.141 (+72% error), $\delta_{3D}^{.05}$ 43.0 → 30.4. **Most impactful component: long-range 2D track supervision.**
- Removing SE(3) motion-basis initialization (i.e., training bases/weights from scratch instead of track-derived init): EPE 0.082 → 0.111, $\delta_{3D}^{.05}$ 43.0 → 39.3.
- Replacing shared SE(3) bases with translational-only bases: EPE 0.082 → 0.093, $\delta_{3D}^{.05}$ 43.0 → 42.3 (rotation matters, but less than track supervision/init).
- Per-Gaussian SE(3) (no basis sharing/low-rank structure) is close to the full method on error (EPE 0.083) but scores slightly higher on some $\delta_{3D}$ thresholds while being far less constrained/generalizable — the basis representation is motivated by robustness/compactness, not by raw EPE alone.
- Per-Gaussian translation only (weakest ablation of the motion representation family): EPE 0.087, $\delta_{3D}^{.05}$ 41.2.

#### Implementation Reality
- **Framework:** PyTorch, built on top of the `gsplat` CUDA rasterization backend (`nerfstudio-project/gsplat`), with CuML used for GPU-accelerated clustering/init utilities.
- **Key files:** `flow3d/scene_model.py` (canonical Gaussian + basis-blended scene representation), `flow3d/trajectories.py` (SE(3) motion basis trajectories), `flow3d/transforms.py` (SE(3) transform operations), `flow3d/params.py` (learnable parameter containers), `flow3d/init_utils.py` (track-based clustering/Procrustes initialization described above), `flow3d/loss_utils.py` (reconstruction/track/rigidity losses), `flow3d/trainer.py` (main optimization loop), `flow3d/renderer.py` (rasterization/rendering).
- **Notable implementation details:** the repo supports an optional 2D Gaussian Splatting mode (`--use_2dgs` flag) not emphasized as a core contribution in the paper; the codebase is dataset-agnostic across NVIDIA, iPhone, and custom captures, and includes a MegaSAM-based preprocessing path for camera pose/depth estimation on custom (non-benchmark) video, i.e., the released pipeline bundles its own pose/depth front end rather than assuming those are always given.

#### Failure Modes & Limitations
- Requires per-scene test-time optimization (no feed-forward inference), which the authors note "hinder[s] streamable applications."
- Depends on off-the-shelf predictions for camera poses, geometry (depth), and 2D motion (tracks); quality degrades when these priors degrade, specifically in textureless regions or under large/fast motion.
- Requires a user-provided or externally computed mask to separate moving objects from static background — it does not discover dynamic regions on its own, though the authors note recent segmentation advances could automate this step.

## Relevance to ADAGS

Makes ADAGS's inactive track-flow hook a central gap.

## Connections

### Deep Dive

**Paper:** https://arxiv.org/abs/2407.13764
**Code:** https://github.com/vye16/shape-of-motion/
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023) rasterization/optimization, combined with off-the-shelf monocular depth, 2D long-range track, and camera-pose priors fused into a single dynamic scene representation.

#### Core Novelty
Instead of giving every Gaussian its own free-form trajectory (which is underconstrained from monocular video) or relying on a per-point deformation MLP, the paper represents the motion of the *entire dynamic scene* as a weighted linear combination of a small number (B=10) of shared SE(3) basis trajectories. This exploits the low-rank structure of real-world motion (rigid/near-rigid parts moving together) to regularize an otherwise ill-posed problem, and lets every dynamic Gaussian's persistent identity (its canonical attributes) be transported through time by a compact, globally shared motion field rather than per-point free parameters. The basis trajectories and per-Gaussian blend weights are additionally initialized from noisy, uncertainty-weighted 3D tracks via clustering and weighted Procrustes alignment, rather than being learned from a random start, which is the second key mechanism driving stability.

#### Mathematical Formulation
- Canonical Gaussian parameters (evaluated once, persistent object state): $\bar g_0 \equiv (\mu_0, R_0, s, o, c)$ — position $\mu_0 \in \mathbb{R}^3$, orientation $R_0 \in SO(3)$, scale $s \in \mathbb{R}^3$, opacity $o \in \mathbb{R}$, color $c \in \mathbb{R}^3$.
- Per-Gaussian rigid transform to time $t$, computed per-Gaussian before rasterization at each frame, as a weighted combination of $B$ shared learned basis transforms:
$$T_{0\to t} = \sum_{b=1}^{B} w^{(b)} T_{0\to t}^{(b)}, \qquad \sum_b \lVert w^{(b)} \rVert = 1$$
  Each basis $T_{0\to t}^{(b)}$ is parameterized as 6D rotation + translation; the same per-Gaussian weight $w^{(b)}$ blends rotation and translation components separately (linear blend skinning-style composition, but over rigid bases rather than a mesh skeleton).
- Resulting per-Gaussian pose at time $t$ (per-Gaussian, before rasterization):
$$\mu_t = R_{0\to t}\mu_0 + t_{0\to t}, \qquad R_t = R_{0\to t}R_0$$
- Reconstruction loss (after rendering, per frame): $\mathcal{L}_{recon} = \lVert \hat I - I\rVert_1 + \lambda_{depth}\lVert \hat D - D\rVert_1 + \lambda_{mask}\lVert \hat M - M\rVert_1$, with $\lambda_{depth}=0.5$, $\lambda_{mask}=1.0$.
- 2D/depth track-supervision losses (after rendering, tying rendered correspondence back to off-the-shelf 2D tracks): $\mathcal{L}_{track\text{-}2d} = \lVert U_{t\to t'} - \hat U_{t\to t'}\rVert_1$ and $\mathcal{L}_{track\text{-}depth} = \lVert \hat d_{t\to t'} - \hat D(U_{t\to t'})\rVert_1$, weights $\lambda_{track\text{-}2d}=2.0$, $\lambda_{track\text{-}depth}=0.1$.
- Rigidity/as-rigid-as-possible regularizer on dynamic Gaussians (sampled pairwise over k-nearest neighbors, as a loss term): $\mathcal{L}_{rigidity} = \lVert \mathrm{dist}(\hat X_t, C_k(\hat X_t)) - \mathrm{dist}(\hat X_{t'}, C_k(\hat X_{t'}))\rVert_2^2$, weight $\lambda_{rigidity}=0.1$, where $C_k(\cdot)$ is the k-nearest-neighbor set.

#### Algorithm / Pipeline Changes
1. Select a canonical reference frame $t_0$ as the frame where the most 3D tracks (lifted from off-the-shelf 2D tracks + monocular depth) are visible.
2. Initialize dynamic Gaussian means $\mu_0$ directly from N randomly sampled noisy 3D track locations at $t_0$ (no random init).
3. Run k-means clustering on the vectorized per-track velocity vectors to propose $B$ candidate rigid motion groups.
4. Initialize each of the $B$ SE(3) basis trajectories via weighted Procrustes alignment between the 3D point set at $t_0$ and at each other frame $t$, restricted to each cluster; weights come from per-track uncertainty and visibility scores (down-weighting occluded/unreliable tracks).
5. Initialize per-Gaussian basis blend weights $w^{(b)}$ to decay exponentially with the Gaussian's distance from its cluster's center.
6. Pre-optimize this initialization for 1000 iterations using an $\ell_1$ fitting loss plus temporal smoothness constraints, before joint scene optimization begins.
7. During main training, at every frame the per-Gaussian transform is recomputed via the basis-blend equation above, Gaussians are rasterized (3DGS/gsplat), and the combined photometric + depth + mask + track + rigidity losses are backpropagated jointly to canonical attributes, basis trajectories, and blend weights. Static background is modeled with a separate, non-moving Gaussian set (100k Gaussians) rendered alongside the 40k dynamic Gaussians.

#### Key Hyperparameters & Design Choices
- Number of SE(3) motion bases $B$: 10.
- Dynamic Gaussians: 40k; static Gaussians: 100k.
- Initialization pre-optimization: 1000 iterations.
- Main training: 500 epochs.
- Learning rates: canonical means $\mu_0$ and basis parameters $1.6\times10^{-4}$; motion coefficients (blend weights) $1\times10^{-2}$; canonical rotation $R_0$ $1\times10^{-3}$; scale $s$ $5\times10^{-3}$.
- Loss weights: $\lambda_{depth}=0.5$, $\lambda_{mask}=1.0$, $\lambda_{track\text{-}2d}=2.0$, $\lambda_{track\text{-}depth}=0.1$, $\lambda_{rigidity}=0.1$; depth-gradient loss weight 1.0; motion smoothness weight 0.1.

#### What Actually Drove the Gains (Ablation Summary)
iPhone dataset, full method: EPE 0.082, $\delta_{3D}^{.05}$ 43.0, $\delta_{3D}^{.10}$ 73.3.
- Removing 2D track supervision entirely is the single most damaging ablation: EPE 0.082 → 0.141 (+72% error), $\delta_{3D}^{.05}$ 43.0 → 30.4. **Most impactful component: long-range 2D track supervision.**
- Removing SE(3) motion-basis initialization (i.e., training bases/weights from scratch instead of track-derived init): EPE 0.082 → 0.111, $\delta_{3D}^{.05}$ 43.0 → 39.3.
- Replacing shared SE(3) bases with translational-only bases: EPE 0.082 → 0.093, $\delta_{3D}^{.05}$ 43.0 → 42.3 (rotation matters, but less than track supervision/init).
- Per-Gaussian SE(3) (no basis sharing/low-rank structure) is close to the full method on error (EPE 0.083) but scores slightly higher on some $\delta_{3D}$ thresholds while being far less constrained/generalizable — the basis representation is motivated by robustness/compactness, not by raw EPE alone.
- Per-Gaussian translation only (weakest ablation of the motion representation family): EPE 0.087, $\delta_{3D}^{.05}$ 41.2.

#### Implementation Reality
- **Framework:** PyTorch, built on top of the `gsplat` CUDA rasterization backend (`nerfstudio-project/gsplat`), with CuML used for GPU-accelerated clustering/init utilities.
- **Key files:** `flow3d/scene_model.py` (canonical Gaussian + basis-blended scene representation), `flow3d/trajectories.py` (SE(3) motion basis trajectories), `flow3d/transforms.py` (SE(3) transform operations), `flow3d/params.py` (learnable parameter containers), `flow3d/init_utils.py` (track-based clustering/Procrustes initialization described above), `flow3d/loss_utils.py` (reconstruction/track/rigidity losses), `flow3d/trainer.py` (main optimization loop), `flow3d/renderer.py` (rasterization/rendering).
- **Notable implementation details:** the repo supports an optional 2D Gaussian Splatting mode (`--use_2dgs` flag) not emphasized as a core contribution in the paper; the codebase is dataset-agnostic across NVIDIA, iPhone, and custom captures, and includes a MegaSAM-based preprocessing path for camera pose/depth estimation on custom (non-benchmark) video, i.e., the released pipeline bundles its own pose/depth front end rather than assuming those are always given.

#### Failure Modes & Limitations
- Requires per-scene test-time optimization (no feed-forward inference), which the authors note "hinder[s] streamable applications."
- Depends on off-the-shelf predictions for camera poses, geometry (depth), and 2D motion (tracks); quality degrades when these priors degrade, specifically in textureless regions or under large/fast motion.
- Requires a user-provided or externally computed mask to separate moving objects from static background — it does not discover dynamic regions on its own.

## Sources

- https://arxiv.org/abs/2407.13764
- https://github.com/vye16/shape-of-motion/
