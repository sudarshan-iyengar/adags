---
type: paper
node_id: paper:luiten2023_dynamic_3d_gaussians
title: "Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis"
authors: ["Jonathon Luiten", "Georgios Kopanas", "Bastian Leibe", "Deva Ramanan"]
year: 2023
venue: "3DV 2024"
external_ids:
  arxiv: "2308.09713"
  doi: null
  s2: null
tags: ["dynamic-gaussians", "persistence", "tracking", "identity"]
added: 2026-07-14T22:18:30Z
---

# Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis

**Paper:** https://arxiv.org/abs/2308.09713
**Code:** https://github.com/JonathonLuiten/Dynamic3DGaussians (depends on a forked rasterizer, https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth)
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023), extended to per-timestep Gaussian parameters with no deformation network.

## One-line thesis

Freezing color, opacity, and size per-Gaussian across time and only letting position and rotation change, under a local-rigidity regularizer between neighboring Gaussians, forces each Gaussian to represent a fixed piece of physical surface — so dense 6-DoF tracking falls out of pure rendering supervision with no flow, correspondence, or tracker input.

## Problem / Gap

Prior dynamic NeRF/Gaussian methods (deformation fields, per-frame independent reconstructions) optimize purely for photometric accuracy at each timestep and have no constraint tying a piece of geometry at time $t$ to "the same" piece of geometry at time $t{-}1$. A per-frame-independent 3D Gaussian fit can freely reassign color/opacity/position to explain the image, so nothing prevents identity drift, and no tracking signal is recoverable. The paper's target failure is specifically this: existing pipelines can synthesize novel views of dynamic scenes but cannot tell you where a specific physical point went.

## Method

Gaussians are initialized once from a multi-view depth-camera point cloud at frame 0, and every subsequent frame is initialized by forward-extrapolating the previous two frames' positions/rotations (constant velocity). Per Gaussian, only center position and orientation quaternion are optimized per timestep; color, opacity, and scale are optimized at frame 0 and then held fixed for the rest of the sequence. Training combines the standard rendering loss with three motion regularizers computed over each Gaussian's k=20 spatial nearest neighbors (fixed from frame-0 positions): a local-rigidity loss that penalizes relative-position drift inconsistent with a rigid transform, a rotation-similarity loss that penalizes neighbors rotating differently, and a long-term local-isometry loss that penalizes pairwise-distance drift relative to frame 0. A background-segmentation loss additionally suppresses foreground Gaussians from explaining background pixels. Because attributes persist and neighbors are regularized to move together, the resulting per-frame center/rotation trajectory of each Gaussian is directly usable as a dense 6-DoF track.

## Assumptions

Requires synchronized multi-view video (27 training + 4 held-out test cameras in the paper's PanopticSports dataset) with known calibration; does not work on monocular video. Assumes every tracked physical element is visible in the initial frame (identity is seeded there and never re-seeded) and that local motion is approximately rigid over the k-nearest-neighbor radius, which is reasonable for humans/objects in a lab volume but not for topology change.

## Limitations / Failure Modes

The authors state the method "is only able to track parts of scenes that are visible in the initial frame" and "would completely fail to reconstruct new objects entering the scene" — there is no mechanism for birth, death, or re-appearance after occlusion. It requires a synchronized multi-camera rig and does not work off-the-shelf on monocular video. View-dependent color (spherical harmonics) is disabled for simplicity, so specular/reflective surfaces are not modeled. The ablation shows that removing the background loss is catastrophic (PSNR 28.7 → 24.1) and removing attribute-fixing/forward-propagation causes large tracking-error blowups (3D MTE 1.9cm → 30.7cm without fixing color/opacity/scale), meaning the persistence assumption is load-bearing, not incidental.

## Reusable Ingredients

- **Attribute freezing after frame 0** — fixing color/opacity/scale and optimizing only pose forces temporal identity onto each primitive; ablation shows this is one of the two most critical components (PSNR 27.14 vs 29.48 full, 3D MTE 30.7cm vs 1.9cm without it).
- **k-NN local-rigidity loss** — a lightweight, correspondence-free way to encourage locally-rigid motion using only a fixed spatial neighbor graph computed once at initialization.
- **Forward-velocity initialization per timestep** — extrapolating position/rotation from the previous two frames gives each new frame's optimization a strong prior and is shown to matter a lot (removing it: PSNR 28.48 vs 29.48, 3D MTE 6.3cm vs 1.9cm).
- **Emergent dense tracking from rendering-only supervision** — no flow/tracker network needed; trajectories are simply the optimized per-Gaussian pose sequence.
- **Distance-weighted neighbor loss** ($w_{i,j}=\exp(-\lambda_w\|\mu_{j,0}-\mu_{i,0}\|_2^2)$) — a reusable way to softly down-weight rigidity constraints between spatially distant "neighbors" instead of using a hard cutoff.

---

### Deep Dive

#### Core Novelty
Relative to vanilla 3D Gaussian Splatting (which has no time axis and no persistence constraint), the paper (1) re-optimizes only $(\mu, q)$ per Gaussian per timestep while freezing $(\text{color}, o, s)$ after frame 0, and (2) adds three k-NN-based motion regularizers so that neighboring Gaussians are constrained to move as a locally-rigid group. The insight is that persistence + local rigidity is exactly the inductive bias needed to make "which Gaussian is which" well-defined over time, which is precisely what a photometric loss alone cannot supply — so 6-DoF tracking becomes a side effect of correctly-regularized rendering rather than a separately-trained task.

#### Mathematical Formulation

Covariance at time $t$ for Gaussian $i$ (rotation-only update, scale fixed):
$$\Sigma_{i,t} = R_{i,t} S_i S_i^\top R_{i,t}^\top$$
where $S_i$ (diagonal scale) is optimized once at frame 0 and frozen; $R_{i,t}$ is derived from the per-timestep quaternion $\hat q_{i,t}$.

Neighbor weight (computed once, from frame-0 positions, used in all three losses below):
$$w_{i,j} = \exp\left(-\lambda_w \lVert \mu_{j,0} - \mu_{i,0}\rVert_2^2\right), \quad \lambda_w = 2000$$
($\lambda_w{=}2000$ corresponds to a soft falloff radius of roughly 2.2 cm.) Evaluated per-Gaussian, before the loss sum, over its $k{=}20$ nearest neighbors fixed at initialization.

Local-rigidity loss (per neighbor pair $i,j$, evaluated every training step at every timestep transition $t{-}1 \to t$):
$$\mathcal{L}^{rigid}_{i,j} = w_{i,j}\left\lVert (\mu_{j,t-1}-\mu_{i,t-1}) - R_{i,t-1}R_{i,t}^{-1}(\mu_{j,t}-\mu_{i,t}) \right\rVert^2$$
Penalizes the relative offset between neighbors $i,j$ changing in a way inconsistent with $i$'s own rigid rotation between consecutive frames.

Rotation-similarity loss:
$$\mathcal{L}^{rot} = \frac{1}{k|\mathcal{S}|}\sum_i\sum_j w_{i,j}\left\lVert \hat q_{j,t}\hat q_{j,t-1}^{-1} - \hat q_{i,t}\hat q_{i,t-1}^{-1} \right\rVert^2$$
Penalizes neighboring Gaussians undergoing different frame-to-frame rotation deltas — a direct constraint that rigid groups rotate together, distinct from the position-only rigidity term above.

Long-term local isometry loss:
$$\mathcal{L}^{iso} = \frac{1}{k|\mathcal{S}|}\sum_i\sum_j w_{i,j}\left\lvert \lVert \mu_{j,0}-\mu_{i,0}\rVert^2 - \lVert \mu_{j,t}-\mu_{i,t}\rVert^2 \right\rvert$$
Anchors pairwise distances back to their frame-0 values (not just the previous frame), which is what suppresses long-horizon drift accumulation that a purely frame-to-frame rigidity term would not catch.

All three are summed with the rendering loss and a background-segmentation loss; the paper does not report the individual scalar loss weights used to combine them ("Not specified in paper").

#### Algorithm / Pipeline Changes
1. Build an initial colored point cloud from 10 synchronized depth cameras at $t{=}0$ (subsampled by factor 2); assign each point's color from its nearest training camera. This seeds Gaussian centers, colors, opacities, and scales.
2. Optimize all Gaussian parameters (position, rotation, color, opacity, scale, plus per-camera color scale/offset) for 10,000 iterations (~4 min) against the 27 training-camera images at $t{=}0$ only.
3. Freeze color, opacity, scale, and the per-camera color-correction parameters for the remainder of the sequence — only position and rotation remain trainable from $t{=}1$ onward.
4. Compute the $k{=}20$ nearest-neighbor graph once, from frame-0 positions, and hold it fixed for every subsequent timestep (used by all three regularizers above).
5. For each new timestep $t>0$: initialize $\mu_{i,t}, \hat q_{i,t}$ by constant-velocity extrapolation from $t{-}1$ and $t{-}2$; reset the Adam optimizer's first-order momentum; run 2,000 optimization iterations (~50 s) minimizing rendering loss + $\mathcal{L}^{rigid}+\mathcal{L}^{rot}+\mathcal{L}^{iso}+\mathcal{L}^{Bg}$.
6. No densification, pruning, clone/split, or opacity-reset steps run after $t{=}0$ — the Gaussian count and correspondence between timesteps is fixed for the whole 150-frame sequence, which is what makes "the same Gaussian at every $t$" a well-posed dense track.
7. Spherical harmonics / view-dependence is disabled throughout (flat per-Gaussian RGB only).

#### Key Hyperparameters & Design Choices
- $k = 20$ nearest neighbors, computed once from frame-0 positions.
- $\lambda_w = 2000$ in the neighbor-weighting kernel (≈2.2 cm effective falloff).
- Frame-0 (initialization) optimization: 10,000 iterations (~4 minutes).
- Per-subsequent-frame optimization: 2,000 iterations (~50 seconds); Adam first-order momentum reset at the start of each new frame.
- Total sequence length: 150 frames, ~2 hours total training time per scene.
- 27 training cameras / 4 held-out test cameras (positions 0, 10, 15, 30) in PanopticSports; images at 640×360.
- Individual loss term weights (relative weighting of $\mathcal{L}^{rigid}, \mathcal{L}^{rot}, \mathcal{L}^{iso}, \mathcal{L}^{Bg}$ against the rendering loss) and learning rates: Not specified in paper.

#### Ablation Summary
(Juggle scene; full method: PSNR 29.48, 3D MTE 1.90cm, 2D MTE 1.54px)
- **Removing attribute-fixing (color/opacity/scale allowed to vary per-frame) — largest degradation:** PSNR 27.14 (−2.34), 3D MTE 30.7cm (12.7cm), 2D MTE 19.15px. This is the single most impactful component.
- **Removing $\mathcal{L}^{Bg}$ (background-segmentation loss):** PSNR 24.14 (−5.34), 3D MTE 8.46cm, 2D MTE 6.40px — also severe, and the largest raw PSNR drop of any ablation.
- **Removing forward-velocity propagation (random/no init instead):** PSNR 28.48 (−1.0), 3D MTE 6.32cm, 2D MTE 5.4px.
- **Removing $\mathcal{L}^{rigid}$:** PSNR 28.51 (−0.97), 3D MTE 4.32cm, 2D MTE 3.80px.
- **Removing $\mathcal{L}^{iso}$:** PSNR 29.36 (−0.12), 3D MTE 1.93cm, 2D MTE 1.72px — small metric effect.
- **Removing $\mathcal{L}^{rot}$:** PSNR 29.43 (−0.05), 3D MTE 1.91cm, 2D MTE 1.55px — smallest metric effect, though the paper notes it still improves qualitative rotational coherence.
- 3DGS-per-frame-independent baseline (3GS-O): PSNR 28.19, 3D MTE 32.81cm, 2D MTE 23.86px — comparable rendering quality but tracking is essentially unusable, confirming that photometric loss alone gives no identity signal.

#### Implementation Reality
- **Framework:** PyTorch, plus a custom CUDA rasterizer (a fork of the original 3DGS rasterizer, `diff-gaussian-rasterization-w-depth`, adding depth output).
- **Key files:** the repo README states "almost all of the code is in `train.py`, in a few core functions, with the overall training loop clearly laid out"; `helpers.py` holds general utilities, `external.py` holds functions adapted from other repos, and `visualize.py` provides an Open3D-based trajectory/scene viewer.
- **Notable implementation details differing from the paper:** the released code "soft fixes" color via a temporal-consistency regularization loss rather than hard-freezing it as the paper describes, and adds a floor-plane loss (not in the paper) that penalizes Gaussians moving below the known ground plane. The authors note the public repo is a more up-to-date, evolving version of the method rather than an exact snapshot of the paper.

#### Failure Modes & Limitations
The paper explicitly states the method "is only able to track parts of scenes that are visible in the initial frame" and "would completely fail to reconstruct new objects entering the scene" — there is no re-seeding mechanism for content that appears mid-sequence or was fully occluded at $t{=}0$. It requires a calibrated multi-camera capture rig and "does not work off-the-shelf on monocular video." Spherical harmonics are disabled, so it cannot model view-dependent appearance (specular highlights, reflections).

---

## Open Questions

Which Gaussian properties should persist under occlusion, and which should be allowed to branch or retire on verified reveals?

## Claims

None yet.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

It shows why surface identity matters, but Phase 8 must combine persistence with evidence-driven birth or reassignment rather than impose persistence everywhere.

## Sources

- https://arxiv.org/abs/2308.09713
- https://dynamic3dgaussians.github.io/
- https://github.com/JonathonLuiten/Dynamic3DGaussians
