---
type: paper
node_id: paper:feng2025_disentangled4dgs
title: "Disentangled 4D Gaussian Splatting: Rendering High-Resolution Dynamic World at 343 FPS"
authors: ["Hao Feng", "Hao Sun", "Wei Xie", "Zhi Zuo", "Zhengzhe Liu"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2503.22159"
tags: [dynamic-gs, disentanglement, high-resolution, realtime]
status: deep-dived
---

# Disentangled 4D Gaussian Splatting: Rendering High-Resolution Dynamic World at 343 FPS

**Paper:** https://arxiv.org/abs/2503.22159
**Code:** Not found (no repository linked from arXiv, paper text, or Papers With Code as of extraction)
**Base method:** 4D Gaussian Splatting via 4D-to-3D conditional slicing (Rotor4DGS / 4D-Rotor Gaussians, Duan et al. 2024), itself built on 3D Gaussian Splatting (Kerbl et al. 2023)

## One-line thesis

Re-parameterizing the 4D Gaussian's temporal-spatial coupling as an explicit per-Gaussian linear velocity plus a reduced-dimension 4D covariance (instead of slicing a full 4x4 spatiotemporal covariance matrix at render time) removes the costliest matrix algebra from the render loop, cutting per-Gaussian storage from 16 to 15 floats and raising throughput to 343 FPS without an accuracy trade-off.

## Problem / Gap

Prior 4D GS methods (4D-Rotor/Rotor4DGS) represent each Gaussian with a full 4D covariance matrix and derive the effective 3D Gaussian at a query timestamp `t` by conditioning (slicing) that matrix, which requires a matrix inverse and multiple matrix products per Gaussian, per frame. This "slicing-first" computation must be redone whenever the timestamp changes, and it also introduces motion inaccuracy near object boundaries because the coupling between temporal and spatial covariance blocks is entangled rather than expressed as a direct velocity. The result is a real accuracy/throughput ceiling: RealTime4DGS and Rotor4DGS are already fast (72-277 FPS) but leave both rendering speed and boundary motion fidelity on the table.

## Method

The paper decomposes the 4D covariance's off-diagonal spatiotemporal coupling term into an explicit 3D mean-velocity vector `V3D`, so each Gaussian carries a 3D mean `mu_3D`, a 3D covariance `Sigma_3D`, a rotor/quaternion, a temporal center `mu_t`, a temporal scale `s_t`, and `V3D`, instead of the full 4x4 slicing algebra. At render time the Gaussian's position at query time `t` is obtained by simple linear extrapolation along `V3D` rather than by conditioning a 4D covariance, and the projected 2D covariance is obtained via a Jacobian computed directly from the projected velocity. Two auxiliary training-time mechanisms are added on top of this representation: a flow-gradient-guided consistency loss that suppresses optical-flow artifacts in regions with no corresponding image edge, and a temporal-splitting-aware densification rule that splits a Gaussian's spatial and temporal scales independently based on the gradient of `t` vs. the gradients of `x,y,z`.

## Assumptions

Assumes multi-view (or dense-enough monocular, per D-NeRF) captures with known camera poses, per-frame optical flow available for the flow-gradient loss, and that scene motion is well-approximated locally by linear (constant-velocity) per-Gaussian trajectories between the temporal center and the query timestamp.

## Limitations / Failure Modes

The paper itself states the representation "still has further compression potential" despite only a >=4.5% storage reduction over Rotor4DGS. It reports degraded rendering quality on "extremely sparse inputs such as monocular synthesized reconstruction data" (i.e., D-NeRF-style single-camera synthetic sequences), and states that "faithfully capturing dynamic details remains challenging" under sparse-view conditions — consistent with the linear-velocity assumption breaking down when there are too few observations to constrain non-linear or fast motion.

## Reusable Ingredients

- **Explicit per-Gaussian mean-velocity parameter (`V3D`)** in place of full 4D covariance slicing — replaces a matrix-inverse-heavy conditioning step with linear extrapolation, cutting compute and storage.
- **Velocity-aware projection Jacobian** — propagates the 3D velocity through the camera projection to get the correct 2D covariance at the query timestamp without re-deriving a 4D-to-2D pipeline per frame.
- **Flow-gradient-guided consistency loss** — penalizes optical-flow gradients that have no corresponding image-intensity edge, a generic way to suppress flow-driven motion artifacts in textureless/boundary regions.
- **Temporal-vs-spatial decoupled densification** — splits Gaussians independently along temporal scale gradient vs. spatial scale gradient, avoiding conflating "needs more spatial detail" with "needs more temporal detail" in a single split criterion.

---

### Deep Dive

#### Core Novelty

The paper's change relative to Rotor4DGS/4D-Rotor Gaussians is to algebraically re-derive the same underlying 4D Gaussian model so that the temporal-spatial coupling is expressed as an explicit velocity vector `V3D` and a reduced 3D covariance `Sigma_3D`, rather than as an opaque block of a 4x4 covariance matrix that must be sliced (conditioned) at every query time. The key insight is that the slicing operation used by prior work is mathematically equivalent to linear motion of the 3D Gaussian's mean over time plus a constant 3D covariance — so if you parameterize the Gaussian this way from the start, you never need to perform the conditioning-matrix inverse and multiplication at render time, and you can propagate `V3D` through the perspective projection directly to get the time-dependent screen-space covariance via a Jacobian.

#### Mathematical Formulation

Prior-work (slicing) 4D Gaussian, evaluated per-Gaussian, per query time `t`:
$$G_{3D}(x,t) = \exp\left(-\tfrac{1}{2}\lambda (x-\mu_t)^2\right) \cdot \exp\left(-\tfrac{1}{2}[t-\mu(t)]^\top \Sigma_{3D}^{-1} [x-\mu(t)]\right)$$
where $\lambda = W^{-1}$, $\Sigma_{3D} = A^{-1} = U - VV^\top/W$, and $\mu_{3D} = (\mu_x,\mu_y,\mu_z)^\top + (t-\mu_t)V/W$. Here $U$, $V$, $W$ are the blocks of the original 4D precision/covariance matrix (spatial-spatial, spatial-temporal, temporal-temporal). This is the baseline being replaced.

Disentangled reparameterization (evaluated once, not per query time, since it defines the stored Gaussian parameters):
- Velocity of the mean: $V_{3D} = V/W$
- 3D covariance: $\Sigma_{3D} = U - VV^\top/W$
- Temporal scale: $s_t = \sqrt{W}$

This shows $V_{3D}$ and $\Sigma_{3D}$ are just renamed/precomputed blocks of the same underlying 4D quantities — the contribution is using them as the stored per-Gaussian state instead of $U,V,W$ directly, so that the position at time $t$ becomes simple linear extrapolation $\mu_{3D} + (t-\mu_t) V_{3D}$ rather than a matrix-conditioning step.

Projection pipeline (evaluated per-Gaussian, per-frame, before rasterization):
1. Camera-space mean: $P_{3D} = \phi(\mu_{3D}) = W_{view}\mu_{3D} + d_{view}$, $P_t = \mu_t$.
2. Camera-space velocity: $V_{view} = \phi'(V_{3D}) = W_{view} V_{3D}$ (linear map, no translation term, since velocity is a direction not a position).
3. Ray-space projection: $(x_0,y_0,z_0)^\top = (P_0/P_2,\, P_1/P_2,\, \|(P_0,P_1,P_2)^\top\|)$ and analogously for velocity: $(v_0,v_1,v_2)^\top = (V_{view,x}/P_2,\, V_{view,y}/P_2,\, \|(V_{view,x},V_{view,y},V_{view,z})^\top\|)$.
4. Jacobian at query time $t_0$, evaluated using $P_{k,t} = P_{3D,k} + dt \cdot V_{view,k}$ with $dt = t_0 - t$:
$$J_k = \begin{bmatrix} 1/P_{k,t_2} & 0 & -P_{k,t_0}/P_{k,t_2}^2 \\ 0 & 1/P_{k,t_2} & -P_{k,t_1}/P_{k,t_2}^2 \\ P_{k,t_0}/l' & P_{k,t_1}/l' & P_{k,t_2}/l' \end{bmatrix}, \quad l' = \|(P_{k,t_0}, P_{k,t_1}, P_{k,t_2})^\top\|$$
5. Screen-space covariance: $\Sigma' = J \, W_{view} \, \Sigma \, W_{view}^\top J^\top$ — the standard EWA splatting covariance transform, but with $J$ now depending on the per-Gaussian velocity-shifted position at the query timestamp instead of a static mean.

Flow-gradient-guided consistency loss (loss term, applied after rendering, using rendered optical flow):
$$L_{fg} = \lambda_{flow} \cdot \frac{1}{N}\sum_{x,y} \|\nabla M(x,y)\| \cdot \left(1 - \|\nabla I(x,y)\|\right)$$
where $M = \sqrt{u^2+v^2+\epsilon}$ is the optical-flow magnitude, $\nabla M$ is the normalized flow-magnitude gradient, and $\nabla I$ is the normalized image-intensity gradient (Sobel). The term is large exactly where flow varies sharply but the image does not — i.e., spurious flow discontinuities not backed by any visible edge — and is penalized.

#### Algorithm / Pipeline Changes

1. Replace the stored per-Gaussian 4D state (previously the blocks needed to reconstruct a 4x4 covariance) with: 3D mean $\mu_{3D}$, rotor/quaternion $q$, 3D scale, temporal center $\mu_t$, temporal scale $s_t$, and 3D velocity $V_{3D}$ — 15 floats instead of 16.
2. At render time, for each query timestamp, skip the 4D covariance conditioning step entirely; compute the time-shifted mean via linear extrapolation $\mu_{3D} + (t-\mu_t)V_{3D}$.
3. Project the mean and the velocity separately through the camera view transform, then through the perspective (ray-space) projection, producing time-dependent projected position and projected velocity.
4. Build the projection Jacobian using the velocity-shifted camera-space position (step above) rather than a static mean, then apply the standard $\Sigma' = J W \Sigma W^\top J^\top$ covariance transform.
5. During training, add the flow-gradient consistency loss term $L_{fg}$ computed from rendered optical flow and rendered RGB image gradients, alongside the standard photometric losses.
6. During adaptive density control, split a Gaussian's spatial scales and temporal scale independently: use the gradient magnitude of the temporal coordinate $t$ to decide whether to split along the temporal axis, and the gradient magnitude of $x,y,z$ to decide whether to split spatially, rather than a single combined split criterion.

#### Key Hyperparameters & Design Choices

- Optimizer: Adam; training steps: 20,000 per scene.
- Densification gradient threshold: 5e-5 on D-NeRF, 2e-5 on the Plenoptic Video dataset.
- Rotor/quaternion initialization: $(1,0,0,0)$ (identity rotation).
- LPIPS backbone: AlexNet for Plenoptic Video / Google Immersive, VGGNet for D-NeRF (metric-reporting choice, not a model hyperparameter).
- $\lambda_{flow}$ (flow-gradient loss weight): not specified in paper (extracted text gives the loss form but not the numeric weight).
- MLP/network dimensions: not applicable — the method is explicitly designed to avoid per-Gaussian network queries; no deformation MLP hidden-dim values are reported for the disentangled path.
- Other training settings follow the original 3DGS defaults (not itemized beyond the above in the extracted text).

#### Ablation Summary

D-NeRF dataset (PSNR / SSIM):
- Base (no edge/flow-gradient loss, no temporal split): 33.20 / 0.95
- + flow-gradient loss only: 33.47 / 0.97
- + temporal split only: 33.40 / 0.97
- Full (both): 33.61 / 0.98

Plenoptic Video dataset (PSNR / DSSIM):
- Base: 32.44 / 0.013
- + flow-gradient loss only: 32.58 / 0.012
- + temporal split only: 32.56 / 0.013
- Full (both): 32.75 / 0.011

Both components contribute comparably (roughly +0.15-0.27 dB PSNR each in isolation on D-NeRF, +0.12-0.14 dB on Plenoptic), and combine near-additively; the flow-gradient loss has a marginally larger single-component effect on both datasets. Neither ablation isolates the core disentangled-velocity reparameterization itself (that is treated as the base architecture in all four rows), so the ablation only quantifies the two auxiliary training-time additions, not the representational contribution.

#### Failure Modes & Limitations

The paper states the representation "still has further compression potential" despite a >=4.5% size reduction over Rotor4DGS, implying the 15-float-per-Gaussian encoding is not considered close to an information-theoretic floor. It reports explicit quality degradation "when handling extremely sparse inputs such as monocular synthesized reconstruction data" (D-NeRF-style single-camera synthetic sequences), and separately notes "faithfully capturing dynamic details remains challenging" in sparse-view scenarios — both point at the linear per-Gaussian velocity assumption being under-constrained when view coverage is low.

## Relevance to ADAGS

Pressures ADAGS to report efficiency under matched point budgets and to avoid broad speed claims unless measured against high-throughput 4DGS variants.

## Connections

- Addresses [[gap_map#G5 - Capacity Allocation Must Be Matched And Dynamic-Aware]]
- Addresses [[gap_map#G6 - Single Global Motion Models Are A Known Weakness]]

## Sources

- https://arxiv.org/abs/2503.22159
