---
type: paper
node_id: paper:zhang2026_se3_bspline_gs
title: "Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos"
authors: ["Xuankai Zhang", "Junjin Xiao", "Shangwei Huang", "Wei-shi Zheng", "Qing Zhang"]
year: 2026
venue: "CVPR 2026"
external_ids:
  arxiv: "2603.25058"
tags: [dynamic-gs, motion-representation, b-spline, monocular]
status: deep-dived
---

# Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos (SE(3) B-spline GS)

**Paper:** https://arxiv.org/abs/2603.25058
**Code:** https://github.com/hhhddddddd/se3bsplinegs
**Base method:** MoSca (4D Motion Scaffolds, Lei et al.) for tracklet-based dynamic-region masking and the underlying track/scaffold formulation, layered on 3D Gaussian Splatting; static Gaussians follow standard 3DGS, dynamic Gaussians are additionally deformed via dual-quaternion blending (DQB) of nearby SE(3) motion bases.

## One-line thesis
Representing each motion basis as a continuous SE(3) cumulative B-spline over a compact, adaptively inserted/pruned set of control points gives dynamic Gaussians smooth, topology-consistent position *and* orientation trajectories from monocular video — fixing the orientation discontinuities that prior spline methods (position-only cubic Hermite splines) and implicit deformation fields leave unresolved.

## Problem / Gap
Prior monocular dynamic-GS methods either warp a canonical Gaussian set through an implicit deformation MLP (which smooths over genuine discontinuities and is hard to interpret/control), or use splines restricted to 3D position only (e.g., cubic Hermite splines), leaving orientation to be interpolated less carefully. The paper identifies "non-continuous Gaussian orientation deformation" as the concrete artifact this causes in regions of complex motion — rotational jumps between frames that position-only spline or implicit-warp methods do not explicitly constrain to be smooth.

## Method
Each dynamic Gaussian is bound to its K-nearest SE(3) "motion bases," each of which is an explicit cumulative B-spline curve in SE(3) built from a compact set of learnable control points (initialized at N_c = 100, then adaptively pruned/inserted during training). At any query time t, the pose T(t) is a cumulative product of matrix exponentials of per-control-point Lie-algebra twists, weighted by B-spline basis functions, applied to an initial pose T_0 from tracklet initialization. Each Gaussian's per-frame rigid transform is obtained by dual-quaternion blending (DQB) of its K nearest motion-basis transforms rather than a canonical-to-observation MLP warp. Training jointly supervises photometric reconstruction, depth, optical-flow tracking, ARAP motion-smoothness, camera-pose smoothness, and a multi-view diffusion (SDS) prior that regularizes novel-view appearance of the (monocular-only) foreground.

## Assumptions
Monocular RGB(-D) video with a single moving camera (camera poses are jointly optimized, with an explicit smoothness loss over consecutive extrinsics); scenes are decomposed into static background (reprojected 3DGS) and dynamic foreground bound to motion bases; relies on off-the-shelf 3D tracklets/depth for initialization and optical-flow supervision, and on MoSca's dynamic-region mask for locating where to densify control points.

## Limitations / Failure Modes
The paper states the method still fails on "dynamic scenes with substantial non-rigid deformation and motion blur," and on "blurry monocular videos with rapid camera or object motion" — i.e., the B-spline motion bases and their DQB blending are not enough when true motion is highly non-rigid or when input frames themselves are degraded by blur. No per-scene quantitative failure breakdown is reported; degradation is described qualitatively (Fig. 7) rather than isolated to specific benchmark scenes.

## Reusable Ingredients
- **SE(3) cumulative B-spline pose curve** (Lovegrove et al. formulation, adapted per-motion-basis): gives closed-form, differentiable, continuous rigid trajectories in both translation and rotation from a sparse control-point set.
- **Reprojection-triggered control-point densification**: uses a rendering-error mask intersected with a dynamic-region mask, then a "≥50% of projected control-point positions fall inside the mask" rule to decide where trajectory capacity is insufficient — a concrete, checkable criterion for local motion-model under-capacity.
- **Trajectory-fit-error pruning**: removes the single control point whose omission increases cumulative trajectory reconstruction error the least, only if that error stays under a fixed threshold — a principled one-point-at-a-time simplification rule instead of heuristic decimation.
- **Soft segment reconstruction (opacity down-weighting by temporal distance)**: `o' = sigmoid(scale·(1-|t_ref - t_obs|))·o` reduces the rendering contribution of a dynamic Gaussian the further its reference frame is from the observation frame, without changing its geometry — a cheap, differentiable way to hedge against motion-model error accumulating over long intervals.
- **DQB fusion of K nearest motion bases** for per-Gaussian rigid transform: an alternative to canonical-space MLP warping that keeps motion explicit and locally interpretable.
- **Foreground-only SDS from a monocular-conditioned diffusion prior** (Zero123-xl) with small camera-center perturbations around the training view: view-augmentation regularization usable in any monocular reconstruction pipeline suffering novel-view overfitting.

---

### Deep Dive

#### Core Novelty
Relative to its MoSca-style tracklet/motion-scaffold base and to prior position-only spline motion models, this paper's specific change is (1) replacing per-basis position-only interpolation with a full SE(3) cumulative B-spline (position *and* orientation jointly, via Lie-algebra twist composition) and (2) making the control-point set itself adaptive (error-triggered insertion, fit-error-triggered pruning) rather than fixed. The key insight is that orientation discontinuities — not just position error — are a distinct, previously under-addressed source of artifacts in complex-motion regions, and that a compact adaptive control-point budget lets the same representation stay both smooth (few control points, globally regularized) and locally expressive (extra control points only where reprojection error concentrates).

#### Mathematical Formulation
- **Twist extraction (Eqs. 1-2)**: For adjacent tracklet poses $Q_i, Q_{i+1} \in SE(3)$, the relative transform is $\Delta Q = Q_i^{-1} Q_{i+1}$ and the per-control-point Lie-algebra twist is $\xi = \log(\Delta Q)$, i.e., control points store the log-map "velocity" between adjacent 3D tracklet poses. Evaluated once per control point during motion-basis construction/initialization.
- **SE(3) cumulative B-spline pose curve (Eq. 3)**:
$$T(t) = \left(\prod_{i=0}^{N_c-1} \exp(\Omega_i(t)\,\xi_i)\right) T_0$$
  computes the pose of a motion basis at continuous time $t$ as a cumulative product of matrix exponentials of scaled twists $\xi_i$, each scaled by a B-spline blending weight $\Omega_i(t)$, applied to the initial pose $T_0$ (the tracklet pose at the first frame). $N_c$ is the (adaptive) control-point count. Evaluated per motion basis at every training/render timestep, before dynamic-Gaussian deformation. The paper does not give the closed-form expression for $\Omega_i(t)$ itself, only citing the cumulative-B-spline construction (Lovegrove et al. 2013); spline order/degree is likewise not explicitly stated in the paper (standard cumulative-B-spline usage implies cubic, i.e., ~4 local control points affect any given $t$, but this is inferred, not stated).
- **Pruning objective (Eqs. 4-5)**: candidate removal selects
$$\hat Q = \arg\min_{Q} \sum_{t=0}^{N_T} \lVert T(t)^{Q} - T(t) \rVert_2^2, \qquad E = \sum_{t=0}^{N_T} \lVert T(t)^{\hat Q} - T(t) \rVert_2^2$$
  i.e., the control point whose removal changes the reconstructed trajectory $T(t)$ least, measured by summed squared pose error over all $N_T$ frames; removal is applied only if $E < \epsilon_{prune}$. Evaluated every $N_{prune}$ iterations, per motion basis.
- **Rendering-error mask (Eq. 6)**: $m_{error}^i = \mathbb{1}[\lvert \hat I^i - I^i \rvert > \epsilon_{error}]$, a per-pixel absolute photometric error mask at view $i$.
- **Complex-motion region mask (Eq. 7)**: $m^i = m_{error}^i \cap m_d^i$, intersecting the rendering-error mask with MoSca's dynamic-region mask $m_d^i$ to localize densification candidates to genuinely dynamic, poorly-reconstructed pixels.
- **Motion-basis projection (Eq. 8)**: $p = K\,P(t)\,T(t)^j$ projects motion-basis $j$'s 3D pose at time $t$ into image space using intrinsics $K$ and extrinsics $P(t)$; used to test whether a motion basis lies inside the complex-motion mask across the sequence (≥50% of projected positions inside mask $M$ triggers insertion for that basis).
- **Dual-quaternion blending of per-Gaussian transform (Eqs. 9-10)**:
$$\Delta Q^g = \mathrm{DQB}\left(\{(w_i, \Delta Q^i)\}_{i=1}^{K}\right), \qquad \mu_g' = \Delta Q^g \mu_g,\ \ R_g' = \Delta Q^g R_g$$
  fuses the K nearest motion-basis transforms $\Delta Q^i$ (with weights $w_i$) via dual-quaternion blending into a single rigid transform $\Delta Q^g$ applied to Gaussian $g$'s mean $\mu_g$ and rotation $R_g$. Evaluated per dynamic Gaussian, per frame, before rasterization.
- **Soft segment reconstruction opacity re-weighting (Eq. 11)**:
$$o' = \mathrm{sigmoid}\big(\mathrm{scale}\cdot(1 - \lvert t_{ref} - t_{obs}\rvert)\big) \cdot o$$
  rescales a dynamic Gaussian's rendering opacity $o$ by a sigmoid of $(1 - $ temporal distance between its reference time $t_{ref}$ and the current observation time $t_{obs})$, scaled by `scale`. Applied per dynamic Gaussian, per frame, immediately before rasterization; it modulates blending weight only — geometry (position/orientation) is unaffected.
- **SDS loss (Eq. 12)**: $\mathcal{L}_{sds} = \mathbb{E}_{t,\epsilon}\left[\lVert \omega(t)\big(\epsilon_\phi(z_t, t, P_tP_s^{-1}, I_s) - \epsilon\big)\rVert_2^2\right]$, a standard score-distillation loss conditioning the Zero123-xl diffusion model on the source-view image $I_s$ and relative camera pose $P_t P_s^{-1}$; applied to renders from perturbed nearby camera poses, foreground region only.
- **Reconstruction loss (Eq. 13)**: $\mathcal{L}_{rec} = (1-\beta)\mathcal{L}_1(\hat I, I) + \beta\,\mathcal{L}_{ssim}(\hat I, I)$, $\beta = 0.2$ (standard form, included for completeness of the total-loss weighting).
- **Camera smoothness (Eq. 15)**: $\sum_{t=0}^{N_T-1} \lVert P_t^{-1}P_{t+1} \rVert_2^2$, penalizing large frame-to-frame camera-extrinsic changes.
- **Total loss (Eq. 16)**: $\mathcal{L} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{geo}\mathcal{L}_{geo} + \lambda_{sds}\mathcal{L}_{sds} + \lambda_{arap}\mathcal{L}_{arap} + \lambda_{track}\mathcal{L}_{track} + \lambda_{smo}\mathcal{L}_{smo}$ (weights below).

#### Algorithm / Pipeline Changes
1. Initialize 3D tracklets and per-tracklet poses $Q_i$ from monocular depth/tracking (MoSca-style scaffold init); compute inter-frame twists $\xi = \log(Q_i^{-1}Q_{i+1})$ to seed each motion basis with $N_c = 100$ control points.
2. At every training step, evaluate each motion basis's continuous pose $T(t)$ via the cumulative SE(3) B-spline (Eq. 3) at the current sampled/rendered timestep.
3. For each dynamic Gaussian, find its K nearest motion bases, fuse their transforms via dual-quaternion blending (Eqs. 9-10), and apply the fused rigid transform to the Gaussian's mean and rotation.
4. Apply soft-segment opacity re-weighting (Eq. 11) to each dynamic Gaussian based on the gap between its reference time and the current observation time, immediately before rasterization — replaces the raw learned opacity for rendering purposes only.
5. Rasterize static (background, standard 3DGS) and dynamic (deformed, opacity-reweighted) Gaussians together.
6. Every $N_{densify} = 500$ iterations: compute the rendering-error mask (Eq. 6), intersect with MoSca's dynamic-region mask (Eq. 7), project each motion basis into each training view (Eq. 8), and if ≥50% of a basis's projected positions fall inside the resulting mask sequence, insert a new control point by copying that basis's existing control-point parameters into $T_0$ and adding a random perturbation.
7. Every $N_{prune} = 500$ iterations: for each motion basis, find the control point whose removal minimizes trajectory reconstruction error (Eqs. 4-5) and remove it if that error is below $\epsilon_{prune} = 5.0$.
8. Periodically render from a training view perturbed by a small random camera-center offset and apply the foreground-only SDS loss (Eq. 12) against Zero123-xl conditioned on the corresponding same-time source-view image.
9. Backpropagate the combined loss (Eq. 16) through motion-basis control points, per-Gaussian attributes, and camera extrinsics jointly via Adam.

Batched inputs/outputs and exact tensor shapes are not stated in the paper beyond what's captured above.

#### Key Hyperparameters & Design Choices
- Initial control points per motion basis: $N_c = 100$.
- Densification check interval: $N_{densify} = 500$ iterations.
- Pruning check interval: $N_{prune} = 500$ iterations.
- Densification trigger: ≥50% of a motion basis's projected positions fall inside the complex-motion mask.
- Rendering-error mask threshold: $\epsilon_{error} = 0.5$.
- Pruning error threshold: $\epsilon_{prune} = 5.0$.
- Soft-segment opacity sigmoid scale: $\mathrm{scale} = 5.0$.
- Loss weights: $\lambda_{rec} = 1.0$, $\lambda_{geo} = 0.075$, $\lambda_{sds} = 0.01$, $\lambda_{arap} = 1.0$, $\lambda_{track} = 1.0$, $\lambda_{smo} = 0.01$; reconstruction-loss blend $\beta = 0.2$ (L1/SSIM).
- Learning rates: $1.6\times10^{-4}$ for SE(3) motion-basis parameters, $3\times10^{-4}$ for camera extrinsics.
- Optimizer: Adam.
- Training length: 8,000 iterations; ~30 minutes on a single RTX 4090.
- Diffusion prior: Zero123-xl-diffusers (built on Stable Diffusion v1.5).
- Rendering resolution for FPS measurement: 480×360 (45.124 FPS reported).
- Spline degree/order and $\Omega_i(t)$'s explicit functional form: not specified in paper.
- Reference-time $t_{ref}$ selection rule beyond "timestamp of the depth map that initializes the dynamic Gaussian": not further specified in paper.

#### Ablation Summary
On the iPhone dataset (mPSNR, full-method baseline presumed ≈20.17 dB per main results table):
- **Adaptive control (insertion/pruning) removed**: mPSNR 18.84 (**-1.33 dB** vs. full method) — the single largest ablation drop, i.e., the adaptive control-point mechanism is the most impactful component.
- **Soft segment reconstruction removed**: mPSNR 19.02 (-1.15 dB).
- **Camera smoothness loss removed**: mPSNR 19.18 (-0.99 dB).
- **SDS loss removed**: mPSNR 19.39 (-0.78 dB).

Ranked by impact: adaptive control-point insertion/pruning > soft segment reconstruction > camera smoothness > SDS diffusion prior.

#### Implementation Reality
- **Framework:** PyTorch 2.5.0, CUDA 12.4, Python 3.10; repository also lists JAX in requirements.
- **Key files:** `mosca_reconstruct.py` (main training loop), `mosca_precompute.py` / `mosca_evaluate.py` (preprocessing and evaluation), `lite_moca_reconstruct.py` (lightweight variant), `mosca_viz.py` (visualization); core logic under `lib_moca`, `lib_mosca`, `lib_prior`, `lib_render`. File naming (`mosca_*`) confirms the implementation is built directly on top of a MoSca (motion-scaffold) codebase rather than a from-scratch 3DGS fork.
- **Notable implementation details:** README acknowledges MoSca, SoM, SplineGS, HiMoR, MarbleGS, and DreamScene4D as prior-work dependencies/baselines. No MLP-depth or densification-schedule discrepancies between paper and code were surfaced in the available repository content; none of the paper's stated numeric choices (N_c=100, thresholds, weights) were contradicted by the repo content inspected.

#### Failure Modes & Limitations
The paper explicitly states the method struggles on "dynamic scenes with substantial non-rigid deformation and motion blur," and separately that it "fails to handle blurry monocular videos with rapid camera or object motion" (qualitative failure shown in Fig. 7, large non-rigid motion). No scene-specific quantitative degradation (e.g., a named benchmark scene with a reported dB drop) is given — failures are reported qualitatively rather than isolated per-scene.

---

## Relevance to ADAGS

Occupies the explicit per-primitive continuous-motion axis (CVPR 2026). Any
motion component in a new ADAGS representation must be positioned against
it; it does not touch presence, identity-across-gaps, or events.

## Connections

- Pressures [[gap_map#G6 - Single Global Motion Models Are A Known Weakness]]

## Sources

- https://arxiv.org/abs/2603.25058
