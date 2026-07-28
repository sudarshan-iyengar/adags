---
type: paper
node_id: paper:zhang2026_continuous_motion
title: "Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos"
authors: ["Xuankai Zhang", "Junjin Xiao", "Shangwei Huang", "Wei-Shi Zheng", "Qing Zhang"]
year: 2026
venue: "CVPR"
external_ids:
  arxiv: "2603.25058"
tags: [dynamic-gs, continuous-motion, se3, motion-basis]
status: deep-dived
---

# Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting

**Paper:** https://arxiv.org/abs/2603.25058
**Code:** https://github.com/hhhddddddd/se3bsplinegs
**Base method:** 3D Gaussian Splatting + a MoSca-style motion scaffold (3D tracklets driving Gaussian deformation), replacing SplineGS's separate-channel Cubic Hermite position/orientation splines with a unified SE(3) Cumulative B-spline; adds a Zero-1-to-3 / DreamScene4D-style multi-view diffusion prior for novel-view regularization.

## One-line thesis

Representing per-tracklet motion as a single SE(3) Cumulative B-spline (rather than separately splining position and rotation) gives mathematically continuous, differentiable rigid-body trajectories that can be adaptively pruned/densified in control-point space, which fixes the jitter and discontinuity artifacts of prior discrete or channel-separated motion representations for monocular dynamic Gaussian Splatting.

## Problem / Gap

Prior monocular dynamic-GS methods (e.g., SplineGS) spline position and rotation as separate channels, which breaks the coupling between translation and rotation and is not truly continuous in SE(3), producing artifacts under interpolation. Track-driven methods (e.g., MoSca) tie Gaussian motion rigidly to a fixed scaffold with no principled mechanism to add/remove motion capacity where scene motion is more or less complex, and monocular-only supervision overfits to the single training camera trajectory, degrading novel-view quality.

## Method

Static background Gaussians are initialized from depth reprojection; dynamic Gaussians are initialized from 3D tracklets (with an added per-Gaussian reference-time parameter). Each tracklet's motion is represented as a Cumulative B-spline in Lie algebra space over a set of compact control points, giving a continuous SE(3) trajectory. Each dynamic Gaussian is deformed by its nearest-neighbor set of control-point bases, blended via Dual Quaternion Blending (DQB). An adaptive control mechanism prunes low-utility control points and densifies new ones in high-motion/high-error regions during training, and a "soft segment reconstruction" scheme down-weights (via opacity) a Gaussian's contribution to frames temporally far from its reference time. A Zero-1-to-3 diffusion model supplies an SDS loss on unobserved views to regularize novel-view synthesis.

## Assumptions

Monocular video capture (single moving or static camera), off-the-shelf 2D/3D tracklets (from a tracker such as TAPNet) and optical flow (RAFT) are available as motion priors, and scene motion is assumed to be locally well-approximated by a sparse set of rigid SE(3) bases rather than dense free-form deformation.

## Limitations / Failure Modes

The paper reports degraded quality on scenes with substantial non-rigid deformation, since motion is represented via a sparse set of rigid SE(3) bases blended by DQB rather than a dense non-rigid field. It also fails under motion blur from fast object or camera movement and on generally blurry monocular video, where the underlying 2D tracks/flow priors become unreliable.

## Reusable Ingredients

- **Unified SE(3) Cumulative B-spline (vs. separate position/rotation splines):** guarantees joint positional+rotational continuity for rigid-body motion segments — reusable anywhere a trajectory is splined rather than per-frame estimated.
- **Adaptive control-point pruning/densification:** allocates motion-representation capacity to regions with high reconstruction error/dynamic content and removes redundant control points below a reconstruction-error threshold — a budget-aware capacity mechanism analogous to Gaussian densification but applied to motion bases, not geometry.
- **Soft segment reconstruction (temporal opacity weighting):** `o' = sigmoid(scale·(1-|t_ref - t_obs|)) * o` softly fades a Gaussian's influence away from its reference time, reducing interference from long-interval motion extrapolation.
- **Multi-view diffusion SDS regularization (Zero-1-to-3) on unseen views:** mitigates single-camera overfitting for monocular dynamic reconstruction.
- **Nearest-neighbor basis assignment + Dual Quaternion Blending:** a lightweight way to let each Gaussian be influenced by multiple nearby rigid motion bases without a learned soft-assignment network.

---

### Deep Dive

#### Core Novelty

Relative to SplineGS (which splines position and rotation as independent channels) and MoSca (which uses a fixed motion scaffold with tracklet-driven deformation but no explicit continuous-time spline formulation), this paper's contribution is (1) computing relative pose transforms between adjacent 3D tracklet poses, mapping them into Lie algebra, and cumulatively composing SE(3) B-spline basis functions so translation and rotation are continuous and coupled in a single trajectory, and (2) making the number/placement of spline control points adaptive (prune + densify) instead of fixed, so motion capacity tracks scene complexity. The insight is that treating rigid motion as a single Lie-group curve (rather than two independently-splined channels) is both more physically correct and gives a natural, differentiable handle (control points) for adaptive capacity allocation.

#### Mathematical Formulation

Core trajectory equation, evaluated per tracklet/motion basis to produce a continuous pose at query time $t$ (used to deform associated Gaussians before rasterization):

$$T(t) = \left(\prod_{i=0}^{N_x-1} \exp(\Omega_i(t)\,\xi_i)\right) T_0$$

- $T(t) \in SE(3)$: the interpolated rigid pose at time $t$.
- $T_0$: the pose at the first control point (base pose).
- $\xi_i \in \mathfrak{se}(3)$: the relative pose transform between adjacent control points, expressed in Lie algebra (twist coordinates).
- $\Omega_i(t)$: the B-spline basis (blending) function for control point $i$, giving the cumulative weight applied to $\xi_i$ at time $t$.
- $\exp(\cdot)$: the Lie-algebra-to-group exponential map, ensuring each incremental transform stays in SE(3).
- The product is a cumulative composition (as in Cumulative B-splines / spline fusion for pose interpolation), which is what gives joint position+rotation continuity, unlike splining position and rotation as separate channels.

Soft segment reconstruction (evaluated per-Gaussian, per-frame, modifying opacity before rasterization):

$$o' = \text{sigmoid}\big(\text{scale} \cdot (1 - |t_{ref} - t_{obs}|)\big) \cdot o$$

- $o$: the Gaussian's base opacity; $o'$: the temporally-weighted opacity used for rendering.
- $t_{ref}$: the Gaussian's reference (initialization) time; $t_{obs}$: the current observed/rendered frame's time.
- $\text{scale}$: a temperature-like constant controlling how sharply opacity falls off with temporal distance (exact value not specified in accessible text).

Pruning criterion: control points are removed when their contribution to reconstruction error falls below threshold $\epsilon_{prune} = 5.0$ (units/definition of "reconstruction error" not specified in accessible text).

Loss weighting (evaluated as the total training objective): $\lambda_{rec}=1.0$ (with SSIM weight $\beta=0.2$ inside the reconstruction term), $\lambda_{geo}=0.075$, $\lambda_{sds}=0.01$, plus motion-smoothness (ARAP + optical-flow tracking) and camera-pose smoothness terms whose individual weights are not specified in accessible text.

#### Algorithm / Pipeline Changes

1. Initialize static Gaussians from background depth reprojection; initialize dynamic Gaussians from 3D tracklet reprojection, each carrying an additional reference-time parameter $t_{ref}$.
2. For each tracklet, fit a set of SE(3) control-point poses; compute relative transforms $\xi_i$ between adjacent control points in Lie algebra space; this defines a Cumulative B-spline trajectory $T(t)$ (Eq. above) — replaces SplineGS's separate position/rotation spline fitting.
3. Every 500 iterations (pruning/densification interval), run the adaptive control step: prune control points whose reconstruction-error contribution is below $\epsilon_{prune}=5.0$; densify (add) new control points in regions with high rendering error and inside dynamic-region masks — this augments/replaces a fixed-capacity motion scaffold with a budget-aware one.
4. At render time, for each dynamic Gaussian, find its nearest-neighbor controlling SE(3) B-spline bases and blend their pose contributions via Dual Quaternion Blending (DQB) to get the Gaussian's deformed pose at time $t$ — runs per-Gaussian, before rasterization.
5. Apply soft segment reconstruction: scale each dynamic Gaussian's opacity by the temporal-distance-based sigmoid weight (Eq. above) before rasterization, so Gaussians contribute less to frames far from their reference time.
6. Render, then supervise with: reconstruction loss (L1+SSIM), depth-based geometry loss, motion smoothness (ARAP + optical-flow tracking), camera-pose smoothness, and — for regions/views not covered by the single training camera — a Zero-1-to-3 diffusion-model SDS loss to regularize novel views.

#### Key Hyperparameters & Design Choices

- Training length: 8,000 iterations, Adam optimizer.
- Initial control-point count: ~100 (adaptively adjusted via prune/densify).
- Prune/densify interval: every 500 iterations.
- Pruning threshold: $\epsilon_{prune} = 5.0$.
- Learning rates: $1.6\times10^{-4}$ for motion bases, $3\times10^{-4}$ for camera parameters.
- Loss weights: $\lambda_{rec}=1.0$ (SSIM component weight $\beta=0.2$), $\lambda_{geo}=0.075$, $\lambda_{sds}=0.01$; other loss weights not specified in accessible text.
- Robustness: tolerates roughly ±15 pixel perturbation added to 2D tracking priors with minimal performance degradation (specific degraded-metric value not specified in accessible text).

#### Ablation Summary

- Adaptive control mechanism (prune + densify): **+1.33 dB** on iPhone dataset (mPSNR) — the single most impactful ablated component reported.
- Soft segment reconstruction: contributes positively (exact delta not specified in accessible text).
- Diffusion-based SDS loss: contributes positively (exact delta not specified in accessible text).
- Camera-pose smoothness regularization: contributes positively (exact delta not specified in accessible text).

#### Implementation Reality

- **Framework:** PyTorch 2.5.0 / CUDA 12.4, Python 3.10; extends a MoSca-style codebase rather than the vanilla gaussian-splatting repo.
- **Key files/dirs:** `mosca_reconstruct.py` (main training pipeline), `mosca_evaluate.py` / `mosca_viz.py` (eval and visualization), `lite_moca_reconstruct.py` (lightweight variant), `lib_mosca/`/`lib_moca/` (motion capture/analysis utilities implementing the scaffold and B-spline logic), `lib_prior/` (prior modeling, likely the diffusion SDS prior), `lib_render/` (Gaussian rasterization), `data_utils/`/`eval_utils/`, `profile/iphone/` (iPhone-dataset configs).
- **Notable implementation details:** repo dependencies explicitly reference MoSca, SplineGS, and DreamScene4D, confirming the paper builds directly on MoSca's tracklet/scaffold machinery and DreamScene4D's approach to multi-view diffusion consistency rather than a from-scratch SDS integration. Repository README does not surface additional undocumented deviations from the paper beyond this dependency structure.

#### Failure Modes & Limitations

The paper itself identifies: (1) degraded quality on scenes with substantial non-rigid deformation, since motion is represented as a sparse set of blended rigid SE(3) bases rather than a dense free-form field; (2) failure under motion blur from rapid object or camera movement; (3) failure on generally blurry monocular video, where the 2D tracking/flow priors the method depends on become unreliable.

## Relevance to ADAGS

Direct competitor to LoRA/scaffold motion basis claims.

## Connections

## Sources

- https://arxiv.org/abs/2603.25058
- https://github.com/hhhddddddd/se3bsplinegs
