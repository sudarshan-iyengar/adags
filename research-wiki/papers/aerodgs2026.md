---
type: paper
node_id: paper:aerodgs2026
title: "AeroDGS: Dynamic 3D Gaussian Splatting from Aerial Imagery"
authors: []
year: 2026
venue: "CVPR"
tags: [dynamic-gs, aerial, niche]
status: deep-dived
---

# AeroDGS: Dynamic 3D Gaussian Splatting from Aerial Imagery

**Paper:** https://arxiv.org/abs/2602.22376 (full title: "AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction", Hanyang Liu & Rongjun Qin, Ohio State University, CVPR 2026)
**Code:** Not found. Project page (https://gdaosu.github.io/aerialdgs/) lists code as "coming soon"; no GitHub repository is linked from the arXiv page or the project page.
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023) for rendering, extended with per-object canonical-space Gaussians and object-level rigid SE(3) motion trajectories in the style of 4DGF / "Dynamic 3D Gaussian Fields for Urban Areas" (Fischer et al. 2024), which the paper explicitly cites as the inspiration for its appearance-field design.

## One-line thesis

Three differentiable physics priors (ground-support, upright-stability, trajectory-smoothness) applied to per-object SE(3) trajectories resolve the depth/pose ambiguity of monocular aerial video, letting a single-camera UAV capture reconstruct moving objects that would otherwise be geometrically under-constrained without multi-view or LiDAR priors.

## Problem / Gap

Ground-based dynamic Gaussian methods (4DGF, Street Gaussians-style pipelines) resolve per-object motion using multi-view or LiDAR priors, which lightweight UAV platforms with a single monocular camera cannot provide. Indoor deformation-field methods (4DGS, deformable 3DGS) target small-scale scenes with large articulated motion under controlled lighting and do not generalize to wide-area outdoor aerial footage with tiny, fast, low-parallax dynamic objects. Feed-forward monocular reconstruction models (VGGT, DUSt3R-style) are trained on ground-level data and produce unreliable motion recovery when applied to aerial imagery, so the depth and 3D position of small aerial dynamic objects remain fundamentally ambiguous under monocular capture.

## Method

A Monocular Geometry Lifting (MGL) module bootstraps the scene from purely 2D foundation-model cues: per-frame monocular depth (UniDepth), segmentation/tracking of movable instances (Grounding DINO + SAM2 + a video object tracker), and long-term background point tracks (CoTracker3) that are triangulated and bundle-adjusted to recover scale-consistent camera poses and a dense point map. Per-object 3D centers and footprints are estimated by PCA-fitting oriented boxes to each instance's point cluster, with object height predicted by a pretrained MLP since height along the optical axis is unobservable monocularly; objects with sub-3m 3D displacement are classified static. The scene is then represented as static Gaussians in world space plus per-object dynamic Gaussians in canonical object space, each object animated by a continuous 6DoF SE(3) trajectory (with a learned residual correction) that is spline-interpolated across time and composed into the world frame for standard 3DGS alpha-compositing rendering. A Physics-Guided Optimization stage adds three regularizers on top of photometric (L1+SSIM) supervision: a ground-support loss that pulls each object's ray-projected center toward the local ground plane, an upright-stability loss that keeps the object's vertical axis aligned with the ground normal (or gravity for non-rigid objects), and a second-order trajectory-smoothness loss that penalizes discrete acceleration of the object center.

## Assumptions

Input is a single monocular UAV video of an urban scene (no multi-view or LiDAR); dynamic content is assumed to be dominated by ground-contact, near-rigid objects (vehicles) that stay upright and move smoothly, since the physics priors directly encode ground contact and upright orientation. The pipeline depends on off-the-shelf 2D foundation models (depth, segmentation, tracking, point-tracking) as bootstrap priors and needs sufficient camera parallax during flight for bundle adjustment to recover a metrically consistent static background.

## Limitations / Failure Modes

The fixed 3m motion threshold for dynamic/static classification misclassifies objects with small localized motion as static, so they are processed by the static pipeline and appear blurred in rendering. The method does not reconstruct pedestrians, since they occupy only a few pixels and appear partially in high-altitude aerial views. Dynamic-region PSNR trails static-region PSNR by a wide margin across all scenes (e.g., 37.91 dB static vs. 17.94 dB dynamic on Downtown-High) — the paper attributes this to positional sensitivity of small objects under nonlinear UAV/object motion rather than a genuine perceptual-quality gap. On the synthetic UAV3D benchmark, overall metrics are markedly lower than on the real-world Aero4D scenes (e.g., 33.61 dB vs. ~34 dB), which the paper attributes to UAV3D's discrete, long-baseline camera trajectories amplifying temporal misalignment.

## Reusable Ingredients

- **Local depth ratio-field scale correction**: computing a sparse scale-ratio field between geometric (triangulated) and predicted monocular depth at tracked points, then interpolating it densely to correct monocular depth-scale drift before back-projection.
- **Ground-support ray-cast loss**: a robust penalty on the signed distance (along the camera ray) between an object's center (offset to its base) and its projected intersection with the local ground plane — a generic way to resolve along-ray position ambiguity for ground-contact objects.
- **Upright-stability loss**: `1 - |dot(vertical_axis, reference_direction)|` as a lightweight, generic pose regularizer for any object expected to stay upright.
- **Second-order trajectory-smoothness loss**: penalizing discrete acceleration (`c_{t+1} - 2c_t + c_{t-1}`) of an object's trajectory to suppress jitter while still allowing objects to exit the scene with retained momentum rather than snapping to a stop.
- **Paired annealing schedule**: linearly increasing the dynamic-region photometric loss weight while linearly decaying the physics-prior weights over the same iteration window — a general curriculum for handing off from strong regularization to photometric fine detail once optimization has stabilized.
- **Shared continuous appearance field**: modeling per-Gaussian appearance as `f(position, view direction, time, instance embedding)` via hash-grid position encoding + spherical harmonics + sinusoidal time embedding, avoiding storing per-Gaussian time-varying spherical harmonics for large, dense dynamic Gaussian sets.

---

### Deep Dive

#### Core Novelty

Relative to prior dynamic Gaussian frameworks that lean on multi-view/LiDAR priors (4DGF) or unconstrained deformation fields (4DGS), AeroDGS's actual contribution is not a new rendering or motion representation but a set of three differentiable physical regularizers imposed on already-standard per-object SE(3) trajectories. The key insight is that aerial monocular capture removes the multi-view geometric constraints ground-based methods rely on to disambiguate object depth/pose, so the missing constraints must instead come from generic physical regularities of urban dynamic objects (they touch the ground, stay upright, and move with bounded acceleration) rather than from additional sensor data.

#### Mathematical Formulation

Monocular back-projection (per pixel, during Monocular Geometry Lifting):
$$X_t(x) = \Pi^{-1}(x, \tilde{D}_t(x), K) \quad (1)$$
where $\tilde{D}_t$ is the scale-corrected depth map (raw monocular depth rescaled by a locally interpolated ratio field fit on triangulated track points) and $K$ is the camera intrinsics matrix. Produces the per-frame point map used for both static geometry and object bounding-box fitting.

Appearance field (evaluated per Gaussian, at rendering time):
$$A_i = f(\mu_i, d, t, e_o) \quad (2)$$
a shared field conditioned on Gaussian center $\mu_i$, view direction $d$, time $t$, and an instance embedding $e_o$ for dynamic objects — encoded via hash-grid position, spherical-harmonic view-dependence, and sinusoidal time encoding.

Object motion, as a continuous SE(3) trajectory (per dynamic object, per frame, before compositing):
$$T_{o,t} = \exp(\xi_o(t)) \quad (3)$$
$$\hat{T}_{o,t} = T_{o,t} \cdot \Delta T_{o,t} \quad (4)$$
$$\mu_{i,t} = \hat{T}_{o,t} \cdot \mu_i \quad (5)$$
where $\xi_o(t) \in \mathfrak{se}(3)$ is the time-dependent twist vector and $\Delta T_{o,t}$ is a jointly-optimized residual correction for small pose deviations. Trajectories are spline-interpolated in SE(3) between timestamps.

Scene composition (per timestep, before rasterization):
$$G(t) = G_{static} \cup \bigcup_{o \in O} \hat{T}_{o,t} \cdot G_o \quad (6)$$

Rendering (standard 3DGS alpha-compositing, unchanged):
$$I_t(x) = \sum_i \alpha_i(x)\, A_i(x) \prod_{j<i}(1 - \alpha_j(x)) \quad (7)$$

Overall training objective:
$$\mathcal{L} = \lambda_{photo}\mathcal{L}_{photo} + \lambda_{sup}\mathcal{L}_{support} + \lambda_{upr}\mathcal{L}_{upright} + \lambda_{traj}\mathcal{L}_{traj} \quad (8)$$

Ground-support loss (per object, per frame — the core novel regularizer):
$$\mathcal{L}_{support} = \mathbb{E}_{o,t}\big[\rho\big(r_{o,t}(c_{o,t} - \hat{c}^g_{o,t})\big)\big] \quad (9)$$
$\rho(\cdot)$ is a robust penalty on the signed distance, measured along the camera viewing ray $r_{o,t}$, between the object center $c_{o,t}$ (shifted upward by half the object height, i.e. anchored to the object's base) and $\hat{c}^g_{o,t}$, its intersection with the locally-inferred ground plane.

Upright-stability loss:
$$\mathcal{L}_{upright} = \mathbb{E}_{o,t}\big[1 - |u_{o,t} \cdot v_{o,t}|\big] \quad (10)$$
$u_{o,t}$ is the object's vertical axis; $v_{o,t}$ is the reference direction — the local ground normal $n_t$ for rigid objects, or the gravity direction $g$ for non-rigid ones.

Trajectory-smoothness loss (second-order / discrete acceleration):
$$\mathcal{L}_{traj} = \mathbb{E}_{o,t}\big[\|c_{o,t+1} - 2c_{o,t} + c_{o,t-1}\|_2^2\big] \quad (11)$$

#### Algorithm / Pipeline Changes

1. Run a monocular depth estimator (UniDepth) per frame to get dense pseudo-depth, plus zero-shot segmentation (Grounding DINO + SAM2) and video tracking to get temporally-consistent 2D instance masks.
2. Triangulate long-term background point tracks (CoTracker3) and refine via local bundle adjustment to obtain scale-consistent camera poses $P_t = [R_t | t_t]$ and background 3D keypoints.
3. Compute a local depth-ratio field between triangulated geometric depth and monocular pseudo-depth at tracked points, interpolate it densely, and rescale the monocular depth map into $\tilde{D}_t$; back-project every pixel via Eq. 1 to build a joint static+dynamic point map.
4. Group pixels by 2D instance ID into per-object point sets, PCA-fit an oriented 3D bounding box per object (center $c_{o,t}$, footprint $(w,\ell)$), and predict height $h$ with a pretrained MLP (since it is unobservable monocularly). Resolve ID switches/occlusions by grounding objects at physically plausible positions along camera rays plus trajectory smoothness. Classify objects with <3m total 3D displacement as static (skip further tracking), the rest as dynamic candidates.
5. Initialize static Gaussians in world space from the static point cloud, and dynamic Gaussians per object in canonical object space from points within the estimated bounding box; initialize per-object trajectories $T_{o,t}$ from the tracked box centers; add a learnable per-frame camera-pose residual $\Delta P_t$.
6. Jointly optimize all Gaussian parameters, object trajectories/residuals, and camera-pose residuals against the combined loss (Eq. 8) via differentiable 3DGS rasterization (Eq. 6-7), for 30K iterations.
7. During optimization, run a two-stage weighting schedule: equal static/dynamic photometric weighting during warm-up, then increase the dynamic-region photometric weight and decay the three physics-prior weights over the 7K-15K iteration window, letting photometric signal take over once the physics priors have stabilized the coarse configuration.

#### Key Hyperparameters & Design Choices

- Training: 30K iterations, Adam optimizer, single NVIDIA RTX 6000 Ada GPU (48 GB).
- Loss weights: $\lambda_{photo}=1.0$, $\lambda_{sup}=0.05$, $\lambda_{upr}=0.1$, $\lambda_{traj}=0.02$.
- Dynamic-region photometric weight increases linearly from 1.0 to 1.3 between iterations 7K and 15K; physics-guided constraint weights decay by 50% over the same window.
- Static/dynamic classification threshold: 3m total 3D displacement.
- Train/validation split: 8:2, full-resolution images.
- Object height prediction: "a pretrained MLP" — architecture, hidden dims, and training data not specified in paper.
- Hash-grid resolution/levels, spherical-harmonic degree, and sinusoidal time-encoding frequency count for the appearance field: not specified in paper.
- Object canonical-space bounding box dimensions $(w, \ell, h)$: derived from PCA fit + predicted height; no fixed values given (per-object, per-scene).

#### Ablation Summary

From Table 3 (Aero4D dataset; note the source PDF's table had a row/column misalignment during text extraction — the mapping below was inferred by cross-checking against the paper's prose, which states the full model achieves the best metrics and that removing initialization causes the largest Dyn-PSNR drop; both checks are consistent with this mapping):

| Variant | PSNR | SSIM | LPIPS | Dyn-PSNR |
|---|---|---|---|---|
| w/o Initialization (COLMAP+fixed-height-prior instead of MGL) | 33.82 | 0.952 | 0.027 | 17.75 |
| w/o Ground Support | 34.51 | 0.967 | 0.022 | 19.23 |
| w/o Upright Stability | 34.10 | 0.963 | 0.025 | 19.35 |
| w/o Trajectory Smoothness | 34.63 | 0.966 | 0.022 | 19.89 |
| w/o Dynamic Mask (weighting schedule) | 34.23 | 0.959 | 0.024 | 19.12 |
| **Ours (Full Model)** | **34.67** | **0.971** | **0.021** | **20.07** |

Most impactful component: replacing the Monocular Geometry Lifting initialization with a COLMAP + fixed-height-prior baseline causes the largest degradation (Dyn-PSNR drops ~2.3 dB relative to the full model), which the paper attributes to plane bias and pose inaccuracy from the coarser initialization. Each physics-guided loss term (ground-support, upright-stability, trajectory-smoothness) and the dynamic-mask weighting schedule contribute smaller, roughly comparable individual gains (~0.2-1.0 dB Dyn-PSNR each).

#### Failure Modes & Limitations

The fixed 3m dynamic/static motion threshold misclassifies small-localized-motion objects as static, causing them to be rendered blurred by the static pipeline. Pedestrians are not reconstructed at all, since they cover too few pixels at typical UAV altitudes. The paper also notes the large static-vs-dynamic PSNR gap across all evaluated scenes is a byproduct of positional sensitivity for small objects under nonlinear UAV/object motion, not necessarily a perceptual quality failure, and that the synthetic UAV3D benchmark's discrete/long-baseline camera trajectories depress absolute metrics relative to the real-world Aero4D scenes.

## Relevance to ADAGS

Another example that application-specific dynamic GS niches are viable when paired with targeted assumptions and evaluation.

## Connections

## Sources

- https://arxiv.org/abs/2602.22376
- https://gdaosu.github.io/aerialdgs/
