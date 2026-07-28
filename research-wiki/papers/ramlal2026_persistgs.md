---
type: paper
node_id: paper:ramlal2026_persistgs
title: "PersistGS: Differentiable Physics for Object Permanence in 4D Gaussian Splatting"
authors: ["Adrian Ramlal", "John S. Zelek"]
year: 2026
venue: "CVPRW / arXiv"
external_ids:
  arxiv: "2606.03479"
tags: [dynamic-gs, object-permanence, occlusion, differentiable-physics, rigid-body]
status: deep-dived
---

# PersistGS: Differentiable Physics for Object Permanence in 4D Gaussian Splatting

**Paper:** https://arxiv.org/abs/2606.03479
**Code:** Not found
**Base method:** Object-decomposed 3D Gaussian Splatting (per-object Gaussians + static background 3DGS) with Gaussians initialized/segmented via MV-SAM3D, coupled to differentiable rigid-body simulation (NVIDIA Newton).

## One-line thesis

A differentiable rigid-body physics simulator, driven by only four scalar/vector parameters (friction, initial velocity) fit from pre-occlusion frames via a centroid-silhouette loss, supplies the SE(3) trajectory of a fully-occluded object's Gaussians, replacing both kinematic extrapolation (which cannot handle contact/bounce events) and photometric-gradient-starved deformation fields.

## Problem / Gap

When a moving object passes fully behind a static occluder relative to every training camera, its Gaussians receive zero photometric gradient and either freeze, drift, or are hallucinated by generative priors that look plausible but are physically wrong. Kinematic extrapolation (e.g. constant velocity) cannot model contact events such as bounces or friction-driven direction changes, and deformation-field learning overfits badly under the sparse multi-view capture (5 cameras) used here.

## Method

The scene is split into static background Gaussians (~25K, standard 3DGS with an inverse-mask loss that excludes the object region) and per-object Gaussians (~511K, produced along with a collision mesh by MV-SAM3D). Four physical parameters — friction coefficient $\mu$ and initial velocity $\mathbf{v}_0=(v_x,v_y,v_z)$ — are fit in log-space from pre-occlusion frames using NVIDIA Newton, a differentiable rigid-body simulator with penalty-based contact, run with 8 substeps/frame; mass, contact stiffness, friction stiffness, and damping are fixed constants. Fitting uses a novel centroid-silhouette loss (see below) instead of photometric loss because it yields far larger gradients on friction. During occlusion the object's Gaussians are rigidly transformed frame-by-frame by the simulator's output SE(3) poses with no photometric supervision; during visible frames a small per-frame residual translation is additionally optimized to correct residual position error, and this residual optimization/photometric supervision resumes smoothly on re-emergence.

## Assumptions

Synchronized multi-view video (5 training + 2 held-out evaluation cameras, 512×512 @ 60fps) of a scene containing one rigid object with a known segmentation/collision mesh (from MV-SAM3D) moving against a static background and occluder; the physics model assumes single-object rigid-body dynamics with contact against static geometry, not deformable or multi-object interactions.

## Limitations / Failure Modes

Physics fitting is blind to parameters that are unobservable in the pre-occlusion window — on `ball_bounce`, friction is unobservable before the first contact ($\partial\mathcal{L}/\partial\mu=0$), so the physics-based and non-physics baselines end up comparable there. On the near-linear `ball_roll` scene (45-frame occlusion, no contact event), a simple non-causal linear-interpolation baseline nearly matches PersistGS (-0.17 dB), implying physics mainly helps when the occluded interval contains a nonlinear contact event (bounce). The method is validated only on rigid, roughly spherical single objects; rotation-aware supervision (via silhouette second moments) is proposed but not validated on objects whose orientation changes visual appearance, and multi-object contact is explicitly unaddressed. Object decomposition currently requires a known/given segmentation rather than automatic discovery.

## Reusable Ingredients

- **Centroid silhouette loss**: supervising the alpha-weighted centroid of the rendered object mask against ground truth gives ~100x larger gradient signal for physical parameters (e.g. friction) than raw photometric loss — useful whenever a physical/kinematic parameter must be fit through a renderer.
- **Observability-aware curriculum**: splitting parameter fitting into phases that isolate which frames can identify which parameter (velocity from pre-contact frames with friction frozen, then friction from a post-contact window with velocity frozen, then joint refinement at reduced LR) avoids conflating under-determined parameters.
- **Residual-disable-under-occlusion pattern**: freezing per-frame correction terms during periods with no supervision signal (rather than letting them drift or extrapolate) and only resuming them when supervision returns.
- **Withheld-camera evaluation protocol for occlusion**: using cameras from which the object is occluded during training, but visible from held-out evaluation cameras, to directly score permanence quality.

---

### Deep Dive

#### Core Novelty

Relative to plain object-decomposed dynamic 3DGS, PersistGS's change is to replace the object's per-frame learned/deformed pose with the output of a differentiable rigid-body physics simulator whose only free parameters (friction, initial velocity) are fit from the visible portion of the trajectory. The key insight is that photometric loss gives almost no gradient signal for physical parameters like friction (contact events are sparse and low-gradient in pixel space), so a geometry-only proxy — the alpha-weighted silhouette centroid — is used instead to fit those parameters, then the resulting physically-consistent trajectory is trusted through occlusion instead of being re-estimated from vanished gradients.

#### Mathematical Formulation

**Centroid silhouette loss** (Eq. 1), evaluated per training iteration on the rendered object-only image, before/alongside the physics-parameter update (not a per-Gaussian rasterization-stage term):
$$\mathcal{L}_{\text{sil}} = \lVert \mathbf{c}_{\text{render}} - \mathbf{c}_{\text{gt}} \rVert_2^2, \qquad \mathbf{c} = \frac{\sum_i \alpha_i \mathbf{p}_i}{\sum_i \alpha_i}$$
where $\alpha_i$ is the rendered alpha (opacity) value at pixel $i$ and $\mathbf{p}_i$ is that pixel's 2D position. $\mathbf{c}$ is thus the alpha-weighted centroid of the object's silhouette in image space, computed for both the render and the ground-truth mask.

**Rigid SE(3) Gaussian transform**, applied per-frame to every canonical object Gaussian, driven by the simulator's output pose $(R_t, t_t)$ (translation + quaternion) — evaluated after physics simulation, before rasterization:
$$\mu' = R_t \mu_{\text{can}} + t_t, \qquad q' = q_{R_t} \otimes q_{\text{can}}$$
Scales are left unchanged (rigid motion preserves shape); spherical-harmonic color is re-evaluated with the view direction rotated into the canonical frame: $\text{color} = SH(R_t^{-1} d_{\text{view}}; C_{\text{can}})$.

**Modified friction contact model**: the simulator's default hard-min friction clamp is replaced with a harmonic combination,
$$f = \frac{k_f v_t \cdot f_c}{k_f v_t + f_c + \epsilon}$$
to keep the friction force differentiable near the stick/slip transition (variables: $k_f$ = friction stiffness constant, $v_t$ = tangential velocity, $f_c$ = Coulomb friction limit, $\epsilon$ = small constant for numerical stability).

#### Algorithm / Pipeline Changes

1. Segment/reconstruct scene into static background (standard 3DGS, ~25K Gaussians, trained 5K iterations with an inverse-object-mask-weighted loss) and object (MV-SAM3D gives Gaussians, ~511K, and a collision mesh), object trained 7K iterations with degree-3 SH, DropGaussian dropout p=0.5, Mip-Splatting 3D smoothing, and Depth Anything V2 depth supervision ($\lambda_{\text{depth}}=0.05$) for sparse-view regularization.
2. From pre-occlusion frames, run a 3-phase curriculum (200 total iterations, 5 random seeds) in NVIDIA Newton to fit $\mu$ and $\mathbf{v}_0$ in log-space using $\mathcal{L}_{\text{sil}}$: Phase 1 (60 iter) fits velocity from pre-contact frames with friction frozen; Phase 2 (60 iter) fits friction from a post-contact window with velocity frozen; Phase 3 (80 iter) jointly refines both at 0.3x the earlier learning rate.
3. Simulate forward with Newton (8 substeps/frame, penalty-based contact, fixed mass=1.0, $k_e=10^5$, $k_f=10^3$, $k_d=10^2$) to produce a per-frame SE(3) pose track for the object across the full sequence, including the occluded interval.
4. Rigidly transform the object's canonical Gaussians by this simulated pose track each frame (equations above) to render the object during occlusion — no photometric gradients flow into object Gaussians during occluded frames.
5. During visible (non-occluded) frames, additionally optimize a small per-frame residual translation (3 optimization passes, LR $10^{-4}$) on top of the rigid transform to correct small errors from imperfect parameter estimation; residual optimization is disabled during occluded frames and re-enabled smoothly once the object re-emerges and photometric supervision resumes.
6. Also bypasses Newton's default per-material property averaging to enable full gradient routing through contact parameters, reported to give ~50x larger gradients.

#### Key Hyperparameters & Design Choices

- Background: 5K iterations, ~25K Gaussians.
- Object: 7K iterations, ~511K Gaussians, SH degree 3, DropGaussian rate 0.5.
- Sparse-view depth loss weight $\lambda_{\text{depth}} = 0.05$ (Depth Anything V2 supervision).
- Physics fitting: 200 iterations total across 3 curriculum phases (60/60/80), 5 random seeds, ~5.5 min/scene on an RTX 5080.
- Phase 3 joint-refinement learning rate: 0.3x the phase 1/2 rate (absolute LR not specified in extracted text).
- Residual translation optimization during visible frames: 3 passes, LR $10^{-4}$.
- Fixed physics constants: mass $m=1.0$, contact stiffness $k_e=10^5$, friction stiffness $k_f=10^3$, damping $k_d=10^2$.
- Simulator substeps: 8 per frame.
- Capture setup: 5 training cameras + 2 evaluation cameras, 512×512, 60 fps.
- Not specified in paper: absolute learning rates for phases 1/2 of physics fitting; MV-SAM3D-specific hyperparameters; background/object Gaussian densification schedule.

#### Ablation Summary

- **Centroid-silhouette loss vs. photometric loss** (most impactful component for parameter fitting): 40% lower mean trajectory RMSE (0.586 vs. 0.984) and +0.41 dB mean PSNR; on `ball_fall` specifically, 3.4x lower trajectory RMSE and ~100x larger friction gradients. This is flagged as the single most impactful design choice — without it, friction is effectively unfittable through the renderer.
- **Sparse-view regularization** (Mip-Splatting smoothing + DropGaussian + depth supervision): removing it costs +3.0 dB PSNR degradation on `ball_bounce` and +4.2 dB on `ball_roll`.
- **Noise tolerance**: PSNR degrades ~1 dB per $\sigma=0.25$ of added input noise.
- **End-to-end occlusion reconstruction** (Table 2, mean across 3 scenes): PersistGS 17.15 dB / 0.314 LPIPS vs. GT-trajectory upper bound 17.34 dB / 0.284, vs. linear interpolation 15.74 dB / 0.381, vs. constant velocity 14.69 dB / 0.491, vs. no-physics 12.01 dB / 0.716 — i.e., PersistGS closes almost all the gap to the GT-trajectory upper bound and beats constant-velocity by ~2.46 dB.

#### Failure Modes & Limitations

Friction is unobservable pre-contact on `ball_bounce`, so physics and non-physics baselines converge there; post-occlusion observations could reportedly buy back ~0.80 dB via iterative refinement (not implemented). On `ball_roll` (45-frame occlusion, no contact event, nearly linear motion), non-causal linear interpolation is competitive (-0.17 dB vs. PersistGS), indicating the physics simulator's advantage is concentrated in occlusion intervals containing nonlinear contact events (bounces), not smooth free motion. Rotation estimation via silhouette second moments is proposed for asymmetric objects but not validated. Multi-object contact dynamics are out of scope. Object decomposition currently depends on a provided/known segmentation (MV-SAM3D) rather than automatic discovery — the paper lists integration with automatic decomposition methods (e.g. Gaussian Grouping) as future work.

## Relevance to ADAGS

Important pressure on Event-Causal Hide/Reveal Gaussians. The distinction should be: PersistGS uses a physics prior for rigid object trajectories through full occlusion; ECHRG should provide a primitive-level visibility-event test that can plug into dynamic-GS backbones without assuming rigid-body simulation.

## Connections

- Pressures [[ideas/event-causal-visibility-gaussians]]
- Addresses [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]

## Sources

- https://arxiv.org/abs/2606.03479
