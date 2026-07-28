---
type: paper
node_id: paper:jiang2024_motiongs
title: "MotionGS: Exploring Explicit Motion Guidance for Deformable 3D Gaussian Splatting"
authors: ["Ruijie Zhu", "Yanzhe Liang", "Hanzhi Chang", "Jiacheng Deng", "Jiahao Lu", "Wenfei Yang", "Tianzhu Zhang", "Yongdong Zhang"]
year: 2024
venue: "NeurIPS"
external_ids:
  arxiv: "2410.07707"
  doi: "10.48550/arXiv.2410.07707"
  s2: null
tags: [dynamic-gs, motion-guidance, flow]
status: deep-dived
---

# MotionGS: Exploring Explicit Motion Guidance for Deformable 3D Gaussian Splatting

**Paper:** https://arxiv.org/abs/2410.07707
**Code:** https://github.com/RuijieZhu94/MotionGS
**Base method:** Deformable 3D Gaussian Splatting (deformation-MLP 3DGS, Yang et al. 2024), built with components from GaussianFlow, MonoGS, CF-3DGS, DynPoint, MiDaS, GMFlow, and MDFlow.

## One-line thesis

Off-the-shelf 2D optical flow, decoupled into a camera-induced component and an object-motion component via rendered depth and camera pose, supplies an explicit per-pixel supervisory target that a rasterizer-derived "Gaussian flow" is regressed against — giving the deformation network a direct motion signal instead of relying solely on photometric reconstruction loss.

## Problem / Gap

Deformation-MLP methods for monocular dynamic 3DGS (e.g. Deformable-3DGS) supervise Gaussian deformation only through appearance (photometric) loss, with no explicit motion signal. For scenes with irregular or fast object motion this is underconstrained and the optimization falls into local optima (blurry or drifting reconstructions), because photometric loss alone cannot disambiguate camera-induced apparent motion from true object motion, nor tell the deformation field which direction a Gaussian should move.

## Method

MotionGS runs two parallel data streams per adjacent frame pair. In the 2D stream, an off-the-shelf optical flow network (GMFlow) computes total 2D flow between frames $t$ and $t+1$; this is decoupled into a "camera flow" (the flow that would be observed from a static scene, computed by reprojecting depth-unprojected points through the two camera poses) and a "motion flow" (total flow minus camera flow), which isolates the component of pixel motion attributable to actual object movement. In the 3D stream, each Gaussian's deformation network predicts its position/rotation/scale at $t+1$; a "Gaussian flow" is computed by projecting each Gaussian's positional change to 2D and alpha-blending per-pixel inside the CUDA rasterizer. An L1 loss between the (stop-gradiented) motion flow and the rendered Gaussian flow supervises the deformation network directly. A camera pose refinement module alternately optimizes the Gaussians (with poses frozen) and a small learnable SE(3) residual on the COLMAP-initialized camera poses (with Gaussians frozen), correcting pose error that would otherwise leak into the flow decoupling.

## Assumptions

Monocular video input with per-scene COLMAP camera pose/point initialization, requiring "sufficient static features" in the scene for COLMAP to succeed. Assumes an off-the-shelf optical flow estimator (GMFlow by default) produces usable dense correspondences between adjacent frames, and that a renderable/estimable per-frame depth map is available to unproject pixels for the camera-flow computation.

## Limitations / Failure Modes

The paper explicitly notes the method still depends on COLMAP-computed poses as initialization and therefore inherits COLMAP's requirement for sufficient static texture/features in the scene; it states future work should target a 3DGS variant that does not require camera pose input at all. The ablations show that naive (non-decoupled) optical flow guidance actually *hurts* quality relative to the no-flow baseline (23.37 vs 23.61 PSNR on NeRF-DS) — only the decoupled motion-flow signal helps — indicating the method is sensitive to camera-flow contamination and would degrade further under noisy pose/depth estimates. Removing the motion mask drops PSNR from 24.12 to 23.13, and swapping the flow network (MDFlow: 23.25, FlowFormer: 23.97) or depth source (monocular MiDaS: 23.58) both underperform the default GMFlow/rendered-depth combination, showing real dependence on upstream estimator quality.

## Reusable Ingredients

- **Optical-flow decoupling (camera flow vs. motion flow):** reprojects depth-unprojected pixels through consecutive camera poses to isolate ego-motion-only flow, then subtracts it from total observed flow to recover a motion-only supervisory signal — directly useful anywhere raw optical flow would otherwise conflate camera and object motion.
- **Rasterizer-native "Gaussian flow":** computes a 2D flow field from the alpha-blended change in each Gaussian's projected position between two timesteps, giving a differentiable quantity that can be regressed against any 2D motion prior without extra rendering passes.
- **Stop-gradient flow loss:** treats the derived motion-flow target as fixed (`sg(·)`) so the loss only shapes the Gaussian/deformation side, avoiding the flow estimator being pulled by the 3D optimization.
- **Alternating pose/geometry optimization:** freezing Gaussians while refining a small SE(3) residual on camera poses (and vice versa) as a way to correct upstream pose error without destabilizing the flow-decoupling supervision.

---

### Deep Dive

#### Core Novelty

Relative to plain deformation-MLP 3DGS, MotionGS adds an explicit, decoupled 2D motion supervisory signal (motion flow) and a matching rasterizer-derived Gaussian flow to regress against it, plus an alternating camera-pose refinement step. The key insight is that raw optical flow is not directly usable as a 3D motion supervisory signal because it mixes camera ego-motion with object motion; only after removing the camera-induced component does flow supervision improve reconstruction (confirmed by the ablation where raw optical flow guidance underperforms the no-flow baseline).

#### Mathematical Formulation

Camera flow — reprojects a pixel unprojected via rendered/estimated depth through consecutive camera poses to obtain the flow expected from ego-motion alone (evaluated once per adjacent frame pair, before the flow loss):
$$x_t = T_t^{-1} K_t^{-1} D_t \tilde{p}_t \qquad (4)$$
$$p_t^{t+1} = \mathrm{proj}(K_{t+1} T_{t+1} x_t) \qquad (5)$$
$$F^{C}_{t \to t+1} = p_t^{t+1} - p_t \qquad (6)$$
where $K$ is camera intrinsics, $T$ is camera extrinsics, $D_t$ is the depth map at frame $t$, $\tilde{p}_t$ is a homogeneous pixel coordinate, and $\mathrm{proj}(\cdot)$ is perspective projection to 2D.

Motion flow — total optical flow minus camera flow, isolating object motion (evaluated per adjacent frame pair, before the flow loss):
$$F^{M}_{t \to t+1} = F_{t \to t+1} - F^{C}_{t \to t+1} = p_{t+1} - p_t^{t+1} \qquad (7)$$
where $F_{t \to t+1}$ is the raw optical flow from GMFlow and $p_{t+1}$ is the pixel's tracked position at $t+1$.

Gaussian flow — the 2D projection of each Gaussian's 3D deformation, alpha-blended per pixel (evaluated per-Gaussian inside the CUDA rasterizer, per training step):
$$\hat{x}_t = \Sigma_{i,t}^{-1}(x_t - \mu_{i,t}) \qquad (10)$$
$$x_{i,t+1} = \Sigma_{i,t+1}\hat{x}_t + \mu_{i,t} \qquad (11)$$
$$F^{G}_{i,t \to t+1} = x_{i,t+1} - x_t \qquad (12)$$
$$F^{G}_{t \to t+1} = \sum_i w_i (x_{i,t+1} - x_t) \qquad (13\text{-}14)$$
where $\mu_{i,t}$ and $\Sigma_{i,t}$ are Gaussian $i$'s center and covariance at time $t$ (from the deformation network's output), and $w_i$ is the standard alpha-blending weight from the rasterizer.

Flow loss — stop-gradient L1 between motion flow and rendered Gaussian flow (a loss term added after rendering, per training step):
$$L_{flow} = \| \mathrm{sg}(F^{M}_{t \to t+1}) - F^{G}_{t \to t+1} \| \qquad (8)$$
$$L = L_{baseline} + \lambda \cdot L_{flow} \qquad (9)$$
where $\mathrm{sg}(\cdot)$ is the stop-gradient operator and $\lambda$ is a loss weight.

Camera pose refinement models a learnable residual on top of the COLMAP pose estimate: $T_{refined} = T + \Delta T$, where $\Delta T$ is a small SE(3) residual optimized while Gaussians are frozen.

#### Algorithm / Pipeline Changes

1. Initialize camera poses and sparse point cloud via COLMAP (unchanged from base Deformable-3DGS pipeline).
2. Per adjacent frame pair $(t, t+1)$, compute raw optical flow $F_{t\to t+1}$ with GMFlow (default) or an alternative flow network.
3. Render (or otherwise obtain) a depth map $D_t$ from the current Gaussian set; unproject pixels and reproject through consecutive camera poses to compute camera flow $F^{C}_{t\to t+1}$ (Eqs. 4-6).
4. Subtract camera flow from raw flow to get motion flow $F^{M}_{t\to t+1}$ (Eq. 7); apply a motion mask to suppress unreliable/static regions (ablation shows this mask is required — removing it drops PSNR by ~1 point).
5. Run the deformation network to predict each Gaussian's attributes at $t+1$; compute per-Gaussian 2D flow and alpha-blend to pixel-space Gaussian flow $F^{G}_{t\to t+1}$ inside the CUDA rasterizer (Eqs. 10-14).
6. Add stop-gradient flow loss $L_{flow}$ to the baseline photometric loss, weighted by $\lambda$ (Eqs. 8-9); backpropagate into the deformation network only.
7. Alternate optimization: optimize the full Gaussian/deformation model for one phase with camera poses frozen, then optimize the SE(3) pose residuals $\Delta T$ with Gaussians frozen (camera pose refinement module).
8. Train for 20,000 total iterations.

#### Key Hyperparameters & Design Choices

- Flow loss weight $\lambda$: 0.5 (NeRF-DS), 0.1 (HyperNeRF). Ablation over $\lambda \in \{0.2, 0.5, 0.8\}$ on NeRF-DS gives 23.46 / 24.12 / 23.75 PSNR respectively — 0.5 is optimal.
- Camera pose refinement learning rates: rotation 3e-3, translation 1e-1.
- Total training iterations: 20,000.
- Default optical flow network: GMFlow. Default depth source: rendered depth from the Gaussians (vs. monocular MiDaS depth, which underperforms).
- Resolution: 480×270 on NeRF-DS; 536×960 (2× downsampled) on HyperNeRF vrig.
- Deformation-network architecture dimensions: Not specified in the fetched summary (inherited from base Deformable-3DGS; not modified as a novel component by this paper).
- Hardware/training cost: single RTX 3090, 1-2 hours per NeRF-DS scene, 9.66-17.73 GB peak memory.

#### Ablation Summary

On NeRF-DS, most impactful component first:

1. **Motion flow guidance** (decoupled): 23.61 → 24.12 PSNR (+0.51 dB) — the single most impactful component, and the one that validates the core thesis (naive flow guidance alone is harmful).
2. **Motion mask**: removing it drops 24.12 → 23.13 PSNR (−0.99 dB), so the mask is nearly as load-bearing as the decoupling itself.
3. **Camera pose refinement** (added on top of motion flow guidance): 24.12 → 24.54 PSNR (+0.42 dB).
4. **Raw/undecoupled optical flow guidance** (negative control): 23.61 → 23.37 PSNR (−0.24 dB) — confirms naive flow supervision without camera-flow removal actively hurts.
5. Flow network choice: GMFlow (24.12, implicit default) > FlowFormer (23.97) > MDFlow (23.25); self-supervised flow loss variant: 23.76.
6. Depth source: rendered depth (default) > monocular MiDaS depth (23.58).

#### Implementation Reality

- **Framework:** PyTorch 1.13.1, extends the Deformable-3DGS repository with a custom CUDA rasterizer modified to emit per-pixel Gaussian flow, plus integrated GMFlow/MiDaS/MDFlow submodules.
- **Key files (per repo structure):** `core_flow/` (optical flow decoupling logic), `core_depth/` (depth estimation), `gaussian_renderer/` (rasterizer with Gaussian-flow computation), `gmflow/` (bundled GMFlow model), `scene/` (dataset loading, supports `nerfies` and `plenopticVideo` scene formats), `train.py` (main entry point with `--use_depth_and_flow` and `--optimize_pose` flags).
- **Notable implementation details:** the repo exposes `--scene_format plenopticVideo`, indicating built-in support for DyNeRF/Neural 3D Video (N3V)-style multi-camera datasets alongside the monocular NeRF-DS/HyperNeRF formats the paper's main results use — worth checking directly against the codebase before assuming the released code is monocular-only.

#### Failure Modes & Limitations

The paper states the method still relies on COLMAP for pose/point initialization and thus inherits COLMAP's need for "sufficient static features" in the scene, explicitly flagging pose-free 3DGS as future work. The ablation results themselves reveal a failure mode: undecoupled optical flow supervision is actively harmful (−0.24 dB vs. no flow guidance at all), so the method's benefit is contingent on accurate camera-flow removal, which in turn depends on depth and pose quality — degraded depth (monocular MiDaS vs. rendered) or degraded flow estimation (MDFlow) both measurably reduce quality.

---

## Relevance to ADAGS

Relevant to `idea:rendered-flow-gated-supervision`.

## Connections

## Sources

- https://arxiv.org/abs/2410.07707
- https://github.com/RuijieZhu94/MotionGS
