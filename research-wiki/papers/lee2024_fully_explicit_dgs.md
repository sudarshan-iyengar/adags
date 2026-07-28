---
type: paper
node_id: paper:lee2024_fully_explicit_dgs
title: "Fully Explicit Dynamic Gaussian Splatting"
authors: ["Jongmin Lee", "Daeseung Lee", "Dogyoon Lee", "Junhyeop Lee", "Sangyoon Son", "Kwanghoon Sohn"]
year: 2024
venue: "NeurIPS"
external_ids:
  arxiv: "2410.15629"
tags: [dynamic-gs, explicit-motion, baseline]
status: deep-dived
---

# Fully Explicit Dynamic Gaussian Splatting

**Paper:** https://arxiv.org/abs/2410.15629
**Code:** https://github.com/juno181/Ex4DGS
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023) + Mip-Splatting codebase, extended to dynamic scenes without a deformation MLP.

## One-line thesis

Sparsely sampling explicit per-Gaussian position/rotation keyframes and reconstructing continuous motion via cubic Hermite / Slerp interpolation lets dynamic Gaussians converge robustly from sparse (first-frame-only) point clouds, avoiding the dense-initialization and implicit-warp-field dependence of prior 4D Gaussian methods.

## Problem / Gap

Implicit deformation-field methods (4DGaussians, 4D-GS/STG-style approaches) model motion as continuous neural warps, which are slow to render or require dense multi-frame point clouds to initialize reliably; when given only sparse (first-frame) COLMAP points, 4DGaussians drops 5.52 dB PSNR and 4DGS drops 1.82 dB versus their dense-initialization results. NeRF-based dynamic methods (NeRFPlayer, HyperReel) retain high quality but render at ~0.05-2 fps, far below real-time. The paper targets a fully explicit, keyframe-based representation that stays robust under sparse initialization while preserving 3DGS-class rendering speed.

## Method

Gaussians are first split into static and dynamic subsets by a motion-magnitude heuristic; static Gaussians get a linear position model over time, dynamic Gaussians are explicitly parameterized only at sparse keyframe timestamps (interval `I`) and interpolated between them with a cubic Hermite spline (position) and Slerp (rotation/quaternion). A two-sided Gaussian temporal-opacity function fades each dynamic Gaussian in/out around its active window, letting objects appear/disappear without persisting everywhere. Training is progressive: the model starts with a short temporal window (10 frames) and grows it every 400 iterations, using linear regression over recent frames to initialize newly exposed keyframes, which avoids local minima from optimizing all keyframes at once against sparse geometry. A point-backtracking step accumulates per-Gaussian rendering error weighted by visibility (transmittance) across training views and prunes Gaussians whose accumulated error exceeds the dataset-average error.

## Assumptions

Multi-view (calibrated, synchronized) video capture, per the Neural 3D Video and Technicolor dataset setups; the method assumes point clouds can be sparse (as sparse as a single first-frame COLMAP reconstruction) but still requires camera calibration across views. It assumes motion is smooth enough between keyframes for spline interpolation and that static/dynamic separation from a simple screen-space motion metric is sufficient without semantic supervision.

## Limitations / Failure Modes

Newly appearing objects with no 3D points and no relevant Gaussians in neighboring frames "can get stuck in local minima" since nothing exists to seed their geometry; the paper suggests additional depth priors as a fix. In monocular video, "every 3D Gaussian is treated as dynamic due to the lack of accurate geometric clues," i.e., the static/dynamic split heuristic breaks down without multi-view triangulation, and the paper suggests semantic cues (masks, optical flow) as a remedy it does not implement.

## Reusable Ingredients

- **Motion-magnitude static/dynamic split** (`∥d∥/∥λ∥²` thresholded at top-η%): unsupervised way to segment dynamic Gaussians from screen-space motion without masks.
- **Cubic Hermite keyframe interpolation for position**: continuous motion from sparse per-Gaussian keyframes instead of a per-frame or MLP-warped field.
- **Two-sided Gaussian temporal opacity**: cheap, closed-form appear/disappear modeling without a learned lifespan network.
- **Progressive temporal-window training with linear-regression keyframe init**: mitigates local minima when optimizing against sparse geometry by growing the problem gradually.
- **Visibility-weighted accumulated-error point backtracking**: a pruning criterion based on transmittance-weighted rendering error accumulated over all training views/frames, not just opacity/size.

---

### Deep Dive

#### Core Novelty

Relative to implicit deformation-field 4D Gaussian methods, this paper replaces the continuous neural warp with fully explicit, sparsely-sampled per-Gaussian keyframe attributes (position, rotation) plus closed-form spline interpolation between them. The key insight: because the representation only needs to fit values at sparse keyframes (not a smooth field everywhere), it is far less sensitive to sparse/incomplete point initialization, and because interpolation is a fixed closed-form function (not a learned MLP), rendering speed matches static 3DGS.

#### Mathematical Formulation

Static Gaussian position, linear over normalized time (evaluated per-Gaussian before rasterization, every frame):
$$\mu(t) = x + t' \cdot d, \quad t' = t/l \in [0,1]$$
where $x$ is the pivot position, $d$ a learned translation vector, $l$ the scene duration.

Dynamic Gaussian position via cubic Hermite interpolation (CHip), evaluated per-Gaussian per-frame between its two bracketing keyframes:
$$\text{CHip}(p_0, m_0, p_1, m_1; t) = (2t^3-3t^2+1)p_0 + (t^3-2t^2+t)m_0 + (-2t^3+3t^2)p_1 + (t^3-t^2)m_1$$
$$\mu(t) = \text{CHip}(p_n, m_n, p_{n+1}, m_{n+1}; t'), \quad n = \lfloor t/I \rfloor,\ t' = (t-nI)/I$$
$$m_n = \frac{p_{n+1}-p_{n-1}}{2I}, \quad m_{n+1} = \frac{p_{n+2}-p_n}{2I}$$
Here $p_n$ are the learned keyframe positions, $I$ the keyframe interval, and $m_n$ the tangents estimated from neighboring keyframes (finite-difference/Catmull-Rom style).

Dynamic Gaussian rotation via Slerp between keyframe quaternions (same pipeline stage):
$$\text{Slerp}(x_0,x_1;t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}x_0 + \frac{\sin(t\Omega)}{\sin\Omega}x_1, \quad \cos\Omega = x_0\cdot x_1$$
$$q(t) = \text{Slerp}(r_n, r_{n+1}; t')$$

Temporal opacity (two-sided Gaussian window), evaluated per-Gaussian per-frame and multiplied into the standard 3DGS opacity before rasterization:
$$\sigma_t(t) = \begin{cases} \exp(-(t-a^o_s)^2/(b^o_s)^2) & t < a^o_s \\ 1 & a^o_s \le t \le a^o_f \\ \exp(-(t-a^o_f)^2/(b^o_f)^2) & t > a^o_f \end{cases}$$
$a^o_s, a^o_f$ are learned onset/offset times, $b^o_s, b^o_f$ learned fade widths.

Static/dynamic classification metric (computed once from initial motion cues, before optimization or early in training):
$$\text{motion\_metric} = \lVert d \rVert / \lVert \lambda \rVert^2$$
$\lambda$ is the point's distance to camera; the top $\eta\%$ (empirically $\eta=2$) by this metric are instantiated as dynamic Gaussians, the rest static.

Point-backtracking accumulated error (computed post-hoc from training-view renders, used to decide pruning, not part of the forward/loss graph):
$$\mathcal{E} = \frac{\sum_k \left[\sigma_i \prod_{j=1}^{i-1}(1-\sigma_j)\, q_k\right]}{\sum_k \left[\sigma_i \prod_{j=1}^{i-1}(1-\sigma_j)\right]}, \qquad \mathcal{E}_{\text{total}} = \frac{1}{|\mathcal{D}|}\sum_{v\in\mathcal{D}} \mathcal{E}_v$$
$q_k$ is a per-pixel rendering error term, the product term is the standard alpha-compositing transmittance up to Gaussian $i$; Gaussians with $\mathcal{E}$ above $\mathcal{E}_{\text{total}}$ are pruned.

#### Algorithm / Pipeline Changes

1. Initialize Gaussians from (possibly sparse, e.g. first-frame-only) COLMAP points, as in static 3DGS.
2. Compute the motion metric per point and convert the top 2% to dynamic Gaussians (keyframed {p_n, r_n, σ_t}); the rest remain static with linear {x, d}.
3. Begin training with a short active temporal window (10 frames); every 400 iterations extend the window by one keyframe interval `I`, initializing new keyframe positions/rotations via linear regression over the previous `ρ` frames rather than from scratch.
4. At each training/render step, evaluate μ(t) and q(t) for dynamic Gaussians via CHip/Slerp between the two bracketing keyframes (replaces a deformation-MLP forward pass used in prior methods), evaluate the temporal opacity window, then feed standard 3DGS rasterization (alpha-blending, Eq. for C and T unchanged from 3DGS/NeRF volume rendering).
5. Periodically (schedule not specified beyond being part of the standard densification/pruning cadence) run point backtracking: accumulate visibility-weighted error per Gaussian across all training views/frames and prune Gaussians above the mean accumulated error, augmenting (not replacing) standard opacity/size-based pruning.
6. Train with a regularization loss (weight λ) on the novel dynamic parameters in addition to standard photometric losses.

#### Key Hyperparameters & Design Choices

- Keyframe interval `I`: 10 (frames).
- Initial progressive-training window: 10 frames; grows by `I` every 400 iterations.
- Dynamic-conversion percentile `η`: 2%.
- Regularization weight `λ`: 0.0001.
- Optimizer: RAdam (inherited from 3DGS).
- Training resolution: half original resolution for N3V; N3V output evaluated at 1352×1014; Technicolor at full 2048×1088.
- Number of frames used for linear-regression keyframe init (`ρ`): Not specified in paper.
- Point-backtracking run frequency/schedule: Not specified in paper.

#### Ablation Summary (Table 3, Cook Spinach scene, PSNR)

- **Dynamic point extraction is the single most impactful component**: removing it costs 3.53 dB (28.58 vs. full 32.11 dB PSNR) and roughly halves model size (58MB vs. 115MB), i.e. the static/dynamic split is what lets the model spend capacity on motion rather than treating everything as static or everything as free-floating dynamic.
- Cubic Hermite position vs. linear position: −0.99 dB (31.12 vs. 32.11).
- Cubic Hermite+Slerp vs. linear position&rotation: −0.79 dB (31.32 vs. 32.11).
- Linear rotation (Slerp removed) vs. full: −0.85 dB (31.26 vs. 32.11).
- No progressive growing: −1.09 dB (31.02 vs. 32.11), the second-largest drop, confirming progressive training's role in avoiding local minima under sparse init.
- No temporal opacity: −0.69 dB (31.42 vs. 32.11) but larger model (186MB vs. 115MB), i.e. temporal opacity also acts as a compaction mechanism.
- No point backtracking: −0.71 dB (31.40 vs. 32.11).
- No regularization: −0.74 dB (31.37 vs. 32.11).

#### Implementation Reality

- **Framework:** PyTorch (2.1.2 tested), CUDA-accelerated rasterization; codebase explicitly built on top of the 3DGS and Mip-Splatting repos ("uses almost its hyperparameters").
- **Key files:** `train.py` (training loop), `render.py` (evaluation/rendering), `convert.py` (dataset conversion), core dynamic-Gaussian and interpolation logic under `gaussian_renderer/` and `scene/`, with dataset-specific settings under `configs/`. Exact file-level breakdown of CHip/Slerp/temporal-opacity/backtracking code was not resolved beyond directory structure — the repo listing did not surface line-level detail.
- **Notable implementation details:** None confirmed beyond the paper; a targeted code read (not performed here) would be needed to confirm whether config-file hyperparameters (e.g. `ρ`, backtracking schedule) match the paper's stated values or add undocumented defaults.

#### Failure Modes & Limitations

Newly appearing objects with no prior 3D points and no nearby-frame Gaussians can get stuck in local minima since there is no geometry to seed them from; the paper proposes additional depth priors as an unimplemented fix. In monocular capture, the motion-metric-based static/dynamic split degrades because "every 3D Gaussian is treated as dynamic due to the lack of accurate geometric clues" — i.e., without multi-view triangulation the heuristic can't distinguish real motion from monocular depth/parallax ambiguity, and the paper suggests (without implementing) semantic masks or optical flow as a remedy.

## Relevance to ADAGS

ADAGS LoRA/scaffold motion should be compared as another explicit-ish low-rank temporal representation.

## Connections

## Sources

- https://arxiv.org/abs/2410.15629
- https://arxiv.org/html/2410.15629
- https://github.com/juno181/Ex4DGS
