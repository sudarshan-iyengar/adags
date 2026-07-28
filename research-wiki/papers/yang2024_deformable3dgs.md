---
type: paper
node_id: paper:yang2024_deformable3dgs
title: "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction"
authors: ["Ziyi Yang", "Xinyu Gao", "Wen Zhou", "Shaohui Jiao", "Yuqing Zhang", "Xiaogang Jin"]
year: 2024
venue: "CVPR"
external_ids:
  arxiv: "2309.13101"
tags: [dynamic-gs, deformation, monocular]
status: deep-dived
---

# Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction

**Paper:** https://arxiv.org/abs/2309.13101
**Code:** https://github.com/ingra14m/Deformable-3D-Gaussians
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023) + a canonical-space/deformation-field decomposition

## One-line thesis

Keeping a single time-independent canonical set of 3D Gaussians and routing all temporal change through a small per-Gaussian MLP that predicts `(δposition, δrotation, δscale)` as a function of `(position, time)` lets Gaussian splatting reach real-time, high-detail monocular dynamic reconstruction, and adding time-dependent noise to the time encoding during early training (annealed to zero) absorbs COLMAP pose-estimation error on real captures without a separate pose-refinement step.

## Problem / Gap

Prior dynamic reconstruction is split between implicit neural representations (NeRF-family), which capture detail poorly and cannot render in real time, and grid/plane-based acceleration methods, which impose low-rank structure that is a poor fit for genuinely dynamic content and remain bottlenecked by ray-casting at high resolution. Neither line combines explicit, detail-preserving geometry with real-time rendering for monocular (single-camera-per-timestep) capture, and real-world monocular data additionally has COLMAP pose noise that naive per-frame or per-Gaussian-time-conditioning tends to overfit to.

## Method

3D Gaussians are optimized in a single canonical space (as in static 3DGS) rather than being re-estimated per frame. A deformation network `F_θ` takes the stop-gradiented canonical position and the current timestamp (both positionally encoded) and outputs per-Gaussian offsets to position, rotation, and scale; these deformed Gaussians are what get rasterized for that frame. Training starts with a 3k-iteration warm-up that optimizes only the canonical 3DGS (deformation network inactive), after which the deformation network is jointly trained with the Gaussians. During this joint phase, an annealing smoothing mechanism injects Gaussian noise into the time encoding, with the noise magnitude decaying linearly to zero over the first 20k of 40k total iterations, to prevent the network from overfitting to per-frame pose noise before local structure is established.

## Assumptions

Monocular video (one camera per timestep, not multi-view-per-timestep) with either exact synthetic poses (D-NeRF) or COLMAP-estimated poses (NeRF-DS, HyperNeRF). Assumes deformation from a shared canonical geometry is adequate, i.e., no topology changes (objects appearing/disappearing) and motion that is smooth enough for a coordinate-MLP to represent well.

## Limitations / Failure Modes

The paper itself reports: (1) reconstruction quality is sensitive to viewpoint diversity — sparse/narrow view coverage causes overfitting of the canonical Gaussians; (2) quality is bounded by COLMAP pose accuracy, since the method has no independent pose-correction mechanism beyond AST; (3) training/memory cost scales with Gaussian count, so large scenes become expensive; (4) the method is evaluated on moderate-motion scenes and the authors note fine, nuanced motion (e.g., facial expressions) is not established to work well.

## Reusable Ingredients

- **Canonical-space + coordinate-MLP deformation**: separates "what a Gaussian looks like" from "where/how it sits at time t," so geometry and appearance are learned once and only deformed — reusable framing for any route that wants a stable base representation plus a lightweight temporal head.
- **Stop-gradient from deformation output back to canonical position**: prevents the deformation network from reshaping canonical geometry itself, keeping canonical space as a stable anchor.
- **Annealed noise injection on the time encoding (AST)**: a cheap, architecture-free way to make a temporal MLP robust to noisy/inaccurate per-frame conditioning (e.g., noisy pose or noisy correspondence) early in training, without a dedicated calibration or refinement module.
- **Warm-up-then-joint training schedule**: stabilizing the base representation before activating the temporal/deformation component reduces the chance of the temporal head compensating for a still-bad base geometry.

---

### Deep Dive

#### Core Novelty
Relative to static 3DGS, the paper adds (a) a canonical-to-time deformation MLP so a single Gaussian set can represent an entire dynamic sequence instead of needing per-frame optimization, and (b) an annealed noise-injection scheme (AST) on the time input specifically to counteract pose-estimation error in real monocular captures. The key insight is that decoupling "what" (canonical Gaussian attributes) from "when" (a small conditioned offset network) lets the explicit, rasterization-friendly representation of 3DGS extend to dynamic scenes without abandoning real-time rendering, while AST treats temporal-pose noise as a training-time regularization problem rather than a geometry problem.

#### Mathematical Formulation

Positional encoding applied to both spatial coordinates and time before the deformation MLP:
$$\gamma(p) = \left(\sin(2^k \pi p), \cos(2^k \pi p)\right)_{k=0}^{L-1}$$
with $L=10$ for position and $L=6$ for time on synthetic (D-NeRF) scenes, and $L=10$ for both on real scenes. Evaluated once per Gaussian per frame, immediately before the deformation MLP.

Deformation network output — per-Gaussian offsets to position, rotation, and scale:
$$(\delta\mathbf{x}, \delta\mathbf{r}, \delta\mathbf{s}) = \mathcal{F}_\theta\big(\gamma(\text{sg}(\mathbf{x})), \gamma(t)\big)$$
where $\text{sg}(\cdot)$ is a stop-gradient on the canonical position $\mathbf{x}$ (so the deformation loss does not update canonical geometry through this path), and $t$ is the frame timestamp. Evaluated per-Gaussian, per-frame, before rasterization; the deformed $(\mathbf{x}+\delta\mathbf{x}, \mathbf{r}+\delta\mathbf{r}, \mathbf{s}+\delta\mathbf{s})$ is what gets rasterized.

Annealing smooth training (AST) noise on the time encoding:
$$\mathcal{X}(i) = \mathcal{N}(0,1) \cdot \beta \cdot \Delta t \cdot \left(1 - \frac{i}{\tau}\right)$$
added as $\gamma(t) + \mathcal{X}(i)$ fed into the deformation MLP, where $i$ is the current training iteration, $\beta = 0.1$, $\Delta t$ is the mean time interval between frames, and $\tau = 20\text{k}$ iterations (noise reaches zero at iteration $\tau$). Evaluated during the joint-training phase only (after the 3k-iteration warm-up), as part of computing the deformation MLP's time input each iteration.

Combined image loss (standard, not itself novel, included for reproduction):
$$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}, \quad \lambda = 0.2$$

#### Algorithm / Pipeline Changes
1. Initialize 3D Gaussians in canonical space exactly as in static 3DGS (from SfM point cloud).
2. Train canonical Gaussians alone for a 3k-iteration warm-up (deformation network not yet active) — stabilizes base geometry/appearance before temporal conditioning is introduced.
3. After warm-up, for every training iteration and every Gaussian: compute $\gamma(\text{sg}(\mathbf{x}))$ and $\gamma(t)$ (with AST noise added to $\gamma(t)$ if $i < \tau$), pass through the deformation MLP $\mathcal{F}_\theta$ (8 fully-connected layers, ReLU, 256-dim hidden, 3 separate output heads for $\delta\mathbf{x}, \delta\mathbf{r}, \delta\mathbf{s}$).
4. Apply the offsets to the canonical Gaussian parameters to get the frame-specific Gaussian set, then rasterize using the standard differentiable 3DGS rasterizer — this step replaces "load the per-frame Gaussians" with "compute per-frame Gaussians from canonical + deformation."
5. Backpropagate the photometric loss jointly through both the deformation MLP and the (non-stop-gradiented) canonical Gaussian attributes.
6. Standard 3DGS adaptive density control (clone/split/prune) continues to run on the canonical Gaussians, with a density-control gradient threshold of $t_{pos}=0.0002$ and a scale-split divisor $\xi=1.6$.

#### Key Hyperparameters & Design Choices
- Total training iterations: 40k (D-NeRF synthetic); real-world (NeRF-DS/HyperNeRF) uses 20k per the released code.
- Warm-up (canonical-only) phase: 3k iterations.
- Deformation MLP: depth $D=8$, hidden width $W=256$, ReLU activations, 3 separate linear output heads for $\delta\mathbf{x}$, $\delta\mathbf{r}$, $\delta\mathbf{s}$.
- Positional encoding: $L=10$ (position) / $L=6$ (time) for synthetic scenes; $L=10$/$L=10$ for real scenes.
- Deformation network learning rate: exponential decay from $8\times10^{-4}$ to $1.6\times10^{-6}$.
- 3D Gaussian attribute learning rates: official 3D-GS defaults (unchanged).
- Adam optimizer $\beta = (0.9, 0.999)$.
- AST noise: $\beta = 0.1$, anneal horizon $\tau = 20\text{k}$ iterations.
- Photometric loss weight $\lambda = 0.2$ (D-SSIM term).
- Density-control gradient threshold $t_{pos} = 0.0002$; scale-split divisor $\xi = 1.6$.
- Additional storage overhead of the deformation network: ~2MB.
- Optional 6DoF rigid-transform mode (`--is_6dof` in the released code) as an alternative deformation parameterization; not detailed further in the paper text extracted.

#### Ablation Summary
D-NeRF synthetic dataset, full method vs. ablations (PSNR / SSIM / LPIPS):
- **w/o AST**: 39.42 / 0.9875 / 0.0247 vs. full method 39.51 / 0.9902 / 0.0124 — AST's largest effect is on LPIPS (perceptual detail), roughly halving it, despite a similar PSNR.
- **w/o δs (scale offset)**: 40.39 / 0.9833 / 0.0323 — removing scale deformation raises PSNR slightly but visibly worsens LPIPS/SSIM, suggesting scale offsets primarily help perceptual/structural fidelity rather than pixel-level error.
- **w/o δr (rotation offset)**: 40.58 / 0.9839 / 0.0278 — similar pattern to w/o δs.
- On the real-world NeRF-DS dataset, AST improves PSNR from 23.97 → 24.11.
- Most impactful component by LPIPS: **AST** (annealing smooth training) — it is the only ablation that changes the ranking on the perceptual metric, not just PSNR, indicating it is load-bearing for detail quality specifically (consistent with its stated purpose of preventing early overfitting to pose noise).

#### Implementation Reality
- **Framework:** PyTorch (`torch==1.13.1+cu116`), extending the official 3D Gaussian Splatting CUDA rasterizer as a submodule.
- **Key files:** `train.py` (main training loop, supports both D-NeRF-style synthetic and real-world datasets via `--is_blender`/dataset flags), `render.py` (inference with render / time-interpolation / view-synthesis modes), `metrics.py` (PSNR/SSIM/LPIPS evaluation), `scene/` (scene and deformation-field management), `gaussian_renderer/` (rasterization pipeline), `lpipsPyTorch/` (perceptual loss, noted in the README as swapped to a faster `lpips` implementation than originally used).
- **Notable implementation details:** the repo exposes an `--is_6dof` flag for an alternative rigid 6DoF deformation parameterization not elaborated on in the paper excerpt available; real-world training defaults to `--iterations 20000` versus 40k for synthetic scenes, a discrepancy not explicitly called out as such in the paper text extracted. A separate official "Lightweight-Deformable-GS" variant in the same author's account claims 50% storage reduction and 200% FPS increase at comparable quality, indicating the released deformation MLP is not minimal/optimized by default.

#### Failure Modes & Limitations
The paper identifies: convergence is "profoundly influenced by the diversity of perspectives," so narrow or sparse-view monocular capture overfits the canonical Gaussians; quality is bounded by upstream pose-estimation (COLMAP) accuracy since AST mitigates but does not eliminate pose-error sensitivity; per-Gaussian temporal deformation means training time and memory scale with Gaussian count, making large or long scenes expensive; and evaluation focuses on moderate-magnitude motion, with the authors explicitly flagging that performance on subtle/nuanced motion such as facial expressions is unresolved.

---

## Relevance to ADAGS

Direct baseline class for LoRA/scaffold motion. ADAGS should show why its reversible route0 plus priors is better for fast dynamic regions.

## Connections


## Sources

- https://arxiv.org/abs/2309.13101
- https://github.com/ingra14m/Deformable-3D-Gaussians
