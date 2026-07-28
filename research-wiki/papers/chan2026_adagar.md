---
type: paper
node_id: paper:chan2026_adagar
title: "AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction"
authors: ["Jiewen Chan", "Zhenjun Zhao", "Yu-Lun Liu"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2601.00796"
tags: [dynamic-scene, gabor, high-frequency, temporal-continuity, representation]
status: deep-dived
---

# AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction

**Paper:** https://arxiv.org/abs/2601.00796
**Code:** https://github.com/JiewenChan/Adaptive_Gabor_Video_Representation
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023), implemented via the Pointrix framework, extended with per-Gaussian Gabor frequency modulation and cubic-Hermite-spline keyframe trajectories in place of a deformation MLP (contrast with Yang et al. 2024 deformation-field 3DGS and 4DGS/Wu et al. 2024).

## One-line thesis

Multiplying each Gaussian's spatial support by a learnable, energy-compensated sum of cosine modulations (an "Adaptive Gabor" term) recovers high-frequency appearance that plain Gaussian primitives low-pass away, while representing per-Gaussian motion as cubic Hermite splines over sparse keyframes (rather than a continuous deformation MLP) preserves temporal continuity without over-smoothing trajectories.

## Problem / Gap

Gaussian primitives are smooth radial basis functions, so a Gaussian-only scene representation acts as a low-pass filter on appearance: fine texture and sharp dynamic edges blur, especially from monocular video with fast motion. Naively replacing Gaussians with standard (uncompensated) Gabor functions destabilizes optimization because the sinusoidal modulation can drive local intensity/energy to zero or oscillate, and continuous deformation-MLP motion models (as in Yang et al. 2024, 4DGS) tend to smooth over per-instant trajectory detail rather than tracking discrete point-track keyframes directly.

## Method

Each primitive's Gaussian envelope is multiplied by a sum of learnable-amplitude cosine terms with fixed frequencies and learnable direction vectors, then passed through an adaptive energy-compensation term that renormalizes so the modulated primitive degrades smoothly to a standard Gaussian as the learned amplitudes go to zero. Per-Gaussian motion is represented as a sparse set of keyframe positions connected by cubic Hermite splines with a monotonicity-preserving ("auto-slope") tangent rule, and a curvature-regularization loss penalizes second-derivative trajectory acceleration to keep motion smooth between keyframes. Initialization samples canonical points from a probability field that combines inverse temporal-support and inverse local-density terms (i.e., favors long-lived, sparsely covered regions) plus grid-based uniform coverage and boundary-aware boosting near motion-mask edges, seeded from monocular depth (DPT) and point tracks (CoTracker). Training runs a 500-iteration warm-up then 10K main iterations, updating spline control points every 100 iterations, optimizing a combined RGB (L1+SSIM), optical-flow-consistency, scale-shift-invariant depth, and curvature loss.

## Assumptions

Single-camera (monocular) video input of a dynamic scene, with off-the-shelf monocular depth (DPT) and point tracking (CoTracker) available to seed initialization and supply flow supervision; assumes scene motion is representable as smooth per-point trajectories between sparse keyframes (not necessarily rigid, but not abrupt/discontinuous either) and that a foreground/motion mask is available to bias initialization near boundaries.

## Limitations / Failure Modes

The paper states spline-based motion modeling assumes smooth trajectories, causing misalignment under abrupt or highly nonlinear motion. It also reports that the Adaptive Gabor Representation can exhibit oscillations in high-frequency regions due to the energy-compensation constraints — i.e., the fix for low-pass blur can itself introduce ringing artifacts in some regions.

## Reusable Ingredients

- **Energy-compensated Gabor modulation of a Gaussian primitive** — adds recoverable high-frequency detail while provably degrading to a vanilla Gaussian at zero amplitude, avoiding the instability of raw Gabor functions.
- **Cubic Hermite spline motion with monotone auto-slope tangents** — keyframe-based per-point trajectories that resist overshoot/oscillation better than a plain cubic spline (ablation shows a large gap: 38.98 dB vs 32.42 dB PSNR).
- **Curvature regularization on trajectory second derivatives** — a lightweight temporal-smoothness prior independent of the interpolation scheme itself.
- **Density/support-aware adaptive initialization sampling** — biasing point spawn probability by inverse temporal support and inverse local density (plus boundary-mask boosting) as an alternative to uniform or random initialization; reported +6.78 dB over random init.

---

### Deep Dive

#### Core Novelty

Relative to standard 3DGS, AdaGaR changes two things: (1) it multiplies each Gaussian's spatial envelope by a learnable, multi-frequency cosine sum with an energy-compensation offset, so the primitive can represent local high-frequency structure while provably reducing to a plain Gaussian when the learned amplitudes vanish; and (2) it replaces continuous per-Gaussian deformation MLPs with sparse-keyframe cubic Hermite splines plus explicit curvature regularization, treating motion as an interpolation problem over a small set of learned control points rather than a globally-smooth field. The key insight is that both fixes target the same root cause — Gaussians/MLPs are smooth-by-construction, so the paper adds mechanisms (frequency modulation, spline keyframes) whose smoothness is explicit and tunable rather than implicit and unavoidable.

#### Mathematical Formulation

- Gabor modulation (per-Gaussian, evaluated before/alongside the Gaussian spatial falloff, per query point $x$ in the primitive's local frame):
$$S(x) = \sum_i \omega_i \cos(f_i \langle d_i, x\rangle)$$
where $\omega_i \in [0,1]$ are learnable amplitude weights, $f_i \in \{1,2\}$ are fixed frequencies, and $d_i$ are learnable direction vectors. This term multiplies the Gaussian support to modulate local intensity/alpha.

- Adaptive energy compensation (same evaluation point, applied to prevent the modulation from driving energy to zero):
$$S_{adap}(x) = b + \frac{1}{N}\sum_i \omega_i \cos(f_i \langle d_i, x\rangle), \qquad b = \gamma + (1-\gamma)\left(1 - \frac{1}{N}\sum_i \omega_i\right)$$
$\gamma \in [0,1]$ controls the degradation floor; as all $\omega_i \to 0$, $S_{adap}(x) \to 1$, i.e., the primitive reduces exactly to a standard Gaussian (proved in the paper's Appendix B).

- Cubic Hermite motion spline (evaluated when querying a Gaussian's position at continuous time $t$, between keyframes $k$ and $k+1$ at parametric position $s$):
$$\Delta(t) = H_{00}(s) y_k + H_{10}(s)\,\delta_k m_k + H_{01}(s) y_{k+1} + H_{11}(s)\,\delta_k m_{k+1}$$
with the auto-slope rule $m_k = \beta \cdot (\delta_{k-1}+\delta_k)/2$ if the neighboring segment slopes $\delta_{k-1}, \delta_k$ share sign, else $m_k = 0$ (this monotonicity guard suppresses overshoot/oscillation at keyframes).

- Curvature regularization loss (per-trajectory, added to the total loss):
$$\mathcal{L}_{curve} = \frac{\sum_k w_k \|y_k''\|_2^2}{\sum_k w_k D + \epsilon}$$
penalizing keyframe second-derivative (acceleration) magnitude, weighted by $w_k$ and normalized by dimensionality $D$.

- Adaptive initialization sampling probability (used once, at point-cloud initialization, to decide where new canonical points are spawned):
$$\Pi(p_i) \propto \frac{1}{\tau_i+\epsilon} + \lambda_t \cdot \frac{1}{\rho_i+\epsilon}$$
where $\tau_i$ is a point's temporal support (visibility duration) and $\rho_i$ is local spatial density; combined with grid-based uniform-coverage modulation and boundary-aware boosting near foreground-mask edges.

- Total training objective:
$$\mathcal{L}_{total} = \lambda_{rgb}\mathcal{L}_{rgb} + \lambda_{flow}\mathcal{L}_{flow} + \lambda_{depth}\mathcal{L}_{depth} + \lambda_{curv}\mathcal{L}_{curv}$$
where $\mathcal{L}_{rgb}$ is an L1+SSIM blend, $\mathcal{L}_{flow} = \sum_j w_j \|\hat{x}^{2j} - x^{2j}\|_1 / (\sum_j w_j + \epsilon)$ supervises projected motion against CoTracker point tracks, and $\mathcal{L}_{depth}$ is scale-shift-invariant alignment to DPT monocular depth.

#### Algorithm / Pipeline Changes

1. Seed a canonical 3D point cloud using monocular depth (DPT) and point tracks (CoTracker), sampling point positions according to the adaptive initialization probability $\Pi(p_i)$ (temporal-support-weighted, density-weighted, grid-modulated, boundary-boosted by a foreground/motion mask).
2. Attach to each canonical point a standard 3DGS parameter set (position, covariance, opacity, color/SH) plus new Gabor parameters: a set of amplitude weights $\omega_i$ (constrained to $[0,1]$ via a straight-through hard-sigmoid) and direction vectors $d_i$, with fixed frequencies $f_i \in \{1,2\}$.
3. At render time, evaluate $S_{adap}(x)$ per-Gaussian and multiply it into the Gaussian's spatial support/alpha before rasterization (described in the project page as "CUDA-style accumulation" that multiplies Gaussian support by the sinusoidal weights to update alpha).
4. Represent each Gaussian's position over time as a small set of learned keyframes; at any query time $t$, interpolate position via the cubic Hermite spline $\Delta(t)$ with auto-slope tangents, replacing what a deformation MLP would otherwise compute.
5. Run a 500-iteration warm-up, then 10K main-optimization iterations; update spline control points every 100 iterations (rather than every iteration), while other Gaussian parameters presumably update every step.
6. Optimize the combined RGB/flow/depth/curvature loss jointly; flow supervision comes from CoTracker point tracks, depth supervision from DPT with scale-shift-invariant alignment.
7. At inference, downstream applications (frame interpolation, depth consistency, video editing, stereo synthesis) query the same canonical Gaussian bank at arbitrary times/views via the spline motion and standard 3DGS rasterization.

#### Key Hyperparameters & Design Choices

- Gabor frequencies: $f_i \in \{1, 2\}$ (fixed, not learned).
- Amplitude weights: $\omega_i \in [0,1]$, parameterized via a straight-through hard-sigmoid.
- Degradation-floor parameter: $\gamma \in [0,1]$ (controls how far $S_{adap}$ can deviate from 1 as amplitudes grow).
- Spline smoothness coefficient: $\beta \in (0,1]$.
- Temporal/spatial init balance: $\lambda_t \in [0,1]$; grid modulation weight $\lambda_g > 0$; boundary-compensation weight $\lambda_b > 0$ (exact values not specified in paper).
- Loss weights $\lambda_{rgb}, \lambda_{flow}, \lambda_{depth}, \lambda_{curv}$: not specified in paper.
- Training schedule: 500-iteration warm-up + 10,000 main iterations; spline control points updated every 100 iterations.
- Hardware/runtime: single NVIDIA RTX 4090, ~90 minutes per video sequence.
- $N$ (number of Gabor frequency components summed per Gaussian): not specified in paper.

#### Ablation Summary

Most impactful single component: **Adaptive Gabor representation itself**, and **cubic Hermite spline motion**, are roughly tied as the largest contributors — both produce multi-dB jumps versus their respective naive alternatives.

- Adaptive Gabor vs. Gaussian baseline: **+0.77 dB PSNR** (36.66 → 37.43), SSIM +0.020, LPIPS −0.018 (0.0421 → 0.0242).
- Adaptive Gabor vs. standard (uncompensated) Gabor: +0.78 dB PSNR (36.65 → 37.43), confirming energy compensation — not just frequency modulation — drives the gain.
- Adaptive Gabor vs. naive "$1+S(x)$" variant: +0.93 dB PSNR (36.50 → 37.43).
- Cubic Hermite spline vs. standard cubic spline: **+6.56 dB PSNR** (32.42 → 38.98) — the largest single delta reported in the ablations, showing the monotonicity/auto-slope guard is critical, not cosmetic.
- Cubic Hermite spline vs. B-spline: +2.30 dB PSNR (36.68 → 38.98).
- Adaptive initialization vs. random initialization: **+6.78 dB PSNR**.
- Full method on Tap-Vid DAVIS: 6.86 dB PSNR improvement over the second-best baseline (from Splatter A Video, CoDeF, Omnimotion, Deformable Sprites, RoDynRF, 4DGS).

#### Implementation Reality

- **Framework:** PyTorch 2.4.0 + CUDA 11.8, built on the Pointrix point-based rendering framework (not a direct fork of the original gaussian-splatting repo), with PyTorch3D for 3D operations and custom CUDA extensions (`simple-knn`, `dptr`) compiled as submodules.
- **Key files/dirs:** `model/` (core Gabor + Gaussian primitive and spline implementations), `video3Dflow/` (optical-flow computation for temporal consistency), `data_preparation/` (depth/track/mask preprocessing for initialization), `loaders/`, `train.py`/`trainer.py` (training loop, configargparse-driven), `criterion.py`/`loss.py` (loss terms), `configs/` (per-sequence YAML/text configs, e.g. DAVIS "blackswan").
- **Notable implementation details:** the repo uses a two-level config hierarchy (a runtime config for dataset/iteration/checkpoint settings, separate from a `--gs_config_file` controlling point parameterization, renderer, optimizer, and initialization toggles) — this split is not described in the paper. Point tracking/nearest-neighbor operations rely on compiled `simple-knn` and `dptr` CUDA extensions rather than pure PyTorch, implying performance-critical Gabor/spline evaluation likely also runs through custom kernels rather than the paper's equations directly in Python.

#### Failure Modes & Limitations

The paper explicitly states: spline-based motion modeling assumes smooth trajectories, so it can misalign under abrupt or highly nonlinear motion. It also notes the Adaptive Gabor Representation may exhibit oscillations in high-frequency regions because of the energy-compensation constraints — the mechanism designed to fix Gaussian low-pass blur can itself introduce ringing under certain frequency/amplitude combinations.

---

## Relevance to ADAGS

This paper makes "dynamic sharpness" less of a loss-design issue and more of a representation-capacity issue. ADAGS should measure edge/detail preservation in dynamic masks and justify whether LoRA/scaffold residuals can overcome Gaussian low-pass behavior.

## Connections

- Addresses [[gap_map#G1 - Dynamic-Region Sharpness Needs A Direct Objective]]
- Addresses [[gap_map#G11 - Representation Frequency Is A New Sharpness Axis]]
- Pressures [[ideas/dynamic-region-diagnostic-benchmark]]

## Sources

- https://arxiv.org/abs/2601.00796
