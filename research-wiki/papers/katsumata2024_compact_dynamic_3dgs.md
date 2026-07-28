---
type: paper
node_id: paper:katsumata2024_compact_dynamic_3dgs
title: "A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis"
authors: ["Kai Katsumata", "Duc Minh Vo", "Hideki Nakayama"]
year: 2024
venue: "ECCV"
external_ids:
  arxiv: "2311.12897"
tags: [dynamic-gs, compact, efficiency]
status: deep-dived
---

# A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis

**Paper:** https://arxiv.org/abs/2311.12897
**Code:** https://github.com/raven38/EfficientDynamic3DGaussian
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023) — forks the official `graphdeco-inria/gaussian-splatting` repo directly.

## One-line thesis

Per-Gaussian position and rotation over time can be replaced with a short Fourier series (position) and a linear function (rotation) of a handful of coefficients, cutting dynamic-3DGS storage from O(TN) (one full parameter set per frame) to O(LN) with L≪T while keeping scale, opacity, and color time-invariant.

## Problem / Gap

Naive dynamic extensions of 3DGS (e.g. per-frame Gaussian clouds as in Dynamic 3D Gaussians) store an independent set of position/rotation/scale/opacity parameters for every timestep, so memory grows linearly with sequence length (they report 6.6GB for Dynamic3DGaussians on DyNeRF) and training/rendering throughput collapses under that volume. NeRF-based dynamic methods (K-Planes, V4D) avoid the per-frame blowup via implicit fields but pay for it in rendering speed (0.54–1.23 FPS) because they still require per-ray network evaluation. Neither line gets both compactness and real-time speed simultaneously.

## Method

Each Gaussian's scale, color (SH coefficients), and opacity stay constant across time, exactly as in static 3DGS. Only position and rotation are made time-varying, but instead of storing per-frame values they are parameterized as closed-form functions of a continuous time variable t: position as a truncated Fourier series with L harmonics per axis, rotation (quaternion) as a first-order (linear) function of t per component. Training runs in two stages — a 3,000-iteration static stage that fits the time-invariant properties plus the t=0 intercepts of position/rotation (treating all frames as one static reference), followed by a 27,000-iteration dynamic stage that optimizes the harmonic/linear coefficients jointly with standard 3DGS densification/pruning. An auxiliary loss supervises the Gaussians' projected 3D scene flow against RAFT optical flow to keep the trajectories physically consistent between frames.

## Assumptions

Assumes a fixed-topology scene where every Gaussian persists across the entire captured time window (no birth/death of geometry), multi-view or monocular video with either known camera poses/SfM points (real scenes) or synthetic random initialization, and that motion is smooth enough over the sequence to be well-approximated by a handful of low-frequency Fourier terms.

## Limitations / Failure Modes

The paper explicitly states topology changes (Gaussians appearing/disappearing, e.g. fluid) are "tough" to model since every Gaussian is assumed to live for the whole sequence. Very long sequences degrade rendering quality because a fixed, small L cannot capture accumulating high-frequency motion over many frames. The representation also "sacrifices the continuity and smoothness of neural field-based volume rendering," costing some generalization performance relative to NeRF-based methods, and quality drops when camera poses are inaccurate. The flow-loss term is dropped entirely on D-NeRF because camera teleportation between frames makes ground-truth optical flow unreliable there.

## Reusable Ingredients

- **Closed-form low-frequency time parameterization** (Fourier for position, linear for rotation) — replaces per-frame storage with O(L) coefficients per Gaussian, directly reusable as a compact alternative to deformation MLPs or per-frame duplication.
- **Two-stage static-then-dynamic training schedule** — first fit a static reference model, then unlock temporal coefficients; stabilizes optimization before temporal degrees of freedom are introduced.
- **3D scene-flow-to-2D-optical-flow supervision** — differencing Gaussian positions at t and t±Δt, projecting through the camera Jacobian, and matching against RAFT flow; a cheap way to inject temporal-consistency signal without per-frame correspondence annotation.
- **Parameter-count vs. quality ablation protocol** (varying L, comparing polynomial/spline alternatives) as a template for compactness-vs-fidelity tradeoff studies.

---

### Deep Dive

#### Core Novelty

Relative to vanilla 3DGS (static-only) and to per-frame dynamic extensions, this paper's change is narrow and specific: keep every Gaussian's scale/color/opacity fixed for all time, and replace the "one position+rotation per frame" storage with a small set of global basis coefficients evaluated at query time t. The insight is that most per-Gaussian motion in these capture setups is low-frequency (smooth trajectories), so a handful of Fourier terms suffice for position and even a single linear term suffices for rotation — turning an O(T) storage problem per Gaussian into an O(L) one while the rendering path (rasterization, alpha compositing) is otherwise untouched 3DGS.

#### Mathematical Formulation

Position as a function of time, evaluated per-Gaussian before rasterization at each queried frame t, for each axis x (analogous for y, z):
$$x(t) = w_{x,0} + \sum_{i=1}^{L} \left[ w_{x,2i-1}\sin(2i\pi t) + w_{x,2i}\cos(2i\pi t) \right]$$
Here $w_{x,\cdot}$ are learned per-Gaussian scalar coefficients and $L$ is the (dataset-level) number of harmonic terms. This gives $3(2L+1)$ learned parameters per Gaussian for position (3 axes).

Rotation (quaternion component $q_x$; analogous for $q_y, q_z, q_w$), evaluated at the same pipeline point:
$$q_x(t) = w_{qx,0} + w_{qx,1}\cdot t$$
i.e. a first-order Taylor/linear model, 2 parameters per quaternion component, 8 total.

Covariance at time t, computed per-Gaussian immediately before projection:
$$\Sigma(t) = R(t)\,S S^{T}\,R(t)^{T}$$
where $S$ is the (time-invariant) scaling matrix and $R(t)$ is the rotation matrix built from the time-varying quaternion $q(t)$ above.

2D projected covariance (standard 3DGS splatting step, applied after $\Sigma(t)$ is formed):
$$\Sigma'(t) = J\,W\,\Sigma(t)\,W^{T}\,J^{T}$$
with $J$ the Jacobian of the projective transform and $W$ the viewing transform.

Scene-flow-based temporal consistency loss, computed after rendering flow from position differences:
$$\hat f^{x}_{\text{fwd}} = x(t+\Delta t) - x(t), \qquad \hat f^{x}_{\text{bwd}} = x(t) - x(t-\Delta t)$$
these 3D flow estimates are projected to 2D through the same Jacobian used for covariance projection, composited via alpha-blending into a rendered optical flow map, and supervised against RAFT-estimated ground-truth flow $F$:
$$L = L_{\text{recon}} + \lambda_{\text{flow}} L_{\text{flow}}(\hat F, F), \qquad L_{\text{recon}} = (1-\lambda)|\hat I - I| + \lambda\, L_{\text{D-SSIM}}$$
$\lambda$ is the standard 3DGS SSIM weight (0.2, unchanged from Kerbl et al.); $\lambda_{\text{flow}}$ is the new flow-loss weight.

Per-Gaussian parameter count under this scheme: $3L + 8 + 3 + 3(k+1)^2 + 1$ (Fourier position + linear rotation + scale + degree-$k$ SH coefficients + opacity), versus $O(T)$ per Gaussian in per-frame storage.

#### Algorithm / Pipeline Changes

1. **Initialization**: real scenes use SfM sparse point cloud (as in standard 3DGS); synthetic scenes use random uniform point initialization.
2. **Static stage (iterations 0–3,000)**: optimize time-invariant Gaussian properties (scale, SH color, opacity) plus only the $t{=}0$ intercept terms $w_{x,0}, w_{qx,0}$, etc., treating the whole capture as a single static frame. This replaces/precedes the standard 3DGS optimization loop, giving a stable geometric starting point before temporal DOFs are unlocked.
3. **Dynamic stage (iterations 3,000–30,000)**: unlock and jointly optimize all harmonic/linear coefficients ($w_{x,1..2L}$, $w_{qx,1}$, etc.) together with the already-unlocked static parameters. Standard 3DGS densification (clone/split on gradient thresholds) and opacity/size pruning continue to run in this stage, now applied to Gaussians whose position/rotation are themselves time-functions rather than fixed points.
4. **Flow supervision**: at each training step, compute the 3D forward/backward position deltas per Gaussian, project to 2D via the projective Jacobian, composite into a rendered flow image, and add the RAFT-supervised flow loss to the reconstruction loss (Sintel-pretrained RAFT provides pseudo-ground-truth flow). This term is omitted for D-NeRF because its camera teleportation between frames breaks the flow ground truth.
5. **Rendering**: for a query time t, evaluate $x(t), q(t)$ per Gaussian in closed form, form $\Sigma(t)$, then run the unmodified 3DGS rasterizer/alpha-compositing pipeline.

#### Key Hyperparameters & Design Choices

- $L$ (Fourier harmonics): 2 for D-NeRF, 5 for DyNeRF/HyperNeRF (dataset-dependent).
- $\lambda$ (SSIM weight in reconstruction loss): 0.2, inherited unchanged from 3DGS.
- $\lambda_{\text{flow}}$ (flow loss weight): 1,000.
- Opacity pruning threshold: 0.005.
- Total iterations: 30,000 (3,000 static + 27,000 dynamic).
- Optical flow source: RAFT, Sintel-pretrained.
- Rotation model order: fixed at linear (order 1) for all datasets — not swept as a hyperparameter in the same way L is.
- Position/rotation/scale/opacity learning rates and densification start/end iterations: not stated in the paper text extracted here; the public code shows position LR starting at 0.00016 decaying to 0.0000016 and densification running from iteration 500 to 15,000, but these are implementation defaults inherited from base 3DGS rather than values discussed in the paper itself.

#### Ablation Summary

- **Effect of L** (Table 4, D-NeRF mean): L=1 → 31.30 dB / 0.965 SSIM; **L=2 → 32.19 dB / 0.971 SSIM (best overall)**; L=3 → 31.74 dB / 0.962 SSIM. Higher L helps some scenes (Jumping Jacks, T-Rex) but hurts others (Lego) — no single L dominates per-scene, 2 is the best mean tradeoff.
- **Position parameterization choice**: linear polynomial 25.37 dB/0.942 SSIM (underfits motion); cubic polynomial 25.95 dB/0.947 SSIM; spline(5) 32.10 dB but only 91 FPS. The Fourier series matches spline-level quality at 150 FPS — **this parameterization choice is the single most load-bearing design decision**, since naive polynomial alternatives lose ~6-7 dB.
- **Flow loss**: removing it reintroduces "ghostly artifacts" and reduces color reconstruction accuracy (qualitative, Figure 6; no isolated delta-PSNR reported).
- **Time-varying scale**: tested and found to give only marginal gains in specific scenes at increased memory cost; not adopted in the final method.

#### Implementation Reality

- **Framework:** PyTorch with custom CUDA extensions; built directly on top of the official `graphdeco-inria/gaussian-splatting` repo, reusing its `diff-gaussian-rasterization` and `simple-knn` submodules.
- **Key files:** `train.py` (main training loop, implements the static/dynamic two-stage schedule), `gaussian_renderer/` (rendering pipeline — where the time-function evaluation of position/rotation feeds into the otherwise-standard rasterizer), `scene/` (scene/Gaussian parameter representation). The public README does not explicitly annotate which files hold the Fourier/linear time-parameterization logic.
- **Notable implementation details:** the released defaults show densification running iterations 500–15,000 and a position learning-rate schedule (0.00016 → 0.0000016) inherited from base 3DGS conventions; these specific values are not discussed in the paper text itself, so treat them as implementation defaults rather than reported/ablated hyperparameters.

#### Failure Modes & Limitations

Topology changes (Gaussian birth/death, e.g. fluid phenomena) are explicitly called out as "tough" because the formulation assumes every Gaussian persists across the full sequence with no start/end time; the authors suggest extending with per-Gaussian lifetime parameters as future work. Compactness degrades rendering quality on very long sequences (a fixed small L cannot track accumulating motion complexity). The closed-form time functions sacrifice the smoothness/continuity properties of neural-field volume rendering, costing generalization relative to NeRF-based competitors. Quality is also sensitive to camera pose accuracy, and the flow-consistency loss is inapplicable on datasets with discontinuous camera trajectories between frames (D-NeRF).

---

## Relevance to ADAGS

Important comparator for LoRA compactness and fixed-budget claims.

## Connections

## Sources

- https://arxiv.org/abs/2311.12897
- Code: https://github.com/raven38/EfficientDynamic3DGaussian
- Project page: https://raven38.github.io/compactdynamic3dgaussian.github.io/
