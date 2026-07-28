---
type: paper
node_id: paper:sandu2026_temporally_aware_densification
title: "Temporally Aware Densification for Dynamic 3D Gaussian Splatting"
authors: ["Vikram Sandu", "Mayurdeep Pathak", "Rajiv Soundararajan"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2606.23212"
tags: [dynamic-gs, densification, temporal-visibility, lifespan, visibility-aware]
status: deep-dived
---

# Temporally Aware Densification for Dynamic 3D Gaussian Splatting

**Paper:** https://arxiv.org/abs/2606.23212
**Code:** Not found (no repository link in the paper; not located on GitHub or Papers With Code)
**Base method:** A Fourier-motion / temporal-RBF-opacity deformation model in the style of periodic dynamic 3DGS methods (e.g. STG, Ex4DGS, Swift4D) with standard 3DGS positional-gradient densification.

## One-line thesis

Short-lived dynamic Gaussians never accumulate enough positional gradient to trigger standard 3DGS densification because the gradient-accumulation denominator counts all frames, not just the frames the Gaussian is actually visible in; normalizing gradient accumulation by per-Gaussian visibility, then relaxing the split/clone threshold and the temporal-capacity allocation in proportion to each Gaussian's temporal lifespan, restores adequate densification signal to exactly the primitives static-style densification starves.

## Problem / Gap

Dynamic Gaussians in periodic/RBF-style deformation models are frequently visible (non-zero opacity) for only a handful of frames out of the full sequence. Static 3DGS densification accumulates positional gradient over all training views/frames and divides by a fixed count, so a Gaussian visible in 5 of 300 frames accumulates a gradient average diluted by the other 295 frames where it contributes nothing — it never crosses the fixed clone/split threshold `τ_pos`, leaving fast, transient, or newly-appearing content permanently under-densified and blurred. This is a mismatch between a densification criterion designed for static, always-visible primitives and dynamic primitives whose defining property is sparse temporal presence.

## Method

The paper introduces three components layered onto an existing periodic dynamic-3DGS pipeline (temporal-RBF opacity gate + Fourier-basis position + polynomial rotation/scale). (1) Visibility-Aware Densification (VAD) reweights the positional-gradient accumulation used for the clone/split decision by each Gaussian's per-frame opacity/visibility and normalizes by the sum of that visibility rather than by frame count, so gradient signal is only diluted by frames where the Gaussian truly wasn't visible. (2) Temporally-Adaptive Thresholding (TAT) makes the clone/split gradient threshold `τ_pos` itself a function of each Gaussian's normalized temporal lifespan `ψ̄ᵢ`, relaxing it for short-lived Gaussians and leaving it near-unchanged for long-lived/static ones. (3) Temporal Offset Warping (TOW) piecewise-linearly warps the time input fed to each Gaussian's Fourier motion basis so that a disproportionate share of frequency capacity is spent near the Gaussian's own temporal center, letting the same fixed Fourier order model faster local motion without adding parameters.

## Assumptions

Multi-view (multiview-temporal) capture with a periodic/RBF-style dynamic-3DGS backbone that already parametrizes each Gaussian's temporal presence via an opacity gate and models motion with a bounded Fourier/polynomial basis; the method assumes densification is still governed by positional-gradient-vs-threshold decisions as in vanilla 3DGS. It is evaluated only on multiview studio/sports capture (N3DV, Interdigital, VRU Basketball).

## Limitations / Failure Modes

The paper explicitly states it has not been tested on long sequences (several minutes) and flags scaling to such durations as open. It also notes the current focus is multiview capture, with monocular/sparse-view extension left as future work. Architecturally, the mechanism is still a densification/gradient-thresholding and temporal-capacity-allocation scheme — it decides *where and how finely* to split/clone/allocate Fourier capacity, not whether a primitive's identity should persist through an occlusion versus being newly grown; it carries no explicit hidden/occluded-surface or causal-visibility reasoning.

## Reusable Ingredients

- **Visibility-weighted gradient normalization** — reweight/normalize the densification gradient signal by per-primitive visibility instead of raw frame count, so sparsely-visible primitives aren't penalized by frames where they weren't rendered.
- **Lifespan-conditioned threshold relaxation** — make a fixed hyperparameter (here, the split/clone threshold) a smooth function of a per-primitive temporal-scale statistic, recovering static behavior in the limit of full visibility.
- **Local time-warping for fixed-capacity bases** — piecewise-linear (or otherwise monotonic) warping of the time axis fed into a fixed-order Fourier/positional basis, to reallocate representational capacity toward a primitive's local temporal neighborhood without adding parameters.
- **Masked/dynamic-region metrics (M-PSNR, M-SSIM)** — evaluating only on optical-flow-identified dynamic pixels, avoiding the dilution of global PSNR by dominant static background.
- **Plug-and-play component validation** — demonstrating a proposed module (VAD) as a drop-in addition to multiple independent baselines (STG, Ex4DGS, Swift4D) to argue mechanism-level generality rather than one-pipeline-specific tuning.

---

### Deep Dive

#### Core Novelty

Relative to vanilla 3DGS-style densification (which accumulates and thresholds positional gradients uniformly across all training samples), this paper's actual change is to make both the gradient statistic and the threshold used for the clone/split decision functions of each Gaussian's temporal visibility/lifespan, and additionally to reparametrize time itself (via warping) so a fixed-capacity Fourier motion basis behaves as if it had more capacity near a Gaussian's active window. The key insight: sparsity of visibility, not sparsity of information, is what suppresses densification signal for dynamic content — correcting the accounting (normalize by visibility) and the criterion (threshold scaled by lifespan) restores the signal without changing what "densification" fundamentally means.

#### Mathematical Formulation

Temporal opacity gate (existing periodic-model component, context for what "visibility" means here), evaluated per-Gaussian per-frame before rasterization:
$$\sigma_i(t) = \sigma_i^s \cdot \exp\!\big(-\psi_i (t - t_i)^2\big)$$
where $\sigma_i^s$ is the Gaussian's static/base opacity, $t_i$ its temporal center, and $\psi_i$ its temporal scale (inverse of lifespan width).

Visibility-Aware Densification (VAD), evaluated during the periodic densification pass (replaces the frame-count-normalized gradient accumulation used for the clone/split decision):
$$\bar{g}_i = \frac{\sum_{t} \sigma_i(t)\, \big\|\nabla_{\mathbf{p}_i} L(t)\big\|}{\sum_{t} \sigma_i(t)}$$
i.e. the positional gradient at each frame is weighted by the Gaussian's visibility (opacity) at that frame and normalized by the sum of visibility rather than by the number of frames, so frames where the Gaussian is invisible contribute ~0 to both numerator and denominator instead of diluting the average. (Exact notation reconstructed from the described mechanism; the paper states this reduces to standard 3DGS gradient accumulation when opacity is 1 across all frames.)

Temporally-Adaptive Thresholding (TAT), evaluated when comparing the accumulated gradient to the split/clone criterion:
$$\tau_{pos}^i = \tau_{pos} \cdot \left(\frac{1}{1 + \beta(1 - \bar{\psi}_i)}\right)^{\alpha}$$
where $\tau_{pos}$ is the base (static) 3DGS gradient threshold, $\bar{\psi}_i$ is Gaussian $i$'s normalized temporal lifespan (scale) in $[0,1]$, $\beta$ is a sensitivity parameter (set to 0.3), and $\alpha$ is a scaling exponent (set to 1.0). Short-lived Gaussians (small $\bar\psi_i$) get a smaller (relaxed) $\tau_{pos}^i$, making them easier to qualify for densification; as $\bar\psi_i \to 1$ (fully visible/static), $\tau_{pos}^i \to \tau_{pos}$, recovering standard behavior.

Temporal Offset Warping (TOW), applied to the time input before it is fed into each Gaussian's Fourier position/motion basis:
$$t' = t_i + W(t - t_i;\ \lambda_t, \rho_t)$$
where $W$ is a piecewise-linear warp with a "near" segment scaled by $s_{near} = \rho_t / \lambda_t$ (stretches time near the Gaussian's center $t_i$, effectively raising the local Fourier frequency resolution) and a "far" segment scaled by $s_{far} = (1-\rho_t)/(1-\lambda_t)$ (compresses time far from the center, enforcing smoothness/low frequency there). $\lambda_t \in (0,1)$ is the normalized focus-window size and $\rho_t \in (0,1)$ is the fraction of total temporal capacity allocated inside that window.

#### Algorithm / Pipeline Changes

1. During the periodic densification pass (every 200 iterations up to iteration 17K), for each Gaussian accumulate the positional gradient across observed frames weighted by that Gaussian's per-frame opacity $\sigma_i(t)$, and normalize by the sum of those opacities instead of the raw frame count (VAD) — this replaces the standard 3DGS running-average gradient accumulator.
2. Compute each Gaussian's normalized temporal lifespan $\bar\psi_i$ from its existing temporal-scale parameter $\psi_i$, and scale the global split/clone threshold $\tau_{pos}$ per-Gaussian via the TAT formula before comparing it against the VAD-weighted gradient — this replaces the single global threshold check with a per-Gaussian threshold check.
3. Before evaluating the Fourier position/motion basis for a Gaussian at query time $t$, warp $t$ relative to the Gaussian's temporal center $t_i$ using the piecewise-linear TOW map (near-window stretch, far-window compression) with fixed $\lambda_t = 50/T$ ($T$ = total frame count) and $\rho_t = 0.75$ — this sits upstream of the existing K=4-mode Fourier position evaluation and the degree-1 polynomial rotation/scale evaluation, changing only the effective time argument, not the basis order.
4. Multiple temporal centers are still initialized uniformly with 20-frame spacing (existing periodic-model initialization), unchanged by the three additions.
5. All three components (VAD, TAT, TOW) are additive/compatible and were also validated as drop-in modules (primarily VAD) on three independent external baselines (STG, Ex4DGS, Swift4D) without modifying those baselines' own architectures.

#### Key Hyperparameters & Design Choices

- TAT: $\alpha = 1.0$, $\beta = 0.3$ (fixed across all datasets).
- TOW: $\lambda_t = 50/T$ (50-frame focus window), $\rho_t = 0.75$ (75% of temporal capacity inside that window); sensitivity analysis shows stable performance within roughly ±0.25 variance around these values.
- VAD: no additional hyperparameters (it is a reweighting/normalization of the existing gradient-accumulation statistic).
- Optimizer: Adam, initial LR $2.6\times10^{-4}$ decaying to $2.6\times10^{-6}$.
- Training: 40K total iterations; densification every 200 iterations until iteration 17K.
- Fourier motion basis: K = 4 modes; rotation/scale modeled as degree-1 polynomials in time offset.
- Temporal center initialization: uniform spacing every 20 frames.
- Hardware: single NVIDIA A4000 (16GB).

#### Ablation Summary

On N3DV (Table 3), incremental additions over the baseline (deformation-only, no VAD/TAT/TOW):
- Baseline: PSNR 31.98, M-PSNR 21.96, M-SSIM 0.781, LPIPS 0.066
- + VAD: PSNR 32.14, M-PSNR 23.40 (**+1.44 M-PSNR**, the single largest jump), M-SSIM 0.832, LPIPS 0.062
- + VAD + TAT: PSNR 32.17, M-PSNR 23.62 (+0.22 further), M-SSIM 0.837, LPIPS 0.061
- + Full (VAD+TAT+TOW): PSNR 32.42, M-PSNR 24.68 (+1.06 further), M-SSIM 0.863, LPIPS 0.059

On Interdigital (Table 3), same pattern: Baseline M-PSNR 24.75 → +VAD 27.32 (**+2.57**, largest single jump) → +TAT 27.51 (+0.19) → +Full 28.87 (+1.36).

VAD is the single most impactful component in both datasets by a clear margin, particularly on the masked/dynamic-region metric it was designed to fix; TOW is the second-largest contributor; TAT contributes the smallest but consistent increment. Generalization study (Table 4): adding VAD alone to independent baselines yields M-PSNR gains of +1.40 (STG/N3DV), +0.77 (STG/Interdigital), +0.39 (Ex4DGS/N3DV), +0.27 (Ex4DGS/Interdigital), +0.30 (Swift4D, both datasets) — larger gains on baselines using multiple temporal anchors, consistent with the stated mechanism.

Efficiency cost (Table 9, N3DV): model size grows from 155MB (baseline) to 182MB (+VAD), 189MB (+VAD+TAT), 204MB (+Full); training time grows from 49 min to 62 min for the full method.

#### Failure Modes & Limitations

The paper states the method "has not yet been tested on longer video sequences spanning several minutes," leaving scaling to such durations as an open question. It also notes the current evaluation is restricted to multiview capture, with monocular or sparse-view settings left as a "practical next step." No scene-specific quantitative failure cases (e.g., a specific dB drop on a specific scene) are reported beyond these two stated scope limitations.

---

## Relevance to ADAGS

Direct novelty pressure on Event-Causal Hide/Reveal Gaussians: the final claim must not sound like visibility-aware densification. The distinction should be that visibility events decide whether the same identity should remain visible or hidden/revealed, not primarily where to split or densify.

## Connections

- Pressures [[ideas/event-causal-visibility-gaussians]]
- Addresses [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]
- Addresses [[gap_map#G5 - Capacity Allocation Must Be Matched And Dynamic-Aware]]

## Sources

- https://arxiv.org/abs/2606.23212
