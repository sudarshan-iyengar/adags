---
type: paper
node_id: paper:shaw2024_swings
title: "SWinGS: Sliding Windows for Dynamic 3D Gaussian Splatting"
authors: ["Richard Shaw", "Michal Nazarczuk", "Jifei Song", "Arthur Moreau", "Sibi Catley-Chandar", "Helisa Dhamo", "Eduardo Perez-Pellitero"]
year: 2024
venue: "ECCV"
tags: [dynamic-gs, sliding-window, static-dynamic]
status: deep-dived
---

# SWinGS: Sliding Windows for Dynamic 3D Gaussian Splatting

**Paper:** https://arxiv.org/abs/2312.13308
**Code:** Not found
**Base method:** 3D Gaussian Splatting (Kerbl et al. 2023), extended with a per-window deformation MLP in the style of deformable-3DGS / D-NeRF-type canonical + deformation-field approaches.

## One-line thesis

Partitioning a long multi-view dynamic sequence into overlapping, adaptively-sized temporal windows — each with its own canonical Gaussian set and deformation MLP — plus a per-Gaussian learned static/dynamic blend of two MLP weight sets, lets a single small deformation network handle arbitrary-length, motion-imbalanced sequences without the quality collapse a single global canonical representation suffers under large motion.

## Problem / Gap

Deformation-field dynamic-GS methods (D-NeRF-style, Nerfies-style) share one canonical representation across an entire sequence, which the authors state becomes "inherently challenging, especially for large motions" and does not scale to arbitrary-length video. A single global deformation MLP must also spend equal capacity on scene regions that barely move (background) and regions that move a lot, degrading dynamics modeling in motion-imbalanced scenes.

## Method

The sequence is greedily split into overlapping windows sized by accumulated multi-view optical flow, so each window gets roughly equal "motion budget" rather than equal frame count. Each window trains its own canonical 3D Gaussian set plus a small positionally-encoded MLP that predicts per-frame $(\Delta x, \Delta r, \Delta s)$ offsets from canonical Gaussians. To stop the deformation MLP from wasting capacity on near-static regions, each Gaussian carries a learned blend parameter $\alpha$ (initialized from an optical-flow-difference binary mask) that mixes two sets of MLP weights (a "static" set and a "dynamic" set) per Gaussian. After all windows are trained, a short sequential fine-tuning pass enforces temporal consistency across window boundaries by rendering the same interpolated novel camera pose from both the outgoing and incoming window's model and minimizing their L1 difference, with the canonical points frozen and only the deformation network updated.

## Assumptions

Requires multiple calibrated, time-synchronized cameras (multi-view capture, not monocular) and per-view optical flow (RAFT) computed across the whole capture for window-size selection and for the static/dynamic $\alpha$ initialization.

## Limitations / Failure Modes

The paper does not include an explicit limitations section or discuss specific failure scenes. Implicitly, each window still relies on one canonical representation, so very large intra-window topology changes remain a plausible failure mode even though the windowing reduces per-window motion magnitude; the method is also inherently multi-view-only (no monocular capability) and adds per-window model/training overhead compared to a single global model.

## Reusable Ingredients

- **Optical-flow-budgeted adaptive windowing:** greedily grow a temporal window until accumulated multi-view RAFT flow exceeds a threshold, so windows carry roughly equal motion rather than equal frame count.
- **Dual-weight-set per-Gaussian blending ($\alpha$):** give each Gaussian a soft mixture over a "static" and a "dynamic" MLP weight set instead of a hard binary static/dynamic split, initialized from a flow-difference mask but left learnable.
- **Cross-window consistency fine-tuning via novel-view interpolation:** render the same interpolated camera pose from adjacent windows' models and supervise their agreement with an L1 loss, updating only the deformation network with canonical points frozen.

---

### Deep Dive

#### Core Novelty
Relative to a single-canonical-representation deformable-3DGS baseline, SWinGS changes two things: (1) it shards the sequence into overlapping, motion-budgeted windows so no one canonical representation/deformation MLP must cover the full sequence's motion range, and (2) it replaces a hard static/dynamic Gaussian split with a learned per-Gaussian soft blend between two MLP weight sets, so capacity allocation between "moves a lot" and "barely moves" regions is continuous and trainable rather than fixed at initialization.

#### Mathematical Formulation
Deformation field (evaluated per-Gaussian, per-frame, before rasterization):
$$\Delta x(t), \Delta r(t), \Delta s(t) = F_\theta(\gamma(x), \gamma(t))$$
where $\gamma(\cdot)$ is sinusoidal positional encoding and $F_\theta$ is the per-window deformation MLP mapping canonical position $x$ and time $t$ to position/rotation/scale offsets applied to the canonical Gaussian before rendering.

Tunable dynamic MLP blend (per-Gaussian layer computation inside $F_\theta$, replacing a single linear layer):
$$y_i = \phi\Big(\sum_{m=1}^{M} \big(\alpha_{i,m}\, w_m^T x_i + \alpha_{i,m}\, b_m\big)\Big)$$
$$\Delta x(t), \Delta r(t), \Delta s(t) = F^{dyn}_\theta(\gamma(x), \gamma(t), \alpha)$$
with $M{=}2$ weight sets (one intended to specialize static, one dynamic) and $\alpha_i$ a per-Gaussian, per-set learnable blend weight initialized from a binary flow-difference mask (L1 pixel flow difference $>0.5$ average → labeled dynamic) but optimized thereafter, so the split is soft rather than fixed.

Adaptive window-size signal (computed once from the whole capture, before window assignment):
$$\hat v_i = \frac{1}{V}\sum_j \sum_i \|f(I_i^j, I_{i+1}^j)\|_2^2$$
where $f$ is RAFT optical flow between consecutive frames, $j$ indexes the $V$ camera views; a greedy pass accumulates $\hat v_i$ across frames and starts a new window once the accumulated value exceeds a threshold, giving windows roughly equal motion budget rather than equal frame count. Consecutive windows overlap by one frame.

Cross-window temporal-consistency loss (evaluated during a post-hoc sequential fine-tuning pass, not during main per-window training):
$$L_{consistency} = \big| I^{w}_{t=0} - I^{w-1}_{t=N_w-1} \big|_1$$
comparing renders of the same interpolated (SE(3)-geodesic) novel camera pose from the incoming window $w$'s model and the outgoing window $w{-}1$'s model at their shared boundary frame; only the deformation network is updated (canonical Gaussians frozen), alternated with ordinary per-frame training at a 75%/25% ratio.

#### Algorithm / Pipeline Changes
1. Precompute per-view RAFT optical flow across the entire capture; compute $\hat v_i$ (Eq. 3) and greedily partition the sequence into overlapping windows (1-frame overlap) once accumulated flow crosses a threshold — replaces fixed/manual window-size choice.
2. For each window independently: initialize a canonical 3D Gaussian set, initialize per-Gaussian $\alpha$ from the flow-difference binary mask, and train jointly with a small deformation MLP $F^{dyn}_\theta$ (depth 4, width 16, 6 frequency bands, 2 skip connections) that predicts $(\Delta x, \Delta r, \Delta s)$ per frame — this MLP replaces/augments the standard static-3DGS optimization loop.
3. Per window: 2K-iteration warm-up with the deformation MLP frozen (canonical stabilizes first), then joint optimization with Gaussian densification active up to 8K iterations, total 15K iterations per window.
4. After all windows are trained, run a sequential fine-tuning pass per window boundary (3K iterations): render the shared boundary frame from an interpolated novel camera pose using both the current and previous window's model, apply $L_{consistency}$ (Eq. 8) 75% of the time and ordinary photometric training 25% of the time, updating only deformation-MLP weights with canonical Gaussians frozen.

#### Key Hyperparameters & Design Choices
- Deformation MLP: depth $D=4$, width $W=16$, positional-encoding frequency bands $m=6$, 2 skip connections.
- Static/dynamic blend: $M=2$ weight sets; $\alpha$ initialized via flow-difference threshold of 0.5 average L1 pixel difference.
- Learning rate for MLP and $\alpha$ parameters: $1\times10^{-4}$, exponentially decayed by a factor of $1\times10^{-2}$ over 20K iterations.
- Per-window training: 15K total iterations; 2K-iteration warm-up (MLP frozen); densification active through 8K iterations.
- Fine-tuning: 3K iterations per window boundary; consistency-loss/standard-training alternation ratio 75%/25%.
- Photometric loss: $L = (1-\lambda)L_1 + \lambda L_{SSIM}$, standard 3DGS $\lambda$ and other Gaussian-parameter learning rates carried over unchanged (paper does not restate the numeric $\lambda$ or those rates as SWinGS-specific).
- Window size for the fixed-size ablation comparisons: 3 and 9 frames; adaptive windowing is the proposed default.

#### Ablation Summary
- Window Size Analysis (Birthday scene, Technicolor): fixed window size 3 → 33.12 dB PSNR / 16.0 hrs train time / t-LPIPS 0.0076; fixed window size 9 → 33.38 dB / 4.0 hrs / t-LPIPS 0.0053; adaptive windowing → 33.44 dB / 3.3 hrs / t-LPIPS 0.0051 (best quality *and* fastest).
- Tunable dynamic MLP (static/dynamic blend) contribution, adaptive windowing with vs. without the dynamic MLP: 33.44 dB vs. 32.76 dB → **+0.68 dB**, with t-LPIPS improving from 0.0062 to 0.0051. This is the single most impactful ablated component reported.
- Temporal consistency fine-tuning (Neural 3D Video): PSNR 32.01 → 32.05 (+0.04 dB, marginal), but SSIM regresses slightly (0.956 → 0.949) and LPIPS regresses (0.085 → 0.093) while a video-quality metric (VQA) improves (0.666 → 0.726, vs. ground-truth VQA 0.763) — i.e., the consistency fine-tune trades a small amount of per-frame fidelity for improved temporal/perceptual smoothness, not a clean PSNR win.

#### Failure Modes & Limitations
The paper does not include a dedicated limitations or failure-case discussion; the only self-critical framing is the general statement that single-canonical-representation deformation fields are "inherently challenging, especially for large motions," which motivates the windowing but is not quantified as a residual SWinGS failure mode.

---

## Relevance to ADAGS

Raises the novelty bar for reversible routing and static-exclusion claims.

## Connections

## Sources

- https://arxiv.org/abs/2312.13308

