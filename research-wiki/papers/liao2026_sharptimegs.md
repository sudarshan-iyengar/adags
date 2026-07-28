---
type: paper
node_id: paper:liao2026_sharptimegs
title: "SharpTimeGS: Sharp and Stable Dynamic Gaussian Splatting via Lifespan Modulation"
authors: ["Zhanfeng Liao", "Jiajun Zhang", "Hanzhang Tu", "Zhixi Wang", "Yunqi Gao", "Hongwen Zhang", "Yebin Liu"]
year: 2026
venue: "CVPR"
external_ids:
  arxiv: "2602.02989"
tags: [dynamic-gs, sharpness, lifespan, densification]
status: deep-dived
---

# SharpTimeGS

**Paper:** https://arxiv.org/abs/2602.02989
**Code:** Not found (the linked GitHub repo `liaozhanfeng/SharpTimeGS` contains only the static project page — `index.html`, `css/`, `js/`, `assets/` — no training/inference source)
**Base method:** Per-primitive-motion 4D Gaussian Splatting with Gaussian-shaped temporal decay for visibility, as in FreeTimeGS (Wang et al. 2025); also compared against 4DGS, STGS (SpaceTimeGS), and Deformable-3DGS.

## One-line thesis

Replacing each Gaussian's Gaussian-shaped temporal decay with a learnable flat-top visibility profile, and using the same learned "lifespan" to scale down each primitive's velocity, removes the forced trade-off in prior motion-based 4DGS between static-region stability (drift, multi-primitive redundancy) and dynamic-region sharpness (motion blur from decay-driven soft visibility).

## Problem / Gap

Motion-based 4DGS methods such as FreeTimeGS give every primitive a linear velocity plus a Gaussian temporal-decay visibility curve. To represent a static or slow region, several overlapping primitives with staggered decay centers are needed to approximate a flat, time-invariant visibility, which is redundant and lets optimization push "static" primitives to small-but-nonzero velocities that accumulate drift over long sequences. Conversely, decay-based soft visibility windows blur fast-moving content near the visibility boundary because primitives remain partially active outside their true occupancy interval. Prior work does not differentiate the optimization treatment of static vs. dynamic regions, so the same decay/motion parameterization is used everywhere.

## Method

Each Gaussian gets a learnable "lifespan" pair (temporal variance $\sigma_t$, temporal radius $r$) in addition to standard 3DGS attributes and a linear velocity $v$. The lifespan defines (a) a flat-top temporal visibility profile that is fully opaque within $\pm r$ of the primitive's center time $T$ and decays only outside that window, and (b) a motion-damping factor $f(\sigma_t,r)$ that divides the velocity, so long-lived (large $\sigma_t,r$) primitives are pinned near-static while short-lived primitives keep unrestricted motion. Initialization seeds dynamic regions (via optical flow + SAM2 segmentation and cross-frame 3D point matching for initial velocity) with short lifespans, and static regions (via COLMAP, zero velocity) with lifespans extended to 3x the sequence length. Training runs in two stages: an initial AbsGS-style densification phase, followed by a velocity-lifespan-aware phase that scores primitives by reconstruction error, opacity, and motion magnitude to reallocate capacity toward fast, short-lived regions while keeping the static point count compact.

## Assumptions

Multi-view (not monocular) capture with COLMAP-recoverable static structure; requires off-the-shelf optical flow and SAM2 segmentation to seed dynamic-region velocities, and per-frame images dense enough for gradient-based densification statistics.

## Limitations / Failure Modes

The paper's own stated limitation is that the pipeline is not real-time to reconstruct: converting a multi-view video into the 4D representation takes several hours despite 100 FPS 4K rendering at inference. It is scoped to novel-view synthesis only (no relighting), and the authors note it "could benefit from stronger geometric priors." No scene-type-specific failure cases or quantitative degradation numbers were reported in the accessible text.

## Reusable Ingredients

- **Flat-top temporal visibility** — replaces pure Gaussian decay with a plateau region ($|t-T|\le r$, opacity unchanged) plus Gaussian falloff only outside it, so primitives don't fade near the edges of their active interval.
- **Lifespan-gated velocity damping** ($f(\sigma_t,r)=\max\{1.0,(\sigma_t+r)^2\}$) — a single learned scalar simultaneously controls temporal extent and motion magnitude, coupling "how long a primitive is visible" to "how much it's allowed to move."
- **Asymmetric static/dynamic initialization** — static points from COLMAP get zero velocity and long lifespan by construction; dynamic points get flow/SAM2/3D-matching-derived velocity and short lifespan, avoiding a single shared prior for both regimes.
- **Velocity-lifespan-aware densification score** combining reconstruction error, opacity, and a saturating function of velocity/lifespan — redirects capacity toward fast, short-lived content instead of uniform gradient-based cloning.

---

### Deep Dive

#### Core Novelty

The paper's change relative to FreeTimeGS-style motion-based 4DGS is to make temporal visibility and motion magnitude both functions of one learned per-primitive lifespan pair $(\sigma_t, r)$, rather than treating visibility as a fixed Gaussian decay and motion as an independently optimized velocity. The insight is that the two failure modes (static drift, dynamic blur) share a common cause — undifferentiated treatment of "how long is this primitive relevant" — so tying visibility shape and velocity damping to the same lifespan parameter lets static primitives naturally freeze (large lifespan → near-zero effective velocity, flat full-opacity visibility) while dynamic primitives naturally stay sharp (small lifespan → full velocity, tight visibility window with hard-edged falloff instead of gradual fade).

#### Mathematical Formulation

Motion with lifespan-scaled velocity, evaluated per-Gaussian before rasterization at query time $t$:
$$X_t = X + \frac{v}{f(\sigma_t, r)}(t - T), \qquad f(\sigma_t, r) = \max\{1.0, (\sigma_t + r)^2\}$$
$X$ is the canonical (center-time) position, $v$ the learned linear velocity, $T$ the primitive's center time, $\sigma_t$ the lifespan variance, $r$ the temporal radius. Large $\sigma_t+r$ drives $f\to\infty$, suppressing displacement; $f$ is floored at 1.0 so short-lived primitives get undamped motion.

Flat-top temporal visibility, evaluated per-Gaussian before rasterization at query time $t$:
$$l(t) = \begin{cases} 1, & |t-T| \le r \\ \exp\!\left(-\dfrac{(|t-T|-r)^2}{\sigma_t^2}\right), & |t-T| > r \end{cases}, \qquad O_t = O \cdot l(t)$$
$O$ is the primitive's base opacity; $O_t$ is the time-modulated opacity used in rasterization. Inside the radius $r$ the primitive is fully opaque (no decay-induced fade); outside it, opacity falls off with a Gaussian tail governed by $\sigma_t$.

Densification score, evaluated per-primitive during the Stage-2 densification pass (after a forward/backward pass, before clone/prune decisions):
$$s = \lambda_e E + \lambda_o O + \lambda_l\left(1 - \exp\!\left(-\frac{\lVert v\rVert + 1}{f(\sigma_t, r)}\right)\right)$$
$E$ is the accumulated reconstruction error attributed to the primitive (exact computation deferred to supplementary material, not in the accessible text); $O$ is opacity; the third term is a saturating function of velocity-over-lifespan-damping, so it grows with true (damped) motion magnitude and saturates near 1 for fast, short-lived primitives. $\lambda_e, \lambda_o, \lambda_l$ are loss/score weights whose numeric values are not stated in the accessible text.

Temporal lifespan extension loss, part of the training objective:
$$\mathcal{L}_t = \frac{1}{N}\sum_i \frac{1}{\sqrt{-2\log(o_{th})\,\sigma_{t,i}^2} + r_i}$$
$o_{th}$ is an opacity truncation threshold (numeric value not stated); this term penalizes primitives for having short effective lifespans, encouraging extension where reconstruction does not require short-lived, high-motion behavior — i.e., it pushes primitives toward the static/long-lived regime by default unless residual error/velocity justifies otherwise.

#### Algorithm / Pipeline Changes

1. Initialize static Gaussians from COLMAP with velocity $v=0$ and lifespan set to cover roughly 3x the total sequence length (long-lived).
2. Initialize dynamic Gaussians using optical flow + SAM2 to segment moving regions, and cross-frame 3D point matching to compute initial per-primitive velocity; lifespan variance $\sigma_t$ initialized to cover about 3 frames, temporal radius $r$ initialized to $1\times10^{-6}$ (near-zero plateau).
3. At render time for query timestamp $t$: displace each primitive via $X_t = X + v/f(\sigma_t,r)\cdot(t-T)$, and modulate opacity via $O_t = O\cdot l(t)$ using the flat-top profile above, before standard Gaussian rasterization.
4. Stage 1 (first 1/3 of training iterations): standard AbsGS gradient-based densification (clone/split by accumulated absolute image-space gradient), no lifespan-aware scoring yet.
5. Stage 2 (remaining iterations): switch to the velocity-lifespan-aware score $s$ above; primitives are ranked and cloned/pruned by this score while keeping the total Gaussian count fixed ("same number of new Gaussians" replaces removed ones), biasing added capacity toward high-error, high-motion, short-lived primitives instead of uniformly across the scene.
6. Total training loss combines reconstruction ($L_1$ + SSIM + perceptual), regularization (scale, opacity, normal consistency, plus the temporal lifespan extension term $\mathcal{L}_t$), and the densification-auxiliary error term $\mathcal{L}_e$ (defined in supplementary material, not accessible here).

#### Key Hyperparameters & Design Choices

- Reconstruction loss weights: $\lambda_1=0.8$ (L1), $\lambda_s=0.2$ (SSIM), $\lambda_p=0.01$ (perceptual).
- Regularization weights ($\lambda_{scale}, \lambda_{opacity}, \lambda_n$ for normal consistency, weight on $\mathcal{L}_t$): Not specified in paper (accessible text).
- Densification score weights $\lambda_e, \lambda_o, \lambda_l$: Not specified in paper (accessible text).
- Opacity truncation threshold $o_{th}$ in $\mathcal{L}_t$: Not specified in paper (accessible text).
- Static-region lifespan initialization: variance covers ~3x total sequence frames; dynamic-region lifespan variance $\sigma_t$ covers ~3 frames; temporal radius $r$ initialized to $1\times10^{-6}$ for dynamic regions.
- Stage 1 / Stage 2 split: Stage 1 is the first 1/3 of total training iterations (AbsGS densification); Stage 2 is the remaining 2/3 (lifespan-velocity-aware densification), exact total iteration count not specified in accessible text.
- Learning rates/optimizer/schedules for $\sigma_t$, $r$, $v$: Not specified in paper (accessible text).
- Reported inference performance: real-time rendering up to 4K at 100 FPS on one RTX 4090.

#### Ablation Summary

From Table 2 (SelfCap dataset), each row removes one component from the full model (full model: PSNR 27.36, SSIM 0.947, LPIPS 0.244):

- **w/o representation** (removes the lifespan-modulated motion/visibility representation itself): PSNR 25.96 (−1.40 dB), SSIM 0.907, LPIPS 0.299 — the single most impactful component; removing the core lifespan representation costs the most PSNR and by far the most LPIPS (+0.055).
- **w/o densification** (no velocity-lifespan-aware densification): PSNR 26.82 (−0.54 dB), SSIM 0.919, LPIPS 0.317 — this configuration has the worst LPIPS of any ablation, indicating the densification strategy matters most for perceptual sharpness specifically.
- **w/o initialization** (no flow/SAM2/3D-matching-based velocity initialization): PSNR 26.83 (−0.53 dB), SSIM 0.927, LPIPS 0.297.
- **w/o lifespan r** (no flat-top plateau, i.e. back to pure Gaussian decay): PSNR 26.76 (−0.60 dB), SSIM 0.927, LPIPS 0.321.

All four components contribute; the full lifespan representation (motion damping + flat-top visibility together) is the largest single contributor to both PSNR and perceptual quality.

#### Failure Modes & Limitations

The only limitation stated in the accessible text is that reconstruction (training) is not real-time — converting multi-view video into the 4D representation takes several hours, even though rendering afterward is real-time (100 FPS at 4K). The paper also notes the method is scoped to novel-view synthesis and does not support relighting, and suggests it "could benefit from stronger geometric priors." No specific failing scene types, camera configurations, or quantitative per-scene degradation were found in the accessible text.

---

## Relevance to ADAGS

Closest pressure on ADAGS's motion-aware densification and blur-reduction claims.

## Connections

## Sources

- https://arxiv.org/abs/2602.02989
