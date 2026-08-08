---
type: paper
node_id: paper:lisowski2026_tom_gs
title: "TOM-GS: Editable Video Representation via Temporal Opacity Modulation of Static 3D Gaussians"
authors: ["Marek Lisowski", "Łukasz Smoliński", "Kornel Howil", "Piotr Biliński", "Marcin Mazur", "Przemysław Spurek"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2607.22717"
tags: [dynamic-gs, temporal-opacity, presence-only, editability]
status: deep-dived
---

# TOM-GS: Editable Video Representation via Temporal Opacity Modulation of Static 3D Gaussians

**Paper:** https://arxiv.org/abs/2607.22717
**Code:** Not found (no repository linked on arXiv, in the paper text, or on Papers With Code/GitHub search as of this deep-dive)
**Base method:** 3D Gaussian Splatting (3DGS), extended with a closed-form per-Gaussian temporal opacity envelope in place of any spatial deformation field; camera poses come from AnyCam (off-the-shelf feed-forward pose estimation).

## One-line thesis

Giving every 3D Gaussian a fixed spatial pose for the whole video and encoding all scene dynamics as a single closed-form Gaussian-shaped opacity curve in time (learnable mean + scale) is sufficient to reconstruct monocular video at high fidelity while keeping the representation compatible with standard (static) 3D editing tools.

## Problem / Gap

Prior editable video representations (VeGaS, Splatter-a-Video) rely on complex spatial deformation fields or "folded" temporal distributions to move Gaussians through time; these deformations constrain the optimization and make the resulting Gaussians awkward to hand off to standard 3D editors (mesh/physics tools expect static geometry). TOM-GS targets this specifically: it removes spatial deformation entirely and asks whether opacity modulation alone — with geometry frozen — can still reconstruct dynamic video content.

## Method

TOM-GS first estimates camera poses for the monocular input video with AnyCam, fixing a static world coordinate frame without manual calibration. It then optimizes a standard 3DGS scene where every Gaussian's spatial parameters (position, rotation, scale, color) are static across the whole sequence, but its opacity is scaled at render time by a closed-form Gaussian bump in time, parameterized by a learnable temporal mean (when the primitive is most visible) and temporal scale (how long it stays visible). Rendering otherwise uses the unmodified 3DGS front-to-back alpha-compositing rasterizer, with the time-varying opacity substituted in per frame. Training uses inverse-PSNR frame sampling to concentrate optimization on poorly-reconstructed (typically dynamic) frames, disables the standard 3DGS periodic opacity reset (since resetting would erase the learned temporal windows), and caps densification at 10M Gaussians.

## Assumptions

Monocular RGB video input with a single, largely static camera-recoverable world frame (poses obtained via AnyCam, not ground-truth calibration). The method assumes each dynamic element's visibility can be captured by one smooth appear/disappear window per Gaussian, and that spatial motion itself can be approximated by fading different static Gaussians in and out rather than moving any Gaussian's position.

## Limitations / Failure Modes

The paper reports difficulty with extended video sequences and abrupt, large changes in content. Because geometry never moves, covering long or highly dynamic sequences requires a large number of distinct Gaussians to represent successive appearances, which increases memory consumption and inference time. The ablation shows the representation degrades sharply without its core mechanisms: replacing the closed-form temporal window with an MLP-based ("neural") opacity predictor conditioned on position and time drops PSNR from 36.50 to 29.19, and removing temporal opacity entirely (static 3DGS) drops it to 24.36.

## Reusable Ingredients

- **Closed-form per-Gaussian temporal opacity bump** (learnable mean + scale, no MLP) — a cheap, interpretable presence signal that needs only two extra scalars per primitive.
- **Inverse-PSNR frame sampling** — reweights training-frame sampling probability by each frame's current reconstruction error, concentrating capacity on hard/dynamic frames without a separate saliency network.
- **Disabling opacity resets when opacity carries temporal meaning** — a general lesson: any method that overloads opacity with a non-appearance signal (here, temporal presence) must audit standard 3DGS heuristics (periodic resets, opacity pruning) that assume opacity is purely a rendering/visibility parameter.
- **Static-geometry-for-editability constraint** — freezing spatial parameters entirely trades motion-modeling capacity for direct compatibility with mesh/physics-based 3D editing pipelines (their Blender tetrahedra-conversion physics-edit path relies on this).

---

### Deep Dive

#### Core Novelty

Relative to deformation-based dynamic 3DGS (and to VeGaS/Splatter-a-Video's folded temporal distributions), TOM-GS removes spatial deformation from the model entirely and pushes 100% of the temporal modeling burden onto a per-Gaussian opacity envelope. The key insight is that "presence" (whether a static primitive is currently contributing to the render) is a weaker, lower-dimensional signal than "motion," and a population of many static Gaussians with staggered, differently-shaped presence windows can jointly approximate what a moving/deforming Gaussian population would produce — while leaving the underlying geometry untouched and therefore directly editable with conventional static-scene tools.

#### Mathematical Formulation

Temporal opacity function (their Eq. 2), evaluated per-Gaussian at render time for each queried frame timestamp, replacing the constant opacity term used in standard 3DGS rasterization:

$$\alpha_i(t) = \alpha_i^{b} \cdot \exp\!\left(-\frac{(t - \mu_i^{\tau})^2}{2 (\sigma_i^{\tau})^2}\right)$$

Where:
- $\alpha_i(t)$ — the effective opacity of Gaussian $i$ used in alpha-compositing at time $t$.
- $\alpha_i^{b}$ — the Gaussian's learnable base opacity (the standard 3DGS opacity parameter, i.e. its peak/maximum achievable opacity).
- $\mu_i^{\tau}$ — a learnable scalar temporal mean: the point in time at which Gaussian $i$ is maximally visible.
- $\sigma_i^{\tau}$ — a learnable scalar temporal scale: controls how long Gaussian $i$ stays visible around $\mu_i^{\tau}$ (a Gaussian-in-time envelope, structurally the temporal analogue of the spatial covariance in 3DGS).
- $t$ — the frame timestamp being rendered.

This is a single unimodal Gaussian bump in time per primitive: each Gaussian has exactly one $(\mu_i^{\tau}, \sigma_i^{\tau})$ pair and therefore exactly one active temporal interval (of soft, Gaussian-tailed extent, not a hard window). A primitive cannot itself have more than one active temporal interval — there is no mixture, periodicity, or multi-window formulation anywhere in the paper. Coverage of content that appears more than once, or of long/complex temporal behavior, is instead handled at the population level: distinct Gaussians (potentially initialized/duplicated separately) each take their own single window, not by making one Gaussian multi-modal in time.

#### Algorithm / Pipeline Changes

1. Run AnyCam feed-forward pose estimation on the input monocular video to obtain a fixed set of camera poses and establish a static world coordinate frame (replaces manual/COLMAP-style calibration).
2. Initialize a standard 3DGS point cloud/Gaussian set in this static world frame; each Gaussian carries the usual spatial parameters (position, rotation, scale, base color) plus two new learnable scalars, $\mu_i^{\tau}$ and $\sigma_i^{\tau}$.
3. At each training/render step, for the sampled frame's timestamp $t$, compute $\alpha_i(t)$ per Eq. 2 for every Gaussian and substitute it for the constant opacity in the standard 3DGS front-to-back alpha-compositing rasterizer; spatial parameters are not touched by time at all.
4. Sample training frames with probability proportional to each frame's current inverse PSNR (i.e., worse-reconstructed frames are sampled more often), rather than uniformly — this augments/replaces the standard uniform frame-sampling used in per-frame dynamic 3DGS training.
5. Optimize with the standard 3DGS L1 + D-SSIM photometric loss; densify as in standard 3DGS but disable the periodic opacity-reset heuristic (since resetting $\alpha_i^b$ towards a low value would destroy already-learned temporal windows) and cap densification once the model reaches 10M Gaussians (training runs 80,000 iterations total, with densification active until 60,000).
6. For editing: spatial edits (manual scaling/duplication/removal, and physics edits via a Gaussian-to-tetrahedra conversion, Blender physics simulation, and tetrahedra-back-to-Gaussian conversion) operate purely on the static spatial parameters; $(\mu_i^{\tau}, \sigma_i^{\tau})$ are carried through unchanged so temporal presence behavior automatically persists through spatial edits.

#### Key Hyperparameters & Design Choices

- Training iterations: 80,000; densification active until iteration 60,000.
- Densification cap: 10,000,000 Gaussians (hard stop).
- Loss: L1 pixel color + D-SSIM (weight for D-SSIM: Not specified in paper).
- Learning rate / schedule / initialization for $\mu_i^{\tau}$ and $\sigma_i^{\tau}$: Not specified in paper.
- Any explicit regularizer (sparsity/entropy) on temporal opacity: Not specified in paper (none mentioned).
- Image resolution used for training/evaluation: Not specified in paper.
- Camera pose source: AnyCam (feed-forward, off-the-shelf, no manual annotation).

#### Ablation Summary

From Table 5 (average PSNR on their benchmark; full model = 36.50 dB):

- **No temporal opacity (falls back to static 3DGS): 24.36 dB (−12.14 dB)** — by far the largest single component, confirming the temporal opacity mechanism is what makes dynamic reconstruction possible at all.
- Neural (MLP-based) opacity modulation instead of the closed-form Gaussian-in-time formula: 29.19 dB (−7.31 dB) — the closed-form parametrization clearly outperforms a learned neural predictor conditioned on position and time.
- Opacity resets enabled (i.e., not disabling the standard 3DGS reset heuristic): 31.54 dB (−4.96 dB).
- No camera poses (i.e., without AnyCam pose estimation): 31.88 dB (−4.62 dB).
- Uniform frame sampling instead of inverse-PSNR sampling: 35.52 dB (−0.98 dB) — described as giving "consistent gains" but the smallest-magnitude contribution among the ablated components.

Most impactful component: the temporal opacity mechanism itself (removing it entirely costs over 12 dB); among the remaining design choices, disabling opacity resets and using estimated camera poses each contribute roughly 4.6–5 dB, while inverse-PSNR sampling contributes under 1 dB.

#### Failure Modes & Limitations

The paper states the method struggles with extended video sequences and with abrupt, significant scene changes. Because spatial geometry is static for the entire sequence and all appearance/disappearance behavior must be encoded via per-Gaussian presence windows, covering long or highly dynamic content requires a correspondingly large number of Gaussians, which the authors note increases memory consumption and inference time.

---

## Relevance to ADAGS

Closest new presence-axis preprint: demonstrates presence-only encoding is
viable, but presence remains a single smooth bump and identity is one
primitive per active episode (flipbook at the extreme). Any latched/
multi-interval presence design must cite it and differentiate on interval
structure, identity pooling, and motion.

## Connections

- Pressures [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]

## Sources

- https://arxiv.org/abs/2607.22717
