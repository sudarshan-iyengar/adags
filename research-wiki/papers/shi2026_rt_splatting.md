---
type: paper
node_id: paper:shi2026_rt_splatting
title: "RT-Splatting: Joint Reflection-Transmission Modeling with Gaussian Splatting"
authors: ["Ji Shi", "Xianghua Ying", "Bowei Xing", "Ruohao Guo", "Wenzhen Yue"]
year: 2026
venue: "CVPR 2026"
external_ids:
  arxiv: "2605.18263"
tags: [gaussian-splatting, occupancy, opacity-factorization, transparency]
status: deep-dived
---

# RT-Splatting: Joint Reflection-Transmission Modeling with Gaussian Splatting

**Paper:** https://arxiv.org/abs/2605.18263
**Code:** https://github.com/sjj118/RT-Splatting
**Base method:** 2D Gaussian Splatting (2DGS); also explicitly extends/compares against the deferred-shading reflection codebases Ref-GS and EnvGS, and contrasts with the multi-stage transparent-object pipeline TransparentGS.

## One-line thesis
Factorizing each Gaussian's opacity into geometric occupancy (probability the ray interacts with the primitive's substance) and optical opacity (conditional probability of absorption/scattering given interaction) lets one set of primitives be read simultaneously as a surface — for sharp specular reflections via first-surface extraction — and as a volume — for clear background transmission — removing the need for separate reflection/transmission representations or optimization stages.

## Problem / Gap
Deferred-shading reflection methods (3DGS-DR, Ref-GS, EnvGS) rely on G-buffers that store only the nearest surface's properties per pixel, which is fundamentally incompatible with seeing *through* semi-transparent material rather than having the background occluded by it. Multi-stage pipelines such as TransparentGS decompose the scene into separate reflection and transmission optimization stages, preventing joint single-pass optimization of transparent and opaque content together. Prior planar reflection-transmission methods assume roughly flat geometry and do not generalize to complex shapes.

## Method
RT-Splatting extends 2DGS by giving each Gaussian two separate scalars — geometric occupancy σ and optical opacity α — instead of one conflated opacity. A first-surface extraction pass aggregates per-pixel surface attributes (e.g. normals) weighted by σ-derived first-hit probabilities, feeding a deferred specular-shading pass that produces color term $C_{spec}$. A separate volumetric forward pass uses the effective opacity $\alpha_{eff}=\sigma\alpha$ to accumulate a subsurface transport term $C_{sub}$ that blends transmitted background and scattered light via a transmissivity ratio τ. The two are composited into a final color with a learnable attenuation factor β. A Specular-Aware Gradient Gating mechanism suppresses image-loss gradients into the transmission branch in high-specular-variance regions, and a SAM2-derived transparency mask weakly supervises α directly, alongside a modified density-control schedule that resets σ and α on alternating 1500-iteration cycles instead of a single combined 3000-iteration reset.

## Assumptions
Assumes thin, semi-transparent specular surfaces (e.g. glass panes, screens) captured with multi-view coverage sufficient for standard 2DGS reconstruction; assumes single-bounce light transport (no refraction, no multi-bounce scattering) and relies on a frozen off-the-shelf SAM2 segmentation model to supply a transparency-region mask as weak supervision.

## Limitations / Failure Modes
The paper states the method is designed for thin semi-transparent surfaces and does not explicitly model refraction or multiple light bounces, so it cannot handle thicker refractive media or multi-bounce transport such as water or solid glass objects. Ablations show removing joint optimization (Δ −1.695 dB PSNR in transparent regions) and removing the occupancy/opacity factorization itself (Δ −1.064 dB) are the two largest quality hits, and removing SAM2-mask supervision causes measurable degradation with reported optimization instability.

## Reusable Ingredients
- Occupancy/opacity factorization ($\alpha_{eff}=\sigma\cdot\alpha$): splits "does the ray hit something" from "does that something absorb/scatter the ray" — a general pattern for any per-primitive existence-vs-effect split.
- First-surface probabilistic aggregation ($p_i=\sigma_i\mathcal{G}_i\prod_{j<i}(1-\sigma_j\mathcal{G}_j)$): extracts single-surface attributes from a volumetric representation for deferred shading without discarding volumetric accumulation used elsewhere.
- Specular-Aware Gradient Gating: variance-triggered, patch-local gradient gating via a stop-gradient blend, protecting one loss branch's supervision from being corrupted by another branch's high-variance signal — reusable wherever two rendering branches compete for the same pixels.
- Off-the-shelf segmentation model (SAM2) as weak mask supervision for a physically-interpretable per-primitive parameter, via a BCE loss.
- Alternating, per-parameter density-control reset schedule (separate σ/α resets) so one channel's reset cadence doesn't destabilize the other, and pruning keyed on occupancy rather than opacity to avoid deleting genuinely-present-but-optically-clear primitives.

---

### Deep Dive

#### Core Novelty
Standard 3DGS/2DGS conflate "is there something here" and "how much light does it let through" into a single opacity scalar, forcing a tradeoff between sharp reflective surfaces and clear transmission. RT-Splatting's key insight is that these are physically distinct quantities — interaction probability vs. conditional absorption/scattering probability — and separating them lets the same primitive set serve two renderer interpretations at once (surface-like via σ for first-surface reflection extraction, volume-like via $\alpha_{eff}=\sigma\alpha$ for transmission compositing) without separate geometry or a multi-stage pipeline.

#### Mathematical Formulation
- **Occupancy-opacity factorization (effective opacity):** $\alpha_{eff} = \sigma \cdot \alpha$, where $\sigma \in [0,1]$ is per-Gaussian geometric occupancy (probability the ray interacts with the primitive's substance) and $\alpha \in [0,1]$ is per-Gaussian optical opacity (conditional probability of absorption/scattering given interaction). Domain of application: $\alpha_{eff}$ replaces the single opacity term used in standard volumetric alpha-compositing for the transmission/volumetric forward-rendering pass; $\sigma$ alone (not $\alpha_{eff}$) drives the separate first-surface extraction used for the reflection pass. This is evaluated per-Gaussian at rasterization time, before/during compositing.
- **First-surface attribute extraction (Eq. 2):** $A = \sum_i p_i \cdot a_i$, where $p_i = \sigma_i \cdot \mathcal{G}_i \cdot \prod_{j=1}^{i-1}(1-\sigma_j \mathcal{G}_j)$ is the probability Gaussian $i$ is the first (nearest) ray-surface hit, $\mathcal{G}_i$ is the Gaussian's spatial falloff term, and $a_i$ is a per-Gaussian surface attribute (e.g. normal). Evaluated per-pixel, feeding G-buffers for the deferred specular-shading pass; uses σ only.
- **Subsurface-transport composition (Eq. 3):** $C_{sub} = \tau \cdot C_{trans} + (1-\tau) \cdot C_{scatter}$, where $\tau \in [0,1]$ is a transmissivity ratio blending transmitted-background color $C_{trans}$ and scattered-light color $C_{scatter}$. Evaluated in the volumetric forward pass.
- **Final pixel color (Eq. 4):** $C = C_{spec} + \beta \cdot C_{sub}$, where $\beta \in [0,1]$ is a learnable attenuation factor suppressing the subsurface term. Final compositing step after both passes.
- **Specular-aware gating weight (Eq. 5):** $g(x) = \exp(-k \cdot \mathrm{Var}_{p \in N(x)}[C_{spec}(p)])$, over a local 3×3 neighborhood $N(x)$ with $k=4$; low when local specular color varies strongly (reflection-dominated regions).
- **Gradient modulation (Eq. 6):** $\partial L_{img}/\partial C_{trans}(x) \leftarrow g(x) \cdot \partial L_{img}/\partial C_{trans}(x)$ — scales the image-loss gradient reaching the transmission branch, applied during backprop.
- **Stop-gradient implementation (Eq. 8):** $\tilde{C}_{trans} = (1-g)\cdot \mathrm{sg}(C_{trans}) + g \cdot C_{trans}$, the practical realization of Eq. 6 via the stop-gradient operator $\mathrm{sg}(\cdot)$, avoiding a custom backward pass.
- **Transparent mask loss (Eq. 7):** $L_{mask} = \mathrm{BCE}(1-M, \alpha)$, where $M$ is a transparency mask from a frozen pretrained SAM2 model; supervises α directly (not σ), keeping occupancy free to represent geometry while opacity is pulled toward the segmented transparency prior.
- **Total loss (Eq. 10):** $L = L_{img} + \lambda_n L_n + \lambda_{mask} L_{mask}$, with appearance loss (Eq. 9) $L_{img} = (1-\lambda) L_1 + \lambda L_{D\text{-}SSIM} + \lambda_{perc} L_{perc}$.

#### Algorithm / Pipeline Changes
1. Each Gaussian is parameterized with two scalars, σ (geometric occupancy) and α (optical opacity), replacing 2DGS's single opacity parameter.
2. Rasterize per-pixel Gaussian order as in 2DGS; compute first-surface hit probabilities $p_i$ from σ only (Eq. 2) and aggregate surface attributes (normals, etc.) into G-buffers.
3. Feed G-buffers into a deferred shading pass (extending Ref-GS/EnvGS-style deferred reflection shading) to produce $C_{spec}$.
4. Separately, run a volumetric forward accumulation pass using $\alpha_{eff}=\sigma\alpha$ per Gaussian to composite the subsurface transport term $C_{sub}$ (Eq. 3), blending transmitted background and scattered light via τ.
5. Composite final color $C = C_{spec} + \beta C_{sub}$ (Eq. 4).
6. At loss time, compute the local gating weight $g(x)$ from the 3×3-neighborhood variance of $C_{spec}$ (Eq. 5) and apply it via the stop-gradient blend (Eq. 8) before computing the image loss on the transmission branch, suppressing gradients from highly specular regions into transmission reconstruction.
7. Supervise α with a BCE mask loss against the SAM2-derived transparency mask (Eq. 7), and normals with a normal-consistency loss $L_n$.
8. Density control alternates every 1500 iterations between resetting σ and resetting α (rather than one combined 3000-iteration reset); pruning is keyed on σ rather than α so highly transparent-but-present primitives are not deleted.

#### Key Hyperparameters & Design Choices
- Gating strength $k = 4$ (Eq. 5); sensitivity analysis reported in the paper's Table 4.
- Gating neighborhood window: 3×3.
- D-SSIM blend weight $\lambda = 0.2$.
- Perceptual loss weight $\lambda_{perc} = 0.01$.
- Normal consistency weight $\lambda_n = 0.05$.
- Transparent mask loss weight $\lambda_{mask} = 0.01$.
- Density-control reset interval: 1500 iterations, alternating between σ and α (vs. a standard single 3000-iteration reset).
- Mask supervision source: pretrained SAM2, frozen, used only to generate mask $M$.
- Architecture dimensions for any learned network components: Not specified in paper.

#### Ablation Summary
(Table 3, transparent-region metrics; full method PSNR 37.983 dB is the reference)
- w/o joint optimization (multi-stage instead of joint): PSNR 36.288 (Δ −1.695 dB) — largest drop overall.
- w/o occupancy-opacity factorization: PSNR 36.919 (Δ −1.064 dB) — second-largest drop; the paper's headline mechanism.
- w/o $L_{mask}$ (SAM2 supervision): PSNR 37.167 (Δ −0.816 dB), reported with optimization instability.
- w/o attenuation β: PSNR 37.541 (Δ −0.442 dB).
- w/o scattering term: PSNR 37.597 (Δ −0.386 dB).
- w/o specular-aware gating: PSNR 37.754 (Δ −0.229 dB), reported with visible floater artifacts.
Most impactful component: removing joint optimization causes the largest PSNR drop, with the occupancy/opacity factorization a close second and the paper's central contribution.

#### Implementation Reality
- **Framework:** PyTorch + custom CUDA, extending the 2DGS (surfel) codebase; uses `nvdiffrast` (NVIDIA's differentiable rasterizer) for surface-side rendering.
- **Key files:** `train.py` (training entry point), `render.py` (inference/rendering), `eval.sh` (combined train+eval automation across datasets), `metrics.py` (quantitative evaluation), `convert.py` (data conversion), `gaussian_renderer/` (rendering pipeline implementation), `scene/` (scene data loading/management). Requires custom submodules `simple-knn` and `diff-surfel-anych` (differentiable surfel rendering).
- **Notable implementation details:** README names Ref-GS and EnvGS as base codebases more explicitly than the paper abstract alone conveys. Evaluation runs via a single `eval.sh` across Ref-Real, NeRF-Casting, EnvGS, Tanks&Temples, and custom capture datasets. Dataset access is via Google Drive rather than an automated download script.

#### Failure Modes & Limitations
The paper states the method is designed for thin semi-transparent surfaces and does not explicitly model refraction or multiple light bounces; it therefore cannot handle thicker refractive media or multi-bounce light transport such as water or solid glass objects.

---

## Relevance to ADAGS

The one verified GS precedent for a "presence is not the same scalar as
visibility contribution" factorization — but applied to transparent/
reflective MATERIALS in static scenes, not to temporal existence or
occlusion state in dynamic scenes. Cite as mechanism-shape neighbor when
claiming an existence/visibility split; the delta is domain and semantics.

## Connections

- Pressures [[gap_map#G9 - Uncertainty And Occlusion Confidence Are Underused In ADAGS]]

## Sources

- https://arxiv.org/abs/2605.18263
