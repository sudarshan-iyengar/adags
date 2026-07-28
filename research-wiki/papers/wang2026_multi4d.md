---
type: paper
node_id: paper:wang2026_multi4d
title: "Multi4D: High-Fidelity Dynamic Gaussian Splatting via Multi-Level Competitive Allocation"
authors: ["Rui Wang", "Quentin Lohmeyer", "Siyu Tang", "Mirko Meboldt"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2606.22197"
tags: [dynamic-gs, allocation, high-fidelity, temporal-consistency]
status: deep-dived
---

# Multi4D: High-Fidelity Dynamic Gaussian Splatting via Multi-Level Competitive Allocation

**Paper:** https://arxiv.org/abs/2606.22197
**Code:** https://github.com/BatFaceWayne/Multi4D (repo exists, README describes the method; code itself is listed as "Coming Soon" — no files to inspect yet)
**Base method:** 3DGS + HexPlane deformation field for canonical/persistent dynamics (deformation-MLP lineage, e.g. 4DGaussians-style), plus explicit 4D spatiotemporal primitives in the style of 4D-primitive methods (e.g. [[papers/li2023_spacetime_gaussians]]/4DGS) used as the transient-detail branch.

## One-line thesis

Splitting the scene into three explicitly typed Gaussian subsets (static, persistent-dynamic via deformation, transient 4D) and letting them compete for photometric-error explanation through shared rasterization gives per-region-appropriate representation, instead of forcing one monolithic representation to trade off temporal identity against high-frequency detail everywhere.

## Problem / Gap

Deformation-based dynamic 3DGS methods (canonical Gaussians warped by an MLP/HexPlane field) preserve temporal correspondence but over-factorize motion into a smooth warp, oversmoothing high-frequency dynamics (fast motion, appearance change). 4D-primitive methods (independent per-time Gaussians) capture fine detail but temporally overparameterize — each primitive is only locally valid in time, so object identity breaks across frames and storage explodes (paper reports 4215k dynamic Gaussians / 2.6 GB for a 4DGS baseline on Neu3D). Neither single representation is right everywhere in a scene at once.

## Method

Multi4D maintains three Gaussian subsets rendered together via one shared rasterizer: static Gaussians 𝒢ₛ (dense COLMAP init, time-invariant), persistent dynamic Gaussians 𝒢_d (sparse init, canonical Gaussians deformed by a HexPlane + shallow MLP field, geometry-only), and transient Gaussians 𝒢_t (4D spatiotemporal primitives, start empty). Because gradients couple through shared transmittance in the joint render, once one subset already explains a region well its residual/positional gradients shrink, which suppresses further densification there in the other subsets — this is the "competitive allocation" mechanism, and it is implicit (an emergent effect of shared rendering + independent per-subset densification/pruning), not a separate routing network. A "velocity-aware periodical lifting" step promotes fast-moving/active persistent-dynamic Gaussians into new transient primitives, seeded with the parent's estimated velocity. Mask-aware utility-based pruning removes low-visible-contribution primitives from each subset. After the geometry/appearance optimization is frozen, 32-d semantic features are trained only on the compact static+persistent subset (𝒢ₚ = 𝒢ₛ ∪ 𝒢_d) with a SAM-contrastive loss, giving cheap, temporally consistent 4D segmentation.

## Assumptions

Primarily targets calibrated multi-view video (Technicolor, Neu3D: dense synchronized camera rigs), with a secondary evaluation on monocular sequences (NeRF-DS) enabled specifically because the persistent-dynamic branch retains a deformation-field motion prior (the paper states 4D-primitive baselines "lack[] holistic motion prior" and are limited to dense camera input, unlike Multi4D). No explicit calibration or minimum-camera-count requirement is stated; COLMAP-quality initialization for the static subset is assumed.

## Limitations / Failure Modes

The paper's own stated limitation: Multi4D reduces dynamic-primitive count substantially but has no explicit attribute compression (no deformation distillation, quantization, or lightweight parameterization), leaving storage/compute headroom unexploited. The competitive-allocation mechanism is implicit/emergent (coupled gradients + independent per-subset densification) rather than an explicit learned router, so its behavior is not directly controllable or auditable — the paper offers no formula for how much residual routes to which subset, only the qualitative claim that shared transmittance suppresses redundant modeling.

## Reusable Ingredients

- **Three-way capacity typing (static / persistent-dynamic / transient)** — separates "what needs identity" from "what needs detail," rather than one representation doing both.
- **Velocity-aware periodical lifting** — promote high-activity deformation-field Gaussians into freeform 4D primitives, seeded with finite-difference velocity for pose/orientation, instead of spawning new primitives from scratch.
- **Peak-visible-contribution utility score for pruning** (Eq. 6-7 below) — a per-Gaussian, per-view "how much did this primitive actually matter to a rendered pixel" score, combining max and mean visibility across views, usable as a general densification/pruning utility signal independent of this paper's 3-subset design.
- **Asymmetric depth-ordering loss** between two representation types — penalizes only one ordering direction (transient in front of / on persistent geometry), a cheap way to encode a prior ordering assumption without a hard constraint.
- **Attenuated composite-supervision target (γ ramp)** — supervising a composite render against `γ·C_gt` with γ<1 early in training, ramping to 1, to prevent one high-capacity subset from overfitting before a lower-capacity subset stabilizes.

---

### Deep Dive

#### Core Novelty
Relative to prior dynamic-3DGS work that picks one representation (deformation field OR 4D primitives) for the whole dynamic scene, Multi4D's change is to run both simultaneously as separate typed Gaussian subsets inside one shared rasterizer, plus a third static subset, and rely on gradient coupling from joint rendering (rather than an explicit gating network) to let each subset specialize on the regions/frequencies it is best suited for. The key insight: because all subsets render into the same image via the same transmittance chain, a subset that already explains a pixel well starves the positional/residual gradients of the other subsets there for free — no extra loss term or router is needed to prevent redundant capacity in the same region.

#### Mathematical Formulation

**Persistent-dynamic deformation** (per-Gaussian, evaluated before rasterization, each frame):
$$(\boldsymbol{\mu}_t, \mathbf{r}_t) = (\boldsymbol{\mu}, \mathbf{r}) + \Phi_g(\boldsymbol{\mu}, t)$$
$\Phi_g$ is the HexPlane-backed deformation MLP; only geometric attributes (position, rotation) are deformed, not opacity/color, keeping identity purely geometric.

**Composite render for separation supervision:**
$$\mathbf{C}_{comp} = \mathbf{M}_d \odot \mathbf{C}_d + (1-\mathbf{M}_d) \odot \mathbf{C}_s$$
$\mathbf{M}_d$ is the rendered dynamic mask; $\mathbf{C}_d, \mathbf{C}_s$ are the dynamic-subset and static-subset renders. Evaluated as a loss input after rendering.

**Separation loss** (Eq. 11), applied during Phase I (0-10k iters):
$$\mathcal{L}_{sep} = \lambda_{comp}\|\mathbf{C}_{comp} - \gamma \mathbf{C}_{gt}\|_1 + \mathcal{L}_{regional}$$
$\gamma = 0.9$ for the first 2000 iterations then $\gamma \to 1$, so the composite is initially supervised toward a slightly attenuated target — keeps the dynamic subset from overfitting to full-brightness ground truth before the static subset's geometry has stabilized. $\mathcal{L}_{regional}$ (Eq. 12) adds spatially-assigned regional supervision (paper does not give its full separate formula beyond this reference).

**Velocity-aware lifting** (run every 50 iterations, iters 6000-10000): active candidates are $\{g_i \in \mathcal{G}_d \mid m'_i(t) > \tau\}$, $\tau=0.05$, on the deformed mask logit; up to $K=2000$ sampled. Velocity via finite difference:
$$\mathbf{v}_i = \frac{\Phi_g(\boldsymbol{\mu}_i, t+\Delta t) - \Phi_g(\boldsymbol{\mu}_i, t)}{\Delta t}$$
New transient primitive: position $\boldsymbol{\mu}_{4D}^{(new)} = [\boldsymbol{\mu}_i(t) + \epsilon,\ t]^T$ ($\epsilon$ a small offset to avoid immediate occlusion by the parent), rotation $\mathbf{r}_{4D}^{(new)} \leftarrow \text{Align}(\mathbf{v}_i)$ (aligns the 4D primitive's temporal axis to the estimated spatiotemporal trajectory).

**Peak visible contribution / pruning utility** (Eq. 6-7), computed per Gaussian $g_i$ per view $I$:
$$w_{i,I} = \max_{\mathbf{u} \in I}\Big(\sigma_i \mathcal{P}_i(g_i,\mathbf{u}) \prod_{j=1}^{i-1}(1-\sigma_j \mathcal{P}_j(g_j,\mathbf{u}))\Big) \cdot M(\mathbf{u})$$
$$s_i = \beta \cdot \max_{I \in \mathcal{I}_s} w_{i,I} + (1-\beta)\cdot \frac{1}{|\mathcal{I}_s|}\sum_{I \in \mathcal{I}_s} w_{i,I}$$
$\sigma_i \mathcal{P}_i$ is the alpha-blended contribution of Gaussian $i$ at pixel $\mathbf{u}$; the product term is the accumulated transmittance from primitives in front of it (standard alpha compositing); $M(\mathbf{u})$ gates by subset type: $\mathbf{M}_d$ for persistent-dynamics, $(1-\mathbf{M}_d)$ for static, 1 for transient. $s_i$ blends the single best-view peak contribution with the mean across a view sample set $\mathcal{I}_s$ (weight $\beta$, value not specified in extracted text). Primitives with $s_i < \tau_{prune}$ are removed. Evaluated periodically as a pruning criterion, per subset.

**Mask-aware opacity regularization** (Eq. 16), a loss term after rendering:
$$\mathcal{L}_\alpha = \lambda_\alpha \|\alpha_d - \mathbb{I}[M_d > \tau]\|_1,\quad \tau = 0.49$$
pushes the rendered opacity of the persistent-dynamic subset toward the binarized dynamic mask.

**Depth ordering loss** (Eq. 17), a loss term after rendering, over pixel set $\mathcal{P}$:
$$\mathcal{L}_{depth} = \lambda_{depth}\frac{1}{|\mathcal{P}|}\sum_{\mathbf{u}\in\mathcal{P}} \max(0,\ D_{4D}(\mathbf{u}) - D_{3D}(\mathbf{u}))$$
one-sided hinge: only penalized when transient-primitive depth $D_{4D}$ is behind persistent-geometry depth $D_{3D}$ — encodes a prior that transient appearance detail should sit on/in front of the persistent surface, without forcing exact equality.

**Diversity loss:**
$$\mathcal{L}_{diversity} = 0.1 \cdot \sum \text{SSIM}(C_{4D}, C_{3D}) \cdot \alpha_{hybrid}$$
(structural similarity between the transient-only and persistent-only renders, weighted by a hybrid-region opacity term — discourages the two dynamic subsets from reconstructing the same content redundantly).

**Total objective:**
$$\mathcal{L}_{total} = \mathcal{L}_{color} + \lambda_{sep}\mathcal{L}_{sep} + \lambda_{reg}\mathcal{L}_{reg} + \lambda_{div}\mathcal{L}_{diversity}$$
$$\mathcal{L}_{color} = 1.0\|\mathbf{C}-\mathbf{C}_{gt}\|_1 + 0.4(1-\text{SSIM}(\mathbf{C},\mathbf{C}_{gt}))$$

#### Algorithm / Pipeline Changes
1. **Initialization ("inverse expressiveness init")**: static subset from dense COLMAP points; persistent-dynamic subset from 10k sparse random points; transient subset starts empty — capacity is seeded inversely proportional to how expressive/flexible each subset's representation already is.
2. **Phase I, iterations 0-2000**: deformation field frozen (persistent-dynamic subset behaves as static geometry) while static/dynamic separation begins forming.
3. **Phase I, iterations 0-10000**: dynamic-static decomposition active via $\mathcal{L}_{sep}$ with the $\gamma$ ramp (0.9 → 1.0 at iter 2000).
4. **Phase I, iterations 6000-10000**: velocity-aware periodical lifting runs every 50 iterations, moving up to 2000 active persistent-dynamic Gaussians into new transient 4D primitives per lifting step.
5. **Phase II, iterations 10000-20000 (refinement)**: dynamic-static decomposition loss disabled (subset boundaries considered settled); mask-aware utility-based pruning continues to run on all subsets.
6. **Every iteration**: all three subsets are rasterized jointly with shared, depth-sorted alpha compositing (standard 3DGS rasterizer, unmodified compositing math) so gradients couple across subsets — this shared render is what implements competitive allocation; no separate routing/gating network exists.
7. **Post-hoc, after reconstruction freezes**: 32-dimensional semantic features trained only on $\mathcal{G}_p = \mathcal{G}_s \cup \mathcal{G}_d$ via a SAM-mask contrastive loss; DBSCAN clustering on a 2% feature sample produces object instances that are tracked automatically through the existing deformation field (no separate tracker needed).

#### Key Hyperparameters & Design Choices
- HexPlane deformation grid resolution: [64, 64, 64, 150] (x,y,z,t)
- Deformation MLP: 1 hidden layer, width 128
- Semantic feature dimension: 32
- Lifting: $\tau=0.05$ (activity threshold on deformed mask logit), $K=2000$ (max samples per lifting step), every 50 iterations, iters 6000-10000
- Mask-aware opacity threshold: $\tau=0.49$
- $\gamma$ (separation-loss target attenuation): 0.9 for iters 0-2000, then 1.0
- Loss weights: $\lambda_{TV}=0.01$ (depth smoothness), $\lambda_\alpha=0.01$ (mask-aware opacity), $\lambda_{depth}=0.01$, scale penalty 0.01, aspect penalty 0.1, diversity loss coefficient 0.1, color loss SSIM weight 0.4
- Adam learning rates (exponentially decayed): Gaussian positions $1.6\times10^{-4} \to 1.6\times10^{-6}$; deformation MLP $1.6\times10^{-4}\to1.6\times10^{-5}$; HexPlane grids $8.0\times10^{-4}\to5.0\times10^{-6}$; spherical harmonics fixed $2.5\times10^{-3}$; opacity fixed $5.0\times10^{-2}$
- Training schedule: Phase I 0-10k iters (formation), Phase II 10k-20k iters (refinement), total 20k iterations
- $\beta$ (peak vs. mean blend weight in pruning utility score $s_i$): Not specified in paper
- $\tau_{prune}$ (pruning threshold on $s_i$): Not specified in paper
- $\lambda_{comp}$, $\lambda_{sep}$, $\lambda_{reg}$, $\lambda_{div}$ top-level weights: Not specified in paper (only the sub-term weights above are given)

#### Ablation Summary
(Neu3D, 4 scenes; PSNR / DSSIM / dynamic-Gaussian-count / storage; full model = 33.92 dB, 0.0197, 165k, 214.7 MB)
- **w/o persistent dynamics ($\mathcal{G}_d$): -0.70 dB** (33.92 → 33.22 PSNR) — the single most impactful component; paper attributes this to loss of motion prior. Also drastically increases dynamic Gaussian count (145k vs 165k is close, but the comparable no-lifting variant shows count is not the main driver — quality is).
- **w/o transient ($\mathcal{G}_t$): -1.06 dB** (33.92 → 32.86) with count collapsing to 25k / 105.4 MB — worse than removing persistent dynamics, showing high-frequency detail capacity matters most for raw PSNR, though at much lower storage.
- **w/o periodical lifting: -0.70 dB** equivalent (33.92 → 33.22... note paper's "w/o 𝒢ₐ" and "w/o Periodical Lifting" rows are reported separately at 32.78/1139k and 33.22/145k respectively) — removing lifting alone costs quality and prevents efficient transient-primitive count control.
- **w/o $\mathcal{L}_{diversity}$: -0.26 dB** (33.92 → 33.66), storage grows to 263.8 MB — diversity loss keeps persistent/transient subsets from redundantly reconstructing the same content.
- **w/o mask-aware pruning: -0.24 dB** (33.92 → 33.68), count balloons to 729k / 527.9 MB — pruning is primarily a compactness lever, not a quality lever.
- **Baseline 4DGS reference: 33.14 dB, 0.0219 DSSIM, 4215k dynamic Gaussians, 2.6 GB** — full Multi4D beats this baseline on quality while using ~4% of the dynamic primitives and ~8% of the storage.

Most impactful component: the transient (4D-primitive) subset for raw quality (-1.06 dB when removed), but the persistent-dynamic subset is what keeps primitive count/storage low while preserving quality — the paper's framing is that both are needed for the quality/compactness trade-off, not just quality alone.

#### Implementation Reality
- **Framework:** Not specified — repository README does not state the underlying framework or base repo it extends.
- **Key files:** Not available — code is listed as "Coming Soon" on the GitHub repo (https://github.com/BatFaceWayne/Multi4D); no source files exist to inspect yet.
- **Notable implementation details:** None extractable; only the paper's own description is available via the README, which restates the paper's method summary rather than adding new detail.

#### Failure Modes & Limitations
The paper explicitly states Multi4D reduces dynamic primitive count substantially but "does not currently incorporate explicit attribute compression," and suggests deformation distillation, Gaussian quantization, or lightweight parameterizations as future work to close this gap. Beyond this, no scene-specific or condition-specific failure cases (e.g. specific scene names with degraded quality) were found in the extracted material.

---

## Relevance to ADAGS

Very direct pressure on ADAGS's motion-aware densification and fixed-budget claims. ADAGS should report where points/capacity move, not only whether PSNR changes.

## Connections

- Addresses [[gap_map#G1 - Dynamic-Region Sharpness Needs A Direct Objective]]
- Addresses [[gap_map#G5 - Capacity Allocation Must Be Matched And Dynamic-Aware]]
- Addresses [[gap_map#G11 - Representation Frequency Is A New Sharpness Axis]]

## Sources

- https://arxiv.org/abs/2606.22197
