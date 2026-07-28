---
type: paper
node_id: paper:liu2021_neuray
title: "Neural Rays for Occlusion-aware Image-based Rendering"
authors: ["Yuan Liu", "Sida Peng", "Lingjie Liu", "Qianqian Wang", "Peng Wang", "Christian Theobalt", "Xiaowei Zhou", "Wenping Wang"]
year: 2021
venue: "CVPR 2022"
external_ids:
  arxiv: "2107.13421"
  doi: null
  s2: null
tags: ["learned-visibility", "occlusion", "image-based-rendering", "ray-representation"]
added: 2026-07-14T22:18:30Z
status: deep-dived
---

# Neural Rays for Occlusion-aware Image-based Rendering

**Paper:** https://arxiv.org/abs/2107.13421
**Code:** https://github.com/liuyuan-pal/NeuRay
**Base method:** IBRNet (Wang et al., CVPR 2021) for on-the-fly radiance field construction from aggregated multi-view image features; also positioned against MVSNeRF (Chen et al., ICCV 2021) and PixelNeRF (Yu et al., CVPR 2021).

## One-line thesis

Representing per-source-ray visibility as the CDF of a learned mixture-logistics distribution over depth lets the renderer down-weight occluded source-view features analytically, and a scene-specific consistency loss between this visibility CDF and the rendered opacity profile further sharpens it during finetuning.

## Problem / Gap

IBRNet and similar generalizable image-based renderers aggregate per-source-view features for a queried 3D point using only appearance similarity or unweighted pooling; they have no explicit notion of whether that point is actually occluded in a given source view. When a point is occluded in some views, the pooled feature mixes valid and invalid (occluded) observations, which the paper shows costs roughly 2.8 dB PSNR in generalization (25.64 dB IBRNet-style baseline vs 28.41 dB with the visibility-aware init network) and even more after finetuning (29.61 dB without visibility vs 32.97 dB with it, on the Lego scene).

## Method

For each 3D point sampled along a rendering ray, NeuRay looks up a "visibility feature" from a per-source-view feature map built either from a plane-swept cost volume (3 neighboring views, depth discretized into 64 planes) or from COLMAP patch-match depth maps. This feature is decoded by a small MLP into the parameters of a two-component mixture of logistic distributions whose CDF along depth defines a visibility function v(z): the probability the point at depth z is visible to that source view. Standard IBRNet-style feature aggregation is then re-weighted by these visibilities before pooling into per-point color/density and before ray compositing, so features from source views where the point is predicted occluded contribute less. At test time, a discretized "hitting probability" derived from the visibility CDF also gives an alternative, visibility-only alpha/density estimate that can substitute for or blend with the network-predicted density. During per-scene finetuning, a consistency loss forces the visibility-derived hitting probability to match the hitting probability implied by the actual rendered/optimized opacity along the ray, sharpening visibility to match the true reconstructed surface.

## Assumptions

Assumes enough overlapping source views to build a cost volume or reliable MVS/COLMAP depth per view, and that per-view feature matching is reliable enough to signal occlusion (textured, matchable surfaces). The method is fundamentally static-scene image-based rendering — no temporal component.

## Limitations / Failure Modes

The paper states NeuRay cannot render regions that are invisible to *all* working source views, since it relies on cross-view feature aggregation rather than a fully generative global representation like vanilla NeRF. It also degrades in textureless or cluttered regions where feature matching fails to localize the correct surface, corrupting both the depth/cost-volume input and the learned visibility. The visibility-based rendering speedup depends on accurate surface estimates, which the paper notes is mainly reliable after per-scene finetuning rather than in the pure generalization setting.

## Reusable Ingredients

- **Mixture-logistics CDF as a differentiable visibility function** — a compact, continuous parameterization of "probability a point at depth z is visible," decoded from a small per-ray feature by an MLP.
- **Visibility-weighted feature aggregation** — down-weighting occluded-view contributions before pooling, rather than trusting all source views equally (directly reusable in any multi-view feature-aggregation renderer).
- **Consistency loss between predicted visibility and rendered opacity** — a self-supervised way to sharpen an auxiliary visibility signal to match the actual optimized geometry during scene-specific finetuning.
- **Dual initialization paths (cost volume vs. COLMAP depth)** — decouples the visibility encoder from a single upstream depth source, letting it fall back to classical MVS when cost-volume compute isn't available.

---

### Deep Dive

#### Core Novelty
Relative to IBRNet, NeuRay's change is architectural and loss-level: it inserts an explicit, geometry-grounded visibility variable — parameterized as a mixture-logistics CDF over depth — between per-view feature extraction and cross-view aggregation, and adds a consistency loss that ties this visibility to the rendered opacity during finetuning. The key insight is that occlusion is a depth-dependent, per-(point, source-view) binary-ish event that can be modeled analytically and continuously (via a smooth CDF) rather than left for a generic aggregation MLP to infer implicitly from appearance alone.

#### Mathematical Formulation

Visibility as a CDF of a mixture of logistic distributions, decoded per point/ray from feature **g** by MLP ℱ (evaluated per sample point along each source ray, before feature aggregation):
$$
t(z; \{\mu_i,\sigma_i,w_i\}) = \sum_i w_i\, S\!\left(\frac{z-\mu_i}{\sigma_i}\right), \qquad v(z) = 1 - t(z)
$$
where $S(\cdot)$ is the sigmoid, $z$ is depth along the source ray, and $(\mu_i,\sigma_i,w_i)$ are the $i$-th logistic component's mean, scale, and mixture weight (paper uses $N_l=2$ components). $t(z)$ is interpreted as the CDF of the surface's depth distribution, so $v(z)=1-t(z)$ is the probability the queried point (at depth $z$) is still visible (not yet occluded by a nearer surface).

Visibility-weighted feature aggregation (per sampled point, before color/density prediction):
$$
f_i = M(\{f_{ij}, v_{ij} \mid j = 1,\dots,N\})
$$
where $f_{ij}$ is the raw feature from source view $j$ for sample $i$, $v_{ij}$ its predicted visibility, $N$ the number of source views, and $M$ the (IBRNet-style) aggregation network — visibility is concatenated/used as an attention-like weight so occluded-view features are suppressed.

Hitting probability directly from the visibility CDF, giving a visibility-only alternative alpha/opacity per ray interval (used at test time and in the consistency loss):
$$
\tilde h_i = t(z_{i+1}) - t(z_i)
$$
Visibility-weighted alpha compositing across source views $j$ for sample $i$:
$$
\tilde\alpha_{ij}(z_0,z_1) = \frac{t(z_1)-t(z_0)}{1-t(z_0)}, \qquad
\hat\alpha_i = \frac{\sum_j \tilde\alpha_{ij}\, v_{ij}}{\sum_j v_{ij}}, \qquad
\hat h_i = \hat\alpha_i \prod_{k<i}(1-\hat\alpha_k)
$$
Final rendered color is standard volume rendering, $\mathbf{c} = \sum_i \mathbf{c}_i h_i$, with the hitting weights $h_i$ derived above rather than from a plain density MLP.

Scene-specific consistency loss (finetuning stage only, applied per ray after rendering):
$$
\ell_{\text{consist}} = \frac{1}{K_t}\sum_i \mathrm{CE}(\tilde h_i, h_i)
$$
where $h_i$ is the hitting probability implied by the network's actual rendered/optimized opacity along the ray and $\tilde h_i$ is the visibility-CDF-derived hitting probability from above; $K_t$ normalizes over sampled points. This is a cross-entropy term pulling the two toward agreement, effectively distilling the optimized surface back into the visibility function.

Pretraining render loss (standard): $\ell_{\text{render}} = \sum \lVert \mathbf{c} - \mathbf{c}_{gt}\rVert^2$.

#### Algorithm / Pipeline Changes
1. For each of $N_s{=}3$ neighboring source views, build an $H\times W\times D$ plane-swept cost volume ($D{=}64$ depth planes) OR, alternatively, take a COLMAP patch-match depth map per view.
2. A CNN processes the cost volume / depth map into a per-view visibility feature map $\mathbf{G}\in\mathbb{R}^{H\times W\times C}$ (this replaces/augments IBRNet's plain image feature extractor).
3. For each point sampled along a rendering ray, and for each of $N_w{=}8$ working (nearby) source views, look up the corresponding feature in $\mathbf{G}$ and decode it with distribution-decoder MLP ℱ into mixture-logistics parameters $(\mu_1,\mu_2,\sigma_1,\sigma_2,w_0)$ (5 sub-networks, 3 FC layers each) — this yields $v_{ij}$ per (point, source view).
4. Visibility values feed into the aggregation network (IBRNet's design, extended with visibility as an extra input) to reweight per-view features before pooling into per-point color/density — replaces IBRNet's visibility-agnostic aggregation.
5. Volume rendering composites the ray using either the network-predicted density or the visibility-derived hitting probabilities $\hat h_i$ above.
6. Generalization pretraining optimizes only $\ell_{\text{render}}$ across many scenes (400k steps).
7. Per-scene finetuning discards the initialization network, keeps a trainable intermediate feature map $\mathbf{G}'\in\mathbb{R}^{H\times W\times C}$ initialized from step 2's output, and jointly optimizes $\ell_{\text{render}} + \ell_{\text{consist}}$ (200k steps) so the visibility function sharpens to match the scene's actual reconstructed opacity.

#### Key Hyperparameters & Design Choices
- Image encoder: ResNet, 13 residual blocks, 32-channel feature output (shallower than IBRNet's encoder).
- Distribution decoder ℱ: 5 sub-networks, 3 FC layers each, predicting $(\mu_1,\mu_2,\sigma_1,\sigma_2,w_0)$ (mixture of $N_l{=}2$ logistics).
- Working views $N_w = 8$; cost-volume neighboring views $N_s = 3$; cost-volume depth planes $D = 64$.
- Coarse and fine sample counts: 64 each (standard two-pass NeRF-style sampling).
- Learning rate: $2\times10^{-4}$ for pretraining, $1\times10^{-4}$ for finetuning, halved every 100k steps.
- Pretraining: 400k steps (~3 days on a single 2080Ti).
- Finetuning: 200k steps (~20 hours at 800×800 resolution).
- Loss weighting between $\ell_{\text{render}}$ and $\ell_{\text{consist}}$: not specified in paper.

#### Ablation Summary
(Numbers as reported/extracted; treat as approximate.)
- Generalization setting, IBRNet-style baseline (no visibility, no depth features): 25.64 dB PSNR.
- + depth features only (no full visibility mechanism): 26.45 dB (+0.81 dB).
- + full visibility-aware init network ("Init-NeuRay"): 28.41 dB (+2.77 dB over baseline) — **largest single contribution**, confirming visibility prediction is the dominant driver of the generalization gain.
- Finetuning on Lego: without visibility: 29.61 dB; without $\ell_{\text{consist}}$: 31.46 dB (+1.85 dB from adding visibility, further +1.51 dB from adding the consistency loss); full NeuRay: 32.97 dB.
- Single-logistic mixture ($N_l{=}1$) vs. full ($N_l{=}2$): 33.05 dB vs. 32.97 dB — roughly equivalent, suggesting the mixture-of-two is not critical and a single logistic suffices for this scene.
- Most impactful component overall: the visibility prediction mechanism itself (+2.77 dB generalization, +1.85 dB finetuning), with the consistency loss a secondary but still significant contributor (+1.51 dB).

#### Implementation Reality
- **Framework:** PyTorch 1.7.1, with OpenCV, TensorFlow 2.4.1 (likely for some data/eval utility), NumPy, SciPy; custom repo (not a fork of the original NeRF/IBRNet codebase, though it reimplements IBRNet-style aggregation internally).
- **Key files:** `network/dist_decoder.py` (the mixture-logistics distribution decoder MLP — the core novel visibility mechanism); `network/vis_encoder.py` (visibility/occlusion encoder producing the per-view visibility feature map from cost volume or depth); `network/aggregate_net.py` (visibility-weighted multi-view feature aggregation); `network/ibrnet.py` (base aggregation architecture it extends); `network/init_net.py` (cost-volume/depth-based initialization network, discarded after pretraining per the paper's description); `network/loss.py` (includes the consistency loss); `network/renderer.py` and `render_ops.py` (volume-rendering/compositing using visibility-derived hitting probabilities); `network/mvsnet/` subdirectory for MVS cost-volume construction; `network/sph_solver.py` (spherical-harmonic solver, likely for view-dependent color, not part of the core visibility novelty).
- **Notable implementation details:** Two parallel initialization paths are implemented (cost-volume-based and COLMAP-depth-based) rather than a single canonical pipeline, letting the same visibility/aggregation stack run without on-the-fly cost-volume compute when COLMAP depth is available. Config files are split cleanly into `configs/train/gen/` (generalization pretraining), `configs/train/ft/` (scene-specific finetuning), and `configs/gen/` (inference), reflecting the two-stage training described in the paper. Checkpoints save every 10,000 steps during training.

#### Failure Modes & Limitations
The paper explicitly states NeuRay cannot reconstruct regions occluded in *every* working source view, since visibility only reweights existing cross-view evidence rather than hallucinating unseen geometry the way a purely generative model might. It also degrades in textureless or cluttered regions, where the underlying feature matching (cost volume or COLMAP patch-match) fails to localize the correct depth/surface, which in turn corrupts the visibility feature input. The authors further note that the rendering speedup enabled by visibility-derived hitting probabilities depends on accurate surface estimates, making it most reliable after per-scene finetuning rather than in the pure cross-scene generalization setting.

---

## Relevance to This Project

It is the closest clean precedent for a learned visibility field, while clarifying that visibility weighting alone is not hidden-surface representation.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Sources

https://arxiv.org/abs/2107.13421
